"""Opt-in hold gain continuity experiment; original acceptance is retained.

The original teacher lowers every left finger's stiffness from 30 to 20 at
hold entry. This candidate keeps the grasp stiffness and records that event.
It changes robot drive gains only, never cube state, contacts, or audit values.
"""
import hashlib
import json
from pathlib import Path

import torch

from settled_wrist_teacher import SettledWristTeacher


class ContactHoldTeacher(SettledWristTeacher):
    hold_stiffness = 30.0
    hold_damping = 1.0
    # Preserve the fragile index contact. Only unload the two fingers with
    # sustained high contact impulses in the 11601 gain-continuity trace.
    opposing_pair_stiffness = 30.0
    gain_ramp_steps = 30
    source_joint_damping = False
    position_iterations = None

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        root = Path(__file__).resolve().parent
        files = ('contact_hold_teacher.py', 'contact_hold_experiment.py')
        snapshot = env.run_dir / 'contact_hold_source'
        snapshot.mkdir()
        hashes = {}
        for name in files:
            data = (root / name).read_bytes()
            (snapshot / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        manifest = {
            'schema': 'contact_hold_gain_continuity_candidate_v1',
            'teacher_class': type(self).__name__,
            'source_sha256': hashes,
            'left_finger_hold_stiffness': self.hold_stiffness,
            'left_finger_hold_damping': self.hold_damping,
            'thumb_little_final_stiffness': self.opposing_pair_stiffness,
            'thumb_little_ramp_steps': self.gain_ramp_steps,
            'source_joint_linear_damping_enabled': self.source_joint_damping,
            'position_iterations': self.position_iterations,
            'original_contact_and_speed_acceptance_retained': True,
            'neural_policy_trained': False,
        }
        (env.run_dir / 'contact_hold_candidate.json').write_text(json.dumps(manifest, indent=2))

    def action(self):
        if self.mode == 'tilt' and self.tick >= len(self.path) and not getattr(self, 'hold_gains_set', False):
            ids = [self.env.ids[i] for i, name in enumerate(self.env.cfg['joint_names']) if name.startswith('left_')]
            if len(ids) != 21:
                raise ValueError('Expected exactly 21 left finger joints')
            self.env.robot.write_joint_stiffness_to_sim_index(
                stiffness=torch.full((1, len(ids)), self.hold_stiffness, dtype=torch.float32), joint_ids=ids)
            self.env.robot.write_joint_damping_to_sim_index(
                damping=torch.full((1, len(ids)), self.hold_damping, dtype=torch.float32), joint_ids=ids)
            self.hold_gains_set = True
            event = dict(kind='contact_hold_gain_continuity', step_in_tilt=self.tick,
                         stiffness=self.hold_stiffness, damping=self.hold_damping)
            self.planner.diagnostics.append(event)
            print('CONTACT_HOLD_GAINS', json.dumps(event), flush=True)
        if self.mode == 'tilt' and self.opposing_pair_stiffness != self.hold_stiffness:
            offset = self.tick - len(self.path)
            if 1 <= offset <= self.gain_ramp_steps:
                pair = [self.env.ids[i] for i, name in enumerate(self.env.cfg['joint_names'])
                        if name.startswith(('left_thumb_', 'left_little_'))]
                if len(pair) != 9:
                    raise ValueError('Expected five thumb and four little-finger joints')
                gain = self.hold_stiffness + (self.opposing_pair_stiffness - self.hold_stiffness) * offset / self.gain_ramp_steps
                self.env.robot.write_joint_stiffness_to_sim_index(
                    stiffness=torch.full((1, len(pair)), gain, dtype=torch.float32), joint_ids=pair)
                if offset == self.gain_ramp_steps:
                    self.planner.diagnostics.append(dict(kind='opposing_pair_ramp_complete', stiffness=gain, step_in_tilt=self.tick))
        return super().action()
