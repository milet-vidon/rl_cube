"""Opt-in hold gain continuity; retain the original contact and safety gates."""
import json
import math
from pathlib import Path

import torch

from isaac_compat import tensor, quaternion_from_native
from settled_wrist_teacher import SettledWristTeacher


class ContactHoldTeacher(SettledWristTeacher):
    hold_damping = 1.0
    damping_ramp_steps = 30

    def __init__(self, env, config, *args, **kwargs):
        super().__init__(env, config, *args, **kwargs)
        if not math.isfinite(self.hold_damping) or not 1. <= self.hold_damping <= 4.:
            raise ValueError('Hold damping must be within [1, 4]')
        state = tensor(env.cube.data.root_state_w)[0]
        initial = dict(q=tensor(env.robot.data.joint_pos)[0, env.ids].tolist(),
                       cube_position=state[:3].tolist(),
                       cube_quaternion_wxyz=quaternion_from_native(state[3:7]).tolist(),
                       applied_target=env.targets[0].tolist(),
                       joint_limits=env.limits[0].tolist())
        (Path(env.run_dir) / 'initial_state_v12.json').write_text(json.dumps(initial, indent=2))

    def _hold_gains(self):
        """Do not drop the finger stiffness at the first hold command."""
        offset = self.tick - len(self.path)
        if offset > self.damping_ramp_steps and getattr(self, 'hold_gains_set', False):
            return
        ids = [self.env.ids[i] for i, n in enumerate(self.env.cfg['joint_names'])
               if n.startswith('left_')]
        damping = 1. + (self.hold_damping - 1.) * min(offset / self.damping_ramp_steps, 1.)
        self.env.robot.write_joint_stiffness_to_sim_index(
            stiffness=torch.full((1, len(ids)), 30., dtype=torch.float32), joint_ids=ids)
        self.env.robot.write_joint_damping_to_sim_index(
            damping=torch.full((1, len(ids)), damping, dtype=torch.float32), joint_ids=ids)
        # Both parent hold paths honor this event; all target and acceptance
        # logic still runs through SettledWristTeacher and TiltTeacher.
        self.hold_gains_set = True
        event = dict(step=int(self.env.episode_length_buf[0]), hold_offset=offset,
                     joint_ids=ids, stiffness=30., damping=damping)
        with (Path(self.env.run_dir) / 'hold_gain_events_v12.jsonl').open('a') as f:
            f.write(json.dumps(event) + '\n')

    def action(self):
        if self.mode == 'tilt' and self.tick >= len(self.path):
            self._hold_gains()
        return super().action()
