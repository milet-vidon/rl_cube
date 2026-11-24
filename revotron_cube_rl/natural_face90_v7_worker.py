"""Guided right-index centering and sweep in the continuous R27 PhysX scene."""
import json
from types import FunctionType

from contact_hold90_cuda_domain import _specialize
import natural_face90_recipe_worker as guided
import natural_face90_v6_worker as previous


ACTION_MODE = 'index_pad_center_hold_and_sweep'


def restore_expert(model, optimizer, checkpoint, contract, req):
    from natural_face90_v6_long import SCHEMA as old_schema

    expected = contract.copy()
    origin = req.get('transfer')
    if origin:
        train = req['role'] == 'train' and req['iteration'] == origin['iteration'] + 1
        frozen = (req['role'] == 'validate' and req['iteration'] == origin['iteration']
                  and (req.get('paired_validation') is True or req.get('fresh_source_validation') is True))
        if (not (train or frozen) or origin['schema'] != old_schema
                or checkpoint['iteration'] != origin['iteration']
                or (req['checkpoint'], req['checkpoint_sha']) != (origin['checkpoint'], origin['sha'])):
            raise ValueError('Invalid source for indexed contact policy transfer')
        expected.update(schema=old_schema, software=origin['software'], action_mode=previous.ACTION_MODE)
    if checkpoint['contract'] != expected:
        raise ValueError('Centering policy checkpoint contract mismatch')
    model.load_state_dict(checkpoint['policy'], strict=True)
    if not origin and req['learn']:
        optimizer.load_state_dict(checkpoint['optimizer'])


# Reuse the pinned PPO initialization, update, and rollout. Each worker binds
# its own imports so an existing campaign retains its checkpoint contract.
_runner_import = ('from natural_face90_recipe_long import fingerprint, worker_software',
                  'from natural_face90_v7_long import fingerprint, worker_software')
_init_ppo = _specialize(guided.PostPhase.__init__, [
    _runner_import,
    ('super().__init__(e, req, out)', 'ScriptedPostPhase.__init__(self, e, req, out)'),
    ('from natural_face90_recipe_guide import ArmExpert, IndexApproach, restore_expert',
     'from natural_face90_recipe_guide import ArmExpert\n    from natural_face90_turn_guide_v4 import TurnGuide as IndexApproach'),
    ("action_mode='bounded_residual_on_index_pad_ik'", 'action_mode=action_mode'),
], {'restore_expert': restore_expert, 'action_mode': ACTION_MODE})
_finish_ppo = _specialize(guided.PostPhase.finish_learning, [_runner_import], {})


class DriveResponse:
    """Positive net angle during one uninterrupted commanded, measured contact."""

    def __init__(self, n, device):
        import torch
        self.start = torch.zeros(n, device=device, dtype=torch.float64)
        self.peak = self.start.clone()
        self.active = torch.zeros_like(self.start, dtype=torch.bool)
        self.previously_isolated = self.active.clone()
        self.drive_frames = torch.zeros_like(self.start, dtype=torch.long)
        self.qualified_frames = self.drive_frames.clone()
        self.distal_frames = self.drive_frames.clone()

    def update(self, angle, qualified, drive, distal, other_digit, last_angle):
        import torch
        from natural_face90_v5_metrics import wrap
        # drive applies over the interval between the last measurement and this
        # one. Count its motion only if both physical endpoints were qualified.
        isolated = qualified & ~other_digit
        active = drive & isolated & self.previously_isolated
        self.start = torch.where(active & ~self.active, last_angle, self.start)
        self.peak = torch.where(active, torch.maximum(self.peak, wrap(angle-self.start)), self.peak)
        self.active = active
        self.previously_isolated = isolated
        self.drive_frames += drive.long()
        self.qualified_frames += qualified.long()
        self.distal_frames += distal.long()


class PostPhase(previous.PostPhase):
    def __init__(self, e, req, out):
        from natural_face90_turn_contacts import install
        _init_ppo(self, e, req, out)
        self.index_tracker = install(e, self.monitor, self.pad_body, self.pad_offset)
        self.index_progress = previous.IndexProgress(e.n, self.device)
        self.response = DriveResponse(e.n, self.device)
        self.turn_log = (out/'turn_contact_evidence.jsonl').open('x', buffering=1)

    def observe(self, active, post, safe, step):
        import torch
        from natural_face90_v5_metrics import rotate, conj
        from natural_face90_turn_guide import pad_near_marker

        result = guided.PostPhase.observe(self, active, post, safe, step)
        m = self.obs
        threshold = self.e.cfg['success']['contact_impulse_min']
        m['index_impulse'] = self.index_tracker.index_impulse.clone()
        m['index_contact'] = m['index_impulse'] > threshold
        m['other_digit_contact'] = self.index_tracker.other_digit_impulse > threshold
        m['static_contact'] = m['left_static_impulse'] > threshold
        learned = post & self.learning
        if not learned.any():
            return result
        core = self.e.views['Core'].get_transforms().double()
        local = rotate(conj(core[:,3:]), self.index_tracker.pad_position_world()-core[:,:3])
        near = pad_near_marker(self.controller.radial, local, m['face_deg'])
        qualified = (learned & safe & m['natural_ready'] & m['index_contact'] & near
                     & m['static_contact'] & m['support'] & m['structure_ok'] & m['allowed_layer_state']
                     & ~m['wrong_contact'] & ~self.retract)
        drive = learned & self.guide.drive_contact
        self.response.update(m['face_deg'], qualified, drive, learned & m['index_contact'],
                             m['other_digit_contact'], self.index_progress.last)
        self.index_progress.update(m['face_deg'], qualified)
        # The pre_* values explain the command for this control step (30 Hz).
        # Impulses, pad pose, and face angle are from its subsequent PhysX readback.
        fields = dict(
            self.guide.control_evidence(),
            drive_command=drive, sweep_command_deg=self.guide.sweep_deg, realigning=self.guide.recovering,
            requested_x_m=self.guide.goal_local[:,0], requested_y_m=self.guide.goal_local[:,1],
            requested_z_m=self.guide.goal_local[:,2], index_lead_m=self.guide.index_lead_m,
            right_target_impulse_Ns=m['target_impulse'], index_distal_impulse_Ns=m['index_impulse'],
            other_digit_target_impulse_Ns=self.index_tracker.other_digit_impulse,
            static_support_impulse_Ns=m['left_static_impulse'], face_deg=m['face_deg'],
            pad_near_marker=near, qualified_index_frame=qualified, safe=safe,
            signed_index_driven_deg=self.index_progress.driven, peak_index_driven_deg=self.index_progress.peak,
            peak_drive_response_deg=self.response.peak,
            pad_x_m=local[:,0], pad_y_m=local[:,1], pad_z_m=local[:,2],
        )
        ids = torch.where(learned)[0]
        values = torch.stack([v[ids].double() for v in fields.values()], 1).cpu().tolist()
        for lane, row in zip(ids.tolist(), values):
            self.turn_log.write(json.dumps(dict(step=step, lane=lane, seed=self.req['seeds'][lane],
                **dict(zip(fields, row))), allow_nan=False)+'\n')
        return result

    def row(self, i):
        return dict(super().row(i),
                    drive_frames=int(self.response.drive_frames[i]),
                    distal_contact_frames=int(self.response.distal_frames[i]),
                    qualified_index_frames=int(self.response.qualified_frames[i]),
                    peak_index_driven_deg=float(self.index_progress.peak[i]),
                    peak_drive_response_deg=float(self.response.peak[i]))

    def finish_learning(self):
        from natural_face90_turn_guide_v4 import RECOVERY_REASONS
        self.turn_log.close()
        result = _finish_ppo(self)
        result['turn_evidence_sha256'] = guided.sha(self.out/'turn_contact_evidence.jsonl')
        result['control_recovery_reasons'] = RECOVERY_REASONS
        return result


run = FunctionType(guided.run.__code__, dict(guided.run.__globals__, PostPhase=PostPhase), 'run')
main = _specialize(guided.main, [_runner_import], {'run': run})

if __name__ == '__main__':
    main()
