"""Continuous guided PPO with native index-tip contact and a contact-driven sweep."""
import json
from types import FunctionType

from contact_hold90_cuda_domain import _specialize
import natural_face90_recipe_worker as guided


ACTION_MODE = 'index_contact_realign_and_sweep'


def restore_turn_expert(model, optimizer, checkpoint, contract, req):
    """Only the first wave migrates the guided weights onto the new action law."""
    from natural_face90_recipe_long import GUIDED_SCHEMA
    expected = contract.copy()
    transfer = req.get('transfer')
    if transfer:
        if (req['role'] != 'train' or transfer['schema'] != GUIDED_SCHEMA
                or checkpoint['iteration'] != transfer['iteration']
                or req['checkpoint_sha'] != transfer['sha']):
            raise ValueError('Invalid guided checkpoint for contact-turn transfer')
        expected.update(schema=GUIDED_SCHEMA, software=transfer['software'],
                        action_mode='bounded_residual_on_index_pad_ik')
    if checkpoint['contract'] != expected:
        raise ValueError('Contact-turn checkpoint contract mismatch')
    model.load_state_dict(checkpoint['policy'], strict=True)
    if not transfer:
        optimizer.load_state_dict(checkpoint['optimizer'])


# The running guided experiment fingerprints its source. Bind its existing
# initialization and update code privately, preserving the native R27 rollout.
_runner_import = ('from natural_face90_recipe_long import fingerprint, worker_software',
                  'from natural_face90_v6_long import fingerprint, worker_software')
_init_ppo = _specialize(guided.PostPhase.__init__, [
    _runner_import,
    ('super().__init__(e, req, out)', 'ScriptedPostPhase.__init__(self, e, req, out)'),
    ('from natural_face90_recipe_guide import ArmExpert, IndexApproach, restore_expert',
     'from natural_face90_recipe_guide import ArmExpert\n    from natural_face90_turn_guide import TurnGuide as IndexApproach'),
    ("action_mode='bounded_residual_on_index_pad_ik'", 'action_mode=action_mode'),
], {'restore_expert': restore_turn_expert, 'action_mode': ACTION_MODE})
_finish_ppo = _specialize(guided.PostPhase.finish_learning, [_runner_import], {})


class IndexProgress:
    """Credit signed angular motion across consecutive qualifying measured frames."""
    def __init__(self, n, device):
        import torch
        self.last = torch.zeros(n, device=device, dtype=torch.float64)
        self.start = self.last.clone()
        self.driven = self.last.clone()
        self.peak = self.last.clone()
        self.previous_contact = torch.zeros_like(self.last, dtype=torch.bool)
        self.seen = self.previous_contact.clone()

    def update(self, angle, contact):
        import torch
        from natural_face90_v5_metrics import wrap
        self.start = torch.where(contact & ~self.seen, angle, self.start)
        self.seen |= contact
        self.driven += torch.where(contact & self.previous_contact, wrap(angle-self.last), 0.)
        net = torch.minimum(self.driven, wrap(angle-self.start))
        self.peak = torch.where(contact, torch.maximum(self.peak,net), self.peak)
        self.previous_contact = contact.clone()
        self.last = angle.clone()


class PostPhase(guided.PostPhase):
    def __init__(self, e, req, out):
        from natural_face90_turn_contacts import install
        _init_ppo(self, e, req, out)
        self.index_tracker = install(e, self.monitor, self.pad_body, self.pad_offset)
        self.index_progress = IndexProgress(e.n, self.device)
        self.turn_log = (out/'turn_contact_evidence.jsonl').open('x', buffering=1)

    def state(self, target, remaining):
        obs, critic, phi, distance = super().state(target, remaining)
        # Preserve the actor input contract for checkpoint transfer. For shaping,
        # attribute progress to measured index contact at the intended surface.
        broad = self.obs['progress'].float().clamp(0, 1)
        index = (self.index_progress.peak.float()/self.req['target_deg']).clamp(0, 1)
        return obs, critic, phi + 6.*(index-broad), distance

    def observe(self, active, post, safe, step):
        import torch
        from natural_face90_v5_metrics import rotate, conj
        from natural_face90_turn_guide import pad_near_marker
        result = super().observe(active, post, safe, step)
        m = self.obs
        threshold = self.e.cfg['success']['contact_impulse_min']
        m['index_impulse'] = self.index_tracker.index_impulse.clone()
        m['index_contact'] = m['index_impulse'] > threshold
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
        self.index_progress.update(m['face_deg'], qualified)
        # Each line aligns the PRE-step guide command to the subsequent native
        # physical readback. Goal position and pad position are in Core coordinates.
        fields = {
            'right_target_impulse_Ns':m['target_impulse'], 'index_distal_impulse_Ns':m['index_impulse'],
            'other_digit_target_impulse_Ns':self.index_tracker.other_digit_impulse,
            'static_support_impulse_Ns':m['left_static_impulse'], 'face_deg':m['face_deg'],
            'pad_near_marker':near, 'qualified_index_frame':qualified, 'safe':safe,
            'signed_index_driven_deg':self.index_progress.driven, 'drive_command':self.guide.drive_contact,
            'sweep_command_deg':self.guide.sweep_deg, 'realigning':self.guide.recovering,
            'requested_x_m':self.guide.goal_local[:,0], 'requested_y_m':self.guide.goal_local[:,1],
            'requested_z_m':self.guide.goal_local[:,2], 'pad_x_m':local[:,0], 'pad_y_m':local[:,1],
            'pad_z_m':local[:,2], 'index_lead_m':self.guide.index_lead_m,
        }
        ids = torch.where(learned)[0]
        values = torch.stack([v.double()[ids] for v in fields.values()],1).cpu().tolist()
        for lane, row in zip(ids.tolist(), values):
            self.turn_log.write(json.dumps(dict(step=step,lane=lane,seed=self.req['seeds'][lane],
                **dict(zip(fields,row))),allow_nan=False)+'\n')
        return result

    def finish_learning(self):
        self.turn_log.close()
        result = _finish_ppo(self)
        result['turn_evidence_sha256'] = guided.sha(self.out/'turn_contact_evidence.jsonl')
        return result


# Inject the single versioned controller dependency into the pinned physical
# loop. Other running workers retain the original function and its globals.
run = FunctionType(guided.run.__code__, dict(guided.run.__globals__, PostPhase=PostPhase), 'run')
main = _specialize(guided.main, [_runner_import], {'run':run})

if __name__ == '__main__':
    main()
