"""Material-point comparison and signed contact-driven PPO in the continuous R27 scene."""
import json
import math
from types import FunctionType

import torch

from contact_hold90_cuda_domain import _specialize
import natural_face90_hover_c_policy as hover
from natural_face90_hover_ab import R70_SHA
import natural_face90_recipe_worker as recipe
from natural_face90_turn_guide import TurnGuide as ArmGuide, _point
import natural_face90_v5_worker as scripted
from natural_face90_v5_metrics import conj, rotate, wrap
import natural_face90_v7_worker as centered


ACTION_MODE = 'fixed_material_point_measured_hover_and_isolated_sweep'


def material_radial(radial, offset_deg):
    """Positive nZ rotation sends Core +X towards -Y."""
    theta = math.radians(offset_deg)
    tangent = torch.stack((radial[:, 1], -radial[:, 0]), dim=1)
    return math.cos(theta) * radial + math.sin(theta) * tangent


def forward_regrasp_point(current, forward, face_deg):
    """Choose a facelet interior on the 30 mm ring in material coordinates."""
    # The 90 mm face has seams at x/y = +/-15 mm. These fixed directions
    # leave >=6.2 mm from a seam and >=15 mm from the outer edge.
    angles = [math.radians(quadrant + offset) for quadrant in (0, 90, 180, 270)
              for offset in (-15, 0, 15, 45)]
    candidates = current.new_tensor([(math.cos(t), math.sin(t)) for t in angles])
    theta = torch.deg2rad(face_deg.double())[:, None]
    tangent = torch.stack((forward[:, 1], -forward[:, 0]), dim=1)
    # Undo the measured face motion to express the workspace's forward side
    # in material coordinates. A detached cube receives no motion command.
    desired = theta.cos() * forward - theta.sin() * tangent
    next_radial = candidates[(desired @ candidates.T).argmax(dim=1)]
    improvement = .030 * ((current - desired).norm(dim=1) - (next_radial - desired).norm(dim=1))
    return next_radial, improvement


def unloaded_after_stall(active, guide, obs, pad_z, oriented, credit, last_switch_credit):
    return (active & guide.recovering & (guide.recovery_reason == 3)
            & (credit >= last_switch_credit + 10.) & oriented & obs['natural_ready'].bool()
            & obs['support'].bool() & obs['static_contact'].bool() & obs['structure_ok'].bool()
            & obs['allowed_layer_state'].bool() & ~obs['wrong_contact'].bool()
            & ~obs['contact_goal_reached'].bool() & (obs['right_total_impulse'] == 0.)
            & ~obs['right_contact'].bool() & ~obs['index_contact'].bool() & ~obs['other_digit_contact'].bool()
            & (pad_z <= -.045 - .020) & (pad_z <= guide.clear_z + .003))


class _TargetSweep(ArmGuide):
    # v4 keeps the pre-contact/recovery conditions and its 16 degree lead cap.
    _path = _specialize(ArmGuide._path, [
        ('torch.full_like(face,18.)', 'torch.full_like(face, self.c.deg + 3.)'),
    ], {})


class MaterialGuide(hover.HoverMeasuredGuide, _TargetSweep):
    def command(self, target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs):
        command = super().command(target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs)
        face = obs['face_deg'].double()
        marker = _point(self.c.radial, face, torch.zeros_like(face))
        self.control_evidence().update(pre_material_marker_x_m=marker[:, 0],
                                       pre_material_marker_y_m=marker[:, 1], pre_material_face_deg=face)
        return command


class SignedDriveResponse(centered.DriveResponse):
    """Sum signed motion only on commanded intervals with isolated contact at both ends."""
    def __init__(self, n, device):
        super().__init__(n, device)
        self.seen = self.active.clone()
        self.first_angle = self.peak.clone()
        self.signed = self.peak.clone()
        self.credit = self.peak.clone()
        self.peak_credit = self.peak.clone()

    def update(self, angle, qualified, drive, distal, other_digit, last_angle):
        super().update(angle, qualified, drive, distal, other_digit, last_angle)
        isolated = self.previously_isolated
        self.first_angle = torch.where(isolated & ~self.seen, angle, self.first_angle)
        delta = wrap(angle - last_angle)
        # Debit retrograde slip between pushes. Otherwise a repeated push and
        # an unrelated forward advance in a gap could cover each other's loss.
        loss = torch.where(self.seen, delta.clamp_max(0.), 0.)
        self.signed += torch.where(self.active, delta, loss)
        self.seen |= isolated
        # A repeated push after slip may not count the same range of angles twice.
        bounded = torch.minimum(self.signed, wrap(angle-self.first_angle))
        # Off-contact slips may reduce the potential. Off-contact movement may
        # not add reward or trigger a release or a change of material point.
        self.credit = torch.where(self.seen, bounded, 0.)
        self.peak_credit = torch.where(self.active, torch.maximum(self.peak_credit, self.credit), self.peak_credit)


def restore_expert(model, optimizer, checkpoint, contract, req):
    """The immutable v3 R70 starts each candidate. Later checkpoints bind the offset and task."""
    if req.get('transfer'):
        if req['checkpoint_sha'] != R70_SHA or req['transfer']['iteration'] != 70 or req['target_deg'] != 15:
            raise ValueError('Expected the archived 15 degree R70 transfer')
        expected = contract.copy()
        del expected['contact_point_offset_deg']
        centered.restore_expert(model, optimizer, checkpoint, expected, req)
        return

    parent = contract.copy()
    promotion = req.get('promote_from_15')
    if promotion:
        if (req['role'] != 'train' or not req['learn'] or req['target_deg'] != 90
                or (promotion['checkpoint'], promotion['sha']) != (req['checkpoint'], req['checkpoint_sha'])
                or promotion['iteration'] + 1 != req['iteration']):
            raise ValueError('Invalid 15 to 90 degree promotion')
        parent['target_deg'] = 15
    expected_iteration = req['iteration'] - int(req['learn'])
    if checkpoint['iteration'] != expected_iteration or checkpoint['contract'] != parent:
        raise ValueError('Material-point checkpoint iteration, offset or task contract mismatch')
    model.load_state_dict(checkpoint['policy'], strict=True)
    if req['learn'] and not promotion:
        optimizer.load_state_dict(checkpoint['optimizer'])


# Bind new contracts privately using the project's pinned rollout implementation.
# The historic C/D scripts must still restore and verify their original inputs.
_runner_import = ('from natural_face90_recipe_long import fingerprint, worker_software',
                  'from natural_face90_contact_point import fingerprint, worker_software')
_init_ppo = _specialize(recipe.PostPhase.__init__, [
    _runner_import,
    ('super().__init__(e, req, out)', 'ScriptedPostPhase.__init__(self, e, req, out)'),
    ('from natural_face90_recipe_guide import ArmExpert, IndexApproach, restore_expert',
     'from natural_face90_recipe_guide import ArmExpert'),
    ('self.guide = IndexApproach(self.controller)', 'self.guide = MaterialGuide(self.controller)'),
    ("target_deg=15, action_dim=56", "target_deg=req['target_deg'], action_dim=56"),
    ("action_mode='bounded_residual_on_index_pad_ik'",
     "action_mode=ACTION_MODE, contact_point_offset_deg=req['contact_point_offset_deg']"),
], dict(restore_expert=restore_expert, ACTION_MODE=ACTION_MODE, MaterialGuide=MaterialGuide))
_finish_ppo = _specialize(recipe.PostPhase.finish_learning, [_runner_import], {})
_init_turn = FunctionType(centered.PostPhase.__init__.__code__,
                          dict(centered.PostPhase.__init__.__globals__, _init_ppo=_init_ppo,
                               DriveResponse=SignedDriveResponse))
_finish_turn = FunctionType(centered.PostPhase.finish_learning.__code__,
                            dict(centered.PostPhase.finish_learning.__globals__, _finish_ppo=_finish_ppo))
_observe_turn = _specialize(centered.PostPhase.observe, [
    ('peak_drive_response_deg=self.response.peak,',
     'peak_drive_response_deg=self.response.peak, isolated_drive_progress_deg=self.response.credit,\n'
     '        peak_isolated_drive_progress_deg=self.response.peak_credit,'),
], {})

# The feature contract encodes measured angle / task target. The source packer
# divides by 15, so pass it an equivalent 15-degree-scale angle for the 90-degree task.
_task_state = _specialize(recipe.PostPhase.state, [
    ("m['face_deg'], m['progress'], reached, remaining",
     "m['face_deg'] * (15. / self.req['target_deg']), m['progress'], reached, remaining"),
], {})


class PostPhase(centered.PostPhase):
    def __init__(self, e, req, out):
        from natural_face90_contact_probe_native import install_probe
        _init_turn(self, e, req, out)
        self.base_radial = torch.zeros_like(self.controller.radial)
        self.selected_radial = torch.zeros_like(self.controller.radial)
        self.last_switch_credit = torch.zeros_like(self.response.credit)
        self.regrasp_count = torch.zeros_like(self.guide.streak)
        self.isolated_goal = torch.zeros_like(self.learning)
        self.regrasp_log = ((out/'contact_point_changes.jsonl').open('x', buffering=1)
                            if req['target_deg'] == 90 else None)
        # Only the paired screening waves collect every raw contact point.
        self.probe = install_probe(self, req['seeds'], out) if req.get('paired_validation') else None

    def select_material_point(self, fresh):
        # The scripted detour is identical across offsets. At the learned handoff,
        # change the shared material radial exactly once before reading the first
        # actor observation; v4's guide, physics observer and probe use that same radial.
        if not fresh.any():
            return
        base = self.controller.radial[fresh]
        self.base_radial[fresh] = base
        selected = material_radial(base, self.req['contact_point_offset_deg'])
        self.selected_radial[fresh] = selected
        self.controller.radial[fresh] = selected

    def regrasp_after_stall(self, active):
        guide = self.guide
        if self.regrasp_log is None or not (active & guide.recovering & (guide.recovery_reason == 3)).any():
            return
        # Verify clearance and orientation in the latest measured state before
        # choosing the target for the NEXT command. All coordinates are Core-local.
        core = self.e.views['Core'].get_transforms().double()
        pad = rotate(conj(core[:, 3:]), self.index_tracker.pad_position_world() - core[:, :3])
        q = self.e.robot.get_dof_positions()[:, self.e.ids].double()
        _, palm, _, _ = self.controller.rp.fk(q[:, guide.arm])
        inward = rotate(core[:, 3:], q.new_tensor([0., 0., 1.]))
        actual_axis = (palm.transpose(1, 2) @ inward[:, :, None]).squeeze(2)
        lead = ((guide.vectors * actual_axis[:, None]).sum(-1) - guide.radii).amin(dim=1)
        oriented = ((guide.axis * actual_axis).sum(-1) > .985) & (lead >= .002)
        obs = self.obs
        credit = self.response.credit
        ready = unloaded_after_stall(active, guide, obs, pad[:, 2], oriented, credit, self.last_switch_credit)
        chosen, improvement = forward_regrasp_point(self.controller.radial, self.base_radial, obs['face_deg'])
        switch = ready & (improvement >= .002)
        for lane in switch.nonzero(as_tuple=True)[0].tolist():
            record = dict(step=self.monitor.last_step + 1, lane=lane, seed=self.req['seeds'][lane],
                          previous_radial_xy=self.controller.radial[lane].tolist(),
                          selected_radial_xy=chosen[lane].tolist(), face_deg=float(obs['face_deg'][lane]),
                          isolated_progress_deg=float(credit[lane]), pad_z_m=float(pad[lane, 2]),
                          right_impulse_Ns=float(obs['right_total_impulse'][lane]),
                          forward_distance_improvement_m=float(improvement[lane]))
            self.regrasp_log.write(json.dumps(record, allow_nan=False) + '\n')
        self.controller.radial[switch] = chosen[switch]
        self.last_switch_credit[switch] = self.response.peak_credit[switch]
        self.regrasp_count += switch.long()

    propose = _specialize(recipe.PostPhase.propose, [
        ('self.learning |= eligible', 'self.select_material_point(fresh)\n'
         '    self.regrasp_after_stall(active & self.learning)\n    self.learning |= eligible'),
        ('super().propose(target,', 'ScriptedPostPhase.propose(self, target,'),
    ], {})

    def state(self, target, remaining):
        obs, critic, phi, distance = _task_state(self, target, remaining)
        broad = self.obs['progress'].float().clamp(0., 1.)
        signed = (self.response.credit.float() / self.req['target_deg']).clamp(-1., 1.)
        contact = self.response.previously_isolated.float()
        # Native qualified contact and signed layer progress replace the original
        # generic progress term. Effort, retreat, horizon and unsafe penalties remain.
        return obs, critic, phi + contact + 6. * (signed - broad), distance

    def observe(self, active, post, safe, step):
        result = _observe_turn(self, active, post, safe, step)
        if self.req['target_deg'] == 90:
            # The raw physical observer continues to measure all nine pieces and
            # its own direct-contact goal. Only the command-side release waits
            # for full isolated index-drive credit in the same at-goal interval.
            physical = self.obs['contact_goal_reached'].bool()
            self.obs['physical_contact_goal_reached'] = physical
            self.isolated_goal |= (physical & self.obs['at_goal'].bool() & (self.obs['face_deg'] >= 90. - 1e-7)
                                   & self.response.active & (self.response.credit >= 90. - 1e-7))
            self.obs['contact_goal_reached'] = self.isolated_goal.clone()
            insufficient = result['success'] & ~self.isolated_goal
            result['success'] = result['success'] & ~insufficient
            result['failed'] = result['failed'] | insufficient
            for lane in insufficient.nonzero(as_tuple=True)[0].tolist():
                result['reason'][lane] = 'insufficient_isolated_drive_angle'
        return result

    def remember(self, step, q, dq, command, core, prefix, post):
        super().remember(step, q, dq, command, core, prefix, post)
        if self.probe is not None:
            self.probe.begin_step(step, post & self.learning)

    def row(self, lane):
        return dict(super().row(lane), peak_isolated_drive_progress_deg=float(self.response.peak_credit[lane]))

    def finish_learning(self):
        result = _finish_turn(self)
        result['hover_ablation'] = dict(operator=hover.baseline.OPERATOR, frames_by_lane=self.guide.hover_frames.tolist())
        result['hover_anchor_ablation'] = dict(operator=hover.OPERATOR, frames_by_lane=self.guide.anchor_frames.tolist())
        result['contact_point'] = dict(offset_deg=self.req['contact_point_offset_deg'], radius_m=.030,
            selected_by_lane=self.learning.tolist(), base_radial_xy_by_lane=self.base_radial.tolist(),
            selected_radial_xy_by_lane=self.selected_radial.tolist())
        if self.regrasp_log is not None:
            self.regrasp_log.close()
            result['contact_point']['regrasp_count_by_lane'] = self.regrasp_count.tolist()
            result['contact_point']['changes_sha256'] = recipe.sha(self.out/'contact_point_changes.jsonl')
        if self.probe is not None:
            result['geometry_probe'] = self.probe.close()
        return result


inputs = _specialize(scripted.inputs, [
    ("req['target_deg']==15,'Frozen R27 and 15 degree goal required'",
     "req['target_deg'] in (15,90),'Frozen R27 and a supported material-point goal required'"),
], {})
run = FunctionType(recipe.run.__code__, dict(recipe.run.__globals__, inputs=inputs, PostPhase=PostPhase))
main = _specialize(recipe.main, [_runner_import], dict(run=run))

if __name__ == '__main__':
    main()
