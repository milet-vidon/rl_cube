"""Measured index-pad approach with bounded right-arm PPO residuals."""
import torch

from contact_hold_cuda_teacher_math import log_rot, mv
from natural_face90_recipe_policy import StateExpert
from natural_face90_v5_controller import _dls, face_point
from natural_face90_v5_metrics import rotate


def restore_expert(model, optimizer, checkpoint, contract, req):
    expected = contract.copy()
    transfer = req.get('transfer')
    if transfer:
        from natural_face90_recipe_policy import SCHEMA
        if req['role'] != 'train' or checkpoint['iteration'] != transfer['iteration']:
            raise ValueError('Invalid legacy checkpoint iteration or role')
        del expected['arm_indices'], expected['action_mode']
        expected.update(schema=SCHEMA, learned_action_dim=28, software=transfer['software'])
    if checkpoint['contract'] != expected:
        raise ValueError('PPO checkpoint contract mismatch')
    model.load_state_dict(checkpoint['policy'], strict=True)
    if not transfer:
        optimizer.load_state_dict(checkpoint['optimizer'])


class ArmExpert(StateExpert):
    """Reuse the legacy network weights, optimizing only executed arm channels."""

    def __init__(self, channels):
        super().__init__()
        if len(channels) != 7 or len(set(channels)) != 7 or not all(0 <= i < 28 for i in channels):
            raise ValueError('Expected seven distinct right-arm actor channels')
        self.register_buffer('arm_channels', torch.tensor(channels), persistent=False)

    def distribution(self, obs):
        full = super().distribution(obs)
        return torch.distributions.Normal(full.loc.index_select(-1, self.arm_channels),
                                          full.scale.index_select(-1, self.arm_channels))


def guide_point(radial, local_tip, face, touch, reached):
    """Align outside the face before applying depth, then sweep on measured contact."""
    angle = face.double()
    surface = face_point(torch.full_like(angle, .030), radial, angle, torch.zeros_like(angle))
    error = (surface[:, :2] - local_tip[:, :2]).norm(dim=1)
    # Clear the 45 mm half-edge while translating across the face. Blend into
    # the surface over 6 mm to avoid switching normal direction frame to frame.
    align = ((.010 - error) / .006).clamp(0, 1)
    depth = -.022 + .0225 * align
    lead = torch.minimum(angle + 4., torch.full_like(angle, 16.))
    goal_angle = torch.where(touch & ~reached, torch.maximum(angle, lead), angle)
    depth = torch.where(reached, torch.full_like(depth, -.055), depth)
    return face_point(torch.full_like(angle, .030), radial, goal_angle, depth)


class IndexApproach:
    def __init__(self, controller):
        self.c = controller
        self.arm = controller.rp.ix
        self.reference = controller.right_ref.clone()
        self.orientation = controller.route_rotation.clone()

    def command(self, target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs):
        c, arm = self.c, self.arm
        q = q.double()
        _, palm, _, _ = c.rp.fk(q[:, arm])
        self.orientation = torch.where(fresh[:, None, None], palm, self.orientation)
        prev = torch.where(fresh[:, None], target[:, arm].double(), self.reference)
        ref_q = q.clone()
        ref_q[:, arm] = prev
        ref_tip, _, origins, axes = c.tip.fk(ref_q[:, c.tip.ix])
        _, ref_palm, _, _ = c.rp.fk(prev)
        goal = guide_point(c.radial, local_tip, obs['face_deg'], obs['right_contact'],
                           obs['contact_goal_reached'])
        # Both terms in the error use measured PhysX world coordinates. FK supplies
        # the arm Jacobian. Keep nominal IK separate from the bounded policy offset.
        error = core[:, :3] + rotate(core[:, 3:], goal) - tip
        delta = _dls(ref_tip, ref_palm, origins, axes, ref_tip + error,
                     log_rot(self.orientation @ ref_palm.transpose(1, 2)), .012)
        delta = c.clearance.steer(prev, delta, ref_tip, origins, axes)
        cap = torch.minimum(c.e.step, c.e.vel / 30).expand_as(target)[:, arm].double()
        lim = c.e.limits.expand(len(q), -1, -1)[:, arm].double()
        factor = (cap / delta.abs().clamp_min(1e-12)).amin(1, keepdim=True).clamp_max(1.)
        nominal = (prev + factor * delta).clamp(q[:, arm] - .04, q[:, arm] + .04)
        nominal = c.clearance.project(prev, nominal.clamp(lim[..., 0], lim[..., 1]))
        self.reference = torch.where(learned[:, None], nominal, self.reference)
        residual = .008 * latent.double().tanh()
        goal_arm = (nominal + residual).clamp(lim[..., 0], lim[..., 1])
        old = target[:, arm].double()
        arm_cmd = c.clearance.project(old, old + (goal_arm - old).clamp(-cap, cap))
        # Hold the entire hand at its last collision-free handoff command.
        result = handoff.clone()
        result[:, arm] = arm_cmd.to(result.dtype)
        return result
