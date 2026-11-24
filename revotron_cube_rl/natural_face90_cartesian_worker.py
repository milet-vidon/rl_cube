"""Continuous R27 pickup followed by Cartesian residual PPO and protected arm IK."""
from types import FunctionType

import torch

from contact_hold90_cuda_domain import _specialize
import natural_face90_cartesian_control as control
import natural_face90_cartesian_policy as policy
import natural_face90_contact_point_worker as material
import natural_face90_pad_preshape_policy as preshape
import natural_face90_recipe_worker as recipe
from natural_face90_recipe_policy import potential
from natural_face90_v5_metrics import conj, rotate


def cartesian_receipt(req):
    return dict(
        actor_dim=policy.ACTOR_DIM, critic_dim=policy.CRITIC_DIM, action_dim=policy.ACTION_DIM,
        input_fields=list(policy.INPUT_FIELDS), action_axes=list(policy.ACTION_AXES),
        residual_norm_cap_m=control.MAX_RESIDUAL_M,
        end_effector='calibrated_right_index_pad', frame='moving_material_point_on_nZ', ik_backend='placo',
        control_hz=30, index_preshape_deg=req['pad_preshape_offset_deg'],
        material_offset_deg=req['contact_point_offset_deg'], geometry_sha256=req['geometry_sha256'],
        scene_sha256=req['scene_sha256'],
    )


def restore_cartesian(model, optimizer, checkpoint, contract, req):
    """A training wave resumes its committed parent. Validation reads that wave's weights."""
    if (req.get('transfer') or req.get('promote_from_15') or checkpoint['contract'] != contract
            or checkpoint['iteration'] != req['iteration'] - int(req['learn'])):
        raise ValueError('Expected a matching Cartesian PPO checkpoint, task and iteration')
    model.load_state_dict(checkpoint['policy'], strict=True)
    if req['learn']:
        optimizer.load_state_dict(checkpoint['optimizer'])


def cartesian_guide(controller, req):
    guide = control.CartesianGuide(controller, req['pad_orientation_geometry'])
    if guide.placo.receipt != req['placo']:
        guide.placo.close()
        raise ValueError('Placo runtime differs from the requested controller')
    return guide


# Use the project's original R27 initialization, calibration, PPO rollout and optimizer
# bookkeeping. These private bindings change only the post-handoff model and action
# contract. The original modules are also used by previously archived experiments.
_runner_import = ('from natural_face90_recipe_long import fingerprint, worker_software',
                  'from natural_face90_cartesian import fingerprint, worker_software')
_init_ppo = _specialize(recipe.PostPhase.__init__, [
    _runner_import,
    ('super().__init__(e, req, out)', 'ScriptedPostPhase.__init__(self, e, req, out)'),
    ('from natural_face90_recipe_guide import ArmExpert, IndexApproach, restore_expert',
     'from natural_face90_cartesian_policy import CartesianExpert'),
    ('self.guide = IndexApproach(self.controller)', 'self.guide = cartesian_guide(self.controller, req)'),
    ('ArmExpert([self.rids.index(i) for i in self.guide.arm])', 'CartesianExpert()'),
    ("target_deg=15, action_dim=56", "target_deg=req['target_deg'], action_dim=56"),
    ("learned_action_dim=7, observation_kind='privileged_simulation_state'",
     "learned_action_dim=policy.ACTION_DIM, observation_kind='cartesian_pad_state_v1'"),
    ("action_mode='bounded_residual_on_index_pad_ik'",
     "action_mode='bounded_material_frame_displacement_before_ik', cartesian=cartesian_receipt(req), "
     "placo=dict(self.guide.placo.receipt)"),
    ('restore_expert(self.model,', 'restore_cartesian(self.model,'),
], dict(cartesian_guide=cartesian_guide, policy=policy,
        cartesian_receipt=cartesian_receipt, restore_cartesian=restore_cartesian))
_init_turn = FunctionType(material._init_turn.__code__, dict(material._init_turn.__globals__, _init_ppo=_init_ppo))
_init_material = FunctionType(material.PostPhase.__init__.__code__,
                              dict(material.PostPhase.__init__.__globals__, _init_turn=_init_turn))

# Collect the same physical turn/contact and preshape receipts, including when the
# new network is learning. The previous approach diagnostic is frozen-R70-only.
_finish_ppo = _specialize(recipe.PostPhase.finish_learning, [_runner_import], {})
_finish_turn = FunctionType(material._finish_turn.__code__,
                            dict(material._finish_turn.__globals__, _finish_ppo=_finish_ppo))
_finish_material = FunctionType(material.PostPhase.finish_learning.__code__,
                                dict(material.PostPhase.finish_learning.__globals__, _finish_turn=_finish_turn))


def finish_material(self):
    stats = _finish_material(self)
    stats['pad_orientation'] = self.guide.receipt()
    stats['cartesian'] = cartesian_receipt(self.req)
    stats['placo'] = dict(self.guide.placo.receipt)
    return stats


_finish_preshape = _specialize(preshape.PadPreshapePhase.finish_learning, [
    ('super().finish_learning()', 'finish_material(self)'),
], dict(finish_material=finish_material))


class CartesianPhase(preshape.PadPreshapePhase):
    # The selected guide is already installed before restoring the new network.
    # The actual tracker must agree with the archived point used to build its hulls.
    __init__ = _specialize(preshape.PadPreshapePhase.__init__, [
        ('super().__init__(e, req, out)', '_init_material(self, e, req, out)'),
        ("self.guide = PreshapeGuide(self.controller, req['pad_orientation_geometry'])",
         "if not torch.allclose(self.pad_offset, self.guide.meshes[2], atol=1e-8, rtol=0.):\n"
         "        raise ValueError('Cartesian controller and measured pad anchors disagree')\n"
         "    self.pad_motion = policy.PadMotion(self.n, self.device)"),
    ], dict(_init_material=_init_material, policy=policy))

    def state(self, target, remaining):
        # state() is called by the control loop and by several evidence writers.
        # last_step identifies the completed physics measurement across those reads.
        core = self.e.views['Core'].get_transforms().double()
        tip = self.index_tracker.pad_position_world()
        local_tip = rotate(conj(core[:, 3:]), tip - core[:, :3])
        self.pad_geometry = core, tip, local_tip
        motion = self.pad_motion.measure(self.monitor.last_step, local_tip, self.learning)
        m = self.obs
        obs = policy.pack_observation(local_tip, motion, self.controller.radial, m, self.response.credit,
                                      self.req['target_deg'], remaining, self.guide.aligned,
                                      self.guide.drive_contact, self.guide.recovering)
        # Keep the measured distance/withdrawal potential and signed, isolated drive
        # credit of the material-point campaign. No Kinematic target motion earns credit.
        marker = policy.material_marker(self.controller.radial, m['face_deg']).clone()
        reached = m['contact_goal_reached'].bool()
        marker[:, 2] = torch.where(reached, -.100, marker[:, 2])
        distance = (marker - local_tip).norm(dim=1)
        broad = m['progress'].float().clamp(0., 1.)
        phi = potential(distance.float(), m['progress'].float(), m['hold_fraction'].float(),
                        m['actual_retraction_m'].float(), reached)
        signed = (self.response.credit.float() / self.req['target_deg']).clamp(-1., 1.)
        phi = phi + self.response.previously_isolated.float() + 6. * (signed - broad)
        return obs, obs, phi, distance

    def finish_learning(self):
        try:
            return _finish_preshape(self)
        finally:
            self.guide.placo.close()


run = FunctionType(material.run.__code__, dict(material.run.__globals__, PostPhase=CartesianPhase))
main = _specialize(recipe.main, [
    ('from natural_face90_recipe_long import fingerprint, worker_software, validate_request',
     'from natural_face90_cartesian import fingerprint, worker_software, validate_request'),
], dict(run=run, __doc__=__doc__))


if __name__ == '__main__':
    main()
