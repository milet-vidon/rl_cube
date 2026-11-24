"""Frozen v4 control with a lane-local knockout of the pre-contact hover residual."""
from types import FunctionType

import torch

from contact_hold90_cuda_domain import _specialize
from natural_face90_turn_guide import TurnGuide as ArmGuide
from natural_face90_turn_guide_v4 import TurnGuide as CenteringGuide
import natural_face90_v7_worker as frozen


OPERATOR = 'zero_right_arm_ppo_residual_during_precontact_hover'


class _ResidualHook(ArmGuide):
    # CenteringGuide.command calls super() after its normal near-pad scaling.
    # Insert one filter after the original recovery mask and immediately before
    # adding the residual to the nominal IK target. All projections stay downstream.
    command = _specialize(ArmGuide.command, [(
        'wanted = (nominal+residual)',
        'residual = self.filter_residual(residual, fresh, learned, obs)\n    wanted = (nominal+residual)',
    )], {})


class HoverAblationGuide(CenteringGuide, _ResidualHook):
    def __init__(self, controller):
        super().__init__(controller)
        if len(self.arm) != 7:
            raise ValueError('Expected seven residual-controlled right arm joints')
        self.seen_target_touch = torch.zeros_like(self.aligned)
        self.hover_frames = torch.zeros_like(self.streak)

    def filter_residual(self, residual, fresh, learned, obs):
        # _path has already classified the command that will be sent this step.
        # Contact flags come from the previous physical readback. A first touch
        # occurring on THIS action ends the intervention on the next decision.
        state = self.control_evidence()
        touch = state['pre_contact'] | state['pre_index_contact'] | state['pre_other_digit_contact']
        self.seen_target_touch = (self.seen_target_touch & ~fresh) | (learned & touch)
        hover = (learned & ~self.seen_target_touch & state['pre_support_ok']
                 & ~self.aligned & ~self.recovering & ~self.drive_contact
                 & ~state['pre_centering'] & ~state['pre_settling']
                 & (self.capture_remaining == 0) & ~obs['contact_goal_reached'].bool())
        applied = torch.where(hover[:, None], 0., residual)
        self.hover_frames += hover.long()
        state.update(
            pre_hover_residual_off=hover,
            pre_arm_residual_baseline_max_rad=residual.abs().amax(dim=1),
            pre_arm_residual_applied_max_rad=applied.abs().amax(dim=1),
            pre_arm_residual_change_max_rad=(applied-residual).abs().amax(dim=1),
        )
        return applied


class PostPhase(frozen.PostPhase):
    def __init__(self, e, req, out):
        super().__init__(e, req, out)
        # Both guides start from the same pristine controller. The v4 actor has
        # already been restored against its original action/checkpoint contract.
        self.guide = HoverAblationGuide(self.controller)

    def finish_learning(self):
        result = super().finish_learning()
        result['hover_ablation'] = dict(operator=OPERATOR, frames_by_lane=self.guide.hover_frames.tolist())
        return result


_rollout = FunctionType(frozen.run.__code__, dict(frozen.run.__globals__, PostPhase=PostPhase))


def run(req, out):
    if (req.get('role') != 'validate' or req.get('learn') is not False
            or req.get('paired_validation') is not True or req.get('iteration') != 70):
        raise ValueError('Hover comparison requires a frozen paired round-70 validation')
    return _rollout(req, out)
