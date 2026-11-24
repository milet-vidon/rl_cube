"""Frozen R70 hover C. Use measured right-arm joints as the nominal IK anchor."""
from types import FunctionType

import torch

import natural_face90_hover_ab_policy as baseline
from contact_hold90_cuda_domain import _specialize
from natural_face90_turn_guide import TurnGuide as ArmGuide


OPERATOR = 'anchor_nominal_right_arm_ik_to_measured_q_during_precontact_hover'

# The B filter evaluates the gate after _path and counts that decision once.
# Move the same residual calculation before the IK so both interventions read
# that single gate, including on the decision following the first physical touch.
_residual = ('residual = .008*latent.double().tanh()\n'
             '    residual = torch.where(self.recovering[:,None],0.,residual)')


class _MeasuredArmCommand(baseline._ResidualHook):
    command = _specialize(ArmGuide.command, [
        (_residual, ''),
        ('prev = self.reference',
         _residual + '\n    residual = self.filter_residual(residual, fresh, learned, obs)\n'
         '    prev = self.select_anchor(q[:,arm], target[:,arm])'),
    ], {})


class HoverMeasuredGuide(baseline.HoverAblationGuide, _MeasuredArmCommand):
    # The v4 command still scales the latent before calling _MeasuredArmCommand.
    def __init__(self, controller):
        super().__init__(controller)
        self.anchor_frames = torch.zeros_like(self.hover_frames)

    def select_anchor(self, measured, previous_target):
        evidence = self.control_evidence()
        hover = evidence['pre_hover_residual_off']
        measured = measured.to(self.reference.dtype)
        anchor = torch.where(hover[:, None], measured, self.reference)
        self.anchor_frames += hover.long()
        evidence.update(
            pre_hover_measured_anchor=hover,
            pre_previous_reference_error_max_rad=(self.reference-measured).abs().amax(dim=1),
            pre_ik_anchor_error_max_rad=(anchor-measured).abs().amax(dim=1),
            pre_previous_command_error_max_rad=(previous_target-measured).abs().amax(dim=1),
        )
        return anchor

    def command(self, target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs):
        result = super().command(target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs)
        self.control_evidence()['sent_command_error_max_rad'] = (
            result[:, self.arm].double() - q[:, self.arm].double()).abs().amax(dim=1)
        return result


class PostPhase(baseline.PostPhase):
    def __init__(self, e, req, out):
        baseline.frozen.PostPhase.__init__(self, e, req, out)
        self.guide = HoverMeasuredGuide(self.controller)

    def finish_learning(self):
        result = super().finish_learning()
        result['hover_anchor_ablation'] = dict(operator=OPERATOR, frames_by_lane=self.guide.anchor_frames.tolist())
        return result


_rollout = FunctionType(baseline.frozen.run.__code__, dict(baseline.frozen.run.__globals__, PostPhase=PostPhase))
# Keep the exact same frozen paired validation guard used by B.
run = FunctionType(baseline.run.__code__, dict(baseline.run.__globals__, _rollout=_rollout), 'run')
