"""Frozen R70 study of a depth hold during unloaded approach recentering."""
from types import FunctionType

import torch

import natural_face90_hover_c_policy as control


OPERATOR = 'hold_depth_for_unloaded_approach_recenter'
INNER_M = .004  # Existing v4 threshold for forward approach.
OUTER_M = .007  # Experimental capture band around the same material marker.


class ApproachRecenterGuide(control.HoverMeasuredGuide):
    def __init__(self, controller):
        super().__init__(controller)
        self.previous_approach = torch.zeros_like(self.aligned)
        self.depth_hold = torch.zeros_like(self.aligned)
        self.depth_cap = torch.zeros_like(self.pad_gap_m)
        self.seen_right_contact = torch.zeros_like(self.aligned)
        self.recenter_frames = torch.zeros_like(self.hover_frames)

    def _path(self, fresh, active, radial, local_tip, orientation_ok, obs):
        goal = super()._path(fresh, active, radial, local_tip, orientation_ok, obs)
        state = self.control_evidence()
        # The observer's "any right" signal includes subthreshold contacts that
        # would not qualify as a drive. D is limited to fully unloaded motion.
        any_right = ((obs['right_total_impulse'] > 1e-7) | state['pre_contact']
                     | state['pre_index_contact'] | state['pre_other_digit_contact']
                     | obs['wrong_contact'].bool())
        self.seen_right_contact = (self.seen_right_contact & ~fresh) | (active & any_right)
        unloaded = (active & ~self.seen_right_contact & state['pre_support_ok'] & orientation_ok
                    & ~self.recovering & ~obs['contact_goal_reached'].bool())
        lateral = state['pre_lateral_error_m']
        hold = (unloaded & ~fresh & (self.previous_approach | self.depth_hold)
                & ~self.aligned & (lateral > INNER_M) & (lateral <= OUTER_M)
                & (state['pre_normal_gap_m'] > 0))
        entry = hold & ~self.depth_hold

        local_z = local_tip[:, 2]
        # In Core coordinates larger z points inward. If the measured pad
        # retreats, lower the stored cap so the next tick cannot chase it in.
        self.depth_cap = torch.where(hold & self.depth_hold,
                                     torch.minimum(self.depth_cap, local_z), local_z)
        self.depth_hold = hold
        self.previous_approach = unloaded & self.aligned
        self.recenter_frames += hold.long()
        state.update(pre_approach_recenter_hold=hold, pre_approach_recenter_entry=entry,
                     pre_measured_pad_z_m=local_z, pre_c_requested_z_m=goal[:, 2],
                     pre_recenter_depth_cap_m=self.depth_cap,
                     pre_recenter_any_right_contact=any_right,
                     pre_recenter_seen_right_contact=self.seen_right_contact)
        # Keep C's in-plane marker, orientation, residual mask, measured-joint
        # anchor and final joint projections. Only its normal target changes.
        z = torch.where(hold, self.depth_cap, goal[:, 2])
        self.goal_local = torch.cat((goal[:, :2], z[:, None]), dim=1)
        return self.goal_local


class PostPhase(control.PostPhase):
    def __init__(self, e, req, out):
        control.baseline.frozen.PostPhase.__init__(self, e, req, out)
        self.guide = ApproachRecenterGuide(self.controller)

    def finish_learning(self):
        result = super().finish_learning()
        result['approach_recenter_ablation'] = dict(
            operator=OPERATOR, inner_m=INNER_M, outer_m=OUTER_M,
            frames_by_lane=self.guide.recenter_frames.tolist(),
        )
        return result


_rollout = FunctionType(control._rollout.__code__, dict(control._rollout.__globals__, PostPhase=PostPhase))
run = FunctionType(control.run.__code__, dict(control.run.__globals__, _rollout=_rollout), 'run')
