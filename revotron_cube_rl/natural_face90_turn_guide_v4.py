"""Close the loop on the measured index pad before loading the nZ face."""
import torch

from natural_face90_turn_guide import TurnGuide as ContactTurnGuide, _point, pad_near_marker


RECOVERY_REASONS = {
    0: 'none', 1: 'other_link_first', 2: 'index_off_marker', 3: 'turn_stalled',
    4: 'support_or_structure_lost', 5: 'loaded_wrist_misaligned', 6: 'reverse_motion',
}


class TurnGuide(ContactTurnGuide):
    def __init__(self, controller):
        super().__init__(controller)
        self.capture_remaining = torch.zeros_like(self.streak)
        self.last_contact_z = torch.zeros_like(self.pad_gap_m)
        self.recovery_reason = torch.zeros_like(self.streak)
        self._evidence = {}

    def _path(self, fresh, active, radial, local_tip, orientation_ok, obs):
        face = obs['face_deg'].double()
        marker = _point(radial, face, torch.zeros_like(face))
        offset = marker[:, :2] - local_tip[:, :2]
        lateral = offset.norm(dim=1)
        near = pad_near_marker(radial, local_tip, face)
        touch = obs['right_contact'].bool()
        index = obs['index_contact'].bool()
        other_digit = obs['other_digit_contact'].bool()
        supported = obs['support'].bool() & obs['natural_ready'].bool()
        good = (supported & obs['static_contact'].bool() & obs['structure_ok'].bool()
                & obs['allowed_layer_state'].bool() & ~obs['wrong_contact'].bool())
        reached = obs['contact_goal_reached'].bool()
        was_recovering = self.recovering & ~fresh
        reverse = ~fresh & (self.window > 0) & (face < self.start_face - .5)

        # Reacquire the narrow 4 mm centre if it drifts before the sweep starts.
        # During a sweep the original moving-marker band and stall guard apply.
        drifted = (self.streak < 3) & (lateral > .004)
        self.aligned &= ~(active & (~orientation_ok | drifted))
        # A loaded wrist must unload before reorientation or loss of left support.
        # Keep the original raw contact readings in the trace and the observer.
        routable = dict(obs, support=supported & ~reverse,
                        right_contact=touch | other_digit,
                        index_contact=index & ~other_digit & ~reverse & good & orientation_ok)
        goal = super()._path(fresh, active, radial, local_tip, orientation_ok, routable)
        reason = torch.full_like(self.streak, 3)  # positive drive stalled
        reason = torch.where(touch & ~index, 1, reason)
        reason = torch.where(touch & index & ~near, 2, reason)
        reason = torch.where(touch & ~orientation_ok, 5, reason)
        reason = torch.where(touch & reverse, 6, reason)
        reason = torch.where(touch & ~good, 4, reason)
        reason = torch.where(other_digit, 1, reason)
        self.recovery_reason = torch.where(self.recovering,
            torch.where(was_recovering, self.recovery_reason, reason), 0)

        free_to_center = (active & good & orientation_ok & ~other_digit & ~reverse
                          & ~self.recovering & ~reached)
        caught = free_to_center & touch & index & near
        # Store the last *measured* legitimate contact depth. For up to six lost
        # samples the fingertip may recenter and reseat here without jumping to
        # the 30 mm standoff. Missing contact still resets the base drive streak.
        normal_gap = marker[:, 2] - local_tip[:, 2]
        nearby = (lateral <= .012) & (normal_gap.abs() <= .010)
        grace = free_to_center & ~touch & nearby & ~fresh & (self.capture_remaining > 0)
        self.capture_remaining = torch.where(caught, 6,
            torch.where(grace, self.capture_remaining - 1, 0))
        self.last_contact_z = torch.where(caught, local_tip[:, 2], self.last_contact_z)
        capture = (caught | grace) & ~self.drive_contact

        # Correct at most 0.75 mm along the face per control target. A new
        # contact holds its measured normal depth; a dropout can probe 0.3 mm
        # beyond that depth. The original maximum surface depth still applies.
        local_z = local_tip[:, 2]
        depth_limit = marker[:, 2] + .0005
        desired_z = torch.minimum(self.last_contact_z + .0003 * grace, depth_limit)
        capture_xy = local_tip[:, :2] + offset * (.00075 / lateral.clamp_min(.00075))[:, None]
        capture_z = torch.minimum(local_z + (desired_z-local_z).clamp(-.00075, .00075), depth_limit)
        capture_goal = torch.cat((capture_xy, capture_z[:, None]), dim=1)
        # Once aligned, advance the normal target relative to the actual pad,
        # not directly from standoff to the face. Retraction remains immediate.
        approach = free_to_center & self.aligned
        goal[:, 2] = torch.where(approach, torch.minimum(goal[:, 2], local_z + .00075), goal[:, 2])
        self.goal_local = torch.where(capture[:, None], capture_goal, goal)
        self._evidence = {
            'pre_lateral_error_m': lateral, 'pre_normal_gap_m': normal_gap,
            'pre_aligned': self.aligned, 'pre_orientation_ok': orientation_ok,
            'pre_contact': touch, 'pre_index_contact': index, 'pre_index_near': near,
            'pre_other_digit_contact': other_digit,
            'pre_contact_streak': self.streak, 'pre_recovering': self.recovering,
            'pre_recovery_reason': self.recovery_reason, 'pre_support_ok': good,
            'pre_centering': capture & ~self.aligned, 'pre_settling': capture & self.aligned,
            'pre_capture_remaining': self.capture_remaining,
        }
        return self.goal_local

    def control_evidence(self):
        return self._evidence

    def command(self, target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs):
        # Within the contact band, the PPO residual must not dominate the
        # sub-millimetre nominal correction. Preserve its original seven axes.
        close = pad_near_marker(self.c.radial, local_tip, obs['face_deg'])
        fine_latent = torch.atanh(.25 * torch.tanh(latent))
        latent = torch.where(close[:, None], fine_latent, latent)
        return super().command(target, q, handoff, fresh, learned, latent, core, tip, local_tip, obs)
