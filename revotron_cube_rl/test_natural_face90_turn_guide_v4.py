"""CPU control regressions at the edge of the measured round-70 contact band."""
import torch

import test_natural_face90_turn_guide as baseline
from natural_face90_turn_guide_v4 import TurnGuide


class ContactCapture(baseline.ContactTurn):
    def setUp(self):
        super().setUp()
        self.g = TurnGuide(self.c)
        self.obs['natural_ready'] = self.yes.clone()
        self.obs['other_digit_contact'] = ~self.yes

    def test_single_contact_near_edge_recenters_and_brief_dropout_reseats_from_measured_depth(self):
        self.tip[:] = self.tip.new_tensor([.040, .004, -.0509])
        goal = self.plan(fresh=self.yes)
        ev = self.g.control_evidence()
        self.assertTrue(ev['pre_centering'].all())
        self.assertFalse(self.g.drive_contact.any())
        torch.testing.assert_close(goal[:, 2], self.tip[:, 2])
        step = (goal[:, :2] - self.tip[:, :2]).norm(dim=1)
        torch.testing.assert_close(step, torch.full_like(step, .00075))

        self.tip[:, 2] -= .002  # outside the sensing band, still near the last real contact
        self.obs['right_contact'] = ~self.yes
        self.obs['index_contact'] = ~self.yes
        for _ in range(6):
            goal = self.plan()
            self.assertTrue(self.g.control_evidence()['pre_centering'].all())
            self.assertFalse(self.g.control_evidence()['pre_index_near'].any())
            self.assertFalse(self.g.drive_contact.any())
            self.assertTrue((self.g.streak == 0).all())
            torch.testing.assert_close(goal[:, 2], self.tip[:, 2] + .00075)
        self.assertTrue((self.plan()[:, 2] <= -.075).all())

    def test_unloaded_approach_rate_and_three_real_samples_after_dropout(self):
        self.tip[:, 2] = -.075
        self.obs['right_contact'] = ~self.yes
        self.obs['index_contact'] = ~self.yes
        torch.testing.assert_close(self.plan(fresh=self.yes)[:, 2], self.tip[:, 2] + .00075)
        self.tip[:, 2] = -.0508
        for touch, expected in ((True, 1), (True, 2), (False, 0), (True, 1), (True, 2), (True, 3)):
            self.obs['right_contact'][:] = touch
            self.obs['index_contact'][:] = touch
            self.plan()
            self.assertTrue((self.g.streak == expected).all())
            self.assertTrue((self.g.drive_contact == (expected == 3)).all())

    def test_support_loss_and_simultaneous_other_digit_contact_cancel_capture(self):
        self.plan(fresh=self.yes)
        self.obs['other_digit_contact'][1] = True
        self.obs['static_contact'][0] = False
        goal = self.plan()
        self.assertTrue(self.g.recovering.all())
        self.assertTrue((goal[:, 2] <= -.075).all())
        self.assertEqual(self.g.recovery_reason.tolist(), [4, 1])
        self.assertFalse(self.g.drive_contact.any())
        self.assertFalse(self.g.capture_remaining.any())

    def test_negative_rotation_under_drive_unloads_immediately(self):
        self.plan(fresh=self.yes)
        self.plan()
        self.plan()
        self.assertTrue(self.g.drive_contact.all())
        self.obs['face_deg'][:] = -.6
        self.plan()
        self.assertTrue(self.g.recovering.all())
        self.assertTrue((self.g.recovery_reason == 6).all())
        self.assertFalse(self.g.drive_contact.any())
