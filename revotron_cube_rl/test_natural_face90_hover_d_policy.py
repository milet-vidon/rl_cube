"""CPU checks for the frozen C/D approach transition on the R70 joint fixture."""
import unittest

import torch

import test_natural_face90_turn_guide as fixtures
from natural_face90_hover_c_policy import HoverMeasuredGuide
from natural_face90_hover_d_policy import ApproachRecenterGuide, run


class ApproachRecenterTest(unittest.TestCase):
    def setUp(self):
        f = fixtures.ContactTurn()
        f.setUp()
        self.f = f
        self.g = ApproachRecenterGuide(f.c)
        self.c = HoverMeasuredGuide(f.c)
        self.obs = dict(f.obs, natural_ready=f.yes.clone(), right_contact=~f.yes,
                        index_contact=~f.yes, other_digit_contact=~f.yes,
                        right_total_impulse=torch.zeros(2, dtype=torch.double))
        f.tip[:, 2] = -.065

    def path(self, offset, *, fresh=None, active=None, oriented=None):
        f = self.f
        f.tip[:, 1] = f.tip.new_tensor(offset)
        args = (~f.yes if fresh is None else fresh, f.yes if active is None else active,
                f.radial, f.tip, f.yes if oriented is None else oriented, self.obs)
        original = self.c._path(*args)
        modified = self.g._path(*args)
        evidence = self.g.control_evidence()
        torch.testing.assert_close(evidence['pre_c_requested_z_m'], original[:, 2], rtol=0, atol=0)
        torch.testing.assert_close(modified[:, :2], original[:, :2], rtol=0, atol=0)
        off = ~evidence['pre_approach_recenter_hold']
        torch.testing.assert_close(modified[off], original[off], rtol=0, atol=0)
        return modified

    def test_no_entry_without_approach_latch_ratcheting_and_reacquisition(self):
        self.path([.005, .006], fresh=self.f.yes)
        self.path([.004, .006])
        self.assertEqual(self.g.aligned.tolist(), [True, False])
        goal = self.path([.007, .005])
        self.assertEqual(self.g.depth_hold.tolist(), [True, False])
        self.assertEqual(self.g.control_evidence()['pre_approach_recenter_entry'].tolist(), [True, False])
        self.assertEqual(goal[0, 2].item(), -.065)
        for measured, cap in ((-.064, -.065), (-.068, -.068), (-.066, -.068)):
            self.f.tip[0, 2] = measured
            self.assertEqual(self.path([.005, .006])[0, 2].item(), cap)
            self.assertFalse(self.g.control_evidence()['pre_approach_recenter_entry'].any())
        goal = self.path([.004, .004])
        self.assertFalse(self.g.depth_hold.any())
        self.assertTrue(self.g.aligned.all())
        torch.testing.assert_close(goal[:, 2], self.f.tip[:, 2] + .00075)
        self.assertEqual(self.g.recenter_frames.tolist(), [4, 0])

    def test_far_drift_support_orientation_retract_and_fresh_override_depth_hold(self):
        self.path([.001, .002], fresh=self.f.yes)
        self.path([.006, .006])
        self.path([.007001, .006], oriented=torch.tensor([True, False]))
        self.assertFalse(self.g.depth_hold.any())
        self.path([.005, .005])
        self.assertFalse(self.g.depth_hold.any())

        self.path([.002, .002])
        self.path([.005, .005])
        self.obs['support'][0] = False
        self.obs['contact_goal_reached'][1] = True
        self.path([.005, .005])
        self.assertFalse(self.g.depth_hold.any())

        self.obs['support'][0] = True
        self.obs['contact_goal_reached'][1] = False
        self.path([.002, .002])
        self.path([.005, .005], fresh=self.f.yes)
        self.assertFalse(self.g.depth_hold.any())
        self.path([.002, .002])
        self.path([.005, .005], active=~self.f.yes)
        self.path([.005, .005])
        self.assertFalse(self.g.depth_hold.any())

    def test_first_raw_or_other_digit_contact_disarms_hold_for_episode(self):
        self.path([.002, .002], fresh=self.f.yes)
        self.path([.005, .005])
        self.obs['right_total_impulse'][0] = 2e-7
        self.obs['other_digit_contact'][1] = True
        self.path([.005, .005])
        self.assertFalse(self.g.depth_hold.any())
        self.assertTrue(self.g.seen_right_contact.all())
        self.obs['right_total_impulse'].zero_()
        self.obs['other_digit_contact'].zero_()
        self.f.tip[:, 2] = -.092  # release any recovery also triggered on lane 1
        self.path([.002, .002])
        self.path([.005, .005])
        self.assertFalse(self.g.depth_hold.any())
        self.path([.002, .002], fresh=self.f.yes)
        self.path([.005, .005])
        self.assertTrue(self.g.depth_hold.all())

    def test_wrong_contact_and_surface_plane_take_precedence(self):
        self.path([.002, .002], fresh=self.f.yes)
        self.path([.005, .005])
        self.obs['wrong_contact'][0] = True
        self.f.tip[1, 2] = -.044  # Beyond the outward face plane at z=-.045.
        self.path([.005, .005])
        self.assertFalse(self.g.depth_hold.any())
        self.assertEqual(self.g.seen_right_contact.tolist(), [True, False])

    def test_mixed_lane_actual_arm_commands_use_same_residual_and_anchor_gates_as_c(self):
        f = self.f
        world_tip = f.c.tip.fk(f.q[:, f.c.tip.ix])[0] + f.c.root
        _, palm, _, _ = f.c.rp.fk(f.q[:, f.c.rp.ix])
        inward = palm[:, 2, :]
        for guide in (self.g, self.c):
            guide.axis.copy_(inward)
            guide.orientation.copy_(palm)
            guide.vectors[:] = inward[:, None] * .12
        for offset in ((.002, .002), (.005, .008)):
            f.tip[:, 1] = f.tip.new_tensor(offset)
            core = torch.cat((world_tip-f.tip, f.q.new_tensor([[0.,0.,0.,1.]]*2)), dim=1)
            args = (f.q, f.q, f.q, ~f.yes, f.yes, f.q.new_ones(2, 7), core,
                    world_tip, f.tip, self.obs)
            modified = self.g.command(*args)
            original = self.c.command(*args)
            for name in ('pre_hover_measured_anchor', 'pre_hover_residual_off',
                         'pre_arm_residual_applied_max_rad'):
                self.assertTrue(torch.equal(self.g.control_evidence()[name],
                                            self.c.control_evidence()[name]))
            torch.testing.assert_close(modified[1], original[1], rtol=0, atol=0)
        self.assertEqual(self.g.depth_hold.tolist(), [True, False])
        self.assertTrue(self.g.control_evidence()['pre_hover_measured_anchor'].all())
        self.assertGreater((modified[0]-original[0]).abs().max().item(), 1e-6)

    def test_frozen_role_guard(self):
        for change in (dict(role='train'), dict(learn=True), dict(iteration=71), dict(paired_validation=False)):
            request = dict(role='validate', learn=False, iteration=70, paired_validation=True)
            with self.subTest(change=change), self.assertRaises(ValueError):
                run(dict(request, **change), None)


if __name__ == '__main__':
    unittest.main()
