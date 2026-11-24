"""Exercise the frozen hover C control path on the two-lane CPU joint fixture."""
import unittest

import torch

import test_natural_face90_turn_guide as fixtures
from natural_face90_hover_ab_policy import HoverAblationGuide
from natural_face90_hover_c_policy import HoverMeasuredGuide, run


class MeasuredHoverTest(unittest.TestCase):
    def test_frozen_gate(self):
        request = dict(role='validate', learn=False, paired_validation=True, iteration=70)
        for change in (dict(role='train'), dict(learn=True), dict(paired_validation=False), dict(iteration=71)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                run(dict(request, **change), None)

    def test_full_command_on_mixed_lanes_and_touch_boundary(self):
        f = fixtures.ContactTurn()
        f.setUp()
        controlled = HoverMeasuredGuide(f.c)
        oracle = HoverAblationGuide(f.c)
        baseline = HoverAblationGuide(f.c)
        obs = dict(f.obs, natural_ready=f.yes.clone(), right_contact=~f.yes,
                   index_contact=~f.yes, other_digit_contact=~f.yes)
        tip = f.c.tip.fk(f.q[:, f.c.tip.ix])[0] + f.c.root
        f.bodies[:, :, :3] += f.c.root
        local = f.q.new_tensor([[.050, .005, -.075]]*2)
        core = torch.cat((tip-local, f.q.new_tensor([[0., 0., 0., 1.]]*2)), -1)
        latent = f.q.new_tensor([[1., -.8, .7, -1., .6, -.8, .9]]*2)
        target = f.q.clone()
        for step, expected_gate in enumerate(((True, True), (False, True), (False, True),
                                              (False, False), (False, False))):
            fresh = f.yes if step == 0 else ~f.yes
            if step == 1:
                obs['right_contact'][0] = True
                obs['index_contact'][0] = True
            if step == 2:
                obs['right_contact'][0] = False
                obs['index_contact'][0] = False
                local[0, 2] = -.1
                core[:, :3] = tip-local
            if step == 3:
                obs['support'][1] = False
            if step == 4:
                obs['support'][1] = True
                obs['contact_goal_reached'][1] = True

            # Compare with the existing B implementation, setting its reference
            # only on the lanes that must use the actual configuration this tick.
            gate = torch.tensor(expected_gate)
            oracle.reference[gate] = f.q[:, oracle.arm][gate]
            args = (target, f.q, f.q, fresh, f.yes, latent, core, tip, local, obs)
            result = controlled.command(*args)
            torch.testing.assert_close(result, oracle.command(*args), rtol=0, atol=0)
            torch.testing.assert_close(controlled.reference, oracle.reference, rtol=0, atol=0)
            original = baseline.command(*args)
            evidence = controlled.control_evidence()
            self.assertTrue(torch.equal(evidence['pre_hover_residual_off'], gate))
            self.assertTrue(torch.equal(evidence['pre_hover_measured_anchor'], gate))
            self.assertTrue(torch.all(evidence['pre_ik_anchor_error_max_rad'][gate] == 0))
            self.assertTrue(torch.equal(evidence['pre_ik_anchor_error_max_rad'][~gate],
                                        evidence['pre_previous_reference_error_max_rad'][~gate]))
            if step == 1:
                # q is stationary while targets have moved. C must actually
                # change the IK output in free hover, keeping B on touched lane 0.
                self.assertGreater(evidence['pre_previous_reference_error_max_rad'][1].item(), .001)
                self.assertGreater((original[1]-result[1]).abs().amax().item(), .0001)
                torch.testing.assert_close(result[0], original[0], rtol=0, atol=0)
            arm_difference = (result[:, controlled.arm]-f.q[:, controlled.arm]).abs().amax(dim=1)
            self.assertTrue(torch.equal(evidence['sent_command_error_max_rad'], arm_difference))
            target = result
        self.assertEqual(controlled.anchor_frames.tolist(), [1, 3])
        self.assertTrue(torch.equal(controlled.anchor_frames, controlled.hover_frames))


if __name__ == '__main__':
    unittest.main()
