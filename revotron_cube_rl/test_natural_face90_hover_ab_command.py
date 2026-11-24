"""Compare the complete 56-joint commands of both frozen controllers on CPU."""
import unittest

import torch
import test_natural_face90_turn_guide as fixtures
from natural_face90_hover_ab_policy import HoverAblationGuide
from natural_face90_turn_guide_v4 import TurnGuide


class HoverCommandComparison(unittest.TestCase):
    def test_approach_and_withdrawal_before_any_logged_touch_keep_the_residual(self):
        f = fixtures.ContactTurn()
        f.setUp()
        g = HoverAblationGuide(f.c)
        obs = dict(f.obs, natural_ready=f.yes.clone(), right_contact=~f.yes,
                   index_contact=~f.yes, other_digit_contact=~f.yes)
        local = f.q.new_tensor([[.050, 0., -.075], [.030, 0., -.090]])
        g._path(f.yes, f.yes, f.radial, local, f.yes, obs)
        residual = torch.full((2, 7), .0025, dtype=torch.float64)
        applied = g.filter_residual(residual, f.yes, f.yes, obs)
        self.assertEqual(g.control_evidence()['pre_hover_residual_off'].tolist(), [True, False])
        self.assertTrue(torch.equal(applied[1], residual[1]))
        obs['contact_goal_reached'] = f.yes.clone()
        g._path(~f.yes, f.yes, f.radial, local, f.yes, obs)
        self.assertTrue(torch.equal(g.filter_residual(residual, ~f.yes, f.yes, obs), residual))

    def test_initial_hover_mixed_lanes_and_return_to_standoff(self):
        f = fixtures.ContactTurn()
        f.setUp()
        baseline, treated, always_on = TurnGuide(f.c), HoverAblationGuide(f.c), TurnGuide(f.c)
        obs = dict(f.obs, natural_ready=f.yes.clone(), other_digit_contact=~f.yes)
        obs['right_contact'] = torch.zeros_like(f.yes)
        obs['index_contact'] = torch.zeros_like(f.yes)
        tip = f.c.tip.fk(f.q[:, f.c.tip.ix])[0] + f.c.root
        f.bodies[:, :, :3] += f.c.root
        local = f.q.new_tensor([[.050, .005, -.075], [.050, .005, -.075]])
        core = torch.cat((tip-local, f.q.new_tensor([[0., 0., 0., 1.]]*2)), -1)
        latent = f.q.new_tensor([[1., -.8, .7, -1., .6, -.8, .9]]*2)
        target = f.q.clone()
        passive = [i for i in range(target.shape[1]) if i not in f.c.rp.ix]
        for step in range(4):
            fresh = f.yes if step == 0 else ~f.yes
            if step == 1:
                obs['right_contact'][0] = True
                obs['index_contact'][0] = True
            if step == 2:
                obs['right_contact'][0] = False
                obs['index_contact'][0] = False
                local[0, 2] = -.10   # release the obstruction after the baseline's retreat
                core[:, :3] = tip - local
            expected_latent = latent.clone()
            # Lane 1 keeps hovering. Lane 0 is treated only on its initial hover.
            expected_latent[1] = 0
            if step == 0:
                expected_latent[0] = 0
            def command(guide, policy_output):
                return guide.command(target, f.q, f.q, fresh, f.yes, policy_output,
                                     core, tip, local, obs)
            expected = command(baseline, expected_latent)
            actual = command(treated, latent)
            unmodified = command(always_on, latent)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual[:, passive], unmodified[:, passive], rtol=0, atol=0)
            self.assertTrue(torch.equal(treated.goal_local, baseline.goal_local))
            if step == 0:
                self.assertGreater(float((actual-unmodified).abs().max()), .0001)
            if step == 3:
                self.assertFalse(treated.recovering[0])
                self.assertLessEqual(float(treated.goal_local[0, 2]), -.075)
                self.assertTrue(torch.equal(actual[0], unmodified[0]))
            target = actual


if __name__ == '__main__':
    unittest.main()
