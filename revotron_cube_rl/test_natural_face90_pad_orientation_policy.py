import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

import natural_face90_pad_orientation_policy as policy
import test_natural_face90_turn_guide as fixtures


def geometry():
    def hull(x, z):
        return [[x, 0., .020 + z], [x - .0001, .0001, .019 + z],
                [x - .0001, -.0001, .019 + z], [x + .0001, 0., .018 + z]]
    return dict(hard_vertices_dip_m=hull(0., .006), rubber_vertices_dip_m=hull(.011, .002),
                anchor_dip_m=[0., 0., .020], outward_axis_dip=[1., 0., 0.])


class PadOrientationTests(unittest.TestCase):
    def test_two_degree_search_in_dip_coordinates_and_original_plan_fallback(self):
        z = torch.tensor([[0., 0., 1.]] * 3, dtype=torch.float64)
        # The DIP frames differ in world orientation. The matching palm frames
        # preserve their DIP-local relationship. Each pose must give the same geometry.
        dip = z.new_tensor([[0., 0., 0., 1.], [0., .5**.5, 0., .5**.5], [0., 0., 0., 1.]])
        palm = torch.eye(3, dtype=z.dtype).repeat(3, 1, 1)
        palm[1] = z.new_tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        landmarks = (.05 * z)[:, None]
        meshes = policy.geometry_tensors(geometry(), z)
        chosen, metrics = policy.choose_axis(z, dip, palm, landmarks, z.new_tensor([.01]), meshes,
                                             torch.tensor([True, True, False]))
        self.assertEqual(metrics['adjusted'].tolist(), [True, True, False])
        self.assertAlmostEqual(metrics['angle_deg'][0].item(), 22.)
        tilt = math.radians(22)
        torch.testing.assert_close(chosen[:2], z.new_tensor([[math.sin(tilt), 0., math.cos(tilt)]] * 2))
        torch.testing.assert_close(chosen[2], z[2], rtol=0, atol=0)
        self.assertTrue((metrics['predicted_gap_mm'][:2] < 6).all())
        self.assertTrue((metrics['rubber_lead_mm'][:2] >= .25).all())
        # Insufficient passive lead rejects the candidate, even on an unloaded lane.
        chosen, metrics = policy.choose_axis(z, dip, palm, landmarks, z.new_tensor([.049]), meshes,
                                             torch.ones(3, dtype=torch.bool))
        self.assertFalse(metrics['feasible'].any())
        torch.testing.assert_close(chosen, z, rtol=0, atol=0)

    def test_original_axis_already_exposes_pad_and_beyond_cap_is_unavailable(self):
        z = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
        args = (z, z.new_tensor([[0., 0., 0., 1.]]), torch.eye(3, dtype=z.dtype)[None],
                .050 * z[:, None], z.new_tensor([.01]))
        sample = geometry()
        for v in sample['hard_vertices_dip_m']:
            v[2] -= .005
        chosen, m = policy.choose_axis(*args, policy.geometry_tensors(sample, z), torch.tensor([True]))
        self.assertTrue(m['feasible'].item())
        self.assertFalse(m['adjusted'].item())
        torch.testing.assert_close(chosen, z, rtol=0, atol=0)
        sample = geometry()
        for v in sample['rubber_vertices_dip_m']:
            v[0] -= .006
        chosen, m = policy.choose_axis(*args, policy.geometry_tensors(sample, z), torch.tensor([True]))
        self.assertFalse(m['feasible'].item())
        torch.testing.assert_close(chosen, z, rtol=0, atol=0)

    def test_loaded_or_near_handoff_exactly_follows_existing_controller(self):
        f = fixtures.ContactTurn()
        f.setUp()
        candidate, baseline = policy.PadOrientationGuide(f.c, geometry()), policy.anchor.ApproachGuide(f.c)
        no = ~f.yes
        obs = dict(f.obs, natural_ready=f.yes, right_contact=no, index_contact=no, other_digit_contact=no,
                   right_total_impulse=f.q.new_tensor([1e-9, 0.]))
        # First lane already has a raw impulse. The other pad is only 15 mm from the surface.
        local = f.q.new_tensor([[.03, 0., -.085], [.03, 0., -.060]])
        tip = f.c.tip.fk(f.q[:, f.c.tip.ix])[0] + f.c.root
        f.bodies[:, :, :3] += f.c.root
        core = torch.cat((tip - local, f.q.new_tensor([[0., 0., 0., 1.]] * 2)), dim=1)
        for fresh in (f.yes, no):
            args = (f.q, f.q, f.q, fresh, f.yes, torch.ones_like(candidate.reference), core, tip, local, obs)
            torch.testing.assert_close(candidate.command(*args), baseline.command(*args), rtol=0, atol=0)
            self.assertEqual(candidate.receipt()['eligible_by_lane'], [False, False])
            self.assertEqual(candidate.receipt()['adjusted_by_lane'], [False, False])
            self.assertEqual(candidate.control_evidence()['pre_pad_orientation_first_plan'].tolist(), fresh.tolist())
            obs['right_total_impulse'].zero_()

    def test_archive_receipt_checks_eligibility_and_one_time_plan(self):
        values = dict(planned=True, eligible=True, feasible=True, adjusted=True, angle_deg=22.,
                      passive_lead_mm=4., predicted_gap_mm=5.8, rubber_lead_mm=.3)
        receipt = {k + '_by_lane': [v, 0] for k, v in values.items()}
        row = dict(lane=0, seed=12, pre_pad_orientation_first_plan=True, pre_support_ok=True,
                   pre_approach_any_touch=False, pre_approach_goal_reached=False, pre_normal_gap_m=.028)
        row.update({'pre_pad_orientation_' + k: v for k, v in values.items()})
        with TemporaryDirectory() as folder:
            trace = Path(folder) / 'turn.jsonl'
            def check(first, second):
                trace.write_text(json.dumps(first) + '\n' + json.dumps(second) + '\n')
                policy.verify_receipt(receipt, trace, dict(seeds=[12, 13]))
            following = dict(row, pre_pad_orientation_first_plan=False)
            check(row, following)
            for first, second in ((dict(row, pre_approach_any_touch=True), following),
                                  (row, dict(following, pre_pad_orientation_angle_deg=20.)),
                                  (row, row)):
                with self.assertRaises(ValueError):
                    check(first, second)


if __name__ == '__main__':
    unittest.main()
