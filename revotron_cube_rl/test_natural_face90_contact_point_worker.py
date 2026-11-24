"""Offline regressions for selecting material points and transferring a frozen specialist."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import natural_face90_contact_point_worker as worker
import natural_face90_v6_long as source
import test_natural_face90_turn_guide as fixtures


class ContactPointWorkerTest(unittest.TestCase):
    def test_signed_offset_and_asynchronous_lanes(self):
        base = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.float64)
        angle = math.radians(7)
        phase = worker.PostPhase.__new__(worker.PostPhase)
        phase.req = {'contact_point_offset_deg': 7}
        phase.controller = SimpleNamespace(radial=base.clone())
        phase.base_radial = torch.zeros_like(base)
        phase.selected_radial = torch.zeros_like(base)
        phase.select_material_point(torch.tensor([True, False]))
        torch.testing.assert_close(phase.controller.radial[1], base[1])
        phase.select_material_point(torch.tensor([False, True]))
        phase.select_material_point(torch.tensor([False, False]))
        expected = base.new_tensor([[math.cos(angle), -math.sin(angle)],
                                    [-math.sin(angle), -math.cos(angle)]])
        torch.testing.assert_close(phase.selected_radial, expected)
        torch.testing.assert_close(phase.base_radial, base)
        torch.testing.assert_close(worker.material_radial(phase.controller.radial, -7), base)

    def test_regrasp_targets_reachable_forward_interior_and_follows_measured_rotation(self):
        forward = torch.tensor([[1., 0.]] * 2, dtype=torch.float64)
        angle = torch.tensor([45., 90.], dtype=torch.float64)
        chosen, improvement = worker.forward_regrasp_point(forward, forward, angle)
        torch.testing.assert_close(chosen, forward.new_tensor([[math.sqrt(.5), math.sqrt(.5)], [0., 1.]]))
        marker = worker._point(chosen, angle, torch.zeros_like(angle))
        torch.testing.assert_close(marker[:, :2], .030 * forward)
        self.assertTrue((improvement > .002).all())
        self.assertTrue(((.030 * chosen.abs() - .015).abs() >= .006).all())
        unchanged, gain = worker.forward_regrasp_point(chosen, forward, angle)
        torch.testing.assert_close(unchanged, chosen)
        torch.testing.assert_close(gain, torch.zeros_like(gain))

    def test_stalled_regrasp_requires_unloaded_withdrawal_and_new_attributed_motion(self):
        yes = torch.ones(7, dtype=torch.bool)
        guide = SimpleNamespace(recovering=yes.clone(), recovery_reason=torch.full((7,), 3),
                                clear_z=torch.full((7,), -.078))
        obs = {k:yes.clone() for k in ('support','natural_ready','static_contact','structure_ok','allowed_layer_state')}
        obs.update({k:~yes for k in ('wrong_contact','contact_goal_reached','right_contact','index_contact','other_digit_contact')})
        obs['right_total_impulse'] = torch.tensor([0., 1e-9, 0., 0., 0., 0., 0.])
        credit = torch.tensor([15., 15., 0., 15., 15., 15., 15.])
        last = torch.tensor([0., 0., 0., 10., 0., 0., 0.])
        pad_z = torch.full((7,), -.076)
        pad_z[4] = -.069
        oriented = yes.clone()
        oriented[5] = False
        guide.recovery_reason[6] = 1
        allowed = worker.unloaded_after_stall(yes, guide, obs, pad_z, oriented, credit, last)
        self.assertEqual(allowed.tolist(), [True, False, False, False, False, False, False])

    def test_unchanged_guide_for_zero_and_extended_sweep_for_qualified_90(self):
        f = fixtures.ContactTurn()
        f.setUp()
        f.obs.update(natural_ready=f.yes, other_digit_contact=~f.yes)
        f.c.radial = worker.material_radial(f.radial, 7)
        goal_ang = torch.full((2,), 84., dtype=torch.double)
        marker = worker._point(f.c.radial, goal_ang, torch.zeros_like(goal_ang))
        f.obs['face_deg'] = goal_ang
        # At the same measured layer state, v4 can only request 18°, whereas
        # the promoted guide can request 93° under qualifying isolated contact.
        for goal, expected in ((15, 18.), (90, 93.)):
            with self.subTest(goal=goal):
                f.c.deg = float(goal)
                guide = worker.MaterialGuide(f.c)
                guide._path(f.yes, f.yes, f.c.radial, marker, f.yes, f.obs)
                for _ in range(60):
                    guide._path(~f.yes, f.yes, f.c.radial, marker, f.yes, f.obs)
                torch.testing.assert_close(guide.sweep_deg, torch.full_like(goal_ang, expected))
                f.obs['other_digit_contact'][0] = True
                guide._path(~f.yes, f.yes, f.c.radial, marker, f.yes, f.obs)
                self.assertFalse(guide.drive_contact[0])
                self.assertTrue(guide.recovering[0])
                f.obs['other_digit_contact'][0] = False

    def test_signed_credit_requires_command_two_isolated_endpoints_and_accounts_for_slip(self):
        track = worker.SignedDriveResponse(1, 'cpu')
        last = torch.zeros(1, dtype=torch.double)
        yes = torch.tensor([True])
        for value,drive,other in [(0,False,False), (2,True,False), (1,True,False), (6,True,True),
                                  (7,True,False), (8,True,False), (0,False,False), (3,True,False)]:
            angle = last.new_tensor([value])
            track.update(angle, yes, torch.tensor([drive]), yes, torch.tensor([other]), last)
            last = angle
        self.assertEqual(track.signed.item(), -3)
        self.assertEqual(track.credit.item(), -3)
        self.assertEqual(track.peak_credit.item(), 2)

    def test_repeated_push_and_forward_drift_cannot_make_up_an_undriven_angle_range(self):
        track = worker.SignedDriveResponse(1, 'cpu')
        last = torch.zeros(1, dtype=torch.double)
        yes = torch.tensor([True])
        for value,contact,drive in [(0,True,False),(30,True,True),(0,False,False),(0,True,False),
                                    (30,True,True),(60,False,False),(60,True,False),(90,True,True)]:
            angle = last.new_tensor([value])
            track.update(angle, torch.tensor([contact]), torch.tensor([drive]), yes, ~yes, last)
            last = angle
        self.assertEqual(track.peak_credit.item(), 60.)

    def test_terminal_geometry_cannot_claim_90_from_one_short_isolated_drive(self):
        phase = worker.PostPhase.__new__(worker.PostPhase)
        phase.req = {'target_deg': 90}
        yes = torch.tensor([True, True])
        phase.response = SimpleNamespace(active=yes, credit=torch.tensor([89.6, 90.2]))
        phase.isolated_goal = ~yes
        phase.obs = dict(contact_goal_reached=yes, at_goal=yes, face_deg=torch.tensor([90.2,90.2]))
        observed = dict(success=yes.clone(), failed=~yes, reason=['', ''])
        with patch.object(worker, '_observe_turn', return_value=observed):
            result = phase.observe(yes, yes, yes, 10)
        self.assertEqual(result['success'].tolist(), [False, True])
        self.assertEqual(result['failed'].tolist(), [True, False])
        self.assertEqual(result['reason'], ['insufficient_isolated_drive_angle', ''])
        self.assertEqual(phase.obs['contact_goal_reached'].tolist(), [False, True])
        self.assertEqual(phase.obs['physical_contact_goal_reached'].tolist(), [True, True])
        # Release persists after the driving finger has withdrawn.
        phase.response.active = ~yes
        phase.response.credit[:] = 89.
        phase.obs['contact_goal_reached'] = yes
        with patch.object(worker, '_observe_turn', return_value=observed):
            phase.observe(yes, yes, yes, 11)
        self.assertEqual(phase.obs['contact_goal_reached'].tolist(), [False, True])

    def test_checkpoint_restore_and_promotion_bind_angle_and_goal(self):
        model, opt = Mock(), Mock()
        contract = dict(schema='contact-point', software={'worker':'v1'}, target_deg=15,
                        action_mode=worker.ACTION_MODE, contact_point_offset_deg=7)
        ck = dict(contract=contract, policy={}, optimizer={}, iteration=75)
        req = dict(iteration=76, learn=True, role='train', target_deg=15, checkpoint='parent', checkpoint_sha='hash')
        worker.restore_expert(model, opt, ck, contract, req)
        opt.load_state_dict.assert_called_once_with({})
        with self.assertRaises(ValueError):
            worker.restore_expert(model, opt, ck, dict(contract, contact_point_offset_deg=-7), req)
        with self.assertRaises(ValueError):
            worker.restore_expert(model, opt, ck, dict(contract, target_deg=90), dict(req, target_deg=90))
        opt.reset_mock()
        provenance = dict(iteration=75, checkpoint='parent', sha='hash')
        worker.restore_expert(model, opt, ck, dict(contract, target_deg=90),
                              dict(req, target_deg=90, promote_from_15=provenance))
        opt.load_state_dict.assert_not_called()

    def test_only_exact_archived_r70_can_initialize_offset(self):
        contract = dict(schema='contact-point', target_deg=15, software={'all':'new'},
                        action_mode=worker.ACTION_MODE, contact_point_offset_deg=-7)
        old = dict(schema=source.SCHEMA, target_deg=15, software={'v3':'old'},
                   action_mode=worker.centered.previous.ACTION_MODE)
        origin = dict(iteration=70, schema=source.SCHEMA, sha=worker.R70_SHA,
                      checkpoint='source.pt', software=old['software'])
        req = dict(iteration=70, role='validate', learn=False, paired_validation=True,
                   checkpoint=origin['checkpoint'], checkpoint_sha=origin['sha'], transfer=origin, target_deg=15)
        ck = dict(contract=old, policy={}, optimizer={}, iteration=70)
        model, opt = Mock(), Mock()
        worker.restore_expert(model, opt, ck, contract, req)
        model.load_state_dict.assert_called_once_with({}, strict=True)
        opt.load_state_dict.assert_not_called()
        with self.assertRaises(ValueError):
            worker.restore_expert(model, opt, ck, contract, dict(req, checkpoint_sha='other'))


if __name__ == '__main__':
    unittest.main()
