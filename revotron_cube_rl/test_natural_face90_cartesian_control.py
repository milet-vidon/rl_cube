import math
import unittest

import torch

import natural_face90_cartesian_control as cart
from natural_face90_pad_preshape_policy import PreshapeGuide
from test_natural_face90_pad_orientation_policy import geometry
import test_natural_face90_pad_preshape_policy as preshape_fixture


class CartesianControlTests(unittest.TestCase):
    def setUp(self):
        # Build the actual URDF FK and clearance controller with two CPU lanes.
        f = preshape_fixture.PreshapeTests()
        f.setUp()
        self.f = f
        root = f.q.new_tensor([[0., 0., f.c.root[2], 0., 0., 0., 1.]] * 2)
        f.c.e.robot.get_root_transforms = lambda: root
        self.g = cart.CartesianGuide(f.c, geometry())
        self.addCleanup(self.g.placo.close)
        self.ref = PreshapeGuide(f.c, geometry())
        self.yes, self.no = f.f.yes, ~f.f.yes
        self.obs = f.obs.copy()
        self.local = f.q.new_tensor([[.030, 0., -.068]] * 2)
        self.tip = f.c.tip.fk(f.q[:, f.c.tip.ix])[0] + f.c.root
        _, palm, _, _ = f.c.rp.fk(f.q[:, f.c.rp.ix])
        f.c.radial[:] = f.q.new_tensor([1., 0.])
        for guide in (self.g, self.ref):
            guide.axis[:] = palm[:, 2, :]
            guide.vectors[:] = .100 * guide.axis[:, None]
            guide.orientation[:] = palm

    def tick(self, guide, latent, fresh=None):
        f = self.f
        core = torch.cat((self.tip - self.local, f.q.new_tensor([[0., 0., 0., 1.]] * 2)), 1)
        return guide.command(f.q, f.q, f.q, self.no if fresh is None else fresh,
                             self.yes, latent, core, self.tip, self.local, self.obs)

    def row(self, i):
        ev = {k:v[i].item() for k,v in self.g.control_evidence().items()}
        ev.update({'requested_' + axis + '_m': self.g.goal_local[i, j].item() for j,axis in enumerate('xyz')})
        ev['drive_command'] = self.g.drive_contact[i].item()
        return ev

    def test_zero_cartesian_preserves_nominal_path_with_real_placo(self):
        for _ in range(2):
            self.tick(self.ref, self.f.q.new_zeros(2, 7))
            new = self.tick(self.g, self.f.q.new_zeros(2, 3))
            self.assertTrue(torch.isfinite(new).all())
            self.assertEqual(self.g.placo.receipt['backend'], 'placo')
            torch.testing.assert_close(self.ref.goal_local, self.g.goal_local, rtol=0, atol=0)
            for i in (0, 1):
                cart.verify_frame(self.row(i))
        with self.assertRaises(ValueError):
            self.tick(self.g, self.f.q.new_zeros(2, 7))

    def test_rotated_translated_lane_maps_to_same_robot_base_ik(self):
        f, g = self.f, self.g
        root = f.q.new_tensor([[0., 0., 1.15, 0., 0., 0., 1.], [2., -3., 4., .5, .5, .5, .5]])
        f.c.e.robot.get_root_transforms = lambda: root
        rotations = cart.quat_matrix(root[:, 3:])
        tip = root[:, :3] + cart.mv(rotations, self.tip - f.c.root)
        core = torch.cat((tip - cart.mv(rotations, self.local), root[:, 3:]), 1)
        g.orientation = rotations @ g.orientation
        action = f.q.new_tensor([[1., .5, -.25]] * 2)
        command = g.command(f.q, f.q, f.q, self.no, self.yes, action, core, tip, self.local, self.obs)
        torch.testing.assert_close(command[0], command[1], rtol=0., atol=1e-10)
        self.assertLess(self.row(1)['pre_placo_fk_pad_error_m'], 1e-10)
        cart.verify_frame(self.row(1))

    def test_cartesian_goal_reaches_ik_and_preserves_handoff_joints(self):
        # On this nZ face +Y, +X are the positive radial/tangent directions.
        self.f.c.radial[:] = self.f.q.new_tensor([0., 1.])
        self.local[:,:2] = self.f.q.new_tensor([0., .030])
        nominal = self.tick(self.g, self.f.q.new_zeros(2, 3))
        self.tick(self.ref, self.f.q.new_zeros(2, 7))
        action = self.f.q.new_tensor([[1., 1., -1.], [0., 0., 2.]])
        command = self.tick(self.g, action)
        diff = self.g.goal_local - self.ref.goal_local
        self.assertGreater(diff[0, 0].item(), 0.)  # clockwise nZ tangent from +Y is +X
        self.assertGreater(diff[0, 1].item(), 0.)
        self.assertLess(diff[0, 2].item(), 0.)
        self.assertLessEqual(diff.norm(dim=1).max().item(), cart.MAX_RESIDUAL_M + 1e-12)
        self.assertAlmostEqual(diff[1, 2].item(), 0., places=12)  # nominal is already at 0.75 mm
        self.assertGreater((command[0] - nominal[0]).abs().max().item(), 1e-7)
        self.assertLessEqual(float((command - self.f.q).abs().max()), .012001)
        passive = [j for j in range(56) if j not in self.g.arm]
        torch.testing.assert_close(command[:,passive], self.f.q[:,passive], rtol=0, atol=0)
        self.assertEqual(self.row(0)['pre_cartesian_mode'], 2)
        self.assertTrue(self.row(0)['pre_approach_anchor'])
        for i in (0, 1):
            cart.verify_frame(self.row(i))
        with self.assertRaises(ValueError):
            cart.verify_frame(dict(self.row(0), requested_z_m=-.04))
        with self.assertRaises(ValueError):
            cart.verify_frame(dict(self.row(0), pre_cartesian_latent_0=-.1))

    def test_standoff_holds_normal_and_disallowed_lanes_ignore_actions(self):
        self.local[:,0] = .040  # outside alignment window, still unloaded
        self.obs['right_total_impulse'] = self.f.q.new_tensor([0., 1e-9])
        large = self.f.q.new_full((2, 3), 8.)
        self.tick(self.g, large)
        self.assertEqual([self.row(i)['pre_cartesian_mode'] for i in (0, 1)], [1, 0])
        self.assertEqual(self.row(0)['pre_cartesian_delta_z_m'], 0.)
        self.assertTrue(all(self.row(1)['pre_cartesian_delta_' + k + '_m'] == 0. for k in 'xyz'))
        self.obs['support'] = self.no
        self.tick(self.g, large)
        for i in (0, 1):
            self.assertEqual(self.row(i)['pre_cartesian_mode'], 0)
            cart.verify_frame(self.row(i))

    def test_drive_respects_nominal_lead_and_off_marker_contact_retracts(self):
        self.obs.update(right_total_impulse=self.f.q.new_tensor([.001, .002]),
                        right_contact=self.yes, index_contact=self.yes)
        self.local[:, 2] = -.045
        self.local[1, 2] = -.053   # distal contact with pad outside the qualification band
        self.g.streak[:] = 3
        self.g.sweep_deg[:] = 15.99
        self.tick(self.g, self.f.q.new_tensor([[-4., 4., 4.], [-4., 4., -4.]]))
        row = self.row(0)
        self.assertEqual(row['pre_cartesian_mode'], 3)
        self.assertEqual(row['pre_cartesian_delta_z_m'], 0.)
        lead = math.degrees(math.atan2(-row['requested_y_m'], row['requested_x_m']))
        self.assertLessEqual(lead, 16. + 1e-9)
        cart.verify_frame(row)
        self.assertEqual(self.row(1)['pre_cartesian_mode'], 0)
        self.assertTrue(self.g.recovering[1])
        cart.verify_frame(self.row(1))
