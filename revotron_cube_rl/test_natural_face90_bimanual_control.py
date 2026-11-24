"""Measured FK, opposed flank geometry and real batched Placo (CPU only)."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from kinematics import RobotKinematics
from natural_face90_bimanual_control import BimanualGuide, flank_targets
from natural_face90_v5_controller import NaturalFace90Controller


class OpposedControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        here = Path(__file__).resolve().parent
        cfg = json.loads((here/'config.json').read_bytes())
        cls.names = cfg['joint_names']
        measured = json.loads((here/'natural_face90_v5_regression.json').read_bytes())['r27_endpoint']['q']
        cls.q = torch.tensor([measured]*3, dtype=torch.float64)
        model = RobotKinematics()
        limits = cls.q.new_tensor([[model.limits[n] for n in cls.names]])
        root = cls.q.new_tensor([[0., 0., cfg['robot_z'], 0., 0., 0., 1.]]).expand(3,-1)
        e = SimpleNamespace(names=cls.names, home=cls.q, cfg=cfg, dt=1/240, dec=8,
                            step=cls.q.new_full((1,56), .025), vel=cls.q.new_full((1,56), 1.57),
                            limits=limits, robot=SimpleNamespace(get_root_transforms=lambda:root))
        cls.guide = BimanualGuide(NaturalFace90Controller(e, [0.]*6))
        cls.addClassCleanup(cls.guide.placo.close)
        cls.core = root.clone()
        cls.core[:, :3] += cls.q.new_tensor([.56, .25, .20])
        cls.q = cls.q.clamp(limits[:,:,0], limits[:,:,1])

    def test_mirrored_flank_pair_tracks_measured_layer_not_whole_cube(self):
        li = self.q.new_tensor([[.05, -.017, .021], [.05,-.017,.021], [.006,-.048,.025]])
        lt = self.q.new_tensor([[-.05, .006, .035], [-.05,.006,.035], [-.011,.048,.031]])
        deg = self.q.new_tensor([0.,90.,0.])
        i,t,normal = flank_targets(li, lt, deg)
        torch.testing.assert_close(i, self.q.new_tensor([[.045,-.017,-.03],[-.017,-.045,-.03],[.006,-.045,-.03]]))
        torch.testing.assert_close(t, self.q.new_tensor([[-.045,.006,-.03],[.006,.045,-.03],[-.011,.045,-.03]]))
        torch.testing.assert_close(((i-t)*normal).sum(1), deg.new_full((3,), .09))
        g = self.guide.geometry(self.q, self.core, deg)
        self.assertEqual(g['index_local'].shape, (3,3))
        self.assertTrue(torch.isfinite(g['palm_opposition_cos']).all())

    def test_qualified_orbit_and_lane_masking_with_actual_placo(self):
        g = self.guide
        q = self.q
        on = torch.tensor([True, True, False])
        closed = torch.tensor([True, False, True])
        contacts = dict(left_static=torch.ones_like(on), thumb=closed, index=closed, bilateral=closed)
        a = q.new_zeros((3,7))
        a[0,6] = 1.
        deg = q.new_zeros(3)
        with patch.object(g.placo, 'solve_batch', wraps=g.placo.solve_batch) as solve:
            new = g.command(q,q,on,a,self.core,deg,contacts)
        qs, demand, rotations, recover = solve.call_args.args
        self.assertEqual(qs.shape, (2,56))
        radius = g.geometry(q,self.core,deg)['index_local'][0]
        angle = math.radians(.35+.5*math.tanh(1.))
        c, s = math.cos(angle), math.sin(angle)
        expected = q.new_tensor([c*radius[0]+s*radius[1], -s*radius[0]+c*radius[1], radius[2]]) - radius
        # The Placo target is capped at 2.5 mm even for a remote synthetic core.
        expected *= min(1., .0025/float(expected.norm()))
        torch.testing.assert_close(q.new_tensor(demand[0]), expected)
        torch.testing.assert_close(g.orbit_rad, q.new_tensor([angle,0.,0.]))
        self.assertFalse(recover.any())
        self.assertTrue(torch.equal(q[2], new[2]))
        fingers = g.finger_ids.tolist()
        passive = [i for i in range(56) if i not in g.arm+fingers]
        self.assertTrue(torch.equal(new[:,passive], q[:,passive]))
        self.assertLessEqual((new[:,g.arm]-q[:,g.arm]).abs().max().item(), .012001)
        self.assertTrue(torch.equal(new[0,fingers],q[0,fingers]))
        self.assertGreater(float((new[1,fingers]-q[1,fingers]).abs().max()), .0001)

    def test_no_drive_without_support_and_opens_on_release(self):
        q, g = self.q, self.guide
        active = torch.ones(3, dtype=torch.bool)
        release = torch.tensor([False, True, False])
        contacts = dict(left_static=torch.tensor([False,True,True]), thumb=active, index=~active, bilateral=~active)
        target = q.clone()
        target[:, g.finger_ids] = .45
        new = g.command(target, target, active, q.new_ones(3,7), self.core, q.new_zeros(3), contacts, retreat=release)
        self.assertTrue(torch.equal(new[0], target[0]))
        self.assertTrue((new[1, g.finger_ids] < target[1, g.finger_ids]).all())
        self.assertTrue((g.orbit_rad==0.).all())

    def test_single_contact_keeps_its_first_measured_cube_relative_pivot(self):
        q, g = self.q, self.guide
        on = torch.ones(3, dtype=torch.bool)
        zeros = torch.zeros_like(on)
        thumb = dict(left_static=on, index=zeros, thumb=on, bilateral=zeros)
        none = dict(thumb, thumb=zeros)
        action = q.new_zeros((3,7))
        deg = q.new_zeros(3)
        g.command(q, q, on, action, self.core, deg, none)
        g.command(q, q, on, action, self.core, deg, thumb)
        first = g.geometry(q, self.core, deg)['thumb_local']
        torch.testing.assert_close(g.pivot_local, first)
        changed = self.core.clone()
        changed[:, 0] += .001
        g.command(q, q, on, action, changed, deg, thumb)
        torch.testing.assert_close(g.pivot_local, first)
        self.assertTrue((g.orbit_rad == 0.).all())
        g.command(q, q, on, action, changed, deg, none)
        g.command(q, q, on, action, changed, deg, thumb)
        torch.testing.assert_close(g.pivot_local, g.geometry(q, changed, deg)['thumb_local'])


if __name__=='__main__':
    unittest.main()
