"""CPU-only operator actions and real Placo; no Kit session or rollout."""
import json
import math
from types import SimpleNamespace
import unittest

import torch

from contact_hold_cuda_teacher_math import log_rot
from kinematics import RobotKinematics
from natural_face90_demo_control import ManualControl, MOVE_M, ROTATE_RAD, face_orbit
from natural_face90_placo import HERE


class ManualControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads((HERE/'config.json').read_bytes())
        cls.names = config['joint_names']
        measured = json.loads((HERE/'natural_face90_v5_regression.json').read_bytes())['r27_endpoint']['q']
        cls.q = torch.tensor([measured], dtype=torch.float64)
        limits = RobotKinematics().limits
        env = SimpleNamespace(n=1, names=cls.names, cfg=config, home=cls.q, ids=list(range(56)), dt=1/240,
                              dec=8, robot=SimpleNamespace(get_dof_positions=lambda: cls.q.clone()),
                              limits=cls.q.new_tensor([[limits[name] for name in cls.names]]),
                              step=cls.q.new_full((1, 56), .025), vel=cls.q.new_full((1, 56), 1.57))
        cls.control = ManualControl(env)
        cls.addClassCleanup(cls.control.close)

    def test_zero_or_finger_only_preserves_full_arm_preload(self):
        prev = self.q.clone()
        prev[:, self.names.index('Joint1_R')] += .02
        self.assertTrue(torch.equal(self.control.step(prev, {}), prev))
        for group in ('thumb', 'index', 'opposition'):
            joints = self.control.fingers[group]
            prev[:, [i for i,_ in joints]] = .4
            nxt = self.control.step(prev, {group:1.})
            diff = (nxt - prev).flatten()
            self.assertEqual(diff.nonzero().flatten().tolist(), sorted(i for i,_ in joints))
            for i, gain in joints:
                self.assertAlmostEqual(diff[i].item(), math.radians(gain))
            self.assertTrue(torch.allclose(self.control.step(nxt, {group:-1.}), prev, atol=1e-12,rtol=0))

    def test_palm_translation_and_rotation_in_robot_base(self):
        ctl = self.control
        palm, palm_R, _, _ = ctl.palm.fk(self.q[:, ctl.arm])
        # The stored R27 endpoint is close to the torso. Exercise a reachable
        # diagonal and check that a command further into the torso is held.
        self.assertTrue(torch.equal(ctl.step(self.q, {'move':[0,1,0]}), self.q))
        for action in ({'move': [-1,-1,0]}, {'rotate':[0,0,1]}):
            target = ctl.step(self.q, action)
            passive = [i for i in range(56) if i not in ctl.arm]
            self.assertTrue(torch.equal(target[:,passive], self.q[:,passive]))
            self.assertLessEqual(float((target-self.q).abs().max()), .0120001)
            new, new_R, _, _ = ctl.palm.fk(target[:, ctl.arm])
            rotation = log_rot(new_R @ palm_R.transpose(1,2))[0]
            if 'move' in action:
                expected = self.q.new_tensor([[-MOVE_M/math.sqrt(2),-MOVE_M/math.sqrt(2),0.]])
                self.assertLess(float((new-palm-expected).norm()), .00015)
                self.assertLess(float(rotation.norm()), .0015)
            else:
                self.assertGreater(float(rotation[2]), .6*ROTATE_RAD)
                self.assertLess(float((new-palm).norm()), .0003)

    def test_invalid_actions_and_joint_limits(self):
        for action in ({'other': 1}, {'move': [1, 0]}, {'rotate': [0,0,float('nan')]}, {'thumb': 1.01},
                       {'move': [1, 0, 0], 'orbit': 1}):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.control.step(self.q, action)
        prev = self.q.clone()
        for i, _ in self.control.fingers['thumb']:
            prev[:,i] = self.control.env.limits[:,i,1]
        self.assertTrue(torch.equal(self.control.step(prev, {'thumb':1.}),prev))

    def test_orbit_of_a_held_point_about_measured_negative_z(self):
        # The robot and cube are tilted 90 degrees about world X. In the
        # robot frame the face axis is -Z and the core is at (0.4,0.3,0).
        h = math.sqrt(.5)
        root = self.q.new_tensor([[.1, .2, .3, h, 0., 0., h]])
        cube = self.q.new_tensor([[.5, .2, .6, h, 0., 0., h]])
        tip = self.q.new_tensor([[.43, .3, -.045]])
        for sign in (1, -1):
            angle = sign * ROTATE_RAD
            delta, rotation = face_orbit(cube, root, tip, angle)
            expected = self.q.new_tensor([[.03*(math.cos(angle)-1.), -.03*math.sin(angle), 0.]])
            torch.testing.assert_close(delta, expected)
            torch.testing.assert_close(log_rot(rotation), self.q.new_tensor([[0., 0., -angle]]))

        from unittest.mock import patch
        ctl = self.control
        with patch.object(ctl.env, 'views', {'Core': SimpleNamespace(get_transforms=lambda: cube)}, create=True), \
             patch.object(ctl.env.robot, 'get_root_transforms', lambda: root, create=True), \
             patch.object(ctl.placo, 'solve_batch', wraps=ctl.placo.solve_batch) as solve:
            result = ctl.step(self.q, {'orbit': 1})
        solve.assert_called_once()
        passive = [i for i in range(56) if i not in ctl.arm]
        self.assertTrue(torch.equal(result[:, passive], self.q[:, passive]))


if __name__ == '__main__':
    unittest.main()
