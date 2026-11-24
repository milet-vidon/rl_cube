"""Exercise real isolated Placo solves without launching Isaac or a trainer."""
import json
import unittest

import numpy as np

from kinematics import RobotKinematics
from natural_face90_placo import HERE, PALM, RIGHT
from natural_face90_placo_client import PlacoClient, self_check


class PlacoTests(unittest.TestCase):
    def test_reference_fk_and_solved_calibrated_tip(self):
        results = self_check()
        self.assertEqual(results['receipt']['version'], '0.9.23')
        self.assertLess(results['max_fk_diff_m'], 1e-8)
        self.assertLess(results['max_position_error_after_step_m'], .00025)

    def test_loaded_locked_joints_and_recoverable_measured_limit_violation(self):
        cfg = json.loads((HERE / 'config.json').read_bytes())
        names = cfg['joint_names']
        q = np.array(json.loads((HERE / 'natural_face90_v5_regression.json').read_bytes())['r27_endpoint']['q'])
        measured = np.tile(q, (3, 1))
        model = RobotKinematics()
        for j in ('Joint6_L', 'right_index_MCP_joint'):
            measured[0, names.index(j)] = model.limits[j][1] + .016
        # An already exceeded right joint can only return at its 30 Hz rate.
        j = 'Joint3_R'
        measured[1, names.index(j)] = model.limits[j][1] + .030
        palm = np.stack([model.fk_link(dict(zip(names, lane)), PALM)[:3, :3] for lane in measured])
        error = np.tile([.0005, 0., 0.], (3, 1))
        client = PlacoClient(names, 1 / 30)
        self.addCleanup(client.close)
        out = client.solve_batch(measured, error, palm, [False] * 3)
        base = client.solve_batch(measured[-1:], error[-1:], palm[-1:], [False])
        np.testing.assert_allclose(out[2], base[0], rtol=0., atol=1e-9)
        np.testing.assert_array_less(np.abs(out - measured[:, [names.index(n) for n in RIGHT]]), .0120001)
        self.assertLess(out[1, RIGHT.index(j)], measured[1, names.index(j)] - .0119)
