"""Static checks for frame transforms, signed flexion and unchanged joint bounds."""
import unittest

import numpy as np

from kinematics import RobotKinematics
from natural_face90_pad_preshape_check import ShapeCheck, pose_matrix


class PreshapeTest(unittest.TestCase):
    def test_finger_flexion_and_joint_limits(self):
        checker = ShapeCheck.__new__(ShapeCheck)
        checker.fk = RobotKinematics()
        q = {f'right_index_{part}_joint': value for part, value in zip(('MCP', 'PIP', 'DIP'), (.24, .22, .16))}
        pose = checker.joint_pose(q, {'index': -6})
        expected = checker.fk.fk({k: v-np.deg2rad(6) for k, v in q.items()})
        np.testing.assert_allclose(pose['right_index_DIP_link'], expected['right_index_DIP_link'])
        np.testing.assert_allclose(pose['right_middle_DIP_link'], checker.fk.fk(q)['right_middle_DIP_link'])
        self.assertIsNone(checker.joint_pose(q, {'index': -40}))
        self.assertEqual(q['right_index_DIP_joint'], .16)

    def test_root_orientation_applied_to_face_normal(self):
        # Recorded pose format is XYZ / XYZW; this rotates +Z into +Y in root coordinates.
        root = pose_matrix((1, 2, 3, 2**-.5, 0, 0, 2**-.5))
        np.testing.assert_allclose(root[:3, :3].T @ (0,0,1), (0,1,0), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
