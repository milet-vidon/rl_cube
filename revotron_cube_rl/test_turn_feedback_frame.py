import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from turn_feedback_frame import planning_frame_layer_angle


def wxyz(rotation):
    return rotation.as_quat()[[3, 0, 1, 2]]


class TurnFeedbackFrameTests(unittest.TestCase):
    def test_unchanged_core_retains_face_angle_and_nonzero_baseline(self):
        core = wxyz(Rotation.from_euler('xyz', [34., -29., 17.], degrees=True))
        for face in (-90., -.31, 0., 15., 90.):
            self.assertAlmostEqual(planning_frame_layer_angle(core, core, face), face, places=10)

    def test_core_corotation_contributes_in_same_turn_direction(self):
        initial = Rotation.from_euler('xyz', [21., 34., -58.], degrees=True)
        current = initial * Rotation.from_euler('z', -.465, degrees=True)
        self.assertAlmostEqual(planning_frame_layer_angle(wxyz(initial), wxyz(current), .027), .492, places=10)

    def test_common_world_rotation_does_not_change_feedback(self):
        initial = Rotation.from_euler('xyz', [21., 34., -58.], degrees=True)
        current = initial * Rotation.from_euler('xyz', [12., -18., -.465], degrees=True)
        world = Rotation.from_euler('xyz', [-71., 24., 109.], degrees=True)
        old = planning_frame_layer_angle(wxyz(initial), wxyz(current), 15.)
        transformed = planning_frame_layer_angle(wxyz(world*initial), wxyz(world*current), 15.)
        self.assertAlmostEqual(old, transformed, places=10)

    def test_tilted_core_matches_original_planner_expression(self):
        initial = Rotation.from_euler('xyz', [21., 34., -58.], degrees=True)
        current = initial * Rotation.from_euler('xyz', [32., -21., -7.], degrees=True)
        palm_reference = Rotation.from_euler('xyz', [-74., 10., 53.], degrees=True)
        layer_world = current * Rotation.from_euler('z', -15., degrees=True)
        # A palm rigidly attached to that layer follows its full world delta.
        palm_target = layer_world * initial.inv() * palm_reference
        original_planner = initial.inv() * palm_target * palm_reference.inv() * initial
        expected = -np.degrees(original_planner.as_rotvec()[2])
        actual = planning_frame_layer_angle(wxyz(initial), wxyz(current), 15.)
        self.assertAlmostEqual(actual, expected, places=10)
        core_projection = -np.degrees((initial.inv()*current).as_rotvec()[2])
        self.assertGreater(abs(actual-(core_projection+15.)), .05)

    def test_nonfinite_angles_and_quaternions_rejected(self):
        identity = [1., 0., 0., 0.]
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):planning_frame_layer_angle(identity, identity, value)
            with self.assertRaises(ValueError):planning_frame_layer_angle([1.,value,0.,0.], identity, 15.)
            with self.assertRaises(ValueError):planning_frame_layer_angle(identity, [1.,value,0.,0.], 15.)

    def test_invalid_shape_zero_norm_and_quaternion_sign(self):
        identity = [1., 0., 0., 0.]
        for bad in ([0.,0.,0.,0.], [1.,0.,0.], [[1.,0.,0.,0.]]):
            with self.assertRaises(ValueError):planning_frame_layer_angle(identity, bad, 15.)
        q=wxyz(Rotation.from_euler('xyz',[10.,20.,30.],degrees=True))
        self.assertAlmostEqual(planning_frame_layer_angle(identity,q,15.),planning_frame_layer_angle(identity,-q,15.))

if __name__=='__main__':unittest.main()
