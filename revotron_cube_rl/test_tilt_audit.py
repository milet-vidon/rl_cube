import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from tilt_audit import assess_tilt, tilt_error


def held_row():
    rotation = Rotation.from_euler('x', 90, degrees=True)
    center = np.array([.5, 0., 1.1])
    points = []
    for side, axis, sign in [('left', 0, -1), ('right', 1, 1)]:
        for finger, offset in zip(('thumb', 'index', 'middle', 'ring', 'little'), (-.02, -.02, -.007, .007, .02)):
            point = np.zeros(3)
            point[axis] = .0285 * (sign if finger == 'thumb' else -sign)
            point[2] = offset
            points.append(dict(side=side, finger=finger, impulse_Ns=.001,
                               position_world_m=(center + rotation.apply(point)).tolist()))
    return dict(phase='bilateral_hold', cube_position=center.tolist(),
                cube_quaternion_wxyz=rotation.as_quat()[[3, 0, 1, 2]].tolist(),
                tilt_reference_wxyz=[1., 0., 0., 0.], finger_contact_points=points,
                height_m=.18, linear_speed=0., angular_speed=0., cube_table_contact=False,
                self_contact_impulse=0., robot_table_impulse=0., arm_soft_limit_hit=False,
                actual_arm_soft_limit_violation=False, hard_limit_hit=[False] * 56, drop=False)


class MeasuredTiltAuditTests(unittest.TestCase):
    def test_90deg_opposition_hold(self):
        self.assertTrue(assess_tilt([held_row() for _ in range(90)], True, full_episode=False)['accepted'])

    def test_wrong_rotation_axis(self):
        row = held_row()
        row['cube_quaternion_wxyz'] = Rotation.from_euler('z', 90, degrees=True).as_quat()[[3, 0, 1, 2]]
        self.assertGreater(tilt_error(row), 100.)
        self.assertFalse(assess_tilt([row] * 90, True, full_episode=False)['accepted'])

    def test_controller_success_counter_cannot_replace_contacts(self):
        row = held_row()
        row['right_hold_steps'] = 900
        row['finger_contact_points'] = [p for p in row['finger_contact_points'] if p['side'] == 'left']
        self.assertFalse(assess_tilt([row] * 90, True, full_episode=False)['accepted'])

    def test_wrong_finger_faces(self):
        row = held_row()
        for p in row['finger_contact_points']:
            if p['side'] == 'right':
                p['finger'] = 'thumb' if p['finger'] != 'thumb' else 'index'
        self.assertFalse(assess_tilt([row] * 90, True, full_episode=False)['accepted'])

    def test_interrupted_hold_does_not_accumulate(self):
        rows = [held_row() for _ in range(100)]
        rows[50]['cube_table_contact'] = True
        self.assertFalse(assess_tilt(rows, True, full_episode=False)['accepted'])

    def test_any_collision_or_limit_event_rejects_episode(self):
        for field, value in [('self_contact_impulse', .001), ('robot_table_impulse', .001), ('arm_soft_limit_hit', True)]:
            rows = [held_row() for _ in range(100)]
            rows[0][field] = value
            self.assertFalse(assess_tilt(rows, True, full_episode=False)['accepted'])

    def test_resumed_hold_cannot_pass_full_episode_gate(self):
        self.assertFalse(assess_tilt([held_row() for _ in range(90)], regrasp=True)['accepted'])

    def test_return_to_table_then_regrasp_still_fails(self):
        rows = [held_row() for _ in range(100)]
        rows[0]['phase'] = 'right_route'
        rows[0]['cube_table_contact'] = True
        self.assertFalse(assess_tilt(rows, True, full_episode=False)['accepted'])


if __name__ == '__main__':
    unittest.main()
