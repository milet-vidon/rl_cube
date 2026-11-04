"""Pure host tests: do not import or start Isaac Sim."""
import copy
import unittest

import numpy as np

from evaluate_reference_replay import replay_schedule, replay_target, target_error


class ReplayTests(unittest.TestCase):
    def fixture(self):
        return [
            {'step': 0, 'phase': 'ready', 'applied_target': [0.] * 56,
             'cube_quaternion_wxyz': [1., 0., 0., 0.], 'tilt_reference_wxyz': None},
            {'step': 1, 'phase': 'tilt_hold', 'applied_target': [.01] * 56,
             'cube_quaternion_wxyz': [.99, 0., .1, 0.],
             'tilt_reference_wxyz': [1., 0., 0., 0.]},
        ]

    def test_exact_applied_target_not_nominal(self):
        row = self.fixture()[1]
        row['nominal_target'] = [2.] * 56
        actual = replay_target(row)
        np.testing.assert_array_equal(actual, np.full(56, .01, dtype=np.float32))
        actual[0] = 4
        self.assertEqual(row['applied_target'][0], .01)

    def test_schedule_is_previous_measured_state(self):
        rows = self.fixture()
        self.assertEqual(replay_schedule(rows), (1, 1))
        rows[1]['tilt_reference_wxyz'] = rows[1]['cube_quaternion_wxyz']
        with self.assertRaises(ValueError):
            replay_schedule(rows)

    def test_refuses_shifted_or_gapped_control_steps(self):
        rows = self.fixture()
        for index, value in ((0, 1), (1, 2)):
            altered = copy.deepcopy(rows)
            altered[index]['step'] = value
            with self.assertRaises(ValueError):
                replay_schedule(altered)

    def test_bad_target_rejected(self):
        for value in ([0.] * 55, [float('nan')] * 56, [float('inf')] * 56):
            with self.assertRaises(ValueError):
                replay_target({'applied_target': value})

    def test_emitted_error_checks_both_hands(self):
        reference = [0.] * 56
        actual = list(reference)
        actual[54] = 2e-6
        error = target_error(actual, reference)
        self.assertEqual(error['max_abs_rad'], 2e-6)
        self.assertEqual(error['worst_joint_index'], 54)
        self.assertEqual(target_error(reference, reference)['max_abs_rad'], 0.)

    def test_nonfinite_emission_rejected(self):
        with self.assertRaises(ValueError):
            target_error([0.] * 55 + [float('nan')], [0.] * 56)


if __name__ == '__main__':
    unittest.main()
