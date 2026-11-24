import math
import unittest

from left_index_continuation import ContinuedIndexSupport, SETTINGS, initial_state, index_envelope_values


class ContinuationTests(unittest.TestCase):
    def fixture(self):
        names = [SETTINGS['joint']]
        center = {'finger_contact_feedback': dict(SETTINGS)}
        row = {'nominal_target': [.0996], 'index_contact_feedback':
               {'offset_rad': -.0004, 'next_offset_rad': -.00045}}
        return center, row, names, [.0996]

    def test_handover_preserves_original_reference(self):
        state = initial_state(*self.fixture())
        policy = ContinuedIndexSupport(state)
        self.assertAlmostEqual(state['reference_target_rad'], .1)
        self.assertAlmostEqual(policy.target, .09955)
        self.assertAlmostEqual(policy.target-state['last_nominal_target_rad'], -.00005)

    def test_high_force_precedes_missing_face(self):
        policy = ContinuedIndexSupport(initial_state(*self.fixture()))
        before = policy.offset
        policy.update(.003, False)
        self.assertAlmostEqual(policy.offset, before+.00005)

    def test_loss_and_release_remain_original_bounds(self):
        policy = ContinuedIndexSupport(initial_state(*self.fixture()))
        for _ in range(1000): policy.update(0., False)
        self.assertEqual(policy.offset, -.012)
        for _ in range(1000): policy.update(.003, True)
        self.assertEqual(policy.offset, 0.)

    def test_nonfinite_contact_rejected(self):
        policy = ContinuedIndexSupport(initial_state(*self.fixture()))
        for value in [math.nan, math.inf, -.001]:
            with self.assertRaises(ValueError): policy.update(value, False)

    def test_changed_protocol_rejected(self):
        values = self.fixture()
        values[0]['finger_contact_feedback']['lower_offset_rad'] = -.02
        with self.assertRaises(ValueError): initial_state(*values)

    def test_projected_handover_mismatch_rejected(self):
        values = self.fixture(); values[3][0] += .0001
        with self.assertRaises(ValueError): initial_state(*values)

    def test_discontinuous_next_offset_rejected(self):
        values = self.fixture(); values[1]['index_contact_feedback']['next_offset_rad'] = -.001
        with self.assertRaises(ValueError): initial_state(*values)

    def test_envelope_covers_measured_target_and_all_offsets(self):
        state = initial_state(*self.fixture())
        values = index_envelope_values(.1002, state)
        self.assertIn(.1002, values)
        self.assertAlmostEqual(min(values), .088)
        self.assertAlmostEqual(max(values), .1002)
        self.assertLessEqual(max(b-a for a,b in zip(values, values[1:])), .001+1e-12)
        self.assertIn(state['reference_target_rad']+state['next_offset_rad'], values)


if __name__ == '__main__':
    unittest.main()
