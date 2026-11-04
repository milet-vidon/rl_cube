import copy
import unittest

from left_index_continuation import initial_state, SETTINGS
from left_support_envelope_v2 import validate_handover_baseline


class HandoverBindingTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {SETTINGS['joint']: .0996}
        self.state = initial_state({'finger_contact_feedback': dict(SETTINGS)},
            {'nominal_target': [.0996], 'index_contact_feedback':
             {'offset_rad': -.0004, 'next_offset_rad': -.00045}},
            [SETTINGS['joint']], [.0996])

    def test_exact_actual_baseline_accepted(self):
        validate_handover_baseline(self.state, self.baseline)

    def test_other_center_baseline_rejected(self):
        with self.assertRaises(ValueError):
            validate_handover_baseline(self.state, {SETTINGS['joint']: .0997})

    def test_changed_reference_rejected(self):
        self.state['reference_target_rad'] += .0001
        with self.assertRaises(ValueError): validate_handover_baseline(self.state, self.baseline)

    def test_nonfinite_or_discontinuous_metadata_rejected(self):
        for key in self.state:
            if not isinstance(self.state[key], float): continue
            state = copy.deepcopy(self.state); state[key] = float('nan')
            with self.assertRaises(ValueError): validate_handover_baseline(state, self.baseline)
        self.state['last_applied_offset_rad'] -= .0002
        with self.assertRaises(ValueError): validate_handover_baseline(self.state, self.baseline)


if __name__ == '__main__':
    unittest.main()
