"""Physical readback regressions for the final quarter-turn gate."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from natural_face90_contact_point import verify_quarter_turn


class QuarterTurnReadback(unittest.TestCase):
    def check_trace(self, samples, peak, success):
        with TemporaryDirectory() as directory:
            trace = Path(directory) / 'trace.jsonl'
            with trace.open('w') as stream:
                for step, (angle, contact, drive) in enumerate(samples):
                    row = dict(seed=1, lane=0, step=step, safe=True, face_deg=angle,
                               qualified_index_frame=contact, index_distal_impulse_Ns=.1 if contact else 0.,
                               other_digit_target_impulse_Ns=0., drive_command=drive)
                    stream.write(json.dumps(row) + '\n')
            verify_quarter_turn(trace, [dict(seed=1, success=success, peak_isolated_drive_progress_deg=peak)])

    def test_reseated_finger_can_continue_a_measured_quarter_turn(self):
        self.check_trace([(0, True, False), (40, True, True), (40, False, False),
                          (40, True, False), (90, True, True)], 90, True)

    def test_detached_motion_does_not_earn_drive_credit(self):
        samples = [(0, True, False), (30, True, True), (70, False, False),
                   (70, True, False), (90, True, True)]
        self.check_trace(samples, 50, False)
        with self.assertRaises(ValueError):
            self.check_trace(samples, 90, True)

    def test_slip_and_repeated_push_cannot_double_count_the_same_angle(self):
        samples = [(0, True, False), (50, True, True), (0, False, False),
                   (0, True, False), (50, True, True)]
        self.check_trace(samples, 50, False)
        with self.assertRaises(ValueError):
            self.check_trace(samples, 50, True)

    def test_late_forward_drift_cannot_cancel_an_earlier_detached_slip(self):
        samples = [(0, True, False), (30, True, True), (0, False, False),
                   (0, True, False), (30, True, True), (60, False, False),
                   (60, True, False), (90, True, True)]
        self.check_trace(samples, 60, False)
        with self.assertRaises(ValueError):
            self.check_trace(samples, 90, True)
