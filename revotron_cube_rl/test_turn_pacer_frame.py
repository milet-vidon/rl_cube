"""CPU tests for coordinate-only integration into the original pacer gates."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from turn_pacer import TurnPacer
from turn_pacer_frame import PlanningFrameFeedback


def quaternion_z(degrees):
    return Rotation.from_euler('z', degrees, degrees=True).as_quat()[[3, 0, 1, 2]].tolist()


def row(step, core, face, contacts=True):
    return {'step': step, 'cube_quaternion_wxyz': core,
            'layer_measurement': {'face_angle_deg': face},
            'opposition_audit': {'right_contacts': {'all_five': contacts}}}


class PlanningFramePacerTests(unittest.TestCase):
    def test_zero_Core_matches_every_original_decision_and_counter(self):
        feedback = PlanningFrameFeedback({'cube_quaternion': quaternion_z(0)})
        original, mapped = TurnPacer([0, .2, .4, .8, 1.]), TurnPacer([0, .2, .4, .8, 1.])
        for step, (angle, contacts) in enumerate(((0, True), (.1, True), (.1, False),
                                                 (.1, True), (.1, True), (.3, True), (.3, True))):
            expected = original.next_index(angle, contacts)
            actual = feedback.next_index(mapped, row(step, quaternion_z(0), angle, contacts))[:2]
            self.assertEqual(actual, expected)
            self.assertEqual(mapped.report(), original.report())

    def test_co_rotation_uses_planning_frame_for_lead(self):
        feedback = PlanningFrameFeedback({'cube_quaternion': quaternion_z(0)})
        mapped, old = TurnPacer([0., .8]), TurnPacer([0., .8])
        mapped.next_index(0, True)
        old.next_index(0, True)
        # A 0.6 degree world layer turn from Core co-rotation is real progress
        # in the planning frame although current-Core relative layer angle is 0.
        actual = feedback.next_index(mapped, row(1, quaternion_z(-.6), 0.))
        self.assertEqual(actual[:2], (1, None))
        self.assertEqual(old.next_index(0., True), (0, None))
        self.assertAlmostEqual(actual[2]['planning_frame_layer_angle_deg'], .6)

    def test_original_reference_and_initial_layer_bias_are_retained(self):
        request = {'cube_quaternion': quaternion_z(11.)}
        feedback = PlanningFrameFeedback(request)
        request['cube_quaternion'][:] = quaternion_z(39.)
        pacer = TurnPacer([0.])
        result = feedback.next_index(pacer, row(7, quaternion_z(10.6), .2))
        self.assertAlmostEqual(result[2]['planning_frame_layer_angle_deg'], .6)
        self.assertEqual(feedback.report()['initial_Core_source'], 'opposition_request.cube_quaternion')
        self.assertFalse(feedback.report()['initial_layer_offset_subtracted'])

    def test_contact_and_stall_gates_unchanged_despite_large_progress(self):
        feedback = PlanningFrameFeedback({'cube_quaternion': quaternion_z(0)})
        pacer = TurnPacer([0., 1.], stall_ticks=2)
        self.assertEqual(feedback.next_index(pacer, row(0, quaternion_z(-10), 0., False))[:2], (0, None))
        self.assertEqual(feedback.next_index(pacer, row(1, quaternion_z(-10), 0., False))[:2], (0, None))
        self.assertEqual(feedback.next_index(pacer, row(2, quaternion_z(-10), 0., False))[:2],
                         (None, 'turn_contact_or_progress_stalled'))
        self.assertEqual(pacer.cursor, 0)

    def test_nonfinite_feedback_does_not_advance(self):
        feedback = PlanningFrameFeedback({'cube_quaternion': quaternion_z(0)})
        for bad in (row(0, quaternion_z(0), float('nan')), row(0, [1, 0, np.inf, 0], 0.)):
            pacer = TurnPacer([0., .5])
            with self.assertRaises(ValueError):
                feedback.next_index(pacer, bad)
            self.assertEqual(pacer.cursor, 0)
            self.assertEqual(pacer.ticks, 0)

    def test_global_frame_change_preserves_adapter_decision(self):
        initial = Rotation.from_euler('xyz', [10, 30, -20], degrees=True)
        current = initial * Rotation.from_euler('z', -1., degrees=True)
        global_rotation = Rotation.from_euler('xyz', [40, -11, 60], degrees=True)
        def wxyz(r):
            return r.as_quat()[[3, 0, 1, 2]].tolist()
        A = PlanningFrameFeedback({'cube_quaternion': wxyz(initial)})
        B = PlanningFrameFeedback({'cube_quaternion': wxyz(global_rotation * initial)})
        a = A.next_index(TurnPacer([.9]), row(0, wxyz(current), .1))
        b = B.next_index(TurnPacer([.9]), row(0, wxyz(global_rotation * current), .1))
        self.assertEqual(a[:2], b[:2])
        self.assertAlmostEqual(a[2]['planning_frame_layer_angle_deg'], b[2]['planning_frame_layer_angle_deg'])


if __name__ == '__main__':
    unittest.main()
