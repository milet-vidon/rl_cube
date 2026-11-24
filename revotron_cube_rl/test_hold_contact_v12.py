"""Regression checks for the gain discontinuity at the hold boundary."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hold_contact_teacher_v12 import ContactHoldTeacher
from settled_wrist_teacher import SettledWristTeacher


class GainContinuityTests(unittest.TestCase):
    def teacher(self, root, tick=10, damping=1.):
        t = ContactHoldTeacher.__new__(ContactHoldTeacher)
        t.mode = 'tilt'; t.tick = tick; t.path = [0] * 10
        t.hold_damping = damping
        t.env = SimpleNamespace(run_dir=root, ids=[2, 3, 4],
                                cfg={'joint_names': ['Joint1_L', 'left_index_MCP_joint', 'right_index_MCP_joint']},
                                episode_length_buf=[1180], robot=Mock())
        return t

    def test_prehold_does_not_change_drives(self):
        with TemporaryDirectory() as root:
            t = self.teacher(root, tick=9)
            with patch.object(SettledWristTeacher, 'action', return_value='native') as parent:
                self.assertEqual(t.action(), 'native')
                parent.assert_called_once()
            t.env.robot.write_joint_stiffness_to_sim_index.assert_not_called()

    def test_hold_keeps_30_and_leaves_arm_and_right_hand_alone(self):
        with TemporaryDirectory() as root:
            t = self.teacher(root)
            with patch.object(SettledWristTeacher, 'action', return_value='native'):
                self.assertEqual(t.action(), 'native')
            call = t.env.robot.write_joint_stiffness_to_sim_index.call_args.kwargs
            self.assertEqual(call['joint_ids'], [3])
            self.assertEqual(call['stiffness'].tolist(), [[30.]])
            self.assertTrue(t.hold_gains_set)
            events = [json.loads(s) for s in (Path(root)/'hold_gain_events_v12.jsonl').read_text().splitlines()]
            self.assertEqual(events[0]['damping'], 1.)

    def test_damping_changes_gradually_and_stops(self):
        with TemporaryDirectory() as root:
            t = self.teacher(root, damping=2.)
            values = []
            for offset in (0, 15, 30, 31):
                t.tick = 10 + offset
                with patch.object(SettledWristTeacher, 'action', return_value=None):
                    t.action()
                values.append(t.env.robot.write_joint_damping_to_sim_index.call_args.kwargs['damping'].item())
            self.assertEqual(values, [1., 1.5, 2., 2.])
            self.assertEqual(t.env.robot.write_joint_damping_to_sim_index.call_count, 3)


if __name__ == '__main__':
    unittest.main()
