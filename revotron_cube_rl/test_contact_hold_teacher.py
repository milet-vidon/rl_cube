import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from contact_hold_teacher import ContactHoldTeacher
from settled_wrist_teacher import SettledWristTeacher


class ContactHoldTests(unittest.TestCase):
    def teacher(self, tick):
        obj = ContactHoldTeacher.__new__(ContactHoldTeacher)
        obj.mode, obj.tick, obj.path = 'tilt', tick, [0] * 10
        names = [f'Joint{i}_L' for i in range(7)] + [f'left_finger_{i}' for i in range(21)]
        names += [f'right_finger_{i}' for i in range(28)]
        obj.env = SimpleNamespace(ids=list(reversed(range(56))), cfg={'joint_names': names}, robot=Mock())
        obj.planner = SimpleNamespace(diagnostics=[])
        return obj

    def test_no_gain_change_before_hold(self):
        obj = self.teacher(9)
        with patch.object(SettledWristTeacher, 'action', return_value='original'):
            self.assertEqual(obj.action(), 'original')
        self.assertEqual(obj.env.robot.mock_calls, [])

    def test_gain_event_once_only_left_fingers_and_parent_runs(self):
        obj = self.teacher(10)
        with patch.object(SettledWristTeacher, 'action', return_value='original') as parent:
            obj.action()
            obj.tick += 1
            obj.action()
        self.assertEqual(parent.call_count, 2)
        method = obj.env.robot.write_joint_stiffness_to_sim_index
        method.assert_called_once()
        self.assertEqual(method.call_args.kwargs['joint_ids'], obj.env.ids[7:28])
        self.assertTrue((method.call_args.kwargs['stiffness'] == 30.).all())
        obj.env.robot.write_joint_damping_to_sim_index.assert_called_once()
        self.assertEqual(len(obj.planner.diagnostics), 1)

    def test_invalid_joint_layout_fails_before_writes(self):
        obj = self.teacher(10)
        obj.env.cfg['joint_names'][7] = 'missing'
        with self.assertRaisesRegex(ValueError, '21 left'):
            obj.action()
        self.assertEqual(obj.env.robot.mock_calls, [])

    def test_selective_ramp_excludes_index_and_stops_at_requested_gain(self):
        obj = self.teacher(9)
        names = obj.env.cfg['joint_names']
        names[7:28] = [f'left_{f}_{j}' for f in ('thumb', 'index', 'middle', 'ring', 'little')
                       for j in range(5 if f == 'thumb' else 4)]
        expected = [obj.env.ids[i] for i, n in enumerate(names) if n.startswith(('left_thumb_', 'left_little_'))]
        obj.opposing_pair_stiffness = 15.
        with patch.object(SettledWristTeacher, 'action', return_value='original') as base:
            for tick in range(9, 43):
                obj.tick = tick
                obj.action()
            self.assertEqual(base.call_count, 34)
        calls = obj.env.robot.write_joint_stiffness_to_sim_index.call_args_list
        self.assertEqual(len(calls), 31)
        for offset, call in enumerate(calls[1:], 1):
            self.assertEqual(call.kwargs['joint_ids'], expected)
            self.assertTrue((call.kwargs['stiffness'] == 30. - .5 * offset).all())


if __name__ == '__main__':
    unittest.main()
