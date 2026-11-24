import unittest
from types import SimpleNamespace
from unittest.mock import patch
from settled_wrist_teacher import SettledWristTeacher
from adaptive_wrist_teacher import AdaptiveWristTeacher
from tilt_regrasp import TiltTeacher

class SettledHoldTests(unittest.TestCase):
    def teacher(self,angle=2.):
        t=SettledWristTeacher.__new__(SettledWristTeacher);t.mode='tilt';t.tick=20;t.path=[0]*10
        t._hold_error_deg=lambda:angle;t.planner=SimpleNamespace(diagnostics=[]);t.hold=90;t.accepted=True
        t.hold_gains_set=True;t.env=SimpleNamespace(metrics={'linear_speed':0.,'angular_speed':0.})
        return t
    def test_deadband_keeps_base_contact_control(self):
        t=self.teacher()
        with patch.object(TiltTeacher,'action',return_value='base') as base,patch.object(AdaptiveWristTeacher,'action',return_value='correct') as original:
            self.assertEqual(t.action(),'base');base.assert_called_once();original.assert_not_called()
        self.assertEqual(t.planner.diagnostics[0]['deadband_deg'],2.5)
    def test_outside_deadband_keeps_bounded_original_correction(self):
        t=self.teacher(2.501)
        with patch.object(AdaptiveWristTeacher,'action',return_value='original') as original:
            self.assertEqual(t.action(),'original');original.assert_called_once()
        self.assertEqual(t.planner.diagnostics,[])
    def test_measured_motion_still_rejects_hold(self):
        for metric,value in (('linear_speed',.03),('angular_speed',.20)):
            t=self.teacher();t.env.metrics[metric]=value
            with patch.object(TiltTeacher,'action',return_value='base'):t.action()
            self.assertEqual(t.hold,0);self.assertFalse(t.accepted)
    def test_prehold_path_unchanged(self):
        t=self.teacher();t.tick=5
        with patch.object(AdaptiveWristTeacher,'action',return_value='path') as original:
            self.assertEqual(t.action(),'path');original.assert_called_once()
if __name__=='__main__':unittest.main()
