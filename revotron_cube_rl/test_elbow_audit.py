"""Regression against visual-only/forged metrics and disjoint success windows."""
import copy
import unittest
from elbow_audit import elbow_metrics
from elbow_gate import combined_hold
from test_palm_handover_audit import palm_row


def outward_row():
    row=palm_row()
    row['measured_elbow']=elbow_metrics([0.,.25,.24],[.04,.43,.02],[.30,.23,.01])
    return row


class ElbowHoldTests(unittest.TestCase):
    def test_actual_outward_three_second_window(self):
        self.assertTrue(combined_hold([outward_row() for _ in range(90)],[])['accepted'])

    def test_vertical_elbow_cannot_pass_with_forged_summary(self):
        row=outward_row()
        row['measured_elbow']['elbow_root_m']=[.04,.25,.02]
        self.assertFalse(combined_hold([row]*90,[])['accepted'])

    def test_contact_and_posture_must_hold_at_the_same_time(self):
        a=outward_row();a['finger_contact_points']=[]
        b=outward_row();b['measured_elbow']['elbow_root_m']=[.04,.25,.02]
        self.assertFalse(combined_hold([a]*90+[b]*90,[])['accepted'])

    def test_one_fast_step_breaks_continuous_window(self):
        rows=[outward_row()for _ in range(150)];rows[75]['angular_speed']=.21
        self.assertFalse(combined_hold(rows,[])['accepted'])

    def test_palm_direction_is_required_with_outward_elbow(self):
        row=outward_row();row['measured_hand_frames']['left_palm_link'][1][0]=1.
        self.assertFalse(combined_hold([row]*90,[])['accepted'])

if __name__=='__main__':unittest.main()
