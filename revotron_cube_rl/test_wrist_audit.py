import copy,unittest
import numpy as np
from elbow_audit import elbow_metrics
from test_palm_handover_audit import palm_row
from wrist_audit import wrist_geometry,assess_wrist


def row(side=-1):
    r=palm_row();r['q']=[float(np.radians(12))]
    r['measured_elbow']=elbow_metrics([0,.25,.24],[.2,.43,.02],[.2+side*.1,.30,.20])
    return r


class WristAuditTests(unittest.TestCase):
    def test_reversed_side_and_all_other_hold_conditions(self):
        result=assess_wrist([row()for _ in range(90)],['Joint6_L'])
        self.assertTrue(result['accepted']);self.assertLess(result['final']['side_bend_deg'],0)

    def test_joint_sign_alone_cannot_prove_reversed_bend(self):
        r=row(1);r['measured_wrist']={'side_bend_deg':-90}
        self.assertFalse(assess_wrist([r]*90,['Joint6_L'])['accepted'])

    def test_wrist_outside_elbow_fails(self):
        r=row();r['measured_elbow']['wrist_root_m'][1]=.46
        self.assertFalse(assess_wrist([r]*90,['Joint6_L'])['accepted'])

    def test_contact_and_wrist_need_the_same_hold_window(self):
        a=row();a['finger_contact_points']=[];b=row(1)
        self.assertFalse(assess_wrist([a]*90+[b]*90,['Joint6_L'])['accepted'])

    def test_flipped_palm_is_not_a_wrist_correction(self):
        r=row();r['measured_hand_frames']['left_palm_link'][1][0]=1.
        self.assertFalse(assess_wrist([r]*90,['Joint6_L'])['accepted'])

if __name__=='__main__':unittest.main()
