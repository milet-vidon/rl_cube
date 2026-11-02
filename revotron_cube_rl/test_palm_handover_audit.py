import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from test_tilt_audit import held_row
from palm_handover_audit import assess_handover


def palm_row():
    row=held_row();old=Rotation.from_euler('x',90,degrees=True);new=Rotation.from_euler('x',-90,degrees=True)
    center=np.array(row['cube_position'])
    for p in row['finger_contact_points']:
        p['position_world_m']=(center+new.apply(old.inv().apply(np.array(p['position_world_m'])-center))).tolist()
    row.update(phase='tilt_hold',tilt_degrees=-90,q=[],cube_quaternion_wxyz=new.as_quat()[[3,0,1,2]].tolist())
    frames={}
    for side,sign in [('L',1),('R',-1)]:
        palm=np.eye(4);palm[:3,:3]=Rotation.from_euler('z',-sign*90,degrees=True).as_matrix();palm[1,3]=sign*.1
        elbow=np.eye(4);elbow[1,3]=sign*.3
        frames[('left' if side=='L' else 'right')+'_palm_link']=palm.tolist();frames['Link4_'+side]=elbow.tolist()
    row['measured_hand_frames']=frames
    return row


class PalmDirectionAuditTests(unittest.TestCase):
    def audit(self,row):return assess_handover([copy.deepcopy(row) for _ in range(90)],[],regrasp=False,full_episode=False)
    def test_reverse_rotation_and_palm_direction_pass(self):self.assertTrue(self.audit(palm_row())['accepted'])
    def test_old_positive_rotation_fails(self):
        row=palm_row();row['cube_quaternion_wxyz']=Rotation.from_euler('x',90,degrees=True).as_quat()[[3,0,1,2]].tolist()
        self.assertFalse(self.audit(row)['accepted'])
    def test_crossed_measured_wrists_fail_even_if_joint_targets_do_not(self):
        row=palm_row();row['measured_hand_frames']['left_palm_link'][1][3]=-.2
        self.assertIn('arms_crossed',self.audit(row)['failures'])
    def test_palm_facing_away_fails(self):
        row=palm_row();row['measured_hand_frames']['left_palm_link'][1][0]=1.
        self.assertIn('no_3s_palm_facing_uncrossed_hold',self.audit(row)['failures'])

if __name__=='__main__':unittest.main()
