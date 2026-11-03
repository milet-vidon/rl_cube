import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from opposed_grasp_audit import right_contacts

def points(angle=0):
 out=[]
 for finger,x in zip(('thumb','index','middle','ring','little'),(.013,.024,.008,-.008,-.024)):
  pos=R.from_euler('z',-angle,degrees=True).apply([x,-.027 if finger=='thumb'else .027,-.020]);out.append({'robot_body':'/World/Robot/right_'+finger+'_DIP_link','cubelet':'nX_nZ_pY','position_world_m':pos.tolist(),'impulse_Ns':.001})
 return out
class OppositionContactTests(unittest.TestCase):
 def test_all_five_target_layer(self):self.assertTrue(right_contacts(points(),[0,0,0],[1,0,0,0])['all_five'])
 def test_index_only_not_grasp(self):self.assertFalse(right_contacts(points()[1:2],[0,0,0],[1,0,0,0])['all_five'])
 def test_little_on_other_layer_fails(self):
  p=points();p[-1]['cubelet']='nX_pY';a=right_contacts(p,[0,0,0],[1,0,0,0]);self.assertFalse(a['all_five']);self.assertEqual(len(a['unexpected_contacts']),1)
 def test_thumb_same_side_fails(self):
  p=points();p[0]['position_world_m'][1]=.027;self.assertFalse(right_contacts(p,[0,0,0],[1,0,0,0])['all_five'])
 def test_proximal_contact_not_distal_grasp(self):
  p=points();p[0]['robot_body']='/World/Robot/right_thumb_MCP_link';self.assertFalse(right_contacts(p,[0,0,0],[1,0,0,0])['all_five'])
 def test_wrist_contact_is_unexpected(self):
  p=points();p.append(p[0]|{'robot_body':'/World/Robot/Link7_R'});self.assertEqual(len(right_contacts(p,[0,0,0],[1,0,0,0])['unexpected_contacts']),1)
 def test_rotated_layer_contact_coordinates(self):self.assertTrue(right_contacts(points(90),[0,0,0],[1,0,0,0],90)['all_five'])
if __name__=='__main__':unittest.main()
