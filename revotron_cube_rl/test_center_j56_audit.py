import json,unittest
from pathlib import Path
import numpy as np
from center_j56_audit import joint_coordination,next_index_offset

class JointCoordinationTests(unittest.TestCase):
    def setUp(self):
        self.names=[f'Joint{i}_L'for i in range(1,8)]
        self.initial=np.radians([89.4,-51.4,-87.5,-126.2,-77.5,11.9,46.3])
        self.current=np.radians([48,-50,-38,-124,-53,29.5,35])
        self.protocol=json.loads(Path(__file__).with_name('center_j56_protocol_v3.json').read_text())
    def check(self):return joint_coordination(self.current,self.initial,self.names,self.protocol)
    def test_outward_roll_inward_bend_and_reduced_flexion(self):self.assertTrue(self.check()['qualified'])
    def test_previous_center_style_rejected(self):
        self.current[4:]=np.radians([-89.5,8.,63.6]);self.assertFalse(self.check()['qualified'])
    def test_wrong_j5_direction_rejected(self):
        self.current[4]=self.initial[4]-.3;self.assertIn('j5_outward',self.check()['failed_conditions'])
    def test_wrong_j6_direction_rejected(self):
        self.current[5]=self.initial[5]-.1;self.assertIn('j6_inward',self.check()['failed_conditions'])
    def test_excessive_remaining_flexion_rejected(self):
        self.current[6]=np.radians(42);self.assertIn('j7_moderate_flexion',self.check()['failed_conditions'])
    def test_unchanged_flexion_rejected(self):
        self.initial[6]=np.radians(35);self.assertIn('j7_less_flexed',self.check()['failed_conditions'])
class ContactFeedbackTests(unittest.TestCase):
    def setUp(self):self.settings=json.loads(Path(__file__).with_name('center_j56_protocol_v3.json').read_text())['finger_contact_feedback']
    def test_missing_contact_closes_slowly(self):self.assertAlmostEqual(next_index_offset(0,0,False,self.settings),-.00005)
    def test_contact_in_band_holds(self):self.assertEqual(next_index_offset(-.003,.001,True,self.settings),-.003)
    def test_excessive_force_opens(self):self.assertAlmostEqual(next_index_offset(-.003,.04,True,self.settings),-.00295)
    def test_high_force_overrides_wrong_face(self):self.assertGreater(next_index_offset(-.003,.04,False,self.settings),-.003)
    def test_closure_is_bounded(self):self.assertEqual(next_index_offset(-.012,0,False,self.settings),-.012)
    def test_opening_is_bounded(self):self.assertEqual(next_index_offset(0,.04,True,self.settings),0)
if __name__=='__main__':unittest.main()
