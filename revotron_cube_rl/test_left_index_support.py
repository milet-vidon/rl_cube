import math
import unittest
from left_index_support import IndexSupport, envelope_values, parameters, index_impulse, LOWER, STEP

class IndexSupportTests(unittest.TestCase):
    def test_unloaded_contact_moves_negative_and_stops(self):
        c=IndexSupport()
        for _ in range(1000):
            previous=c.offset;c.update(0.)
            self.assertGreaterEqual(c.offset,LOWER)
            self.assertLessEqual(c.offset,0.)
            self.assertLessEqual(previous-c.offset,STEP+1e-12)
        self.assertEqual(c.offset,LOWER)

    def test_supported_contact_holds_overload_returns_to_baseline(self):
        c=IndexSupport();c.update(0.);self.assertEqual(c.update(.001),-STEP)
        self.assertEqual(c.update(.003),0.)
        self.assertEqual(c.update(.003),0.)

    def test_invalid_input_keeps_controller_state(self):
        for value in (math.nan,math.inf,-.001):
            c=IndexSupport()
            with self.assertRaises(ValueError):c.update(value)
            self.assertEqual(c.offset,0.)

    def test_actual_and_applied_target_discrepancy_is_covered(self):
        for actual,target in ((.0065,.0062),(.005,.0062)):
            values=envelope_values(actual,target)
            for value in (actual,target,target+LOWER):self.assertIn(value,values)
            self.assertLessEqual(max(b-a for a,b in zip(values,values[1:])),.001+1e-12)
            self.assertEqual(min(values),min(actual,target+LOWER))
            self.assertEqual(max(values),max(actual,target))

    def test_nonfinite_baseline_is_rejected(self):
        for values in ((math.nan,0),(0,math.inf)):
            with self.assertRaises(ValueError):envelope_values(*values)

    def test_only_original_topology_face_and_index_count(self):
        row={'cube_position':[0.,0.,0.],'cube_quaternion_wxyz':[1.,0.,0.,0.],
             'finger_contact_points':[
                 dict(side='left',finger='index',position_world_m=[.028,-.028,.027],impulse_Ns=.001),
                 dict(side='left',finger='index',position_world_m=[0.,-.028,.027],impulse_Ns=.01),
                 dict(side='right',finger='index',position_world_m=[.028,-.028,.027],impulse_Ns=.02),
                 dict(side='left',finger='ring',position_world_m=[.028,-.028,.027],impulse_Ns=.03)]}
        self.assertAlmostEqual(index_impulse(row),.001)

if __name__=='__main__':unittest.main()
