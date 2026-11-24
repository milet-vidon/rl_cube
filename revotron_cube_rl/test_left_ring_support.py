import math
import unittest
from left_ring_support import RingSupport, envelope_values, parameters, UPPER, STEP

class RingSupportTests(unittest.TestCase):
    def test_lost_contact_reaches_bound_without_exceeding(self):
        c=RingSupport()
        for _ in range(1000):
            previous=c.offset;c.update(0.)
            self.assertGreaterEqual(c.offset,0.)
            self.assertLessEqual(c.offset,UPPER)
            self.assertLessEqual(c.offset-previous,STEP+1e-12)
        self.assertEqual(c.offset,UPPER)

    def test_supported_contact_holds_and_overload_releases(self):
        c=RingSupport();c.update(0.);self.assertEqual(c.update(.001),STEP)
        self.assertEqual(c.update(.003),0.)
        self.assertEqual(c.update(.003),0.)

    def test_invalid_contact_cannot_move_controller(self):
        for value in (math.nan,math.inf,-.001):
            c=RingSupport()
            with self.assertRaises(ValueError):c.update(value)
            self.assertEqual(c.offset,0.)

    def test_envelope_includes_actual_target_and_all_feedback(self):
        for actual,target in ((.57,.573),(.573,.57)):
            values=envelope_values(actual,target)
            for value in (actual,target,target+UPPER):self.assertIn(value,values)
            self.assertLessEqual(max(b-a for a,b in zip(values,values[1:])),.001+1e-12)
            self.assertEqual(min(values),min(actual,target))
            self.assertEqual(max(values),max(actual,target+UPPER))

    def test_nonfinite_baseline_is_rejected(self):
        for values in ((math.nan,0),(0,math.inf)):
            with self.assertRaises(ValueError):envelope_values(*values)

if __name__=='__main__':unittest.main()
