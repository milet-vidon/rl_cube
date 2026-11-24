import unittest
import numpy as np
from opposed_path_validation import samples,assemble,require_valid,PHASES

class ValidationContractTests(unittest.TestCase):
    def test_exact_interpolation_and_boundary(self):
        names=[str(i) for i in range(56)];initial={n:0. for n in names};point=np.zeros(56);point[0]=.025
        phases=[(phase,[point.copy()]) for phase in PHASES]
        jobs,turn_end=samples(initial,names,phases)
        self.assertEqual(len(jobs),9);self.assertTrue(np.array_equal(jobs[0][3],point/3));self.assertTrue(np.array_equal(jobs[2][3],point));self.assertEqual(turn_end,[7])
    def test_failure_sort_and_original_message(self):
        rows=[{'index':i,'phase':'right_close','step':i,'theta_deg':0.,'valid':i not in (1,3),'failure_pair':['finger','cube',-.003] if i in (1,3) else None} for i in range(4)]
        r=assemble(rows[::-1],[],0.,2);self.assertEqual(r['checks'],2);self.assertEqual(r['executed_checks'],4)
        self.assertEqual(r['failure'],"right_close step 1: ('finger', 'cube', -0.003)")
        with self.assertRaisesRegex(ValueError,'right_close step 1'):require_valid(r)
    def test_angle_endpoints_preserved(self):
        rows=[{'index':i,'phase':'opposed_face_turn','step':i,'theta_deg':i*.25,'valid':True,'failure_pair':None} for i in range(5)]
        r=assemble(rows,[1,4],0.,1);self.assertEqual(require_valid(r),(5,[.25,1.]))
    def test_missing_results_rejected(self):
        with self.assertRaises(ValueError):assemble([{'index':1}],[],0.,1)
    def test_nonfinite_path_rejected(self):
        names=[str(i) for i in range(56)];initial={n:0. for n in names};phases=[(p,[[float('nan')]*56]) for p in PHASES]
        with self.assertRaises(ValueError):samples(initial,names,phases)

if __name__=='__main__':unittest.main()
