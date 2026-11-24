import unittest
from unittest.mock import patch
import numpy as np
from adaptive_wrist_teacher import AdaptiveWristPlanner
from wrist_direction import WristDirectionPlanner

class WristRecoveryTests(unittest.TestCase):
    def planner(self):
        p=AdaptiveWristPlanner.__new__(AdaptiveWristPlanner);p.wrist_goal=np.radians(12.);p.diagnostics=[]
        frames={n:np.eye(4) for n in ('Link2_L','Link4_L','Link7_L','left_palm_link')}
        frames['Link2_L'][:3,3]=[0,.25,0];frames['Link4_L'][:3,3]=[.1,.45,0];frames['Link7_L'][:3,3]=[.2,.2,.2]
        p.k=type('FK',(),{'fk':lambda _,q:frames})();return p
    def test_fallback_preserves_palm_request_and_restores_goal(self):
        p=self.planner();position=np.array([.4,.2,.1]);rotation=np.eye(3);seen=[]
        def solve(obj,q,side,pos,rot):
            self.assertIs(pos,position);self.assertIs(rot,rotation);seen.append(obj.wrist_goal)
            if len(seen)==1:raise ValueError('original pose failure')
            return {'Joint6_L':np.radians(10.)}
        with patch.object(WristDirectionPlanner,'fit_tilt',solve):result=p.fit_tilt({},'left',position,rotation)
        self.assertAlmostEqual(p.wrist_goal,np.radians(12.));self.assertAlmostEqual(result['Joint6_L'],np.radians(10.))
        self.assertEqual(len(p.diagnostics),1)
    def test_infeasible_search_raises_and_restores_goal(self):
        p=self.planner()
        with patch.object(WristDirectionPlanner,'fit_tilt',side_effect=ValueError('infeasible')),patch('adaptive_wrist_teacher.recover_bounded_pose',side_effect=ValueError('infeasible')):
            with self.assertRaisesRegex(ValueError,'infeasible'):p.fit_tilt({},'left',np.zeros(3),np.eye(3))
        self.assertAlmostEqual(p.wrist_goal,np.radians(12.));self.assertEqual(p.diagnostics,[])
    def test_original_success_unchanged(self):
        p=self.planner();answer={'original':True}
        with patch.object(WristDirectionPlanner,'fit_tilt',return_value=answer) as solve:
            self.assertIs(p.fit_tilt({},'left',np.zeros(3),np.eye(3)),answer);self.assertEqual(solve.call_count,1)

if __name__=='__main__':unittest.main()
