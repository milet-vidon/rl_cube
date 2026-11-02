import unittest
from types import SimpleNamespace
import numpy as np
import torch
from side_control import SegmentController

def environment():
    return SimpleNamespace(targets=torch.zeros(1,56),scales=torch.ones(1,56)*.025,contact_impulse=[0.,0.],metrics={},ids=list(range(56)),cfg={'joint_names':['Joint1_L']+['finger']*55,'success':{'contact_impulse_min':.00001}},robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1,56),joint_vel=torch.zeros(1,56))))

class PhaseSafetyTests(unittest.TestCase):
    def test_lift_requires_ten_bilateral_samples(self):
        e=environment();z=np.zeros(56,dtype=np.float32)
        c=SegmentController(e,[('contact',z,z,1),('lift',z,z+.1,10)])
        for _ in range(15):c.action()
        self.assertEqual(c.index,0)
        e.contact_impulse=[.01,.01]
        for _ in range(9):c.action()
        self.assertEqual(c.index,0)
        c.action();self.assertEqual(c.index,1)

    def test_contact_does_not_override_unsettled_arm(self):
        e=environment();e.contact_impulse=[.01,.01];e.robot.data.joint_vel[0,0]=.5
        z=np.zeros(56,dtype=np.float32);c=SegmentController(e,[('contact',z,z,1),('lift',z,z+.1,10)])
        for _ in range(20):c.action()
        self.assertEqual(c.index,0)

    def test_lost_grip_stops_further_commands(self):
        e=environment();z=np.zeros(56,dtype=np.float32);c=SegmentController(e,[('lift',z,z+.1,100)])
        for _ in range(10):action=c.action()
        self.assertEqual(c.failure_reason,'lift_contact_lost')
        self.assertTrue(torch.equal(action,torch.zeros_like(action)))
        self.assertTrue(torch.equal(c.action(),torch.zeros_like(action)))

if __name__=='__main__':unittest.main()
