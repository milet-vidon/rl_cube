"""The diagnostic must change gates, never the observed contacts used for control."""
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from contact_hold_functional_worker import methods, MODES
from test_contact_hold_grace_worker import controller
from opposition_teacher import OppositionController
from isaac_compat import quaternion_to_native
from tilt_regrasp import TiltTeacher


def fingerless_pickup(phase,left,*,height=.15,on_table=False):
    c=controller(phase)
    names=[phase,'next'];c.phases=[(n,np.zeros((2,56),np.float32)) for n in names]
    c.env.contact_impulse[:]=[left,0.]
    c.env.metrics.update(height_m=height,cube_table_contact=on_table)
    return c


def wrist_case(left):
    c=controller('primary_lift'); env=c.env;q=env.targets
    env.contact_impulse=[left,0.];env.metrics=dict(height_m=.2,cube_table_contact=False)
    env.cube=NS(data=NS(root_state_w=torch.tensor([[0.,0.,1.,*quaternion_to_native([1.,0.,0.,0.])]])))
    class RawFingerFeedback:
        def __init__(self):self.seen=[]
        def update(self,actual,goal,primary,position,quat):
            self.seen.append(dict(primary));return goal
    feedback=RawFingerFeedback()
    return NS(env=env,mode='tilt',path=np.zeros((1,56),np.float32),tick=1,
              planner=NS(clear_steps=0,tilt_steps=1,tilt_degrees=0.),base=NS(finger_arrival=feedback),
              reference=np.array([1.,0.,0.,0.]), regrasp=False,lost=0,hold=0,
              failure_reason=None,finger_contacts={'left':{}},accepted=False)


class Functional(unittest.TestCase):
    def test_squeeze_modes_and_lift_readiness(self):
        for mode in MODES:
            fn,_,_=methods(mode)
            for phase in ('squeeze','primary_lift'):
                c=fingerless_pickup(phase,.01)
                for _ in range(10):fn(c)
                self.assertEqual(c.index,int(mode=='from_reset' or phase=='primary_lift'))
            for height,on_table in ((.079,False),(.15,True)):
                c=fingerless_pickup('primary_lift',.01,height=height,on_table=on_table)
                for _ in range(12):fn(c)
                self.assertEqual(c.index,0)
            c=fingerless_pickup('squeeze',0.);c.finger_contacts={'left':dict.fromkeys(('thumb','middle','index','little','ring'),.01)}
            for _ in range(10):fn(c)
            self.assertEqual(c.index,int(mode=='matched_lift'))

    def test_loss_uses_aggregate_not_fingers_and_resets(self):
        for mode in MODES:
            fn,_,_=methods(mode);c=controller('primary_lift')
            for i in range(41):
                c.env.contact_impulse=[.011 if i==16 else 1e-5,0.]
                fn(c)
                if i==16:self.assertEqual(c.lost,0)
                if i<36:self.assertIsNone(c.failure_reason)
                else:self.assertEqual(c.failure_reason,'lost_primary_support')
            c=controller('primary_lift');c.env.contact_impulse=[.01,0.]
            for _ in range(21):fn(c)
            self.assertIsNone(c.failure_reason)
            self.assertFalse(c.finger_contacts)

    def test_real_feedback_observations_and_tilt_timers(self):
        _,fn,_=methods('from_reset')
        c=wrist_case(.012);f=c.base.finger_arrival
        for i in range(90):fn(c);self.assertEqual(c.accepted,i==89)
        self.assertEqual(f.seen,[{}]*90)
        old=wrist_case(.012);TiltTeacher.action(old)
        self.assertEqual(old.lost,1);self.assertEqual(old.hold,0)
        c=wrist_case(0.)
        for i in range(20):
            fn(c);self.assertEqual(c.failure_reason,'primary_support_lost_during_tilt' if i==19 else None)
        self.assertFalse(c.accepted)
        c=wrist_case(.012);c.env.metrics['cube_table_contact']=True
        for _ in range(90):fn(c)
        self.assertEqual(c.hold,0)

    def test_originals_unmodified(self):
        orig=OppositionController.action,TiltTeacher.action
        for mode in MODES:
            _,_,r=methods(mode)
            self.assertTrue(all(d['original']!=d['diagnostic'] for d in r.values()))
        self.assertEqual(orig,(OppositionController.action,TiltTeacher.action))

if __name__=='__main__':unittest.main()
