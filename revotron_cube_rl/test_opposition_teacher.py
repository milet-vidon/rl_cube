import unittest
import numpy as np
import torch
from types import SimpleNamespace
from opposition_acceptance import assess_opposition,FINGERS
from opposition_teacher import timed_path,OppositionController

def row():
    contacts=[{'side':'left','finger':f,'impulse_Ns':.001,'position_world_m':[-.0285,-.02,.02] if f=='thumb' else [.0285,0.,0.]} for f in FINGERS]
    return {'cube_position':[0.,0.,0.],'cube_quaternion_wxyz':[1.,0.,0.,0.],
            'finger_contact_impulses':{'left':{f:.001 for f in FINGERS}},'finger_contact_points':contacts}

class OppositionChecks(unittest.TestCase):
    def test_contact_on_same_face_does_not_count_as_opposition(self):
        r=row();r['finger_contact_points'][0]['position_world_m']=[.0285,-.02,.02]
        self.assertFalse(assess_opposition([r]*90)['passed'])
    def test_missing_fourth_finger_rejected(self):
        r=row();r['finger_contact_impulses']['left']['little']=0.
        self.assertFalse(assess_opposition([r]*90)['passed'])
    def test_thumb_corner_and_all_opposing_fingers_pass(self):
        self.assertTrue(assess_opposition([row()]*90)['passed'])
    def test_retiming_respects_joint_velocity_and_acceleration(self):
        q=timed_path(np.array([[0.,0.],[1.,-.7],[1.5,-1.]]),.1,30,
                     np.array([.6,.4]),np.array([1.2,.8]),np.array([[-2,2],[-2,2]]))
        self.assertTrue(np.all(np.max(np.abs(np.diff(q,axis=0)),axis=0)*30<=[.6001,.4001]))
        self.assertTrue(np.all(np.max(np.abs(np.diff(q,n=2,axis=0)),axis=0)*900<=[1.201,.801]))
    def test_support_waits_for_every_primary_finger(self):
        zero=torch.zeros((1,56));names=[f'Joint{i}_L' for i in range(1,8)]+[f'finger_{i}' for i in range(49)]
        env=SimpleNamespace(targets=zero.clone(),scales=torch.ones_like(zero),ids=list(range(56)),contact_impulse=[.01,.01],
             cfg={'joint_names':names,'control_hz':30,'success':{'contact_impulse_min':1e-5}},
             robot=SimpleNamespace(data=SimpleNamespace(joint_pos=zero,joint_vel=zero)))
        c=OppositionController(env,[('squeeze',np.zeros((1,56),dtype=np.float32)),('support',np.zeros((100,56),dtype=np.float32))])
        c.finger_contacts={'left':{f:.01 for f in FINGERS}};c.finger_contacts['left']['index']=0.
        for _ in range(15):c.action()
        self.assertEqual(c.index,0)
        c.finger_contacts['left']['index']=.01
        for _ in range(10):c.action()
        self.assertEqual(c.index,1)
    def test_interpolation_cannot_overshoot_safe_joint_knots(self):
        q=timed_path(np.array([[0.],[.99],[1.],[.99]]),2.,30,
                     np.array([.6]),np.array([1.2]),np.array([[0.,1.]]))
        self.assertGreaterEqual(float(q.min()),0.)
        self.assertLessEqual(float(q.max()),1.)

if __name__=='__main__':unittest.main()
