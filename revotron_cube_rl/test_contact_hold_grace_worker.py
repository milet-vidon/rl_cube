"""CPU-only regression for the one branch that differs in the diagnostic."""
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from contact_hold_grace_worker import patched_action, ORIGINAL, TREATMENT
from opposition_teacher import OppositionController


def controller(phase):
    q=torch.zeros((1,56))
    env=NS(targets=q.clone(), scales=torch.ones_like(q), ids=list(range(56)),
           cube=None, metrics={}, cfg={'joint_names':['Joint1_L']+['finger']*55,
           'success':{'contact_impulse_min':1e-5}}, contact_impulse=[0.,0.],
           robot=NS(data=NS(joint_pos=q.clone(),joint_vel=q.clone())))
    return OppositionController(env,[(phase,np.full((120,56),.1,dtype=np.float32))])


def run(action, phase, contact):
    c=controller(phase); log=[]
    for frame,good in enumerate(contact,1):
        c.finger_contacts={'left':dict.fromkeys(('thumb','middle','ring','little'),.02) if good else {}}
        result=action(c)
        log.append((frame,c.failure_reason,c.lost,c.tick,float(result[0,0])))
        if c.failure_reason:break
    return log


class Grace(unittest.TestCase):
    def test_missing_then_recovery(self):
        original=OppositionController.action; treatment,old,new=patched_action()
        self.assertEqual(new.replace(TREATMENT, ORIGINAL),old)
        missing=[False]*30
        old_log=run(original,'primary_lift',missing);new_log=run(treatment,'primary_lift',missing)
        self.assertEqual(old_log[:9],new_log[:9])
        self.assertEqual(old_log[-1][:4],(10,'lost_primary_grasp',10,9))
        self.assertEqual(new_log[-1][:4],(20,'lost_primary_grasp',20,19))
        regains=run(treatment,'primary_lift',[False]*10+[True]+missing)
        self.assertEqual(regains[10][1:4],(None,0,11))
        self.assertEqual(regains[-1][0:3],(31,'lost_primary_grasp',20))
        self.assertIs(OppositionController.action,original)

    def test_other_phases_unchanged(self):
        treatment,_,_=patched_action();old=OppositionController.action
        for phase in ('squeeze','ready','lift','hold'):
            with self.subTest(phase=phase):
                self.assertEqual(run(treatment,phase,[False]*30),run(old,phase,[False]*30))


if __name__=='__main__':unittest.main()
