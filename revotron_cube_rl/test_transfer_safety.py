import unittest
import torch
from joint_safety import project_target
from acceptance import assess_pickup

class SafetyTests(unittest.TestCase):
    def test_arbitrary_targets_obey_position_step_and_velocity(self):
        torch.manual_seed(12)
        limits=torch.tensor([[[-1.,1.]]*56]);previous=torch.zeros(1,56)
        for _ in range(100):
            request=torch.randn(1,56)*50
            applied,hits=project_target(previous,request,limits,torch.full((1,56),.1),torch.full((1,56),.3),.1)
            self.assertTrue((applied.abs()<=1.000001).all())
            self.assertTrue(((applied-previous).abs()<=.030001).all())
            self.assertTrue(torch.equal(hits,request.abs()>1))
            previous=applied
        with self.assertRaises(ValueError):
            project_target(previous,torch.full((1,56),float('nan')),limits,torch.ones(1,56),torch.ones(1,56),.1)

    def test_lift_cannot_hide_collision_or_limit_failure(self):
        row=dict(success=True,robot_table_impulse=0.,arm_soft_limit_hit=False,actual_arm_soft_limit_violation=False,
                 hard_limit_hit=[False]*56,drop=False,bilateral_contact=True,cube_table_contact=False,height_m=.1,
                 joint_velocity=[0.]*56,joint_position_error=[0.]*56,hold_steps=30)
        cfg={'control_hz':30,'success':{'hold_seconds':1.,'lift_m':.08}}
        rows=[dict(row) for _ in range(30)]
        self.assertTrue(assess_pickup(rows,cfg)['accepted'])
        for key,value in [('robot_table_impulse',.01),('arm_soft_limit_hit',True),('actual_arm_soft_limit_violation',True),('hard_limit_hit',[True]*56),('drop',True),('cube_table_contact',True),('bilateral_contact',False)]:
            bad=[dict(r) for r in rows];bad[0][key]=value
            self.assertFalse(assess_pickup(bad,cfg)['accepted'],key)

if __name__=='__main__': unittest.main()
