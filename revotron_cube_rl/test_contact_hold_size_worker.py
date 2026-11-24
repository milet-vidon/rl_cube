"""Offline domain isolation and vertical reference checks; no Kit or GPU."""
import copy
import math
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch

import contact_hold_size_worker as w


class WroteReset(Exception): pass
class CalculatedHeight(Exception): pass


class SizeWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.methods,_=w.method_set()

    def method(self,name,globs):
        ns={};exec(self.methods['CubePickupEnv.'+name],globs,ns);return ns[name]

    def test_reset_keeps_xy_yaw_randomization_and_clearance(self):
        reset=self.method('reset',dict(torch=torch,math=math,tensor=lambda x:x,quaternion_to_native=lambda q:q))
        seed=21611;rng=np.random.default_rng(seed)
        expected_xy=rng.uniform(-.015,.015,2);expected_angle=rng.uniform(-.15,.15)
        def check(s):
            self.assertAlmostEqual(float(s[0,2])-.9-.045,.002,places=6)
            np.testing.assert_allclose(s[0,:2],[.48+expected_xy[0],expected_xy[1]],atol=1e-7)
            np.testing.assert_allclose(s[0,3:7],[math.cos(expected_angle/2),0,0,math.sin(expected_angle/2)],atol=1e-7)
            self.assertTrue(torch.equal(s[0,7:],torch.zeros(6)));raise WroteReset
        start=torch.ones(1,13);start[0,:2]=torch.tensor([.48,0.])
        env=NS(cfg={'table_z':.9},ids=[0],home=torch.zeros(1,1),rng=np.random.default_rng(seed),
               robot=NS(reset=lambda:None,data=NS(default_joint_pos=torch.ones(1,1)),write_joint_state_to_sim=lambda *a:None,
                        set_joint_position_target=lambda q:None),
               cube=NS(reset=lambda:None,data=NS(default_root_state=start),write_root_state_to_sim=check))
        with self.assertRaises(WroteReset):reset(env)

    def test_detected_height_goes_into_reward_at_real_half_extent(self):
        action=torch.zeros(1,56);cube=torch.zeros(1,13);cube[0,:3]=torch.tensor([.48,0.,.945]);q=torch.zeros(1,56)
        def check(_,ds,h,contacts,table,*args,**kw):
            self.assertAlmostEqual(h,0.,places=6);self.assertTrue(np.all(contacts==[0.,0.]));raise CalculatedHeight
        step=self.method('step',dict(torch=torch,tensor=lambda x:x,project_target=lambda a,b,*args:(b,None),pickup_reward=check))
        env=NS(cfg={'table_z':.9,'cube_xyz':[.48,0.,.93],'joint_names':['j']*56},scales=action,max_target_delta=action,
               limits=None,command_velocity_limits=None,control_dt=1/30,targets=action,last_actions=action,
               ids=list(range(56)),robot=NS(set_joint_position_target=lambda *a,**k:None,data=NS(joint_vel=q)),
               contact_impulse=np.zeros(2),decimation=0,episode_length_buf=torch.zeros(1),_render=lambda:None,
               cube=NS(data=NS(root_state_w=cube)),palm_pos=lambda _:torch.zeros(1,3),max_episode_length=3500,hold=0,
               previous_potential=0.)
        with self.assertRaises(CalculatedHeight):step(env,action)

    def test_cube_hash_only_runtime_difference(self):
        model={'runtime_friction':[.9,.8],'runtime_core_kinematic':False};base={'source_sha256':w.BASE_ASSET,'model':model}
        new=dict(base,source_sha256='scaled');self.assertTrue(w.matches_domain('cube_asset.json',base,new,'scaled'))
        self.assertFalse(w.matches_domain('cube_asset.json',base,base,w.BASE_ASSET))
        tampered=copy.deepcopy(new);tampered['model']['runtime_friction']=[0.,0.]
        self.assertFalse(w.matches_domain('cube_asset.json',base,tampered,'scaled'))
        self.assertFalse(w.matches_domain('config.json',{'success':1},{'success':0},'scaled'))


if __name__=='__main__':unittest.main()
