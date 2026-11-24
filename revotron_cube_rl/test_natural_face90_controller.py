"""CPU contract/geometry tests; none claims a physical rollout succeeded."""
import json
import unittest
from types import SimpleNamespace
from pathlib import Path
import torch
from natural_face90_controller import NaturalFace90Controller,face_point,PRESENT_FRAMES
BASE=Path(__file__).resolve().parent

class Plant:
    """Read-only sensors; no cube actuator, reset or joint target setter available."""
    def __init__(self,n):
        self.cfg=json.loads((BASE/'config.json').read_bytes());self.names=self.cfg['joint_names'];self.ids=list(range(56))
        self.home=torch.tensor([self.cfg['home'][s] for s in self.names],dtype=torch.float32)[None].repeat(n,1)
        self.pos=self.home.clone();self.robot=SimpleNamespace(get_dof_positions=lambda:self.pos)
        self.step=torch.full((1,56),.012);self.vel=30*self.step
        self.limits=torch.tensor([-3.15,3.15])[None,None,:].expand(1,56,2)
        self.core_positions=torch.tensor([.45,.15,1.32,0.,0.,0.,1.],dtype=torch.float32)[None].repeat(n,1)
    def core(self):return self.core_positions

class TestNaturalFaceController(unittest.TestCase):
    def setUp(self):
        self.plant=Plant(2); self.target=self.plant.home.clone(); self.robot_snapshot=self.plant.pos.clone();self.cube_snapshot=self.plant.core().clone()
        self.ctrl=NaturalFace90Controller(self.plant,torch.zeros((2,6)),15)
        self.active=torch.tensor([True,False]);b=torch.tensor([True,True]);z=~b
        self.obs=dict(natural_ready=z.clone(),right_contact=z.clone(),wrong_contact=z.clone(),face_deg=torch.zeros(2),structure_ok=b.clone(),support=b.clone(),contact_goal_reached=z.clone())
        self.ctrl.begin(self.active,self.target,self.obs)
    def tearDown(self):
        torch.testing.assert_close(self.robot_snapshot,self.plant.pos);torch.testing.assert_close(self.cube_snapshot,self.plant.core())
    def test_turn_sign_and_normal_offset(self):
        r=torch.tensor([.033]);radial=torch.tensor([[1.,0.]])
        a=face_point(r,radial,torch.tensor([90.]),torch.tensor([.001]))
        self.assertTrue(torch.allclose(a,torch.tensor([[0.,-.033,-.044]]),atol=1e-7))
    def test_real_robot_kinematics_and_bounds(self):
        req,withdraw,failed,_=self.ctrl.action(self.target,self.active,self.obs)
        self.assertTrue(torch.isfinite(req).all());self.assertFalse(withdraw.any() or failed.any())
        self.assertLessEqual(float((self.target-req).abs().max()),.012001)
        self.assertTrue(torch.equal(req[1],self.target[1]))
        ids=[i for i,s in enumerate(self.plant.names) if s.startswith('left_')]
        self.assertTrue(torch.equal(req[0,ids],self.target[0,ids]))
        self.assertLess(float((req[0]-self.target[0]).abs().max()),.001)  # ramp, not a goal step
        for _ in range(PRESENT_FRAMES):req,_,_,_=self.ctrl.action(self.target,self.active,self.obs)
        self.assertGreater(float((req[0]-self.target[0]).abs().max()),.001)
    def test_ready_bit_alone_cannot_skip_new_presentation(self):
        self.obs['natural_ready'][0]=True
        for _ in range(5):
            req,_,_,_=self.ctrl.action(self.target,self.active,self.obs)
        self.assertEqual(self.ctrl.phase[0],0)
        self.assertTrue(torch.equal(req[0,self.ctrl.fIDs],self.target[0,self.ctrl.fIDs]))
    def test_cannot_reinitialize_live_prefix(self):
        with self.assertRaises(ValueError):self.ctrl.begin(self.active,self.target,self.obs)
    def test_measured_contact_and_angle_gate_order(self):
        self.ctrl.phase[0]=3 # arrive at pad engagement
        for _ in range(4):self.ctrl.action(self.target,self.active,self.obs)
        self.assertEqual(self.ctrl.phase[0],3);self.assertEqual(self.ctrl.arc[0],0)
        self.obs['right_contact'][0]=True
        for _ in range(3):self.ctrl.action(self.target,self.active,self.obs)
        self.assertEqual(self.ctrl.phase[0],4);self.assertEqual(self.ctrl.arc[0],0)
        for _ in range(50):self.ctrl.action(self.target,self.active,self.obs)
        self.assertEqual(self.ctrl.phase[0],4);self.assertGreater(self.ctrl.arc[0],10)
        self.obs['face_deg'][0]=15.2; self.obs['right_contact'][0]=False
        unchanged=self.ctrl.arc[0].item()
        for _ in range(20):self.ctrl.action(self.target,self.active,self.obs)
        self.assertEqual(self.ctrl.phase[0],4);self.assertEqual(self.ctrl.arc[0],unchanged)
        self.obs['right_contact'][0]=True
        for _ in range(16):self.ctrl.action(self.target,self.active,self.obs)
        self.assertEqual(self.ctrl.phase[0],4)  # face angle alone may include pre-contact drift
        self.obs['contact_goal_reached'][0]=True
        for i in range(15):
            _,w,fail,_=self.ctrl.action(self.target,self.active,self.obs)
            self.assertFalse(w[0] or fail[0])
        self.assertEqual(self.ctrl.phase[0],5)
        self.assertTrue(self.ctrl.action(self.target,self.active,self.obs)[1][0])
    def test_wrong_piece_fails_and_freezes(self):
        self.ctrl.phase[0]=4;self.obs['wrong_contact'][0]=True
        req,withdraw,failed,why=self.ctrl.action(self.target,self.active,self.obs)
        self.assertTrue(failed[0]);self.assertFalse(withdraw[0]);self.assertEqual(why[0],'wrong_target_contact')
        self.assertTrue(torch.equal(req,self.target))
    def test_parameters_fail_closed(self):
        for par in (torch.zeros(5),torch.full((6,),1.01),torch.full((6,),float('nan'))):
            with self.assertRaises(ValueError):NaturalFace90Controller(self.plant,par,15)
    def test_r27_measured_endpoint_can_present_and_reach_selected_face(self):
        """Ideal tracking only, no physical contact/success is invented. R27 seed 20000000
        final link poses provide 14 arm DOFs, open right index, and Core XYZW. These
        small values are sufficient for FK; the ~1,900-frame replay is not a dependency.
        """
        import math
        import numpy as np
        from scipy.spatial.transform import Rotation
        from kinematics import RobotKinematics
        from joint_safety import soft_limits_from_hard,max_target_delta,project_target
        from contact_hold_cuda_teacher_math import Chain,mv,quat_matrix,log_rot
        plant=Plant(3);n=plant.names;k=RobotKinematics()
        params=torch.tensor([[-1.,-1.,0,0,0,0],[0.]*6,[1.,1.,0,0,0,0]])
        ctrl=NaturalFace90Controller(plant,params,15)
        left=[1.5473652,-.8934016,-1.5469697,-2.2120186,-1.3158931,.2106551,.7792378]
        right=[1.9078921,-1.198981,-1.0527213,-1.701763,.0536655,-.6174434,.610913]
        plant.pos[:,ctrl.lp.ix]=torch.tensor(left);plant.pos[:,ctrl.rp.ix]=torch.tensor(right)
        plant.pos[:,[i for i,s in enumerate(n) if s.startswith('right_index_')]]=0.
        plant.core_positions[:]=torch.tensor([.5420599,.2918692,1.3174783,-.7535825,-.0869958,.1451253,.6352037])
        plant.limits=soft_limits_from_hard(torch.tensor([k.limits[s] for s in n])[None],n,plant.cfg)
        arm=torch.tensor([s.startswith('Joint') for s in n])[None]
        plant.step=max_target_delta(n,torch.where(arm,.025,.060),plant.cfg)
        plant.vel=torch.where(arm,3.1416*plant.cfg['arm_velocity_limit_scale'],10.)
        target=plant.pos.clamp(plant.limits[:,:,0],plant.limits[:,:,1]).clone()
        target[:,ctrl.lp.ix]+=torch.tensor([-.003647,-.000319,-.003761,-.004405,-.000300,-.002940,-.001665])
        active=torch.ones(3,dtype=torch.bool);zero=torch.zeros_like(active);z=torch.zeros(3,dtype=torch.float64)
        obs=dict(natural_ready=zero.clone(),support=active,structure_ok=active,face_deg=z,
                 right_contact=zero,wrong_contact=zero,contact_goal_reached=zero)
        def pose(chain):
            p,R,*_=chain.fk(plant.pos[:,chain.ix].double())
            return p+ctrl.root,R
        ctrl.begin(active,target,obs);p0,L0=pose(ctrl.lp);core0=plant.core().double().clone()
        rel=mv(L0.mT,core0[:,:3]-p0);rot=L0.mT@quat_matrix(core0[:,3:])
        elbows=[Chain(k,s,n,torch.device('cpu')) for s in ('Link4_L','Link4_R','Link7_L','Link7_R')]
        history=[];count=zero.long();reached=zero.clone()
        for _ in range(270):
            p,L=pose(ctrl.lp);r,R=pose(ctrl.rp);pad,_=pose(ctrl.tip);c=p+mv(L,rel);C=L@rot
            cq=torch.from_numpy(Rotation.from_matrix(C.numpy()).as_quat())
            plant.core_positions=torch.cat((c,cq),1) # assumption: no Core/palm slip or layer rotation
            direction=torch.nn.functional.normalize(r-p);le,re,lw,rw=[pose(ch)[0] for ch in elbows]
            change=log_rot(L@L0.mT).norm(dim=1)*180/math.pi
            ok=((p0[:,1].abs()-p[:,1].abs()>=.018)&(core0[:,1].abs()-c[:,1].abs()>=.010)&
                (change>=4.)&(change<=35.)&((direction*L[:,:,0]).sum(1)>=.40)&
                (-(direction*R[:,:,0]).sum(1)>=.40)&((L[:,:,0]*R[:,:,0]).sum(1)<=.10)&
                (le[:,1]-lw[:,1]>=.020)&(rw[:,1]-re[:,1]>=.020)&(p[:,1]-r[:,1]>=.030)&(le[:,1]-re[:,1]>=.040))
            count=torch.where(ok,count+1,0);history=(history+[plant.core_positions])[-30:]
            h=torch.stack(history);qq=h[:,:,3:];qq=torch.where((qq*qq[-1:]).sum(2,keepdim=True)<0,-qq,qq)
            span=(h[:,:,:3].amax(0)-h[:,:,:3].amin(0)).norm(dim=1)
            angle=4*180/math.pi*torch.asin(((qq.amax(0)-qq.amin(0)).norm(dim=1)*.5).clamp(0,1))
            obs['natural_ready']|=(count>=30)&(span<=.008)&(angle<=3.)
            local=mv(C.mT,pad-c);dst=face_point(z+.030,ctrl.radial,z,z+.00085)
            transit=(ctrl.phase==1)|(ctrl.phase==2)
            self.assertFalse((transit&(local.abs().amax(1)<.065)).any())  # 20 mm clear of cube through hover
            reached|=(ctrl.phase==3)&((dst-local).norm(dim=1)<=.0008)&(local[:,2]>=-.045)
            if reached.all():break  # stop at geometric reach; no synthetic contact is supplied
            req,withdraw,failed,_=ctrl.action(target,active,obs)
            self.assertFalse(withdraw.any() or failed.any())
            target,clamped=project_target(target,req,plant.limits,plant.step,plant.vel,1/30)
            self.assertFalse(clamped.any());plant.pos=target.clone()
        self.assertTrue(reached.all());self.assertFalse(obs['right_contact'].any())
if __name__=='__main__':unittest.main()
