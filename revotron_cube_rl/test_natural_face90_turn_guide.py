"""CPU geometry and controller tests. No simulation scene or trainer is started."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from kinematics import RobotKinematics
from natural_face90_v5_controller import NaturalFace90Controller
from natural_face90_turn_guide import TurnGuide, pad_near_marker


class ContactTurn(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parent
        cfg = json.loads((root/'config.json').read_text())
        names = cfg['joint_names']
        endpoint = json.loads((root/'natural_face90_v5_regression.json').read_text())
        self.q = torch.tensor(endpoint['r27_endpoint']['q'],dtype=torch.double)[None].repeat(2,1)
        labs = ['Link7_R']+[f'right_{f}_{j}_link' for f in ('thumb','index','middle','ring','little')
                            for j in ('MCP','PIP','DIP')]
        step = torch.full((1,56),.012)
        robot = SimpleNamespace(shared_metatype=SimpleNamespace(link_names=labs))
        plant = SimpleNamespace(cfg=cfg,names=names,home=self.q,step=step,vel=30*step,
                                limits=torch.tensor([-3.15,3.15]).expand(1,56,2),robot=robot)
        self.c = NaturalFace90Controller(plant,torch.zeros(2,6))
        self.q[:,self.c.fIDs] = self.c.fGoal
        fk = RobotKinematics()
        angles = dict(zip(names,self.q[0].tolist()))
        link = np.stack([fk.fk_link(angles,name) for name in labs])
        poses = torch.tensor(np.c_[link[:,:3,3],Rotation.from_matrix(link[:,:3,:3]).as_quat()])
        self.bodies = poses[None].repeat(2,1,1)
        self.c.e.robot.get_link_transforms = lambda:self.bodies
        self.g = TurnGuide(self.c)
        self.yes = torch.ones(2,dtype=torch.bool)
        self.radial = torch.tensor([[1.,0.]]*2,dtype=torch.double)
        self.c.radial.copy_(self.radial)
        self.tip = torch.tensor([[.03,0.,-.045]]*2,dtype=torch.double)
        self.obs = {k:self.yes.clone() for k in ('support','static_contact','structure_ok',
                                                  'allowed_layer_state','right_contact','index_contact')}
        self.obs.update(face_deg=torch.zeros(2),wrong_contact=~self.yes,contact_goal_reached=~self.yes)

    def plan(self, fresh=None, active=None, oriented=None):
        return self.g._path(~self.yes if fresh is None else fresh,
                            self.yes if active is None else active,self.radial,self.tip,
                            self.yes if oriented is None else oriented,self.obs)

    def test_attributed_near_pad_sweeps_while_static_marker_and_observer_stay_measured(self):
        self.obs['index_contact'][1]=False
        self.plan(fresh=self.yes)
        for _ in range(90): goal=self.plan()
        self.assertAlmostEqual(self.g.sweep_deg[0].item(),16.,places=9)
        self.assertAlmostEqual(self.g.sweep_deg[1].item(),0.,places=9)
        self.assertLess(goal[0,1].item(),-.008)
        self.assertAlmostEqual(goal[0,2].item(),-.0445,places=9)
        self.assertTrue(pad_near_marker(self.radial,goal,self.obs['face_deg'])[0])
        self.assertFalse(self.obs['contact_goal_reached'].any())
        self.assertTrue(self.g.recovering[1])

    def test_round50_passive_touch_retracts_from_current_pad_instead_of_loading_again(self):
        self.tip[0] = torch.tensor([.0373,.0016,-.0871])
        self.obs['index_contact'][0]=False
        goal=self.plan(fresh=self.yes)
        self.assertTrue(self.g.recovering[0]); self.assertFalse(self.g.drive_contact[0])
        self.assertAlmostEqual(goal[0,2].item(),self.tip[0,2].item()-.014,places=9)
        torch.testing.assert_close(goal[0,:2],self.tip[0,:2])
        for _ in range(20):torch.testing.assert_close(self.plan()[0],goal[0])
        self.tip[0]=goal[0];self.obs['right_contact'][0]=False
        self.plan()
        self.assertFalse(self.g.recovering[0]);self.assertFalse(self.g.drive_contact[0])

    def test_loss_static_support_geometry_and_release_control_advancement(self):
        self.plan(fresh=self.yes)
        for _ in range(12): self.plan()
        previous=self.g.sweep_deg.clone()
        self.obs['right_contact'][0]=False
        self.plan();self.assertFalse(self.g.drive_contact[0])
        self.assertEqual(self.g.sweep_deg[0],previous[0])
        for _ in range(8):self.plan()
        self.assertEqual(self.g.sweep_deg[0],0)
        self.obs['right_contact'][0]=True;self.obs['static_contact'][0]=False
        for _ in range(4):goal=self.plan()
        self.assertFalse(self.g.drive_contact[0]);self.assertLessEqual(goal[0,2],-.075)
        self.obs['contact_goal_reached'][1]=True
        self.assertAlmostEqual(self.plan()[1,2].item(),-.100,places=8)
        self.obs['static_contact'][0]=True;self.obs['wrong_contact'][0]=True
        self.plan();self.assertFalse(self.g.drive_contact[0])
        self.obs['wrong_contact'][0]=False;self.obs['structure_ok'][0]=False
        self.plan();self.assertFalse(self.g.drive_contact[0])

    def test_stationary_layer_unloads_after_bounded_drive(self):
        self.plan(fresh=self.yes)
        for _ in range(127):g=self.plan()
        self.assertTrue(self.g.recovering.all())
        self.assertLess(g[0,2].item(),-.06)
        self.assertFalse(self.g.drive_contact.any())

    def test_wrist_reorients_at_standoff_and_passive_joints_stay_at_handoff(self):
        core=torch.tensor([[.45,.15,1.32,0.,0.,0.,1.]]*2,dtype=torch.double)
        tip=self.c.tip.fk(self.q[:,self.c.tip.ix])[0]+self.c.root
        # FK link translations are relative to robot base; mimic true world poses.
        shift=torch.tensor([[0.,0.,0.],[8.,-4.,0.]],dtype=torch.double)
        self.bodies[:,:,:3]+=self.c.root+shift[:,None,:]
        local=tip-core[:,:3];core[:,:3]+=shift;tip+=shift
        self.obs['right_contact']=~self.yes; self.obs['index_contact']=~self.yes
        target=self.q.clone()
        for i in range(6):
            cmd=self.g.command(target,self.q,self.q,self.yes if i==0 else ~self.yes,
                               self.yes,torch.full((2,7),8.),core,tip,local,self.obs)
            torch.testing.assert_close(cmd[0],cmd[1])
            passive=[j for j in range(56) if j not in self.c.rp.ix]
            torch.testing.assert_close(cmd[:,passive],self.q[:,passive])
            self.assertLessEqual(float((cmd-target).abs().max()),.012001)
            target=cmd
        planned_margin=((self.g.vectors*self.g.axis[:,None]).sum(-1)-self.g.radii).amin(1)
        self.assertTrue((planned_margin>.002).all())
        self.assertTrue((self.g.goal_local[:,2]<=-.075).all())
        inactive=self.g.command(target,self.q,self.q,~self.yes,~self.yes,torch.zeros(2,7),
                                core,tip,local,self.obs)
        torch.testing.assert_close(inactive,target)
