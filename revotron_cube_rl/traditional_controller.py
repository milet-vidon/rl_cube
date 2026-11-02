"""Joint-trajectory teacher using initial cube pose; not a visual policy."""
from isaac_compat import tensor, quaternion_to_native, quaternion_from_native, physx_config, simulation_config, runtime_versions, LAB3
import numpy as np
import torch
from kinematics import RobotKinematics

class GraspController:
    def __init__(self,env,width=.04,backoff=.065,curl=.0,thumb=1.7,lift=.25,relax=0.,approach_height=.068):
        self.env=env;self.k=RobotKinematics();self.width=width;self.relax=relax
        c=env.cfg;xyz=tensor(env.cube.data.root_pos_w)[0].cpu().numpy().copy();q=dict(c['home'])
        self.phases=[]
        for name,w,z,duration in [('align',.12,c['table_z']+approach_height,2.),('close',width,c['table_z']+approach_height,3.),('lift',width,c['table_z']+lift,3.),('relax',width+relax,c['table_z']+lift,1.),('hold',width+relax,c['table_z']+lift,4.)]:
            for side,sgn in [('left',1),('right',-1)]:
                q[side+'_thumb_CMP_joint']=thumb
                for f in ('index','middle','ring','little'):
                    for joint,factor in [('MCP',1.),('PIP',1.2),('DIP',.7)]:
                        q[f'{side}_{f}_{joint}_joint']=0. if name=='align' else curl*factor
                q=self.k.fit_arm(q,side,np.array([xyz[0]-backoff,xyz[1]+sgn*w,z-c['robot_z']]),np.array([[0.,0,1],[-sgn,0,0],[0,-sgn,0]]))
            self.phases.append((name,torch.tensor([[q[n] for n in c['joint_names']]]),round(duration*c['control_hz'])))
        self.phase=0;self.tick=0;self.begin=env.targets.clone();self.name='align'
    def action(self):
        self.name,goal,steps=self.phases[self.phase]
        self.tick+=1;u=min(self.tick/steps,1.);u=u*u*(3-2*u)
        desired=(1-u)*self.begin+u*goal
        action=((desired-self.env.targets)/self.env.scales).clamp(-1,1)
        if self.tick>=steps and self.phase<len(self.phases)-1:
            self.phase+=1;self.tick=0;self.begin=goal.clone()
        return action

class DownwardController(GraspController):
    """Six-finger opposed grasp, with elevated wrists and sampled Cartesian paths."""
    def __init__(self,env,width=.096,lift=.12,relax=0.,**unused):
        self.env=env;self.k=RobotKinematics();self.width=width;self.relax=relax
        c=env.cfg;xyz=tensor(env.cube.data.root_pos_w)[0].cpu().numpy().copy();q=dict(c['home'])
        self.phases=[]
        # 23 mm pad pitch avoids adjacent finger mesh intersections (18 mm did not).
        fingers={'index':[.0211649,.7180875,.2224492,.0532635],
                 'middle':[.0532475,.5906342,.5489092,.2871546],
                 'ring':[.0772269,.6455451,.4083752,.1884852],
                 'little':[0.,.55,.45,.25]}
        thumb={'CMP':1.20,'CMR':.30,'MCP':.30,'PIP':.22,'DIP':.14}
        thumb={'CMP':1.20,'CMR':.30,'MCP':.30,'PIP':.22,'DIP':.14}
        poses={s:self.k.fk(q)[s+'_palm_link'][:3,3].copy() for s in ('left','right')}
        from scipy.spatial.transform import Rotation, Slerp
        rotations={s:self.k.fk(q)[s+'_palm_link'][:3,:3].copy() for s in poses}
        specifications=[('raise',.15,1.28,2.,False),('orient',.15,1.28,4.,True),
                        ('lower',.14,xyz[2]+.17530,2.,True),('close',width,xyz[2]+.17530,2.,True),
                        ('lift',width,xyz[2]+.17530+lift,3.,True),
                        ('hold',width+relax,xyz[2]+.17530+lift,3.,True)]
        for name,w,z,duration,down in specifications:
            destinations={s:np.array([xyz[0]-.01609,xyz[1]+sgn*w,z-c['robot_z']]) for s,sgn in [('left',1),('right',-1)]}
            dest_rots={s:np.array([[0.,-sgn,0],[-sgn,0,0],[0,0,-1]]) if down else rotations[s].copy() for s,sgn in [('left',1),('right',-1)]}
            starts={s:poses[s].copy() for s in poses}
            slerps={s:Slerp([0.,1.],Rotation.from_matrix(np.stack([rotations[s],dest_rots[s]]))) for s in poses}
            # Small Cartesian segments avoid the large arc of a single joint-space interpolation.
            segments=12 if name!='hold' else 1
            for j in range(1,segments+1):
                u=j/segments;v=u*u*(3-2*u)
                for side in ('left','right'):
                    if down:
                        for f,angles in fingers.items():
                            for joint,value in zip(('MPR','MCP','PIP','DIP'),angles):
                                q[f'{side}_{f}_{joint}_joint']=value*(v if name=='orient' else 1.)
                    q=self.k.fit_arm(q,side,(1-v)*starts[side]+v*destinations[side],slerps[side]([v]).as_matrix()[0])
                self.phases.append((name,torch.tensor([[q[n] for n in c['joint_names']]]),round(duration*c['control_hz']/segments)))
            poses=destinations;rotations=dest_rots
        self.phase=0;self.tick=0;self.begin=env.targets.clone();self.name='raise'


class StagedDownwardController(DownwardController):
    """Explicit arm-first FSM: orient/lower with open fingers, then close."""
    def __init__(self, env, width=.096, lift=.12, relax=0., grasp_mode='parallel',
                 tilt_deg=0., settle=.75, approach_height=.068, pregrasp_scale=.35, **unused):
        self.env=env; self.k=RobotKinematics(); self.width=width; self.relax=relax
        if grasp_mode not in ('parallel','oblique'):
            raise ValueError('grasp_mode must be parallel or oblique')
        self.grasp_mode=grasp_mode; self.tilt_deg=float(tilt_deg); self.settle=float(settle)
        self.pregrasp_scale=float(pregrasp_scale)
        if not 0.0 <= self.pregrasp_scale <= 1.0:
            raise ValueError('pregrasp_scale must be within [0,1]')
        c=env.cfg; xyz=tensor(env.cube.data.root_pos_w)[0].cpu().numpy().copy(); q=dict(c['home'])
        fingers={'index':[.0211649,.7180875,.2224492,.0532635],
                 'middle':[.0532475,.5906342,.5489092,.2871546],
                 'ring':[.0772269,.6455451,.4083752,.1884852],
                 'little':[0.,.55,.45,.25]}
        thumb={'CMP':1.20,'CMR':.30,'MCP':.30,'PIP':.22,'DIP':.14}
        poses={s:self.k.fk(q)[s+'_palm_link'][:3,3].copy() for s in ('left','right')}
        from scipy.spatial.transform import Rotation, Slerp
        rotations={s:self.k.fk(q)[s+'_palm_link'][:3,:3].copy() for s in poses}
        self.phases=[]
        z_offset=.17530 + (float(approach_height)-.068)
        specs=[('raise',.15,1.28,2.,False),('orient',.15,1.28,4.,True),
               ('lower_clearance',.14,xyz[2]+z_offset+.10,2.,True),
               ('settle',.14,xyz[2]+z_offset+.10,self.settle,True),
               ('close',.14,xyz[2]+z_offset+.10,2.,True),
               ('lower_contact',.14,xyz[2]+z_offset,2.,True),
               ('squeeze',width,xyz[2]+z_offset,2.,True),
               ('verify_contact',width,xyz[2]+z_offset,.2,True),
               ('lift',width,xyz[2]+z_offset+lift,3.,True),
               ('hold',width+relax,xyz[2]+z_offset+lift,3.,True)]
        for name,w,z,duration,down in specs:
            destinations={s:np.array([xyz[0]-.01609,xyz[1]+sgn*w,z-c['robot_z']])
                          for s,sgn in [('left',1),('right',-1)]}
            base_rot={s:np.array([[0.,-sgn,0],[-sgn,0,0],[0,0,-1]])
                      for s,sgn in [('left',1),('right',-1)]}
            dest_rots={}
            for s,sgn in [('left',1),('right',-1)]:
                if not down:
                    dest_rots[s]=rotations[s].copy()
                elif grasp_mode=='oblique':
                    dest_rots[s]=Rotation.from_euler('y',sgn*self.tilt_deg,degrees=True).as_matrix() @ base_rot[s]
                else:
                    dest_rots[s]=base_rot[s]
            starts={s:poses[s].copy() for s in poses}
            slerps={s:Slerp([0.,1.],Rotation.from_matrix(np.stack([rotations[s],dest_rots[s]]))) for s in poses}
            segments=12 if name!='hold' else 1
            for j in range(1,segments+1):
                u=j/segments; v=u*u*(3-2*u)
                for side in ('left','right'):
                    # Fingers remain open through settle. They are only
                    # commanded after the arm reaches the final palm pose.
                    if down:
                        # A partially curled pre-grasp keeps the open fingers
                        # above the table. The actual grip closes only after
                        # the arm pose gate, interpolating from pre-grasp to
                        # full closure over the two-second close phase.
                        if name in ('orient','lower_clearance','settle'):
                            close_scale=self.pregrasp_scale
                        elif name=='close':
                            close_scale=self.pregrasp_scale+(1.-self.pregrasp_scale)*v
                        elif name in ('lower_contact','squeeze','verify_contact','lift','hold'):
                            close_scale=1.
                        else:
                            close_scale=0.
                        for joint,value in thumb.items():
                            # This six-finger grasp does not use thumbs; preserve the collision-free home posture.
                            q[f'{side}_thumb_{joint}_joint']=c['home'][f'{side}_thumb_{joint}_joint']
                        for f,angles in fingers.items():
                            for joint,value in zip(('MPR','MCP','PIP','DIP'),angles):
                                q[f'{side}_{f}_{joint}_joint']=value*close_scale
                    q=self.k.fit_arm(q,side,(1-v)*starts[side]+v*destinations[side],
                                     slerps[side]([v]).as_matrix()[0])
                self.phases.append((name,torch.tensor([[q[n] for n in c['joint_names']]]),
                                    max(1,round(duration*c['control_hz']/segments))))
            poses=destinations; rotations=dest_rots
        self.phase=0; self.tick=0; self.begin=env.targets.clone(); self.name='raise'
        self.gate_wait=0; self.contact_streak=0; self.failure_reason=None

    def _arm_settled(self, goal):
        q=tensor(self.env.robot.data.joint_pos)[:,self.env.ids]
        arm=[i for i,n in enumerate(self.env.cfg['joint_names']) if n.startswith('Joint')]
        err=float((q-goal)[:,arm].abs().max())
        vel=float(tensor(self.env.robot.data.joint_vel)[:,self.env.ids][:,arm].abs().max())
        return err < .035 and vel < .18

    def action(self):
        if self.failure_reason:
            return torch.zeros_like(self.env.targets)
        metrics=getattr(self.env,'metrics',{})
        if metrics.get('robot_table_impulse',0.)>1e-6:
            self.failure_reason='robot_table_contact'
            return torch.zeros_like(self.env.targets)
        self.name,goal,steps=self.phases[self.phase]
        self.tick+=1; u=min(self.tick/steps,1.); u=u*u*(3-2*u)
        desired=(1-u)*self.begin+u*goal
        action=((desired-self.env.targets)/self.env.scales).clamp(-1,1)
        if self.tick>=steps and self.phase<len(self.phases)-1:
            next_name=self.phases[self.phase+1][0]
            ready=True
            if next_name=='close':
                ready=self._arm_settled(goal)
            if self.name=='verify_contact' and next_name=='lift':
                both=all(x>self.env.cfg['success']['contact_impulse_min'] for x in self.env.contact_impulse)
                self.contact_streak=self.contact_streak+1 if both else 0
                ready=self._arm_settled(goal) and self.contact_streak>=3
            if not ready:
                self.gate_wait+=1
                if self.gate_wait>=round(3*self.env.cfg['control_hz']):
                    self.failure_reason='settle_or_contact_gate_timeout'
                self.tick=steps-1
                return action
            self.gate_wait=0
            self.phase+=1; self.tick=0; self.begin=goal.clone()
        return action
