"""Experimental asymmetric teacher: left thumb opposition, right support.

Noro-inspired named-joint posture costs and quintic path timing. Main-arm lift
uses J1..J4, holding J5..J7 at the grasp targets; support follows the resulting
rigid grasp transform. The certified side-grasp controller is unchanged.
"""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp
from proximal_planner import ProximalPlanner
from isaac_compat import tensor,quaternion_from_native

ROOT=Path(__file__).resolve().parent

def timed_path(points,seconds,hz,velocity,acceleration,limits):
    # Preserve the bounds of the IK knots; an unconstrained cubic can overshoot
    # a soft limit between two otherwise valid configurations.
    points=np.asarray(points);curve=PchipInterpolator(np.linspace(0,1,len(points)),points,axis=0)
    for _ in range(8):
        n=max(2,int(np.ceil(seconds*hz)));u=np.linspace(0,1,n+1)
        q=curve(10*u**3-15*u**4+6*u**5)
        if np.any(q<limits[:,0]) or np.any(q>limits[:,1]):raise ValueError('Spline crossed joint limit')
        v=np.max(np.abs(np.diff(q,axis=0)),axis=0)*hz
        a=np.max(np.abs(np.diff(q,n=2,axis=0)),axis=0)*hz**2
        ratio=max(float(np.max(v/velocity)),float(np.sqrt(np.max(a/acceleration))))
        if ratio<=1.:return q.astype(np.float32)
        seconds*=ratio*1.03
    raise ValueError('Unable to time path within velocity/acceleration limits')

class OppositionPlanner(ProximalPlanner):
    def __init__(self,grasp_config):
        super().__init__();self.grasp=json.loads(Path(grasp_config).read_text())
        self.primary_closed=self.grasp['joint_targets']
        self.primary_open=self.grasp['open_joint_targets']
        self.fingers.update(self.grasp.get('support_joint_targets',{}))
        # Task-specific shoulder posture envelope, inside the user's unchanged
        # global soft limits. Deep rearward shoulder flexion touched the torso.
        if 'support_joint2_floor_deg' in self.grasp:
            self.k.limits['Joint2_R'][0]=max(self.k.limits['Joint2_R'][0],np.radians(self.grasp['support_joint2_floor_deg']))
        self.diagnostics=[]

    def support_after_lift(self,q,cube_xyz,cube_quaternion):
        """Replan against the held cube, after contact has changed its pose."""
        names=self.cfg['joint_names'];hz=self.cfg['control_hz']
        r=Rotation.from_quat(np.asarray(cube_quaternion)[[1,2,3,0]]).as_matrix()
        position=np.asarray(cube_xyz)-[0,0,self.cfg['robot_z']]+r@self.grasp['support_center_offset']
        rotation=r@np.asarray(self.grasp['support_rotation'])
        limits=np.array([self.k.limits[n] for n in names])
        velocity=np.array([.65 if n.startswith('Joint') else 1.5 for n in names])
        acceleration=np.array([2. if n.startswith('Joint') else 6. for n in names])
        phases=[]
        for phase,target in [('support_clear',position+r@np.array([0.,-.11,.015])),('support',position)]:
            start=self.k.fk_link(q,'right_palm_link');interpolate=Slerp([0,1],Rotation.from_matrix([start[:3,:3],rotation]))
            points=[np.array([q[n] for n in names])]
            for u in np.linspace(0,1,25)[1:]:
                q=self.fit(q,'right',(1-u)*start[:3,3]+u*target,interpolate([u]).as_matrix()[0])
                points.append(np.array([q[n] for n in names]))
            phases.append((phase,timed_path(points,2.4,hz,velocity,acceleration,limits)))
        endpoint=np.array([[q[n] for n in names]],dtype=np.float32)
        phases.extend([('contact',np.repeat(endpoint,13,axis=0)),('hold',np.repeat(endpoint,180,axis=0))])
        return phases

    def proximal_lift(self,q,start,pos):
        names=[f'Joint{i}_L' for i in range(1,5)]
        q0=np.array([q[n] for n in names]);lim=np.array([self.k.limits[n] for n in names])
        def residual(x):
            t=self.k.fk_link(q|dict(zip(names,x)),'left_palm_link')
            return np.r_[30*(t[:3,3]-pos),.1*Rotation.from_matrix(start[:3,:3]@t[:3,:3].T).as_rotvec(),.001*(x-q0)]
        fit=least_squares(residual,q0,bounds=(lim[:,0]+.002,lim[:,1]-.002),max_nfev=200)
        q=q|dict(zip(names,fit.x));t=self.k.fk_link(q,'left_palm_link')
        err=np.linalg.norm(t[:3,3]-pos);angle=np.linalg.norm(Rotation.from_matrix(start[:3,:3]@t[:3,:3].T).as_rotvec())
        if err>.002 or angle>np.radians(25):raise ValueError('Proximal-only lift outside position/tilt envelope')
        return q

    def plan(self,xyz,yaw=0.):
        q=self.cfg['home'].copy();names=self.cfg['joint_names'];hz=self.cfg['control_hz']
        yaw_r=Rotation.from_euler('z',yaw).as_matrix()
        base=np.array([[0.,0,1],[-1.,0,0],[0,-1.,0]])
        grasp_frame=yaw_r@base@Rotation.from_euler('z',self.grasp['frame_roll_deg'],degrees=True).as_matrix()
        main_rot=grasp_frame@np.array(self.grasp['cube_rotation_palm']).T
        main_pos=np.asarray(xyz)-main_rot@np.array(self.grasp['center_palm']);main_pos[2]-=self.cfg['robot_z']
        support_rot=yaw_r@np.array(self.grasp.get('support_rotation',self.rotation('right')))
        def support_pos(width):
            offset=np.array(self.grasp.get('support_center_offset',[-.175*np.cos(self.tilt),-.098,.175*np.sin(self.tilt)+.025]))
            offset[1]+=.098-width
            p=np.asarray(xyz)+yaw_r@offset
            p[2]-=self.cfg['robot_z'];return p
        goals=[('ready',.23,.06,'home',2.8),('preshape',.23,.06,'open',2.4),
               ('lower',.23,.04,'open',2.0),('approach',.20,.015,'open',1.6),
               ('squeeze',.20,0.,'closed',2.4),('support',.098,0.,'closed',1.4),
               ('contact',.098,0.,'closed',.4)]
        primary_first=bool(self.grasp.get('primary_lift_before_support',False))
        if primary_first:goals=goals[:5]
        phases=[];limits=np.array([self.k.limits[n] for n in names])
        vel=np.array([.65 if n.startswith('Joint') else 1.5 for n in names])
        acc=np.array([2.0 if n.startswith('Joint') else 6. for n in names])
        for phase,width,above,posture,seconds in goals:
            target={'left':main_pos+np.array([0,0,above]),'right':support_pos(width)+np.array([0,0,above+(.15 if primary_first else 0.)])}
            rot={'left':main_rot,'right':support_rot};start={s:self.k.fk_link(q,s+'_palm_link') for s in rot}
            slerp={s:Slerp([0,1],Rotation.from_matrix([start[s][:3,:3],rot[s]])) for s in rot}
            fstart=q.copy();fgoal=q.copy()
            if posture!='home':
                fgoal.update({n:v for n,v in self.fingers.items() if n.startswith('right_')})
                fgoal.update(self.primary_open if posture=='open' else self.primary_closed)
            points=[np.array([q[n] for n in names])]
            for u in np.linspace(0,1,13)[1:]:
                for n in names:
                    if not n.startswith('Joint'):q[n]=(1-u)*fstart[n]+u*fgoal[n]
                for s in rot:
                    try:q=self.fit(q,s,(1-u)*start[s][:3,3]+u*target[s],slerp[s]([u]).as_matrix()[0])
                    except ValueError as exc:raise ValueError(f'{phase} {s} u={u}: {exc}') from exc
                points.append(np.array([q[n] for n in names]))
            phases.append((phase,timed_path(points,seconds,hz,vel,acc,limits)))
        main_start=self.k.fk_link(q,'left_palm_link');support_start=self.k.fk_link(q,'right_palm_link')
        if primary_first:
            support_start=np.eye(4);support_start[:3,3]=support_pos(.098);support_start[:3,:3]=support_rot
        wrist={n:q[n] for n in ['Joint5_L','Joint6_L','Joint7_L']};points=[np.array([q[n] for n in names])]
        lift_forward=float(self.grasp.get('lift_forward_m',.05));lift_height=float(self.grasp.get('lift_height_m',.12))
        for u in np.linspace(0,1,25)[1:]:
            q=self.proximal_lift(q,main_start,main_start[:3,3]+yaw_r@np.array([lift_forward*u,0.,lift_height*u]))
            desired=self.k.fk_link(q,'left_palm_link')@np.linalg.inv(main_start)@support_start
            if not primary_first:q=self.fit(q,'right',desired[:3,3],desired[:3,:3])
            points.append(np.array([q[n] for n in names]))
        assert all(q[n]==v for n,v in wrist.items())
        phases.append(('primary_lift' if primary_first else 'lift',timed_path(points,4.,hz,vel,acc,limits)))
        if primary_first:
            begin=self.k.fk_link(q,'right_palm_link');interpolate=Slerp([0,1],Rotation.from_matrix([begin[:3,:3],desired[:3,:3]]))
            points=[np.array([q[n] for n in names])]
            for u in np.linspace(0,1,25)[1:]:
                q=self.fit(q,'right',(1-u)*begin[:3,3]+u*desired[:3,3],interpolate([u]).as_matrix()[0])
                points.append(np.array([q[n] for n in names]))
            phases.append(('support',timed_path(points,2.4,hz,vel,acc,limits)))
            phases.append(('contact',np.repeat(np.array([[q[n] for n in names]],dtype=np.float32),13,axis=0)))
        phases.append(('hold',np.repeat(np.array([[q[n] for n in names]],dtype=np.float32),180,axis=0)))
        return phases

class OppositionController:
    def __init__(self,env,phases):
        self.env=env;self.phases=phases;self.index=0;self.tick=0;self.name=phases[0][0]
        self.failure_reason=None;self.wait=0;self.contact_streak=0;self.lost=0;self.finger_contacts={}
        self.nominal_target=None
        self.primary_streak=0
        self.settle_tolerance=.035
        self.finger_arrival=None
    def action(self):
        if self.failure_reason:return torch.zeros_like(self.env.targets)
        self.name,path=self.phases[self.index];goal=torch.from_numpy(path[min(self.tick,len(path)-1)]).reshape(1,56)
        q=tensor(self.env.robot.data.joint_pos)[:,self.env.ids]
        v=tensor(self.env.robot.data.joint_vel)[:,self.env.ids]
        arm=[i for i,n in enumerate(self.env.cfg['joint_names']) if n.startswith('Joint')]
        self.nominal_target=goal.clone()
        if self.finger_arrival and (self.name in ('support_clear','support','contact','primary_lift','lift','hold') or (self.name=='squeeze' and self.tick>=len(path)-1)):
            state=tensor(self.env.cube.data.root_state_w)[0]
            corrected=self.finger_arrival.update(q[0].numpy(),goal[0].numpy(),self.finger_contacts.get('left',{}),state[:3].numpy(),quaternion_from_native(state[3:7]).numpy())
            goal=torch.from_numpy(corrected).reshape(1,56)
        both=all(x>self.env.cfg['success']['contact_impulse_min'] for x in self.env.contact_impulse)
        primary=self.finger_contacts.get('left',{})
        topology=primary.get('thumb',0.)>1e-5 and sum(primary.get(f,0.)>1e-5 for f in ['index','middle','ring','little'])>=3
        all_primary=all(primary.get(f,0.)>1e-5 for f in ['thumb','index','middle','ring','little'])
        self.primary_streak=self.primary_streak+1 if all_primary else 0
        self.contact_streak=self.contact_streak+1 if both and topology else 0
        if self.name in ('lift','hold'):
            self.lost=0 if both and primary.get('thumb',0.)>1e-5 else self.lost+1
            if self.lost>=10:self.failure_reason='lost_thumb_or_support';return torch.zeros_like(self.env.targets)
        if self.name=='primary_lift':
            self.lost=0 if topology else self.lost+1
            if self.lost>=10:self.failure_reason='lost_primary_grasp';return torch.zeros_like(self.env.targets)
        self.tick+=1
        if self.tick>=len(path) and self.index<len(self.phases)-1:
            settled=float((q-goal)[:,arm].abs().max())<self.settle_tolerance and float(v[:,arm].abs().max())<.18
            ready=settled and (self.contact_streak>=10 if self.name=='contact' else True)
            if self.name=='squeeze':ready=ready and self.primary_streak>=10
            if self.name=='primary_lift':
                ready=ready and self.primary_streak>=10 and self.env.metrics.get('height_m',0)>=.08 and not self.env.metrics.get('cube_table_contact',True)
            if ready:self.index+=1;self.tick=0;self.wait=0
            else:
                self.tick=len(path)-1;self.wait+=1
                if self.wait>120:self.failure_reason='contact_topology_or_settle_timeout'
        return (goal-self.env.targets)/self.env.scales

class OppositionTeacher(OppositionController):
    def __init__(self,env,grasp_config):
        self.planner=OppositionPlanner(grasp_config)
        if self.planner.grasp.get('arm_integral_tracking',False):
            raise ValueError('Arm integral tracking was rejected for oscillation; use the damped PD configuration.')
        xyz=tensor(env.cube.data.root_pos_w)[0].numpy().copy()
        quat=quaternion_from_native(tensor(env.cube.data.root_state_w)[0,3:7]).numpy()
        yaw=Rotation.from_quat(quat[[1,2,3,0]]).as_euler('xyz')[2]
        super().__init__(env,self.planner.plan(xyz,yaw))
        self.settle_tolerance=float(self.planner.grasp.get('arm_settle_tolerance_rad',.035))
        if self.planner.grasp.get('contact_anchors'):
            from opposition_feedback import FingerArrival
            self.finger_arrival=FingerArrival(self.planner.k,self.planner.grasp,env.cfg['joint_names'],env.cfg['robot_z'])

    def action(self):
        before=self.index;action=super().action()
        if self.name=='primary_lift' and self.index!=before and self.planner.grasp.get('replan_support_after_lift',False):
            q=dict(zip(self.env.cfg['joint_names'],self.nominal_target[0].tolist()))
            state=tensor(self.env.cube.data.root_state_w)[0]
            suffix=self.planner.support_after_lift(q,state[:3].numpy(),quaternion_from_native(state[3:7]).numpy())
            self.phases=self.phases[:self.index]+suffix
        return action
