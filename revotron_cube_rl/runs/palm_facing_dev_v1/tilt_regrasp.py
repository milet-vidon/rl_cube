"""Experimental 90 degree rightward object tilt after the certified left grasp.

This module leaves the accepted v20 pickup and its evidence unchanged.
"""
import json, os, subprocess, tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from opposition_teacher import OppositionPlanner,timed_path
from proximal_planner import PROXIMAL_WEIGHTS
from opposition_teacher import OppositionTeacher,OppositionController
from isaac_compat import tensor,quaternion_from_native
import torch

def set_right_arm_gains(env,regrasp):
    """Keep pickup unchanged; use a damped, stiffer arm for loaded regrasp."""
    names=[f'Joint{i}_R' for i in range(1,8)]
    ids=[env.ids[env.cfg['joint_names'].index(n)] for n in names]
    stiffness=[300. if regrasp else env.cfg['arm_stiffness'][n] for n in names]
    damping=20. if regrasp else 8.
    env.robot.write_joint_stiffness_to_sim_index(stiffness=torch.tensor([stiffness],dtype=torch.float32),joint_ids=ids)
    env.robot.write_joint_damping_to_sim_index(damping=torch.full((1,7),damping,dtype=torch.float32),joint_ids=ids)

class TiltPlanner(OppositionPlanner):
    tilt_degrees=90.
    grasp_symmetry=np.array([[0.,-1.,0.],[-1.,0.,0.],[0.,0.,-1.]])
    right_contact_axis=1
    right_thumb_sign=1
    pregrasp_offset_cube=np.array([-.10,0.,-.06])
    little_contact_z=None
    keep_right_side=False
    def __init__(self,grasp_config):
        super().__init__(grasp_config)
        # The old single-finger support used a posture-specific -100 degree
        # shoulder floor. Regrasp uses a different shoulder orientation and
        # explicit full-body mesh planning, within the user's original limits.
        self.k.limits['Joint2_R']=list(self.cfg['arm_soft_limits']['Joint2_R'])

    def regrasp_path(self,q,measured,cube_position,cube_quaternion):
        """Mirror the actual held grasp onto a free pair of cube faces."""
        names=self.cfg['joint_names'];m=np.diag([1.,-1.,1.,1.])
        Q=self.grasp_symmetry.copy()
        cube=np.eye(4);cube[:3,:3]=Rotation.from_quat(np.asarray(cube_quaternion)[[1,2,3,0]]).as_matrix()
        cube[:3,3]=np.asarray(cube_position)-[0,0,self.cfg['robot_z']]
        lp=np.linalg.inv(cube)@self.k.fk_link(measured,'left_palm_link')
        symmetry=np.eye(4);symmetry[:3,:3]=Q
        local=symmetry@m@lp@m;target=cube@local
        closed={n.replace('left_','right_'):v for n,v in measured.items() if n.startswith('left_')}
        opened={n:closed[n]+self.primary_open[n.replace('right_','left_')]-self.primary_closed[n.replace('right_','left_')] for n in closed}
        # The inherited little-finger preshape grazed the cube during lateral
        # insertion. Keep it extended until the palm reaches the grasp pose.
        opened.update(right_little_MCP_joint=.3,right_little_PIP_joint=0.,right_little_DIP_joint=0.)
        final=self.fit_tilt(q,'right',target[:3,3],target[:3,:3])
        if self.little_contact_z is not None:
            anchor=self.grasp['contact_anchors']['little']
            link=anchor['link'].replace('left_','right_')
            point=np.diag([1.,-1.,1.])@np.asarray(anchor['point_local'])
            little_names=[n for n in closed if n.startswith('right_little_')]
            ref=np.array([closed[n] for n in little_names])
            def little_position(values):
                frame=self.k.fk_link(final|closed|dict(zip(little_names,values)),link)
                return cube[:3,:3].T@(frame[:3,:3]@point+frame[:3,3]-cube[:3,3])
            target_little=little_position(ref);target_little[2]=self.little_contact_z
            bounds=np.array([self.k.limits[n] for n in little_names])
            lo=np.maximum(bounds[:,0]+.02,ref-.5);hi=np.minimum(bounds[:,1]-.02,ref+.5)
            fit=least_squares(lambda x:np.r_[80*(little_position(x)-target_little),.03*(x-ref)],np.clip(ref,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=100)
            if np.linalg.norm(little_position(fit.x)-target_little)>.003:
                raise ValueError('Right little finger cannot clear the opposing hand')
            closed.update(zip(little_names,fit.x))
        # A joint-angle 'opening' inherited from the left pickup moved the
        # mirrored thumb inward. Open geometrically along its face normal.
        thumb_names=[n for n in closed if n.startswith('right_thumb_')]
        anchor=self.grasp['contact_anchors']['thumb']
        thumb_link=anchor['link'].replace('left_','right_')
        thumb_point=np.diag([1.,-1.,1.])@np.asarray(anchor['point_local'])
        def thumb_position(values):
            frame=self.k.fk_link(final|dict(zip(thumb_names,values)),thumb_link)
            return frame[:3,:3]@thumb_point+frame[:3,3]
        thumb_target=thumb_position([closed[n] for n in thumb_names])+cube[:3,self.right_contact_axis]*self.right_thumb_sign*.020
        thumb_ref=np.array([opened[n] for n in thumb_names])
        thumb_limits=np.array([self.k.limits[n] for n in thumb_names])
        thumb_lo=np.maximum(thumb_limits[:,0]+.02,thumb_ref-.7)
        thumb_hi=np.minimum(thumb_limits[:,1]-.02,thumb_ref+.7)
        thumb_fit=least_squares(lambda x:np.r_[80*(thumb_position(x)-thumb_target),.05*(x-thumb_ref)],
            np.clip(thumb_ref,thumb_lo+1e-6,thumb_hi-1e-6),bounds=(thumb_lo,thumb_hi),max_nfev=100)
        if np.linalg.norm(thumb_position(thumb_fit.x)-thumb_target)>.002:
            raise ValueError('Right thumb cannot reach the open pregrasp clearance')
        opened.update(zip(thumb_names,thumb_fit.x))
        # Approach along the two tangents of the grasped faces. Keeping the
        # offset in cube coordinates avoids pushing into a face as yaw varies.
        pre_offset=cube[:3,:3]@self.pregrasp_offset_cube
        pre=target.copy();pre[:3,3]+=pre_offset
        approach=self.fit_tilt(final,'right',pre[:3,3],pre[:3,:3]);approach.update(opened)
        limits=np.array([self.k.limits[n] for n in names])
        velocity=np.array([.35 if n.startswith('Joint') else 1. for n in names])
        acceleration=np.array([.7 if n.startswith('Joint') else 4. for n in names])
        def timed(points,seconds):return timed_path(points,seconds,self.cfg['control_hz'],velocity,acceleration,limits)
        request={'initial':measured,'opened':opened,'goal':approach,'cube_position':np.asarray(cube_position).tolist(),'cube_quaternion':np.asarray(cube_quaternion).tolist(),'keep_right_side':self.keep_right_side}
        # The preinstalled convex-geometry package uses the host NumPy ABI.
        # Keep it in a separate process from Isaac's conda environment.
        with tempfile.TemporaryDirectory(prefix='cube_regrasp_') as directory:
            source=Path(directory)/'request.json';destination=Path(directory)/'route.json';source.write_text(json.dumps(request))
            environment=os.environ.copy()
            for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD'):environment.pop(key,None)
            environment['PYTHONPATH']='/opt/ros/jazzy/lib/python3.12/site-packages'
            environment['LD_LIBRARY_PATH']='/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu'
            completed=subprocess.run(['/usr/bin/python3',str(Path(__file__).with_name('tilt_collision_plan.py')),str(source),str(destination)],env=environment,capture_output=True,text=True,timeout=120)
            if completed.returncode:raise ValueError('Collision route search failed: '+completed.stderr[-1500:])
            route=json.loads(destination.read_text())
        phases=[];opened_q=q|opened
        phases.append(('right_preshape',timed([[q[n] for n in names],[opened_q[n] for n in names]],2.)))
        q=opened_q
        for waypoint in route['waypoints'][1:]:
            endpoint=q|dict(zip(route['right_names'],waypoint))
            phases.append(('right_route',timed([[q[n] for n in names],[endpoint[n] for n in names]],2.5)))
            q=endpoint
        points=[[q[n] for n in names]]
        for u in np.linspace(0,1,25)[1:]:
            q=self.fit_tilt(q,'right',(1-u)*pre[:3,3]+u*target[:3,3],target[:3,:3])
            points.append([q[n] for n in names])
        phases.append(('right_approach',timed(points,3.)))
        end=q|closed;phases.append(('right_close',timed([[q[n] for n in names],[end[n] for n in names]],2.)))
        phases.append(('bilateral_hold',np.repeat(np.array([[end[n] for n in names]],dtype=np.float32),181,axis=0)))
        return phases,{'Q':Q.tolist(),'right_palm_in_cube':local.tolist(),'right_closed':closed,'collision_route':route,'thumb_open_face_normal_offset_m':.020,'pregrasp_offset_cube_m':self.pregrasp_offset_cube.tolist(),'pregrasp_offset_world_m':pre_offset.tolist(),'right_contact_axis':self.right_contact_axis,'right_thumb_sign':self.right_thumb_sign,'little_contact_z':self.little_contact_z}

    def fit_tilt(self,q,side,position,rotation):
        names=[f'Joint{i}_{"L" if side=="left" else "R"}' for i in range(1,8)]
        limits=np.array([self.k.limits[n] for n in names]);margin=np.radians(2.)
        initial=np.clip([q[n] for n in names],limits[:,0]+margin,limits[:,1]-margin)
        def residual(x):
            candidate=q|dict(zip(names,x));t=self.k.fk_link(candidate,side+'_palm_link')
            clearance=np.minimum(x-limits[:,0],limits[:,1]-x)
            elbow=self.k.fk_link(candidate,'Link4_'+('L' if side=='left' else 'R'))[:3,3]
            # Conservative elbow waypoint envelope; PhysX remains the final
            # whole-body collision gate, including the entire forearm mesh.
            radius=np.hypot(elbow[0]/.20,elbow[1]/.24)
            torso_penalty=5.*max(1.08-radius,0.) if side=='left' else 0.
            return np.r_[50*(t[:3,3]-position),6*Rotation.from_matrix(rotation@t[:3,:3].T).as_rotvec(),
                         .008*np.sqrt(PROXIMAL_WEIGHTS)*(x-initial),.12*np.maximum(np.radians(10)-clearance,0.),torso_penalty]
        fitted=least_squares(residual,initial,bounds=(limits[:,0]+margin,limits[:,1]-margin),max_nfev=160,
                             ftol=1e-9,xtol=1e-9,gtol=1e-9)
        result=q|dict(zip(names,fitted.x));t=self.k.fk_link(result,side+'_palm_link')
        if np.linalg.norm(t[:3,3]-position)>.002 or Rotation.from_matrix(rotation@t[:3,:3].T).magnitude()>np.radians(1):
            raise ValueError('Tilt IK outside position/orientation envelope')
        return result

    def tilt_path(self,q,cube_position,shift=(0.,0.,0.),degrees=None):
        degrees=self.tilt_degrees if degrees is None else degrees
        center=np.asarray(cube_position)-[0,0,self.cfg['robot_z']]
        start=self.k.fk_link(q,'left_palm_link');names=self.cfg['joint_names']
        limits=np.array([self.k.limits[n] for n in names])
        velocity=np.array([.40 if n.startswith('Joint') else 1.5 for n in names])
        acceleration=np.array([.8 if n.startswith('Joint') else 6. for n in names])
        parking=None;self.clear_steps=0
        if np.linalg.norm(shift)>0:
            parked=self.k.fk_link(q,'right_palm_link');points=[np.array([q[n] for n in names])]
            for u in np.linspace(0,1,25)[1:]:
                q=self.fit_tilt(q,'right',parked[:3,3]+u*np.array([0.,-.15,.08]),parked[:3,:3])
                points.append(np.array([q[n] for n in names]))
            parking=timed_path(points,3.,self.cfg['control_hz'],velocity,acceleration,limits)
            self.clear_steps=len(parking)
        points=[np.array([q[n] for n in names])]
        for u in np.linspace(0,1,49)[1:]:
            r=Rotation.from_euler('x',degrees*u,degrees=True).as_matrix()
            pos=center+r@(start[:3,3]-center)
            q=self.fit_tilt(q,'left',pos,r@start[:3,:3]);points.append(np.array([q[n] for n in names]))
        path=timed_path(points,6.,self.cfg['control_hz'],velocity,acceleration,limits)
        if parking is not None:path=np.concatenate([parking,path])
        self.tilt_steps=len(path)
        if np.linalg.norm(shift)>0:
            # Rotate in the proven open workspace, then translate with the
            # elbow held outside the torso envelope. Combining both motions
            # made the elbow sweep through the torso in development v2.
            start=self.k.fk_link(q,'left_palm_link');points=[np.array([q[n] for n in names])]
            for u in np.linspace(0,1,25)[1:]:
                q=self.fit_tilt(q,'left',start[:3,3]+u*np.asarray(shift),start[:3,:3])
                points.append(np.array([q[n] for n in names]))
            transfer=timed_path(points,4.,self.cfg['control_hz'],velocity,acceleration,limits)
            path=np.concatenate([path,transfer])
        return path

class TiltTeacher:
    planner_type=TiltPlanner
    def __init__(self,env,config,shift=(0.,0.,0.),regrasp=False):
        self.env=env;self.base=OppositionTeacher(env,config);self.planner=self.planner_type(config)
        self.shift=shift;self.mode='pickup';self.name='ready';self.failure_reason=None
        self.finger_contacts={};self.nominal_target=None;self.reference=None;self.path=None
        self.tick=0;self.wait=0;self.lost=0;self.hold=0;self.accepted=False;self.tilt_error_deg=90.
        self.regrasp=regrasp;self.right_phases=[];self.right_index=0;self.right_tick=0;self.right_hold=0
        self.regrasp_geometry=None;self.right_arrival=None;self.contact_points=[]
    def action(self):
        if self.mode=='pickup':
            self.base.finger_contacts=self.finger_contacts;before=self.base.index
            action=OppositionController.action(self.base)
            self.name=self.base.name;self.nominal_target=self.base.nominal_target
            self.failure_reason=self.base.failure_reason
            if self.name=='primary_lift' and self.base.index!=before:
                state=tensor(self.env.cube.data.root_state_w)[0]
                self.reference=quaternion_from_native(state[3:7]).numpy().copy()
                q=dict(zip(self.env.cfg['joint_names'],self.nominal_target[0].tolist()))
                self.path=self.planner.tilt_path(q,state[:3].numpy(),self.shift)
                self.mode='tilt';self.tick=0
            return action
        self.name='tilt_90' if self.tick<len(self.path) else 'tilt_hold'
        if self.tick<self.planner.clear_steps:self.name='right_clear'
        if self.planner.tilt_steps<=self.tick<len(self.path):self.name='align_right'
        goal=self.path[min(self.tick,len(self.path)-1)].copy()
        if self.mode=='regrasp':
            self.name,right_path=self.right_phases[self.right_index]
            goal=right_path[min(self.right_tick,len(right_path)-1)].copy()
        self.nominal_target=torch.from_numpy(goal.copy()).reshape(1,56)
        state=tensor(self.env.cube.data.root_state_w)[0];quat=quaternion_from_native(state[3:7]).numpy()
        actual=tensor(self.env.robot.data.joint_pos)[0,self.env.ids].numpy()
        primary=self.finger_contacts.get('left',{})
        if self.base.finger_arrival:
            goal=self.base.finger_arrival.update(actual,goal,primary,state[:3].numpy(),quat)
        all_fingers=all(primary.get(f,0)>1e-5 for f in ('thumb','index','middle','ring','little'))
        retained=primary.get('thumb',0)>1e-5 and sum(primary.get(f,0)>1e-5 for f in ('index','middle','ring','little'))>=3
        self.lost=0 if retained else self.lost+1
        if self.lost>=10:self.failure_reason='primary_grasp_lost_during_tilt'
        ref=Rotation.from_quat(self.reference[[1,2,3,0]]).as_matrix()
        current=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        target=Rotation.from_euler('x',self.planner.tilt_degrees,degrees=True).as_matrix()@ref
        self.tilt_error_deg=float(np.degrees(Rotation.from_matrix(target@current.T).magnitude()))
        safe_pose=self.env.metrics.get('height_m',0)>=.08 and not self.env.metrics.get('cube_table_contact',True)
        qualified=self.tick>=len(self.path) and all_fingers and safe_pose and self.tilt_error_deg<=5.
        self.hold=self.hold+1 if qualified else 0
        self.accepted=self.hold>=90 and not self.regrasp
        if self.mode=='tilt' and self.hold>=90 and self.regrasp:
            set_right_arm_gains(self.env,True)
            q=dict(zip(self.env.cfg['joint_names'],goal.tolist()));measured=dict(zip(self.env.cfg['joint_names'],actual.tolist()))
            self.right_phases,self.regrasp_geometry=self.planner.regrasp_path(q,measured,state[:3].numpy(),quat)
            self.right_arrival=MirrorArrival(self.planner.k,self.env.cfg,self.planner.grasp,measured,state[:3].numpy(),quat,self.regrasp_geometry,self.contact_points)
            self.mode='regrasp'
        elif self.mode=='regrasp':
            right=self.finger_contacts.get('right',{})
            if self.name in ('right_close','bilateral_hold') and self.right_arrival:
                goal=self.right_arrival.update(actual,goal,right,state[:3].numpy(),quat)
            right_all=all(right.get(f,0)>1e-5 for f in ('thumb','index','middle','ring','little'))
            self.right_hold=self.right_hold+1 if qualified and right_all and self.name=='bilateral_hold' else 0
            self.accepted=self.right_hold>=90
            self.right_tick+=1
            if self.right_tick>=len(right_path) and self.right_index<len(self.right_phases)-1:
                self.right_index+=1;self.right_tick=0
            if self.name=='bilateral_hold' and self.right_tick>360:self.failure_reason='right_grasp_hold_timeout'
        self.tick+=1
        if self.mode=='tilt' and self.tick>len(self.path)+300:self.failure_reason='tilt_hold_timeout'
        return (torch.from_numpy(goal).reshape(1,56)-self.env.targets)/self.env.scales

class MirrorArrival:
    """Bounded right fingertip correction from measured left contact geometry."""
    def __init__(self,k,cfg,grasp,measured,position,quat,geometry,points):
        self.k=k;self.names=cfg['joint_names'];self.robot_z=cfg['robot_z'];self.tick=0;self.targets={}
        self.reference=geometry['right_closed'];self.anchors={};m=np.diag([1.,-1.,1.]);self.symmetry=np.array(geometry['Q'])@m
        cube_r=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix();position=np.asarray(position)-[0,0,self.robot_z]
        for f,anchor in grasp['contact_anchors'].items():
            contacts=[p for p in points if p['side']=='left' and p['finger']==f]
            if not contacts:raise ValueError('Missing measured left contact for mirrored grasp')
            contact=max(contacts,key=lambda p:p['impulse_Ns']);world=np.array(contact['position_world_m'])-[0,0,self.robot_z]
            t=k.fk_link(measured,anchor['link'])
            target_cube=self.symmetry@cube_r.T@(world-position)
            # Aim 1.5 mm inside the nominal face, as in the accepted left
            # arrival controller. A target exactly on the measured surface
            # left three right fingertips hovering without contact force.
            axis=geometry.get('right_contact_axis',1);thumb_sign=geometry.get('right_thumb_sign',1)
            target_cube[axis]=.027*thumb_sign*(1 if f=='thumb' else -1)
            tangent=[i for i in range(3) if i!=axis]
            if f=='index':target_cube[tangent]=np.clip(target_cube[tangent],-.023,.023)
            if f=='little' and geometry.get('little_contact_z') is not None:target_cube[2]=geometry['little_contact_z']
            self.anchors[f]={'link':anchor['link'].replace('left_','right_'),'point_local':m@t[:3,:3].T@(world-t[:3,3]),'target_cube':target_cube}
    def update(self,measured,nominal,contacts,position,quat):
        q=dict(zip(self.names,measured));out=nominal.copy();r=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()
        pos=np.asarray(position)-[0,0,self.robot_z]
        for f,anchor in self.anchors.items():
            names=[n for n in self.reference if n.startswith('right_'+f+'_')]
            if self.tick%3==0 and contacts.get(f,0)<=1e-5:
                ref=np.array([self.reference[n] for n in names]);lim=np.array([self.k.limits[n] for n in names]);radius=np.array([(.10 if f=='index' else .005) if 'MPR' in n else (.45 if f=='index' else .25) for n in names])
                lo=np.maximum(lim[:,0]+.02,ref-radius);hi=np.minimum(lim[:,1]-.02,ref+radius);target=pos+r@anchor['target_cube']
                def residual(x):
                    t=self.k.fk_link(q|dict(zip(names,x)),anchor['link']);point=t[:3,3]+t[:3,:3]@anchor['point_local']
                    return np.r_[np.array([60.,15.,60.])*(self.symmetry.T@r.T@(point-target)),.015*(x-ref)]
                fit=least_squares(residual,np.clip([q[n] for n in names],lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=30);self.targets.update(zip(names,fit.x))
            for n in names:
                i=self.names.index(n);previous=self.targets.get('_applied_'+n,float(nominal[i]));target=previous if contacts.get(f,0)>1e-5 else self.targets.get(n,previous)
                out[i]=previous+np.clip(target-previous,-.005,.005);self.targets['_applied_'+n]=float(out[i])
        self.tick+=1;return out
