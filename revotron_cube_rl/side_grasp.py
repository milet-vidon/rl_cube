"""Opposed side-grasp geometry, with explicit wrist posture and contact gates."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from kinematics import RobotKinematics
from isaac_compat import tensor

ROOT=Path(__file__).resolve().parent

class SidePlanner:
    def __init__(self, wrist_deg=35., tilt_deg=15., finger_mode='curled'):
        self.k=RobotKinematics();self.cfg=json.loads((ROOT/'config.json').read_text())
        self.wrist=np.radians(wrist_deg);self.tilt=np.radians(tilt_deg);self.finger_mode=finger_mode
        self.fingers=self.finger_posture()

    def rotation(self,side):
        s=1 if side=='left' else -1
        return Rotation.from_euler('y',self.tilt).as_matrix()@np.array([[0.,0,1],[-s,0,0],[0,-s,0]])

    def fit(self,q,side,pos,rot,wrist=None):
        names=[f'Joint{i}_{"L" if side=="left" else "R"}' for i in range(1,8)]
        limits=np.array([self.k.limits[n] for n in names]);q0=np.clip([q[n] for n in names],limits[:,0]+.002,limits[:,1]-.002)
        def residual(x):
            t=self.k.fk_link(q|dict(zip(names,x)),side+'_palm_link')
            return np.r_[20*(t[:3,3]-pos),3*Rotation.from_matrix(rot@t[:3,:3].T).as_rotvec(),.04*(x[6]-(self.wrist if wrist is None else wrist)),.001*(x-q0)]
        fit=least_squares(residual,q0,bounds=(limits[:,0]+.002,limits[:,1]-.002),max_nfev=160,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        result=q|dict(zip(names,fit.x));t=self.k.fk_link(result,side+'_palm_link')
        err=np.linalg.norm(t[:3,3]-pos);ang=np.linalg.norm(Rotation.from_matrix(rot@t[:3,:3].T).as_rotvec())
        if err>.002 or ang>np.radians(1):raise ValueError(f'Side IK infeasible: {side}, pos={err}, angle={np.degrees(ang)}')
        return result

    def finger_posture(self):
        q=self.cfg['home'].copy()
        for side,sgn in [('left',1),('right',-1)]:
            palm_inv=np.linalg.inv(self.k.fk_link(q,side+'_palm_link'))
            for finger,y in [('index',-.028),('middle',0.),('ring',.028)]:
                names=[f'{side}_{finger}_{j}_joint' for j in ['MPR','MCP','PIP','DIP']]
                limits=np.array([self.k.limits[n] for n in names]);initial=np.array([0.,.6,.45,.2] if self.finger_mode=='curled' else [0.,.25,.1,.05])
                def residual(x):
                    t=palm_inv@self.k.fk_link(q|dict(zip(names,x)),f'{side}_{finger}_tip_Link')
                    if self.finger_mode=='curled':return np.r_[50*(t[:3,3]-np.array([.060,sgn*y,.167])),.01*(x-initial)]
                    return np.r_[50*(t[:3,3]-np.array([.028,sgn*y,.195])),.2*(sum(x[1:])-.4),.01*(x-initial)]
                fit=least_squares(residual,initial,bounds=(limits[:,0]+.003,limits[:,1]-.003),max_nfev=160)
                q.update(zip(names,fit.x))
            for joint,v in zip(['MPR','MCP','PIP','DIP'],[-.30,1.1,1.1,.8]):q[f'{side}_little_{joint}_joint']=v
            q[f'{side}_thumb_CMP_joint']=.03
        return {n:float(v) for n,v in q.items() if not n.startswith('Joint')}

    def plan(self,xyz,width=.096,palm_offset=0.):
        q=self.cfg['home'].copy();ts=self.k.fk(q)
        poses={s:ts[s+'_palm_link'][:3,3] for s in ['left','right']}
        rots={s:ts[s+'_palm_link'][:3,:3] for s in poses}
        self.segments=[];self.phase_info=[]
        reach=.175 if self.finger_mode=='curled' else .200
        z=xyz[2]+reach*np.sin(self.tilt)+palm_offset
        x=xyz[0]-reach*np.cos(self.tilt)
        specs=[('ready',.15,1.08,2.4,False),('preshape',.15,1.08,1.6,True),
               ('lower',.14,z,2.,True),('approach',.11 if self.finger_mode=='curled' else .09,z,1.6,True),
               ('squeeze',width,z,2.,True),('contact',width,z,.4,True),
               ('lift',width,z+.12,3.,True),('hold',width,z+.12,4.,True)]
        begin=np.array([q[n] for n in self.cfg['joint_names']])
        for phase,w,zz,seconds,closed in specs:
            # Carry the cube slightly forward while lifting, keeping the wrist
            # housings clear of the torso at the near edge of the reset region.
            xx=x+(.035 if phase in ('lift','hold') else 0.)
            destinations={s:np.array([xx,xyz[1]+sgn*w,zz-self.cfg['robot_z']]) for s,sgn in [('left',1),('right',-1)]}
            dest_rots={s:self.rotation(s) for s in poses}
            slerps={s:Slerp([0,1],Rotation.from_matrix([rots[s],dest_rots[s]])) for s in poses}
            start_fingers={n:q[n] for n in self.fingers}
            segments=12 if phase not in ('contact','hold') else 1
            for j in range(1,segments+1):
                u=j/segments;v=u*u*(3-2*u)
                if closed:
                    for n,target in self.fingers.items():q[n]=(1-v)*start_fingers[n]+v*target
                for s in poses:
                    initial_wrist=self.cfg['home']['Joint7_'+('L' if s=='left' else 'R')]
                    wrist=(1-v)*initial_wrist+v*self.wrist if phase=='ready' else self.wrist
                    q=self.fit(q,s,(1-v)*poses[s]+v*destinations[s],slerps[s]([v]).as_matrix()[0],wrist)
                goal=np.array([q[n] for n in self.cfg['joint_names']],dtype=np.float32)
                steps=max(1,round(seconds*self.cfg['control_hz']/segments))
                self.segments.append((phase,begin.copy(),goal.copy(),steps));begin=goal
            poses=destinations;rots=dest_rots
        return self.segments


from side_control import SegmentController

class SideTeacher(SegmentController):
    def __init__(self,env,width=.096,palm_offset=0.,wrist_deg=35.,tilt_deg=15.,finger_mode='curled'):
        self.planner=SidePlanner(wrist_deg,tilt_deg,finger_mode)
        xyz=tensor(env.cube.data.root_pos_w)[0].numpy().copy()
        super().__init__(env,self.planner.plan(xyz,width,palm_offset))
