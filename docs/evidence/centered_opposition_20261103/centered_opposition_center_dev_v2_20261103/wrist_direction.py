"""Reverse the J6 wrist side bend while retaining palm-facing and elbow-out.

This is a new candidate, not an extension of the old elbow certificate. The
accepted table grasp and -90 degree turn are replayed by their original
controller; a bounded Cartesian adjustment then changes the arm redundancy.
All trajectories still pass through joint_safety and PhysX target PD.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from elbow_handover import ElbowOutPlanner,ElbowOutTeacher
from opposition_teacher import timed_path


class WristDirectionPlanner(ElbowOutPlanner):
    wrist_goal=None
    final_wrist_goal=np.radians(12.)

    def fit_tilt(self,q,side,position,rotation):
        if side!='left' or self.wrist_goal is None:
            return super().fit_tilt(q,side,position,rotation)
        names=[f'Joint{i}_L'for i in range(1,8)];limits=np.array([self.k.limits[n]for n in names]);margin=np.radians(2.)
        lo,hi=limits[:,0]+margin,limits[:,1]-margin;ref=np.clip([q[n]for n in names],lo+1e-7,hi-1e-7)
        def residual(x):
            f=self.k.fk(q|dict(zip(names,x)));p=f['left_palm_link'];s,e,w=[f[n][:3,3]for n in ['Link2_L','Link4_L','Link7_L']]
            return np.r_[100*(p[:3,3]-position),12*Rotation.from_matrix(rotation@p[:3,:3].T).as_rotvec(),
                         35*max(.18-(e-s)[1],0),35*max(.05-(e-w)[1],0),5*(x[5]-self.wrist_goal),.005*(x-ref)]
        fit=least_squares(residual,ref,bounds=(lo,hi),max_nfev=180,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        result=q|dict(zip(names,fit.x));p=self.k.fk_link(result,'left_palm_link')
        if np.linalg.norm(p[:3,3]-position)>.002 or Rotation.from_matrix(rotation@p[:3,:3].T).magnitude()>np.radians(1):
            raise ValueError('Wrist-side reversal IK violates palm pose envelope')
        return result

    def wrist_workspace(self,q):
        names=[f'Joint{i}_L'for i in range(1,8)];limits=np.array([self.k.limits[n]for n in names]);margin=np.radians(2.)
        lo,hi=limits[:,0]+margin,limits[:,1]-margin;ref=np.clip([q[n]for n in names],lo+1e-7,hi-1e-7);start=self.k.fk_link(q,'left_palm_link')
        def residual(x):
            f=self.k.fk(q|dict(zip(names,x)));p=f['left_palm_link'];s,e,w=[f[n][:3,3]for n in ['Link2_L','Link4_L','Link7_L']]
            outside=p[:3,3]-np.clip(p[:3,3],[.20,.05,-.08],[.55,.39,.30])
            return np.r_[2*(p[:3,3]-start[:3,3]),40*Rotation.from_matrix(start[:3,:3]@p[:3,:3].T).as_rotvec(),
                         50*max(.18-(e-s)[1],0),50*max(.05-(e-w)[1],0),15*(x[5]-self.final_wrist_goal),100*outside,.003*(x-ref)]
        fit=least_squares(residual,ref,bounds=(lo,hi),max_nfev=240,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        result=q|dict(zip(names,fit.x));f=self.k.fk(result);target=f['left_palm_link']
        displacement=target[:3,3]-start[:3,3]
        if np.linalg.norm(displacement)>.25 or f['Link4_L'][1,3]-f['Link2_L'][1,3]<.16:
            raise ValueError('Wrist reversal requires an excessive workspace displacement')
        return target[:3,3]

    def tilt_path(self,q,cube_position,shift=(0.,0.,0.),degrees=None):
        self.wrist_goal=None
        path=super().tilt_path(q,cube_position,shift,degrees)
        names=self.cfg['joint_names'];q=dict(zip(names,path[-1]));start=self.k.fk_link(q,'left_palm_link');end=self.wrist_workspace(q)
        start_wrist=q['Joint6_L'];points=[path[-1].copy()];self.wrist_adjust_start=len(path)
        for u in np.linspace(0,1,41)[1:]:
            self.wrist_goal=(1-u)*start_wrist+u*self.final_wrist_goal
            q=self.fit_tilt(q,'left',(1-u)*start[:3,3]+u*end,start[:3,:3]);points.append(np.array([q[n]for n in names]))
        self.wrist_goal=self.final_wrist_goal
        self.wrist_displacement=(end-start[:3,3]).tolist()
        limits=np.array([self.k.limits[n]for n in names]);velocity=np.array([.25 if n.startswith('Joint')else 1.5 for n in names]);acceleration=2*velocity
        return np.concatenate([path,timed_path(points,6.,self.cfg['control_hz'],velocity,acceleration,limits)])


class WristDirectionTeacher(ElbowOutTeacher):
    planner_type=WristDirectionPlanner
    def action(self):
        result=super().action()
        if self.mode=='tilt' and getattr(self.planner,'wrist_adjust_start',10**9)<self.tick<=len(self.path):
            self.name='wrist_reconfigure'
        return result
