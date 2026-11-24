"""Outward upper-arm configuration for the palm-facing handover.

Retain the certified table pickup. Once clear of the table, move the held
object into an ipsilateral handover workspace, then rotate it by -90 degrees.
The elbow target is expressed relative to the actual shoulder frame, not the
other elbow. All waypoints still go through the normal joint safety/PD layer.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from palm_handover import PalmFacingPlanner, PalmFacingTeacher
from opposition_teacher import timed_path
from proximal_planner import PROXIMAL_WEIGHTS


class ElbowOutPlanner(PalmFacingPlanner):
    elbow_shift = np.array([-.045,.20,.055])
    elbow_target = .18

    def fit_tilt(self,q,side,position,rotation):
        if side != 'left':
            return super().fit_tilt(q,side,position,rotation)
        names=[f'Joint{i}_L' for i in range(1,8)]
        limits=np.array([self.k.limits[n] for n in names]);margin=np.radians(2.)
        initial=np.clip([q[n] for n in names],limits[:,0]+margin,limits[:,1]-margin)
        goal=getattr(self,'elbow_goal',self.elbow_target)
        def residual(x):
            candidate=q|dict(zip(names,x));f=self.k.fk(candidate)
            t=f['left_palm_link'];arm=f['Link4_L'][:3,3]-f['Link2_L'][:3,3]
            return np.r_[100*(t[:3,3]-position),12*Rotation.from_matrix(rotation@t[:3,:3].T).as_rotvec(),
                         35*max(goal-arm[1],0.),.008*np.sqrt(PROXIMAL_WEIGHTS)*(x-initial),
                         .015*x[5]]
        fit=least_squares(residual,initial,bounds=(limits[:,0]+margin,limits[:,1]-margin),
                          max_nfev=180,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        result=q|dict(zip(names,fit.x));f=self.k.fk(result);t=f['left_palm_link']
        error=np.linalg.norm(t[:3,3]-position)
        angle=Rotation.from_matrix(rotation@t[:3,:3].T).magnitude()
        if error>.002 or angle>np.radians(1.):
            raise ValueError(f'Outward elbow IK infeasible: position {error:.6f} m, angle {np.degrees(angle):.3f} deg, target {goal:.3f} m')
        return result

    def tilt_path(self,q,cube_position,shift=(0.,0.,0.),degrees=None):
        if np.linalg.norm(shift)>1e-9:
            raise ValueError('Use the frozen elbow workspace shift, not an extra ad-hoc shift')
        names=self.cfg['joint_names'];hz=self.cfg['control_hz']
        limits=np.array([self.k.limits[n] for n in names])
        velocity=np.array([.35 if n.startswith('Joint') else 1.5 for n in names])
        acceleration=np.array([.7 if n.startswith('Joint') else 6. for n in names])
        f=self.k.fk(q);start=f['left_palm_link'].copy()
        initial_out=(f['Link4_L'][:3,3]-f['Link2_L'][:3,3])[1]
        points=[np.array([q[n] for n in names])]
        for u in np.linspace(0,1,31)[1:]:
            self.elbow_goal=(1-u)*initial_out+u*self.elbow_target
            q=self.fit_tilt(q,'left',start[:3,3]+u*self.elbow_shift,start[:3,:3])
            points.append(np.array([q[n] for n in names]))
        outward=timed_path(points,4.,hz,velocity,acceleration,limits)
        center=np.asarray(cube_position)-[0,0,self.cfg['robot_z']]+self.elbow_shift
        start=self.k.fk_link(q,'left_palm_link');points=[np.array([q[n] for n in names])]
        self.elbow_goal=self.elbow_target
        for u in np.linspace(0,1,46)[1:]:
            rotation=Rotation.from_euler('x',self.tilt_degrees*u,degrees=True).as_matrix()
            q=self.fit_tilt(q,'left',center+rotation@(start[:3,3]-center),rotation@start[:3,:3])
            points.append(np.array([q[n] for n in names]))
        turn=timed_path(points,6.,hz,velocity,acceleration,limits)
        self.clear_steps=0;self.outward_steps=len(outward);self.tilt_steps=len(outward)+len(turn)
        return np.concatenate([outward,turn])


class ElbowOutTeacher(PalmFacingTeacher):
    planner_type=ElbowOutPlanner

    def action(self):
        result=super().action()
        if self.mode=='tilt' and self.tick<=getattr(self.planner,'outward_steps',0):
            self.name='elbow_outward'
        return result
