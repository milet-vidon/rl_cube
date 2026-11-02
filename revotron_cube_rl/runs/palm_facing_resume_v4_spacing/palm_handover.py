"""Palm-facing, noncrossing handover after the unchanged certified left pickup."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tilt_regrasp import TiltPlanner, TiltTeacher, MirrorArrival, set_right_arm_gains


class PalmFacingPlanner(TiltPlanner):
    tilt_degrees = -90.
    grasp_symmetry = Rotation.from_euler('x',160.,degrees=True).as_matrix()
    right_palm_offset_cube = np.array([.060,.005,-.005])
    right_contact_axis = 0
    right_thumb_sign = -1
    pregrasp_offset_cube = np.array([0.,0.,-.10])
    little_contact_z = -.025
    keep_right_side = True
    refine_contact_geometry = True

    def refine_right_hand(self,arm,closed,cube,measured,points):
        """Fit each pad to its own face region, with clearance for the wrists."""
        mirror=np.diag([1.,-1.,1.]);anchors={};opened={}
        for finger,anchor in self.grasp['contact_anchors'].items():
            contacts=[p for p in points if p['side']=='left' and p['finger']==finger]
            if not contacts:raise ValueError('Missing measured left '+finger+' contact')
            contact=max(contacts,key=lambda p:p['impulse_Ns'])
            world=np.array(contact['position_world_m'])-[0,0,self.cfg['robot_z']]
            left=self.k.fk_link(measured,anchor['link'])
            local=mirror@left[:3,:3].T@(world-left[:3,3])
            goal=self.grasp_symmetry@mirror@cube[:3,:3].T@(world-cube[:3,3])
            sign=-1 if finger=='thumb' else 1
            goal[0]=sign*.027;goal[1:]=np.clip(goal[1:],-.022,.022)
            if finger=='little':goal[2]=self.little_contact_z
            link=anchor['link'].replace('left_','right_')
            names=[n for n in closed if n.startswith('right_'+finger+'_')]
            ref=np.array([closed[n] for n in names]);limits=np.array([self.k.limits[n] for n in names])
            lo=limits[:,0]+.02;hi=limits[:,1]-.02
            def position(v):
                frame=self.k.fk_link(arm|closed|dict(zip(names,v)),link)
                return cube[:3,:3].T@(frame[:3,3]+frame[:3,:3]@local-cube[:3,3])
            fit=least_squares(lambda v:np.r_[np.array([120.,80.,40.])*(position(v)-goal),.025*(v-ref)],np.clip(ref,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=150)
            actual=position(fit.x)
            if abs(actual[0]-goal[0])>.002 or np.max(np.abs(actual[1:]))>.0275:
                raise ValueError(f'Right {finger} cannot reach an interior face: {actual}')
            closed.update(zip(names,fit.x))
            # Open away from the face, so insertion cannot scrape the held cube.
            open_goal=actual.copy();open_goal[0]+=sign*.018
            ref=fit.x.copy()
            fit=least_squares(lambda v:np.r_[np.array([200.,10.,10.])*(position(v)-open_goal),.025*(v-ref)],np.clip(ref,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=150)
            if sign*position(fit.x)[0]<.038:raise ValueError(f'Insufficient pregrasp finger clearance: {finger} {position(fit.x)}')
            opened.update(zip(names,fit.x))
            anchors[finger]={'link':link,'point_local':local.tolist(),'target_cube':goal.tolist()}
        return closed,opened,anchors


class PalmFacingTeacher(TiltTeacher):
    planner_type=PalmFacingPlanner
