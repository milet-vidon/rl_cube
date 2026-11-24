"""Palm-facing, noncrossing handover after the unchanged certified left pickup."""
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tilt_regrasp import TiltPlanner, TiltTeacher, MirrorArrival, set_right_arm_gains


class PalmFacingPlanner(TiltPlanner):
    tilt_degrees = -90.
    grasp_symmetry = Rotation.from_euler('x',150.,degrees=True).as_matrix()
    right_palm_offset_cube = np.array([.060,.015,-.025])
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
            for i,n in enumerate(names):
                if 'MPR' in n:lo[i]=max(lo[i],ref[i]-.02);hi[i]=min(hi[i],ref[i]+.02)
            def position(v):
                frame=self.k.fk_link(arm|closed|dict(zip(names,v)),link)
                return cube[:3,:3].T@(frame[:3,3]+frame[:3,:3]@local-cube[:3,3])
            def residual(v):
                pos=position(v)
                low=[-.025,-.028] if finger=='little' else [-.025,-.025]
                high=[.025,-.023] if finger=='little' else [.025,.025]
                outside=pos[1:]-np.clip(pos[1:],low,high)
                return np.r_[120*(pos[0]-goal[0]),100*outside,3*(pos[1:]-goal[1:]),.025*(v-ref)]
            fit=least_squares(residual,np.clip(ref,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=150)
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
            anchors[finger]={'link':link,'point_local':local.tolist(),'target_cube':(np.array([sign*.027,actual[1],actual[2]])).tolist()}
        return closed,opened,anchors


class PalmFacingTeacher(TiltTeacher):
    planner_type=PalmFacingPlanner

    def action(self):
        if self.mode=='tilt' and self.tick>=len(self.path):
            if not getattr(self,'hold_gains_set',False):
                names=[n for n in self.env.cfg['joint_names'] if n.startswith('left_')]
                ids=[self.env.ids[self.env.cfg['joint_names'].index(n)] for n in names]
                self.env.robot.write_joint_stiffness_to_sim_index(stiffness=torch.full((1,len(ids)),20.,dtype=torch.float32),joint_ids=ids)
                self.hold_gains_set=True
            if (self.tick-len(self.path))%10==0:
                # Correct actual object slip after the turn, without resetting
                # the object or bypassing the joint-target safety layer.
                from isaac_compat import tensor,quaternion_from_native
                state=tensor(self.env.cube.data.root_state_w)[0]
                actual=Rotation.from_quat(quaternion_from_native(state[3:7]).numpy()[[1,2,3,0]])
                desired=Rotation.from_euler('x',-90.,degrees=True)*Rotation.from_quat(self.reference[[1,2,3,0]])
                error=(desired*actual.inv()).as_rotvec();angle=np.linalg.norm(error)
                if angle>np.radians(1.):
                    delta=Rotation.from_rotvec(error*min(.5,np.radians(1.)/angle)).as_matrix()
                    commanded=dict(zip(self.env.cfg['joint_names'],self.path[-1].tolist()))
                    palm=self.planner.k.fk_link(commanded,'left_palm_link')
                    if not hasattr(self,'hold_center'):
                        self.hold_center=state[:3].numpy().copy()-[0,0,self.env.cfg['robot_z']]
                        self.hold_reference_palm=palm.copy()
                    center=self.hold_center
                    position=center+delta@(palm[:3,3]-center);rotation=delta@palm[:3,:3]
                    # Accumulate corrections on commanded pose, not measured
                    # servo sag: otherwise each correction integrates gravity
                    # deflection and the hand drifts toward the table.
                    correction_angle=Rotation.from_matrix(rotation@self.hold_reference_palm[:3,:3].T).magnitude()
                    if np.linalg.norm(position-self.hold_reference_palm[:3,3])>.04 or correction_angle>np.radians(15.):
                        raise ValueError('Post-turn correction exceeded bounded pose envelope')
                    fitted=self.planner.fit_tilt(commanded,'left',position,rotation)
                    for n in (f'Joint{i}_L' for i in range(1,8)):
                        self.path[-1,self.env.cfg['joint_names'].index(n)]=fitted[n]
        action=super().action()
        if self.mode=='tilt' and (self.env.metrics.get('linear_speed',0.)>=.03 or self.env.metrics.get('angular_speed',0.)>=.20):
            self.hold=0;self.accepted=False
        return action
