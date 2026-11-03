"""Opposed mirrored grasp with corrected wrist mesh geometry and finger checks."""
import json,os,subprocess,tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from elbow_handover import ElbowOutPlanner,ElbowOutTeacher
from opposition_teacher import timed_path

class SafeHandoverPlanner(ElbowOutPlanner):
    grasp_symmetry=Rotation.from_euler('y',-36.29058,degrees=True).as_matrix()@Rotation.from_euler('z',90.,degrees=True).as_matrix()@Rotation.from_euler('x',180.,degrees=True).as_matrix()
    right_palm_offset_cube=np.array([-.010672381884205133,.0007206692882940169,-.007796934040110594])
    right_contact_axis=1
    right_thumb_sign=-1
    pregrasp_offset_cube=np.array([0.,0.,.10])
    little_contact_z=None
    keep_right_side=True
    refine_contact_geometry=True


    def tilt_path(self,q,cube_position,shift=(0.,0.,0.),degrees=None):
        path=super().tilt_path(q,cube_position,shift,degrees)
        names=self.cfg['joint_names'];q=dict(zip(names,path[-1]));start=self.k.fk_link(q,'left_palm_link')
        points=[path[-1]]
        for u in np.linspace(0,1,41)[1:]:
            q=self.fit_tilt(q,'left',start[:3,3]+u*np.array([-.18,-.04,-.04]),start[:3,:3])
            points.append(np.array([q[n]for n in names]))
        limits=np.array([self.k.limits[n]for n in names])
        velocity=np.array([.25 if n.startswith('Joint') else 1. for n in names]);acceleration=2*velocity
        return np.concatenate([path,timed_path(points,5.,self.cfg['control_hz'],velocity,acceleration,limits)])

    def refine_right_hand(self,arm,closed,cube,measured,points):
        config=json.loads(Path(__file__).with_name('safe_contact_candidate_v2.json').read_text())
        anchors={};opened={};mirror=np.diag([1.,-1.,1.])
        for finger,a in self.grasp['contact_anchors'].items():
            contacts=[p for p in points if p['side']=='left' and p['finger']==finger]
            if not contacts:raise ValueError('Missing left contact for '+finger)
            world=np.array(max(contacts,key=lambda p:p['impulse_Ns'])['position_world_m'])-[0,0,self.cfg['robot_z']]
            frame=self.k.fk_link(measured,a['link']);local=mirror@frame[:3,:3].T@(world-frame[:3,3])
            link=a['link'].replace('left_','right_');names=[n for n in closed if n.startswith('right_'+finger+'_')]
            ref=np.array([config['right_closed'][n]for n in names]);limits=np.array([self.k.limits[n]for n in names]);lo=limits[:,0]+.02;hi=limits[:,1]-.02
            for i,n in enumerate(names):
                if 'MPR' in n:lo[i]=max(lo[i],ref[i]-.02);hi[i]=min(hi[i],ref[i]+.02)
            def position(v):
                f=self.k.fk_link(arm|closed|dict(zip(names,v)),link)
                return cube[:3,:3].T@(f[:3,3]+f[:3,:3]@local-cube[:3,3])
            goal=np.array(config['finger_points_cube'][finger]);goal[1]=(-1 if finger=='thumb' else 1)*.027
            fit=least_squares(lambda v:np.r_[100*(position(v)-goal),.05*(v-ref)],np.clip(ref,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=100)
            if np.linalg.norm(position(fit.x)-goal)>.002:raise ValueError('Right pad target unreachable: '+finger)
            closed.update(zip(names,fit.x));anchors[finger]={'link':link,'point_local':local.tolist(),'target_cube':goal.tolist()}
            open_goal=goal.copy();open_goal[1]+=(-1 if finger=='thumb' else 1)*.018;ref=fit.x
            fit=least_squares(lambda v:np.r_[np.array([10.,200.,10.])*(position(v)-open_goal),.02*(v-ref)],np.clip(ref,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=120)
            if abs(position(fit.x)[1])<.038:raise ValueError('Insufficient open-pad clearance: '+finger)
            opened.update(zip(names,fit.x))
        return closed,opened,anchors

    def regrasp_path(self,q,measured,cube_position,cube_quaternion,points=None):
        """Mirror the actual held grasp onto a free pair of cube faces."""
        names=self.cfg['joint_names'];m=np.diag([1.,-1.,1.,1.])
        Q=self.grasp_symmetry.copy()
        cube=np.eye(4);cube[:3,:3]=Rotation.from_quat(np.asarray(cube_quaternion)[[1,2,3,0]]).as_matrix()
        cube[:3,3]=np.asarray(cube_position)-[0,0,self.cfg['robot_z']]
        lp=np.linalg.inv(cube)@self.k.fk_link(measured,'left_palm_link')
        symmetry=np.eye(4);symmetry[:3,:3]=Q
        local=symmetry@m@lp@m;local[:3,3]+=self.right_palm_offset_cube;target=cube@local
        closed={n.replace('left_','right_'):v for n,v in measured.items() if n.startswith('left_')}
        opened={n:closed[n]+self.primary_open[n.replace('right_','left_')]-self.primary_closed[n.replace('right_','left_')] for n in closed}
        # The inherited little-finger preshape grazed the cube during lateral
        # insertion. Keep it extended until the palm reaches the grasp pose.
        opened.update(right_little_MCP_joint=.3,right_little_PIP_joint=0.,right_little_DIP_joint=0.)
        seed=q|dict(zip([f'Joint{i}_R' for i in range(1,8)],np.radians([-18.53,-110.2,30.75,-58.57,150.72,44.78,80.51])))
        final=self.fit_tilt(seed,'right',target[:3,3],target[:3,:3])
        contact_anchors=None
        if self.refine_contact_geometry:
            closed,opened,contact_anchors=self.refine_right_hand(final,closed,cube,measured,points or [])
        else:
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
            completed=subprocess.run(['/usr/bin/python3',str(Path(__file__).with_name('safe_collision_plan.py')),str(source),str(destination)],env=environment,capture_output=True,text=True,timeout=120)
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
        validation=self.validate_full_path(measured,cube_position,cube_quaternion,phases)
        return phases,{'path_validation':validation,'Q':Q.tolist(),'right_contact_anchors':contact_anchors,'right_palm_offset_cube_m':self.right_palm_offset_cube.tolist(),'right_palm_in_cube':local.tolist(),'right_closed':closed,'collision_route':route,'thumb_open_face_normal_offset_m':.020,'pregrasp_offset_cube_m':self.pregrasp_offset_cube.tolist(),'pregrasp_offset_world_m':pre_offset.tolist(),'right_contact_axis':self.right_contact_axis,'right_thumb_sign':self.right_thumb_sign,'little_contact_z':self.little_contact_z}

    def validate_full_path(self,initial,position,quaternion,phases):
        request={'initial':initial,'cube_position':np.asarray(position).tolist(),'cube_quaternion':np.asarray(quaternion).tolist(),
                 'joint_names':self.cfg['joint_names'],'phases':[(n,p.tolist())for n,p in phases]}
        with tempfile.TemporaryDirectory(prefix='cube_path_audit_')as directory:
            src=Path(directory)/'request.json';dst=Path(directory)/'audit.json';src.write_text(json.dumps(request))
            env=os.environ.copy()
            for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD'):env.pop(key,None)
            env['PYTHONPATH']='/opt/ros/jazzy/lib/python3.12/site-packages';env['LD_LIBRARY_PATH']='/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu'
            result=subprocess.run(['/usr/bin/python3',str(Path(__file__).with_name('safe_path_audit.py')),str(src),str(dst)],env=env,capture_output=True,text=True,timeout=180)
            report=json.loads(dst.read_text()) if dst.exists() else {'error':result.stderr[-1500:]}
            self.last_path_validation=report
            if result.returncode:raise ValueError('Full regrasp geometry rejected: '+json.dumps(report))
            return report


class SafeHandoverTeacher(ElbowOutTeacher):
    planner_type=SafeHandoverPlanner
