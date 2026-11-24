"""Isolated, guided right index-pad trial for a real nZ layer turn on a 90 mm cube.

This module never steps the simulation, actuates the cube, or changes the left
student. GuidedFaceTrial.action is called once per physical control frame. A
physical hold only unlocks planning: planning, approach, push commands and
measured layer rotation are different events. All original stop gates remain.
"""
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
EDGE, TARGET = .090, 15.
SOURCES = {'face_push_plan.py':'0bf9b55361f59113264ce29260298fa0d9fb883cda16fac66481de1ab82f89eb',
           'safe_collision_plan.py':'2ca8f8278d50b03a612ec9e9bbad746900541b86a29eb799042212c4fdd6dda4'}
PHASES = ('right_index_preshape','right_index_route','right_index_approach','right_face_push','right_index_retreat','face_hold')


def adapted_source():
    """Keep the pinned planner's IK, joint bounds, margins and full swept checks.

    Change ONLY layer dimensions/centers, and the pad target locations on the
    cube. Distances measured on the robot (pad-local offsets etc.) do not scale.
    Keep the penetration at the nZ face at its original 1.5 mm, not 2.37 mm.
    """
    for name,expected in SOURCES.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Changed reference swept planner: '+name)
    src=(ROOT/'face_push_plan.py').read_text()
    replacements=[
        ('(.057,.057,.038)','(.09,.09,.06)'), ('[0,0,.0095]','[0,0,.015]'),
        ('(.057,.057,.019)','(.09,.09,.03)'), ('[0,0,-.019]','[0,0,-.030]'),
        ('    for corner in corners:', '''    if request['contact_style'] != 'face' or request['edge_m'] != .09 or request['target_deg'] != 15.:
        raise ValueError('Only the 90 mm, nZ, 15-degree face-pad trial is supported')
    corners=[[x*.09/.057, y*.09/.057, -.045+.0015] for x,y,z in corners]
    for corner in corners:'''),
    ]
    for old,new in replacements:
        if src.count(old)!=1:raise ValueError('Pinned planner transformation no longer unique')
        src=src.replace(old,new)
    return src


def host_plan(request, dest):
    ns={'__name__':'size90_checked_planner','__file__':str(ROOT/'face_push_plan.py')}
    exec(compile(adapted_source(),str(ROOT/'face_push_plan.py'),'exec'),ns)
    try:result=ns['plan'](request)
    except ValueError as exc:result={'passed':False,'failure':str(exc),'physics_validated':False}
    result.update(edge_m=EDGE,planner_sources=SOURCES,planner_adaptation_sha256=hashlib.sha256(adapted_source().encode()).hexdigest())
    with Path(dest).open('x') as f:json.dump(result,f,allow_nan=False)
    return result


def plan_for_row(run, names, row, cfg):
    if not (cfg['robot_z']==1.15 and math.isclose(cfg['table_z']-cfg['robot_z'],-.25,abs_tol=1e-7)):
        raise ValueError('Planner table/root frames differ from measured world')
    payload=dict(initial=dict(zip(names,row['q'],strict=True)),joint_names=names,cube_position=row['cube_position'],
                 cube_quaternion=row['cube_quaternion_wxyz'],face='nZ',target_deg=TARGET,edge_m=EDGE,
                 contact_style='face',input_step=row['step'])
    request=run/'guided_face_request.json'; plan=run/'guided_face_plan.json'
    with request.open('x') as f:json.dump(payload,f,allow_nan=False)
    host=os.environ.copy()
    host.update(PYTHONPATH='/opt/ros/jazzy/lib/python3.12/site-packages',
                LD_LIBRARY_PATH='/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu')
    with (run/'guided_face_preflight.log').open('x') as f:
        try:
            p=subprocess.run(['/usr/bin/python3',str(Path(__file__).resolve()),'--host-plan',str(request),str(plan)],
                             env=host,stdout=f,stderr=subprocess.STDOUT,timeout=190,check=False)
        except subprocess.TimeoutExpired:
            return {'passed':False,'failure':'right_contact_planning_timeout'}
    if not plan.is_file():return {'passed':False,'failure':f'planner_process_{p.returncode}_no_plan'}
    result=json.loads(plan.read_bytes())
    if p.returncode or not result['passed']:return dict(passed=False,failure=result.get('failure') or 'swept_path_rejected')
    return result


def validated_actions(plan):
    if (not plan['passed'] or plan['target_deg']!=TARGET or plan['edge_m']!=EDGE or plan['face']!='nZ' or
            plan['contact_pad']!='right_index_DIP_rubber_link' or plan['contact_style']!='face' or
            plan['checks']<=0 or tuple(p for p,_ in plan['phases'])!=PHASES):
        raise ValueError('Wrong right contact plan')
    out=[]
    for phase,path in plan['phases']:
        path=np.asarray(path,dtype=float)
        if path.ndim!=2 or not len(path) or path.shape[1]!=56 or not np.isfinite(path).all():
            raise ValueError('Invalid checked right arm trajectory')
        out.extend((phase,target) for target in path)
    return out


def guard(plan_anchor,row,points,previous_phase):
    """Stop a fixed-snapshot plan when its support, collision scene or contact changes."""
    q0=np.asarray(plan_anchor['cube_quaternion_wxyz']);q=np.asarray(row['cube_quaternion_wxyz'])
    ang=np.degrees(2*np.arccos(np.clip(abs(q0@q)/(np.linalg.norm(q0)*np.linalg.norm(q)),0,1)))
    if np.linalg.norm(np.asarray(row['cube_position'])-plan_anchor['cube_position'])>.002 or ang>2:
        return 'cube_moved_outside_contact_plan'
    if points and previous_phase in ('right_index_preshape','right_index_route'):return 'premature_right_contact'
    if any('right_index_DIP' not in p['robot_body'] or 'nZ' not in p['cubelet'].split('_') for p in points):
        return 'unplanned_right_contact_body_or_layer'
    if row['contact_impulse'][1]>1e-5 and not points:return 'unmapped_right_cube_contact'
    return None


class GuidedFaceTrial:
    def __init__(self, run, edge_m=EDGE):
        if edge_m!=EDGE:raise ValueError('Fixed-size 90 mm trial only')
        self.run=Path(run);self.window=deque(maxlen=30);self.status='waiting_for_hold'
        self.reason='no_continuous_1s_left_hold';self.anchor=None;self.path=[];self.cursor=0
        self.last_step=-1;self.last_phase=None;self.sensor=None;self.issued=0
        self.palm_history=[]

    def action(self,env,previous,student_action,step):
        """Return (56-channel action, post-step ledger fields), preserving all 28 left actions."""
        from contact_hold_simple_rotation import build_evidence
        if step!=self.last_step+1 or ((previous is None)!=(step==0)) or (previous is not None and previous['step']!=step-1):
            raise ValueError('Guided turn called out of sequence')
        if env.cfg['control_hz']!=30:raise ValueError('Expected 30 Hz control')
        self.last_step=step
        points=list(self.sensor.points) if self.sensor is not None else []
        if self.sensor is not None:self.sensor.points.clear()
        phase=None
        if previous is not None and self.status=='waiting_for_hold':
            # Fast reject only. Acceptance uses the exact offline window predicate.
            if previous['height_m']<.02 or previous['cube_table_contact'] or previous['contact_impulse'][0]<=env.cfg['success']['contact_impulse_min']:
                self.window.clear()
            else:
                self.window.append(previous)
            if len(self.window)==30:
                ev=build_evidence(list(self.window),env.cfg)
                if ev['start_by_end'][-1]==0:
                    self.anchor=previous
                    plan=plan_for_row(self.run,env.cfg['joint_names'],previous,env.cfg)
                    if plan['passed']:
                        from face_push_runtime import LayerContacts
                        self.path=validated_actions(plan);self.sensor=LayerContacts()
                        self.status='following_plan';self.reason=None
                        from contact_hold_physical_metrics import _xyz_in_palm
                        self.palm_history=[_xyz_in_palm(r,env.cfg,'left') for r in self.window]
                    else:self.status='preflight_rejected';self.reason=plan['failure']
        if self.status=='following_plan':
            from contact_hold_size_face_score import face_samples
            from contact_hold_physical_metrics import _xyz_in_palm
            ok=bool(face_samples([previous],'nZ')[1][0])
            ev=build_evidence([previous],env.cfg,assembly_check=lambda r:ok)
            reason=None
            if not ev['support'][0]:reason='measured_left_support_or_coherence_lost'
            pt=_xyz_in_palm(previous,env.cfg,'left')
            if not reason and max(math.dist(pt,p) for p in self.palm_history)>.010000000001:
                reason='palm_slip_outside_controlled_hold'
            left=[i for i,n in enumerate(env.cfg['joint_names']) if n.endswith('_L') or n.startswith('left_')]
            if len(left)!=28:raise ValueError('Unexpected joint names')
            if np.max(np.abs(np.asarray(previous['q'])[left]-np.asarray(self.anchor['q'])[left]))>.025:
                reason=reason or 'left_robot_moved_outside_swept_scene'
            reason=reason or guard(self.anchor,previous,points,self.last_phase)
            if reason:self.status='aborted';self.reason=reason
            else:self.palm_history.append(pt)
        requested=None; right_delta=0.
        if self.status=='following_plan':
            if self.cursor==len(self.path):
                self.status='plan_completed';self.reason=None
            else:
                import torch
                phase,target=self.path[self.cursor];self.cursor+=1
                right=[i for i,n in enumerate(env.cfg['joint_names']) if n.endswith('_R') or n.startswith('right_')]
                if len(right)!=28:raise ValueError('Unexpected right joint names')
                before=env.targets[:,right]
                goal=before.new_tensor(target[right]).reshape_as(before)
                if torch.max(torch.abs(goal-before)).item()>.15:
                    self.status='aborted';self.reason='right_robot_target_outside_plan';phase=None
                else:
                    student_action=student_action.clone()
                    student_action[:,right]=(goal-before)/env.scales[:,right]
                    right_delta=float(torch.max(torch.abs(goal-before)).item())
                    requested=goal[0].tolist()
                    if phase=='right_face_push':self.issued+=1
        if self.status in ('aborted','plan_completed') and self.sensor is not None:
            # A rejected/outdated plan has no validated return route. Retain
            # right targets in place; never jump back to the original home path.
            right=[i for i,n in enumerate(env.cfg['joint_names']) if n.endswith('_R') or n.startswith('right_')]
            student_action=student_action.clone()
            student_action[:,right]=0.
        self.last_phase=phase
        return student_action,dict(guided_face_state=self.status,guided_face_reason=self.reason,
                    guided_face_phase_commanded=phase,guided_face_supported_at_post_step=self.anchor['step'] if self.anchor else None,
                    guided_face_requested_right_max_delta_rad=right_delta,
                    guided_face_push_command_moved_target=bool(phase=='right_face_push' and right_delta>1e-7),
                    guided_face_requested_right_target=requested,guided_face_target_deg=TARGET,
                    guided_face_push_commanded=(phase=='right_face_push'),guided_face_total_push_commands=self.issued,
                    guided_face_previous_contact_step=(previous['step'] if points else None),guided_face_previous_contacts=points)



def summarize_commands(rows):
    """Read post-step command fields, not cube angles. This never runs a solver.

    The verified 56-joint contract is [left arm 7, left fingers 21, right
    arm 7, right fingers 21]. An *actual* movement frame means the robot's
    projected/applied target changed, not that the cube necessarily moved.
    """
    if not rows or any(r['step']!=rows[0]['step']+i for i,r in enumerate(rows)):
        raise ValueError('Missing step in commanded trajectory')
    if any('guided_face_state' not in r for r in rows):
        raise ValueError('Missing guided-stage instrumentation')
    left_sources={r['executed_left_source'] for r in rows}
    if not left_sources.issubset({'student','student_plus_size_plan_delta'}):
        raise ValueError('Unexpected left controller in turn audit')
    targets=np.asarray([r['applied_target'] for r in rows])
    if targets.shape!=(len(rows),56) or not np.isfinite(targets).all():
        raise ValueError('Invalid applied targets')
    deltas=np.abs(np.diff(targets,axis=0,prepend=targets[:1]))
    arm=deltas[:,:7].max(axis=1);right=deltas[:,28:].max(axis=1)
    # Do not infer deliberate whole turns during approach, right clearing,
    # regrasp, elbow transport, or a static tilt_hold. Report wrist adjustment
    # separately in the phase set; neither phase proves physical rotation.
    tilt=np.array([r['phase'] in ('tilt_90','wrist_reconfigure') for r in rows])
    push=np.array([r['guided_face_push_commanded'] for r in rows],dtype=bool)
    if not all((r['guided_face_phase_commanded']=='right_face_push')==p for p,r in zip(push,rows)):
        raise ValueError('Inconsistent push action instrumentation')
    if int(push.sum())!=rows[-1]['guided_face_total_push_commands']:
        raise ValueError('Incomplete guided action ledger')
    issued=push & (right>1e-7); tilt_moved=tilt & (arm>1e-7)
    def first(mask):
        ids=np.flatnonzero(mask)
        return rows[int(ids[0])]['step'] if ids.size else None
    preflight=next((r['guided_face_supported_at_post_step'] for r in rows
                    if r['guided_face_supported_at_post_step'] is not None),None)
    return dict(actual_guided_face_push_frames=int(issued.sum()),guided_face_push_phase_frames=int(push.sum()),
                preflight_requested=preflight is not None, hold_confirmed_post_step=preflight,
                first_applied_guided_face_push_step=first(issued),last_reason=rows[-1]['guided_face_reason'],
                last_face_state=rows[-1]['guided_face_state'],
                guided_face_target_deg=TARGET, whole_planner_phase_requested=bool(tilt.any()),
                whole_planner_phase_frames=int(tilt.sum()),whole_planner_phases=sorted({r['phase'] for r,b in zip(rows,tilt) if b}),
                actual_student_left_arm_changing_in_tilt_frames=int(tilt_moved.sum()),
                first_applied_whole_tilt_step=first(tilt_moved),
                max_applied_tilt_left_arm_delta_rad=float(arm[tilt].max()) if tilt.any() else 0.,
                executed_left_sources=sorted(left_sources),
                physical_rotation_not_inferred_from_command=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-plan',nargs=2,required=True,type=Path)
    a=parser.parse_args()
    result=host_plan(json.loads(a.host_plan[0].read_bytes()),a.host_plan[1])
    raise SystemExit(0 if result['passed'] else 2)
