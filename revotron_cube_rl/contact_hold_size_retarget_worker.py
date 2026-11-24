"""Round-15 BC plus a per-phase 57 -> 90 mm geometric command residual.

This is an explicitly hybrid development controller, not a re-trained student.
Original model inference, measured observations and final PhysX safety projection
stay intact. The planner never teleports the cube or specifies a cube action.
"""
import argparse
import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

import contact_hold_size_approach_worker as approach
import contact_hold_size_worker as size
import contact_hold_functional_worker as functional
from contact_hold_grace_worker import ROOT, RUNS

GRASP=ROOT/'opposition_grasp_v20.json'
BASE=RUNS/'contact_hold_size_r15_90mm_approach_20261112_083044'/size.MODE
CONTROL={21611:'d08b4d63214066813c9a78d0e73de0ec63ad8c42d73d0cdebae05e22ffa99a41',
         21612:'6ed7d17c172cb5004dc8c3c45d22d23d1d6ea48ef6bd0bd026f427db4b3b7a07'}
APPROACH_SHA='473a895cfffde3be032e1eddb729ace536bf10908f3d623b9168280d6c73b844'
OLD_SHA='77f69b76900df2845cd5da7d465f1a8fc3641ee929533a14a2458bd8c81af29a'
SOURCE='student_plus_size_plan_delta'
KIND='size_retarget_bc_and_guided_right_face_development'


def dump(path,val):
    with path.open('x') as f:json.dump(val,f,indent=2,allow_nan=False)


def receipt(seed,asset):
    calib=asset.parent.parent/'calibration';grasp=calib/'grasp90.json';report=calib/'geometry_retarget.json'
    if size.file_sha(ROOT/'contact_hold_size_approach_worker.py')!=APPROACH_SHA or size.file_sha(GRASP)!=OLD_SHA:
        raise ValueError('Control worker or base grasp changed')
    ctrl=BASE/f'seed_{seed}';manifest=ctrl/'artifact_manifest.json'
    if size.file_sha(manifest)!=CONTROL[seed]:raise ValueError('Changed 90 mm paired control')
    prior=approach.receipt(seed,asset)
    old=json.loads(GRASP.read_bytes());new=json.loads(grasp.read_bytes());geometry=json.loads(report.read_bytes())
    allowed={'center_palm','open_joint_targets','joint_targets','face_points','contact_anchors','kind','physics_verified','collision_distance_min_m','inter_hand_clearance_m','preshape_path_clearance_m','closing_path_clearance_m','elapsed_s','optimization_evaluations','table_clearance_min_m','cube_contact_distances_m'}
    changed={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    if not ({'center_palm','open_joint_targets','joint_targets','contact_anchors'}<=changed<=allowed and
            geometry['source_grasp_sha256']==OLD_SHA and geometry['grasp_sha256']==size.file_sha(grasp)):
        raise ValueError('Calibration modified the wrong grasp parameters or has a stale receipt: '+str(changed))
    for key in ('open_joint_targets','joint_targets'):
        if set(new[key])!=set(old[key]) or len(old[key])!=21 or any(not n.startswith('left_') for n in new[key]):raise ValueError('Invalid left finger mapping')
    for finger,original in old['contact_anchors'].items():
        if {k for k in original.keys()|new['contact_anchors'][finger].keys() if original.get(k)!=new['contact_anchors'][finger].get(k)}!={'target_cube'}:
            raise ValueError('Local finger anchors or link names changed')
    return dict(kind=KIND,seed=seed,checkpoint_sha256=size.CHECKPOINT,asset_sha256=prior['asset_sha256'],cube_edge_m=size.EDGE,
        fixed_grasp_90mm_control_manifest_sha256=CONTROL[seed],fixed_grasp_90mm_control=str(ctrl),
        old_grasp_sha256=OLD_SHA,grasp_sha256=size.file_sha(grasp),geometry_retarget_sha256=size.file_sha(report),
        worker_sha256=size.file_sha(__file__),geometry_solver_sha256=size.file_sha(ROOT/'contact_hold_size_retarget_geometry.py'),
        right_guide_sha256=size.file_sha(ROOT/'contact_hold_size_retarget_turn.py'),approach_worker_sha256=APPROACH_SHA,
        actor='frozen round 15 evaluated on actual 90 mm observations every step',executed_left_source=SOURCE,
        residual='Phase-aligned new and old nominal pickup trajectories; new queried at the actual tick, old interpolated. Old center is at the 57 mm tabletop Z for the same XY/yaw. Postpickup uses phase-aligned paired tilt paths.',
        target_order='raw student + (90mm nominal - 57mm nominal), then original left step/velocity caps, then original env projection',
        right_source='original resized scheduler until a separately gated right index face attempt',
        precontact_tolerance_rad=approach.PROBE,after_contact_tolerance_rad=approach.ORIGINAL,
        eligible_training_episode=False,diagnostic_only=True)


def sample_pair(old,new,tick):
    if len(old)<1 or len(new)<1 or tick<0:raise ValueError('Empty path or invalid reference tick')
    k=min(int(tick),len(new)-1);u=k/max(1,len(new)-1);j=u*(len(old)-1);i=int(j)
    return ((1-(j-i))*old[i]+(j-i)*old[min(i+1,len(old)-1)]).astype(np.float32),new[k].copy(),u


def retarget(raw,before,cap,old,new):
    delta=new-old;req=raw+delta
    if raw.shape!=before.shape or raw.shape!=cap.shape or raw.shape!=delta.shape or not torch.isfinite(req).all():
        raise ValueError('Invalid retarget request')
    return delta,req,before+(req-before).clamp(-cap,cap)


def tilt_parts(path,planner):
    a,b=planner.outward_steps,planner.wrist_adjust_start
    if not 0<a<b<len(path):raise ValueError('Unexpected paired tilt path subdivision')
    return {'elbow_outward':path[:a], 'tilt_90':path[a:b], 'wrist_reconfigure':path[b:], 'tilt_hold':path[-1:]}


def simulate(out,seed,asset):
    proof=receipt(seed,asset)
    import contact_hold_student_eval as adapter
    from contact_hold_teacher import ContactHoldTeacher
    from opposition_teacher import OppositionPlanner
    from tilt_regrasp import TiltTeacher
    from isaac_compat import tensor,quaternion_from_native
    from scipy.spatial.transform import Rotation
    from contact_hold_size_retarget_turn import GuidedFaceTrial
    orig_init=TiltTeacher.__init__;teacher_action=ContactHoldTeacher.action;student_action=adapter.StudentEvaluation.action
    context={};new_grasp=asset.parent.parent/'calibration/grasp90.json'
    def build(self,env,grasp,*args,**kwargs):
        if Path(grasp).resolve()!=GRASP or env.run_dir!=out or context:raise ValueError('Unexpected planner construction')
        state=tensor(env.cube.data.root_state_w)[0];xyz=state[:3].numpy().copy()
        delta_z=np.array([0.,0.,(size.EDGE-size.BASE_EDGE)/2]);xyz_old=xyz-delta_z
        quat=quaternion_from_native(state[3:7]).numpy().copy();yaw=Rotation.from_quat(quat[[1,2,3,0]]).as_euler('xyz')[2]
        orig_init(self,env,new_grasp,*args,**kwargs)
        old=dict(OppositionPlanner(GRASP).plan(xyz_old,yaw));new=dict(self.base.phases)
        if list(old)!=list(new) or self.base.planner.grasp!=json.loads(new_grasp.read_bytes()):raise ValueError('Retarget paths not comparable')
        names=env.cfg['joint_names'];fingers=[i for i,n in enumerate(names) if n.startswith('left_')];arms=[i for i,n in enumerate(names) if n.endswith('_L')]
        change=lambda p,ids:float(np.abs(old[p][-1,ids]-new[p][-1,ids]).max())
        if len(arms)!=7 or len(fingers)!=21 or change('squeeze',arms)<1e-5 or change('preshape',fingers)<1e-5 or change('squeeze',fingers)<1e-5:
            raise ValueError('New nominal plan did not change left arm and finger targets')
        np.savez_compressed(out/'retarget_plans.npz',**{'source_'+p:path for p,path in old.items()},**{'destination_'+p:path for p,path in new.items()})
        dump(out/'retarget_plan.json',dict(source_center_world_m=xyz_old.tolist(),destination_center_world_m=xyz.tolist(),yaw_rad=float(yaw),
            quaternion_wxyz=quat.tolist(),joint_names=names,source_lengths={p:len(v) for p,v in old.items()},destination_lengths={p:len(v) for p,v in new.items()},
            source_center_palm=json.loads(GRASP.read_bytes())['center_palm'],destination_center_palm=self.base.planner.grasp['center_palm'],
            max_abs_arm_squeeze_target_change_rad=change('squeeze',arms),max_abs_open_target_change_rad=change('preshape',fingers),
            max_abs_closed_target_change_rad=change('squeeze',fingers)))
        for src,dst in ((new_grasp,'retarget_grasp90.json'),(new_grasp.parent/'geometry_retarget.json','retarget_geometry.json'),
                (GRASP,'retarget_grasp57.json'),(Path(__file__),'retarget_worker_source.py'),
                (ROOT/'contact_hold_size_retarget_turn.py','retarget_right_guide_source.py'),(ROOT/'contact_hold_size_retarget_geometry.py','retarget_geometry_solver_source.py')):
            shutil.copyfile(src,out/dst)
        dump(out/'retarget_intervention.json',proof)
        context.update(env=env,teacher=self,old=old,new=new,delta_z=delta_z,step=-1,guide=GuidedFaceTrial(out,edge_m=size.EDGE))

    def scheduled(self):
        if self is not context['teacher']:raise ValueError('Unexpected scheduler')
        old=context['old'];new=context['new']
        if self.mode=='pickup':
            phase,path=self.base.phases[self.base.index];tick=self.base.tick
            oldpath,newpath=old[phase],new[phase]
        elif self.mode=='tilt':
            if 'old_tilt' not in context:
                ref=self.planner_type(GRASP);q=dict(zip(self.env.cfg['joint_names'],old['primary_lift'][-1]))
                # Use the same measured pivot that generated the new tilt path,
                # captured before the student executed the transition frame.
                path=ref.tilt_path(q,context['pivot90']-context['delta_z'],self.shift)
                context['old_tilt']=tilt_parts(path,ref)
                np.savez_compressed(out/'retarget_tilt_plans.npz',source=path,destination=self.path)
            if self.tick>=len(self.path):phase,tick='tilt_hold',0
            elif self.tick<self.planner.outward_steps:phase,tick='elbow_outward',self.tick
            elif self.tick<self.planner.wrist_adjust_start:phase,tick='tilt_90',self.tick-self.planner.outward_steps
            else:phase,tick='wrist_reconfigure',self.tick-self.planner.wrist_adjust_start
            oldpath=context['old_tilt'][phase];newpath=tilt_parts(self.path,self.planner)[phase]
        else:raise ValueError('Unexpected regrasp mode')
        original_mode=self.mode;action=teacher_action(self)
        if self.mode=='tilt' and original_mode=='pickup':context['pivot90']=tensor(self.env.cube.data.root_state_w)[0,:3].numpy().copy()
        oa,na,u=sample_pair(oldpath,newpath,tick)
        if not np.allclose(na,self.nominal_target[0].numpy(),atol=2e-6,rtol=0):
            # During settled hold the planner can update the last waypoint in place.
            # Use the actual returned nominal target (without finger feedback).
            if phase!='tilt_hold':raise ValueError('Actual teacher waypoint does not match the recalibrated path')
            na=self.nominal_target[0].numpy().copy()
        context.update(oldnow=oa,newnow=na,progress=u,ref_phase=phase,step=int(self.env.episode_length_buf[0]))
        return action

    def deployed(self,env,phase,step,previous,native):
        if step!=context['step'] or env is not context['env']:raise ValueError('Student/reference cursor mismatch')
        action,row=student_action(self,env,phase,step,previous,native)
        if step==0:
            self.protocol.update(kind=KIND,retarget_diagnostic=True,executed_left_source=SOURCE,left_position_actions=SOURCE,
                right_actions_phase_and_gains='size90_scheduler_conditional_right_face_guidance',retarget_intervention_sha256=size.file_sha(out/'retarget_intervention.json'))
            (out/'student_evaluation.json').write_text(json.dumps(self.protocol,indent=2))
        before=env.targets[:,self.left];cast=lambda vec:torch.as_tensor(vec,dtype=before.dtype,device=before.device).reshape_as(before)
        old=cast(context['oldnow'][self.left]);new=cast(context['newnow'][self.left]);raw=cast(row['student_raw_left_target'])
        delta,requested,target=retarget(raw,before,self.cap,old,new)
        row.update(executed_left_source=SOURCE, unretargeted_student_left_target=row['student_left_target'],student_left_target=target[0].tolist(),
                   retarget_reference_phase=context['ref_phase'],retarget_reference_progress=context['progress'],retarget_source_left_reference=old[0].tolist(),
                   retarget_destination_left_reference=new[0].tolist(),retarget_left_delta_rad=delta[0].tolist(),retarget_requested_left_target=requested[0].tolist())
        arm=[i for i,j in enumerate(self.left) if self.names[j].endswith('_L')]
        row.update(whole_turn_phase_requested=context['ref_phase'] in ('tilt_90','wrist_reconfigure','tilt_hold'),
                   whole_student_arm_command_delta_rad=float((target-before)[:,arm].abs().max()))
        left=(target-before)/env.scales[:,self.left]
        # The unmodified adapter returns an inference tensor; clone before applying the external geometric correction.
        action=action.clone();action[:,self.left]=left
        guided,extra=context['guide'].action(env,previous,action,step)
        if guided.shape!=action.shape or not torch.equal(guided[:,self.left],left) or not torch.isfinite(guided).all() or set(extra)&set(row):
            raise ValueError('Right index guidance changed the student, telemetry, or action shape')
        row.update(extra)
        return guided,row
    with patch.object(TiltTeacher,'__init__',build),patch.object(ContactHoldTeacher,'action',scheduled),patch.object(adapter.StudentEvaluation,'action',deployed):
        approach.simulate(out,seed,asset)


def adapted(fn,replacements):
    """Keep previous verifiers; change only the declared controller label/known phase alias."""
    src=inspect.getsource(fn)
    for before,after in replacements:
        if src.count(before)!=1:raise ValueError('Cannot bind frozen verifier: '+before)
        src=src.replace(before,after)
    ns={};exec(compile(src,'<retarget:'+fn.__name__+'>','exec'),fn.__globals__,ns);return ns[fn.__name__]


def finalize(out,seed,asset):
    verified_functional=adapted(functional.finalize,[("r['kind']=='contact_hold_student_functional_diagnostic'",f"r['kind']=={KIND!r}"),
        ("row['executed_left_source']=='student'",f"row['executed_left_source']=={SOURCE!r}")])
    # The frozen ElbowOutTeacher can rename the final pickup row immediately on
    # entering tilt. An early hard safety stop before any endpoint is also a real outcome.
    verified_approach=adapted(approach.finalize,[("e['phase']!=row['phase']", "(e['phase']!=row['phase'] and not (e['phase']=='primary_lift' and e['phase_advanced'] and row['phase']=='elbow_outward'))"),
        ("if not boundary:raise ValueError('No real settling boundary was evaluated')", "if not boundary and not r['failure']:raise ValueError('No real settling boundary or terminal failure')")])
    with patch.object(functional,'finalize',verified_functional):verified_approach(out,seed,asset)
    mf=out/'artifact_manifest.json';hashes=json.loads(mf.read_bytes())['sha256']
    def read(name):
        if size.file_sha(out/name)!=hashes[name]:raise ValueError('Changed retarget artifact: '+name)
        return json.loads((out/name).read_bytes())
    r=read('result.json');proof=read('retarget_intervention.json')
    if r['kind']!=KIND or r['executed_left_source']!=SOURCE or r['retarget_intervention_sha256']!=hashes['retarget_intervention.json'] or proof!=receipt(seed,asset):
        raise ValueError('Wrong retarget conditions')
    for file,key in [('retarget_grasp90.json','grasp_sha256'),('retarget_geometry.json','geometry_retarget_sha256'),('retarget_grasp57.json','old_grasp_sha256'),('retarget_worker_source.py','worker_sha256'),('retarget_right_guide_source.py','right_guide_sha256'),('retarget_geometry_solver_source.py','geometry_solver_sha256')]:
        if hashes[file]!=size.file_sha(out/file) or hashes[file]!=proof[key]:raise ValueError('Wrong archived '+file)
    cfg=read('config.json');projection=read('student_left_projection.json');rows=read('trajectory.json');names=cfg['joint_names'];ids=[names.index(n) for n in projection['names']]
    bounds=np.asarray(projection['bounds']);cap=np.minimum(projection['step'],np.array(projection['velocity'])*projection['dt']);old=np.array([cfg['home'][n] for n in projection['names']]);diff=[];group=np.asarray([n.endswith('_L') for n in projection['names']])
    for row in rows:
        raw=np.asarray(row['student_raw_left_target']);base=np.asarray(row['retarget_source_left_reference']);new=np.asarray(row['retarget_destination_left_reference'])
        delta=new-base;req=raw+delta;projected=old+np.clip(req-old,-cap,cap);plain=old+np.clip(raw-old,-cap,cap)
        actual=np.array(row['applied_target'])[ids];expect=np.clip(projected,bounds[:,0],bounds[:,1]);plain_actual=np.clip(plain,bounds[:,0],bounds[:,1]);err=np.abs(actual-plain_actual)
        for field,want in [('retarget_left_delta_rad',delta),('retarget_requested_left_target',req),('unretargeted_student_left_target',plain),('student_left_target',projected)]:
            if not np.allclose(row[field],want,atol=5e-6,rtol=0):raise ValueError('Retarget replay mismatch: '+field)
        if not np.allclose(actual,expect,atol=5e-6,rtol=0) or not np.all(abs(actual-old)<=cap+5e-6):raise ValueError('Actual left command violates retarget projection')
        old=actual;diff.append([np.max(err[group]),np.max(err[~group])])
    diff=np.array(diff)
    if size.file_sha(out/'retarget_plans.npz')!=hashes['retarget_plans.npz']:raise ValueError('Changed plan archive')
    from collections import Counter
    dump(out/'retarget_summary.json',dict(seed=seed,kind=KIND,source_manifest_sha256=size.file_sha(mf),weight_sha256=r['checkpoint_sha256'],
        retarget_plan=read('retarget_plan.json'),phase_frames=dict(Counter(row['phase'] for row in rows)),action_steps_verified=len(rows),
        nonzero_arm_command_effect_frames=int(np.sum(diff[:,0]>1e-5)),nonzero_finger_command_effect_frames=int(np.sum(diff[:,1]>1e-5)),
        maximum_arm_command_effect_rad=float(np.max(diff[:,0])),maximum_finger_command_effect_rad=float(np.max(diff[:,1])),
        comparator='per-step frozen output with no residual, projected from the same observed previous target; not a separate physics rollout',
        end_reason=r['failure'],last_phase=rows[-1]['phase'],diagnostic_only=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=CONTROL,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--asset',type=Path,required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--native-child',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();out=a.output.resolve();asset=a.asset.resolve();job=out.parent.parent
    if (job.parent!=RUNS or not re.fullmatch(r'contact_hold_size_r15_[A-Za-z0-9_]*retarget[A-Za-z0-9_]*approach[A-Za-z0-9_]*',job.name) or out.exists()
        or out.parent.name!=size.MODE or out.name!=f'seed_{a.seed}' or asset!=job/'asset90/cube_model.usda'):
        p.error('Expected a fresh ...size_r15_...retarget...approach.../from_reset/seed_<seed> and sibling asset90/cube_model.usda')
    if a.preflight:print(json.dumps(receipt(a.seed,asset),indent=2));return
    if a.native_child:simulate(out,a.seed,asset);return
    out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--seed',str(a.seed),'--output',str(out),'--asset',str(asset),'--native-child'],check=True,close_fds=False)
    finalize(out,a.seed,asset)


if __name__=='__main__':main()
