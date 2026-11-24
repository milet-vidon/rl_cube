"""Verify actual student commands, then compare supported whole and face rotation to the previous 90 mm grasp."""
import json
from pathlib import Path

import numpy as np
import torch
from contact_hold_grace_job import RUNS, SEEDS, digest, read
from contact_hold_grace_worker import CHECKPOINT
from contact_hold_policy import restored
from phase_contact_policy import features
from contact_hold_simple_rotation import summarize_rotation
from contact_hold_size_face_score import summarize_face

BASE=RUNS/'contact_hold_size_r15_90mm_approach_20261112_083044'

def checked(run):
    m=read(run/'artifact_manifest.json')
    def get(name):
        if digest(run/name)!=m['sha256'][name]:raise ValueError(f'Manifest mismatch: {run/name}')
        return read(run/name)
    return m,get

def replay_raw(run, rows, hz=30.):
    """Independently reconstruct the network output from archived previous post-step observations."""
    ckpt=torch.load(run/'evaluated_best.pt',map_location='cpu',weights_only=True)
    model=restored(ckpt).eval()
    inputs=np.stack([features(prev,r['phase'],r['phase_elapsed']) for prev,r in zip(rows,rows[1:])])
    with torch.inference_mode():
        actual=model(torch.from_numpy(inputs)).numpy()
    ref=np.asarray([r['student_raw_left_target'] for r in rows[1:]])
    err=np.max(np.abs(actual-ref))
    if not (np.isfinite(err) and err<2e-4):raise ValueError(f'Frozen policy did not replay: {err}')
    audit=dict(frames_excluding_initial=len(inputs),max_abs_raw_output_error_rad=float(err))
    if rows[0]['executed_left_source']=='canonical_student_plus_size_plan_delta':
        canonical=[];last=np.zeros(56)
        for prev,row in zip(rows,rows[1:]):
            d=np.asarray(prev['canonical_destination_reference56'])-prev['canonical_source_reference56']
            dv=(d-last)*hz;last=d
            if not (np.allclose(row['canonical_prev_delta56'],d,atol=1e-6,rtol=1e-5) and
                    np.allclose(row['canonical_prev_velocity_delta56'],dv,atol=1e-4,rtol=1e-5)):raise ValueError('Invalid sensor offset history')
            p=dict(prev)
            for k,delta in [('q',d),('joint_velocity',dv),('applied_target',d),('cube_position',np.array([0.,0.,.0165]))]:
                p[k]=np.asarray(prev[k])-delta
            canonical.append(features(p,row['phase'],row['phase_elapsed']))
        with torch.inference_mode():out=model(torch.from_numpy(np.stack(canonical))).numpy()
        ref=np.asarray([row['canonical_student_raw_left_target'] for row in rows[1:]])
        ce=float(np.max(np.abs(out-ref)))
        if not(np.isfinite(ce) and ce<2e-4):raise ValueError(f'Canonical policy did not replay: {ce}')
        audit['max_abs_canonical_output_error_rad']=ce
    return audit

def score(run,seed):
    m,get=checked(run)
    r=get('result.json');rows=get('trajectory.json');cfg=get('config.json');asset=get('cube_asset.json');materials=asset['model']
    runtime=get('size_runtime.json')
    if not(m['full_episode'] and r['full_episode'] and r['seed']==seed and not r['execution_interrupted'] and
           r['checkpoint_sha256']==digest(run/'evaluated_best.pt')==CHECKPOINT and r['steps']==len(rows) and
           r['cube_pose_writes_during_control']==0 and not r['eligible_training_episode']
           and asset['source_sha256']==digest(run.parent.parent/'asset90/cube_model.usda') and runtime['verified']
           and len(materials['bodies'])==27 and materials['runtime_friction']==[.9,.8]
           and abs(runtime['total_mass_kg']-.0685)<1e-6): raise ValueError('Unverified or incomplete 90mm trial')
    if max(abs(q['height_m']-(q['cube_position'][2]-cfg['table_z']-.045)) for q in rows)>1e-6:raise ValueError('Wrong height zero')
    source=rows[0]['executed_left_source']
    if source not in ('student_plus_size_plan_delta','canonical_student_plus_size_plan_delta') or any(q['executed_left_source']!=source for q in rows):raise ValueError('Unexpected left controller')
    raw_audit=replay_raw(run,rows,cfg['control_hz'])
    hold=summarize_rotation(rows,cfg);face=summarize_face(rows,cfg)
    phases=list(dict.fromkeys(q['phase'] for q in rows))
    left_contact=[q['step'] for q in rows if q['contact_impulse'][0]>cfg['success']['contact_impulse_min']]
    return dict(seed=seed,run=str(run),manifest_sha256=digest(run/'artifact_manifest.json'),network_replay=raw_audit,
                checkpoint_sha256=CHECKPOINT,steps=len(rows),end_reason=r['failure'],phases=phases,
                left_contact_frames=len(left_contact),first_left_contact_step=next(iter(left_contact),None),
                peak_height_m=max(x['height_m'] for x in rows),final_table_contact=bool(rows[-1]['cube_table_contact']),
                controller=source,hold=hold,single_face=face,measured_phase_end_aperture=measured_aperture(rows,cfg,read(run.parent.parent/'calibration/grasp90.json')),command_diagnostic=commands(run,rows))

def measured_aperture(rows,cfg,grasp):
    """Measured joint configurations -> pad-anchor span along the hand's closing axis (not planned targets)."""
    from kinematics import RobotKinematics
    k=RobotKinematics();normal=np.array(grasp['cube_rotation_palm'])[:,2]
    anchors=grasp['contact_anchors'];phase_rows={}
    for row in rows:phase_rows[row['phase']]=row
    out={}
    for phase,row in phase_rows.items():
        q=dict(zip(cfg['joint_names'],row['q'],strict=True))
        palm=k.fk_link(q,'left_palm_link');project=palm[:3,:3]@normal
        contacts={}
        for finger,a in anchors.items():
            t=k.fk_link(q,a['link']);p=t[:3,:3]@a['point_local']+t[:3,3]
            contacts[finger]=float(p@project)
        gaps={f:round(1000*(x-contacts['thumb']),3) for f,x in contacts.items() if f!='thumb'}
        out[phase]=dict(step=row['step'],thumb_to_finger_mm=gaps)
    return out

def commands(run,rows):
    # Set explicitly from actual post-step command evidence; do not infer layer commands from angles.
    from contact_hold_size_retarget_turn import summarize_commands
    if rows[0]['executed_left_source']=='canonical_student_plus_size_plan_delta':
        # The frozen command audit recognizes the earlier adapter label. The score above
        # verifies the actual label and network; only alias that label in this read-only count.
        output=summarize_commands([dict(r,executed_left_source='student_plus_size_plan_delta') for r in rows])
        output['executed_left_sources']=['canonical_student_plus_size_plan_delta']
    else:output=summarize_commands(rows)
    output['planned_request_file_exists']=(run/'guided_face_request.json').is_file()
    return output

def write_report(job):
    plan=read(job/'pair_plan.json')
    if not all(digest(Path(p))==s for p,s in plan['sha256'].items()):raise ValueError('Common two-seed calibration was not preserved')
    ctrl_path=BASE/'size_comparison.json'
    if digest(ctrl_path)!=read(BASE/'state.json')['report_sha256']:raise ValueError('Changed original 90mm control panel')
    base=read(ctrl_path)
    episodes={}
    for seed in SEEDS:
        out=job/'from_reset'/f'seed_{seed}';c=base['episodes'][str(seed)]['90']
        new=score(out,seed);g=read(out/('canonical_summary.json' if (out/'canonical_summary.json').exists() else 'retarget_summary.json'))
        old=dict(seed=seed,manifest_sha256=c['manifest_sha256'],peak_height_m=c['original']['max_height_m'],
                 hold_s=c['practical']['best_controlled_hold']['seconds'],end_reason=c['end_reason'],phases=c['phases'],
                 whole_turn=c['practical']['ever_hold_and_turn'],single_face=c['single_face']['qualified'])
        _,get=checked(BASE/'from_reset'/f'seed_{seed}')
        old['measured_phase_end_aperture']=measured_aperture(get('trajectory.json'),get('config.json'),read(job/'calibration/grasp90.json'))
        episodes[str(seed)]=dict(previous_same_seed_same_size=old,retargeted=new,geometry_action_audit=g)
    dst=job/'retarget_report.json'
    with dst.open('x') as f:json.dump(dict(kind='paired_90mm_retarget_development',diagnostic_only=True,
                  weights_sha256=CHECKPOINT,grasp_sha256=digest(job/'calibration/grasp90.json'),
                  original_control_report_sha256=digest(ctrl_path),episodes=episodes),f,indent=2,allow_nan=False)
    return dst
