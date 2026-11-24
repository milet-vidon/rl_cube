"""Verify a frozen student rollout and replay teacher queries only on collection states."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from contact_hold_dataset import PICKUP_PHASES, sha, verified_json
from joint_safety import project_target
from phase_contact_policy import features

ROOT = Path(__file__).resolve().parent
INITIAL = ROOT/'runs/contact_hold_round2_20261107_155006_131092588'
INITIAL_SHA = '9deeac71f530ea0adb26a2600019f811e49e41145fc192d29089a6b4320eef78'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    tmp = p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def checkpoint(run):
    run = Path(run)
    r = read(run/'result.json')
    if not (r['completed'] and r['checkpoint_reload_verified'] and sha(run/'best.pt') == r['checkpoint_sha256']):
        raise ValueError('Incomplete or changed fit: '+str(run))
    return r['checkpoint_sha256']


def inspect(run, fit, seed, role):
    run = Path(run)
    m = read(run/'artifact_manifest.json')
    result, protocol, rows = [verified_json(run,n,m) for n in ('result.json','student_evaluation.json','trajectory.json')]
    digest = checkpoint(fit)
    if not (role in ('collection','validation','test') and role == protocol['evaluation_role'] == result['evaluation_role']
            and result['split'] == m['split'] == ('independent' if role == 'test' else 'development')
            and result['kind'] == 'contact_hold_bc_closed_loop_evaluation'
            and result['full_episode'] and m['full_episode'] and not result['execution_interrupted']
            and seed == result['seed'] == protocol['seed'] == m['seed']
            and Path(result['training_run']).resolve() == Path(fit).resolve()
            and result['checkpoint_sha256'] == protocol['checkpoint_sha256'] == m['sha256']['evaluated_best.pt'] == digest
            and sha(run/'evaluated_best.pt') == digest
            and result['cube_pose_writes_during_control'] == 0
            and result['eligible_training_episode'] is False and result['corrective_labels_qualified_for_training'] is False
            and result['student_executed_steps'] == result['steps'] == len(rows) and rows
            and all(r['step']==i and r['executed_left_source']=='student' and r['phase'] in PICKUP_PHASES for i,r in enumerate(rows))
            and len({r['pose_reset_count'] for r in rows})==1):
        raise ValueError('Rollout identity, completeness, or split mismatch: '+str(run))
    for n in ('environment.py','joint_safety.py','contact_hold_student_eval.py','single_face_experiment.py'):
        if sha(ROOT/n) != m['sha256'][n]:
            raise ValueError('Rollout implementation mismatch: '+n)
    safe = all(not unsafe(r) for r in rows)
    passed = bool(safe and result['dynamic_grasp_passed'] and result['failure'] is None and result['wrist']['accepted']
                  and result['measured']['accepted'] and result['joint_wrist_hold_s']>=3.0)
    record = dict(run=str(run.resolve()), fit=str(Path(fit).resolve()), seed=seed, role=role,
                  checkpoint=digest, manifest=sha(run/'artifact_manifest.json'), passed=passed,safe=safe,
                  phase=rows[-1]['phase'], phase_rank=max(PICKUP_PHASES.index(r['phase']) for r in rows),
                  hold_s=float(result['joint_wrist_hold_s']), failure=result['failure'], steps=len(rows))
    return record, m, rows


def unsafe(r):
    return (r['drop'] or r['self_contact_impulse']>1e-6 or r['robot_table_impulse']>1e-6 or
            r['arm_soft_limit_hit'] or r['actual_arm_soft_limit_violation'] or any(r['hard_limit_hit'])
            or r['layer_measurement']['maximum_relative_rotation_deg']>30)


def transitions(rows, spec, left, bounds):
    """Last two reached phases; reject the terminal action and anything after unsafe state."""
    b = torch.as_tensor(spec['bounds'],dtype=torch.float32)[None]
    step,vel = [torch.as_tensor(spec[k],dtype=torch.float32)[None] for k in ('step','velocity')]
    dt = spec['dt']
    if (b.shape!=(1,28,2) or step.shape!=(1,28) or vel.shape!=(1,28) or not 0<dt<1 or
            not all(torch.isfinite(v).all() for v in (b,step,vel)) or (b[:,:,0]>=b[:,:,1]).any()):
        raise ValueError('Invalid runtime projector')
    phases = list(dict.fromkeys(r['phase'] for r in rows))
    selected = phases[-2:]
    xs,ys,masks,valid,steps = [],[],[],[],[]
    start = 0
    for i,r in enumerate(rows):
        if unsafe(r):
            break
        if not i or rows[i-1]['phase']!=r['phase']:
            start = i
        clock = min((i-start)/300.,1.)
        if clock!=r['phase_elapsed']:
            raise ValueError('Clock mismatch')
        if not i or i==len(rows)-1 or r['phase'] not in selected:
            continue
        prev = rows[i-1]
        before = torch.tensor(prev['applied_target'])[None,left]
        student = torch.tensor(r['student_left_target'])[None]
        raw = torch.tensor(r['student_raw_left_target'])[None]
        cap = torch.minimum(step,vel*dt)
        torch.testing.assert_close(before+(raw-before).clamp(-cap,cap),student, atol=1e-6,rtol=0)
        replay,_ = project_target(before,student,b,step,vel,dt)
        torch.testing.assert_close(replay,torch.tensor(r['applied_target'])[None,left],atol=1e-6,rtol=0)
        teacher = torch.tensor(r['teacher_left_requested_not_executed'])[None]
        target,hit = project_target(before,teacher,b,step,vel,dt)
        y = target[0].numpy()
        mask = ~hit[0].numpy() & ~np.asarray(r['soft_limit_hit'],bool)[left] & (y>=bounds[:,0]-1e-6) & (y<=bounds[:,1]+1e-6)
        if mask.any():
            xs.append(features(prev,r['phase'],clock));ys.append(y);masks.append(mask)
            valid.append(((i-start)//10)%5 == 4);steps.append(i)
    return dict(x=np.asarray(xs,np.float32).reshape(-1,198),y=np.asarray(ys,np.float32).reshape(-1,28),
                mask=np.asarray(masks,bool).reshape(-1,28),val=np.asarray(valid,bool),steps=np.asarray(steps,int))


def collect(entry, out, allowed):
    if entry['role']!='collection' or entry['seed'] not in allowed:
        raise ValueError('Validation and held-out episodes are never training data')
    run = Path(entry['run'])
    rec,m,rows = inspect(run,entry['fit'],entry['seed'],'collection')
    if rec != entry:
        raise ValueError('Collection manifest changed')
    spec = verified_json(run,'student_left_projection.json',m)
    cfg = verified_json(run,'config.json',m)
    saved = torch.load(run/'evaluated_best.pt',map_location='cpu',weights_only=True)
    meta = saved['metadata']
    left = meta['output_indices']
    if cfg['joint_names']!=meta['joint_names'] or spec['names']!=[cfg['joint_names'][i] for i in left]:
        raise ValueError('Correction joint order mismatch')
    bounds = torch.stack([saved['model'][n] for n in ('lower','upper')],dim=1).numpy()
    arrays = transitions(rows,spec,left,bounds)
    if not (~arrays['val']).any():
        return None
    out = Path(out)
    tmp = out.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays)
    if out.exists():
        same = sha(tmp)==sha(out)
        tmp.unlink()
        if not same:
            raise ValueError('Archived correction data no longer reproduces')
    else:
        tmp.replace(out)
    return dict(path=str(out.resolve()),sha256=sha(out),source=entry,training_samples=int((~arrays['val']).sum()),
                validation_samples=int(arrays['val'].sum()),scope='projected_counterfactual_left_joint_labels_only')
