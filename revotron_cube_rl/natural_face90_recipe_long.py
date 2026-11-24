"""Paper-inspired state PPO experiment; no vision or real-hardware claims.

Every iteration collects continuous real PhysX episodes from the normal table
reset, using the frozen R27 and natural-presentation prefix. Only measured
post-presentation transitions enter PPO. Validation never updates the network.
"""
import argparse
from contextlib import ExitStack, contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from collections import Counter
from natural_face90_v5_long import (
    HERE, ROOT, PARENT, ZERO, FILES as BASE_FILES, exclusive, gpu, read, save,
    digest, verified_rows, stats, final_seeds,
)

FILES = (*BASE_FILES, 'natural_face90_recipe_long.py', 'natural_face90_recipe_worker.py',
         'natural_face90_recipe_policy.py', 'run_natural_face90_recipe_long.sh',
         'face_contact_calibration_v1.json', 'natural_face90_recipe_frames.py',
         'natural_face90_recipe_contacts.py', 'natural_face90_recipe_guide.py')

GUIDED_SCHEMA = 'natural_face90_recipe_guided_ppo_v2'
LEGACY_SCHEMA = 'natural_face90_recipe_state_ppo_v1'
LEGACY_CHANGED = {
    'natural_face90_recipe_long.py': 'e8ff31dfbeac4b01219afc58bd695717376422e6f746e2a94a4100d77faa0b87',
    'natural_face90_recipe_worker.py': 'f8f39b9a272461b7a5bd518811fb163bd133cbbbad37ef298240e6207cd1fdf7',
}


def fingerprint(): return {n:digest(HERE/n) for n in FILES}
def worker_software(pins): return {k:v for k,v in pins.items() if k.endswith('.py')}


def transfer_origin(folder, current_pins):
    """Import a committed v1 iteration; keep its optimizer and experiment history separate."""
    folder = folder.resolve()
    cfg = read(folder.parent/'config.json')
    result = read(folder/'result.json')
    iteration = result['iteration']
    pins = {k:v for k,v in current_pins.items() if k != 'natural_face90_recipe_guide.py'}
    pins.update(LEGACY_CHANGED)
    if (type(iteration) is not int or iteration < 1 or folder.name != f'iteration_{iteration:04d}'
            or cfg['schema'] != LEGACY_SCHEMA or cfg['parent_sha'] != PARENT or cfg['source_sha256'] != pins):
        raise ValueError('Unsupported source experiment or changed legacy source pins')
    slot = folder/'train'
    rec = read(slot/'accepted.json')
    attempt = (slot/rec['attempt']).resolve()
    report = attempt/'report.json'
    req = read(slot/'request.json')
    if (attempt.parent != slot or rec['request_sha'] != digest(slot/'request.json')
            or rec['report_sha'] != digest(report) or req['iteration'] != iteration
            or req['role'] != 'train' or req['software'] != worker_software(pins)
            or req['recipe_schema'] != LEGACY_SCHEMA or req['parent_sha'] != PARENT):
        raise ValueError('Invalid transfer source training receipt')
    _, learn = verify_result(report, req)
    if result['learning'] != learn:
        raise ValueError('Source iteration disagrees with accepted training report')
    return dict(iteration=iteration, folder=str(folder), checkpoint=learn['output_checkpoint'],
                sha=learn['output_checkpoint_sha'], software=req['software'])


@contextmanager
def available_gpu():
    # Recheck the native locks and compute-process list on every attempt.
    # Catch contention only at acquisition. Never retry a failed physical rollout.
    with ExitStack() as stack:
        while True:
            try:
                fds = stack.enter_context(gpu())
                break
            except RuntimeError as exc:
                if not (isinstance(exc.__cause__, BlockingIOError) or str(exc).startswith('GPU 0 正被使用；')):
                    raise
                print('等待 GPU 空闲，60 秒后重查：', exc, flush=True)
                time.sleep(60)
        yield fds


def worker_failure(code, dest, log):
    detail=''
    failed=dest/'failed.json'
    if failed.is_file():
        try:
            error=read(failed).get('error')
            if isinstance(error,str):detail='; '+error[:2000]
        except (ValueError,OSError):
            pass
    return RuntimeError(f'PhysX/PPO exited {code}{detail}; inspect {log}')


def validate_request(req):
    if (req.get('recipe_schema')!=GUIDED_SCHEMA or
        type(req.get('learn')) is not bool or req['learn']!=(req['role'] in ('train','smoke')) or
        req['target_deg']!=15 or any(p!=ZERO for p in req['parameters']) or
        type(req.get('iteration')) is not int or req['iteration']<1 or
        type(req.get('policy_seed')) is not int or not 0<=req['policy_seed']<2**31):
        raise ValueError('Invalid PPO specialist request')
    if (req.get('checkpoint') is None)!=(req.get('checkpoint_sha') is None):
        raise ValueError('Checkpoint path/hash pair required')
    if req['role']=='validate' and req['checkpoint'] is None:
        raise ValueError('Validation requires a frozen checkpoint')
    transfer = req.get('transfer')
    if transfer and not (req['role']=='train' and req['iteration']==transfer['iteration']+1
                         and req['checkpoint']==transfer['checkpoint'] and req['checkpoint_sha']==transfer['sha']):
        raise ValueError('Transfer is only valid for the first training wave after the source iteration')


def verify_result(report, req):
    rows=verified_rows(report,req);rep=read(report);learn=rep['expert_learning']
    if (learn['kind']!='state_PPO_specialist' or learn['deterministic']==req['learn'] or
        learn['input_checkpoint_sha']!=req['checkpoint_sha'] or
        learn['transition_evidence_sha']!=digest(report.parent/'policy_transitions.jsonl')):
        raise ValueError('Invalid measured PPO evidence')
    if req['learn']:
        ck=report.parent/'checkpoint.pt'
        if (learn['actor_updated'] is not True or learn['actor_max_delta']<=0 or
            learn['real_transitions']!=learn['learned_transitions'] or
            learn['output_checkpoint']!=str(ck) or learn['output_checkpoint_sha']!=digest(ck)):
            raise ValueError('No verified real-data PPO parameter update')
    elif learn['actor_updated'] is not False or 'output_checkpoint' in learn:
        raise ValueError('Evaluation attempted to update policy')
    return rows,learn


def wave(slot, req, fd):
    validate_request(req);slot.mkdir(parents=True,exist_ok=True)
    plan=slot/'request.json'
    if plan.exists():
        if read(plan)!=req: raise ValueError('Changed wave request')
    else: save(plan,req)
    receipt=slot/'accepted.json'
    if receipt.exists():
        rec=read(receipt);report=slot/rec['attempt']/'report.json'
        if rec['request_sha']!=digest(plan) or rec['report_sha']!=digest(report):
            raise ValueError('Changed accepted PPO wave')
        return verify_result(report,req)
    if req['checkpoint'] is not None and digest(Path(req['checkpoint']))!=req['checkpoint_sha']:
        raise ValueError('Changed PPO parent checkpoint')
    dest=slot/('attempt_'+uuid.uuid4().hex[:12])
    with available_gpu() as fds, (slot/(dest.name+'.log')).open('x') as log:
        env=dict(os.environ,CUBE_DASH_GPU_FD=str(fds[0]),CUBE_DASH_ISAAC_FD=str(fds[1]),PYTHONUNBUFFERED='1')
        command=[sys.executable,'-B','-u',str(HERE/'natural_face90_recipe_worker.py'),
                 '--request',str(plan),'--out',str(dest)]
        print('State PPO',req['role'],'iteration=',req['iteration'],'lanes=',len(req['seeds']),'log=',log.name,flush=True)
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,*fds),env=env)
        if result.returncode: raise worker_failure(result.returncode,dest,log.name)
    report=dest/'report.json';rows,learn=verify_result(report,req)
    save(receipt,dict(attempt=dest.name,request_sha=digest(plan),report_sha=digest(report)))
    return rows,learn


def request(cfg,role,seeds,k,checkpoint):
    result = dict(recipe_schema=cfg['schema'],role=role,learn=role!='validate',
        parent_sha=PARENT,target_deg=15,parameters=[ZERO.copy() for _ in seeds],seeds=seeds,
        iteration=k,policy_seed=cfg['rng'],software=worker_software(cfg['source_sha256']),
        checkpoint=checkpoint,checkpoint_sha=digest(Path(checkpoint)) if checkpoint else None)
    origin = cfg.get('origin')
    if origin and k==origin['iteration']+1 and role=='train':
        result['transfer']=origin
    return result


def panel(home,cfg,k,checkpoint,seeds,fd):
    rows=[]
    for offset in range(0,len(seeds),cfg['lanes']):
        chunk=seeds[offset:offset+cfg['lanes']]
        rs,_=wave(home/f'wave_{offset:04d}',request(cfg,'validate',chunk,k,checkpoint),fd)
        rows.extend(rs)
    return stats(rows)


def campaign(home,cfg,fd):
    f=home/'state.json'
    origin = cfg.get('origin') or {}
    state=read(f) if f.exists() else dict(iteration=origin.get('iteration',0),
        checkpoint=origin.get('checkpoint'),status='running',best_validation=None)
    if origin and not f.exists():
        save(f,state)
    if state['status']!='running':
        print('Finished:',state['status']);return
    for k in range(state['iteration']+1,cfg['iterations']+1):
        if fingerprint()!=cfg['source_sha256']: raise ValueError('Sources changed; use a fresh experiment')
        folder=home/f'iteration_{k:04d}'
        seeds=list(range(45_000_000+(k-1)*32,45_000_000+(k-1)*32+cfg['lanes']))
        rows,learn=wave(folder/'train',request(cfg,'train',seeds,k,state['checkpoint']),fd)
        checkpoint=learn['output_checkpoint']; train=stats(rows)
        result=dict(iteration=k,training_rollouts_before_update=train,learning=learn,
                    end_reasons=dict(Counter(r['end_reason'] for r in rows)))
        if k%cfg['validate_every']==0 or k==cfg['iterations'] or (origin and k==origin['iteration']+1):
            valid=panel(folder/'validate',cfg,k,checkpoint,
                        list(range(65_000_000+(k-1)*32,65_000_000+(k-1)*32+cfg['validation_seeds'])),fd)
            result['validation_after_update']=valid
            def rank(s):return (s['successes'],-s['unsafe'],s['progress'],s['natural'])
            if state['best_validation'] is None or rank(valid)>rank(state['best_validation']):
                state['best_validation']=valid
                save(home/'best_candidate.json',dict(checkpoint=checkpoint,sha=digest(Path(checkpoint)),
                    validation=valid,iteration=k,qualified=False,source_sha256=cfg['source_sha256']))
            if valid['unsafe']==0 and valid['successes']>=math.ceil(.875*cfg['validation_seeds']):
                seeds=final_seeds(home,ROOT/'runs/natural_face90_test_registry')
                test=panel(folder/'independent_test',cfg,k,checkpoint,seeds,fd)
                result['independent_test']=test
                save(home/'final_test.json',dict(checkpoint=checkpoint,sha=digest(Path(checkpoint)),seeds=seeds,results=test))
                state['status']='qualified_15deg_state_specialist' if test['unsafe']==0 and test['successes']>=28 else 'independent_test_not_met'
                if state['status']=='qualified_15deg_state_specialist':
                    save(home/'qualified_policy.json',dict(checkpoint=checkpoint,sha=digest(Path(checkpoint)),
                        results=test,vision=False,simulation_only=True,source_sha256=cfg['source_sha256']))
        state.update(iteration=k,checkpoint=checkpoint)
        if k==cfg['iterations'] and state['status']=='running':state['status']='iteration_limit'
        save(folder/'result.json',result);save(f,state)
        print('iteration',k,'status',state['status'],'train',train,'PPO',learn,flush=True)
        if 'validation_after_update' in result: print('validation',result['validation_after_update'],flush=True)
        if state['status']!='running':break


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=200)
    p.add_argument('--lanes',type=int,choices=(2,4,8,16,32),default=32)
    p.add_argument('--validate-every',type=int,default=10)
    p.add_argument('--validation-seeds',type=int,choices=(8,16,32),default=16)
    p.add_argument('--rng',type=int,default=327)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--continue-from',type=Path,metavar='ITERATION_DIR',
                   help='Start a separate guided run from an accepted original PPO iteration')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--check',action='store_true');mode.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not 1<=args.iterations<=2000 or not 1<=args.validate_every<=args.iterations or not 0<=args.rng<2**31:
        p.error('Invalid iteration/rng bounds')
    from natural_face90_v5_worker import inputs
    source, *_=inputs(dict(role='smoke',seeds=[20_000_000,20_000_001],parameters=[ZERO,ZERO],target_deg=15,parent_sha=PARENT))
    from natural_face90_recipe_frames import check_source_scene
    frame=check_source_scene(source)
    from natural_face90_recipe_contacts import check_capacity_source
    contact_capacity=check_capacity_source(args.lanes)
    cfg={k:getattr(args,k) for k in ('iterations','lanes','validate_every','validation_seeds','rng','smoke')}
    cfg.update(schema=GUIDED_SCHEMA,parent_sha=PARENT,source_sha256=fingerprint())
    cfg['raw_contact_capacity']=contact_capacity
    home=args.run_dir.resolve()
    origin_path=args.continue_from
    if args.resume and origin_path is None and (home/'config.json').is_file():
        stored=read(home/'config.json').get('origin')
        if stored: origin_path=Path(stored['folder'])
    if origin_path:
        cfg['origin']=transfer_origin(origin_path,cfg['source_sha256'])
        if args.smoke or args.iterations<=cfg['origin']['iteration']:
            p.error('Continuation needs a higher total iteration count and training mode')
    if args.check:
        print('CHECK OK: source pins, original gates, frozen R27/domain, calibrated contact frame',frame,
              '; raw contact capacity',contact_capacity,'; source iteration',
              (cfg.get('origin') or {}).get('iteration'),'; no PhysX or training started');return
    if home==ROOT/'runs' or ROOT/'runs/contact_hold90_long_v2' in (home,*home.parents):
        p.error('Use a separate experiment directory')
    if args.resume:
        if not (home/'config.json').is_file():p.error('No configured run to resume')
    else:home.mkdir(parents=True,exist_ok=False)
    with exclusive(home/'campaign.lock') as fd:
        if args.resume:
            if read(home/'config.json')!=cfg: raise ValueError('Resume requires identical config and sources')
        else:save(home/'config.json',cfg)
        if args.smoke:
            seeds=list(range(20_000_000,20_000_000+args.lanes))
            rows,learn=wave(home/'smoke',request(cfg,'smoke',seeds,1,None),fd)
            result=dict(diagnostic_only=True,summary=stats(rows),end_reasons=dict(Counter(r['end_reason'] for r in rows)),
                        learning=learn,rows=rows,checkpoint_validated=False)
            save(home/'smoke.json',result)
            print('smoke diagnostics:',json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)
        else:campaign(home,cfg,fd)


if __name__=='__main__':main()
