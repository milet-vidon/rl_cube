"""Train a goal-conditioned opposing-grasp policy from the frozen R27 pickup."""
import argparse
from pathlib import Path
import re

from contact_hold90_cuda_domain import _specialize
import natural_face90_recipe_long as recipe
from natural_face90_recipe_long import HERE, ROOT, PARENT, ZERO, digest, read, save, worker_software
from natural_face90_v7_long import accepted_report

SCHEMA = 'natural_face90_recipe_bimanual_ppo_v1'
HOME_RUN = ROOT/'runs'/'r27_recipe_bimanual_ppo'
TRAIN_BASE, VALID_BASE = 50_400_000, 68_600_000
GOALS = (2, 15, 90)
FILES = tuple(dict.fromkeys((*recipe.FILES, 'natural_face90_v5_contacts.py', 'natural_face90_v7_long.py',
    'natural_face90_demo_control.py', 'natural_face90_demo_usd.py', 'natural_face90_turn_contacts.py',
    'config.json', 'opposition_grasp_v20.json', 'natural_face90_v5_regression.json',
    'natural_face90_placo.py', 'natural_face90_placo_client.py',
    'natural_face90_bimanual.py', 'natural_face90_bimanual_control.py', 'natural_face90_bimanual_contacts.py',
    'natural_face90_bimanual_policy.py','natural_face90_bimanual_worker.py','run_natural_face90_bimanual.sh')))


def fingerprint():
    return {name: digest(HERE/name) for name in FILES}


def block(base, iteration, count):
    first = base + 32*(iteration-1)
    return list(range(first, first+count))


def request(cfg, role, seeds, k, goal, ck, ck_sha, independent=False):
    return dict(recipe_schema=SCHEMA, role=role, learn=role=='train', parent_sha=PARENT, seeds=seeds,
                parameters=[ZERO.copy() for _ in seeds], iteration=k, target_deg=goal, policy_seed=cfg['rng'],
                software=worker_software(cfg['source_sha256']),
                asset_sha256={n:h for n,h in cfg['source_sha256'].items() if not n.endswith('.py')},
                placo=cfg['placo'], checkpoint=ck,
                checkpoint_sha=ck_sha, independent_test=independent)


def validate_request(req):
    k, seeds, role = req['iteration'], req['seeds'], req['role']
    pins = fingerprint()
    if (req['recipe_schema'] != SCHEMA or req['parent_sha'] != PARENT or req['target_deg'] not in GOALS
            or role not in ('train','validate') or req['learn'] is not (role=='train')
            or type(req['independent_test']) is not bool or type(k) is not int or not 1<=k<=2000
            or type(req['policy_seed']) is not int or not 0<=req['policy_seed']<2**31
            or not isinstance(req['placo'],dict) or req['software'] != worker_software(pins)
            or req['asset_sha256'] != {n:h for n,h in pins.items() if not n.endswith('.py')}
            or req['parameters'] != [ZERO]*len(seeds) or not 2<=len(seeds)<=32
            or not all(type(s) is int for s in seeds) or len(seeds)!=len(set(seeds))):
        raise ValueError('Unexpected bimanual training contract')
    if role=='train':
        valid_seeds = not req['independent_test'] and seeds==block(TRAIN_BASE,k,len(seeds))
    elif req['independent_test']:
        valid_seeds = all(69_000_000<=s<71_000_000 for s in seeds)
    else:
        valid_seeds = set(seeds).issubset(block(VALID_BASE,k,16))
    if not valid_seeds:
        raise ValueError('Seeds do not belong to this wave')
    fresh = role=='train' and k==1
    ck, ck_sha = req['checkpoint'], req['checkpoint_sha']
    if fresh:
        if ck is not None or ck_sha is not None or req['target_deg']!=2:
            raise ValueError('Start the right-hand policy from scratch at +2°')
    elif not isinstance(ck,str) or not isinstance(ck_sha,str) or not re.fullmatch('[0-9a-f]{64}',ck_sha):
        raise ValueError('Expected a committed bimanual policy checkpoint')


def verify_wave(report, req):
    from natural_face90_bimanual_worker import verify_wave as verify
    return verify(report, req)


# Same project-wide GPU leases, per-attempt logs and atomic acceptance receipts.
_collect = _specialize(recipe.wave, [
    ("HERE/'natural_face90_recipe_worker.py'", "HERE/'natural_face90_bimanual_worker.py'"),
], dict(validate_request=validate_request,verify_result=verify_wave))


def wave(slot, req, fd):
    if (slot/'accepted.json').is_file():
        validate_request(req)
        return verify_wave(accepted_report(slot,req),req)
    return _collect(slot,req,fd)


def score(rows):
    safe = [row for row in rows if not row['unsafe']]
    return dict(recipe.stats(rows),
                bilateral_episodes=sum(row['bilateral_frames']>0 for row in safe),
                goal_episodes=sum(row['bimanual_goal_reached'] for row in safe),
                held_episodes=sum(row['bimanual_turn_held'] and row['success'] for row in safe),
                peak_bilateral_drive_deg=max((row['peak_bilateral_drive_deg'] for row in safe),default=0.))


def qualifies(goal, panel):
    if panel['unsafe'] or not panel['episodes']:
        return False
    if goal==2:
        return panel['goal_episodes']>=2
    return panel['held_episodes']*8>=7*panel['episodes']


def panel(folder,cfg,k,goal,ck,ck_sha,seeds,fd,independent=False):
    rows = []
    for i in range(0,len(seeds),cfg['lanes']):
        req=request(cfg,'validate',seeds[i:i+cfg['lanes']],k,goal,ck,ck_sha,independent)
        chunk,_=wave(folder/f'wave_{i:04d}',req,fd)
        rows.extend(chunk)
    return score(rows)


def initial():
    return dict(iteration=0,checkpoint=None,checkpoint_sha=None,target_deg=2,status='not_started')


def restore(home):
    state=read(home/'state.json')
    k=state['iteration']
    if k==0:
        if state not in (initial(),dict(initial(),status='running')):
            raise ValueError('Unexpected initial campaign state')
        return state
    folder=home/f'iteration_{k:04d}'
    if digest(folder/'result.json')!=state['result_sha']:
        raise ValueError('Committed iteration receipt changed')
    result=read(folder/'result.json')
    req=read(folder/'train'/'request.json')
    report=accepted_report(folder/'train',req)
    learn=read(report)['expert_learning']
    if (k!=result['iteration'] or result['learning']!=learn or result['next_goal']!=state['target_deg']
            or result['status']!=state['status'] or learn['output_checkpoint']!=state['checkpoint']
            or learn['output_checkpoint_sha']!=state['checkpoint_sha'] or digest(Path(state['checkpoint']))!=state['checkpoint_sha']):
        raise ValueError('State differs from its accepted bimanual checkpoint')
    return state


def campaign(home,cfg,fd):
    state=restore(home)
    if state['status'] not in ('not_started','running'):
        print('Campaign status:',state['status'],flush=True)
        return
    if state['status']=='not_started':
        state.update(status='running')
        save(home/'state.json',state)
    for k in range(state['iteration']+1,cfg['iterations']+1):
        goal=state['target_deg']
        folder=home/f'iteration_{k:04d}'
        req=request(cfg,'train',block(TRAIN_BASE,k,cfg['lanes']),k,goal,state['checkpoint'],state['checkpoint_sha'])
        rows,learn=wave(folder/'train',req,fd)
        ck,ck_sha=learn['output_checkpoint'],learn['output_checkpoint_sha']
        result=dict(iteration=k,target_deg=goal,training_rollouts_before_update=score(rows),learning=learn)
        if k==1 or k%cfg['validate_every']==0 or k==cfg['iterations']:
            valid=panel(folder/'validate',cfg,k,goal,ck,ck_sha,block(VALID_BASE,k,16),fd)
            result['validation_after_update']=valid
            passed=qualifies(goal,valid)
            if passed and goal in (15,90):
                seeds=recipe.final_seeds(home/f'goal_{goal}',ROOT/'runs/natural_face90_test_registry')
                test=panel(folder/'independent_test',cfg,k,goal,ck,ck_sha,seeds,fd,True)
                result['independent_test']=test
                passed=qualifies(goal,test)
                if not passed:
                    state['status']='independent_test_not_met'
            if passed:
                if goal==90:
                    state['status']='qualified_90deg_state_specialist'
                    save(home/'qualified_policy.json',dict(checkpoint=ck,sha=ck_sha,iteration=k,
                         target_deg=goal,results=result['independent_test'],simulation_only=True))
                else:
                    state['target_deg']=GOALS[GOALS.index(goal)+1]
            print(f"Frozen +{goal}°: {valid['bilateral_episodes']}/{valid['episodes']} bilateral, "
                  f"{valid['goal_episodes']} reached, {valid['held_episodes']} held, {valid['unsafe']} unsafe",flush=True)
        if k==cfg['iterations'] and state['status']=='running':
            state['status']='iteration_limit'
        result.update(next_goal=state['target_deg'],status=state['status'])
        save(folder/'result.json',result)
        state.update(iteration=k,checkpoint=ck,checkpoint_sha=ck_sha,result_sha=digest(folder/'result.json'))
        save(home/'state.json',state)
        print('Bimanual PPO',k,state['status'],'next target',state['target_deg'],flush=True)
        if state['status']!='running':
            break


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('check','prepare','long'))
    p.add_argument('--run-dir',type=Path,default=HOME_RUN)
    p.add_argument('--updates',type=int)
    p.add_argument('--lanes',type=int,choices=(4,8,16,32))
    p.add_argument('--validate-every',type=int)
    args=p.parse_args(argv)
    home=args.run_dir.resolve()
    if args.run_dir.is_symlink() or home.parent!=ROOT/'runs' or not re.fullmatch(r'r27_recipe_bimanual[A-Za-z0-9_-]*',home.name):
        p.error('Use a separate r27_recipe_bimanual directory under runs/')
    prior=read(home/'config.json') if (home/'config.json').is_file() else {}
    count=args.updates if args.updates is not None else prior.get('iterations',300)
    cadence=args.validate_every if args.validate_every is not None else prior.get('validate_every',5)
    lanes=args.lanes or prior.get('lanes',16)
    if not 1<=count<=2000 or not 1<=cadence<=2000:
        p.error('Invalid update count or validation interval')

    from natural_face90_v5_worker import inputs
    from natural_face90_placo_client import self_check
    from natural_face90_demo_usd import check_demo_usd
    from natural_face90_recipe_contacts import check_capacity_source
    from natural_face90_bimanual_policy import ACTION_AXES, ACTOR_DIM
    from natural_face90_bimanual_control import ACTION_AXES as CONTROL_AXES
    from natural_face90_bimanual_worker import verify_wave as worker_verifier
    if ACTION_AXES != CONTROL_AXES or not callable(worker_verifier):
        raise ValueError('Control and policy interfaces differ')
    source,*_=inputs(dict(parent_sha=PARENT,role='train',seeds=block(TRAIN_BASE,1,lanes),
                          parameters=[ZERO]*lanes,target_deg=15))
    cfg=dict(schema=SCHEMA,rng=327,parent_sha=PARENT,source_sha256=fingerprint(),placo=self_check()['receipt'],
             scene=check_demo_usd(source),raw_contact_capacity=check_capacity_source(lanes),
             actor_dim=ACTOR_DIM,action_dim=len(ACTION_AXES),action_axes=list(ACTION_AXES),
             curriculum=list(GOALS),validation_seeds=16,iterations=count,validate_every=cadence,lanes=lanes)
    validate_request(request(cfg,'train',block(TRAIN_BASE,1,lanes),1,2,None,None))
    if args.mode=='check':
        print('CPU CHECK OK: R27, mounted USD, Placo, batched contact sources and bimanual worker',flush=True)
        return
    home.mkdir(exist_ok=True)
    with recipe.exclusive(home/'campaign.lock') as fd:
        if (home/'config.json').exists():
            if read(home/'config.json')!=cfg:
                raise ValueError('Run directory has different code or settings; select a fresh --run-dir')
        else:
            if set(x.name for x in home.iterdir())!={'campaign.lock'}:
                raise ValueError('Selected directory contains another experiment')
            save(home/'config.json',cfg)
        if not (home/'state.json').is_file():
            save(home/'state.json',initial())
        if args.mode=='prepare':
            print('Bimanual PPO configuration verified:',home,read(home/'state.json')['status'],flush=True)
            return
        campaign(home,cfg,fd)


if __name__=='__main__':
    main()
