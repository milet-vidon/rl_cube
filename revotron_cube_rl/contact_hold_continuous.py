"""Collect teacher queries, update the student, and measure physical success until a stopping rule."""
import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from contact_hold_dataset import sha
from contact_hold_loop_data import ROOT, INITIAL, INITIAL_SHA, checkpoint, collect, inspect, read, write
from contact_hold_loop_fit import FILES

LOCKS=ROOT.parent/'work/teacher_optimization_20261103'
DEFAULT=dict(collection_seeds=[11601,11603],validation_seeds=[21611,21612],test_seeds=[32611,32612,32613],
             updates=500,batch_size=65536,lr=5e-5,patience=6,confirmations=2,target=1.)
DONE=('confirmed_on_test','heldout_target_not_met','plateau')
# Keep the per-campaign lock held by an active child even if its launcher dies.
CHILD_LOCKS=()


def seeds(arg):
    v=[int(s) for s in arg.split(',')]
    if not v or len(set(v))!=len(v) or any(n<0 for n in v):
        raise argparse.ArgumentTypeError('Use unique nonnegative seeds')
    return v


def validate(cfg):
    a,b,c=(set(cfg[k]) for k in ('collection_seeds','validation_seeds','test_seeds'))
    if not all((a,b,c)) or (a|{11601})&(b|c) or b&c:
        raise ValueError('Training (including legacy 11601), validation and final-test seeds must be disjoint')
    if (min(cfg['updates'],cfg['patience'],cfg['confirmations'])<1 or cfg['batch_size']<4 or cfg['batch_size']%4
            or not 0<cfg['lr']<=1 or not 0<cfg['target']<=1):
        raise ValueError('Invalid fit parameters or stopping rule')


def panel(records):
    if not records or len({r['seed'] for r in records})!=len(records) or len({r['checkpoint'] for r in records})!=1:
        raise ValueError('Expected unique seeds and one frozen model per panel')
    return dict(rate=sum(r['passed'] for r in records)/len(records),
                score=[sum(r['passed'] for r in records),sum(r['safe'] for r in records),sum(r['phase_rank'] for r in records),
                       sum(int(10*r['hold_s']) for r in records)],
                phases=dict(Counter(r['phase'] for r in records)),episodes=records)


def advance(state, val, fit, cfg):
    state['round']+=1
    state['current']=str(fit)
    if val['score']>state['best_score']:
        state.update(best_score=val['score'],champion=str(fit),stale=0)
    else:
        state['stale']+=1
    state['streak']=state['streak']+1 if val['rate']>=cfg['target'] else 0
    if state['streak']>=cfg['confirmations']:
        state['status']='final_test'
    elif val['rate']<cfg['target'] and state['stale']>=cfg['patience']:
        state['status']='plateau'
    else:
        state['status']='running'


def lock(file):
    f=open(file,'a')
    try:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        f.close();raise
    return f


@contextmanager
def gpu_slot(fitting):
    announced=False
    while True:
        with ExitStack() as stack:
            try:
                fs=([stack.enter_context(lock(LOCKS/n)) for n in ('contact_hold_gpu_training.lock','isaac_gui_evaluation.lock')]
                    if fitting else [])
                busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
            except BlockingIOError:
                busy='another contact-hold process'
            if not busy:
                yield [f.fileno() for f in fs];return
        if not announced:
            print('WAIT_GPU',busy,flush=True);announced=True
        time.sleep(15)


def evaluate(slot, fit, seed, role):
    slot=Path(slot);slot.mkdir(parents=True,exist_ok=True)
    pointer=slot/'attempt.json'
    while True:
        if pointer.exists():
            old=read(pointer);run=ROOT/'runs'/old['name']
            if (run/'artifact_manifest.json').exists():
                # An interrupted Isaac run may still have a complete manifest.
                m=read(run/'artifact_manifest.json')
                if m['full_episode']:
                    return inspect(run,fit,seed,role)[0]
        with gpu_slot(False):
            if shutil.disk_usage(ROOT/'runs').free<3*1024**3:
                raise RuntimeError('Less than 3 GiB free; clear space and resume')
            name=f'contact_hold_student_eval_{seed}_loop_{uuid.uuid4().hex}'
            write(slot/f'attempt_{name}.json',dict(name=name))
            write(pointer,dict(name=name))
            env=dict(os.environ,CONTACT_HOLD_EVAL_ROLE=role,CONTACT_HOLD_EVAL_NAME=name)
            print('EVALUATE',role,seed,fit,flush=True)
            rc=subprocess.run(['bash',str(ROOT/'run_contact_hold_evaluation.sh'),str(fit),str(seed)],
                              env=env,pass_fds=CHILD_LOCKS).returncode
        if rc==75: time.sleep(15);continue
        if rc: raise RuntimeError('Isaac exited with status '+str(rc)+'; resume to retry this slot')
        if not (ROOT/'runs'/name/'artifact_manifest.json').is_file():
            raise RuntimeError('Isaac exited without a complete evaluation')
        return inspect(ROOT/'runs'/name,fit,seed,role)[0]


def run_panel(path,fit,seeds_,role):
    v=panel([evaluate(path/str(seed),fit,seed,role) for seed in seeds_])
    write(path/'panel.json',v)
    print('PANEL',role,{'rate':v['rate'],'phases':v['phases'],'fit':str(fit)},flush=True)
    return v


def fit_block(planfile):
    out=planfile.parent
    if not (out/'result.json').exists():
        with gpu_slot(True) as fds:
            subprocess.run([sys.executable,'-u',str(ROOT/'contact_hold_loop_fit.py'),str(planfile)],
                           check=True,pass_fds=(*CHILD_LOCKS,*fds))
    checkpoint(out)
    if read(out/'experiment.json')['continuous_plan_sha256']!=sha(planfile):
        raise ValueError('Fit no longer matches its plan')


def ensure_unseen(seeds_, allowed=()):
    # Do not reuse an earlier logged contact-hold reset as a new held-out test.
    allowed=set(allowed)
    for seed in seeds_:
        for path in (ROOT/'runs').glob(f'contact_hold_student_eval_{seed}_*'):
            if path.is_dir() and path.name not in allowed:
                raise ValueError(f'Final test seed {seed} already used in {path}; choose unused seeds for a new campaign')


def drive(home,state,cfg,limit):
    frozen_code=read(home/'config.json')['implementation']
    def save():
        if any(sha(ROOT/n)!=h for n,h in frozen_code.items()):
            raise RuntimeError('Training implementation changed during campaign')
        write(home/'state.json',state)
    if state['status'] in DONE: return
    if state['best_score'] is None:
        b=run_panel(home/'baseline',INITIAL,cfg['validation_seeds'],'validation')
        state.update(best_score=b['score'],champion=str(INITIAL),streak=int(b['rate']>=cfg['target']))
        if state['streak']>=cfg['confirmations']: state['status']='final_test'
        save()
    while state['status']!='final_test' and state['round']<limit:
        k=state['round']+1
        out=home/f'round_{k:04d}'
        fit=out/'fit';fit.mkdir(parents=True,exist_ok=True)
        parent=Path(state['current'])
        seed=cfg['collection_seeds'][(k-1)%len(cfg['collection_seeds'])]
        state.update(status='running',stage=f'collect round {k} seed {seed}');save()
        r=evaluate(out/'collection',parent,seed,'collection')
        corpus=list(state['corpus'])
        if not any((x['source']['checkpoint'],x['source']['seed'])==(r['checkpoint'],seed) for x in corpus):
            review=collect(r,out/'corrections.npz',cfg['collection_seeds'])
            if review is not None:
                corpus.append(review)
                write(out/'review.json',review)
        if not corpus:
            # Try the rest of the collection panel if the first episode failed before a safe labelled state.
            for other in cfg['collection_seeds']:
                if other==seed: continue
                rr=evaluate(out/f'collection_{other}',parent,other,'collection')
                review=collect(rr,out/f'corrections_{other}.npz',cfg['collection_seeds'])
                if review:
                    corpus.append(review);break
            if not corpus: raise RuntimeError('No eligible transitions in collection panel; inspect the archived failures')
        plan=dict(round=k,parent=str(parent),parent_sha=checkpoint(parent),corpus=corpus,
                  **{key:cfg[key] for key in ('updates','batch_size','lr','collection_seeds')})
        planfile=fit/'plan.json'
        if planfile.exists() and read(planfile)!=plan: raise ValueError('Round plan changed since interruption')
        if not planfile.exists(): write(planfile,plan)
        state['stage']=f'fit round {k}';save()
        fit_block(planfile)
        state['stage']=f'validate round {k}';save()
        val=run_panel(out/'validation',fit,cfg['validation_seeds'],'validation')
        advance(state,val,fit,cfg)
        state.update(corpus=corpus,stage='round complete',last_rate=val['rate'])
        write(out/'round.json',dict(round=k,fit=str(fit),collection=r,validation=val,champion=state['champion']))
        save()
        if state['status']=='plateau': return
    if state['status']=='final_test':
        frozen=Path(state['champion'])
        state['stage']='final test';save()
        # Previous test attempts in this campaign are the only resumable ones.
        allowed=[read(p)['name'] for p in (home/'final').glob('*/attempt*.json')]
        ensure_unseen(cfg['test_seeds'],allowed)
        final=run_panel(home/'final',frozen,cfg['test_seeds'],'test')
        state.update(status='confirmed_on_test' if final['rate']>=cfg['target'] else 'heldout_target_not_met',
                     stage='finished',heldout_rate=final['rate'],frozen_checkpoint_sha256=checkpoint(frozen))
    else:
        state.update(status='max_rounds',stage='round limit')
    save()


def main():
    global CHILD_LOCKS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--status',action='store_true',help='print current stage, champion and latest validation rate, then exit')
    p.add_argument('--max-rounds',type=int,help='total completed rounds, extendable on resume (default 30)')
    for key in DEFAULT:
        typ=seeds if key.endswith('seeds') else type(DEFAULT[key])
        p.add_argument('--'+key.replace('_','-'),type=typ)
    a=p.parse_args()
    home=a.run_dir.resolve()
    if a.status:
        if not (home/'state.json').exists(): p.error('No campaign state yet')
        import json
        s=read(home/'state.json')
        print(json.dumps({k:s.get(k) for k in ('status','stage','round','max_rounds','champion','last_rate','heldout_rate','stale','streak')},indent=2))
        return
    if a.max_rounds is not None and a.max_rounds<1: p.error('max-rounds must be positive')
    given={k:getattr(a,k) for k in DEFAULT if getattr(a,k) is not None}
    if a.resume:
        if not (home/'config.json').is_file(): p.error('No such campaign')
    else:
        validate(DEFAULT|given)
        ensure_unseen((DEFAULT|given)['test_seeds'])
        home.mkdir(parents=True,exist_ok=False)
    with lock(home/'campaign.lock') as campaign_lock:
        CHILD_LOCKS=(campaign_lock.fileno(),)
        manifest={n:sha(ROOT/n) for n in FILES}
        if not a.resume:
            if checkpoint(INITIAL)!=INITIAL_SHA: raise ValueError('Initial round-two weights changed')
            write(home/'config.json',dict(config=DEFAULT|given,implementation=manifest,initial_sha=INITIAL_SHA,
                                          initial_max_rounds=a.max_rounds or 30))
        saved=read(home/'config.json');cfg=saved['config']
        if manifest!=saved['implementation'] or any(v!=cfg[k] for k,v in given.items()):
            raise ValueError('Frozen campaign code/config changed; use a new run directory')
        default_state=dict(status='running',stage='baseline',round=0,current=str(INITIAL),champion=str(INITIAL),
                           best_score=None,stale=0,streak=0,corpus=[],max_rounds=saved['initial_max_rounds'])
        state=read(home/'state.json') if (home/'state.json').exists() else default_state
        limit=a.max_rounds or state['max_rounds']
        if limit<state['round']: p.error('max-rounds is less than completed rounds')
        state['max_rounds']=limit
        write(home/'state.json',state)
        print('CAMPAIGN',home,'RESUME_WITH: bash run_contact_hold_continuous.sh --run-dir',home,'--resume',flush=True)
        drive(home,state,cfg,limit)
        print('STOP',state['status'],'champion=',state['champion'],flush=True)


if __name__=='__main__':
    main()
