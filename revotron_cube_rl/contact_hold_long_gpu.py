"""Long blocks of CUDA BC with audited multi-seed demonstrations and cumulative DAgger queries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import contact_hold_continuous as loop
from contact_hold_dataset import audit_demo, training_pairs, sha, verified_json
from contact_hold_loop_data import ROOT, INITIAL, INITIAL_SHA, checkpoint, collect, read, write
from contact_hold_loop_fit import verify_collection
from contact_hold_policy import restored
from train_contact_hold_bc import atomic_save

RUNS=ROOT/'runs'
BASE=RUNS/'contact_hold_pair15_pos64_11601_20261107_120008_511634810'
STAGES=tuple(range(11603,11621))
VAL=[21611,21612]
TEST=[32621,32622,32623]
FILES=(*loop.FILES,Path(__file__).name,'contact_hold_experiment.py','contact_hold_teacher.py')
ALLOWED=[11601,*STAGES]


def qualify(home):
    found={11601:BASE}
    for seed in STAGES:
        complete=False
        for p in sorted(RUNS.glob(f'contact_hold_teacher_{seed}_*')):
            if not (p/'result.json').is_file():continue
            r=read(p/'result.json')
            if r['seed']!=seed or r['split']!='development' or r['execution_interrupted'] or not r['full_episode']:continue
            complete=True
            if r['dynamic_grasp_passed'] and r['failure'] is None:
                audit_demo(p); found[seed]=p;break
        if len(found)>=3:break
        if not complete:
            with loop.gpu_slot(False):
                name=f'contact_hold_teacher_{seed}_{time.time_ns()}'
                with (home/(name+'.log')).open('x') as f:
                    print('TEACHER',seed,'log:',f.name,flush=True)
                    subprocess.run([sys.executable,'-u',str(ROOT/'contact_hold_experiment.py'),'--name',name,
                        '--seed',str(seed),'--position-iterations','64','--opposing-pair-stiffness','15'],
                        stdout=f,stderr=subprocess.STDOUT,check=True,pass_fds=loop.CHILD_LOCKS)
                p=RUNS/name;r=read(p/'result.json')
                if not r['full_episode'] or r['execution_interrupted']:raise ValueError('Incomplete teacher evaluation')
                if r['dynamic_grasp_passed'] and r['failure'] is None:
                    audit_demo(p);found[seed]=p
                print('TEACHER_RESULT',seed,r['failure'],flush=True)
        if len(found)>=3:break
    if len(found)<3:raise RuntimeError('Not enough accepted teacher seeds; no optimization started')
    return [audit_demo(p)[-1] for p in found.values()]


def historic():
    records={}
    for d in ('contact_hold_auto','contact_hold_ready_diverse_v1'):
        for c in read(RUNS/d/'state.json')['corpus']:
            verify_collection(c,ALLOWED)
            records[c['source']['manifest']]=c
    return list(records.values())


def tensors(plan, meta,device):
    demo_seeds=[s['seed'] for s in plan['demos']]
    if (len(set(demo_seeds))<3 or len(set(demo_seeds))!=len(demo_seeds) or
            11601 not in demo_seeds or not set(demo_seeds)<=set(ALLOWED)):
        raise ValueError('Need at least three distinct accepted development seeds including 11601')
    arrays=[];contract0=None
    for s in plan['demos']:
        p=Path(s['path'])
        runtime=verified_json(p,'result.json',read(p/'artifact_manifest.json'))
        if runtime['split']!='development':raise ValueError('A held-out rollout cannot enter training')
        r,n,i,contract,a=audit_demo(p)
        if (a!=s or n!=meta['joint_names'] or i!=meta['output_indices'] or
            a['candidate']!=meta['sources'][0]['candidate'] or (contract0 is not None and contract0!=contract)):
            raise ValueError('Incompatible or changed teacher')
        contract0=contract
        x,y,val=training_pairs(r,i); arrays.append((x,y,np.ones_like(y,dtype=bool),val))
    corrections=[]
    for c in plan['corpus']:
        verify_collection(c,ALLOWED)
        with np.load(c['path'],allow_pickle=False) as a:
            corrections.append(tuple(a[k] for k in ('x','y','mask','val')))
    result=[]
    for group in (arrays,corrections):
        x,y,m,v=[np.concatenate(items) for items in zip(*group)]
        if not v.any() or v.all() or not m.any(axis=1).all(): raise ValueError('Invalid split or mask')
        result.append([tuple(torch.from_numpy(a[sel]).to(device) for a in (x,y,m)) for sel in (~v,v)])
    return result


def fit(pfile,device='cuda:0'):
    p=read(pfile);out=pfile.parent;sig=sha(pfile);parent=Path(p['parent'])
    if checkpoint(parent)!=p['parent_sha']:raise ValueError('Parent changed')
    ck=torch.load(parent/'best.pt',weights_only=True,map_location='cpu');meta=ck['metadata']
    for n,h in (meta['source_sha256']|p['code']).items():
        if sha(ROOT/n)!=h:raise ValueError('Changed training code: '+n)
    groups=tensors(p,meta,device)
    m=restored(ck).to(device)
    for group in groups:
        for _,y,mask in group:
            if torch.any(mask & ((y<m.lower-1e-6)|(y>m.upper+1e-6)|~torch.isfinite(y))):
                raise ValueError('Target outside policy bounds')
    opt=torch.optim.AdamW(m.parameters(),lr=p['lr'],weight_decay=1e-5)
    gen=torch.Generator(device=device).manual_seed(20261110)
    base=int(meta.get('long_gpu_updates',0)); start=0
    def restore(path):
        s=torch.load(path,weights_only=True,map_location='cpu')
        m.load_state_dict(s['model']); opt.load_state_dict(s['optimizer']);gen.set_state(s['rng'])
        return s
    if (out/'last.pt').exists():
        s=restore(out/'last.pt')
        if s['plan']!=sig or s['device']!=device:raise ValueError('Incompatible resume')
        start=s['step']
    elif base:
        if sha(parent/'last.pt')!=read(parent/'result.json')['last_sha256']:raise ValueError('Parent optimizer changed')
        s=restore(parent/'last.pt')
        if s['plan']!=meta['plan_sha'] or s['device']!=device:raise ValueError('Bad parent optimizer lineage')
    def loss(pred,y,mask):
        return ((((pred-y)/m.scale).square()*mask).sum(1)/mask.sum(1)).mean()
    tic=time.monotonic(); mb=p['batch']//2
    print('GPU_FIT_START',base+start,'target',base+p['updates'],'batch',p['batch'],flush=True)
    for k in range(start+1,p['updates']+1):
        ids=[torch.randint(len(tr[0]),(mb,),generator=gen,device=device) for tr,_ in groups]
        preds=m(torch.cat([tr[0][j] for (tr,_),j in zip(groups,ids)])).split(mb)
        l=sum(loss(pred,tr[1][j],tr[2][j]) for pred,(tr,_),j in zip(preds,groups,ids))
        opt.zero_grad(set_to_none=True);l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        if k%250==0 or k==p['updates']:
            with torch.no_grad(): vals=[float(loss(m(x),y,mask)) for _,(x,y,mask) in groups]
            if not np.isfinite(vals).all(): raise ValueError('Nonfinite validation')
            elapsed=time.monotonic()-tic
            print('GPU_FIT',base+k,'updates/s',round((k-start)/elapsed,2),'val',vals,flush=True)
            atomic_save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=gen.get_state(),
                             step=k,device=device,plan=sig),out/'last.pt')
    end=base+p['updates']
    meta=dict(meta,sources=p['demos'],source_sha256=meta['source_sha256']|p['code'],plan_sha=sig,
              algorithm='multiseed_teacher_BC_and_online_corrections',long_gpu_updates=end,
              training_seeds=sorted({s['seed'] for s in p['demos']}|{c['source']['seed'] for c in p['corpus']}),
              updates=p['updates'],batch_size=p['batch'],learning_rate=p['lr'],device=device,
              parent_run=str(parent),parent_checkpoint_sha256=p['parent_sha'],optimizer_reset=not bool(base),
              group_loss_weights=[1.,1.],training_samples=sum(len(tr[0]) for tr,_ in groups),
              validation_samples=sum(len(v[0]) for _,v in groups),checkpoint_selection='round_end_then_PhysX_panel')
    meta.pop('correction_reviews',None)
    meta['validation_kind']='temporal_blocks_of_seen_development_trajectories'
    atomic_save(dict(model={n:t.cpu() for n,t in m.state_dict().items()},metadata=meta,metrics={'update':end}),out/'best.pt')
    frozen=restored(torch.load(out/'best.pt',map_location='cpu',weights_only=True))
    with torch.no_grad():
        for _,(x,_,_) in groups:torch.testing.assert_close(frozen(x.cpu()),m(x).cpu(),rtol=1e-5,atol=1e-5)
    write(out/'experiment.json',meta)
    write(out/'result.json',dict(completed=True,checkpoint_reload_verified=True,checkpoint_sha256=sha(out/'best.pt'),
          last_sha256=sha(out/'last.pt'),best_update=end,updates=p['updates'],global_update=end,closed_loop_success_demonstrated=False))


def campaign(home, cfg, resume):
    if not resume:home.mkdir(parents=True,exist_ok=False)
    with loop.lock(home/'campaign.lock') as f:
        loop.CHILD_LOCKS=(f.fileno(),)
        code={n:sha(ROOT/n) for n in FILES}
        if not resume:
            if checkpoint(INITIAL)!=INITIAL_SHA:raise ValueError('Unexpected starting checkpoint')
            loop.ensure_unseen(TEST)
            write(home/'settings.json',dict(cfg=cfg,code=code))
        saved=read(home/'settings.json')
        if saved['code']!=code:raise ValueError('Campaign code changed')
        cfg=saved['cfg'];statefile=home/'state.json'
        if statefile.exists():s=read(statefile)
        else:
            demos=qualify(home)
            s=dict(round=0,status='running',current=str(INITIAL),champion=str(INITIAL),stale=0,streak=0,
                   corpus=historic(),demos=demos,best_score=None)
            write(statefile,s)
        if s['status'] in loop.DONE:return print('STOP',s['status'],s['champion'])
        if s['best_score'] is None:
            panel=loop.run_panel(home/'baseline',INITIAL,VAL,'validation')
            s['best_score']=panel['score'];write(statefile,s)
        while s['round']<cfg['rounds'] and s['status']!='final_test':
            k=s['round']+1;d=home/f'round_{k:03d}';out=d/'fit';out.mkdir(exist_ok=True,parents=True)
            parent=Path(s['current']);seed=cfg['collection'][(k-1)%len(cfg['collection'])]
            pf=out/'plan.json'
            if not pf.exists():
                rec=loop.evaluate(d/'collection',parent,seed,'collection');corp=list(s['corpus'])
                if not any((c['source']['checkpoint'],c['source']['seed'])==(rec['checkpoint'],seed) for c in corp):
                    extra=collect(rec,d/'corrections.npz',ALLOWED)
                    if extra:corp.append(extra)
                write(pf,dict(parent=str(parent),parent_sha=checkpoint(parent),code=code,demos=s['demos'],
                              corpus=corp,batch=cfg['batch'],updates=cfg['updates'],lr=cfg['lr']))
            plan=read(pf)
            if (plan['parent_sha']!=checkpoint(parent) or plan['parent']!=str(parent) or plan['code']!=code or
                    plan['demos']!=s['demos'] or plan['corpus'][:len(s['corpus'])]!=s['corpus']):
                raise ValueError('Changed round plan or resume lineage')
            if not (out/'result.json').exists():
                with loop.gpu_slot(True) as fds:
                    subprocess.run([sys.executable,'-u',__file__,'--fit',str(pf)],check=True,pass_fds=(*loop.CHILD_LOCKS,*fds))
            checkpoint(out)
            if read(out/'experiment.json')['plan_sha']!=sha(pf):raise ValueError('Fit identity mismatch')
            panel=loop.run_panel(d/'validation',out,VAL,'validation')
            loop.advance(s,panel,out,cfg);s.update(last_rate=panel['rate'],corpus=read(pf)['corpus'])
            write(statefile,s)
            if s['status']=='plateau':break
        if s['status']=='final_test':
            loop.ensure_unseen(TEST,[read(p)['name'] for p in (home/'final').glob('*/attempt*.json')])
            v=loop.run_panel(home/'final',Path(s['champion']),TEST,'test')
            s.update(status='confirmed_on_test' if v['rate']>=cfg['target'] else 'heldout_target_not_met',heldout_rate=v['rate'])
        elif s['status']!='plateau':s['status']='max_rounds'
        write(statefile,s);print('STOP',s['status'],'champion',s['champion'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--rounds',type=int,default=40);p.add_argument('--updates',type=int,default=5000)
    p.add_argument('--batch-size',type=int,default=32768);p.add_argument('--patience',type=int,default=12)
    p.add_argument('--fit',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.fit:fit(a.fit)
    else:
        if not a.run_dir or min(a.rounds,a.updates,a.patience)<1 or a.batch_size<2 or a.batch_size%2:p.error('Bad run directory or counts')
        campaign(a.run_dir.resolve(),dict(rounds=a.rounds,updates=a.updates,batch=a.batch_size,patience=a.patience,
                 lr=5e-5,collection=[11603,11604,11605,11606,11601,11607,11608,11609],target=1.,confirmations=2),a.resume)
