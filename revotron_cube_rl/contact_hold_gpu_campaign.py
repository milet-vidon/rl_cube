"""CUDA multi-environment pickup data collection, BC updates, original full-task validation."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

import contact_hold_continuous as loop
from contact_hold_dataset import sha
from contact_hold_loop_data import ROOT, checkpoint, inspect, read, write
from contact_hold_long_gpu import tensors
from contact_hold_policy import restored
from train_contact_hold_bc import atomic_save

BASE=ROOT/'runs/contact_hold_long_v1/round_039/fit'
DIGEST='8468adccd589d2e84634b27f65d6563f7287618c9b901f01a43e06a4f3441f58'
FILES=('contact_hold_gpu_campaign.py','contact_hold_gpu_sample.py','contact_hold_gpu_scene.py','run_contact_hold_gpu_parallel.sh')
VAL=(21611,21612)
TEST=(51611,51612,51613)


def identity():
    return {k:sha(ROOT/k) for k in FILES}


def verify_batch(entry):
    path=Path(entry['path']);planpath=path.parent/'collect.json';p=read(planpath)
    if sha(path/'report.json')!=entry['sha'] or sha(planpath)!=entry['plan_sha']:raise ValueError('Changed rollout receipt')
    r=read(path/'report.json')
    if not (r['schema']=='cuda_wrist_query_v1' and r['scope']=='reset_to_wrist_hold_or_termination' and r['plan_sha256']==entry['plan_sha']
            and r['checkpoint']==p['parent_sha']==checkpoint(Path(p['parent']))
            and r['num_envs']==len(p['seeds'])>=2 and r['seeds']==p['seeds']
            and len(set(r['seeds']))==len(r['seeds']) and all(41000<=s<50000 for s in r['seeds'])
            and r['student_start_phase']==[(0,4,0,5)[j%4] for j in range(r['num_envs'])]
            and r['physics_device']=='cuda:0' and r['role']=='collection' and r['reset_count_per_lane']==1
            and r['eligible_success_demonstration'] is False and not r['full_episode'] and r['cross_lane_contacts']==0
            and r['physics_hz']==240 and r['control_hz']==30 and r['action_dim']==56 and r['solver']['enableGPUDynamics']
            and r['solver']['broadphaseType']=='GPU' and r['solver']['solverType']=='TGS'
            and r['solver']['minPositionIterationCount']==64 and r['solver']['minVelocityIterationCount']==4
            and sha(path/'batch.npz')==r['data_sha256'] and p['code']==identity()):
        raise ValueError('Ineligible GPU collection')
    with np.load(path/'batch.npz',allow_pickle=False) as data:
        a={k:data[k] for k in ('x','y','mask','val','lane','step','phase')}
    n=r['samples']
    if (a['x'].shape!=(n,198) or a['y'].shape!=(n,28) or a['mask'].shape!=(n,28) or
            a['mask'].dtype!=bool or a['val'].dtype!=bool or not a['mask'].any(1).all() or
            any(a[k].shape!=(n,) for k in ('val','lane','step','phase')) or not np.isfinite(a['x']).all() or
            not np.isfinite(a['y']).all() or not ((0<=a['phase'])&(a['phase']<10)&(a['step']>0)&
            (a['lane']>=0)&(a['lane']<r['num_envs'])).all() or
            (a['phase']<np.asarray(r['student_start_phase'])[a['lane']]).any() or not np.array_equal(a['x'][:,-13:-1].argmax(1),a['phase'])):
        raise ValueError('Malformed or wrongly aligned GPU labels')
    return a,r


def fit(pf):
    plan=read(pf);out=pf.parent;sig=sha(pf);parent=Path(plan['parent'])
    if checkpoint(BASE)!=DIGEST or checkpoint(parent)!=plan['parent_sha'] or identity()!=plan['code']:
        raise ValueError('Changed fit lineage')
    c=torch.load(parent/'best.pt',weights_only=True,map_location='cpu');meta=c['metadata']
    for n,h in meta['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Changed parent dependency '+n)
    oldplan=read(BASE/'plan.json')
    if sha(BASE/'plan.json')!=read(BASE/'experiment.json')['plan_sha']:raise ValueError('Changed legacy dataset')
    groups=tensors(oldplan,meta,'cuda:0')
    vectors=[]; extra_seeds=set()
    for entry in plan['vector_corpus']:
        a,r=verify_batch(entry);vectors.append(a);extra_seeds.update(r['seeds'])
    a={k:np.concatenate([v[k] for v in vectors]) for k in ('x','y','mask','val','phase')}
    dev='cuda:0'
    # Split BOTH validation and optimization before oversampling contact-phase transitions.
    for sel in (np.ones_like(a['val']),(a['phase']==4)|(a['phase']==5)):
        if not (sel&~a['val']).any():continue
        groups.append([tuple(torch.from_numpy(a[k][sel&v]).to(dev) for k in ('x','y','mask')) for v in (~a['val'],a['val'])])
    if len(groups)<3:raise ValueError('No trainable parallel data')
    m=restored(c).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=5e-5,weight_decay=1e-5)
    for group in groups:
        for _,y,mask in group:
            if (mask&((y<m.lower-1e-6)|(y>m.upper+1e-6))).any():raise ValueError('Label outside policy bounds')
    gen=torch.Generator(device=dev)
    last=out/'last.pt';start=0;base=meta['long_gpu_updates']
    pth=last if last.exists() else parent/'last.pt'
    if pth!=last and sha(pth)!=read(parent/'result.json')['last_sha256']:raise ValueError('Parent optimizer changed')
    s=torch.load(pth,weights_only=True,map_location='cpu')
    if s['plan']!=(sig if pth==last else meta['plan_sha']) or s['device']!=dev:raise ValueError('Bad restart state')
    if pth==last:start=s['step']
    m.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);gen.set_state(s['rng'])
    b=plan['batch'];sizes=[b//2,b//4,b//8,b//8] if len(groups)==4 else [b//2,b//4,b//4]
    def loss(pred,y,mask):return ((((pred-y)/m.scale).square()*mask).sum(-1)/mask.sum(-1)).mean()
    tr=[g[0] for g in groups]
    for step in range(start+1,plan['updates']+1):
        ids=[torch.randint(len(x),(size,),device=dev,generator=gen) for (x,_,_),size in zip(tr,sizes)]
        parts=m(torch.cat([x[i] for (x,_,_),i in zip(tr,ids)])).split(sizes)
        losses=[loss(pred,y[i],mask[i]) for pred,(_,y,mask),i in zip(parts,tr,ids)]
        objective=sum(l*size/b for l,size in zip(losses,sizes))
        opt.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%250==0 or step==plan['updates']:
            with torch.inference_mode():
                scores=[float(loss(m(x),y,mask)) if len(x) else None for _,(x,y,mask) in groups]
            if any(v is not None and not math.isfinite(v) for v in scores):raise ValueError('Invalid validation')
            print('GPU_FIT',base+step,scores,flush=True)
            atomic_save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=gen.get_state(),step=step,plan=sig,device=dev),last)
    end=base+plan['updates']
    meta=dict(meta,source_sha256=meta['source_sha256']|identity(),plan_sha=sig,algorithm='GPU_vector_pickup_DAgger_BC',
              parent_run=str(parent),parent_checkpoint_sha256=plan['parent_sha'],long_gpu_updates=end,optimizer_reset=False,
              batch_size=b,updates=plan['updates'],group_loss_weights=[n/b for n in sizes],training_seeds=sorted(set(meta['training_seeds'])|extra_seeds),
              training_samples=sum(len(g[0][0]) for g in groups[:3]),validation_samples=sum(len(g[1][0]) for g in groups[:3]))
    write(out/'experiment.json',meta)
    atomic_save(dict(metadata=meta,model={k:v.cpu() for k,v in m.state_dict().items()},metrics=dict(update=end)),out/'best.pt')
    reloaded=restored(torch.load(out/'best.pt',weights_only=True,map_location='cpu'))
    with torch.inference_mode():
        for _,(x,_,_) in groups:
            for chunk in x.split(4096):
                if len(chunk):torch.testing.assert_close(m(chunk).cpu(),reloaded(chunk.cpu()),atol=1e-5,rtol=1e-5)
    write(out/'result.json',dict(completed=True,checkpoint_reload_verified=True,checkpoint_sha256=sha(out/'best.pt'),
          last_sha256=sha(last),best_update=end,global_update=end,updates=plan['updates'],device=dev,closed_loop_success_demonstrated=False))


def campaign(home,cfg,resume):
    if not resume:
        loop.ensure_unseen(TEST)
        home.mkdir(parents=True,exist_ok=False)
    with loop.lock(home/'campaign.lock') as fd:
        loop.CHILD_LOCKS=(fd.fileno(),)
        if checkpoint(BASE)!=DIGEST:raise ValueError('Champion is not the measured round-39 checkpoint')
        if not resume:write(home/'settings.json',dict(cfg=cfg,code=identity(),base=DIGEST))
        settings=read(home/'settings.json')
        if settings['code']!=identity():raise ValueError('Code changed since campaign creation')
        cfg=settings['cfg'];file=home/'state.json'
        if file.exists():s=read(file)
        else:
            basepanel=read(BASE.parent/'validation/panel.json')
            if [e['seed'] for e in basepanel['episodes']]!=list(VAL):raise ValueError('Wrong baseline seeds')
            score=loop.panel([inspect(Path(e['run']),BASE,e['seed'],'validation')[0] for e in basepanel['episodes']])['score']
            s=dict(current=str(BASE),champion=str(BASE),status='running',round=0,best_score=score,stale=0,streak=0,vector_corpus=[])
            write(file,s)
        if s['status'] in loop.DONE:return print('STOP',s['status'],s['champion'])
        while s['round']<cfg['rounds'] and s['status']!='final_test':
            k=s['round']+1;d=home/f'round_{k:03}';out=d/'fit';out.mkdir(parents=True,exist_ok=True)
            parent=Path(s['current']);n=cfg['envs']; cp=d/'collect.json'
            seeds=list(range(41000+(k-1)*n,41000+k*n))
            plan=dict(parent=str(parent),parent_sha=checkpoint(parent),code=identity(),seeds=seeds)
            if cp.exists() and read(cp)!=plan:raise ValueError('Sampler resume mismatch')
            if not cp.exists():write(cp,plan)
            ptr=d/'attempt.json'
            dest=d/read(ptr)['name'] if ptr.exists() else None
            if dest is None or not (dest/'report.json').exists():
                if shutil.disk_usage(home).free<3*1024**3:raise RuntimeError('Not enough free disk space; free space then resume')
                dest=d/('samples_'+uuid.uuid4().hex);write(ptr,dict(name=dest.name))
                print('GPU_SAMPLE',k,n,seeds,flush=True)
                s['stage']=f'collect {k}';write(file,s)
                with loop.gpu_slot(True) as fds, (d/(dest.name+'.log')).open('x') as log:
                    subprocess.run([sys.executable,'-u',str(ROOT/'contact_hold_gpu_sample.py'),'--plan',str(cp),'--output',str(dest)],
                                   stdout=log,stderr=subprocess.STDOUT,check=True,pass_fds=(*loop.CHILD_LOCKS,*fds))
            entry=dict(path=str(dest),sha=sha(dest/'report.json'),plan_sha=sha(cp))
            _,report=verify_batch(entry)
            print('GPU_SAMPLES',report['samples'],'ends',report['ends'],flush=True)
            bank=s['vector_corpus']+[entry]
            pf=out/'plan.json';fp=dict(parent=str(parent),parent_sha=plan['parent_sha'],vector_corpus=bank,
                                        code=identity(),batch=cfg['batch'],updates=cfg['updates'])
            if pf.exists() and read(pf)!=fp:raise ValueError('Fit resume mismatch')
            if not pf.exists():write(pf,fp)
            if not (out/'result.json').exists():
                s['stage']=f'fit {k}';write(file,s)
                with loop.gpu_slot(True) as fds:
                    subprocess.run([sys.executable,'-u',__file__,'--fit',str(pf)],check=True,pass_fds=(*loop.CHILD_LOCKS,*fds))
            checkpoint(out)
            if read(out/'experiment.json')['plan_sha']!=sha(pf):raise ValueError('Model and fit plan disagree')
            s['stage']=f'validate {k}';write(file,s)
            val=loop.run_panel(d/'validation',out,VAL,'validation')
            loop.advance(s,val,out,cfg);s.update(vector_corpus=bank,last_rate=val['rate'],stage='round_complete')
            write(file,s)
            if s['status']=='plateau':break
        if s['status']=='final_test':
            loop.ensure_unseen(TEST,[read(p)['name'] for p in (home/'final').glob('*/attempt*.json')])
            final=loop.run_panel(home/'final',Path(s['champion']),TEST,'test')
            s.update(heldout_rate=final['rate'],status='confirmed_on_test' if final['rate']>=cfg['target'] else 'heldout_target_not_met')
        elif s['status']!='plateau':s['status']='max_rounds'
        write(file,s);print('STOP',s['status'],s['champion'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--status',action='store_true')
    p.add_argument('--num-envs',type=int,default=4);p.add_argument('--rounds',type=int,default=40)
    p.add_argument('--updates',type=int,default=5000);p.add_argument('--batch-size',type=int,default=32768)
    p.add_argument('--patience',type=int,default=12);p.add_argument('--fit',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.fit:fit(a.fit)
    elif not a.run_dir:p.error('--run-dir is required')
    elif a.status:
        s=read(a.run_dir/'state.json');print(json.dumps({k:v for k,v in s.items() if k!='vector_corpus'},indent=2))
    else:
        if not (2<=a.num_envs<=16 and a.rounds*a.num_envs<=9000 and min(a.updates,a.rounds,a.patience)>0 and a.batch_size>=8 and a.batch_size%8==0):p.error('Invalid run configuration')
        campaign(a.run_dir.resolve(),dict(envs=a.num_envs,rounds=a.rounds,updates=a.updates,batch=a.batch_size,
                                         patience=a.patience,target=1.,confirmations=2),a.resume)
