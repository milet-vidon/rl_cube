"""Recover the last validated round; one logical round may use several isolated PhysX waves."""
import argparse
import json
import math
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
import numpy as np
import torch
import contact_hold_continuous as loop
import contact_hold_wide_campaign as wide
from contact_hold_wide_campaign import BASE, DIGEST, VAL, TEST, show_sampling_progress
from contact_hold_wide_replay import Bank
from contact_hold_cuda_monitor import monitor
from contact_hold_dataset import sha
from contact_hold_loop_data import ROOT, checkpoint, read, write
from contact_hold_long_gpu import tensors
from contact_hold_policy import restored
from train_contact_hold_bc import atomic_save

from contact_hold_cuda_campaign import HORIZON, sampler_code, verify_wave, FILES as PREPARATION_FILES
FILES=PREPARATION_FILES+('contact_hold_cuda_train.py','contact_hold_cuda_wave_merge.py','run_contact_hold_cuda_long.sh')

def identity():
    return {k:sha(ROOT/k) for k in FILES}

def verify_batch(entry):
    if 'waves' not in entry:return wide.verify_batch(entry)
    start=1_000_000+2048*entry['round']
    from contact_hold_cuda_wave_merge import merge
    seeds=[]
    def samples():
        for e in entry['waves']:
            a,r=verify_wave(e);p=read(Path(e['path']).parent/'collect.json')
            if r['checkpoint']!=entry['parent_sha'] or len(seeds)!=p['lane_offset']:raise ValueError('Wave lineage mismatch')
            a['lane']=a['lane']+len(seeds);seeds.extend(r['seeds'])
            yield a,r
    arrays=merge(samples(),start)
    if seeds!=list(range(start,start+entry['envs'])):raise ValueError('Incomplete or overlapping seed allocation')
    return arrays,{'seeds':seeds,'checkpoint':entry['parent_sha']}

def collect(rd,k,parent,cfg,s,statefile):
    total=cfg['envs']; offset=0; waves=[];retried=set();base=1_000_000+k*2048
    while offset<total:
        # Widths are multiples of four: never leave a singleton or move the handoff schedule.
        cap=s.get('effective_envs',total);n=min(total-offset,cap)
        b=rd/f'wave_{offset:04d}_{n:04d}';b.mkdir(exist_ok=True)
        p=dict(parent=str(parent),parent_sha=checkpoint(parent),code=sampler_code(),seeds=list(range(base+offset,base+offset+n)),lane_offset=offset)
        cp=b/'collect.json'
        if cp.exists() and read(cp)!=p:raise ValueError('Wave resume mismatch')
        if not cp.exists():write(cp,p)
        ptr=b/'attempt.json';dest=b/read(ptr)['name'] if ptr.exists() else None
        # A completed wave before a later capacity reduction is still reusable after a restart.
        if (rd/f'accepted_{offset:04d}.json').is_file():
            e=read(rd/f'accepted_{offset:04d}.json'); _,r=verify_wave(e)
            if r['checkpoint']!=p['parent_sha'] or r['seeds']!=list(range(base+offset,base+offset+r['num_envs'])):raise ValueError('Changed accepted wave')
            waves.append(e);offset+=r['num_envs'];continue
        # Only retry a demonstrably signalled worker, at most once per invocation.
        # Keep the old UUID, log and failure receipt for review.
        if dest is not None and dest.with_suffix('.resources.json').is_file() and (offset,n) not in retried:
            old=read(dest.with_suffix('.resources.json'))
            if old.get('kind')=='worker_exit' and old.get('exit_code') in (-9,-15,-2):
                print('RETRY_INTERRUPTED',offset,n,'discarding',dest.name,flush=True)
                retried.add((offset,n));dest=None
        if dest is None or not dest.with_suffix('.resources.json').exists():
            if shutil.disk_usage(rd).free<3*1024**3:raise RuntimeError('Need 3 GiB free for collection')
            dest=b/('samples_'+uuid.uuid4().hex);write(ptr,dict(name=dest.name));log=dest.with_suffix('.log')
            print('GPU_SAMPLE',k,'wave',offset,'parallel',n,'round_envs',total,'log',log,flush=True)
            with loop.gpu_slot(True) as fds,show_sampling_progress(log):
                m=monitor([sys.executable,'-u',str(ROOT/'contact_hold_cuda_sample.py'),'--plan',str(cp),'--output',str(dest),'--steps',str(HORIZON)],
                          log,(*loop.CHILD_LOCKS,*fds))
                write(dest.with_suffix('.resources.json'),m)
        tel=dest.with_suffix('.resources.json');t=read(tel)
        if not t['ok']:
            if t['kind']!='memory_pressure' or n<=4:raise RuntimeError(f"Collection rejected ({t['reason']}): {tel}")
            new=max(4,4*(n//8));s.update(effective_envs=new)
            s.setdefault('backoffs',[]).append(dict(round=k,offset=offset,envs=n,next=new,receipt=str(tel),reason=t['reason']))
            write(statefile,s);print('RETRY_SMALLER',n,'->',new,'same seed offset',offset,flush=True);continue
        e=dict(path=str(dest),sha=sha(dest/'report.json'),plan_sha=sha(cp),resources_sha=sha(tel))
        _,r=verify_wave(e);write(rd/f'accepted_{offset:04d}.json',e);waves.append(e);offset+=n
        from collections import Counter
        print('GPU_SAMPLES',r['samples'],'ends',dict(Counter(r['ends'])),flush=True)
    entry=dict(waves=waves,round=k,envs=total,parent_sha=checkpoint(parent));verify_batch(entry);return entry

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
    replay=Bank(16384,197621);extra_seeds=set()
    for entry in plan['vector_corpus']:
        a,r=verify_batch(entry);replay.add({k:a[k] for k in ('x','y','mask','val','phase')});extra_seeds.update(r['seeds'])
    a=replay.finish()
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
    meta=dict(meta,source_sha256=meta['source_sha256']|identity(),plan_sha=sig,algorithm='batched_cuda_feedback_DAgger_BC_v3_weighted_waves',
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


def campaign(home,prepare=False):
    with loop.lock(home/'campaign.lock') as fd:
        loop.CHILD_LOCKS=(fd.fileno(),)
        settings=read(home/'settings.json');cfg=settings['cfg'];file=home/'state.json';s=read(file)
        if settings['code']!=identity() or settings['base']!=checkpoint(BASE):raise ValueError('Changed training dependencies')
        if s['status'] in loop.DONE:return print('STOP',s['status'],s['champion'])
        while s['round']<cfg['rounds'] and s['status']!='final_test':
            k=s['round']+1;d=home/f'round_{k:03}';out=d/'fit';out.mkdir(parents=True,exist_ok=True)
            parent=Path(s['current']);s['stage']=f'collect {k}';write(file,s)
            e=collect(d,k,parent,cfg,s,file);bank=s['vector_corpus']+[e]
            pf=out/'plan.json';fp=dict(parent=str(parent),parent_sha=e['parent_sha'],vector_corpus=bank,code=identity(),batch=cfg['batch'],updates=cfg['updates'])
            if pf.exists() and read(pf)!=fp:raise ValueError('Fit restart mismatch')
            if not pf.exists():write(pf,fp)
            if prepare:s['stage']=f'ready_to_fit {k}';write(file,s);print('PREPARED',pf,flush=True);return
            if not (out/'result.json').exists():
                s['stage']=f'fit {k}';write(file,s)
                with loop.gpu_slot(True) as fds:
                    subprocess.run([sys.executable,'-u',__file__,'--fit',str(pf)],check=True,pass_fds=(*loop.CHILD_LOCKS,*fds))
            checkpoint(out)
            if read(out/'experiment.json')['plan_sha']!=sha(pf):raise ValueError('Wrong fit plan')
            s['stage']=f'validate {k}';write(file,s)
            val=loop.run_panel(d/'validation',out,VAL,'validation')
            loop.advance(s,val,out,cfg);s.update(vector_corpus=bank,last_rate=val['rate'],stage='round_complete')
            write(file,s)
            if s['status']=='plateau':break
        if s['status']=='final_test':
            loop.ensure_unseen(TEST,[read(p)['name'] for p in (home/'final').glob('*/attempt*.json')])
            panel=loop.run_panel(home/'final',Path(s['champion']),TEST,'test')
            s.update(heldout_rate=panel['rate'],status='confirmed_on_test' if panel['rate']>=cfg['target'] else 'heldout_target_not_met')
        elif s['status']!='plateau':s['status']='max_rounds'
        write(file,s);print('STOP',s['status'],s['champion'],flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path);ap.add_argument('--from-run',type=Path,default=ROOT/'runs/contact_hold_wide_v1')
    ap.add_argument('--num-envs',type=int,default=1536);ap.add_argument('--rounds',type=int,default=60);ap.add_argument('--updates',type=int,default=5000)
    ap.add_argument('--prepare',action='store_true',help='Collect the next round and stop before fitting')
    ap.add_argument('--reuse-prepared',type=Path,help='Import only verified, complete next-round wave receipts, preserving seeds and parent')
    ap.add_argument('--preview',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--status',action='store_true')
    ap.add_argument('--fit',type=Path,help=argparse.SUPPRESS)
    a=ap.parse_args()
    if a.fit:return fit(a.fit)
    if not a.run_dir:ap.error('--run-dir required')
    home=a.run_dir.resolve()
    if a.status:return print(json.dumps({k:v for k,v in read(home/'state.json').items() if k!='vector_corpus'},indent=2))
    if a.resume:return campaign(home,a.prepare)
    src=a.from_run.resolve()
    if home.exists() or home.is_relative_to(src) or src.is_relative_to(home):ap.error('choose a fresh sibling run directory (or --resume)')
    if a.reuse_prepared:
        import fcntl
        old=a.reuse_prepared.resolve()
        with (old/'campaign.lock').open('rb') as guard:
            try:fcntl.flock(guard,fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError:
                print('WAIT_PREPARED',old,'waiting for verified collection',flush=True)
                fcntl.flock(guard,fcntl.LOCK_SH)
    with loop.lock(src/'campaign.lock'):
        cfg,s=wide.trace_source(src)
        if not (4<=a.num_envs<=2048 and a.num_envs%4==0 and s['round']<a.rounds<=2048 and a.updates>0):
            ap.error('invalid parallel size or schedule')
        cfg=dict(cfg,envs=a.num_envs,rounds=a.rounds,updates=a.updates);cfg.pop('tiers',None)
        link=dict(source=str(src),completed_round=s['round'],current=s['current'],best_sha=checkpoint(Path(s['current'])),
                  optimizer_sha=sha(Path(s['current'])/'last.pt'),num_envs=cfg['envs'],batch=cfg['batch'],updates=cfg['updates'],rounds=cfg['rounds'])
        prepared=[]
        if a.reuse_prepared:
            old=a.reuse_prepared.resolve()
            if old==home or home.is_relative_to(old) or old.is_relative_to(home):ap.error('choose a separate training directory')
            k=s['round']+1
            # Prepared receipts are inputs only; do not depend on its old fit implementation or an untrained checkpoint.
            prior=read(old/'state.json');fp=read(old/f'round_{k:03}'/'fit/plan.json')
            if (fp['code']!=read(old/'settings.json')['code'] or prior['stage']!=f'ready_to_fit {k}' or prior['round']!=s['round'] or fp['parent']!=s['current'] or
                fp['vector_corpus'][:-1]!=s['vector_corpus'] or fp['parent_sha']!=link['best_sha']):
                raise ValueError('Prepared round does not extend this validated checkpoint')
            bundle=fp['vector_corpus'][-1]
            if (bundle['round'],bundle['parent_sha'],bundle['envs'])!=(k,link['best_sha'],cfg['envs']):
                raise ValueError('Prepared round size or parent differs')
            verify_batch(bundle)
            offset=0
            for receipt in bundle['waves']:
                _,r=verify_wave(receipt);prepared.append((offset,receipt));offset+=r['num_envs']
            link['prepared_wave_source']=str(old);s['effective_envs']=prior['effective_envs']
        print('CUDA_RESUME_CONFIG',link,flush=True)
        if a.preview:return
        s.setdefault('effective_envs',cfg['envs']);home.mkdir(parents=True,exist_ok=False)
        if prepared:
            rd=home/f"round_{s['round']+1:03}";rd.mkdir()
            for offset,e in prepared:write(rd/f'accepted_{offset:04d}.json',e)
        write(home/'lineage.json',link);write(home/'settings.json',dict(cfg=cfg,base=DIGEST,code=identity()));write(home/'state.json',s)
    campaign(home,a.prepare)

if __name__=='__main__': main()
