"""CUDA multi-environment pickup data collection, BC updates, original full-task validation."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import torch

import contact_hold_continuous as loop
import contact_hold_gpu_campaign as legacy
from contact_hold_wide_replay import Bank
from contact_hold_wide_capacity import monitored
from contact_hold_dataset import sha
from contact_hold_loop_data import ROOT, checkpoint, inspect, read, write
from contact_hold_long_gpu import tensors
from contact_hold_policy import restored
from train_contact_hold_bc import atomic_save

BASE=ROOT/'runs/contact_hold_long_v1/round_039/fit'
DIGEST='8468adccd589d2e84634b27f65d6563f7287618c9b901f01a43e06a4f3441f58'
FILES=legacy.FILES+('contact_hold_wide_campaign.py','contact_hold_wide_sample.py','contact_hold_wide_scene.py','contact_hold_wide_planner.py','contact_hold_wide_replay.py','contact_hold_wide_capacity.py','contact_hold_wide_scene_probe.py','contact_hold_wide_qualify.py','contact_hold_vram_probe.py','run_contact_hold_wide.sh')
VAL=(21611,21612)
TEST=(51611,51612,51613)


def identity():
    return {k:sha(ROOT/k) for k in FILES}

@contextmanager
def show_sampling_progress(log):
    """Mirror milestones in the user's terminal; the complete Isaac log stays on disk."""
    stop=threading.Event()
    def follow():
        while not log.exists():
            if stop.wait(.25):return
        with log.open(errors='replace') as f:
            while True:
                line=f.readline()
                if line:
                    if line.startswith(('NATIVE_CLONES','GPU_SCENE','PLAN_LANES','VECTOR_STEP','GPU_COLLECTION_COMPLETE')):
                        print(line.rstrip(),flush=True)
                elif stop.wait(.5):return
    t=threading.Thread(target=follow,daemon=True);t.start()
    try:yield
    finally:stop.set();t.join(timeout=2)


def verify_batch(entry):
    path=Path(entry['path']);planpath=path.parent/'collect.json';p=read(planpath)
    if sha(path/'report.json')!=entry['sha'] or sha(planpath)!=entry['plan_sha']:raise ValueError('Changed rollout receipt')
    r=read(path/'report.json')
    if r['schema']=='cuda_wrist_query_v1':return legacy.verify_batch(entry)
    telemetry=path.parent/(path.name+'.resources.json')
    if not telemetry.is_file() or not read(telemetry)['ok'] or sha(telemetry)!=entry['resources_sha']:raise ValueError('No passing native PhysX receipt')
    if not (r['schema']=='wide_cuda_wrist_query_v2' and r['recipe']=='native_env_ids_two_contact_views' and r['scope']=='reset_to_wrist_hold_or_termination' and r['plan_sha256']==entry['plan_sha']
            and r['checkpoint']==p['parent_sha']==checkpoint(Path(p['parent']))
            and len(r['eligible_per_stratum'])==4 and sum(min(8192,v) for v in r['eligible_per_stratum'])==r['samples'] and r['num_envs']==len(p['seeds'])>=2 and r['num_envs']<=2048 and r['seeds']==p['seeds']
            and len(set(r['seeds']))==len(r['seeds']) and all(1_000_000<=s<9_000_000 for s in r['seeds'])
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
    meta=dict(meta,source_sha256=meta['source_sha256']|identity(),plan_sha=sig,algorithm='native_wide_GPU_DAgger_BC_v2',
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
            parent=Path(s['current'])
            while True:
                n=s.get('effective_envs',cfg['envs']);block=d/f'collection_{n:04d}';block.mkdir(exist_ok=True)
                cp=block/'collect.json';seeds=list(range(1_000_000+k*2048,1_000_000+k*2048+n))
                plan=dict(parent=str(parent),parent_sha=checkpoint(parent),code=identity(),seeds=seeds)
                if cp.exists() and read(cp)!=plan:raise ValueError('Sampler resume mismatch')
                if not cp.exists():write(cp,plan)
                ptr=block/'attempt.json';dest=block/read(ptr)['name'] if ptr.exists() else None
                telemetry=block/(dest.name+'.resources.json') if dest else None
                if telemetry is None or not telemetry.is_file():
                    if shutil.disk_usage(home).free<3*1024**3:raise RuntimeError('Need 3 GiB free disk space to resume')
                    dest=block/('samples_'+uuid.uuid4().hex);write(ptr,dict(name=dest.name))
                    s.update(effective_envs=n,stage=f'collect {k}');write(file,s)
                    telemetry=block/(dest.name+'.resources.json');log=block/(dest.name+'.log')
                    print('GPU_SAMPLE',k,'native environments',n,'log',log,flush=True)
                    with loop.gpu_slot(True) as fds, show_sampling_progress(log):
                        trace=monitored([sys.executable,'-u',str(ROOT/'contact_hold_wide_sample.py'),'--plan',str(cp),'--output',str(dest)],
                                        log,(*loop.CHILD_LOCKS,*fds))
                        write(telemetry,trace)
                trace=read(telemetry)
                if trace['ok']:
                    entry=dict(path=str(dest),sha=sha(dest/'report.json'),plan_sha=sha(cp),resources_sha=sha(telemetry))
                    break
                why=trace['reason'];smaller=[v for v in cfg['tiers'] if v<n]
                # Discard this attempt; never fit on a PhysX run that may have dropped contact pairs.
                err=(block/(dest.name+'.log')).read_text(errors='replace').lower()
                pressure=(why=='device_or_host_memory_headroom' or any(msg in err for msg in
                    ('out of memory','memoryconfig::','contact buffer exhausted','buffer overflow','capacity exceeded')))
                if not pressure or not smaller:raise RuntimeError('Collection rejected: '+str(telemetry))
                s['effective_envs']=max(smaller);s.setdefault('backoffs',[]).append(dict(round=k,envs=n,reason=why,receipt=str(telemetry)))
                write(file,s);print('RETRY_SMALLER',s['effective_envs'],why,flush=True)
            _,report=verify_batch(entry)
            from collections import Counter
            print('GPU_SAMPLES',report['samples'],'ends',dict(Counter(report['ends'])),flush=True)
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

def qualification(path):
    from contact_hold_wide_qualify import sampler_code
    q=read(path);n=q['n'];d=path.parent/f'n{n}';plan=d/'collect.json';report=d/'samples/report.json'
    t=d/'telemetry.json';rows=Path(q['capacity'])/'trials.json'
    if not (q['kind']=='wide_real_sampler_short_probe' and q['n_steps']>=12 and q['code']==sampler_code()
            and q['plan_sha']==sha(plan) and q['report_sha']==sha(report) and sha(t)==q['resource_sha']
            and read(t)==q['telemetry'] and read(t)['ok'] and sha(rows)==q['trials_sha']
            and sha(d/'samples/batch.npz')==read(report)['data_sha256'] and read(report)['num_envs']==n
            and read(report)['samples']==q['samples']>0 and read(report)['cross_lane_contacts']==0
            and read(report)['plan_sha256']==sha(plan) and read(plan)['parent_sha']==checkpoint(Path(read(plan)['parent']))
            and read(t)['peak_device_mib']<=read(t)['total_mib']-640 and read(t)['min_host_available_mib']>=5000):
        raise ValueError('Not a current measured sampler qualification')
    tiers=sorted({r['n'] for r in read(rows) if r['ok'] and 2<=r['n']<=n and r['code']['contact_hold_wide_scene.py']==q['code']['contact_hold_wide_scene.py']
                  and r['code']['contact_hold_wide_scene_probe.py']==sha(ROOT/'contact_hold_wide_scene_probe.py')
                  and r['peak_device_mib']<=r['total_mib']-768 and r['min_host_available_mib']>=6000 and r['physics']['cross']==0})
    if n not in tiers:raise ValueError('No capacity-tested tier for the indicated sampler')
    return q,tiers

def trace_source(src):
    # A previous high-memory branch inherits earlier rounds by absolute receipt, not by copying directories.
    # Verify that complete chain even when its round directories span multiple campaigns.
    from start_contact_hold_high_vram import validated_panel
    st=read(src/'state.json');settings=read(src/'settings.json');cfg=settings['cfg']
    if settings['code'] not in (identity(),legacy.identity()) or settings['base']!=checkpoint(BASE) or settings['base']!=DIGEST:
        raise ValueError('Source campaign dependencies or baseline changed')
    bank=st['vector_corpus']
    if len(bank)!=st['round'] or not len(bank) or st['status'] in ('final_test','confirmed_on_test','heldout_target_not_met'):
        raise ValueError('Need completed development rounds; cannot branch from final test')
    prev=BASE;champion=BASE;best=validated_panel(BASE.parent/'validation',BASE)['score']
    for k,entry in enumerate(bank,1):
        _,r=verify_batch(entry);block=Path(entry['path']).parent
        rd=block.parent if r['schema']=='wide_cuda_wrist_query_v2' else block
        fitdir=rd/'fit';plan=fitdir/'plan.json';pf=read(plan)
        if (rd.name!=f'round_{k:03}' or pf['parent']!=str(prev) or pf['parent_sha']!=checkpoint(prev)
                or r['checkpoint']!=pf['parent_sha'] or pf['vector_corpus']!=bank[:k] or
                pf['code'] not in (identity(),legacy.identity()) or read(fitdir/'experiment.json')['plan_sha']!=sha(plan)
                or sha(fitdir/'last.pt')!=read(fitdir/'result.json')['last_sha256']):
            raise ValueError('Broken optimizer, data or model lineage in '+str(rd))
        checkpoint(fitdir);panel=validated_panel(rd/'validation',fitdir)
        if panel['score']>best:best=panel['score'];champion=fitdir
        prev=fitdir
    if st['current']!=str(prev) or st['champion']!=str(champion) or best!=st['best_score'] or st['last_rate']!=panel['rate']:
        raise ValueError('Source state disagrees with validation evidence')
    loop.ensure_unseen(TEST)
    return cfg,dict(current=str(prev),champion=str(champion),best_score=best,round=len(bank),vector_corpus=bank,last_rate=panel['rate'],
                    status='running',stage='inherited_completed_validation',stale=0,streak=0)

def main():
    p=argparse.ArgumentParser(description='Measured native CUDA multi-environment cube training')
    p.add_argument('--from-run',type=Path,default=ROOT/'runs/contact_hold_parallel_vram_v1');p.add_argument('--run-dir',type=Path)
    p.add_argument('--qualification',type=Path,default=ROOT/'runs/contact_hold_wide_qual_a/qualified.json')
    p.add_argument('--num-envs',default='auto',help='auto: use hardware-tested environment count')
    p.add_argument('--rounds',type=int);p.add_argument('--updates',type=int)
    p.add_argument('--resume',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--preview',action='store_true')
    p.add_argument('--fit',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.fit:return fit(a.fit)
    if not a.run_dir:p.error('--run-dir is required')
    home=a.run_dir.resolve()
    if a.status:
        print(json.dumps({k:v for k,v in read(home/'state.json').items() if k!='vector_corpus'},indent=2));return
    if a.resume:return campaign(home,{},True)
    src=a.from_run.resolve()
    if home.exists() or home.is_relative_to(src) or src.is_relative_to(home):p.error('choose a new sibling run directory')
    q,tiers=qualification(a.qualification.resolve());n=q['n'] if a.num_envs=='auto' else int(a.num_envs)
    if n not in tiers:p.error('select auto or a measured passing tier: '+str(tiers))
    cfg,state=trace_source(src);cfg=dict(cfg,envs=n,tiers=[v for v in tiers if v<=n])
    if a.rounds is not None:cfg['rounds']=a.rounds
    if a.updates is not None:cfg['updates']=a.updates
    if not (state['round']<cfg['rounds']<=2048 and cfg['updates']>=1):p.error('invalid training schedule')
    state['effective_envs']=n
    link=dict(kind='wide_measured_native_branch_v2',source=str(src),completed_round=state['round'],current=state['current'],
              parent_sha=checkpoint(Path(state['current'])),qualification=str(a.qualification.resolve()),qualification_sha=sha(a.qualification),
              num_envs=n,batch=cfg['batch'],updates=cfg['updates'],total_rounds=cfg['rounds'],fallback_tiers=cfg['tiers'])
    print('WIDE_CONFIG',link,flush=True)
    if a.preview:return
    from contact_hold_vram_probe import gpu
    device,total,_,_=gpu()
    if (device,total)!=(q['telemetry']['gpu'],q['telemetry']['total_mib']):raise ValueError('Rerun sampler calibration on this device')
    try:guard=loop.lock(src/'campaign.lock')
    except BlockingIOError:p.exit(2,'Source is active. Stop its terminal with Ctrl+C after a completed validation round; retry.\n')
    with guard:
        if read(src/'state.json')['round']!=state['round']:raise ValueError('Source advanced while selecting a checkpoint; retry')
        home.mkdir(parents=True,exist_ok=False)
        write(home/'settings.json',dict(cfg=cfg,base=DIGEST,code=identity()));write(home/'state.json',state);write(home/'lineage.json',link)
        campaign(home,cfg,True)

if __name__=='__main__':main()
