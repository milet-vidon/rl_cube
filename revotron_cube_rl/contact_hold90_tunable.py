"""Opt-in configurable 90 mm DAgger campaign. The pinned legacy campaign is not modified."""
import argparse
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import uuid

from contact_hold90_long import DOMAIN, GRASP, ROOT, SCENE, FIT, CHECKPOINT, attested, config, eval_summary, gpu_slot, live_progress, lock, monitor, read, record, sha, write, SAMPLE_FILES
from contact_hold90_controls import DEFAULTS, ENGINE, initialize, freeze_round, validate_params
from contact_hold_loop_data import checkpoint

EXTRA_CODE=('contact_hold90_tunable.py','contact_hold90_weighted_fit.py','contact_hold90_controls.py',
            'contact_hold90_objective.py','contact_hold90_reward_sample.py','run_contact_hold90_tunable.sh')

def derive_parent(parent, expected=None):
    """Select only a completed physical-round checkpoint; allocate unused collection seeds."""
    import torch
    from contact_hold90_cuda_fit import ALGORITHM
    parent=Path(parent)
    p=parent.resolve()
    if p!=parent.absolute():raise ValueError('Checkpoint path must not be a symbolic link')
    if p==FIT:
        attested(p/'best.pt',CHECKPOINT)
        if expected and expected!=CHECKPOINT:raise ValueError('Starting checkpoint changed')
        return p,CHECKPOINT,0
    if len(p.parents)<3 or p.parents[2]!=ROOT/'runs' or p.name!='fit' or not re.fullmatch(r'round_[0-9]{3,4}',p.parent.name) or not p.parents[1].name.startswith('contact_hold90_'):
        raise ValueError('Parent must be a verified completed 90 mm fit')
    k=int(p.parent.name[6:]);home=p.parents[1];settings=read(home/'settings.json')
    if k<1 or k>read(home/'state.json')['completed_round'] or not (p.parent/'validation_summary.json').is_file():raise ValueError('Source round is not completed')
    if settings.get('warm_start_sha256',settings.get('source_weights_sha256'))!=CHECKPOINT:raise ValueError('Unexpected model lineage')
    h=checkpoint(p)
    if expected and h!=expected:raise ValueError('Starting checkpoint changed')
    meta=torch.load(p/'best.pt',map_location='cpu',weights_only=True)['metadata']
    if meta.get('algorithm')!=ALGORITHM or meta.get('warm_start_sha256')!=CHECKPOINT:raise ValueError('Not a verified descendant of R15')
    for name,h0 in meta['source_sha256'].items():
        if Path(name).name!=name or sha(ROOT/name)!=h0:raise ValueError('Parent checkpoint source changed')
    seeds=[i for i in (*meta['training_seeds'],*meta['loss_val_seeds_90']) if 10000000<=i<20000000]
    if not seeds:raise ValueError('No source training seeds')
    return p,h,(max(seeds)-10000000)//2048+1

def configuration(params,parent,expected=None):
    a=argparse.Namespace(rounds=params['rounds'],num_envs=params['envs'],val_envs=params['val_envs'],
            updates_per_round=params['updates'],batch_size=params['batch'],horizon=params['horizon'],lr=params['lr'],replay_rounds=params['replay_rounds'])
    c=config(a);p,h,offset=derive_parent(parent,expected)
    if offset+params['rounds']>10000000//2048:raise ValueError('Insufficient unused seed range')
    c.update(params,engine=ENGINE,source_fit=str(p),source_weights_sha256=h,warm_start_sha256=CHECKPOINT,seed_block_offset=offset)
    c['code'].update({name:sha(ROOT/name) for name in EXTRA_CODE});return c

def roundsamples(home,s,c,k,parent,role,handle):
    rd=home/f'round_{k:03d}'/role;rd.mkdir(parents=True,exist_ok=True)
    if role=='collection':total=c['envs'];base=10000000+2048*(k-1+c['seed_block_offset'])
    else:total=c['val_envs'];base=20000000
    offset=0; paths=[]; ck=attested(parent/'best.pt')
    while offset<total:
        mark=rd/f'accepted_{offset:04d}.json'
        width=read(mark)['width'] if mark.exists() else min(total-offset,s['wave_cap'])
        if not 0<width<=total-offset:raise ValueError('Wrong completed wave width')
        seg=rd/f'wave_{offset:04d}_{width:04d}';seg.mkdir(exist_ok=True)
        p=dict(parent=str(parent),parent_sha=ck,scene_source=c['scene_source'],grasp=c['grasp'],seeds=list(range(base+offset,base+offset+width)),
               lane_offset=offset,code={name:c['code'][name] for name in SAMPLE_FILES},role=role)
        if role=='collection':p['code']['contact_hold90_reward_sample.py']=c['code']['contact_hold90_reward_sample.py']
        cp=seg/'plan.json'
        if cp.is_file() and read(cp)!=p:raise ValueError('Wave plan mismatch')
        if not cp.exists():write(cp,p)
        if mark.exists():
            e=read(mark);dest=Path(e['dest'])
            if sha(dest/'report.json')!=e['report_sha'] or sha(cp)!=e['plan_sha']:raise ValueError('Changed wave report')
        else:
            if shutil.disk_usage(rd).free<2*1024**3:raise RuntimeError('Less than 2 GiB of disk space left')
            dest=seg/('sample_'+uuid.uuid4().hex);log=dest.with_suffix('.log')
            print('START',role,'round',k,'envs',width,'offset',offset,'log',log,flush=True)
            record(home,s,stage=role,active_round=k,current_envs=width,log=str(log))
            with gpu_slot(True) as fds,live_progress(log):
                met=monitor([sys.executable,'-B','-u',str(ROOT/('contact_hold90_reward_sample.py' if role=='collection' else 'contact_hold90_cuda_sample.py')),'--plan',str(cp),'--output',str(dest),
                             '--role',role,'--steps',str(c['horizon'])],log,(handle,*fds))
            write(dest.with_suffix('.resources.json'),met)
            if not met['ok']:
                if met['kind']=='memory_pressure' and width>=8:
                    record(home,s,wave_cap=max(4,4*(width//8)));continue
                raise RuntimeError(f"{role}: {met['reason']}; see {log}")
            e=dict(dest=str(dest),report_sha=sha(dest/'report.json'),plan_sha=sha(cp),width=width)
        report=read(dest/'report.json')
        if report['seeds']!=p['seeds'] or report['parent_sha']!=ck or report['role']!=role:raise ValueError('Wrong seed, parent or role in completed wave')
        if (report['plan_sha256']!=sha(cp) or report['cube_edge_m']!=.09 or report['asset_sha256']!=c['cube_sha256'] or report['grasp_sha256']!=c['grasp_sha256'] or report['physics_device']!='cuda:0' or report['cross_lane_contacts']!=0 or (role=='validate' and any(v['teacher_executed_left_steps'] for v in report['per_lane']))):raise ValueError('Unexpected domain or teacher actions in validation')
        if role=='collection' and (sha(dest/'batch.npz')!=report['data_sha256'] or report.get('reward_sampler_sha256')!=c['code']['contact_hold90_reward_sample.py']):raise ValueError('Changed sampling data or reward source')
        if not mark.exists():write(mark,e)
        paths.append(dest);offset+=len(report['seeds'])
    return paths

def campaign(home,c,s,fd):
    if any(sha(ROOT/n)!=h for n,h in c['code'].items()) or sha(GRASP)!=c['grasp_sha256'] or sha(SCENE/'dynamic_training_scene.usda')!=c['scene_sha256']:
        raise ValueError('Source code, calibration or scene differs from this campaign; do not mix runs')
    while True:
        k=s['completed_round']+1
        frozen=freeze_round(home,k)
        if frozen is None:
            record(home,s,status='completed',stage='round_complete',active_round=None,log=None)
            return
        c=c|frozen['params']
        if c['seed_block_offset']+k>10000000//2048:raise ValueError('No unused collection seed block available')
        record(home,s,active_round=k,control_revision=frozen['revision'])
        parent=Path(s['current']);attested(parent/'best.pt',s['current_sha'])
        d=home/f'round_{k:03d}'
        samples=roundsamples(home,s,c,k,parent,'collection',fd)
        npzs=[str(p/'batch.npz') for p in samples]
        trainlist=d/'collection_paths.json';write(trainlist,npzs)
        replay=[p for i in range(max(1,k-c['replay_rounds']+1),k+1) for p in read(home/f'round_{i:03d}'/'collection_paths.json')]
        fitdir=d/'fit';record(home,s,active_round=k,stage='fit',log=str(d/'fit.log'))
        if not (fitdir/'result.json').exists():
            cmd=[sys.executable,'-B','-u',str(ROOT/'contact_hold90_weighted_fit.py'),'--parent',str(parent/'best.pt'),'--data',*replay,
                 '--outdir',str(fitdir),'--batch',str(c['batch']),'--updates',str(c['updates']),'--lr',str(c['lr']),'--control-receipt',str(d/'control.json')]
            print('GPU_FIT',k,'batch',c['batch'],'updates',c['updates'],'log',d/'fit.log',flush=True)
            with gpu_slot(True) as fds,(d/'fit.log').open('a') as f,live_progress(d/'fit.log'):subprocess.run(cmd,check=True,pass_fds=(fd,*fds),stdout=f,stderr=subprocess.STDOUT)
        fit=read(fitdir/'result.json');h=attested(fitdir/'best.pt',fit['checkpoint_sha256'])
        val=[read(p/'report.json') for p in roundsamples(home,s,c,k,fitdir,'validate',fd)]
        metrics=eval_summary(val)
        key=metrics['selection_score']
        if s.get('best_score') is None or key>s['best_score']:s.update(best_score=key,best=str(fitdir/'best.pt'),best_round=k)
        write(d/'validation_summary.json',metrics)
        record(home,s,status='running',stage='round_complete',active_round=None,
               completed_round=k,current=str(fitdir),current_sha=h,validation=metrics,log=None)
        print('ROUND_COMPLETE',k,metrics,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--control-config',type=Path);ap.add_argument('--parent',type=Path,default=FIT)
    ap.add_argument('--parent-sha');ap.add_argument('--resume',action='store_true');ap.add_argument('--preview',action='store_true')
    a=ap.parse_args();home=a.run_dir.resolve()
    if home.parent!=ROOT/'runs' or not re.fullmatch(r'contact_hold90_[A-Za-z0-9_-]{1,80}',home.name) or home!=a.run_dir.absolute():
        ap.error('choose runs/contact_hold90_<name> without symlinks')
    if a.resume:
        if a.control_config or a.parent!=FIT or a.parent_sha or a.preview:ap.error('resume uses the frozen run settings; change parameters in the dashboard')
        c=read(home/'settings.json')
        if c.get('engine')!=ENGINE:ap.error('旧版实验请用原启动脚本续训，或从已完成轮派生新实验')
    else:
        params=validate_params(read(a.control_config) if a.control_config else {})
        c=configuration(params,a.parent,a.parent_sha)
        if a.preview:print(json.dumps({k:v for k,v in c.items() if k!='code'},indent=2));return
        home.mkdir(exist_ok=False)
    with lock(home/'campaign.lock') as f:
        if a.resume:
            s=read(home/'state.json')
            if s.get('status')=='completed':print('Already completed',s.get('best'));return
        else:
            initialize(home,params)
            write(home/'settings.json',c)
            s=dict(status='running',stage='init',completed_round=0,wave_cap=c['envs'],current=c['source_fit'],current_sha=c['source_weights_sha256'],best_score=None)
        s.pop('last_error',None);record(home,s,status='running')
        try:campaign(home,c,s,f.fileno())
        except (Exception,KeyboardInterrupt) as e:
            record(home,s,status='interrupted',last_error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':main()
