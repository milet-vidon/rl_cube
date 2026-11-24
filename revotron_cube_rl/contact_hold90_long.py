"""GPU DAgger/BC for the 90 mm physical cube. New campaign, initialized from frozen round 15."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from contact_hold_loop_data import ROOT,read,write
from contact_hold_dataset import sha
from contact_hold_grace_worker import CHECKPOINT,FIT
from contact_hold_continuous import lock,gpu_slot
from contact_hold90_cuda_sample import FILES as SAMPLE_FILES
from contact_hold_cuda_monitor import monitor
from contact_hold90_progress import live_progress

DOMAIN=ROOT/'runs/contact_hold_size_r15_90mm_retarget_approach_20261112_084150_canonical_run'
SCENE=DOMAIN/'from_reset/seed_21611'
GRASP=DOMAIN/'calibration/grasp90.json'
CODE=('contact_hold90_long.py','contact_hold90_cuda_sample.py','contact_hold90_cuda_fit.py','contact_hold90_cuda_domain.py',
      'contact_hold_cuda_scene.py','contact_hold_cuda_scene_reduce.py','contact_hold_cuda_ik.py','contact_hold_cuda_ik_kernel.py',
      'contact_hold_cuda_teacher.py','contact_hold_cuda_teacher_math.py','contact_hold_cuda_teacher_turn.py',
      'contact_hold_cuda_monitor.py','contact_hold_continuous.py','contact_hold_wide_replay.py','contact_hold_policy.py','joint_safety.py',
      'contact_hold90_progress.py','run_contact_hold90_long.sh')
def stamp():return datetime.now().astimezone().isoformat(timespec='seconds')
def status(home):
    s=read(home/'state.json')
    print(json.dumps(s,indent=2,ensure_ascii=False));return s
def config(a):
    if (not 1<=a.rounds<=2000 or not 4<=a.num_envs<=2048 or a.num_envs%4 or not 4<=a.val_envs<=256 or a.val_envs%4 or
        not 1<=a.updates_per_round or not 16<=a.batch_size<=262144 or not 16<=a.horizon<=3500 or not .0000001<=a.lr<=.001 or not 1<=a.replay_rounds<=16):
        raise ValueError('Invalid round, env, update, replay, batch, horizon or learning rate setting')
    if sha(FIT/'best.pt')!=CHECKPOINT:raise ValueError('Wrong starting checkpoint')
    stage=read(SCENE/'artifact_manifest.json')['sha256'];receipt=read(DOMAIN/'calibration/geometry_retarget.json')
    if receipt['grasp_sha256']!=sha(GRASP) or sha(SCENE/'dynamic_training_scene.usda')!=stage['dynamic_training_scene.usda']:raise ValueError('Changed physical scene or grasp')
    return dict(rounds=a.rounds,envs=a.num_envs,val_envs=a.val_envs,updates=a.updates_per_round,batch=a.batch_size,horizon=a.horizon,
                lr=a.lr,replay_rounds=a.replay_rounds,source_fit=str(FIT),source_weights_sha256=CHECKPOINT,
                scene_source=str(SCENE),scene_sha256=stage['dynamic_training_scene.usda'],grasp=str(GRASP),grasp_sha256=sha(GRASP),
                cube_sha256=read(SCENE/'cube_asset.json')['source_sha256'],code={name:sha(ROOT/name) for name in dict.fromkeys((*CODE,*SAMPLE_FILES))})
def record(home,s,**v):
    s.update(v,updated=stamp());write(home/'state.json',s)
def attested(path,expected=None):
    if not path.is_file():raise ValueError(f'Missing checkpoint: {path}')
    h=sha(path)
    if expected is not None and h!=expected:raise ValueError('Changed checkpoint: '+str(path))
    return h
def roundsamples(home,s,c,k,parent,role,handle):
    rd=home/f'round_{k:03d}'/role;rd.mkdir(parents=True,exist_ok=True)
    if role=='collection':total=c['envs'];base=10000000+2048*(k-1)
    else:total=c['val_envs'];base=20000000
    offset=0; paths=[]; ck=attested(parent/'best.pt')
    while offset<total:
        mark=rd/f'accepted_{offset:04d}.json'
        width=read(mark)['width'] if mark.exists() else min(total-offset,s['wave_cap'])
        if not 0<width<=total-offset:raise ValueError('Wrong completed wave width')
        seg=rd/f'wave_{offset:04d}_{width:04d}';seg.mkdir(exist_ok=True)
        p=dict(parent=str(parent),parent_sha=ck,scene_source=c['scene_source'],grasp=c['grasp'],seeds=list(range(base+offset,base+offset+width)),
               lane_offset=offset,code={name:c['code'][name] for name in SAMPLE_FILES},role=role)
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
                met=monitor([sys.executable,'-B','-u',str(ROOT/'contact_hold90_cuda_sample.py'),'--plan',str(cp),'--output',str(dest),
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
        if role=='collection' and sha(dest/'batch.npz')!=report['data_sha256']:raise ValueError('Changed sampling data')
        if not mark.exists():write(mark,e)
        paths.append(dest);offset+=len(report['seeds'])
    return paths

def campaign(home,c,s,fd):
    if any(sha(ROOT/n)!=h for n,h in c['code'].items()) or sha(GRASP)!=c['grasp_sha256'] or sha(SCENE/'dynamic_training_scene.usda')!=c['scene_sha256']:
        raise ValueError('Source code, calibration or scene differs from this campaign; do not mix runs')
    for k in range(s['completed_round']+1,c['rounds']+1):
        parent=Path(s['current']);attested(parent/'best.pt',s['current_sha'])
        d=home/f'round_{k:03d}'
        samples=roundsamples(home,s,c,k,parent,'collection',fd)
        npzs=[str(p/'batch.npz') for p in samples]
        trainlist=d/'collection_paths.json';write(trainlist,npzs)
        replay=[p for i in range(max(1,k-c['replay_rounds']+1),k+1) for p in read(home/f'round_{i:03d}'/'collection_paths.json')]
        fitdir=d/'fit';record(home,s,active_round=k,stage='fit',log=str(d/'fit.log'))
        if not (fitdir/'result.json').exists():
            cmd=[sys.executable,'-B','-u',str(ROOT/'contact_hold90_cuda_fit.py'),'--parent',str(parent/'best.pt'),'--data',*replay,
                 '--outdir',str(fitdir),'--batch',str(c['batch']),'--updates',str(c['updates']),'--lr',str(c['lr'])]
            print('GPU_FIT',k,'batch',c['batch'],'updates',c['updates'],'log',d/'fit.log',flush=True)
            with gpu_slot(True) as fds,(d/'fit.log').open('a') as f,live_progress(d/'fit.log'):subprocess.run(cmd,check=True,pass_fds=(fd,*fds),stdout=f,stderr=subprocess.STDOUT)
        fit=read(fitdir/'result.json');h=attested(fitdir/'best.pt',fit['checkpoint_sha256'])
        val=[read(p/'report.json') for p in roundsamples(home,s,c,k,fitdir,'validate',fd)]
        metrics=eval_summary(val)
        key=metrics['selection_score']
        if s.get('best_score') is None or key>s['best_score']:s.update(best_score=key,best=str(fitdir/'best.pt'),best_round=k)
        write(d/'validation_summary.json',metrics)
        record(home,s,status='running' if k<c['rounds'] else 'completed',stage='round_complete',active_round=None,
               completed_round=k,current=str(fitdir),current_sha=h,validation=metrics,log=None)
        print('ROUND_COMPLETE',k,metrics,flush=True)

def eval_summary(reports):
    # Inputs are independent, pure-left-student episodes on the fixed validation panel.
    rows=[d for r in reports for d in r['per_lane']]
    if not rows or any(r['role']!='validate' for r in reports) or len({e['seed'] for e in rows})!=len(rows):raise ValueError('Wrong validation panel')
    n=len(rows)
    holds=sum(bool(e['ever_hold']) for e in rows);ending=sum(bool(e['ending_hold']) for e in rows);turns=sum(bool(e['ever_whole_turn']) for e in rows)
    gap=sum(max(0.,e['max_height_m'] or 0.) for e in rows)/n
    length=sum((e.get('physical_rotation') or {}).get('best_controlled_hold',{}).get('seconds',0.) for e in rows)/n
    return dict(envs=n,mean_controlled_hold_s=length,held=holds,hold_rate=holds/n,still_held_at_end=ending,whole_turn=turns,whole_turn_rate=turns/n,
                mean_peak_height_m=gap,face_twist_trained=False,selection_score=[holds,ending,turns,round(length,3),round(gap,4)])

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',type=Path,required=True)
    actions=ap.add_mutually_exclusive_group();actions.add_argument('--resume',action='store_true');actions.add_argument('--status',action='store_true');actions.add_argument('--preview',action='store_true')
    ap.add_argument('--num-envs',type=int,default=1536);ap.add_argument('--val-envs',type=int,default=32)
    ap.add_argument('--rounds',type=int,default=100);ap.add_argument('--updates-per-round',type=int,default=1000)
    ap.add_argument('--batch-size',type=int,default=32768);ap.add_argument('--horizon',type=int,default=3500)
    ap.add_argument('--replay-rounds',type=int,default=3);ap.add_argument('--lr',type=float,default=3e-5)
    a=ap.parse_args();home=a.run_dir.resolve()
    if a.status:status(home);return
    if not a.resume:
        if not home.parent==ROOT/'runs' or not home.name.startswith('contact_hold90_'):ap.error('choose a fresh runs/contact_hold90_<name>')
        c=config(a)
        if a.preview:print(json.dumps({k:v for k,v in c.items() if k!='code'},indent=2));return
        home.mkdir(exist_ok=False)
    with lock(home/'campaign.lock') as f:
        if not a.resume:
            write(home/'settings.json',c);s=dict(status='running',stage='init',completed_round=0,wave_cap=c['envs'],
                    current=str(FIT),current_sha=CHECKPOINT,best_score=None);record(home,s)
        else:c=read(home/'settings.json');s=read(home/'state.json')
        if s['status']=='completed':print('Already completed',s.get('best'));return
        s.pop('last_error',None)
        record(home,s,status='running')
        try:campaign(home,c,s,f.fileno())
        except (Exception,KeyboardInterrupt) as err:
            record(home,s,status='interrupted',last_error=f'{type(err).__name__}: {err}');raise
if __name__=='__main__':main()
