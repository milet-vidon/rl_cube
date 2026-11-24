"""Standalone read-only SSH command for the 5090 training source.

Standard library only. No imports from the trainer, no checkpoint deserialization,
no writes, no simulation/control endpoints. Install under a forced-command key.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import shlex
import socket
import subprocess
import sys

ROOT = Path('/home/brainco/rl_cube/github_project/revotron_cube_rl/runs')
NAME = re.compile(r'r27_(?:recipe|natural_face90)[A-Za-z0-9_-]{0,150}\Z')
SCRIPT = re.compile(r'natural_face90_(?:(?:(?:v\d+|recipe)_)?(?:long|worker)|contact_point(?:_train|_worker)?|approach_anchor_probe|pad_(?:orientation|preshape)_probe|hover_d|bimanual(?:_worker)?|cartesian(?:_contact)?(?:_worker)?)\.py\Z')
LIMIT = 2*1024*1024
SWEEP = 'r70_paired_material_point_sweep_v1'
POINT_PPO = 'natural_face90_contact_point_ppo_v1'
SWEEP_RESULT = 'contact_point_summary.json'
ANCHOR = 'r70_plus7_approach_anchor_v1'
ANCHOR_RESULT = 'approach_anchor_summary.json'
PAD_ORIENTATION = 'r70_plus7_pad_orientation_v1'
PAD_RESULT = 'pad_orientation_summary.json'
PRESHAPE = 'r70_plus7_pad_preshape_v1'
PRESHAPE_RESULT = 'pad_preshape_summary.json'
HOVER_CD = 'natural_face90_r70_hover_cd_v1'
CARTESIAN_PROBE = 'natural_face90_cartesian_neutral_v1'
CARTESIAN_PPO = 'natural_face90_recipe_cartesian_ppo_v1'
CARTESIAN_CONTACT = 'natural_face90_cartesian_contact_curriculum_v1'
CARTESIAN_TO15 = 'natural_face90_cartesian_contact_to15_v1'
BIMANUAL = 'natural_face90_recipe_bimanual_ppo_v1'
CARTESIAN_TRAINING = (CARTESIAN_CONTACT,CARTESIAN_TO15,CARTESIAN_PPO)
CARTESIAN_RESULT = 'cartesian_probe_summary.json'
FROZEN_LAYOUTS = {
    SWEEP: (('plus7','minus7'),SWEEP_RESULT,'sweep','material_point_sweep'),
    ANCHOR: (('probe',),ANCHOR_RESULT,'approach_anchor','frozen_approach_anchor'),
    PAD_ORIENTATION: (('probe',),PAD_RESULT,'pad_orientation','frozen_pad_orientation'),
    PRESHAPE: (('probe',),PRESHAPE_RESULT,'pad_preshape','frozen_pad_preshape'),
    HOVER_CD: (('treatment',),'hover_cd_summary.json','hover_cd','frozen_hover_cd'),
    CARTESIAN_PROBE: (('probe',),CARTESIAN_RESULT,'cartesian_probe','cartesian_neutral'),
}


def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')
def stamp(p):
    try: return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec='seconds')
    except OSError: return None


def file_at(path, home):
    try:
        p=path.resolve()
        return p if p.is_relative_to(home.resolve()) and p.is_file() else None
    except (OSError,ValueError,RuntimeError): return None


def load(path, home):
    p=file_at(path,home)
    if p is None: return {}
    try:
        with p.open('rb') as f: raw=f.read(LIMIT+1)
        if len(raw)>LIMIT: return {}
        d=json.loads(raw,parse_constant=lambda _:None)
        return d if isinstance(d,dict) else {}
    except (OSError,ValueError,RecursionError):return {}


def tail(path,home,limit=128*1024):
    p=file_at(path,home)
    if p is None:return []
    try:
        with p.open('rb') as f:
            offset=max(0,os.fstat(f.fileno()).st_size-limit);f.seek(offset);raw=f.read(limit)
        return raw.decode(errors='replace').splitlines()[1 if offset else 0:]
    except OSError:return []


def run_home(root,name):
    if not isinstance(name,str) or not NAME.fullmatch(name):raise ValueError('Invalid remote experiment name')
    p=(root/name).resolve()
    if p.parent!=root.resolve() or not p.is_dir():raise ValueError('Remote experiment does not exist')
    return p


def process_run(args, cwd, root):
    script=next((Path(x).name for x in args if SCRIPT.fullmatch(Path(x).name)),None)
    if script is None or '--check' in args:
        return None
    if script=='natural_face90_bimanual.py':
        options=args[next(i for i,arg in enumerate(args) if Path(arg).name==script)+1:]
        if 'long' not in options:
            return None
        for i,option in enumerate(options):
            if option.startswith('--run-dir='):
                return (cwd/option.split('=',1)[1]).resolve()
            if option=='--run-dir' and i+1<len(options):
                return (cwd/options[i+1]).resolve()
        return root/'r27_recipe_bimanual_ppo'
    if script in ('natural_face90_cartesian.py','natural_face90_cartesian_contact.py'):
        # Modes may accept both input and output directories. Attribute work to its destination.
        cli=iter(args[next(i for i,arg in enumerate(args) if Path(arg).name==script)+1:])
        paths={}
        modes=[]
        for arg in cli:
            if arg.startswith('--'):
                flag,equals,value=arg.partition('=')
                paths[flag]=value if equals else next(cli,None)
            else:
                modes.append(arg)
        if script=='natural_face90_cartesian_contact.py':
            if modes==['train']:
                return (cwd/(paths.get('--run-dir') or paths.get('--contact-dir') or root/'r27_recipe_cartesian_contact_ppo')).resolve()
            if modes==['train15']:
                return (cwd/(paths.get('--run-dir') or root/'r27_recipe_cartesian_contact_to15')).resolve()
            return None
        if modes==['probe']:
            return (cwd/(paths.get('--probe-dir') or root/'r27_recipe_cartesian_probe')).resolve()
        if modes==['train']:
            return (cwd/(paths.get('--run-dir') or root/'r27_recipe_cartesian_ppo15')).resolve()
        return None
    if script=='natural_face90_hover_d.py' and '--worker' in args:
        return (cwd/args[args.index('--worker')+1]).resolve()
    for flag in ('--run-dir','--request'):
        if flag in args:
            return (cwd/args[args.index(flag)+1]).resolve()
    if script=='natural_face90_contact_point.py':
        return root/'r27_recipe_contact_point_sweep70'
    if script=='natural_face90_approach_anchor_probe.py':
        return root/'r27_recipe_approach_anchor_plus7'
    if script=='natural_face90_pad_orientation_probe.py':
        return root/'r27_recipe_pad_orientation_plus7'
    if script=='natural_face90_pad_preshape_probe.py':
        return root/'r27_recipe_pad_preshape_plus7'
    if script=='natural_face90_hover_d.py':
        return root/'r27_recipe_hover_d70'
    if script=='natural_face90_contact_point_train.py':
        return root/('r27_recipe_contact_point_ppo90' if '--from-qualified15' in args else 'r27_recipe_contact_point_ppo70')
    return None


def processes(root):
    result={};root=root.resolve()
    for file in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=[x.decode(errors='replace') for x in file.read_bytes().split(b'\0') if x]
            cwd=file.parent/'cwd'
            dest=process_run(args,cwd,root)
            if dest is None or not dest.is_relative_to(root):continue
            name=dest.relative_to(root).parts[0]
            if not NAME.fullmatch(name):continue
            state=(file.parent/'stat').read_text().split(') ',1)[1].split()[0]
            if state=='Z':continue
            row=dict(pid=int(file.parent.name),state=state)
            if '--out' in args:row['attempt']=str((cwd/args[args.index('--out')+1]).resolve())
            result.setdefault(name,[]).append(row)
        except (OSError,ValueError,IndexError):continue
    return result


def attempts(home, slots=()):
    if slots==('treatment',):
        path=home/'treatment'
        return [path] if path.is_dir() and path.resolve().is_relative_to(home) else []
    patterns=tuple(slot+'/attempt_*' for slot in slots) or ('smoke/attempt_*','smoke/wave_*/attempt_*',
              'iteration_*/train/attempt_*','iteration_*/validate/wave_*/attempt_*',
              'iteration_*/independent_test/wave_*/attempt_*',
              'round_*/train/wave_*/attempt_*','round_*/validate/wave_*/attempt_*',
              'round_*/final_test/wave_*/attempt_*')
    items=[]
    for pattern in patterns:
        for p in home.glob(pattern):
            if p.is_dir() and p.resolve().is_relative_to(home):items.append(p)
    return sorted(items,key=lambda p:p.stat().st_mtime,reverse=True)


def frozen_result(home, plan, filename):
    summary=load(home/filename,home)
    if summary.get('plan')!=plan or summary.get('updated') is not False or summary.get('independent') is not False:
        return {}
    if filename==SWEEP_RESULT:
        angle=summary.get('selected_offset_deg')
        if (not isinstance(summary.get('angles'),dict) or set(summary['angles'])!={'0','7','-7'} or
                (summary.get('decision'),angle) not in (('no_candidate',None),('selected',7),('selected',-7))):
            return {}
    elif plan.get('schema')==HOVER_CD:
        comparison=summary.get('comparison')
        if not isinstance(comparison,dict) or not isinstance(comparison.get('group'),dict):
            return {}
    elif not all(isinstance(summary.get(k),dict) for k in ('baseline','candidate')):
        return {}
    elif plan.get('schema') in (PAD_ORIENTATION,PRESHAPE,CARTESIAN_PROBE) and not isinstance(summary.get('previous'),dict):
        return {}
    elif plan.get('schema')==CARTESIAN_PROBE and (summary.get('schema')!=CARTESIAN_PROBE or
            not isinstance(summary.get('cartesian'),dict) or type(summary.get('paired_contact_gate_met')) is not bool):
        return {}
    return summary


def header(home, live):
    cfg=load(home/'config.json',home);state=load(home/'state.json',home)
    if not cfg:
        cfg=load(home/'plan.json',home)
    schema=cfg.get('schema')
    if not isinstance(schema,str):
        raise ValueError('Unsupported remote training schema')
    layout=FROZEN_LAYOUTS.get(schema)
    slots=layout[0] if layout else ()
    if not (layout or schema==POINT_PPO or schema in CARTESIAN_TRAINING or str(schema).startswith(('natural_face90_search_','natural_face90_recipe_'))):
        raise ValueError('Unsupported remote training schema')
    batch=None
    for proc in live:
        if proc.get('attempt'):
            p=Path(proc['attempt']).resolve()
            if p.is_relative_to(home) and p.is_dir():batch=p;break
    if batch is None:batch=next(iter(attempts(home,slots)),None)
    failed=load(batch/'failed.json',home) if batch else {}
    final=load(home/'smoke.json',home)
    summary=frozen_result(home,cfg,layout[1]) if layout else {}
    if summary:status='completed'
    elif failed:status='failed'
    elif live:status='paused' if all(x['state'] in ('T','t') for x in live) else 'running'
    elif schema==BIMANUAL and state.get('status')=='not_started':status='not_started'
    elif state.get('status') in ('interrupted','failed'):status=state['status']
    elif final or state.get('status') not in (None,'running'):status='completed'
    else:status='interrupted'
    entry=dict(id=home.name,kind='diagnostic' if cfg.get('smoke') else 'training',status=status,
        algorithm='state_ppo' if schema==POINT_PPO or schema in CARTESIAN_TRAINING or 'recipe' in schema else 'parameter_search',
        recorded_status=state.get('status'),completed=state.get('iteration',state.get('round',0)),
        planned=cfg.get('iterations',cfg.get('rounds')),lanes=cfg.get('lanes'),
        updated_at=stamp(batch/'progress.json') if batch else stamp(home),
        pids=[p['pid'] for p in live],error=failed.get('error'))
    if not entry['updated_at']:entry['updated_at']=stamp(home)
    if slots:
        completed=0
        for folder in slots:
            attempt=load(home/folder/'accepted.json',home).get('attempt')
            if isinstance(attempt,str) and re.fullmatch('attempt_[a-f0-9]{8,32}',attempt):
                completed+=int(file_at(home/folder/attempt/'report.json',home) is not None)
        entry.update(kind=layout[2],algorithm=layout[3],
                     recorded_status=summary.get('decision'),lanes=len(cfg.get('seeds',[])),
                     planned=len(slots),completed=len(slots) if summary else completed)
        if summary:
            entry['updated_at']=stamp(home/layout[1])
            entry['error']=None
    elif schema==BIMANUAL:
        entry.update(kind='bimanual_training',target_deg=state.get('target_deg'))
    elif schema==POINT_PPO or schema in CARTESIAN_TRAINING:
        entry['target_deg']=cfg.get('target_deg')
        if schema in CARTESIAN_TRAINING:
            entry['kind']={CARTESIAN_PPO:'cartesian_training',CARTESIAN_CONTACT:'cartesian_contact',
                           CARTESIAN_TO15:'cartesian_contact_to15'}[schema]
    return entry,cfg,state,batch


def catalog(root,live):
    result=[]
    for p in root.glob('r27_*'):
        try:entry,*_=header(run_home(root,p.name),live.get(p.name,[]));result.append(entry)
        except (ValueError,OSError):continue
    return sorted(result,key=lambda x:(x['status']=='running',x['updated_at'] or '',x['id']),reverse=True)


def select(d,keys): return {k:d[k] for k in keys if k in d}


def aggregate(rows):
    rows=[r for r in rows if isinstance(r,dict)]
    return dict(episodes=len(rows),successes=sum(r.get('success') is True for r in rows),
                unsafe=sum(r.get('unsafe') is True for r in rows),
                prefix_held=sum(r.get('prefix_endpoint_held') is True for r in rows),
                natural_observed=sum(r.get('natural_ready') is True for r in rows),
                touched_target=sum((r.get('target_contact_steps') or 0)>0 for r in rows))


def gpu():
    try:
        out=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits','-i','0'],capture_output=True,text=True,timeout=2,check=True)
        name,use,used,total=[v.strip() for v in out.stdout.splitlines()[0].split(',')]
        return dict(available=True,name=name,util_percent=int(use),used_mib=int(used),total_mib=int(total))
    except (OSError,ValueError,IndexError,subprocess.SubprocessError):return dict(available=False)


LEARN=('kind','learned_lanes','learned_transitions','real_transitions','actor_updated','actor_max_delta',
       'optimizer_steps','mean_reward','policy_loss','value_loss','approx_kl','kl_early_stop','deterministic',
       'input_checkpoint','input_checkpoint_sha','output_checkpoint','output_checkpoint_sha')
ROW=('lane','seed','prefix_endpoint_held','natural_ready','target_contact_steps','final_face_deg',
     'peak_signed_face_deg','success','unsafe','end_reason','post_steps',
     'drive_frames','qualified_index_frames','distal_contact_frames','peak_drive_response_deg',
     'peak_isolated_drive_progress_deg','bilateral_frames','bilateral_substeps','opposed_substeps','peak_bilateral_drive_deg',
     'bimanual_goal_reached','bimanual_turn_held')
CARTESIAN_FIELDS=('actor_dim','critic_dim','action_dim','action_axes','residual_norm_cap_m',
                  'control_hz','material_offset_deg','index_preshape_deg','end_effector','frame','ik_backend')


def point_stages(home, decision, live):
    """Report only runs linked to this sweep. Missing downstream runs have no run ID."""
    options={15:[],90:[]}
    qualified15=False
    for path in home.parent.glob('r27_recipe*'):
        if path.is_symlink() or not path.is_dir():
            continue
        cfg=load(path/'config.json',path)
        target=cfg.get('target_deg')
        if cfg.get('schema')!=POINT_PPO or cfg.get('sweep_dir')!=str(home) or target not in (15,90):
            continue
        entry,*_=header(path,live.get(path.name,[]))
        if target==15 and entry['recorded_status']=='qualified_15deg_state_specialist':
            qualified15=qualified15 or bool(load(path/'qualified_policy.json',path))
        options[target].append(entry)
    result=[]
    for target in (15,90):
        if options[target]:
            latest=max(options[target],key=lambda x:(x['status']=='running',x['updated_at'] or '',x['id']))
            result.append(dict(target_deg=target,status=latest['status'],reason=latest['recorded_status'],
                               run_id=latest['id']))
        elif target==90 and not qualified15:
            result.append(dict(target_deg=90,status='blocked',reason='等待 +15° 独立验收通过',run_id=None))
        elif target==15 and decision!='selected':
            reason='两个落点均未通过训练门槛' if decision=='no_candidate' else '等待冻结选点结果'
            result.append(dict(target_deg=15,status='blocked',reason=reason,run_id=None))
        else:
            result.append(dict(target_deg=target,status='not_started',reason='已满足上一阶段条件，等待手动启动',run_id=None))
    return result


def sweep_details(home, cfg, live):
    recorded=frozen_result(home,cfg,SWEEP_RESULT)
    decision=recorded.get('decision')
    return dict(decision=decision,selected_offset_deg=recorded.get('selected_offset_deg'),
                fixed_C_no_contact_horizon_seeds=recorded.get('fixed_C_no_contact_horizon_seeds',[]),
                angles=recorded.get('angles',{}),stages=point_stages(home,decision,live))


def snapshot(root,name,live):
    home=run_home(root,name);entry,cfg,state,batch=header(home,live.get(name,[]))
    out=dict(**entry,source_path=str(home),target='90 mm 魔方 · 自然换姿 → 右手单层 +15° → 撤手保持',
             settings=select(cfg,('schema','smoke','lanes','iterations','rounds','validate_every','validation_seeds','population',
                                  'target_deg','contact_point_offset_deg','pad_preshape_offset_deg')),
             hardware=gpu(),progress={},log=[],history=[],episodes=[],live_episodes=[],learning={},summary=None)
    final=load(home/'smoke.json',home);chosen=load(home/'best_candidate.json',home)
    out['best_candidate']=select(chosen,('checkpoint','sha','iteration','round','validation','qualified'))
    out['qualified']=bool(load(home/'qualified_policy.json',home))
    if cfg['schema']==SWEEP:
        out['target']='冻结 R70 · 原点、+7°、−7° 材料落点配对比较'
        out['sweep']=sweep_details(home,cfg,live)
    elif cfg['schema']==ANCHOR:
        out['target']='冻结 R70 · +7° 落点接近阶段配对诊断'
        out['settings'].update(target_deg=15,contact_point_offset_deg=7)
        recorded=frozen_result(home,cfg,ANCHOR_RESULT)
        out['approach_anchor']=select(recorded,('baseline','candidate','new_unsafe_seeds','qualifying_fixed_seeds'))
    elif cfg['schema'] in (PAD_ORIENTATION,PRESHAPE):
        out['target']='冻结 R70 · +7° 落点' + ('右食指预形与朝向对照' if cfg['schema']==PRESHAPE else '指垫朝向对照')
        out['settings'].update(target_deg=15,contact_point_offset_deg=7)
        recorded=frozen_result(home,cfg,FROZEN_LAYOUTS[cfg['schema']][1])
        out['pad_orientation']=select(recorded,('baseline','previous','candidate','pad_orientation',
            'new_unsafe_seeds','qualifying_fixed_seeds','paired_contact_gate_met'))
        if cfg['schema']==PRESHAPE:
            out['preshape_screen']=select(cfg.get('handoff_geometry',{}),('handoffs','fixed_handoffs','counts'))
    elif cfg['schema']==HOVER_CD:
        out['target']='冻结 R70 · C/D 首次触层与接近回中比较'
        out['hover_cd']=frozen_result(home,cfg,'hover_cd_summary.json').get('comparison',{})
    elif cfg['schema']==CARTESIAN_PROBE:
        out['target']='Placo 逆运动学 · +7° 材料落点 · 零策略增量配对验证'
        out['settings'].update(target_deg=15,contact_point_offset_deg=7,pad_preshape_offset_deg=-6)
        recorded=frozen_result(home,cfg,CARTESIAN_RESULT)
        out['cartesian_probe']=select(recorded,('baseline','previous','candidate','qualifying_fixed_seeds',
            'new_unsafe_seeds','paired_contact_gate_met','rollout_qualifying_fixed_seeds'))
        out['cartesian']=select(recorded.get('cartesian',{}),CARTESIAN_FIELDS)
    elif cfg['schema']==BIMANUAL:
        out['settings'].update(select(cfg,('actor_dim','action_dim')))
        out['settings']['target_deg']=state.get('target_deg')
        out['target']=f"左手支撑其余两层，右手对握 nZ 层 · 当前目标 +{state.get('target_deg')}° · 最终 +90°"
        out['qualified']=out['qualified'] and state.get('status')=='qualified_90deg_state_specialist'
    elif cfg['schema']==CARTESIAN_CONTACT:
        out['target']='右食指安全首次触层 → 推动目标层至少 +2°'
        out['qualified']=False
    elif cfg['schema'] in (POINT_PPO,CARTESIAN_PPO,CARTESIAN_TO15):
        out['target']=f"90 mm 魔方 · 右食指驱动 nZ 单层 +{cfg.get('target_deg')}° → 撤手保持"
        if cfg['schema']==CARTESIAN_TO15:
            out['qualified']=out['qualified'] and state.get('status')=='qualified_15deg_state_specialist'
    if batch:
        request=load(batch/'request.json',home);report=load(batch/'report.json',home)
        if not request and (cfg['schema'] in FROZEN_LAYOUTS or cfg['schema']==POINT_PPO or cfg['schema'] in CARTESIAN_TRAINING):
            request=load(batch.parent/'request.json',home)
        if cfg['schema']==SWEEP:
            out['settings'].update(select(request,('target_deg','contact_point_offset_deg')))
        out.update(role=request.get('role'),iteration=request.get('iteration'),attempt=batch.name,
            progress=load(batch/'progress.json',home),log=tail(batch.with_suffix('.log'),home)[-100:],
            log_updated_at=stamp(batch.with_suffix('.log')),
            learning=select(report.get('expert_learning',{}),LEARN))
        if (cfg['schema']==CARTESIAN_PROBE or cfg['schema'] in CARTESIAN_TRAINING) and not out.get('cartesian'):
            out['cartesian']=select(report.get('expert_learning',{}).get('cartesian',{}),CARTESIAN_FIELDS)
        rows=report.get('rows',[])
        if not isinstance(rows,list):rows=[]
        if isinstance(rows,list) and rows:
            out['summary']=aggregate(rows)
            out['episodes']=[select(r,ROW) for r in rows if isinstance(r,dict)]
        receipt=load(batch.parent/'accepted.json',home)
        out['report_accepted']=bool(out['hover_cd']) if cfg['schema']==HOVER_CD else receipt.get('attempt')==batch.name
        # Bounded live observer tail. These are ongoing measurements, never
        # promoted to final per-episode successes or independent validation.
        latest={}
        for line in tail(batch/'post_measurements.jsonl',home):
            try:
                r=json.loads(line,parse_constant=lambda _:None)
                if type(r.get('lane')) is not int:continue
                latest[r['lane']]=select(r,('lane','seed','step','controller_phase','natural_ready','face_deg',
                       'right_contact','wrong_contact','structure_ok','hold_streak'))
            except (ValueError,AttributeError):continue
        out['live_episodes']=[latest[k] for k in sorted(latest)]
        for line in reversed(out['log']):
            if 'NATURAL90_READY' in line and not out['progress']:out['stage']='initializing';break
        if out['progress']:out['stage']='rollout'
        if rows:out['stage']='reported'
    if final:
        out['summary']=aggregate(final['rows']) if final.get('rows') else final.get('summary')
        out['learning']=select(final.get('learning',out['learning']),LEARN)
    for file in sorted(home.glob('iteration_*/result.json'))[-200:]:
        r=load(file,home)
        row=dict(index=r.get('iteration'),train=r.get('training_rollouts_before_update'),
                 validation=r.get('validation_after_update'),independent_test=r.get('independent_test'),
                 learning=select(r.get('learning',{}),LEARN))
        if cfg['schema']==BIMANUAL:
            if type(row['index']) is not int or row['index']>entry['completed']:continue
            row['target_deg']=r.get('target_deg')
            if isinstance(row['validation'],dict):
                out['bimanual_validation']=dict(row['validation'],iteration=row['index'],target_deg=row['target_deg'])
        if cfg['schema']==CARTESIAN_CONTACT:
            if type(row['index']) is not int or row['index']>entry['completed']:continue
            panel=r.get('paired_gate')
            if isinstance(panel,dict):
                row['paired_gate']=select(panel,('fixed_11','all_16','qualifying_fixed_seeds',
                                                 'new_unsafe_seeds','paired_contact_gate_met','fixed_peak_drive_deg'))
                out['contact_gate']=dict(row['paired_gate'],iteration=row['index'])
        out['history'].append(row)
    if cfg['schema']==CARTESIAN_CONTACT:
        receipt=load(home/'contact_ready.json',home)
        gate=out.get('contact_gate',{})
        paired=receipt.get('paired_gate')
        out['contact_ready']=(state.get('status')=='contact_ready' and gate.get('paired_contact_gate_met') is True
                              and receipt.get('iteration')==state.get('iteration')==gate.get('iteration')
                              and bool(receipt.get('sha')) and receipt['sha']==state.get('checkpoint_sha')
                              and bool(receipt.get('checkpoint')) and receipt['checkpoint']==state.get('checkpoint')
                              and isinstance(paired,dict) and
                              all(paired.get(k)==v for k,v in gate.items() if k!='iteration'))
    if not out['history']:
        for file in sorted(home.glob('round_*/selection.json'))[-200:]:
            r=load(file,home);v=r.get('validation',[]);i=r.get('selected')
            out['history'].append(dict(index=int(file.parent.name.split('_')[-1]),
                validation=v[i] if type(i) is int and 0<=i<len(v) else None,learning={}))
    return out


def handle(command,root=ROOT):
    args=shlex.split(command)
    if args==['catalog']:
        live=processes(root);data=dict(runs=catalog(root,live))
    elif len(args)==2 and args[0]=='snapshot':
        # Validate before looking at other processes or touching hardware.
        run_home(root,args[1]);data=snapshot(root,args[1],processes(root))
    else:raise ValueError('Only catalog and snapshot <experiment> are supported')
    return dict(schema='cube_remote_monitor_v1',host=socket.gethostname(),sampled_at=now(),data=data)


if __name__=='__main__':
    try:
        command=os.environ.get('SSH_ORIGINAL_COMMAND') or ' '.join(sys.argv[1:])
        print(json.dumps(handle(command),ensure_ascii=False,allow_nan=False,separators=(',',':')))
    except (OSError,ValueError,KeyError) as error:
        print(json.dumps(dict(error=str(error)[:500]),ensure_ascii=False));sys.exit(1)
