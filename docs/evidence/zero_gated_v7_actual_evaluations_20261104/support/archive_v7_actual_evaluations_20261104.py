"""Preserve three failed physical evaluations and the original zero-step launch error."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil

ROOT=Path('/home/brainco/rl_cube/github_project');CODE=ROOT/'revotron_cube_rl';WORK=ROOT/'work/teacher_optimization_20261103'
RUNS={
 'left_zero_gated_residual_v7_eval_9501_20261104':('0920f7e4952b5bf534021ce8240a8467a303bffda4b7ec1750eb830b3aea4054','v7_closed_loop_eval.log'),
 'left_zero_gated_residual_v7_seed20261105_eval_9501_20261104_v2':('10afd5a60ec587eb040068a520dad171ce412d681d2ff46251ae74b8ea84e7bb','v7_seed20261105_closed_loop_eval_v2.log'),
 'left_zero_gated_residual_v7_long_eval_9501_20261104':('a7efd54313d40c8991e8c468c4312442501ba71b9a4d427170318bcfd29a99ee','v7_long_closed_loop_eval.log')}
ERROR_RUN='left_zero_gated_residual_v7_seed20261105_eval_9501_20261104'
p=argparse.ArgumentParser();p.add_argument('--sidecar',action='append',default=[]);p.add_argument('--execute',action='store_true');a=p.parse_args()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(path.read_text())
def require(condition,message):
    if not condition:raise ValueError(message)

items=[];outcomes=[];inventories={}
for name,(checkpoint,log)in RUNS.items():
    run=CODE/'runs'/name;result=read(run/'result.json')
    require(result['steps']==result['physical_steps']==result['fully_audited_transition_steps']==752,'Wrong completed physical count: '+name)
    require(result['teacher_query_count']==753 and result['failure']=='contact_topology_or_settle_timeout','Wrong failure record: '+name)
    require(result['left_teacher_position_action_steps']==result['cube_pose_writes_during_control']==0,'Unexpected intervention: '+name)
    require(result['original_wrist_gate_qualified']is False and result['eligible_training_episode']is False
            and result['pure_student_episode_claim']is False and result['corrective_labels_qualified_for_training']is False,'Original false flags changed: '+name)
    require(result['evaluated_policy_checkpoint_sha256']==checkpoint==sha(run/'checkpoints/evaluated_policy.pt'),'Actual checkpoint differs: '+name)
    original_manifest=read(run/'artifact_manifest.json')['sha256']
    for relative,digest in original_manifest.items():require(sha(run/relative)==digest,'Original source-run manifest mismatch: '+name+'/'+relative)
    count=0;final=None
    with (run/'evaluation_transitions.jsonl').open()as stream:
        for line in stream:
            require(line.endswith('\n'),'Incomplete final evaluation journal')
            r=json.loads(line)
            if r['record_type']=='executed_transition':count+=1
            if r['record_type']=='evaluation_final':final=r['summary']
    require(count==752 and final is not None and final['failure']==result['failure'],'Incomplete evaluation evidence: '+name)
    raw=[p for p in sorted(run.rglob('*'))if p.is_file()];inventories[name]=[str(p.relative_to(run))for p in raw]
    items.extend((path,'runs/'+name+'/'+str(path.relative_to(run)),'completed_failed_physics_evaluation')for path in raw)
    items.append((WORK/log,'launch_logs/'+log,'actual_launch_log'))
    outcomes.append({'run':str(run.relative_to(ROOT)),'original_result_sha256':sha(run/'result.json'),
        'checkpoint_sha256':checkpoint,'physical_steps':752,'teacher_queries':753,'maximum_lift_m':result['maximum_lift_m'],
        'failure':result['failure'],'original_wrist_gate_qualified':False,'left_teacher_position_action_steps':0,
        'cube_pose_writes_during_control':0,'raw_final_done':result['last_environment_termination']['raw_done'],
        'original_false_eligibility_flags_preserved':True})
error_run=CODE/'runs'/ERROR_RUN;error=read(error_run/'evaluation_error.json')
require(error['type']=='FileNotFoundError'and'best.pt'in error['message'],'Unexpected zero-physics initialization failure')
require(not(error_run/'result.json').exists(),'Do not fabricate a result for zero-step error')
require(not(error_run/'evaluation_transitions.jsonl').exists()and not(error_run/'trajectory.jsonl').exists(),'Zero-step failure unexpectedly contains physics transitions')
require('snapshot_bytes'in error['traceback']and'prepare'in error['traceback'],'Initialization failure is not before runtime creation')
raw=[p for p in sorted(error_run.rglob('*'))if p.is_file()];inventories[ERROR_RUN]=[str(p.relative_to(error_run))for p in raw]
items.extend((path,'runs/'+ERROR_RUN+'/'+str(path.relative_to(error_run)),'original_zero_physics_initialization_error')for path in raw)
items.append((WORK/'v7_seed20261105_closed_loop_eval.log','launch_logs/v7_seed20261105_closed_loop_eval.log','original_wrong_checkpoint_path_launch_log'))
outcomes.append({'run':str(error_run.relative_to(ROOT)),'original_error_sha256':sha(error_run/'evaluation_error.json'),
    'error_type':'FileNotFoundError','failure_stage':'prepare.snapshot_bytes before EvaluationRuntime creation',
    'physical_steps_diagnostic_inference':0,'no_original_result_file_exists':True,'result_file_fabricated':False,
    'wrong_model_basename':'best.pt','corrected_actual_rerun_name':list(RUNS)[1]})
required=('diagnose_v7_actual_contact_feedback_20261104.py','v7_actual_contact_feedback_diagnosis_20261104.json',
          'launch_v7_closed_loop_eval_20261104.sh')
for name in required+tuple(a.sidecar):
    require(Path(name).name==name,'Only sidecar basenames accepted')
    items.append((WORK/name,'support/'+name,'independent_actor_prefix_or_contact_diagnostic'))
items.append((Path(__file__).resolve(),'support/'+Path(__file__).name,'archive_tool'))
require(len({str(source.resolve())for source,_,_ in items})==len(items),'Duplicate archive source')
for source,relative,_ in items:
    require(source.is_file()and not source.is_symlink(),'Missing/linked archive source '+str(source))
    source.resolve().relative_to(ROOT)
    require('..'not in Path(relative).parts,'Unsafe archive target')
for process in Path('/proc').glob('[0-9]*'):
    try:cmd=(process/'cmdline').read_bytes().split(b'\0')
    except(FileNotFoundError,PermissionError,ProcessLookupError):continue
    if any(b'evaluate_state_zero_gated_v7.py'in item for item in cmd):
        require(not any(any(item.decode(errors='replace').endswith('/'+name)for name in RUNS)for item in cmd),'An archived evaluation is still running')
output=ROOT/'docs/evidence/zero_gated_v7_actual_evaluations_20261104'
if not a.execute:
    print(json.dumps({'ready_for_complete_archive':True,'files':len(items),'outcomes':outcomes,
        'support_files':list(required)+a.sidecar,'destination':str(output)},indent=2));raise SystemExit
output.mkdir(parents=True,exist_ok=False);entries=[]
for source,relative,category in items:
    digest=sha(source);size=source.stat().st_size
    compress=source.suffix in('.jsonl','.log','.usda')or(source.suffix=='.json'and size>32768)
    target=output/(relative+('.gz'if compress else''));target.parent.mkdir(parents=True,exist_ok=True)
    if compress:
        with source.open('rb')as src,target.open('wb')as raw:
            with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0,compresslevel=6)as out:shutil.copyfileobj(src,out,1024*1024)
        h=hashlib.sha256()
        with gzip.open(target,'rb')as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        require(h.hexdigest()==digest,'Gzip restoration differs')
    else:shutil.copyfile(source,target);require(sha(target)==digest,'Copy differs')
    require(sha(source)==digest and source.stat().st_size==size,'Source changed while archiving')
    entries.append({'source':str(source.relative_to(ROOT)),'archive_file':str(target.relative_to(output)),'category':category,
        'encoding':'gzip_mtime_zero'if compress else'original_bytes','source_sha256':digest,'source_bytes':size,
        'archive_sha256':sha(target),'archive_bytes':target.stat().st_size,'raw_bytes_verified_after_copy_or_decompression':True})
manifest={'schema':'three_failed_v7_actual_evaluations_and_zero_step_path_error_archive_v1','entries':entries,
    'outcomes':outcomes,'complete_run_file_inventories':inventories,'original_results_modified':False,
    'counterfactual_labels_qualified_by_this_archive':False,'whole_episode_training_qualified':False,
    'closed_loop_wrist_success':False,'face_turn_or_cube_solving_success':False,'independent_generalization_success':False,
    'git_stage_commit_or_push_performed':False}
path=output/'archive_manifest.json';path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
print(json.dumps({'archive':str(output),'manifest_sha256':sha(path),'files':len(entries),
    'raw_bytes':sum(r['source_bytes']for r in entries),'archived_bytes':sum(r['archive_bytes']for r in entries)},indent=2))
