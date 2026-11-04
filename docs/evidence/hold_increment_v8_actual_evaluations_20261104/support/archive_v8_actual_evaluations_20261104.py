"""Archive both completed V8 failures, including late files and incomplete query evidence."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil

ROOT=Path('/home/brainco/rl_cube/github_project');CODE=ROOT/'revotron_cube_rl';WORK=ROOT/'work/teacher_optimization_20261103'
DEST=ROOT/'docs/evidence/hold_increment_v8_actual_evaluations_20261104'
RUNS={20261104:(1842,'tilt_hold_timeout','6280460b32b1d87a7bf2512820ad1b23a83437261bf5881a6c64b441de3d4d5b'),
      20261105:(1725,'evaluation_protocol_error:Post-turn correction exceeded bounded pose envelope','56eba2c26ceff4d85513da3970bfa11ec83d87448f47e9c7b3abafc878f74456')}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def require(test,message):
    if not test:raise ValueError(message)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    items=[];outcomes=[];inventories={}
    for seed,(steps,failure,cp)in RUNS.items():
        name=f'left_hold_increment_v8_seed{seed}_eval_9501_20261104';run=CODE/'runs'/name;r=read(run/'result.json')
        require(r['physical_steps']==r['steps']==r['fully_audited_transition_steps']==steps,'Physical/audit count mismatch')
        require(r['failure']==failure and r['left_teacher_position_action_steps']==r['cube_pose_writes_during_control']==0,'Outcome or intervention mismatch')
        for field in ('original_wrist_gate_qualified','eligible_training_episode','pure_student_episode_claim','corrective_labels_qualified_for_training'):
            require(r[field]is False,'Original False flag altered: '+field)
        require(r['evaluated_policy_checkpoint_sha256']==cp==sha(run/'checkpoints/evaluated_policy.pt'),'Actual model mismatch')
        for rel,h in read(run/'artifact_manifest.json')['sha256'].items():require(sha(run/rel)==h,'Original run manifest mismatch: '+rel)
        count=0;final=None;abort=None
        for line in (run/'evaluation_transitions.jsonl').open():
            require(line.endswith('\n'),'Incomplete journal line');row=json.loads(line)
            if row['record_type']=='executed_transition':
                require(abort is None and row['step']==count and row['transition_finished']is True,'Out-of-order/incomplete transition');count+=1
            elif row['record_type']=='abort':require(abort is None,'Duplicate abort');abort=row
            elif row['record_type']=='evaluation_final':require(final is None,'Duplicate final');final=row['summary']
        require(count==steps and final is not None and final['failure']==failure,'Incomplete original evaluation')
        require(abort is not None and abort['step']==steps and not abort['physics_step_completed_for_unfinished_decision'],'Failed query unexpectedly executed')
        missing_audit='original_controller_audit'not in abort['unfinished_decision']
        require(missing_audit==(seed==20261105),'Unexpected failed-query evidence shape')
        raw=[p for p in sorted(run.rglob('*'))if p.is_file()];inventories[name]=[str(p.relative_to(run))for p in raw]
        items.extend((p,'runs/'+name+'/'+str(p.relative_to(run)),'complete_failed_actual_evaluation')for p in raw)
        log=f'v8_seed{seed}_closed_loop_eval.log';items.append((WORK/log,'launch_logs/'+log,'actual_launch_log'))
        outcomes.append({'run':str(run.relative_to(ROOT)),'result_sha256':sha(run/'result.json'),'checkpoint_sha256':cp,
            'physical_steps':steps,'teacher_query_count_original':r['teacher_query_count'],'failure':failure,'maximum_lift_m':r['maximum_lift_m'],
            'original_wrist_gate_qualified':False,'left_teacher_position_action_steps':0,'cube_pose_writes_during_control':0,
            'original_gain_event_count':r['original_teacher_gain_schedule_events'],'last_raw_done':r['last_environment_termination']['raw_done'],
            'failed_query_physics_executed':False,'failed_query_original_controller_audit_missing':missing_audit,
            'source_result_or_failed_query_fabricated':False,
            'qualification_scope':'First 04 independently reviewed safe-prefix candidate only; archive grants no training eligibility'if seed==20261104 else'05 original strict reader rejected missing controller audit on exception query; no prefix/episode training qualification granted here'})
    support=['launch_hold_increment_v8_eval_20261104.sh','queue_v8_two_closed_loop_evals_20261104.sh','queue_v8_two_closed_loop_evals_20261104.state','queue_v8_two_closed_loop_evals_20261104.completed','queue_v8_two_closed_loop_evals_20261104.log',
        'diagnose_hold_increment_actual_contact_v8.py','diagnose_hold_increment_actual_gates_v8.py','diagnose_v8_seed05_pose_envelope.py','join_v8_actual_contact_modes.py',
        'v8_seed20261104_actual_contact_diagnosis_20261104.json','v8_seed20261105_actual_contact_diagnosis_20261104.json',
        'v8_seed20261104_actual_hold_gate_decomposition_20261104.json','v8_seed20261105_actual_hold_gate_decomposition_20261104.json',
        'v8_seed20261105_original_pose_envelope_reconstruction_20261104.json','v8_seed20261104_contact_mode_causal_join_20261104.json',
        'trusted_eval_dataset_registry_v2.py','trusted_registry_v2_independent_review_20261104.json','trusted_registry_v2_root_review_20261104.json']
    for name in support:items.append((WORK/name,'support/'+name,'independent_posthoc_audit_or_launch_evidence'))
    side=WORK/'v8_seed20261104_actual_registry_v2_audit_20261104'
    require((side/'report.json').is_file()and(side/'branch_replay.jsonl').is_file(),'Missing completed first-prefix replay')
    for path in sorted(side.rglob('*')):
        if path.is_file():items.append((path,'support/'+side.name+'/'+str(path.relative_to(side)),'first_actual_prefix_and_single_forward_branch_audit'))
    items.append((Path(__file__).resolve(),'support/'+Path(__file__).name,'archive_tool'))
    require(len({str(p.resolve())for p,_,_ in items})==len(items),'Duplicate source')
    for source,relative,_ in items:
        require(source.is_file()and not source.is_symlink(),'Missing or symlinked source: '+str(source));source.resolve().relative_to(ROOT)
        require('..'not in Path(relative).parts,'Unsafe archive relative path')
    for proc in Path('/proc').glob('[0-9]*'):
        try:cmd=(proc/'cmdline').read_bytes().split(b'\0')
        except(OSError,PermissionError):continue
        if any(b'evaluate_state_hold_increment_v8.py'in p for p in cmd):
            require(not any(any(x.decode(errors='replace').endswith('/'+n)for n in inventories)for x in cmd),'Selected evaluation still active')
    if not args.execute:
        print(json.dumps({'ready':True,'destination':str(DEST),'files':len(items),'outcomes':outcomes,'run_file_counts':{n:len(v)for n,v in inventories.items()}},indent=2));return
    DEST.mkdir(parents=True,exist_ok=False);entries=[]
    for source,relative,kind in items:
        digest=sha(source);size=source.stat().st_size;compressed=source.suffix in('.jsonl','.log','.usda')or(source.suffix=='.json'and size>32768)
        target=DEST/(relative+('.gz'if compressed else''));target.parent.mkdir(parents=True,exist_ok=True)
        if compressed:
            with source.open('rb')as incoming,target.open('wb')as raw:
                with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0,compresslevel=6)as outgoing:shutil.copyfileobj(incoming,outgoing,1024*1024)
            h=hashlib.sha256()
            with gzip.open(target,'rb')as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            require(h.hexdigest()==digest,'Gzip restored bytes differ')
        else:shutil.copyfile(source,target);require(sha(target)==digest,'Copy differs')
        require(sha(source)==digest and source.stat().st_size==size,'Source changed during archive')
        entries.append({'source':str(source.relative_to(ROOT)),'archive_file':str(target.relative_to(DEST)),'category':kind,
            'encoding':'gzip_mtime_zero'if compressed else'original_bytes','source_sha256':digest,'source_bytes':size,
            'archive_sha256':sha(target),'archive_bytes':target.stat().st_size,'raw_bytes_verified_after_copy_or_decompression':True})
    for name,inventory in inventories.items():require(inventory==[str(p.relative_to(CODE/'runs'/name))for p in sorted((CODE/'runs'/name).rglob('*'))if p.is_file()],'Late run inventory changed during archive')
    manifest={'schema':'two_failed_v8_actual_evaluations_complete_archive_v1','entries':entries,'outcomes':outcomes,
        'complete_run_file_inventories':inventories,'original_results_modified':False,'counterfactual_labels_qualified_by_this_archive':False,
        'whole_episode_training_qualified':False,'closed_loop_wrist_success':False,'face_turn_or_cube_solving_success':False,
        'independent_generalization_success':False,'git_stage_commit_or_push_performed':False}
    path=DEST/'archive_manifest.json';path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'archive':str(DEST),'manifest_sha256':sha(path),'entries':len(entries),'files':len(entries)+1,
        'raw_bytes':sum(x['source_bytes']for x in entries),'archived_bytes':sum(x['archive_bytes']for x in entries)},indent=2))


if __name__=='__main__':main()
