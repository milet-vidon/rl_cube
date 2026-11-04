"""Archive completed V9 after CPU checks; trainer-author preparation, later cross-agent review required."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil

ROOT=Path('/home/brainco/rl_cube/github_project');WORK=ROOT/'work/teacher_optimization_20261103'
p=argparse.ArgumentParser()
p.add_argument('--run-name',required=True)
p.add_argument('--log-name',required=True)
p.add_argument('--inventory-name',required=True)
p.add_argument('--journal-audit-name',required=True)
p.add_argument('--peer-audit-name',required=True)
p.add_argument('--peer-audit-sha256',required=True)
p.add_argument('--preflight-directory')
p.add_argument('--archive-name',required=True)
p.add_argument('--execute',action='store_true')
a=p.parse_args()
for key,value in vars(a).items():
    if value is not None and (key.endswith('_name')or key=='preflight_directory'):
        if Path(value).name!=value:raise ValueError('Only basename paths accepted: '+key)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(path.read_text())
def require(condition,message):
    if not condition:raise ValueError(message)

run=ROOT/'revotron_cube_rl/runs'/a.run_name;result=read(run/'result.json');journal=read(WORK/a.journal_audit_name)
require(journal['reviewer_role']=='trainer_author_self_check_pending_cross_agent_review','Audit reviewer role must be truthful')
require(journal['optimizer_steps_per_epoch']==9 and journal['last_state_artifact_validation']['optimizer_steps']==45000,'Completed nine-history update count differs')
require(journal['audit_passed']is True and isinstance(journal['epochs'],int)and journal['epochs']>=1,'Completed journal has not passed audit')
require(journal['training_run']==str(run),'Wrong run journal audit')
require(result['checkpoint_sha256']==journal['checkpoint_sha256']==sha(run/'model.pt'),'Completed model SHA differs')
require(journal['result_sha256']==sha(run/'result.json')and journal['training_journal_sha256']==sha(run/'training.jsonl'),'Completed result/journal changed')
require(journal['last_state_sha256']==result['last_state_sha256']==sha(run/'last_state.pt'),'Last optimizer/RNG artifact changed')
require(journal['last_state_artifact_validation']['adamw_moments_and_steps_validated']is True and
        journal['last_state_artifact_validation']['private_shuffle_state_matches_all_recorded_epoch_orders']is True,
        'Last-state artifact is not fully inventoried')
require(journal['exact_resume_demonstrated']is False,'Do not claim exact resumed training')
require(sha(WORK/a.peer_audit_name)==a.peer_audit_sha256,'Pinned independent CPU audit differs')
peer=read(WORK/a.peer_audit_name)
require(peer['schema']=='completed_v9_nine_history_cpu_product_replay_v1','Wrong nine-history CPU replay schema')
require(peer['checkpoint_sha256']==result['checkpoint_sha256'] and peer['completed_metadata_metrics_reproduced_exactly']is True,
        'Independent completed-model replay does not match the selected model')
require(peer['final_cpu_sequential_metrics']==result['final_cpu_sequential_metrics'] and
        peer['offline_candidate_qualified']==result['offline_candidate_qualified'],'Independent replay outcome differs')
log=WORK/a.log_name
require('V9_COMPLETE' in log.read_text(errors='replace'),'Training log has no final completion event')
for process in Path('/proc').glob('[0-9]*'):
    try:cmd=(process/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if any(b'train_arm_hold_increment_v9.py'in item for item in cmd):
        require(not any(item.decode(errors='replace').endswith('/'+a.run_name)for item in cmd),'Training process is still active')
inventory_path=WORK/a.inventory_name;inventory=read(inventory_path);items=[]
require(inventory['training_run']==str(run.relative_to(ROOT)),'Wrong frozen archive inventory')
for relative,info in inventory['frozen_files'].items():
    source=ROOT/relative
    require(sha(source)==info['sha256'],'Frozen archive evidence changed: '+relative)
    items.append((source,'frozen/'+relative,info['category']))
for source in sorted(run.rglob('*')):
    if source.is_file():items.append((source,'training_run/'+str(source.relative_to(run)),'completed_training_run'))
for source in (log,inventory_path,WORK/a.journal_audit_name,WORK/a.peer_audit_name,
               WORK/'prepare_arm_hold_increment_archive_inventory_v9.py',
               WORK/'v9_completed_journal_host_tests_20261104.log',
               Path(__file__).resolve()):
    items.append((source,'completion_review/'+source.name,'completed_review_and_archive_tool'))
preflight_included=False
if result['offline_candidate_qualified']:
    require(a.preflight_directory is not None,'Qualified candidate requires actual-checkpoint preflight')
if a.preflight_directory is not None:
    preflight=WORK/a.preflight_directory
    require(preflight.is_dir(),'Missing completed actual checkpoint evaluation preflight')
    identity=read(preflight/'evaluation_preflight.json')
    require(identity['evaluated_policy_checkpoint_sha256']==result['checkpoint_sha256'],'Preflight model differs')
    require(identity['policy']=='v9'and identity['all_checkpoint_source_dependencies_snapshotted_exactly']is True,'Incomplete V9 preflight source binding')
    for source in sorted(preflight.rglob('*')):
        if source.is_file():items.append((source,'evaluation_preflight/'+str(source.relative_to(preflight)),'actual_checkpoint_preflight_no_physics'))
    preflight_included=True
require(len({str(source.resolve())for source,_,_ in items})==len(items),'Duplicate archive source')
for source,relative,_ in items:
    source.resolve().relative_to(ROOT)
    require(not source.is_symlink(),'Refuse symlink archive input')
    require('..'not in Path(relative).parts,'Unsafe archive relative path')
output=ROOT/'docs/evidence'/a.archive_name
if not a.execute:
    print(json.dumps({'ready':True,'files':len(items),'destination':str(output),'execute_required':True}));raise SystemExit
output.mkdir(parents=True,exist_ok=False)
manifest=[]
for source,relative,category in items:
    before=sha(source);raw_size=source.stat().st_size
    compress=source.suffix in ('.jsonl','.log')
    destination=output/(relative+('.gz'if compress else ''));destination.parent.mkdir(parents=True,exist_ok=True)
    if compress:
        with source.open('rb')as src,destination.open('wb')as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6)as out:
                shutil.copyfileobj(src,out,1024*1024)
        h=hashlib.sha256()
        with gzip.open(destination,'rb')as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        require(h.hexdigest()==before,'Compressed evidence failed raw SHA recovery')
    else:
        shutil.copyfile(source,destination)
        require(sha(destination)==before,'Copied evidence differs')
    require(sha(source)==before and source.stat().st_size==raw_size,'Archive input changed during copy')
    manifest.append({'source':str(source.relative_to(ROOT)),'archive_file':str(destination.relative_to(output)),
        'category':category,'encoding':'gzip_mtime_zero'if compress else 'original_bytes',
        'source_sha256':before,'source_bytes':raw_size,'archive_sha256':sha(destination),'archive_bytes':destination.stat().st_size,
        'raw_bytes_verified_after_copy_or_decompression':True})
value={'schema':'completed_v9_training_and_last_optimizer_state_author_prepared_archive_v1','training_run':str(run.relative_to(ROOT)),
    'checkpoint_sha256':result['checkpoint_sha256'],'last_state_sha256':result['last_state_sha256'],'best_epoch':result['best_epoch'],
    'offline_candidate_qualified':result['offline_candidate_qualified'],'entries':manifest,
    'source_dataset_references':inventory['source_dataset_references'],'source_datasets_duplicated_here':False,
    'reviewer_role':'trainer_author_archive_preparation_pending_cross_agent_review',
    'independent_archive_cross_review_performed_here':False,
    'original_result_files_modified':False,'complete_training_run_included':True,'epochs':journal['epochs'],
    'complete_declared_epoch_journal_included':True,'complete_last_optimizer_and_rng_artifact_included':True,
    'actual_checkpoint_evaluation_preflight_included':preflight_included,'exact_resume_demonstrated':False,
    'resume_cli_implemented':False,'training_resume_performed':False,
    'complete_training_run_file_inventory':[str(p.relative_to(run))for p in sorted(run.rglob('*'))if p.is_file()],
    'git_stage_commit_or_push_performed':False,'closed_loop_success_demonstrated':False,
    'independent_generalization_demonstrated':False,'face_turn_or_cube_solving_success_demonstrated':False}
path=output/'archive_manifest.json';path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
print(json.dumps({'archive':str(output),'manifest_sha256':sha(path),'files':len(manifest),
    'raw_bytes':sum(x['source_bytes']for x in manifest),'archived_bytes':sum(x['archive_bytes']for x in manifest)},indent=2))
