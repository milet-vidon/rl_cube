"""Archive only a completed, independently checked training run; never git-stage."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil

ROOT=Path('/home/brainco/rl_cube/github_project');WORK=ROOT/'work/teacher_optimization_20261103'
p=argparse.ArgumentParser()
p.add_argument('--run-name',default='teacher_zero_gated_residual_v7_20261104')
p.add_argument('--log-name',default='v7_gpu_train.log')
p.add_argument('--inventory-name',default='v7_training_archive_inventory_20261104_v2.json')
p.add_argument('--journal-audit-name',default='v7_completed_training_journal_audit_20261104.json')
p.add_argument('--peer-audit-name',default='v7_completed_checkpoint_cpu_replay_20261104.json')
p.add_argument('--peer-audit-sha256',required=True)
p.add_argument('--preflight-directory',default='v7_completed_eval_preflight_20261104')
p.add_argument('--archive-name',default='zero_gated_v7_training_20261104')
p.add_argument('--execute',action='store_true')
a=p.parse_args()
for key,value in vars(a).items():
    if key.endswith('_name')or key=='preflight_directory':
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
require(journal['audit_passed']is True and journal['epochs']==5000,'Completed journal has not passed audit')
require(journal['training_run']==str(run),'Wrong run journal audit')
require(result['checkpoint_sha256']==journal['checkpoint_sha256']==sha(run/'model.pt'),'Completed model SHA differs')
require(journal['result_sha256']==sha(run/'result.json')and journal['training_journal_sha256']==sha(run/'training.jsonl'),'Completed result/journal changed')
require(sha(WORK/a.peer_audit_name)==a.peer_audit_sha256,'Pinned independent CPU audit differs')
log=WORK/a.log_name
require('V7_COMPLETE' in log.read_text(errors='replace'),'Training log has no final completion event')
for process in Path('/proc').glob('[0-9]*'):
    try:cmd=(process/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if any(b'train_zero_gated_residual_v7.py'in item for item in cmd):
        require(not any(item.decode(errors='replace').endswith('/'+a.run_name)for item in cmd),'Training process is still active')
inventory_path=WORK/a.inventory_name;inventory=read(inventory_path);items=[]
for relative,info in inventory['frozen_files'].items():
    source=ROOT/relative
    require(sha(source)==info['sha256'],'Frozen archive evidence changed: '+relative)
    items.append((source,'frozen/'+relative,info['category']))
for source in sorted(run.rglob('*')):
    if source.is_file():items.append((source,'training_run/'+str(source.relative_to(run)),'completed_training_run'))
for source in (log,inventory_path,WORK/a.journal_audit_name,WORK/a.peer_audit_name,
               WORK/'audit_v7_training_journal_20261104.py',Path(__file__).resolve()):
    items.append((source,'completion_review/'+source.name,'completed_review_and_archive_tool'))
preflight=WORK/a.preflight_directory
require(preflight.is_dir(),'Missing completed actual checkpoint evaluation preflight')
for source in sorted(preflight.rglob('*')):
    if source.is_file():items.append((source,'evaluation_preflight/'+str(source.relative_to(preflight)),'actual_checkpoint_preflight_no_physics'))
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
value={'schema':'completed_v7_training_evidence_archive_v1','training_run':str(run.relative_to(ROOT)),
    'checkpoint_sha256':result['checkpoint_sha256'],'best_epoch':result['best_epoch'],
    'offline_candidate_qualified':result['offline_candidate_qualified'],'entries':manifest,
    'source_dataset_references':inventory['source_dataset_references'],'source_datasets_duplicated_here':False,
    'original_result_files_modified':False,'complete_training_run_included':True,'all_5000_epoch_journal_included':True,
    'git_stage_commit_or_push_performed':False,'closed_loop_success_demonstrated':False,
    'independent_generalization_demonstrated':False,'face_turn_or_cube_solving_success_demonstrated':False}
path=output/'archive_manifest.json';path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
print(json.dumps({'archive':str(output),'manifest_sha256':sha(path),'files':len(manifest),
    'raw_bytes':sum(x['source_bytes']for x in manifest),'archived_bytes':sum(x['archive_bytes']for x in manifest)},indent=2))
