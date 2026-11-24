"""Freeze all 1803-step failed actual-run files and reviewed prefix evidence."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil

ROOT=Path('/home/brainco/rl_cube/github_project');WORK=ROOT/'work/teacher_optimization_20261103'
NAME='left_hold_increment_v8_wide128_seed20261105_eval_9501_20261104'
RUN=ROOT/'revotron_cube_rl/runs'/NAME
DEST=ROOT/'docs/evidence/hold_increment_v8_wide128_seed20261105_actual_20261104'
CP='cba7ebc92496e6e32d33e83f29357677fcbf8d3037f456403a3734093dce82bb'
FAILURE='evaluation_protocol_error:Post-turn correction exceeded bounded pose envelope'
PINS={
 'v8_wide128_seed20261105_actual_actor_diagnostic_20261104/report.json':'ab7d6a53bba933d24f9ce9542cfc427955155b48f59d86030a14060c6468634f',
 'v8_wide128_seed20261105_actual_actor_diagnostic_20261104/branch_replay.jsonl':'1e43ea5cf0a3b00f63cec65ebf5c622a7a099ee955d725cde7e17ee3a04e3cd1',
 'exception_prefix_candidate_wide128_seed05_20261104_01.json':'a90afc2ee408d4fe58704eb2d92e9adb484f6addd37d6951a0186dcbfd6cb6aa',
 'exception_prefix_independent_review_20261104.json':'56ea5f4423692ca5a6edadec06c3f0b325ad19d1a5e5a9b28afc83a64cc4bdb6',
 'v8_wide128_seed20261105_actual_contact_diagnosis_20261104.json':'7063b3c2cd493d41ca1963f8e0f59e457da51d462e14b72d3cb146734d91e8ad',
 'v8_wide128_seed20261105_actual_hold_gate_decomposition_20261104.json':'bd714593a7aa8e9e3dd9571345af37d6ee90451dbc49edbd38ee0993054240f1',
 'v8_wide128_seed20261105_contact_mode_causal_join_20261104.json':'f0bc785850acfb5525181e949bf016e685886c7310978f494bb2a1e2caa1e89b',
 'v8_wide128_seed20261105_original_pose_envelope_reconstruction_20261104.json':'dbc87cfaa80f9f2def10d3dd73e89956968dc4dce6e3af5d65a07e3777747a28'}
ASSETS={
 'isaac_mujoco_cube/cube_model.usda':'c41baa3cd75db5a497c5878d0e18bb6b3316124e0f1935867de8305b44e4eb11',
 'revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd':'c066a65ff5ac77a8fb0a5fb3658c71f151b64777cdc9a8f66a0739d4033170a2',
 'revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda':'1ea674035491ae19875c61813ef8cfb1fd26c4b0b58de3ae3a2a3f04b741a143'}
SUPPORT=(
 'launch_hold_increment_v8_eval_20261104.sh','v8_wide128_seed20261105_actual_queue_20261104.py',
 'v8_wide128_seed20261105_actual_queue_20261104.json','v8_wide128_seed20261105_actual_queue_20261104.log',
 'v8_wide128_seed20261105_closed_loop_eval_20261104.log',
 'diagnose_hold_increment_actual_contact_v8.py','diagnose_hold_increment_wide128_actual_gates_v8.py',
 'diagnose_v8_wide128_seed05_pose_envelope.py','join_v8_wide128_actual_contact_modes.py',
 'replay_wide128_v8_actual_diagnostic_20261104.py','test_wide128_v8_actual_diagnostic_20261104.py',
 'exception_prefix_dataset_v1.py','test_exception_prefix_dataset_v1.py','test_exception_prefix_dataset_peer_v1.py',
 'exception_prefix_independent_host_tests_20261104_v2.log',
 'v10_saved_exception_evidence_no_gru_20261104.json',
 'v9_representation_only_on_wide128_seed05_actual_20261104.json',
 'v8_hidden64_and_wide128_actual_failure_comparison_20261104.json')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(path.read_text())
def require(ok,message):
 if not ok:raise ValueError(message)

def inspect():
 result=read(RUN/'result.json')
 require(sha(RUN/'result.json')=='8ac1cbe8d3225d841a909d293f1211da619dc71cb6591806c89674a047ddfb7c','Original result changed')
 require(result['physical_steps']==result['fully_audited_transition_steps']==result['steps']==result['teacher_query_count']==1803,'Actual counters changed')
 require(result['failure']==FAILURE and result['left_teacher_position_action_steps']==result['cube_pose_writes_during_control']==0,'Actual outcome/intervention changed')
 for field in ('original_wrist_gate_qualified','eligible_training_episode','pure_student_episode_claim','corrective_labels_qualified_for_training'):
  require(result[field]is False,'Original false flag promoted: '+field)
 require(result['evaluated_policy_checkpoint_sha256']==CP==sha(RUN/'checkpoints/evaluated_policy.pt'),'Actual model changed')
 original_manifest=read(RUN/'artifact_manifest.json')['sha256']
 for rel,h in original_manifest.items():require(sha(RUN/rel)==h,'Original artifact manifest differs: '+rel)
 count=0;abort=None;final=None
 for line in (RUN/'evaluation_transitions.jsonl').open():
  require(line.endswith('\n'),'Incomplete journal line');row=json.loads(line);kind=row['record_type']
  if kind=='executed_transition':
   require(abort is None and final is None and row['step']==count and row['transition_finished']is True,'Out-of-order executed transition');count+=1
  elif kind=='abort':require(abort is None and final is None,'Duplicate/late abort');abort=row
  elif kind=='evaluation_final':require(final is None and abort is not None,'Final ordering changed');final=row['summary']
  else:raise ValueError('Unknown journal record '+kind)
 require(count==1803 and final is not None and final['failure']==FAILURE,'Incomplete final episode')
 require(abort['step']==1803 and abort['physics_step_completed_for_unfinished_decision']is False,'Exception query executed physics')
 require(abort['unfinished_decision']=={'step':1803,'record_type':'decision_attempt','left_teacher_position_action_executed':False},'Exception query acquired invented evidence')
 for rel,h in PINS.items():require(sha(WORK/rel)==h,'Reviewed support changed: '+rel)
 for rel,h in ASSETS.items():require(sha(ROOT/rel)==h,'Baseline asset changed: '+rel)
 prefix=read(WORK/'exception_prefix_candidate_wide128_seed05_20261104_01.json')
 require(prefix['selected_completed_candidate_transitions']==1803 and prefix['training_qualification_granted']is False,'Prefix promoted or count changed')
 require(prefix['exception_query_after_state_known']is False and prefix['exception_query_label_fabricated']is False,'Exception evidence invented')
 files=sorted(p for p in RUN.rglob('*')if p.is_file());inventory=[str(p.relative_to(RUN))for p in files]
 items=[(p,Path('run')/p.relative_to(RUN),'complete_actual_run')for p in files]
 items +=[(WORK/n,Path('support')/n,'posthoc_evidence_or_source')for n in dict.fromkeys((*SUPPORT,*PINS))]
 items +=[(ROOT/n,Path('original_assets')/n,'original_pinned_baseline_asset')for n in ASSETS]
 items.append((Path(__file__).resolve(),Path('support')/Path(__file__).name,'archive_tool'))
 require(len({str(p.resolve())for p,_,_ in items})==len(items),'Duplicate source')
 for p,rel,_ in items:
  require(p.is_file()and not p.is_symlink()and p.resolve().is_relative_to(ROOT),'Missing/unsafe source '+str(p))
  require(not rel.is_absolute()and '..'not in rel.parts,'Unsafe archive path')
 return result,prefix,original_manifest,inventory,items

README='''# V8 hidden128 seed20261105 actual failed episode

This archive preserves every file from the completed 1,803-step ordinary-cube evaluation, including files created after its original artifact manifest. The model checkpoint is cba7ebc9..., maximum recorded lift is approximately 0.368715 m, and the original wrist gate did not pass. The original teacher raised "Post-turn correction exceeded bounded pose envelope" before returning the next action. The failed query has no controller audit, after-state, model action or supervision label; no physics was executed for it. All original False qualification and whole-episode flags remain unchanged.

The archived diagnostic replays all 1,803 actual 284-feature recurrent inputs and executed actions, preserving their bitwise evidence. Its branch journal and the contact, wrist-gate, original pose-envelope and causal-join analyses remain posthoc diagnostics. The separate reviewed exception-prefix consumer rechecks completed safe transitions and excludes the unreturned query. Its report grants no training authorization; V10 later admits only the completed prefix under its own qualification. This archive does not modify the failed original result or promote it to success.

Original source snapshots, plans, checkpoints, three pinned baseline assets and all run files are retained exactly, with restored-byte hashes for compressed files. The environment is the ordinary rigid-body cube baseline, not the native-articulation branch. No new simulation, model training, neural generalization, right-hand rotation or cube-solving success is claimed.
'''

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
 result,prefix,original_manifest,inventory,items=inspect()
 if not args.execute:
  print(json.dumps({'ready':True,'destination':str(DEST),'items':len(items),'run_files':len(inventory),'raw_bytes':sum(p.stat().st_size for p,_,_ in items)}));return
 require(not DEST.exists(),'Archive already exists; never overwrite');DEST.mkdir(parents=True)
 entries=[]
 for source,relative,category in items:
  digest=sha(source);size=source.stat().st_size
  compressed=source.suffix in ('.jsonl','.log','.usda')or(source.suffix=='.json'and size>32768)
  target=DEST/(str(relative)+('.gz'if compressed else''));target.parent.mkdir(parents=True,exist_ok=True)
  if compressed:
   with source.open('rb')as src,target.open('wb')as raw:
    with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0,compresslevel=6)as out:shutil.copyfileobj(src,out,1024*1024)
   h=hashlib.sha256()
   with gzip.open(target,'rb')as restored:
    for b in iter(lambda:restored.read(1024*1024),b''):h.update(b)
   require(h.hexdigest()==digest,'Restored gzip bytes differ')
  else:shutil.copyfile(source,target);require(sha(target)==digest,'Copy bytes differ')
  require(sha(source)==digest and source.stat().st_size==size,'Source changed while archiving')
  entries.append({'source':str(source.relative_to(ROOT)),'archive_file':str(target.relative_to(DEST)),'category':category,
   'encoding':'gzip_mtime_zero'if compressed else'original_bytes','source_sha256':digest,'source_bytes':size,
   'archive_sha256':sha(target),'archive_bytes':target.stat().st_size,'raw_bytes_verified_after_copy_or_decompression':True})
 require(inventory==[str(p.relative_to(RUN))for p in sorted(RUN.rglob('*'))if p.is_file()],'Late run inventory changed')
 (DEST/'README.md').write_text(README,encoding='utf-8',newline='\n')
 manifest={'schema':'complete_failed_v8_wide128_actual_and_exception_prefix_evidence_v1','entries':entries,
  'generated_files':{'README.md':sha(DEST/'README.md')},'complete_run_file_inventory':inventory,'run':str(RUN.relative_to(ROOT)),
  'original_run_artifact_manifest_sha256':sha(RUN/'artifact_manifest.json'),'original_artifact_count':len(original_manifest),
  'result_sha256':sha(RUN/'result.json'),'checkpoint_sha256':CP,'actual_steps':1803,'returned_teacher_queries':1803,
  'failed_unreturned_query_step':1803,'failed_query_physics_executed':False,'failed_query_label_fabricated':False,
  'failure':FAILURE,'maximum_lift_m':result['maximum_lift_m'],'left_teacher_position_action_steps':0,
  'original_gain_event_count':result['original_teacher_gain_schedule_events'],'cube_pose_writes_during_control':0,
  'original_results_modified':False,'original_failure_flags_preserved':True,'counterfactual_labels_qualified_by_this_archive':False,
  'whole_episode_training_qualified':False,'closed_loop_wrist_success':False,'face_turn_or_cube_solving_success':False,
  'independent_generalization_success':False,'git_stage_commit_or_push_performed':False,'pinned_posthoc_evidence':PINS,'pinned_original_assets':ASSETS}
 path=DEST/'archive_manifest.json';path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
 print(json.dumps({'archive':str(DEST),'manifest_sha256':sha(path),'entries':len(entries),'files':len(entries)+2,
  'raw_bytes':sum(e['source_bytes']for e in entries),'archive_bytes':sum(e['archive_bytes']for e in entries)}))

if __name__=='__main__':main()
