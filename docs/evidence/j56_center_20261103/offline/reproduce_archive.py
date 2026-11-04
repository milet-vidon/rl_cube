from pathlib import Path
import json,gzip,hashlib,shutil,sys
repo=Path('/home/brainco/rl_cube/github_project');root=repo/'revotron_cube_rl';run=root/'runs'/(sys.argv[1]if len(sys.argv)>1 else 'center_j56_dev_v3_20261103');out=repo/'docs/evidence/j56_center_20261103';result=json.loads((run/'result.json').read_text());out.mkdir(parents=True,exist_ok=True);dest=out/run.name;dest.mkdir(exist_ok=True)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
experiment=json.loads((run/'experiment.json').read_text());original=json.loads((run/'artifact_manifest.json').read_text())['sha256'];mapping={}
selected=set(experiment['source_archive_paths'].values())|{'config.json','result.json','experiment.json','asset_contract.json','runtime.json','runtime_robot_asset.json','cube_asset.json','center_result.json','center_request.json','center_plan.json','left_plan.json','trajectory.json','artifact_manifest.json','center_preflight.log'}
for name in sorted(selected):
 source=run/name
 if not source.exists():continue
 if name in original:assert sha(source)==original[name],name
 compressed=name in ('trajectory.json','left_plan.json','center_plan.json','center_request.json');target=dest/(name+'.gz'if compressed else name)
 if compressed:
  with source.open('rb')as f,target.open('wb')as g:
   with gzip.GzipFile(filename='',fileobj=g,mode='wb',mtime=0)as zipped:shutil.copyfileobj(f,zipped)
 else:shutil.copy2(source,target)
 mapping[name]={'path':target.name,'encoding':'gzip'if compressed else 'raw','original_sha256':sha(source),'stored_sha256':sha(target)}
for name,expected in experiment['source_sha256'].items():assert mapping[experiment['source_archive_paths'][name]]['original_sha256']==expected,name
(dest/'archive_manifest.json').write_text(json.dumps({'source_run':run.name,'files':mapping,'omitted':{name:'Duplicate journal, generated cube asset or full movie; source, complete trajectory and hashes retained.'for name in original if name not in mapping}},indent=2))
rows=json.loads((run/'trajectory.json').read_text());old=root/'runs/centered_opposition_center_dev_v2_20261103';oldrows=json.loads((old/'trajectory.json').read_text());names=json.loads((run/'config.json').read_text())['joint_names'];oldnames=json.loads((old/'config.json').read_text())['joint_names'];last=rows[-1]
comparison={'seed':9501,'split':'development','previous_run':old.name,'new_run':run.name,'old_configuration_sha256':sha(old/'config.json'),'new_configuration_sha256':sha(run/'config.json'),'old_final_joint_deg':{f'J{i}':float(oldrows[-1]['q'][oldnames.index(f'Joint{i}_L')]*180/3.141592653589793)for i in range(1,8)},'new_final_joint_deg':{f'J{i}':float(last['q'][names.index(f'Joint{i}_L')]*180/3.141592653589793)for i in range(1,8)},'new_center':result['push'].get('center'),'center_transport_passed':result.get('center_transport_passed',False),'failure':result['failure'],'self_contact_impulse_sum_Ns':result['self_contact_impulse_sum_Ns'],'robot_table_impulse_sum_Ns':result['robot_table_impulse_sum_Ns'],'soft_limit_hit_steps':sum(bool(r['arm_soft_limit_hit']or r['actual_arm_soft_limit_violation'])for r in rows),'hard_limit_hit_steps':sum(any(r['hard_limit_hit'])for r in rows),'drop_steps':sum(bool(r['drop'])for r in rows),'elapsed_s':result['elapsed_s'],'cube_pose_writes_during_control':result['cube_pose_writes_during_control'],'independent_trials_this_revision':0,'previous_independent_pipeline_successes':0,'previous_independent_pipeline_total':3,'right_regrasp_validated':False,'face_turn_validated':False,'neural_policy_trained':False};(out/(run.name+'_comparison.json')).write_text(json.dumps(comparison,indent=2))
offline=out/'offline';offline.mkdir(exist_ok=True)
for name in ('probe.py','candidates.json','collision_candidates.json','right_feasibility.py','right_feasibility.json','check_index_adjustment.py','j56_request_v2.json','j56_plan_v2.json','j56_request_v3.json','j56_plan_v3.json','contact_summary.json'):
 source=Path('/tmp/j56_center_20261103')/name
 if source.is_file():
  target=offline/(name+'.gz'if source.stat().st_size>300000 else name)
  if target.suffix=='.gz':target.write_bytes(gzip.compress(source.read_bytes(),mtime=0))
  else:shutil.copy2(source,target)
print(json.dumps({'archived':str(out),'source_files':len(experiment['source_sha256']),'passed':comparison['center_transport_passed'],'final_joints':comparison['new_final_joint_deg']}))
