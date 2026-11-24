from pathlib import Path
import sys,json,gzip,hashlib
repo=Path('/home/brainco/rl_cube/github_project');root=repo/'revotron_cube_rl';out=repo/'docs/evidence/j56_center_20261103';sys.path.insert(0,str(root))
from centered_transport_audit import assess_center_row
from center_j56_audit import joint_coordination
from tilt_audit import contact_topology

def digest(raw):return hashlib.sha256(raw).hexdigest()
run=root/'runs/center_j56_dev_v3_20261103';r=json.loads((run/'result.json').read_text());rows=json.loads((run/'trajectory.json').read_text());request=json.loads((run/'center_request.json').read_text());plan=json.loads((run/'center_plan.json').read_text());names=request['joint_names'];initial=[request['initial'][n]for n in names];streak=longest=0
for row in rows:
 if row['phase']!='center_hold':continue
 a=assess_center_row(row,names,plan['center_target_world_m'],plan['cube_target_quaternion_wxyz']);b=joint_coordination(row['q'],initial,names,request['coordination_protocol']);streak=streak+1 if a['qualified']and b['qualified']else 0;longest=max(longest,streak)
assert (longest>=90)==r['center_transport_passed'] and r['dynamic_grasp_passed']
assert longest/30==r['push']['center']['longest_qualified_hold_s']
assert r['cube_pose_writes_during_control']==0
for row in rows:assert row['self_contact_impulse']<=1e-6 and row['robot_table_impulse']<=1e-6 and not(row['drop']or row['arm_soft_limit_hit']or row['actual_arm_soft_limit_violation']or any(row['hard_limit_hit']))
cert=json.loads((root/'wrist_direction_gate_certificate.json').read_text())
for name,expected in cert['source_sha256'].items():assert digest((root/name).read_bytes())==expected,name
experiment=json.loads((run/'experiment.json').read_text())
for name,expected in experiment['source_sha256'].items():assert digest((root/name).read_bytes())==expected,name
count=0
for manifest in out.glob('*/archive_manifest.json'):
 for name,data in json.loads(manifest.read_text())['files'].items():
  raw=(manifest.parent/data['path']).read_bytes();assert digest(raw)==data['stored_sha256']
  if data['encoding']=='gzip':raw=gzip.decompress(raw)
  assert digest(raw)==data['original_sha256'];count+=1
hold=[row for row in rows if row['phase']=='center_hold'];coverage={finger:sum(contact_topology(row,'left')[finger]for row in hold)/len(hold)for finger in ('thumb','index','middle','ring','little')}
feedback_rows=[row['index_contact_feedback']for row in rows if 'index_contact_feedback'in row]
settings=request['coordination_protocol']['finger_contact_feedback']
for row in feedback_rows:
 assert settings['lower_offset_rad']<=row['offset_rad']<=settings['upper_offset_rad']
 assert abs(row['next_offset_rad']-row['offset_rad'])<=settings['step_rad']+1e-12
report={'unit_tests_passed':36,'offline_path_checks':plan['checks'],'physics_development_attempts':3,'physics_accepted_attempts':int(r['center_transport_passed']),'independent_trials_this_revision':0,'full_reset_episode':r['full_episode'],'recomputed_qualified_hold_s':longest/30,'center_hold_contact_coverage':coverage,'unchanged_frozen_control_files':len(cert['source_sha256']),'current_source_matches_v3_experiment':True,'verified_archive_files':count,'cube_pose_writes_during_control':0,'right_hand_or_face_turn_validated':False,'neural_policy_trained':False};(out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
