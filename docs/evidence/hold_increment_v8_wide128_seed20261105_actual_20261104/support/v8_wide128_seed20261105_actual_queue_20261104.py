from pathlib import Path
import hashlib,json,subprocess,time
W=Path('/home/brainco/rl_cube/github_project/work/teacher_optimization_20261103');C=W.parents[1]/'revotron_cube_rl'
STATE=W/'v8_wide128_seed20261105_actual_queue_20261104.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(status,**kw):STATE.write_text(json.dumps(dict(status=status,time=time.time(),**kw),indent=2)+'\n')
def sims():
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   a=(p/'cmdline').read_bytes().split(b'\x00');c=b' '.join(a).decode(errors='replace')
   if 'python' in Path(a[0].decode(errors='replace')).name and any(t in c for t in ['evaluate_state_','single_face_experiment.py','replay_native_v20','bilateral_contact_sign_smoke']):found.append(int(p.name))
  except(FileNotFoundError,PermissionError):pass
 return found
try:
 assert not STATE.exists()
 pins={'v8_wide128_seed20261105_completed_cpu_audit_20261104.json':'109ed880059dc0ee581e37916334dc145c610977c02c5aae6097608f53a9e55f','v8_wide128_seed20261105_completed_eval_preflight_20261104/evaluation_preflight.json':'ce0bca4dcbdb9c01c7e5776ed7b6bc0e39d564312433e704f4581dfb5798f4ba','v8_wide128_seed20261105_root_training_authorization_20261104.json':'0a5a9963c32a062c0cf13e0085ee3c25463bb0a7f62b4873a8717b7611147a91','launch_hold_increment_v8_eval_20261104.sh':'654361c1894d3a6e9a4734bbea76ad1f349b524b9f0fae61050099feb862ec6b'}
 for n,h in pins.items():assert sha(W/n)==h,n
 audit=json.loads((W/'v8_wide128_seed20261105_completed_cpu_audit_20261104.json').read_text())
 assert audit['offline_candidate_qualified'] and audit['completed_metadata_metrics_reproduced_exactly']
 pre=json.loads((W/'v8_wide128_seed20261105_completed_eval_preflight_20261104/evaluation_preflight.json').read_text())
 assert pre['baseline_runtime_source_root']==str(C)and pre['checkpoint_source_dependency_count']==18 and pre['all_checkpoint_source_dependencies_snapshotted_exactly']
 model=C/'runs/teacher_hold_increment_v8_wide128_seed20261105_20261104/model.pt';model_sha='cba7ebc92496e6e32d33e83f29357677fcbf8d3037f456403a3734093dce82bb';assert sha(model)==model_sha
 output='runs/left_hold_increment_v8_wide128_seed20261105_eval_9501_20261104';assert not(C/output).exists()
 save('waiting_for_native_off_v3_exit',reviewed_pins=pins)
 while Path('/proc/62329').exists():time.sleep(15)
 assert (C/'runs/native_v20_raw_replay_off_actual_v3_20261104/raw_replay_result.json').exists(),'OFF final evidence missing; root must investigate'
 assert not sims(),('Other simulator active',sims())
 for n,h in pins.items():assert sha(W/n)==h,n
 log=W/'v8_wide128_seed20261105_closed_loop_eval_20261104.log'
 with log.open('xb')as f:
  p=subprocess.Popen(['bash',str(W/'launch_hold_increment_v8_eval_20261104.sh'),str(model),model_sha,str(W/'v8_wide128_seed20261105_root_training_authorization_20261104.json'),pins['v8_wide128_seed20261105_root_training_authorization_20261104.json'],output,'evaluate'],cwd=C,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  save('running',pid=p.pid,run=output,checkpoint_sha256=model_sha,reviewed_pins=pins)
  code=p.wait()
 save('finished',exit_code=code,run=output,checkpoint_sha256=model_sha,reviewed_pins=pins)
except Exception as e:
 save('failed',error=repr(e));raise
