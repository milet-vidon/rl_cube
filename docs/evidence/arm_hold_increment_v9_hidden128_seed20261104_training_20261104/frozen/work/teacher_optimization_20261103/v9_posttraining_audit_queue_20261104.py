"""Only audit completed V9 artifacts; no simulation or optimizer."""
from pathlib import Path
import hashlib,json,subprocess,time,sys
W=Path('/home/brainco/rl_cube/github_project/work/teacher_optimization_20261103')
C=W.parents[1]/'revotron_cube_rl'
hidden=int(sys.argv[1]);assert hidden in (64,128)
seed=20261104
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
review=W/f'v9_hidden{hidden}_root_training_review_20261104.json'
expected={64:'4f0bc359b0c560aaa6ea9a74e28f2ab31b563df1ab2b205a5c01ce6f6cf6f68a',128:'6500760e5414d42aa4280b09b224386856d2b3dad6f408a87bffef8f92ed41fa'}
assert sha(review)==expected[hidden]
job=json.loads(review.read_text())
run=C/job['run'];audit=W/f'v9_hidden{hidden}_seed{seed}_completed_cpu_audit_20261104.json'
state=W/f'v9_hidden{hidden}_seed{seed}_posttraining_queue_20261104.json'
assert not state.exists() and not audit.exists()
def save(status,**kw):state.write_text(json.dumps(dict(status=status,seed=seed,hidden_size=hidden,time=time.time(),**kw),indent=2)+'\n')
try:
 save('waiting_for_training_result')
 while not (run/'result.json').exists():time.sleep(15)
 r=json.loads((run/'result.json').read_text());model=run/'model.pt';model_sha=sha(model)
 assert model_sha==r['checkpoint_sha256']
 tool=W/'audit_completed_arm_hold_increment_v9.py';assert sha(tool)=='19a1bfbffda935dac166456ed112f0766d0f1f2a5ea9f50901285931677cf223'
 py=Path('/home/brainco/rl_cube/.conda/rl_cube_6_2/bin/python')
 q=W/'v9_trainer_preflight_20261104_01/qualification_candidate.json'
 assert sha(q)=='6283a3d6e1a1feccea809c5150eac49aaa5784299cfb48a8d8b396be07f07cf7'
 assert sha(Path(job['authorization']))==job['authorization_sha256']
 save('auditing',checkpoint_sha256=model_sha)
 command=[str(py),'-u','-c',"import sys,runpy;sys.path.append('../work/teacher_optimization_20261103');runpy.run_path('../work/teacher_optimization_20261103/audit_completed_arm_hold_increment_v9.py',run_name='__main__')",'--checkpoint',str(model),'--checkpoint-sha256',model_sha,'--qualification',str(q),'--qualification-sha256',sha(q),'--authorization',job['authorization'],'--authorization-sha256',job['authorization_sha256'],'--output',str(audit)]
 subprocess.run(command,cwd=C,check=True)
 verdict=json.loads(audit.read_text());assert verdict['offline_candidate_qualified'] is True
 launcher=W/'launch_arm_hold_increment_v9_eval_20261104.sh';assert sha(launcher)=='220b84a5805a606b462d45cbe462da1bd4070acda40df1746091f236ee5c4d67'
 preflight=W/f'v9_hidden{hidden}_seed{seed}_completed_eval_preflight_20261104';assert not preflight.exists()
 save('evaluation_preflight_only',checkpoint_sha256=model_sha,cpu_audit_sha256=sha(audit))
 subprocess.run(['bash',str(launcher),str(model),model_sha,job['authorization'],job['authorization_sha256'],str(preflight),'preflight'],cwd=C,check=True)
 save('complete_ready_for_root_physical_evaluation_review',checkpoint_sha256=model_sha,cpu_audit_sha256=sha(audit),evaluation_preflight=str(preflight))
except Exception as e:
 save('failed',error=repr(e));raise
