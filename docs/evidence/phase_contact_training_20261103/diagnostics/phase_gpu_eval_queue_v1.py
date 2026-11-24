"""One queued development evaluation after current Isaac exits; no concurrent simulator."""
from pathlib import Path
import subprocess,time,json,sys,os,fcntl
ROOT=Path('/home/brainco/rl_cube/github_project');CODE=ROOT/'revotron_cube_rl';WORK=ROOT/'work/teacher_optimization_20261103'
status=WORK/'phase_gpu_eval_queue_v1.json'
def write(state,**kw):
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'updated_unix':time.time(),**kw},indent=2));os.replace(tmp,status)
def active():
 names={'single_face_experiment.py','evaluate_teacher_bootstrap.py','evaluate_phase_contact.py'}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=(p/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if any(Path(a.decode(errors='replace')).name in names for a in args):return True
 return False
with (WORK/'isaac_eval_queue.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 training=CODE/'runs/teacher_phase_contact_gpu_v1_20261103'
 while not (training/'result.json').exists() or active():
  write('waiting_for_checkpoint_and_simulator');time.sleep(15)
 r=json.loads((training/'result.json').read_text())
 if not r['completed']:raise RuntimeError('Training did not complete')
 out='runs/teacher_phase_contact_gpu_v1_eval_9501'
 if (CODE/out).exists():raise RuntimeError('Refusing to overwrite evaluation')
 command=[sys.executable,'evaluate_phase_contact.py','--checkpoint',str(training/'best.pt'),'--source','runs/center_j56_dev_v3_20261103','--output',out,'--seed','9501']
 with (WORK/'phase_contact_gpu_v1_eval_9501.log').open('w') as log:
  p=subprocess.Popen(command,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write('evaluating',pid=p.pid,command=command);code=p.wait()
 result=CODE/out/'result.json';write('evaluation_complete' if code==0 and result.exists() else 'evaluation_error',returncode=code,result=json.loads(result.read_text()) if result.exists() else None)
