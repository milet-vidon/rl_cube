from pathlib import Path
import os,sys,time,json,hashlib,subprocess,fcntl
ROOT=Path('/home/brainco/rl_cube/github_project');CODE=ROOT/'revotron_cube_rl';WORK=ROOT/'work/teacher_optimization_20261103';status=WORK/'native_fulltask_v18_queue.json'
def write(state,**extra):
 p=status.with_suffix('.tmp');p.write_text(json.dumps({'state':state,'updated_unix':time.time(),**extra},indent=2));os.replace(p,status)
expected={'single_face_experiment.py':'36d98c64fbeb0110506d11aa93b98a3fa392e198716c5a2d0b86ce815e69917f','centered_opposition_runtime.py':'30e22dbb33e1161cf86662e6cd2b98acf850482327df476dffca24d6ea0eaf33','opposed_grasp_runtime.py':'0b3de71cf3c5df1ae30bc046baaae087797d3af3d70a1d243cf24150d84f458c','full_task_episode.py':'937df2b0d02f7a23b69d863a2a7977e86a4b188743bf9c2a7b0cafb46b8e03d4','passive_cube_articulation.py':'949c3ede71c6ac4cd32b359b08ab110c72c354888d32541136aa2330edce980c'}
try:
 with (WORK/'isaac_eval_queue.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  prior=json.loads((WORK/'left_dagger_baseline_mixture_v3_status.json').read_text())
  while (Path('/proc')/str(prior['pid'])).exists():write('waiting_for_mixture_exit',pid=prior['pid']);time.sleep(15)
  prior_run=CODE/'runs/left_dagger_baseline_mixture_v3_9501_20261104'
  if not (prior_run/'result.json').exists():raise RuntimeError('Mixture exited without a complete result; inspect before next simulator')
  for n,h in expected.items():
   if hashlib.sha256((CODE/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen native/fulltask source changed: '+n)
  active=[];names={'single_face_experiment.py','collect_left_dagger_baseline_v3.py','collect_left_dagger_baseline_v2.py','evaluate_phase_basis_v4.py','passive_cube_env_smoke.py','finite_angle_cube_native_probe.py'}
  for p in Path('/proc').iterdir():
   if not p.name.isdigit() or p.name==str(os.getpid()):continue
   try:args=(p/'cmdline').read_bytes().split(b'\0')
   except OSError:continue
   if any(Path(x.decode(errors='replace')).name in names for x in args):active.append(p.name)
  if active:raise RuntimeError('Another simulator active: '+str(active))
  name='teacher_native_fulltask_v18_20261104'
  if (CODE/'runs'/name).exists():raise RuntimeError('Refuse to overwrite native teacher run')
  command=[sys.executable,'single_face_experiment.py','--name',name,'--seed','9501','--max-steps','4500','--controller','opposed','--push-degrees','15','--center-style','j56','--opposition-planner','joint','--ring-inset','0.001','--right-hand-friction','1.2,1.0','--right-finger-stiffness','10','--position-iterations','64','--cube-clearance-refinement','--turn-feedback-pacing','--turn-feedback-planning-frame','--left-ring-support','--preflight-workers','12','--full-task-episode','--passive-cube-articulation','--articulation-baseline',str(WORK/'articulation_freefall_probe_v1_20261103/baseline/result.json'),'--articulation-baseline-sha256','0bc715f5f009a9dcf6b90c7d4f5f97dc22043354aebe403aa5a2ad94ed9615c8']
  with (WORK/'native_fulltask_v18.log').open('w') as log:
   p=subprocess.Popen(command,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write('running',pid=p.pid,command=command,input_sha256=expected,scope='native passive physics candidate and explicit full 15deg task; fixed 4500 total control-step horizon; not a single-parameter ablation');code=p.wait()
  rp=CODE/'runs'/name/'result.json';r=json.loads(rp.read_text()) if rp.exists() else None
  write('complete' if code==0 and r is not None else 'failed',returncode=code,result_status={k:r.get(k) for k in ('steps','failure','face_turn_validated','task_episode')} if r else None)
except BaseException as e:
 write('queue_error',error=str(e));raise
