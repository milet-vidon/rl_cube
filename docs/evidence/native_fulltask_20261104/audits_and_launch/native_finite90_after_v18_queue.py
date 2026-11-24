import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
root=Path("/home/brainco/rl_cube/github_project")
w=root/"work/teacher_optimization_20261103"
status=w/"native_finite90_after_v18_queue.json"
def write(d):status.write_text(json.dumps(d,indent=2)+"\n")
def live(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
expected={"finite_angle_cube_native_probe.py":"fc2c49a3464228d8ad2c0d8fb96c123ea1776b297c55a9a959fb0289d0faa870","finite_cube_pose_math.py":"1527f6c7e29e28e8264e12c65aed5ef2d4c82330a370f709359157094ce73940","passive_cube_articulation.py":"949c3ede71c6ac4cd32b359b08ab110c72c354888d32541136aa2330edce980c","cube_articulation_probe_v2.py":"bea19566ff4071dd26882f578703995ed7a74856f3218507fe9b3df8bee77881"}
write({"state":"waiting_for_v18_exit","pid":os.getpid(),"wait_pid":1178403,"scope":"reset-only native90 pose representation; not robot turn","source_sha256":expected})
try:
 while live(1178403):time.sleep(5)
 r=json.loads((root/"revotron_cube_rl/runs/teacher_native_fulltask_v18_20261104/result.json").read_text())
 if not r.get("task_episode",{}).get("done"):raise RuntimeError("v18 has no audited terminal result; manual investigation required")
 for n,h in expected.items():
  if hashlib.sha256((w/n).read_bytes()).hexdigest()!=h:raise RuntimeError("Frozen probe source changed: "+n)
 output=w/"native_initial_face90_probe_v1_20261104"
 if output.exists():raise RuntimeError("Unique output already exists")
 command=[sys.executable,str(w/"finite_angle_cube_native_probe.py"),"--execute-physics","--face","nZ","--angle-degrees","90","--steps","24","--output",str(output)]
 with (w/"native_finite90_after_v18.log").open("w") as log:
  child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  write({"state":"running_finite90_probe","pid":os.getpid(),"child_pid":child.pid,"command":command,"source_sha256":expected,"robot_turn_success_claim":False})
  code=child.wait()
 write({"state":"complete" if code==0 else "failed","returncode":code,"source_sha256":expected,"robot_turn_success_claim":False,"workflow":json.loads((output/"workflow.json").read_text()) if (output/"workflow.json").exists() else None})
except BaseException:
 write({"state":"failed","error":traceback.format_exc()});raise
