"""One bounded simulation follow-up; never trains on unsuccessful trajectories."""
from pathlib import Path
import hashlib, json, os, subprocess, time

ROOT=Path('/home/brainco/rl_cube/github_project')
CODE=ROOT/'revotron_cube_rl'
WORK=ROOT/'work/teacher_optimization_20261103'
SOURCE=CODE/'runs/teacher_joint_swept_clearance_v8_20261103'
NAME='teacher_joint_matched_friction_v9_20261103'
STATUS=WORK/'contact_comparison_status.json'

def status(state, **kwargs):
    data={'state':state,'time_unix':time.time(),'source_run':SOURCE.name,
          'next_run':NAME,'scope':'simulation development only; no training',**kwargs}
    temporary=STATUS.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2));temporary.replace(STATUS)

def running_isaac_experiments():
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:fields=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(f==b'single_face_experiment.py' or f.endswith(b'/single_face_experiment.py') for f in fields):result.append(int(proc.name))
    return result

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

try:
    status('waiting_for_v8_completion')
    deadline=time.monotonic()+3600
    while not (SOURCE/'artifact_manifest.json').exists() or running_isaac_experiments():
        if time.monotonic()>deadline:raise TimeoutError('Follow-up wait expired; no existing process was stopped')
        time.sleep(5)
    result=json.loads((SOURCE/'result.json').read_text())
    permitted_failures=('right_five_finger_hold_gate_not_met','opposed_face_turn_hold_gate_not_met')
    eligible=(result['full_episode'] and not result['execution_interrupted']
              and result['center_transport_passed'] and result['failure'] in permitted_failures
              and result['cube_pose_writes_during_control']==0
              and result['robot_table_impulse_sum_Ns']<=1e-6
              and result['self_contact_impulse_sum_Ns']<=1e-6)
    if not eligible:
        status('stopped_after_v8_review_required',failure=result['failure'],steps=result['steps'])
    else:
        manifest=json.loads((SOURCE/'artifact_manifest.json').read_text())
        for name, expected in manifest['sha256'].items():
            if digest(SOURCE/name)!=expected:raise ValueError('Prior evidence hash mismatch: '+name)
        experiment=json.loads((SOURCE/'experiment.json').read_text())
        for name, expected in experiment['source_sha256'].items():
            if digest(CODE/name)!=expected:raise ValueError('Code changed before comparison: '+name)
        if (CODE/'runs'/NAME).exists():raise FileExistsError(NAME)
        if running_isaac_experiments():raise RuntimeError('Another Isaac experiment started')
        command=['./run_linux.sh','face-turn','--opposition-planner','joint','--ring-inset','.001',
                 '--center-style','j56','--right-hand-friction','1.2,1.0','--right-finger-stiffness','20',
                 '--position-iterations','64','--cube-clearance-refinement',
                 '--matched-internal-friction','.4,.3','--name',NAME,'--seed','9501']
        status('running_v9',command=command,source_failure=result['failure'],source_sha256=experiment['source_sha256'])
        with (WORK/'matched_friction_v9.log').open('x') as log:
            completed=subprocess.run(command,cwd=CODE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        path=CODE/'runs'/NAME/'result.json'
        outcome=json.loads(path.read_text()) if path.exists() else None
        status('completed_v9' if outcome else 'v9_runtime_failed',returncode=completed.returncode,
               failure=outcome['failure'] if outcome else None,
               steps=outcome['steps'] if outcome else None,
               push=outcome['push'] if outcome else None)
except Exception as error:
    status('failed',error=type(error).__name__+': '+str(error))
    raise
