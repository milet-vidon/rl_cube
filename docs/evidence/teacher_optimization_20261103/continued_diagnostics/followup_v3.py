"""Serialized PhysX follow-up; successful evidence only enters additional BC."""
from pathlib import Path
import json, subprocess, sys, time
ROOT=Path('/home/brainco/rl_cube/github_project/revotron_cube_rl')
WORK=ROOT.parent/'work/teacher_optimization_20261103'
sys.path.insert(0,str(ROOT))
from teacher_bootstrap import dataset
status=[]
def run(name,argv):
    with (WORK/(name+'.log')).open('w') as log:
        rc=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    status.append({'name':name,'returncode':rc})
    (WORK/'followup_v3_status.json').write_text(json.dumps(status,indent=2))
    print('COMPLETED',name,rc,flush=True)
    return rc
while Path('/proc/319429').exists():time.sleep(5)
name='teacher_coverage_9901_20261103_bounded_v3'
run(name,[str(ROOT/'run_linux.sh'),'center-teacher','--center-style','j56','--adaptive-wrist','--name',name,'--seed','9901'])
while Path('/proc/323388').exists():time.sleep(5)
base=ROOT/'runs/center_j56_dev_v3_20261103'
model=ROOT/'runs/teacher_left_bc_applied_v7_20261103'
if (model/'result.json').exists():
    name='teacher_left_bc_applied_v7_eval_9501'
    run(name,[sys.executable,'-u',str(ROOT/'evaluate_teacher_bootstrap.py'),'--checkpoint',str(model/'best.pt'),'--source',str(base),'--output',str(ROOT/'runs'/name),'--seed','9501'])
name='teacher_joint_traction_v3_20261103'
run(name,[str(ROOT/'run_linux.sh'),'face-turn','--opposition-planner','joint','--ring-inset','.001','--center-style','j56','--hand-friction','1.2,1.0','--name',name,'--seed','9501'])
accepted=[]
for seed in (9901,9502,9503):
    for suffix in ('bounded_v3','adaptive_v2','v1'):
        candidate=ROOT/'runs'/f'teacher_coverage_{seed}_20261103_{suffix}'
        try:dataset(candidate,'applied_target')
        except (ValueError,FileNotFoundError,KeyError):continue
        accepted.append(candidate);break
(WORK/'followup_v3_sources.json').write_text(json.dumps([str(p) for p in accepted],indent=2))
if accepted:
    output=ROOT/'runs/teacher_left_bc_multiseed_v8_20261103'
    args=[sys.executable,'-u',str(ROOT/'teacher_bootstrap.py'),'--source',str(base),'--output',str(output),'--updates','100000','--state-noise','.005','--box-output','--applied-targets']
    for source in accepted:args+=['--extra-source',str(source)]
    if run(output.name,args)==0:
        seed=json.loads((accepted[0]/'result.json').read_text())['seed'];name=f'teacher_left_bc_multiseed_v8_eval_{seed}'
        run(name,[sys.executable,'-u',str(ROOT/'evaluate_teacher_bootstrap.py'),'--checkpoint',str(output/'best.pt'),'--source',str(base),'--output',str(ROOT/'runs'/name),'--seed',str(seed)])
else:print('NO_QUALIFIED_NEW_DEMONSTRATION; no multiseed promotion',flush=True)
