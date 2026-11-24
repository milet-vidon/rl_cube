"""Complete the control-deadband trial, then train only from accepted evidence."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path('/home/brainco/rl_cube/github_project/revotron_cube_rl');WORK=ROOT.parent/'work/teacher_optimization_20261103'
sys.path.insert(0,str(ROOT))
from teacher_bootstrap import dataset
while Path('/proc/327328').exists():time.sleep(5)
status=[]
def run(name,command):
    print('START',name,flush=True)
    with (WORK/(name+'.log')).open('w') as log:rc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    status.append({'run':name,'returncode':rc});(WORK/'settled_pipeline_status.json').write_text(json.dumps(status,indent=2))
    print('EXIT',name,rc,flush=True)
    return rc
name='teacher_coverage_9901_20261103_settled_v4'
run(name,[str(ROOT/'run_linux.sh'),'center-teacher','--center-style','j56','--adaptive-wrist','--settled-hold','--name',name,'--seed','9901'])
source=ROOT/'runs'/name
try:dataset(source,'applied_target')
except Exception as e:
    print('SOURCE_REJECTED',str(e),flush=True);sys.exit(0)
base=ROOT/'runs/center_j56_dev_v3_20261103';output=ROOT/'runs/teacher_left_bc_multiseed_v9_20261103'
if run(output.name,[sys.executable,'-u',str(ROOT/'teacher_bootstrap.py'),'--source',str(base),'--extra-source',str(source),'--output',str(output),'--updates','100000','--state-noise','.005','--box-output','--applied-targets'])==0:
    name='teacher_left_bc_multiseed_v9_eval_9901'
    run(name,[sys.executable,'-u',str(ROOT/'evaluate_teacher_bootstrap.py'),'--checkpoint',str(output/'best.pt'),'--source',str(base),'--output',str(ROOT/'runs'/name),'--seed','9901'])
