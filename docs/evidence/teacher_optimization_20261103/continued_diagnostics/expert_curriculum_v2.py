"""Single-instance development collection with gated adaptive-wrist retries."""
from pathlib import Path
import json,subprocess,time,sys,shutil
ROOT=Path('/home/brainco/rl_cube/github_project/revotron_cube_rl')
WORK=ROOT.parent/'work/teacher_optimization_20261103'
sys.path.insert(0,str(ROOT))
from teacher_bootstrap import dataset
while Path('/proc/316600').exists():time.sleep(5)
accepted=[];attempts=[]
for seed in (9901,9502,9503):
    for adaptive in (False,True):
        name=f'teacher_coverage_{seed}_20261103_'+('adaptive_v2' if adaptive else 'v1')
        run=ROOT/'runs'/name
        if not run.exists():
            command=[str(ROOT/'run_linux.sh'),'center-teacher','--center-style','j56','--name',name,'--seed',str(seed)]
            if adaptive:command+=['--adaptive-wrist']
            with (WORK/(name+'.log')).open('w') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            print('COLLECTION_EXIT',name,result.returncode,flush=True)
        try:
            dataset(run,'applied_target');accepted.append(run)
            attempts.append({'run':name,'accepted':True,'adaptive':adaptive});break
        except Exception as e:attempts.append({'run':name,'accepted':False,'reason':str(e),'adaptive':adaptive})
        finally:
            (WORK/'coverage_selection_v2.json').write_text(json.dumps({'attempts':attempts,'accepted':[str(p) for p in accepted]},indent=2))
        print('REJECTED_SOURCE',name,flush=True)
if not accepted:
    print('NO_QUALIFIED_NEW_DEMONSTRATION; training not started',flush=True)
    sys.exit(0)
base=ROOT/'runs/center_j56_dev_v3_20261103';output=ROOT/'runs/teacher_left_bc_multiseed_v6_20261103'
args=[sys.executable,'-u',str(ROOT/'teacher_bootstrap.py'),'--source',str(base),'--output',str(output),'--updates','100000','--state-noise','.005','--box-output','--applied-targets']
for source in accepted:args+=['--extra-source',str(source)]
subprocess.run(args,cwd=ROOT,check=True)
subprocess.run([sys.executable,'-u',str(ROOT/'evaluate_teacher_bootstrap.py'),'--checkpoint',str(output/'best.pt'),'--source',str(base),'--output',str(ROOT/'runs/teacher_left_bc_multiseed_v6_eval_9901'),'--seed','9901'],cwd=ROOT,check=True)
