#!/usr/bin/env bash
set -euo pipefail
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
source /home/brainco/rl_cube/activate.sh
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
while kill -0 316600 2>/dev/null; do sleep 10; done
for seed in 9901 9502 9503; do
 ./run_linux.sh center-teacher --center-style j56 --name "teacher_coverage_${seed}_20261103_v1" --seed "$seed" > "../work/teacher_optimization_20261103/coverage_${seed}.log" 2>&1 || true
done
python - <<'PY'
from pathlib import Path
import json,subprocess
from teacher_bootstrap import dataset
base=Path('runs/center_j56_dev_v3_20261103');accepted=[];rejected=[]
for seed in (9901,9502,9503):
 p=Path(f'runs/teacher_coverage_{seed}_20261103_v1')
 try:dataset(p,'applied_target');accepted.append(p)
 except Exception as e:rejected.append({'run':str(p),'reason':str(e)})
report={'accepted_demonstrations':[str(p) for p in accepted],'rejected_demonstrations':rejected,'new_model_started':bool(accepted)}
Path('../work/teacher_optimization_20261103/coverage_selection.json').write_text(json.dumps(report,indent=2))
if accepted:
 args=['python','-u','teacher_bootstrap.py','--source',str(base),'--output','runs/teacher_left_bc_multiseed_v6_20261103','--updates','100000','--state-noise','.005','--box-output','--applied-targets']
 for p in accepted:args+=['--extra-source',str(p)]
 subprocess.run(args,check=True)
 subprocess.run(['python','-u','evaluate_teacher_bootstrap.py','--checkpoint','runs/teacher_left_bc_multiseed_v6_20261103/best.pt','--source',str(base),'--output','runs/teacher_left_bc_multiseed_v6_eval_9901','--seed','9901'],check=True)
else:
 print('No additional qualified full demonstrations. Refusing to train on failed episodes.',flush=True)
PY
