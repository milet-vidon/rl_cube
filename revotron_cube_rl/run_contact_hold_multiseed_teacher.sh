#!/usr/bin/env bash
# Teacher coverage experiment. Only physically accepted development episodes enter BC.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /home/brainco/rl_cube/activate_6_2.sh
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
stamp=$(date +%Y%m%d_%H%M%S_%N)
bundle="runs/contact_hold_multiseed_$stamp"
mkdir "$bundle"
sources=(runs/contact_hold_pair15_pos64_11601_20261107_120008_511634810)
for seed in 11603 11604 11605; do
  name="contact_hold_teacher_${seed}_$stamp"
  python -u contact_hold_experiment.py --name "$name" --seed "$seed" --position-iterations 64 --opposing-pair-stiffness 15 2>&1 | tee "$bundle/teacher_$seed.log"
  verdict=$(python - "runs/$name" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'result.json').read_text());t=json.loads((p/'trajectory.json').read_text())
if not r['full_episode'] or r['execution_interrupted'] or not t:raise SystemExit('Incomplete teacher evaluation')
print('accepted' if r['dynamic_grasp_passed'] and r['failure'] is None else f"failed phase={t[-1]['phase']} reason={r['failure']}")
PY
)
  echo "$seed $verdict" | tee -a "$bundle/teacher_results.txt"
  if [[ "$verdict" == accepted ]]; then
    python check_contact_hold_run.py "runs/$name" --require-qualified --output "$bundle/audit_$seed.json"
    sources+=("runs/$name")
  fi
done
(( ${#sources[@]} >= 3 )) || { echo "Fewer than two new accepted demos; see $bundle/teacher_results.txt"; exit 1; }
CONTACT_HOLD_BATCH_SIZE=${CONTACT_HOLD_BATCH_SIZE:-32768} bash run_contact_hold_training.sh "${sources[@]}" | tee "$bundle/train.log"
fit=$(sed -n 's/^OUTPUT=//p' "$bundle/train.log")
[[ -f "$fit/result.json" ]]
for seed in 21611 21612; do
  CONTACT_HOLD_EVAL_ROLE=validation bash run_contact_hold_evaluation.sh "$fit" "$seed" | tee "$bundle/validation_$seed.log"
done
