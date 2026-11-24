#!/usr/bin/env bash
# Collect one development demonstration; train only after independent replay
# of the original acceptance checks succeeds. Do not overwrite historical runs.
set -euo pipefail
if (( $# > 1 )) || ! [[ ${1:-11601} =~ ^[0-9]+$ ]]; then
  echo "usage: bash run_contact_hold_cycle.sh [development-reset-seed]" >&2
  exit 2
fi
hold_seed=${1:-11601}
hold_code="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/brainco/rl_cube/activate_6_2.sh
cd "$hold_code"
export REVOTRON_USD_PATH="$hold_code/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$hold_code/../isaac_mujoco_cube"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
hold_name="contact_hold_pair15_pos64_${hold_seed}_$(date +%Y%m%d_%H%M%S_%N)"
hold_run="$hold_code/runs/$hold_name"
echo "TEACHER_RUN=$hold_run"
python -u contact_hold_experiment.py --name "$hold_name" --seed "$hold_seed" \
  --position-iterations 64 --opposing-pair-stiffness 15 2>&1 | tee "$hold_run.log"
python check_contact_hold_run.py "$hold_run" --require-qualified --output "$hold_run.audit.json"
bash run_contact_hold_training.sh "$hold_run"
