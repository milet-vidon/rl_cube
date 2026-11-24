#!/usr/bin/env bash
# Evaluate one frozen BC policy from reset; never creates a training demonstration.
set -euo pipefail
if (( $# < 1 || $# > 2 )) || ! [[ ${2:-11601} =~ ^[0-9]+$ ]]; then
  echo 'Usage: bash run_contact_hold_evaluation.sh TRAINING_RUN [RESET_SEED]' >&2; exit 2
fi
eval_training="$(realpath -e "$1")"
eval_seed=${2:-11601}
eval_code="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/brainco/rl_cube/activate_6_2.sh
cd "$eval_code"
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>../work/teacher_optimization_20261103/isaac_gui_evaluation.lock
flock -n 9 || { echo 'Another Isaac evaluation is active' >&2; exit 75; }
eval_jobs="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
[[ -z "$eval_jobs" ]] || { echo "A GPU workload is active: $eval_jobs" >&2; exit 75; }
eval_name="${CONTACT_HOLD_EVAL_NAME:-contact_hold_student_eval_${eval_seed}_$(date +%Y%m%d_%H%M%S_%N)}"
[[ "$eval_name" =~ ^contact_hold_student_eval_[0-9]+_[a-zA-Z0-9_]+$ && ! -e "runs/$eval_name" && ! -e "runs/$eval_name.log" ]] || { echo 'Invalid or reused evaluation name' >&2; exit 2; }
eval_split=development
if [[ ${CONTACT_HOLD_EVAL_ROLE:-evaluation} == test ]]; then eval_split=independent; fi
echo "OUTPUT=$PWD/runs/$eval_name"
python -u single_face_experiment.py --bc-evaluation-run "$eval_training" --seed "$eval_seed" --name "$eval_name" \
  --controller opposed --push-degrees 0 --adaptive-wrist --settled-hold --position-iterations 64 \
  --split "$eval_split" --max-steps 3500 2>&1 | tee "runs/$eval_name.log"
