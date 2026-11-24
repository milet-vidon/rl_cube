#!/usr/bin/env bash
# GPU fit for the reviewed seed 11601 correction set. Argument: fresh output directory.
set -euo pipefail
if (( $# < 1 || $# > 2 )) || [[ $# == 2 && $2 != --round2 ]]; then
  echo 'Usage: bash run_contact_hold_corrected.sh NEW_TRAINING_DIR [--round2]' >&2; exit 2
fi
fit_script=train_contact_hold_corrected.py
fit_default_updates=20000
if [[ ${2:-} == --round2 ]]; then
  fit_script=train_contact_hold_round2.py
  fit_default_updates=2000
fi
fit_output="$(realpath -m "$1")"
cd "$(dirname "${BASH_SOURCE[0]}")"
source /home/brainco/rl_cube/activate_6_2.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>../work/teacher_optimization_20261103/contact_hold_gpu_training.lock
flock -n 9 || { echo 'Another contact-hold fit is active' >&2; exit 1; }
exec 8>../work/teacher_optimization_20261103/isaac_gui_evaluation.lock
flock -n 8 || { echo 'An Isaac evaluation is active' >&2; exit 1; }
fit_jobs="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
[[ -z "$fit_jobs" ]] || { echo "A CUDA workload is active: $fit_jobs" >&2; exit 1; }
fit_batch="${CONTACT_HOLD_BATCH_SIZE:-65536}"
fit_updates="${CONTACT_HOLD_UPDATES:-$fit_default_updates}"
[[ ! -e "$fit_output" && ! -e "$fit_output.log" ]] || { echo 'Output or log exists' >&2; exit 1; }
mkdir -p "$(dirname "$fit_output")"
echo "OUTPUT=$fit_output BATCH_SIZE=$fit_batch UPDATES=$fit_updates"
python -u "$fit_script" --output "$fit_output" --device cuda:0 \
  --batch-size "$fit_batch" --updates "$fit_updates" 2>&1 | tee "$fit_output.log"
if [[ ${2:-} == --round2 ]]; then
  python - "$fit_output/result.json" <<'PY'
import json, sys
if json.load(open(sys.argv[1]))['best_update'] == 0:
    raise SystemExit('Parent retained on validation; skip repeat of the known failed rollout.')
PY
fi
