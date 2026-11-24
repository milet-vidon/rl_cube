#!/usr/bin/env bash
# Usage: bash run_contact_hold_training.sh /absolute/accepted/run [another/run ...]
# Set CONTACT_HOLD_BATCH_SIZE to override the default of 2048.
# This is a new successful-demonstration state BC fit, not frozen V11 resume.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  printf '%s\n' 'Usage: bash run_contact_hold_training.sh ACCEPTED_RUN [ACCEPTED_RUN ...]' >&2
  exit 2
fi
fit_batch_size="${CONTACT_HOLD_BATCH_SIZE:-2048}"
if ! [[ "$fit_batch_size" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' 'CONTACT_HOLD_BATCH_SIZE must be a positive integer' >&2
  exit 2
fi
fit_code="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fit_sources=()
for fit_source in "$@"; do
  fit_sources+=(--source "$(realpath "$fit_source")")
done
source /home/brainco/rl_cube/activate_6_2.sh
cd "$fit_code"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>"../work/teacher_optimization_20261103/contact_hold_gpu_training.lock"
flock -n 9 || { printf '%s\n' 'Another contact-hold training is active' >&2; exit 1; }
python - <<'PY'
import subprocess, torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; use the GPU host terminal')
jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
if jobs:
    raise SystemExit('Wait for the current CUDA workload to finish: '+jobs)
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY
fit_output="$fit_code/runs/contact_hold_bc_$(date +%Y%m%d_%H%M%S_%N)"
printf 'OUTPUT=%s\nLOG=%s.log\nBATCH_SIZE=%s\n' "$fit_output" "$fit_output" "$fit_batch_size"
python -u train_contact_hold_bc.py "${fit_sources[@]}" --output "$fit_output" \
  --device cuda:0 --updates 20000 --hidden-size 256 --batch-size "$fit_batch_size" \
  --learning-rate 0.0003 --seed 20261106 2>&1 | tee "$fit_output.log"
