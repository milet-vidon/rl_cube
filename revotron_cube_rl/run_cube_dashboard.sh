#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -x /home/brainco/anaconda3/envs/rl_cube/bin/python ]]; then
  cube_python=/home/brainco/anaconda3/envs/rl_cube/bin/python
else
  cube_python="${CONDA_PREFIX:?请先 conda activate rl_cube}/bin/python"
fi
# Dashboard model inference is CPU-only; hardware telemetry reads NVML via nvidia-smi.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$cube_python" -B -m cube_dashboard.server "$@"
