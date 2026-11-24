#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /home/brainco/rl_cube/activate_6_2.sh
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec python -B -u contact_hold90_reset_probe.py "$@"
