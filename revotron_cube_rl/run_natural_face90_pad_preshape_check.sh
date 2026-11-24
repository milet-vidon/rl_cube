#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

runtime=/home/nexus/rlcube_validation/pair_20261116_v1/runtime
if [[ -x "$runtime/bin/python" ]]; then
    python_bin="$runtime/bin/python"
    export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    python_bin="${ISAACLAB_PYTHON:-/home/brainco/rl_cube/.conda/rl_cube_6_2/bin/python}"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$python_bin" -B natural_face90_pad_preshape_check.py "$@"
