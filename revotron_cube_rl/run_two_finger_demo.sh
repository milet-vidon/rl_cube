#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
for option in "$@"; do
    case "$option" in
        --replay|--replay=*|--help|-h) exec python3 -B -u natural_face90_demo.py "$@" ;;
    esac
done
runtime=/home/nexus/rlcube_validation/pair_20261116_v1
if [[ -x "$runtime/runtime/bin/python" && -d "$runtime/kit-user" ]]; then
    export ISAACLAB_PYTHON="$runtime/runtime/bin/python" OMNI_USER_HOME="$runtime/kit-user" OV_USER_HOME="$runtime/kit-user" XDG_CACHE_HOME="$runtime/cache"
    export PATH="$runtime/runtime/bin:$PATH" LD_LIBRARY_PATH="$runtime/runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_CACHE_PATH="$runtime/cache/cuda" WARP_CACHE_PATH="$runtime/cache/warp" TORCH_HOME="$runtime/cache/torch"
else
    source /home/brainco/rl_cube/activate_6_2.sh
fi
set -u
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 PXR_WORK_THREAD_LIMIT=1
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "$ISAACLAB_PYTHON" -B -u natural_face90_demo.py "$@"
