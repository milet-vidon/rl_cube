#!/usr/bin/env bash
# Source this file to activate the project conda environment.
_rl_cube_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_rl_conda_base="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
if [[ ! -f "${_rl_conda_base}/etc/profile.d/conda.sh" ]]; then
  echo "Set CONDA_BASE to your conda installation root." >&2
  return 1
fi
source "${_rl_conda_base}/etc/profile.d/conda.sh"
unset _rl_conda_base
conda activate "${_rl_cube_root}/.conda/rl_cube"
export ISAACLAB_PYTHON="${CONDA_PREFIX}/bin/python"
unset ISAACLAB_LAUNCHER
export REVOTRON_USD_PATH="${_rl_cube_root}/revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd"
export ISAAC_MUJOCO_CUBE_ROOT="${_rl_cube_root}/isaac_mujoco_cube"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_USER_HOME="${_rl_cube_root}/work/isaac_userdata"
export OV_USER_HOME="${OMNI_USER_HOME}"
export OMP_NUM_THREADS=4
unset _rl_cube_root
