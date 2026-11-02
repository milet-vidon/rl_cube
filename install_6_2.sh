#!/usr/bin/env bash
# Reproduce the isolated, officially paired Isaac 6.2 / Lab 3.0 installation.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${PROJECT_ROOT}/.conda/rl_cube_6_2"
OVERRIDES="${PROJECT_ROOT}/vendor/IsaacLab-3.0/tools/wheel_builder/uv-overrides.txt"
if [[ ! -f "${ENV_PREFIX}/conda-meta/history" ]]; then
  "${CONDA_EXE:-conda}" create --prefix "${ENV_PREFIX}" \
    --override-channels -c conda-forge --repodata-fn current_repodata.json \
    --solver classic python=3.12 pip 'libstdcxx-ng>=13' -y
fi
if [[ ! -f "${OVERRIDES}" ]]; then
  git clone --branch v3.0.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git "${PROJECT_ROOT}/vendor/IsaacLab-3.0"
fi
"${ENV_PREFIX}/bin/python" -m pip install 'uv==0.13.6'
UV_CACHE_DIR="${PROJECT_ROOT}/work/upgrade_6_2/uv-cache" "${ENV_PREFIX}/bin/uv" pip install \
  --python "${ENV_PREFIX}/bin/python" 'isaaclab[isaacsim]==3.0.0' \
  --overrides "${OVERRIDES}" --extra-index-url https://pypi.nvidia.com \
  --torch-backend cu130 --index-strategy unsafe-best-match
# The broad upstream overrides may otherwise select a NumPy newer than Numba supports.
UV_CACHE_DIR="${PROJECT_ROOT}/work/upgrade_6_2/uv-cache" "${ENV_PREFIX}/bin/uv" pip install \
  --python "${ENV_PREFIX}/bin/python" --constraint "${PROJECT_ROOT}/constraints_6_2.txt" \
  numpy packaging typeguard
