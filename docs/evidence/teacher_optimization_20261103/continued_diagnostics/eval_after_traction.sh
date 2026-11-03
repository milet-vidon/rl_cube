#!/usr/bin/env bash
set -euo pipefail
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
source /home/brainco/rl_cube/activate.sh
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
while kill -0 323390 2>/dev/null; do sleep 5; done
exec python -u evaluate_teacher_bootstrap.py --checkpoint runs/teacher_left_bc_applied_v7_20261103/best.pt --source runs/center_j56_dev_v3_20261103 --output runs/teacher_left_bc_applied_v7_eval_9501 --seed 9501
