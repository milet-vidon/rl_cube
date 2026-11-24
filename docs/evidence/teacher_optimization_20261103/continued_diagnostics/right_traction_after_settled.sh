#!/usr/bin/env bash
set -euo pipefail
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
source /home/brainco/rl_cube/activate.sh
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
while kill -0 328185 2>/dev/null; do sleep 5; done
exec ./run_linux.sh face-turn --opposition-planner joint --ring-inset .001 --center-style j56 --right-hand-friction 1.2,1.0 --name teacher_joint_right_traction_v4_20261103 --seed 9501
