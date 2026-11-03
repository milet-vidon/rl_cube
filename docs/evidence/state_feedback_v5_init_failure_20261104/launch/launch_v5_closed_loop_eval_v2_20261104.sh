set -eu
source /home/brainco/rl_cube/activate.sh
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
from pathlib import Path
import hashlib
import os
robot=Path(os.environ['REVOTRON_USD_PATH']).resolve()
cube_root=Path(os.environ['ISAAC_MUJOCO_CUBE_ROOT']).resolve()
assert robot==Path.cwd().parent/'revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda'
assert cube_root==Path.cwd().parent/'isaac_mujoco_cube'
expected_assets={robot:'1ea674035491ae19875c61813ef8cfb1fd26c4b0b58de3ae3a2a3f04b741a143',robot.parent/'revotron_bimanual_revo3_56dof.usd':'c066a65ff5ac77a8fb0a5fb3658c71f151b64777cdc9a8f66a0739d4033170a2',cube_root/'cube_model.usda':'c41baa3cd75db5a497c5878d0e18bb6b3316124e0f1935867de8305b44e4eb11'}
for asset,digest in expected_assets.items():assert hashlib.sha256(asset.read_bytes()).hexdigest()==digest,asset
print('EXPLICIT_SOURCE_ASSET_PATHS_AND_HASHES_VERIFIED',str(robot),str(cube_root),flush=True)
expected={'state_residual_eval_runtime_v5.py':'90a1ef7d5d8cfdebd5e05df5ec598d48cea7ae7bcfaf145a357d61a5e7727d64','evaluate_state_residual_v5.py':'9bff577c826107574f6d1e435a07ba4427b7aa5b05777b315a85b4a0261ec139','test_state_residual_eval_v5.py':'08b9a31188c7816dac80636f5bce17a373b39ad77924d0dfeed84bfa9bbc7b4f'}
for name,digest in expected.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
import left_dagger_baseline_runtime_v3 as base
assert base.ROOT.resolve()==Path.cwd(),(base.ROOT,Path.cwd())
print('EXACT_EVALUATION_SOURCE_AND_BASELINE_ROOT_VERIFIED',flush=True)
PY
exec python evaluate_state_residual_v5.py --policy v5 --checkpoint runs/teacher_state_feedback_residual_v5_long_20261104/model.pt --expected-checkpoint-sha256 1c663e17f6157273ee4c80b1e6ebf1c292b6c9972278041936b56943212db98c --baseline runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt --qualification ../work/teacher_optimization_20261103/v5_mixture_inspection_20261104_02/qualification.json --qualification-sha256 71f98935291201335bfe24f8a19b133ee475f7e0dc5060ce553c97f93fd617d5 --dataset runs/left_dagger_baseline_mixture_v3_9501_20261104 --source runs/center_j56_dev_v3_20261103 --seed 9501 --max-steps 3500 --output runs/left_state_residual_v5_long_eval_9501_20261104_v2
