#!/usr/bin/env bash
set -eu
checkpoint="${1:?Explicit completed V8 checkpoint path required}"
checkpoint_sha="${2:?Explicit checkpoint SHA256 required}"
authorization="${3:?Explicit reviewed training authorization path required}"
authorization_sha="${4:?Explicit authorization SHA256 required}"
output="${5:?Explicit new evaluation/preflight output path required}"
mode="${6:-preflight}"
if [[ ! "$checkpoint_sha" =~ ^[0-9a-f]{64}$ || ! "$authorization_sha" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'Invalid explicit artifact SHA256' >&2
  exit 2
fi
extra=()
case "$mode" in
  preflight) extra+=(--preflight-only) ;;
  evaluate) ;;
  *) printf '%s\n' 'Mode must be preflight or evaluate' >&2; exit 2 ;;
esac
source /home/brainco/rl_cube/activate.sh
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
from pathlib import Path
import hashlib,os
robot=Path(os.environ['REVOTRON_USD_PATH']).resolve()
cube_root=Path(os.environ['ISAAC_MUJOCO_CUBE_ROOT']).resolve()
assert robot==Path.cwd().parent/'revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda'
assert cube_root==Path.cwd().parent/'isaac_mujoco_cube'
assets={robot:'1ea674035491ae19875c61813ef8cfb1fd26c4b0b58de3ae3a2a3f04b741a143',robot.parent/'revotron_bimanual_revo3_56dof.usd':'c066a65ff5ac77a8fb0a5fb3658c71f151b64777cdc9a8f66a0739d4033170a2',cube_root/'cube_model.usda':'c41baa3cd75db5a497c5878d0e18bb6b3316124e0f1935867de8305b44e4eb11'}
for path,digest in assets.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,path
work=Path('../work/teacher_optimization_20261103')
expected={'state_hold_increment_eval_runtime_v8.py': '8c3c9459254c12b0b992889bd95aea52bf3a12d4143fd40731641a70ee46072c', 'evaluate_state_hold_increment_v8.py': '54a9f1ae958cacd55434397e72b04c10743c8a16c1be24d74f3518010ed497c3', 'hold_increment_policy_v8.py': 'b708e69d928bf073f8f7ee5f50edd9e1fe0f96213bfe3a49f51cfeff4c0561ae', 'train_hold_increment_v8.py': '055f8f0d6ed724c2932a2effa59d402c2e22472491519f9a1b09fb553e5a42cb', 'trusted_eval_dataset_registry_v1.py': 'e19a5fd879eb9d7e814d14ec1dff906f02d773868b1ff91a1bdd9ce759ff617c', 'inspect_hold_increment_branches_v8.py': 'e1b8b286b9344648f780d5c7732da71468afe1089995ecc96bc7e9040e9a62fc'}
for name,digest in expected.items():assert hashlib.sha256((work/name).read_bytes()).hexdigest()==digest,name
import left_dagger_baseline_runtime_v3 as baseline
assert baseline.ROOT.resolve()==Path.cwd(),(baseline.ROOT,Path.cwd())

import importlib,sys
sys.path.append(str(work.resolve()))
for name,digest in expected.items():
 module=importlib.import_module(Path(name).stem)
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==digest,(name,module.__file__)
print('V8_EVALUATION_SOURCES_BASELINE_ROOT_AND_ASSETS_VERIFIED',flush=True)
PY
exec python -c "import sys,runpy;sys.path.append('../work/teacher_optimization_20261103');runpy.run_path('../work/teacher_optimization_20261103/evaluate_state_hold_increment_v8.py',run_name='__main__')" \
  --policy v8 --checkpoint "$checkpoint" --expected-checkpoint-sha256 "$checkpoint_sha" \
  --authorization "$authorization" --authorization-sha256 "$authorization_sha" \
  --baseline runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt \
  --qualification ../work/teacher_optimization_20261103/v8_trainer_preflight_20261104_01/qualification_candidate.json \
  --qualification-sha256 810b793afb4b827bfcea72778c66f55883552f143b0fda4bd2245eb5aaa2d1b9 \
  --dataset runs/left_dagger_baseline_mixture_v3_9501_20261104 \
  --dataset runs/left_state_residual_v5_long_eval_9501_20261104_v2 \
  --dataset runs/left_phase_gated_residual_v6_eval_9501_20261104 \
  --dataset runs/left_phase_gated_residual_v6_long_eval_9501_20261104 \
  --dataset runs/left_phase_v4_matched_eval_9501_20261104_v2 \
  --dataset runs/left_zero_gated_residual_v7_eval_9501_20261104 \
  --dataset runs/left_zero_gated_residual_v7_seed20261105_eval_9501_20261104_v2 \
  --dataset runs/left_zero_gated_residual_v7_long_eval_9501_20261104 \
  --source runs/center_j56_dev_v3_20261103 --seed 9501 --max-steps 3500 \
  --output "$output" "${extra[@]}"
