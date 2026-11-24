#!/usr/bin/env bash
set -eu
checkpoint="${1:?Explicit completed V7 checkpoint path required}"
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
expected={
 'state_zero_gated_eval_runtime_v7.py':'caadc319e816bdb3c00b5a744b71fbaf7c5f60ffd2d8f1893593fe441767b66b',
 'evaluate_state_zero_gated_v7.py':'6cda789c883a171d565b3a47d5ca5901ad9b4945221b281024eae7f66405b7f0',
 'zero_gated_residual_policy_v7.py':'714c7d6b6ca151afa052ef426cc27360b61ca62c1914cd7b7bfe84a63233256e',
 'eval_prefix_dataset_v7.py':'61c6b43cf0b2af18144723f6f3b2587dea04299feb75611e1a09c7c073b67883',
 'inspect_residual_dataset_v7.py':'ad88eb44646887898082e3b500db8e9d43f0ab1bc3eb8a71ab53f1f9f6f0c511'}
for name,digest in expected.items():assert hashlib.sha256((work/name).read_bytes()).hexdigest()==digest,name
import left_dagger_baseline_runtime_v3 as baseline
assert baseline.ROOT.resolve()==Path.cwd(),(baseline.ROOT,Path.cwd())
print('V7_EVALUATION_SOURCES_BASELINE_ROOT_AND_ASSETS_VERIFIED',flush=True)
PY
exec python -c "import sys,runpy;sys.path.append('../work/teacher_optimization_20261103');runpy.run_path('../work/teacher_optimization_20261103/evaluate_state_zero_gated_v7.py',run_name='__main__')" \
  --policy v7 --checkpoint "$checkpoint" --expected-checkpoint-sha256 "$checkpoint_sha" \
  --authorization "$authorization" --authorization-sha256 "$authorization_sha" \
  --baseline runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt \
  --qualification ../work/teacher_optimization_20261103/v7_five_history_inspection_20261104_03/qualification_candidate.json \
  --qualification-sha256 9dc17535d63a0a7667b1b3aca1853421e6d0a5b8dbf44560462611130582474f \
  --dataset runs/left_dagger_baseline_mixture_v3_9501_20261104 \
  --dataset runs/left_state_residual_v5_long_eval_9501_20261104_v2 \
  --dataset runs/left_phase_gated_residual_v6_eval_9501_20261104 \
  --dataset runs/left_phase_gated_residual_v6_long_eval_9501_20261104 \
  --dataset runs/left_phase_v4_matched_eval_9501_20261104_v2 \
  --source runs/center_j56_dev_v3_20261103 --seed 9501 --max-steps 3500 \
  --output "$output" "${extra[@]}"
