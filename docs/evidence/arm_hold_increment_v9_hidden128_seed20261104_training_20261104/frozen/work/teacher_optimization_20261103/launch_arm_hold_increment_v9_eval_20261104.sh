#!/usr/bin/env bash
set -eu
checkpoint="${1:?Explicit completed V9 checkpoint path required}"
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
import hashlib,json,os
robot=Path(os.environ['REVOTRON_USD_PATH']).resolve()
cube_root=Path(os.environ['ISAAC_MUJOCO_CUBE_ROOT']).resolve()
assert robot==Path.cwd().parent/'revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda'
assert cube_root==Path.cwd().parent/'isaac_mujoco_cube'
assets={robot:'1ea674035491ae19875c61813ef8cfb1fd26c4b0b58de3ae3a2a3f04b741a143',robot.parent/'revotron_bimanual_revo3_56dof.usd':'c066a65ff5ac77a8fb0a5fb3658c71f151b64777cdc9a8f66a0739d4033170a2',cube_root/'cube_model.usda':'c41baa3cd75db5a497c5878d0e18bb6b3316124e0f1935867de8305b44e4eb11'}
for path,digest in assets.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,path
work=Path('../work/teacher_optimization_20261103')
expected={'state_arm_hold_increment_eval_runtime_v9.py': '27e69ce8c204811b3aa44e86fa84155b3d35fd591778a886ed3dcd015d4006b7', 'evaluate_state_arm_hold_increment_v9.py': 'a550548bc4b00c536fe8eee7b4859ee0fe5d1b4cb3cf98a261ee08cd66cc878f', 'arm_hold_increment_policy_v9.py': '27c113212f657aac6e6d3b1caadcf26619e1e8db32d12da4a00f618985948b9c', 'train_arm_hold_increment_v9.py': 'e92df59dc5e50a5f003cceea2ab9f2a9023fde3b1c7e49e41832542a924f98ab', 'trusted_eval_dataset_registry_v2.py': 'e076e877897420e651f691a10419049815d094c27b8196afeec77320d938d89e', 'inspect_arm_hold_increment_branches_v9.py': 'f0bb25423d56d0c13152522367df7fa720c6d6212ef8f1a116d12d77fcb264a5'}
for name,digest in expected.items():assert hashlib.sha256((work/name).read_bytes()).hexdigest()==digest,name
import left_dagger_baseline_runtime_v3 as baseline
assert baseline.ROOT.resolve()==Path.cwd(),(baseline.ROOT,Path.cwd())

import importlib,sys
sys.path.append(str(work.resolve()))
for name,digest in expected.items():
 module=importlib.import_module(Path(name).stem)
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==digest,(name,module.__file__)
import train_arm_hold_increment_v9 as trainer
qualification_path=work/'v9_trainer_preflight_20261104_01/qualification_candidate.json'
assert hashlib.sha256(qualification_path.read_bytes()).hexdigest()=='6283a3d6e1a1feccea809c5150eac49aaa5784299cfb48a8d8b396be07f07cf7'
qualification=json.loads(qualification_path.read_text())
assert qualification['source_sha256']==trainer.source_hashes() and len(qualification['source_sha256'])==22
assert qualification['training_authorized'] is False and qualification['eligible_for_rl_episode_training'] is False
for name,digest in qualification['source_sha256'].items():
 module=importlib.import_module(Path(name).stem)
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==digest,(name,module.__file__)
print('V9_EVALUATION_SOURCES_BASELINE_ROOT_ASSETS_AND_QUALIFICATION_VERIFIED',flush=True)
PY
exec python -c "import sys,runpy;sys.path.append('../work/teacher_optimization_20261103');runpy.run_path('../work/teacher_optimization_20261103/evaluate_state_arm_hold_increment_v9.py',run_name='__main__')" \
  --policy v9 --checkpoint "$checkpoint" --expected-checkpoint-sha256 "$checkpoint_sha" \
  --authorization "$authorization" --authorization-sha256 "$authorization_sha" \
  --baseline runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt \
  --qualification ../work/teacher_optimization_20261103/v9_trainer_preflight_20261104_01/qualification_candidate.json \
  --qualification-sha256 6283a3d6e1a1feccea809c5150eac49aaa5784299cfb48a8d8b396be07f07cf7 \
  --dataset runs/left_dagger_baseline_mixture_v3_9501_20261104 \
  --dataset runs/left_state_residual_v5_long_eval_9501_20261104_v2 \
  --dataset runs/left_phase_gated_residual_v6_eval_9501_20261104 \
  --dataset runs/left_phase_gated_residual_v6_long_eval_9501_20261104 \
  --dataset runs/left_phase_v4_matched_eval_9501_20261104_v2 \
  --dataset runs/left_zero_gated_residual_v7_eval_9501_20261104 \
  --dataset runs/left_zero_gated_residual_v7_seed20261105_eval_9501_20261104_v2 \
  --dataset runs/left_zero_gated_residual_v7_long_eval_9501_20261104 \
  --dataset runs/left_hold_increment_v8_seed20261104_eval_9501_20261104 \
  --source runs/center_j56_dev_v3_20261103 --seed 9501 --max-steps 3500 \
  --output "$output" "${extra[@]}"
