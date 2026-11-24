set -euo pipefail
AUTHORIZATION_PATH=${1:?Supply the independently reviewed root authorization JSON}
AUTHORIZATION_SHA256=${2:?Supply its exact SHA256}
source /home/brainco/rl_cube/activate.sh
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD:../work/teacher_optimization_20261103"
WORK=/home/brainco/rl_cube/github_project/work/teacher_optimization_20261103
python - <<'PY'
from pathlib import Path
import hashlib
root=Path('../work/teacher_optimization_20261103')
expected={'train_zero_gated_residual_v7.py':'9878b9d2c6f769624ba3889ea43797eacd02b203f65d8dcb3d765d90b1f9636e',
          'zero_gated_residual_policy_v7.py':'714c7d6b6ca151afa052ef426cc27360b61ca62c1914cd7b7bfe84a63233256e',
          'eval_prefix_dataset_v7.py':'61c6b43cf0b2af18144723f6f3b2587dea04299feb75611e1a09c7c073b67883',
          'inspect_residual_dataset_v7.py':'ad88eb44646887898082e3b500db8e9d43f0ab1bc3eb8a71ab53f1f9f6f0c511'}
for name,digest in expected.items():
 actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
 if actual!=digest:raise RuntimeError('Frozen v7 source differs: '+name)
PY
exec python -u "$WORK/train_zero_gated_residual_v7.py" \
  --run runs/left_dagger_baseline_mixture_v3_9501_20261104 \
  --run runs/left_state_residual_v5_long_eval_9501_20261104_v2 \
  --run runs/left_phase_gated_residual_v6_eval_9501_20261104 \
  --run runs/left_phase_gated_residual_v6_long_eval_9501_20261104 \
  --run runs/left_phase_v4_matched_eval_9501_20261104_v2 \
  --baseline runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt \
  --qualification "$WORK/v7_five_history_inspection_20261104_03/qualification_candidate.json" \
  --qualification-sha256 9dc17535d63a0a7667b1b3aca1853421e6d0a5b8dbf44560462611130582474f \
  --authorization "$AUTHORIZATION_PATH" --authorization-sha256 "$AUTHORIZATION_SHA256" \
  --output runs/teacher_zero_gated_residual_v7_long_20261104 \
  --epochs 10000 --learning-rate 0.0003 --hidden-size 64 --seed 20261104 \
  --regression-weight 1000000 --bce-weight 1 --candidate-interval 100 --device cuda:0 --train
