#!/usr/bin/env bash
# This launcher accepts only a separately reviewed, hash-pinned authorization.
set -eu
authorization="${1:?Reviewed V9 training authorization path required}"
authorization_sha="${2:?Reviewed authorization SHA256 required}"
qualification="${3:?Reviewed V9 qualification path required}"
qualification_sha="${4:?Reviewed qualification SHA256 required}"
coverage="${5:?Reviewed nine-history coverage path required}"
coverage_sha="${6:?Reviewed coverage SHA256 required}"
output="${7:?Unique output run required}"
training_seed="${8:?Explicit training initialization seed required}"
hidden_size="${9:-64}"
epochs="${10:-5000}"
source /home/brainco/rl_cube/activate.sh
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD"
python - "$authorization" "$authorization_sha" "$qualification" "$qualification_sha" "$coverage" "$coverage_sha" "$output" <<'PY'
from pathlib import Path
import hashlib,json,sys
import left_dagger_baseline_runtime_v3 as baseline_runtime
assert baseline_runtime.ROOT.resolve()==Path.cwd(), 'Wrong frozen runtime source root'
for path,digest in zip(sys.argv[1:7:2],sys.argv[2:7:2]):
    assert len(digest)==64 and hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, path
auth=json.loads(Path(sys.argv[1]).read_text())
assert auth['schema']=='root_reviewed_v9_nine_history_development_bc_authorization_v1'
assert auth['approved_for_development_bc_training'] is True
sys.path.append(str(Path('../work/teacher_optimization_20261103').resolve()))
import train_arm_hold_increment_v9 as trainer
assert trainer.source_hashes()==auth['source_sha256'], 'Frozen V9 training sources changed'
assert not Path(sys.argv[7]).exists(), 'Use a fresh run directory'
print('V9_REVIEWED_AUTHORIZATION_SOURCES_AND_RUNTIME_ROOT_VERIFIED',flush=True)
PY
exec python -u -c "import sys,runpy;sys.path.append('../work/teacher_optimization_20261103');runpy.run_path('../work/teacher_optimization_20261103/train_arm_hold_increment_v9.py',run_name='__main__')" \
  --run runs/left_dagger_baseline_mixture_v3_9501_20261104 \
  --run runs/left_state_residual_v5_long_eval_9501_20261104_v2 \
  --run runs/left_phase_gated_residual_v6_eval_9501_20261104 \
  --run runs/left_phase_gated_residual_v6_long_eval_9501_20261104 \
  --run runs/left_phase_v4_matched_eval_9501_20261104_v2 \
  --run runs/left_zero_gated_residual_v7_eval_9501_20261104 \
  --run runs/left_zero_gated_residual_v7_seed20261105_eval_9501_20261104_v2 \
  --run runs/left_zero_gated_residual_v7_long_eval_9501_20261104 \
  --run runs/left_hold_increment_v8_seed20261104_eval_9501_20261104 \
  --baseline runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt \
  --coverage "$coverage" --coverage-sha256 "$coverage_sha" \
  --qualification "$qualification" --qualification-sha256 "$qualification_sha" \
  --authorization "$authorization" --authorization-sha256 "$authorization_sha" \
  --output "$output" --seed "$training_seed" --epochs "$epochs" \
  --learning-rate .0003 --hidden-size "$hidden_size" --regression-weight 1000000 --ce-weight 1 \
  --candidate-interval 100 --device cuda:0 --train
