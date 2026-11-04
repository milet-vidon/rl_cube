set -eu
source /home/brainco/rl_cube/activate.sh
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
from pathlib import Path
import hashlib
expected={'state_residual_eval_runtime_v5.py':'90a1ef7d5d8cfdebd5e05df5ec598d48cea7ae7bcfaf145a357d61a5e7727d64','evaluate_state_residual_v5.py':'9bff577c826107574f6d1e435a07ba4427b7aa5b05777b315a85b4a0261ec139','test_state_residual_eval_v5.py':'08b9a31188c7816dac80636f5bce17a373b39ad77924d0dfeed84bfa9bbc7b4f'}
for name,digest in expected.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
import left_dagger_baseline_runtime_v3 as base
assert base.ROOT.resolve()==Path.cwd(),(base.ROOT,Path.cwd())
print('EXACT_EVALUATION_SOURCE_AND_BASELINE_ROOT_VERIFIED',flush=True)
PY
exec python evaluate_state_residual_v5.py --policy v4 --checkpoint runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt --expected-checkpoint-sha256 d2493b00df27b8d84ae69da3b399b3beb926b77fdd87a9b51aa6a2b1b23a36a4 --baseline runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt --qualification ../work/teacher_optimization_20261103/v5_mixture_inspection_20261104_02/qualification.json --qualification-sha256 71f98935291201335bfe24f8a19b133ee475f7e0dc5060ce553c97f93fd617d5 --dataset runs/left_dagger_baseline_mixture_v3_9501_20261104 --source runs/center_j56_dev_v3_20261103 --seed 9501 --max-steps 3500 --output runs/left_phase_v4_matched_eval_9501_20261104
