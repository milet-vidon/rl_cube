set -e
source /home/brainco/rl_cube/activate.sh
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python ../work/teacher_optimization_20261103/train_phase_residual_v5.py --run runs/left_dagger_baseline_mixture_v3_9501_20261104 --baseline runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt --output runs/teacher_state_feedback_residual_v5_20261104 --residual-bound 0.0005,0.0005,0.0005,0.0005,0.0005,0.0005,0.0005,1e-06,1e-06,1e-06,1e-06,1e-06,1e-06,1e-06,1e-06,1e-06,1e-06,0.006,0.006,0.006,1e-06,1e-06,1e-06,1e-06,1e-06,1e-06,1e-06,1e-06 --hidden-size 64 --epochs 200 --learning-rate 0.0003 --seed 20261104 --device cuda:0 --qualification ../work/teacher_optimization_20261103/v5_mixture_inspection_20261104_02/qualification.json --qualification-sha256 71f98935291201335bfe24f8a19b133ee475f7e0dc5060ce553c97f93fd617d5 --train
