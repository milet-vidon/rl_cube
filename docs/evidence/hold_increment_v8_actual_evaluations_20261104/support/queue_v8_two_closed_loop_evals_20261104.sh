#!/usr/bin/env bash
set -u
WORK=/home/brainco/rl_cube/github_project/work/teacher_optimization_20261103
STATE="$WORK/queue_v8_two_closed_loop_evals_20261104.state"
printf 'seed20261104_running\n' > "$STATE"
bash "$WORK/launch_hold_increment_v8_eval_20261104.sh" "$PWD/runs/teacher_hold_increment_v8_seed20261104_20261104/model.pt" 6280460b32b1d87a7bf2512820ad1b23a83437261bf5881a6c64b441de3d4d5b "$WORK/v8_seed20261104_root_training_authorization_20261104.json" 55b3bf394fe39e98eb7f277aa96805879186129ead473ad210d9682af9221205 runs/left_hold_increment_v8_seed20261104_eval_9501_20261104 evaluate > "$WORK/v8_seed20261104_closed_loop_eval.log" 2>&1
exit_20261104=$?
printf 'seed20261104_finished exit=%s\n' "$exit_20261104" >> "$WORK/queue_v8_two_closed_loop_evals_20261104.completed"
printf 'seed20261105_running\n' > "$STATE"
bash "$WORK/launch_hold_increment_v8_eval_20261104.sh" "$PWD/runs/teacher_hold_increment_v8_seed20261105_20261104/model.pt" 56eba2c26ceff4d85513da3970bfa11ec83d87448f47e9c7b3abafc878f74456 "$WORK/v8_seed20261105_root_training_authorization_20261104.json" d6d5961f23215140a21658c51c6105e0ad24eb7db1a2df0c65f688708335e6ef runs/left_hold_increment_v8_seed20261105_eval_9501_20261104 evaluate > "$WORK/v8_seed20261105_closed_loop_eval.log" 2>&1
exit_20261105=$?
printf 'seed20261105_finished exit=%s\n' "$exit_20261105" >> "$WORK/queue_v8_two_closed_loop_evals_20261104.completed"
printf 'finished seed04_exit=%s seed05_exit=%s\n' "$exit_20261104" "$exit_20261105" > "$STATE"
exit "$exit_20261105"
