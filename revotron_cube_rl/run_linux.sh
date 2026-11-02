#!/usr/bin/env bash
set -euo pipefail

# Linux entry point for this project. The selected Python must be the Isaac Sim /
# Isaac Lab environment that can import isaaclab and isaacsim. Set ISAACLAB_PYTHON
# when it is not the current `python3`. This wrapper deliberately does not install
# Isaac Sim: use the official Isaac Lab installation for the pinned release first.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ISAACLAB_PYTHON:-python3}"
ISAAC_LAUNCHER="${ISAACLAB_LAUNCHER:-}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export OMNI_USER_HOME="${OMNI_USER_HOME:-${HOME}/.cache/isaaclab/userdata}"
export OV_USER_HOME="${OV_USER_HOME:-${OMNI_USER_HOME}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export REVOTRON_USD_PATH="${REVOTRON_USD_PATH:-${ROOT}/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd}"
export ISAAC_MUJOCO_CUBE_ROOT="${ISAAC_MUJOCO_CUBE_ROOT:-${ROOT}/../isaac_mujoco_cube}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_isaac() {
  local script="$1"
  shift
  # Official Isaac Lab checkouts commonly expose `isaaclab.sh -p script.py`.
  # A direct Isaac Python executable remains supported for container/conda setups.
  if [[ -n "${ISAAC_LAUNCHER}" ]]; then
    exec "${ISAAC_LAUNCHER}" -p "${ROOT}/${script}" "$@"
  fi
  exec "${PYTHON_BIN}" -u "${ROOT}/${script}" "$@"
}

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {check|env-check|smoke|evaluate|train|state|side-policy|opposition-teacher|tilt-teacher|tilt-replay|dagger|sweep|turn|script.py} [arguments...]" >&2
  exit 2
fi
ENTRY="$1"
shift
case "$ENTRY" in
  check)
    exec "$PYTHON_BIN" -u "${ROOT}/linux_migration_check.py" "$@"
    ;;
  env-check)
    exec "$PYTHON_BIN" -u "${ROOT}/linux_env_check.py" "$@"
    ;;
  smoke)
    run_isaac train.py --mode smoke "$@"
    ;;
  evaluate)
    run_isaac train.py --mode evaluate "$@"
    ;;
  train)
    run_isaac train.py --mode train "$@"
    ;;
  state)
    run_isaac state_learning.py "$@"
    ;;
  side-policy)
    run_isaac side_policy_evaluate.py "$@"
    ;;
  opposition-teacher)
    run_isaac opposition_experiment.py --grasp-config "${ROOT}/opposition_grasp_v20.json" "$@"
    ;;
  tilt-teacher)
    run_isaac palm_handover_experiment.py "$@"
    ;;
  tilt-replay)
    run_isaac palm_handover_replay.py "$@"
    ;;
  dagger)
    run_isaac collect_dagger.py "$@"
    ;;
  sweep)
    run_isaac sweep_controller.py "$@"
    ;;
  turn)
    run_isaac robot_hand_turn_stage.py "$@"
    ;;
  *.py)
    case "$ENTRY" in
      tilt_experiment.py|tilt_replay.py|tilt_gate.py|opposition_experiment.py|opposition_gate.py|side_experiment.py|side_waypoints.py|side_policy_evaluate.py|side_gate.py|state_learning.py|audit_dynamic_cube.py|audit_teacher.py|position_train.py|train.py|collect_dagger.py|sweep_controller.py|iterative_training.py|queue_continuation.py|robot_hand_turn_stage.py|linux_migration_check.py|linux_env_check.py)
        if [[ "$ENTRY" == "linux_migration_check.py" || "$ENTRY" == "linux_env_check.py" ]]; then
          exec "$PYTHON_BIN" -u "${ROOT}/${ENTRY}" "$@"
        fi
        run_isaac "$ENTRY" "$@"
        ;;
      *)
        echo "unsupported project entry point: ${ENTRY}" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "unsupported project entry point: ${ENTRY}" >&2
    exit 2
    ;;
esac
