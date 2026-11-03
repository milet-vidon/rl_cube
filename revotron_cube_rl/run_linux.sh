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
  echo "usage: $0 {check|env-check|smoke|evaluate|train|state|side-policy|opposition-teacher|tilt-teacher|wrist-teacher|elbow-regression|regrasp-development|wrist-check|tilt-replay|elbow-check|evidence-check|dagger|sweep|face-turn|turn|script.py} [arguments...]" >&2
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
  tilt-teacher|wrist-teacher)
    run_isaac wrist_direction_experiment.py "$@"
    ;;
  elbow-regression)
    run_isaac elbow_handover_experiment.py "$@"
    ;;
  regrasp-development)
    echo "Legacy right regrasp development: this has not been adapted to the reversed wrist posture." >&2
    export REVOTRON_USD_PATH="${ROOT}/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
    run_isaac safe_handover_experiment.py --regrasp "$@"
    ;;
  wrist-check)
    exec "$PYTHON_BIN" "$ROOT/wrist_gate.py" --protocol wrist_direction_protocol_v1.json --runs runs/wrist_direction_dev_v1_20261103 runs/wrist_direction_independent_v1_20261103 runs/wrist_direction_independent_remaining_v1_20261103 "$@"
    ;;
  evidence-check)
    exec "$PYTHON_BIN" "${ROOT}/verify_portable_evidence.py" "$@"
    ;;
  elbow-check)
    exec "$PYTHON_BIN" "${ROOT}/elbow_gate.py" --protocol elbow_out_protocol_v1.json --runs runs/elbow_out_dev_v1_20261103 runs/elbow_out_independent_v1_20261103 "$@"
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
  face-turn)
    run_isaac single_face_experiment.py --push-degrees 15 --contact-style edge "$@"
    ;;
  turn)
    echo "Legacy fixture-backed turn diagnostic; use face-turn for the free-core robot contact experiment." >&2
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
