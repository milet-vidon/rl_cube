# Linux migration audit (2026-11-02)

This audit records what is portable from the Windows Isaac project and what still needs a Linux runtime check.

## Verified without Isaac runtime

From `outputs/revotron_cube_rl`:

- `python linux_migration_check.py` -> `LINUX_MIGRATION_CHECK PASSED`
- `python test_reward_gates.py` -> 8 tests passed
- `python test_arm_safety.py` -> `ARM_SAFETY_TESTS_PASSED`
- `python -m compileall -q .` -> passed
- `active_training_process.json` -> `status: idle`

The static checker resolves assets through `REVOTRON_USD_PATH` and `ISAAC_MUJOCO_CUBE_ROOT`; it does not start Isaac Sim.

## Known validated artifacts

- `runs/traditional_visual_verified` and `runs/traditional_long_hold`: traditional controller pickup gates passed with zero robot-table impulse on the recorded runs.
- `runs/state_phase_knn_user_softlimit_seed1945`: real chest RGB reduced to location/scale features plus phase/state-aware kNN teacher; 1/1 pickup, approximately 11.28 cm lift, zero robot-table impulse. This is a retrieval teacher, not a learned PPO policy.
- `runs/turning_stage_cpu`: free-core cube with external fixtures passed +Z and +X 90-degree layer checks. Revo3/Tianji bilateral contact and RL turning remain unvalidated.

## Linux prerequisites

1. Ubuntu 22.04/24.04 x86_64 with a supported NVIDIA driver and GPU.
2. A matching Isaac Sim/Isaac Lab installation and its Python interpreter; set `ISAACLAB_PYTHON` to that interpreter.
3. Preserve the project layout or set:
   - `REVOTRON_USD_PATH` to the bimanual Revo3 USD.
   - `ISAAC_MUJOCO_CUBE_ROOT` to the `isaac_mujoco_cube` checkout containing `upstream/mujoco_cube-main/cube_3x3x3.xml` and `assets/sticker.png`.
4. Accept the Isaac EULA and set writable `OMNI_USER_HOME`/`OV_USER_HOME` paths.

## Reproduction order on Linux

```bash
cd outputs/revotron_cube_rl
chmod +x run_linux.sh
python3 linux_migration_check.py
python3 test_reward_gates.py
python3 test_arm_safety.py
ISAACLAB_PYTHON=/opt/isaaclab/isaaclab.sh ./run_linux.sh sweep_controller.py --grasp downward --widths 0.096 --seeds 17 --visual --name linux_traditional_smoke
```

Replace `/opt/isaaclab/isaaclab.sh` with the executable Python entry point supplied by the local Isaac Lab installation. Use a unique run name for every trial. Do not start a second Isaac process while one is active.

For a visual teacher evaluation after the traditional smoke succeeds, use the matching project checkpoint/config and run the existing evaluation script from the Isaac Python environment. Keep the `legacy_downward` DAgger teacher as the default until staged grasp passes the robot-table impulse gate.

## Not yet portable claims

A Linux machine has not yet run Isaac Sim for this project. Do not claim Linux runtime success from the static checker. Existing neural BC/PPO candidates have not passed independent pickup success; do not use them as the migration baseline. The staged parallel controller can lift in one audit but has nonzero robot-table impulse, and the tested 12-degree oblique target is IK-infeasible under the current soft limits.
