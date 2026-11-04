"""Standalone opt-in reset/contact integration probe. Does not start a teacher.

Not executed during development. Reserve the sole Isaac slot before launching.
The default environment's source and shared asset_contract.json remain intact;
its configuration is copied byte-for-byte into this unique run's I/O directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-physics", action="store_true")
    parser.add_argument("--project", type=Path, default=Path("/home/brainco/rl_cube/github_project"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-evidence", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=9501)
    parser.add_argument("--cube-position-iterations", type=int, choices=[64], default=64)
    parser.add_argument("--cube-velocity-iterations", type=int, choices=[4], default=4)
    parser.add_argument("--right-hand-friction")
    parser.add_argument("--matched-internal-friction")
    parser.add_argument("--source-core-inertia", action="store_true")
    parser.add_argument("--source-joint-damping", action="store_true")
    args = parser.parse_args()
    if not args.execute_physics:
        parser.error("No application launched; reserve one Isaac slot and pass --execute-physics")
    if args.source_core_inertia or args.source_joint_damping:
        parser.error("This backend rejects legacy source-Core-inertia and D6/drive damping options")
    if not 1 <= args.steps <= 240:
        parser.error("Use 1..240 physics steps (at most one simulated second)")
    args.project = args.project.resolve(); args.output = args.output.resolve()
    code = args.project / "revotron_cube_rl"
    sys.path.insert(0, str(code))
    from passive_cube_articulation import load_baseline, verify_baseline_inputs
    baseline = load_baseline(args.baseline_evidence, args.baseline_sha256)
    verify_baseline_inputs(baseline)
    args.output.mkdir(parents=True, exist_ok=False)
    io_root = args.output / "environment_io"
    io_root.mkdir()
    shutil.copy2(code / "config.json", io_root / "config.json")
    robot_path = Path(os.environ.get("REVOTRON_USD_PATH", str(args.project / "revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd"))).resolve()
    cube_root = Path(os.environ.get("ISAAC_MUJOCO_CUBE_ROOT", str(args.project / "isaac_mujoco_cube"))).resolve()
    sources = [Path(__file__).resolve(), Path(__file__).with_name("passive_cube_articulation.py").resolve()]
    sources += [code / name for name in ("config.json", "environment.py", "articulated_cube_env.py", "dynamic_cube.py",
        "isaac_compat.py", "runtime_platform.py", "solver_precision.py", "joint_safety.py", "cube_core_inertia.py",
        "cube_joint_damping.py", "right_contact_environment.py", "right_contact_material.py", "matched_cube_material.py")]
    archive = args.output / "source"; archive.mkdir()
    for path in sources:
        shutil.copy2(path, archive / path.name)
    report = {"status": "starting", "scope": "isolated opt-in robot-environment reset/contact smoke",
              "teacher_started": False, "face_turn_validated": False, "checks": {},
              "source_sha256": {str(path): sha(path) for path in sources},
              "assets_sha256": {str(robot_path): sha(robot_path), str(cube_root / "cube_model.usda"): sha(cube_root / "cube_model.usda")},
              "baseline": {"path": str(args.baseline_evidence.resolve()), "sha256": args.baseline_sha256},
              "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "isolated_environment_io": str(io_root)}
    save(args.output / "result.json", report)
    app = None; env = None; contacts = []; rows = []
    try:
        from isaaclab.app import AppLauncher
        from runtime_platform import platform_kit_args
        app = AppLauncher(headless=True, enable_cameras=False,
                          kit_args=platform_kit_args("--/app/useFabricSceneDelegate=false")).app
        import numpy as np
        import omni.physx
        from pxr import PhysicsSchemaTools
        from unittest.mock import patch
        from isaac_compat import tensor
        import environment
        from articulated_cube_env import DynamicPickupEnv
        from passive_cube_articulation import passive_cube_articulation
        from solver_precision import position_iterations, verify_solver
        from matched_cube_material import matched_cube_material

        def check(name, passed):
            report["checks"][name] = bool(passed)
            if not passed:
                raise RuntimeError("Integration smoke check failed: " + name)

        def coefficients(value):
            return tuple(map(float, value.split(","))) if value else None

        constructor = DynamicPickupEnv
        options = {}
        if args.right_hand_friction:
            from right_contact_environment import RightContactEnv
            constructor = RightContactEnv
            options["right_hand_friction"] = coefficients(args.right_hand_friction)
        internal = coefficients(args.matched_internal_friction)
        with patch.object(environment, "ROOT", io_root), patch.dict(os.environ, {
                "REVOTRON_USD_PATH": str(robot_path), "ISAAC_MUJOCO_CUBE_ROOT": str(cube_root)}):
            with passive_cube_articulation(args.baseline_evidence, args.baseline_sha256,
                    solver_position_iterations=args.cube_position_iterations,
                    solver_velocity_iterations=args.cube_velocity_iterations):
                with position_iterations(64), matched_cube_material(internal):
                    env = constructor(args.output, seed=args.seed, vision=False, preview=False,
                                      auto_reset=False, arm_damping=8.0, **options)
        check("robot_remains_56_DOF", env.robot.num_joints == 56)
        env.cube.verify_initialized()
        check("native_cube_backend_verified", env.cube.model["passive_articulation"]["backend_verified"])
        report["solver"] = verify_solver(env, 64)
        cube_paths = set(env.cube.model["bodies"])
        callback_errors = []
        tick = -1

        def callback(headers, data):
            try:
                for header in headers:
                    actors = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.actor0, header.actor1)]
                    if not any(path in cube_paths for path in actors) or header.num_contact_data <= 0:
                        continue
                    impulse = sum(float(np.linalg.norm(np.array(point.impulse))) for point in
                        data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data])
                    if not np.isfinite(impulse):
                        raise ValueError("Non-finite contact impulse")
                    if impulse > 0:
                        contacts.append({"tick": tick, "actors": actors, "points": int(header.num_contact_data), "impulse_Ns": impulse})
            except Exception:
                callback_errors.append(traceback.format_exc())

        subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(callback)
        before_reset = env.cube.pose_reset_count
        env.reset()
        check("one_guarded_cube_write_per_episode_reset", env.cube.pose_reset_count == before_reset + 1)
        check("full_reset_state_readback_passed", env.cube.model["passive_articulation"]["last_reset_readback"]["passed"])
        baseline_resets = env.cube.pose_reset_count
        try:
            env.cube.write_root_state_to_sim(tensor(env.cube.data.default_root_state).clone())
        except RuntimeError as exc:
            check("control_phase_reset_rejected", "episode reset" in str(exc))
        else:
            check("control_phase_reset_rejected", False)
        for tick in range(args.steps):
            env._physics()
            root = tensor(env.cube.data.root_state_w)[0].tolist()
            bodies = env.cube.states()
            check("all_original_body_states_present", set(bodies) == {Path(path).name for path in cube_paths})
            finite = np.isfinite(root).all() and all(np.isfinite(value).all() for body in bodies.values() for value in body.values())
            check("finite_body_states", finite)
            check("no_control_phase_cube_pose_write", env.cube.pose_reset_count == baseline_resets)
            active = env.cube._array(env.cube._art.root_view.get_dof_actuation_forces())
            check("zero_cube_actuation_during_control", np.isfinite(active).all() and np.allclose(active, 0, atol=1e-10))
            rows.append({"tick": tick, "root_native_state": root, "cubelets": bodies,
                         "cube_pose_reset_count": env.cube.pose_reset_count,
                         "robot_table_impulse_Ns": float(env.robot_table_impulse)})
        check("callback_has_no_errors", not callback_errors)
        table = [row for row in contacts if any(path == "/World/Table" or path.startswith("/World/Table/") for path in row["actors"])]
        post_reset_table = [row for row in table if row["tick"] >= 0]
        check("positive_cube_table_contacts_after_reset_observed", bool(post_reset_table))
        report["contacts"] = {"positive_events": len(contacts), "table_positive_events": len(table),
                              "post_reset_table_positive_events": len(post_reset_table),
                              "callback_errors": callback_errors, "subscribed": subscription is not None}
        report["cube_model"] = env.cube.model
        report["asset_contract"] = env.contract
        report["source_unchanged"] = all(sha(path) == value for path, value in report["source_sha256"].items())
        check("source_and_config_unchanged", report["source_unchanged"])
        report["status"] = "integration_smoke_passed_pending_teacher_validation"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save(args.output / "trajectory.json", rows)
        save(args.output / "contacts.json", contacts)
        if env is not None:
            report["cube_model"] = env.cube.model
        save(args.output / "result.json", report)
        if env is not None:
            env.close()
        if app is not None:
            from isaac_compat import close_application
            close_application(app)


if __name__ == "__main__":
    main()
