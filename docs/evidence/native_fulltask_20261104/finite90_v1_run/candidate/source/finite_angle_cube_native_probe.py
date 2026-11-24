"""Draft isolated native reset probe; NOT a robot face-turn experiment.

Fresh ordinary-body baseline -> process exit -> passive articulation. Calibrate
all 66 scalar-coordinate axes with reset-only epsilon FK responses, then create
one coordinated 90/180 degree initial face pose. All state writes stop before
physics stepping. Requires a separately reserved Isaac slot; not yet executed.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_candidate(args, report, facade, math_helpers):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from pxr import Usd, UsdPhysics, PhysicsSchemaTools
    import isaaclab.sim as sim_utils
    import omni.physx
    from dynamic_cube import spawn_dynamic_cube
    from isaac_compat import simulation_config, physx_config, prepare_single_world, normalize_windows_asset_paths

    def check(name, condition, detail=None):
        report["checks"][name] = {"passed": bool(condition), "detail": detail}
        if not condition:
            raise RuntimeError("Finite initial-state probe check failed: " + name)

    baseline = facade.load_baseline(args.baseline_evidence, args.baseline_sha256)
    facade.verify_baseline_inputs(baseline)
    sim = sim_utils.SimulationContext(simulation_config(dt=1 / 240, device="cpu", use_fabric=False,
        render_interval=1, gravity=(0., 0., -9.81),
        physx=physx_config(solver_type=1, min_position_iteration_count=64,
                          max_position_iteration_count=64, min_velocity_iteration_count=4)))
    stage = sim.stage
    sim.set_setting("/physics/disableContactProcessing", False)
    sim.set_setting("/physics/updateToUsd", True)
    sim.set_setting("/physics/updateVelocitiesToUsd", True)
    asset = args.project / "isaac_mujoco_cube/cube_model.usda"
    source_stage = Usd.Stage.Open(str(asset))
    material = UsdPhysics.MaterialAPI(source_stage.GetPrimAtPath("/World/Material"))
    original_friction = (material.GetStaticFrictionAttr().Get(), material.GetDynamicFrictionAttr().Get())
    cube = facade.PassiveCubeHandle(stage, asset, (0., 0., 1.), original_friction,
        baseline=baseline, spawn=spawn_dynamic_cube, solver_position_iterations=64, solver_velocity_iterations=4)
    report["source_friction_preserved"] = list(original_friction)
    report["texture_overrides"] = normalize_windows_asset_paths(stage)
    prepare_single_world(sim)
    stage.GetRootLayer().Export(str(args.output / "runtime_overrides.usda"))
    sim.reset()
    cube.verify_initialized()
    report["cube_model"] = cube.model
    report["baseline_evidence"] = {"path": str(args.baseline_evidence.resolve()), "sha256": args.baseline_sha256}
    save(args.output / "result.json", report)
    cube.allow_pose_reset = True
    cube.reset()
    cube.write_root_state_to_sim(torch.tensor([[0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]]))
    cube.allow_pose_reset = False
    art = cube._art
    view = art.root_view
    paths = list(view.link_paths[0]); indices = {path: index for index, path in enumerate(paths)}
    masses = np.array([baseline["rigid_baseline"][path]["mass"] for path in paths])
    zero = torch.zeros((1, 66), dtype=torch.float32)
    initial_writes_open = True
    writes = []

    def set_initial_q(coordinates, reason):
        if not initial_writes_open:
            raise RuntimeError("All cube state writes are closed during physics stepping")
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if coordinates.shape != (66,) or not np.isfinite(coordinates).all():
            raise ValueError("Invalid initial generalized coordinates")
        art.write_joint_state_to_sim_index(position=torch.from_numpy(coordinates[None, :].copy()), velocity=zero)
        art.update(0.)
        writes.append({"reason": reason, "nonzero_dof_indices": np.flatnonzero(coordinates).tolist(),
                       "nonzero_q_rad": coordinates[np.flatnonzero(coordinates)].tolist()})

    def snapshot():
        # Reading body data after writes must perform FK; stale FK is a hard failure
        # in the epsilon calibration, never repaired by an unrecorded physics step.
        poses = cube._array(art.data.body_link_pose_w)[0]
        com_world = cube._array(art.data.body_com_pos_w)[0]
        velocities = cube._array(art.data.body_com_vel_w)[0]
        core_rotation = Rotation.from_quat(math_helpers.checked_quaternion(poses[0, 3:]))
        relative = {path: (core_rotation.inv() * Rotation.from_quat(math_helpers.checked_quaternion(poses[index, 3:]))).as_quat()
                    for index, path in enumerate(paths)}
        centers = {path: core_rotation.inv().apply(com_world[index] - poses[0, :3])
                   for index, path in enumerate(paths) if path != "/World/Core"}
        marker_error = 0.
        for index, path in enumerate(paths):
            marker = math_helpers.transform_marker(poses[index], baseline["rigid_baseline"][path]["local_com"][:3])
            marker_error = max(marker_error, float(np.linalg.norm(marker - com_world[index])))
        effort = cube._array(view.get_dof_actuation_forces())
        q = cube._array(view.get_dof_positions())[0]
        qd = cube._array(view.get_dof_velocities())[0]
        return {"poses": poses, "relative": relative, "centers": centers, "com_world": com_world,
                "velocities": velocities, "q": q, "qd": qd, "effort": effort,
                "marker_vs_native_COM_error_m": marker_error,
                "finite": all(np.isfinite(value).all() for value in (poses, com_world, velocities, q, qd, effort))}

    set_initial_q(np.zeros(66), "zero_reference")
    reference = snapshot()
    check("zero_reference_finite", reference["finite"])
    calibration = []
    for dof in range(66):
        joint_path = view.dof_paths[0][dof]
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(joint_path))
        children = joint.GetBody1Rel().GetTargets()
        check("dof_has_single_original_child_" + str(dof), len(children) == 1 and str(children[0]) in indices)
        body = str(children[0])
        perturbation = np.zeros(66); perturbation[dof] = args.epsilon
        set_initial_q(perturbation, "axis_calibration_" + str(dof))
        observed = snapshot()
        report["calibration_progress"] = {"completed_dofs": len(calibration), "current_dof": dof,
            "joint_path": joint_path, "body_path": body, "native_q_readback": observed["q"].tolist(),
            "reference_relative_quaternion_xyzw": reference["relative"][body].tolist(),
            "observed_relative_quaternion_xyzw": observed["relative"][body].tolist()}
        save(args.output / "result.json", report)
        record = math_helpers.calibrate_native_axis(dof, joint_path, body,
            reference["relative"][body], observed["relative"][body], args.epsilon)
        record["native_dof_name"] = view.shared_metatype.dof_names[dof]
        record["reference_relative_quaternion_xyzw"] = reference["relative"][body].tolist()
        record["perturbed_relative_quaternion_xyzw"] = observed["relative"][body].tolist()
        calibration.append(record)
        unaffected = max(math_helpers.orientation_error_degrees(observed["relative"][path], reference["relative"][path])
                         for path in paths if path != body)
        check("only_expected_child_rotates_" + str(dof), observed["finite"] and unaffected < .001)
        check("root_stays_fixed_during_initial_FK_" + str(dof), np.allclose(observed["poses"][0], reference["poses"][0], atol=1e-7))
    save(args.output / "native_axis_calibration.json", calibration)
    target_q, assignment = math_helpers.face_initial_coordinates(calibration, args.face, args.angle_degrees)
    set_initial_q(target_q, "coordinated_finite_face_initial_condition")
    observed = snapshot()
    selected = {record["body_path"] for record in assignment}
    expected_orientations = {path: math_helpers.expected_body_orientation(reference["relative"][path],
        args.face, args.angle_degrees, path in selected) for path in paths}
    expected_centers = math_helpers.face_marker_expectations(reference["centers"], args.face, args.angle_degrees)

    def compare(state):
        orientation_error = max(math_helpers.orientation_error_degrees(state["relative"][path], expected_orientations[path]) for path in paths)
        center_error = max(float(np.linalg.norm(state["centers"][path] - expected_centers[path])) for path in expected_centers)
        origin_error = float(np.linalg.norm(state["poses"][:, :3] - state["poses"][0, :3], axis=1).max())
        speed = float(np.linalg.norm(state["velocities"][:, 3:] - state["velocities"][0, 3:], axis=1).max())
        return {"max_relative_orientation_error_deg": orientation_error, "max_center_marker_error_m": center_error,
                "max_actor_origin_separation_m": origin_error, "max_relative_link_angular_speed_rad_s": speed}

    initial_error = compare(observed)
    check("finite_pose_representation_correct", observed["finite"] and initial_error["max_relative_orientation_error_deg"] < .05, initial_error)
    check("finite_face_marker_permutation_correct", initial_error["max_center_marker_error_m"] < 1e-6)
    check("marker_transform_matches_native_COM", observed["marker_vs_native_COM_error_m"] < 1e-6)
    check("initial_velocities_zero", np.allclose(observed["qd"], 0., atol=1e-7))
    save(args.output / "initial_state.json", {"body_paths": paths,
        "reference_actor_poses_xyz_xyzw": reference["poses"].tolist(),
        "actual_actor_poses_xyz_xyzw": observed["poses"].tolist(),
        "actual_COM_world": observed["com_world"].tolist(),
        "expected_relative_quaternions_xyzw": {key: value.tolist() for key, value in expected_orientations.items()},
        "expected_center_markers_core": {key: value.tolist() for key, value in expected_centers.items()}})
    initial_mass_center = np.sum(masses[:, None] * observed["com_world"], axis=0) / masses.sum()
    initial_mass_velocity = np.sum(masses[:, None] * observed["velocities"][:, :3], axis=0) / masses.sum()
    report["initial_condition"] = {"face": args.face, "angle_degrees": args.angle_degrees,
        "assignment": assignment, "requested_q": target_q.tolist(), "actual_q": observed["q"].tolist(),
        "initial_error": initial_error, "cube_initial_write_count": len(writes) + 1,
        "initial_write_log": writes, "joint_poses_not_driven_during_steps": True,
        "marker_definition": "Source-baseline local COM marker, verified on this symmetric cube's 3x3x3 grid; actor origins are never used as cubie centers"}
    initial_writes_open = False
    contacts = []; callback_errors = []; tick = 0

    def callback(headers, data):
        try:
            for h in headers:
                actors = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (h.actor0, h.actor1)]
                if not any(actor in indices for actor in actors) or h.num_contact_data <= 0:
                    continue
                impulse = sum(float(np.linalg.norm(np.array(point.impulse))) for point in
                    data[h.contact_data_offset:h.contact_data_offset + h.num_contact_data])
                if not np.isfinite(impulse):
                    raise ValueError("Non-finite contact impulse")
                if impulse > 0:
                    contacts.append({"step": tick, "actors": actors, "points": int(h.num_contact_data), "impulse_Ns": impulse})
        except Exception:
            callback_errors.append(traceback.format_exc())

    subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(callback)
    rows = []
    for tick in range(1, args.steps + 1):
        sim.step(render=False)
        cube.update(1 / 240)
        state = snapshot()
        errors = compare(state)
        rows.append({"step": tick, "time": tick / 240, **errors,
            "root_pose_xyzw": state["poses"][0].tolist(), "native_q": state["q"].tolist(),
            "native_qd": state["qd"].tolist(), "max_actuation_effort": float(np.abs(state["effort"]).max()),
            "body_link_poses_xyz_xyzw": state["poses"].tolist(), "body_COM_positions_world": state["com_world"].tolist(),
            "mass_center": (np.sum(masses[:, None] * state["com_world"], axis=0) / masses.sum()).tolist(),
            "mass_center_velocity": (np.sum(masses[:, None] * state["velocities"][:, :3], axis=0) / masses.sum()).tolist(),
            "finite": state["finite"], "marker_vs_native_COM_error_m": state["marker_vs_native_COM_error_m"]})
        if not state["finite"]:
            break
    save(args.output / "trajectory.json", rows); save(args.output / "contacts.json", contacts)
    report["cube_model"] = cube.model
    report["contacts"] = {"positive_events": len(contacts), "errors": callback_errors, "subscribed": subscription is not None}
    check("all_steps_finite", len(rows) == args.steps and all(row["finite"] for row in rows))
    check("zero_active_actuation", max(row["max_actuation_effort"] for row in rows) <= 1e-10)
    check("contact_callback_valid", not callback_errors)
    duration = rows[-1]["time"]
    delta_z = rows[-1]["mass_center"][2] - initial_mass_center[2]
    delta_vz = rows[-1]["mass_center_velocity"][2] - initial_mass_velocity[2]
    expected_z = initial_mass_velocity[2] * duration - .5 * 9.81 * duration**2
    check("finite_pose_freefall_mass_center", abs(delta_z - expected_z) <= 2 * 9.81 / 240 * duration + .001
          and abs(delta_vz + 9.81 * duration) < .03,
          {"delta_z": delta_z, "expected_z": expected_z, "delta_vz": delta_vz, "expected_delta_vz": -9.81 * duration})
    check("finite_initial_pose_stable", max(row["max_relative_orientation_error_deg"] for row in rows) <= .25
          and max(row["max_center_marker_error_m"] for row in rows) <= 1e-4
          and max(row["max_relative_link_angular_speed_rad_s"] for row in rows) <= .05)
    report["scope_limitations"] = ["This initializes an already turned pose; no continuous contact-driven trajectory was executed.",
        "No robot was spawned, no teacher gate was evaluated, no formula/vision solver was implemented.",
        "Raw spherical generalized-coordinate sums are not used as orientation measurements."]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute-physics", action="store_true")
    p.add_argument("--project", type=Path, default=Path("/home/brainco/rl_cube/github_project"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--face", choices=("pX", "nX", "pY", "nY", "pZ", "nZ"), default="nZ")
    p.add_argument("--angle-degrees", type=float, choices=(90., 180.), required=True)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--epsilon", type=float, default=.001)
    p.add_argument("--phase", choices=("paired", "baseline", "candidate"), default="paired")
    p.add_argument("--baseline-evidence", type=Path); p.add_argument("--baseline-sha256")
    args = p.parse_args()
    if not args.execute_physics: p.error("No Isaac launched; reserve a slot and pass --execute-physics")
    if not 1 <= args.steps <= 96: p.error("Use 1..96 steps")
    if args.epsilon != .001: p.error("Frozen calibration protocol uses epsilon=.001 rad")
    if args.phase == "candidate" and (not args.baseline_evidence or not args.baseline_sha256): p.error("Candidate requires fresh baseline and SHA256")
    args.project = args.project.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    directory = Path(__file__).resolve().parent
    common = [sys.executable, str(Path(__file__).resolve()), "--execute-physics", "--project", str(args.project),
              "--face", args.face, "--angle-degrees", str(args.angle_degrees), "--steps", str(args.steps)]
    if args.phase == "paired":
        workflow = {"status": "starting", "sequential_processes": True, "phases": [], "robot_turn_success_claim": False}
        try:
            for phase in ("baseline", "candidate"):
                output = args.output / phase
                command = common + ["--phase", phase, "--output", str(output)]
                if phase == "candidate":
                    baseline = args.output / "baseline/result.json"
                    command += ["--baseline-evidence", str(baseline), "--baseline-sha256", sha(baseline)]
                workflow["status"] = "running_" + phase; save(args.output / "workflow.json", workflow)
                with (args.output / (phase + ".log")).open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                workflow["phases"].append({"phase": phase, "returncode": completed.returncode})
                if completed.returncode: raise RuntimeError(phase + " process failed")
                evidence = json.loads((output / "result.json").read_text())
                expected = "rigid_baseline_measured" if phase == "baseline" else "initial_pose_representation_probe_passed_not_robot_turn"
                if evidence.get("status") != expected or evidence.get("source_assets_unchanged") is not True:
                    raise RuntimeError(phase + " result did not complete")
            workflow["status"] = "paired_initial_pose_probe_passed_not_robot_turn"
        except BaseException:
            workflow["status"] = "failed"; workflow["error"] = traceback.format_exc(); raise
        finally:
            save(args.output / "workflow.json", workflow)
        return
    sys.path.insert(0, str(args.project / "revotron_cube_rl"))
    base = module_from_file("_finite_baseline_probe", directory / "cube_articulation_probe_v2.py")
    facade = module_from_file("_finite_passive_facade", directory / "passive_cube_articulation.py")
    helpers = module_from_file("_finite_pose_math", directory / "finite_cube_pose_math.py")
    asset = args.project / "isaac_mujoco_cube/cube_model.usda"
    xml = args.project / "isaac_mujoco_cube/upstream/mujoco_cube-main/cube_3x3x3.xml"
    sources = [Path(__file__).resolve(), directory / "cube_articulation_probe_v2.py",
               directory / "passive_cube_articulation.py", directory / "finite_cube_pose_math.py", asset, xml]
    sources += [args.project / "revotron_cube_rl" / name for name in ("dynamic_cube.py", "isaac_compat.py", "runtime_platform.py")]
    archive = args.output / "source"; archive.mkdir()
    for source in sources:
        if source.suffix == ".py": shutil.copy2(source, archive / source.name)
    report = {"status": "starting", "checks": {}, "provenance_sha256": {str(path): sha(path) for path in sources},
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "robot_spawned": False, "control_phase_cube_state_writes": 0, "face_turn_success_claim": False}
    save(args.output / "result.json", report)
    if sha(asset) != base.EXPECTED_USD or sha(xml) != base.EXPECTED_XML: raise RuntimeError("Source asset changed")
    app = None
    try:
        from isaaclab.app import AppLauncher
        from runtime_platform import platform_kit_args
        app = AppLauncher(headless=True, enable_cameras=False, kit_args=platform_kit_args("--/app/useFabricSceneDelegate=false")).app
        if args.phase == "baseline":
            args.scenario = "freefall"
            base.run(args, report)
        else:
            run_candidate(args, report, facade, helpers)
            report["status"] = "initial_pose_representation_probe_passed_not_robot_turn"
    except BaseException:
        report["status"] = "failed"; report["error"] = traceback.format_exc(); raise
    finally:
        report["source_assets_unchanged"] = sha(asset) == base.EXPECTED_USD and sha(xml) == base.EXPECTED_XML
        save(args.output / "result.json", report)
        if app is not None:
            from isaac_compat import close_application
            close_application(app)


if __name__ == "__main__":
    main()
