"""Opt-in isolated PhysX probe. Never imported by the task environment.

Preserves the original 27 rigid-body paths, 6 hinges, 20 spherical joints,
colliders, materials, and Core mass/inertia. Adds a floating articulation rooted
on Core and registered per-axis passive properties. This ports only the linear
MuJoCo damping coefficient: polynomial b and c are deliberately NOT migrated.

Compile this file without executing it until the coordinator reserves the sole
Isaac slot. --execute-physics is required to launch Isaac. Passing this probe is
not a robot grasp/turn gate and must not authorize task-environment integration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

PROJECT = Path("/home/brainco/rl_cube/github_project")
EXPECTED_USD = "c41baa3cd75db5a497c5878d0e18bb6b3316124e0f1935867de8305b44e4eb11"
EXPECTED_XML = "b7b32a4a8624241817aafcbdba20217b7250fb9c570440af6fbdcf5ec084c023"
ARMATURE = 0.0001
DRY_FRICTION = 0.001
LINEAR_DAMPING = 0.0005


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-physics", action="store_true")
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=("freefall", "kick", "floor-contact"), default="freefall")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--kick-dof", type=int, default=0)
    parser.add_argument("--kick-velocity", type=float, default=0.05)
    args = parser.parse_args()
    if not args.execute_physics:
        parser.error("No simulation launched: reserve one Isaac slot and pass --execute-physics.")
    if not 1 <= args.steps <= 480 or not 0 <= args.kick_dof < 66:
        parser.error("Use 1..480 steps and a DOF index in 0..65.")
    if not 0 < args.kick_velocity <= 0.1:
        parser.error("Initial kick velocity must be in (0, 0.1] rad/s.")
    if args.scenario != "floor-contact" and args.steps > 96:
        parser.error("Limit airborne probes to 96 steps (0.4 seconds).")
    return args


def run(args, report):
    # These imports require an already launched application.
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from pxr import Usd, UsdPhysics, PhysxSchema, PhysicsSchemaTools, UsdGeom
    import omni.physx
    from dynamic_cube import spawn_dynamic_cube
    from isaac_compat import (
        tensor, physx_config, simulation_config, prepare_single_world,
        normalize_windows_asset_paths, runtime_versions,
    )

    def array(value):
        value = tensor(value)
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy().copy()
        if hasattr(value, "numpy"):
            return value.numpy().copy()
        return np.asarray(value).copy()

    checks = report["checks"]

    def check(name, condition, detail=None):
        checks[name] = {"passed": bool(condition), "detail": detail}
        if not condition:
            raise RuntimeError(f"Probe check failed: {name}: {detail}")

    dt = 1 / 240
    sim = sim_utils.SimulationContext(simulation_config(
        dt=dt, device="cpu", use_fabric=False, render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        physx=physx_config(solver_type=1, min_position_iteration_count=64,
                           max_position_iteration_count=64, min_velocity_iteration_count=4),
    ))
    stage = sim.stage
    sim.set_setting("/physics/disableContactProcessing", False)
    sim.set_setting("/physics/updateToUsd", True)
    sim.set_setting("/physics/updateVelocitiesToUsd", True)
    usd_path = args.project / "isaac_mujoco_cube/cube_model.usda"
    start_z = 0.12 if args.scenario == "floor-contact" else 1.0
    model = spawn_dynamic_cube(stage, usd_path, (0.0, 0.0, start_z))
    body_paths = set(model["bodies"])
    check("source_has_27_bodies", len(body_paths) == 27)
    check("SI_units", UsdGeom.GetStageMetersPerUnit(stage) == 1.0
          and UsdPhysics.GetStageKilogramsPerUnit(stage) == 1.0)

    def immutable_contract():
        result = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path == "/World/Material" or path == "/World/Core" or path.startswith("/World/Cube/"):
                # Prim type, collision geometry, joint frames, mass, material binding.
                attrs = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                         if a.GetName().startswith(("physics:", "physxCollision:", "physxMeshCollision:", "physxRigidBody:"))
                         or a.GetName() in ("points", "faceVertexIndices", "faceVertexCounts", "extent")}
                rels = {r.GetName(): [str(p) for p in r.GetTargets()] for r in prim.GetRelationships()
                        if r.GetName().startswith(("physics:", "material:binding"))}
                result[path] = {"type": prim.GetTypeName(), "attrs": attrs, "rels": rels}
        # Joints can also live outside the /World/Cube subtree.
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                result[str(prim.GetPath())] = {
                    "type": prim.GetTypeName(),
                    "attrs": {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                              if a.GetName().startswith("physics:")},
                    "rels": {r.GetName(): [str(p) for p in r.GetTargets()]
                             for r in prim.GetRelationships() if r.GetName().startswith("physics:")},
                }
        return result

    before = immutable_contract()
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    types = Counter(p.GetTypeName() for p in joints)
    check("original_joint_types", dict(types) == {"PhysicsRevoluteJoint": 6, "PhysicsSphericalJoint": 20}, dict(types))
    check("no_existing_articulation", not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse()))
    children = []
    joint_records = []
    for prim in joints:
        joint = UsdPhysics.Joint(prim)
        parents = [str(p) for p in joint.GetBody0Rel().GetTargets()]
        child = [str(p) for p in joint.GetBody1Rel().GetTargets()]
        check("tree_" + str(prim.GetPath()), parents == ["/World/Core"] and len(child) == 1
              and child[0] in body_paths and not joint.GetExcludeFromArticulationAttr().Get())
        children.extend(child)
        check("no_drive_" + str(prim.GetPath()), not any(s.startswith("PhysicsDriveAPI:") for s in prim.GetAppliedSchemas()))
        axes = ("angular",) if prim.IsA(UsdPhysics.RevoluteJoint) else ("rotX", "rotY", "rotZ")
        values = {"armature": ARMATURE, "staticFrictionEffort": DRY_FRICTION,
                  "dynamicFrictionEffort": DRY_FRICTION,
                  "viscousFrictionCoefficient": LINEAR_DAMPING * math.pi / 180}
        for axis in axes:
            # PhysxSchemaAddition is codeless in this installed 110.4.4 package.
            # GetAttribute must resolve an actual registered schema attribute;
            # do not CreateAttribute or author an arbitrary custom property.
            allowed = prim.CanApplyAPI("PhysxJointAxisAPI", axis)
            if not allowed or not prim.ApplyAPI("PhysxJointAxisAPI", axis):
                raise RuntimeError(f"Unregistered or inapplicable PhysxJointAxisAPI: {prim.GetPath()} {axis}")
            for prop, value in values.items():
                attr = prim.GetAttribute(f"physxJointAxis:{axis}:{prop}")
                if not attr or attr.IsCustom():
                    raise RuntimeError(f"Schema did not register {axis}:{prop}")
                attr.Set(value)
        joint_records.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                              "body0": parents, "body1": child, "axes": axes,
                              "collision_enabled": joint.GetCollisionEnabledAttr().Get(),
                              "usd_values_per_axis": values})
    check("tree_children_unique", len(set(children)) == 26 and set(children) == body_paths - {"/World/Core"})
    core = stage.GetPrimAtPath("/World/Core")
    UsdPhysics.ArticulationRootAPI.Apply(core)
    props = PhysxSchema.PhysxArticulationAPI.Apply(core)
    props.CreateEnabledSelfCollisionsAttr(True)
    props.CreateSolverPositionIterationCountAttr(64)
    props.CreateSolverVelocityIterationCountAttr(4)
    check("original_joint_geometry_collision_and_inertia_preserved", before == immutable_contract())
    report["usd"] = {"model": model, "joints": joint_records, "immutable_contract": before,
                     "self_collision_authored": props.GetEnabledSelfCollisionsAttr().Get(),
                     "root_kinematic": UsdPhysics.RigidBodyAPI(core).GetKinematicEnabledAttr().Get()}

    contact_records = []
    step_number = 0
    contact_error = []

    def on_contact(headers, data):
        try:
            for h in headers:
                actors = [str(PhysicsSchemaTools.intToSdfPath(a)) for a in (h.actor0, h.actor1)]
                if not any(a in body_paths for a in actors):
                    continue
                impulse = sum(float(np.linalg.norm(np.array(p.impulse)))
                              for p in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data])
                contact_records.append({"step": step_number, "actors": actors,
                                        "contact_count": int(h.num_contact_data), "impulse": impulse})
        except Exception:
            contact_error.append(traceback.format_exc())

    for path in body_paths:
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.0)
    contact_subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    if args.scenario == "floor-contact":
        floor = sim_utils.CuboidCfg(size=(1.0, 1.0, 0.02), collision_props=sim_utils.CollisionPropertiesCfg())
        floor.func("/World/ProbeFloor", floor, translation=(0.0, 0.0, -0.01))

    # One articulation owns all links. Never instantiate any RigidObject here.
    cube = Articulation(ArticulationCfg(
        prim_path="/World/Core", spawn=None, actuators={}, joint_ordering=None, body_ordering=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, start_z),
                                                joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    ))
    report["texture_path_overrides"] = normalize_windows_asset_paths(stage)
    prepare_single_world(sim)
    stage.GetRootLayer().Export(str(args.output / "runtime_overrides.usda"))
    report["status"] = "initializing_backend"
    save(args.output / "result.json", report)
    sim.reset()
    view = cube.root_view
    meta = view.shared_metatype
    links = list(view.link_paths[0])
    report["backend"] = {
        "versions": runtime_versions(), "count": view.count, "fixed_base": bool(meta.fixed_base),
        "link_paths": view.link_paths, "dof_paths": view.dof_paths,
        "dof_names": list(meta.dof_names), "link_names": list(meta.link_names),
        "joint_names": list(meta.joint_names), "joint_dof_counts": list(meta.joint_dof_counts),
        "joint_dof_offsets": list(meta.joint_dof_offsets), "root_paths": view.prim_paths,
    }
    # Freeze the installed implementation used for this candidate, not only its version labels.
    import inspect
    import omni.physics.tensors as tensor_module
    backend_files = [Path(inspect.getfile(type(cube))), Path(inspect.getfile(type(cube.data))),
                     Path(inspect.getfile(type(view))), Path(tensor_module.__file__)]
    report["backend_implementation_sha256"] = {str(p): sha(p) for p in backend_files}
    check("single_floating_27_link_66_dof_articulation", view.count == 1 and not meta.fixed_base
          and cube.num_bodies == 27 and cube.num_joints == 66)
    check("all_original_paths_in_backend", set(links) == body_paths, links)
    check("Core_backend_root", links[0] == "/World/Core", links[0])
    backend = report["backend"]
    readers = {
        "armatures": view.get_dof_armatures, "friction_properties": view.get_dof_friction_properties,
        "types": view.get_dof_types, "stiffness": view.get_dof_stiffnesses,
        "drive_damping": view.get_dof_dampings, "position_targets": view.get_dof_position_targets,
        "velocity_targets": view.get_dof_velocity_targets, "actuation_forces": view.get_dof_actuation_forces,
        "masses": view.get_masses, "inertias": view.get_inertias, "local_coms": view.get_coms,
    }
    actual = {name: array(reader()) for name, reader in readers.items()}
    backend.update({name: value.tolist() for name, value in actual.items()})
    report["status"] = "validating_backend_readback"
    save(args.output / "result.json", report)
    check("rotational_DOFs_only", actual["types"].shape == (1, 66) and np.all(actual["types"] == 0))
    check("backend_armature", actual["armatures"].shape == (1, 66)
          and np.allclose(actual["armatures"], ARMATURE, rtol=1e-5, atol=1e-10))
    expected_friction = np.array([DRY_FRICTION, DRY_FRICTION, LINEAR_DAMPING])
    check("backend_static_dynamic_and_linear_viscous", actual["friction_properties"].shape == (1, 66, 3)
          and np.allclose(actual["friction_properties"], expected_friction, rtol=1e-5, atol=1e-10),
          {"expected_static_dynamic_viscous_SI": expected_friction.tolist()})
    for name in ("stiffness", "drive_damping", "position_targets", "velocity_targets", "actuation_forces"):
        check("zero_active_" + name, np.all(np.isfinite(actual[name])) and np.allclose(actual[name], 0, atol=1e-10))
    check("positive_link_mass_inertia", np.all(actual["masses"] > 0)
          and np.all(np.linalg.eigvalsh(actual["inertias"].reshape(27, 3, 3)) > 0))
    core_mass = UsdPhysics.MassAPI(core).GetMassAttr().Get()
    core_diagonal = np.array(UsdPhysics.MassAPI(core).GetDiagonalInertiaAttr().Get())
    check("Core_mass_preserved", np.isclose(actual["masses"][0, 0], core_mass, rtol=1e-5))
    check("Core_principal_inertia_preserved", np.allclose(
        np.sort(np.linalg.eigvalsh(actual["inertias"][0, 0].reshape(3, 3))),
        np.sort(core_diagonal), rtol=1e-5, atol=1e-12))

    # Initial-condition writes only. No cube state/target/effort writes after this.
    root_pose = torch.tensor([[0., 0., start_z, 0., 0., 0., 1.]], dtype=torch.float32)
    q = torch.zeros((1, 66), dtype=torch.float32)
    qd = torch.zeros_like(q)
    if args.scenario == "kick":
        qd[0, args.kick_dof] = args.kick_velocity
    cube.reset()
    cube.write_root_link_pose_to_sim_index(root_pose=root_pose)
    cube.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((1, 6)))
    cube.write_joint_state_to_sim_index(position=q, velocity=qd)
    cube.update(0.0)
    masses = actual["masses"][0]
    rows = []

    def observe(step):
        com_pos = array(cube.data.body_com_pos_w)[0]
        com_vel = array(cube.data.body_com_vel_w)[0]
        joint_q = array(view.get_dof_positions())[0]
        joint_v = array(view.get_dof_velocities())[0]
        center = np.sum(masses[:, None] * com_pos, axis=0) / masses.sum()
        center_v = np.sum(masses[:, None] * com_vel[:, :3], axis=0) / masses.sum()
        effort = array(view.get_dof_actuation_forces())[0]
        return {"step": step, "time": step * dt, "mass_center": center.tolist(),
                "mass_center_velocity": center_v.tolist(), "root_pose": array(view.get_root_transforms())[0].tolist(),
                "joint_position": joint_q.tolist(), "joint_velocity": joint_v.tolist(),
                "max_joint_speed": float(np.abs(joint_v).max()), "joint_speed_norm": float(np.linalg.norm(joint_v)),
                "max_link_angular_speed": float(np.linalg.norm(com_vel[:, 3:], axis=1).max()),
                "max_active_effort": float(np.abs(effort).max()),
                "finite": bool(np.all(np.isfinite(com_pos)) and np.all(np.isfinite(com_vel))
                               and np.all(np.isfinite(joint_q)) and np.all(np.isfinite(joint_v)))}

    rows.append(observe(0))
    for step_number in range(1, args.steps + 1):
        sim.step(render=False)
        cube.update(dt)
        rows.append(observe(step_number))
        if not rows[-1]["finite"]:
            break
    save(args.output / "trajectory.json", rows)
    save(args.output / "contacts.json", contact_records)
    report["contact_callback"] = {
        "subscribed": contact_subscription is not None, "errors": contact_error,
        "event_count": len(contact_records),
        "sibling_events": sum(all(p in body_paths and p != "/World/Core" for p in r["actors"]) for r in contact_records),
        "floor_events": sum("/World/ProbeFloor" in r["actors"] for r in contact_records),
        "note": "Zero events alone cannot verify functioning self-collision or contact callbacks.",
    }
    check("finite_passive_trajectory", all(r["finite"] for r in rows))
    check("no_active_effort_during_probe", max(r["max_active_effort"] for r in rows) <= 1e-10)
    check("contact_callback_no_errors", not contact_error)
    if args.scenario in ("freefall", "kick"):
        t = rows[-1]["time"]
        dz = rows[-1]["mass_center"][2] - rows[0]["mass_center"][2]
        vz = rows[-1]["mass_center_velocity"][2] - rows[0]["mass_center_velocity"][2]
        # Semi-implicit integration differs from continuous ballistic position by O(dt*t).
        expected_dz = rows[0]["mass_center_velocity"][2] * t - 0.5 * 9.81 * t * t
        check("freefall_mass_center", abs(dz - expected_dz) <= 2 * 9.81 * dt * max(t, dt) + 0.001
              and abs(vz + 9.81 * t) < 0.03,
              {"dz": dz, "expected_continuous_dz": expected_dz, "delta_vz": vz, "expected_delta_vz": -9.81 * t})
    if args.scenario == "freefall":
        check("zero_internal_velocity_stable", max(r["max_joint_speed"] for r in rows) < 0.05,
              {"peak_joint_speed": max(r["max_joint_speed"] for r in rows), "threshold_rad_s": 0.05})
    elif args.scenario == "kick":
        report["kick_response"] = {
            "dof_index": args.kick_dof, "dof_path": view.dof_paths[0][args.kick_dof],
            "initial_speed_norm": rows[0]["joint_speed_norm"], "final_speed_norm": rows[-1]["joint_speed_norm"],
            "note": "Diagnostic only: coupled DOFs need not decay monotonically; no isolated damping equivalence claim.",
        }
    else:
        check("actual_cube_floor_contact_observed", report["contact_callback"]["floor_events"] > 0)
    report["trajectory_sha256"] = sha(args.output / "trajectory.json")
    report["contacts_sha256"] = sha(args.output / "contacts.json")
    report["remaining_validation"] = [
        "Actual sibling collision response if no sibling events were observed.",
        "66 distinct DOF path/axis response and finite rotations through 90/180 degrees; spherical coordinate semantics differ from MuJoCo.",
        "A/B passive response with isolated parameters; no friction algorithm equivalence assumed.",
        "Robot integration facade, unchanged limits, contact safety, held-out grasp/turn gates.",
    ]


def main():
    args = parse_args()
    args.project = args.project.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.project / "revotron_cube_rl"))
    usd = args.project / "isaac_mujoco_cube/cube_model.usda"
    xml = args.project / "isaac_mujoco_cube/upstream/mujoco_cube-main/cube_3x3x3.xml"
    deps = [Path(__file__).resolve(), usd, xml] + [args.project / "revotron_cube_rl" / name for name in
            ("dynamic_cube.py", "isaac_compat.py", "runtime_platform.py")]
    report = {"status": "starting", "checks": {}, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "provenance_sha256": {str(p): sha(p) for p in deps}, "physics_steps_requested": args.steps,
              "direct_control_during_steps": False, "initial_state_writes_only": True,
              "damping_semantics": {"source_polynomial_abc": [0.0005] * 3,
                                    "ported_linear_a": LINEAR_DAMPING, "omitted_quadratic_b_and_cubic_c": [0.0005, 0.0005]},
              "task_success_claim": False}
    save(args.output / "result.json", report)
    if sha(usd) != EXPECTED_USD or sha(xml) != EXPECTED_XML:
        raise RuntimeError("Source asset changed: audit new source before running this frozen probe")
    app = None
    try:
        from isaaclab.app import AppLauncher
        from runtime_platform import platform_kit_args
        app = AppLauncher(headless=True, enable_cameras=False,
                          kit_args=platform_kit_args("--/app/useFabricSceneDelegate=false")).app
        run(args, report)
        report["status"] = "probe_passed_pending_integration_validation"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["source_assets_unchanged"] = sha(usd) == EXPECTED_USD and sha(xml) == EXPECTED_XML
        save(args.output / "result.json", report)
        if app is not None:
            from isaac_compat import close_application
            close_application(app)


if __name__ == "__main__":
    main()
