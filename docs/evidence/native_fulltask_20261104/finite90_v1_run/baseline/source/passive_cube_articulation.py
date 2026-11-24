"""Draft opt-in cube facade; not installed into the robot environment.

Enable this factory context OUTSIDE existing material/precision contexts. The
original DynamicCubeHandle is never constructed. Runtime acceptance requires a
new integrated robot probe; the isolated probe is not a grasp/turn certificate.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import math

ARMATURE = 0.0001
DRY_FRICTION = 0.001
LINEAR_DAMPING = 0.0005


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_baseline(path, expected_sha256):
    """Require measured, completed ordinary-body evidence, never USD zeros."""
    path = Path(path).resolve()
    if sha(path) != expected_sha256:
        raise ValueError("Ordinary-body baseline SHA256 mismatch")
    data = json.loads(path.read_text())
    if (data.get("status") != "rigid_baseline_measured"
            or data.get("baseline_kind") != "fresh_ordinary_rigid_objects_actual_backend_mass_properties"
            or data.get("source_assets_unchanged") is not True
            or not data.get("checks") or not all(v["passed"] for v in data["checks"].values())):
        raise ValueError("Ordinary-body baseline did not complete its verified measurements")
    bodies = data.get("rigid_baseline", {})
    if len(bodies) != 27 or "/World/Core" not in bodies:
        raise ValueError("Baseline must contain Core and the 26 original cubelet paths")
    data["evidence_path"] = str(path)
    data["evidence_sha256"] = expected_sha256
    return data


def verify_baseline_inputs(baseline):
    """Include original probe/MJCF/source files as well as the cube USD."""
    fingerprint = baseline["matching_fingerprint"]
    for group in ("source_sha256", "physics_implementation_sha256"):
        if not fingerprint.get(group):
            raise ValueError("Baseline source/runtime fingerprint is incomplete")
        for name, expected in fingerprint[group].items():
            if sha(name) != expected:
                raise ValueError("Measured baseline input changed: " + name)


def validate_measured_mass_properties(baseline, paths, masses, inertias, coms):
    """Pure CPU comparison; tensors are actual backend reads, not config caches."""
    import numpy as np
    expected = baseline["rigid_baseline"]
    if len(paths) != 27 or len(set(paths)) != 27 or set(paths) != set(expected):
        raise RuntimeError("Articulation links do not match the 27 measured body paths")
    if paths[0] != "/World/Core":
        raise RuntimeError("Core must be backend root/link zero")
    masses = np.asarray(masses).reshape(27)
    inertias = np.asarray(inertias).reshape(27, 9)
    coms = np.asarray(coms).reshape(27, 7)
    if not all(np.isfinite(v).all() for v in (masses, inertias, coms)):
        raise RuntimeError("Non-finite backend mass, inertia or COM")
    if not (masses > 0).all() or not (np.linalg.eigvalsh(inertias.reshape(27, 3, 3)) > 0).all():
        raise RuntimeError("Non-positive physical mass/inertia")
    for index, path in enumerate(paths):
        prior = expected[path]
        if not np.isclose(masses[index], prior["mass"], rtol=1e-5):
            raise RuntimeError("Measured baseline mass changed: " + path)
        if not np.allclose(inertias[index], prior["inertia"], rtol=1e-5, atol=1e-12):
            raise RuntimeError("Measured baseline inertia changed: " + path)
        if not np.allclose(coms[index, :3], prior["local_com"][:3], rtol=1e-5, atol=1e-7):
            raise RuntimeError("Measured baseline COM translation changed: " + path)
        actual_q = coms[index, 3:7]
        expected_q = np.asarray(prior["local_com"][3:7])
        if (not np.isfinite(expected_q).all() or not np.isclose(np.linalg.norm(actual_q), 1., atol=1e-6)
                or not np.isclose(np.linalg.norm(expected_q), 1., atol=1e-6)):
            raise RuntimeError("COM quaternion is not finite and unit length: " + path)
        if min(np.linalg.norm(actual_q - expected_q), np.linalg.norm(actual_q + expected_q)) > 1e-5:
            raise RuntimeError("Measured baseline COM rotation changed: " + path)
    return {path: {"mass": float(masses[i]), "inertia": inertias[i].tolist(), "local_com": coms[i].tolist()}
            for i, path in enumerate(paths)}


class _RootData:
    def __init__(self, handle):
        self.handle = handle

    @property
    def default_root_state(self):
        return self.handle._art.data.default_root_state

    @property
    def root_state_w(self):
        h = self.handle
        # Preserve old RigidObject semantics: actor/link origin pose, COM velocity.
        return h._torch.cat((h._tensor(h._art.data.root_link_pose_w),
                             h._tensor(h._art.data.root_com_vel_w)), dim=-1)

    @property
    def root_pos_w(self):
        return self.handle._tensor(self.handle._art.data.root_link_pose_w)[..., :3]


class _CoreData(_RootData):
    @property
    def body_inertia(self):
        h = self.handle
        h.verify_initialized()
        return h._torch.as_tensor(h._array(h._art.root_view.get_inertias())[:, 0:1, :])


class _CoreAdapter:
    def __init__(self, handle):
        self.data = _CoreData(handle)


class PassiveCubeHandle:
    def __init__(self, stage, path, position, friction, hand_friction=None, *, baseline, spawn,
                 solver_position_iterations=64, solver_velocity_iterations=4):
        import numpy as np
        import torch
        from pxr import UsdPhysics, PhysxSchema, UsdShade, UsdGeom
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaac_compat import tensor, quaternion_from_native, runtime_versions

        self._np, self._torch, self._tensor = np, torch, tensor
        self._from_native = quaternion_from_native
        self._baseline = baseline
        self._verified = False
        self.allow_pose_reset = False
        self.pose_reset_count = 0
        self._stage = stage
        if (solver_position_iterations, solver_velocity_iterations) != (64, 4):
            raise ValueError("Only the independently probed explicit cube solver 64/4 is supported")
        self._solver_iterations = (solver_position_iterations, solver_velocity_iterations)
        self.model = spawn(stage, path, position)
        if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdPhysics.GetStageKilogramsPerUnit(stage) != 1.0:
            raise ValueError("Verified passive properties require SI stage units")
        # Detect source_core_inertia / legacy joint-damping wrappers before creating
        # any physics view; they are separate, incompatible candidates.
        if self.model.get("core_inertia_override") or self.model.get("passive_joint_damping"):
            raise ValueError("Passive articulation rejects source_core_inertia and legacy source_joint_damping")
        verify_baseline_inputs(baseline)
        asset_hashes = baseline["matching_fingerprint"]["source_sha256"]
        expected_asset = next((value for name, value in asset_hashes.items() if name.endswith("/cube_model.usda")), None)
        if not expected_asset or sha(path) != expected_asset:
            raise ValueError("Cube asset differs from the measured ordinary-body baseline")
        if runtime_versions() != baseline["matching_fingerprint"]["runtime_versions"]:
            raise ValueError("Runtime versions differ from measured baseline")
        if set(self.model["bodies"]) != set(baseline["rigid_baseline"]):
            raise ValueError("Original body paths changed")

        # Preserve DynamicCubeHandle's explicit material behavior and wrapper API.
        material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath("/World/Material"))
        self.model["source_friction"] = [material.GetStaticFrictionAttr().Get(), material.GetDynamicFrictionAttr().Get()]
        material.CreateStaticFrictionAttr(friction[0])
        material.CreateDynamicFrictionAttr(friction[1])
        self.model.update(runtime_friction=list(friction), friction_basis="explicit runtime cube friction; unchanged facade behavior")
        if hand_friction is not None:
            hand = UsdShade.Material.Define(stage, "/World/RobotHandContactMaterial")
            physics = UsdPhysics.MaterialAPI.Apply(hand.GetPrim())
            physics.CreateStaticFrictionAttr(hand_friction[0]); physics.CreateDynamicFrictionAttr(hand_friction[1])
            physics.CreateRestitutionAttr(0.0)
            PhysxSchema.PhysxMaterialAPI.Apply(hand.GetPrim()).CreateFrictionCombineModeAttr("max")
            bound = []
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith(("/World/Robot/left_", "/World/Robot/right_")) and prim.HasAPI(UsdPhysics.CollisionAPI):
                    binding = UsdShade.MaterialBindingAPI.Apply(prim)
                    binding.Bind(hand, UsdShade.Tokens.strongerThanDescendants, "physics")
                    resolved, _ = binding.ComputeBoundMaterial("physics")
                    if resolved.GetPath() != hand.GetPath():
                        raise RuntimeError("Hand contact material binding failed")
                    bound.append(str(prim.GetPath()))
            if not bound:
                raise ValueError("No hand collision shapes found")
            self.model.update(runtime_hand_friction=list(hand_friction), hand_material_colliders=bound,
                              hand_friction_combine_mode="max")

        joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)
                  and str(p.GetPath()).startswith("/World/Joints/")]
        if sum(p.IsA(UsdPhysics.RevoluteJoint) for p in joints) != 6 or sum(p.IsA(UsdPhysics.SphericalJoint) for p in joints) != 20 or len(joints) != 26:
            raise ValueError("Require original six hinges and twenty native spherical joints; no D6")
        children = []
        authored = []
        self._cube_joint_paths = [str(p.GetPath()) for p in joints]
        for prim in joints:
            joint = UsdPhysics.Joint(prim)
            parent = [str(p) for p in joint.GetBody0Rel().GetTargets()]
            child = [str(p) for p in joint.GetBody1Rel().GetTargets()]
            if parent != ["/World/Core"] or len(child) != 1 or joint.GetExcludeFromArticulationAttr().Get():
                raise ValueError("Original Core-to-cubelet tree changed")
            children.extend(child)
            if any(a.startswith("PhysicsDriveAPI:") for a in prim.GetAppliedSchemas()):
                raise ValueError("Cube drive API is incompatible with passive articulation")
            axes = ("angular",) if prim.IsA(UsdPhysics.RevoluteJoint) else ("rotX", "rotY", "rotZ")
            for axis in axes:
                if not prim.CanApplyAPI("PhysxJointAxisAPI", axis) or not prim.ApplyAPI("PhysxJointAxisAPI", axis):
                    raise RuntimeError("Registered PhysxJointAxisAPI unavailable")
                values = {"armature": ARMATURE, "staticFrictionEffort": DRY_FRICTION,
                          "dynamicFrictionEffort": DRY_FRICTION,
                          "viscousFrictionCoefficient": LINEAR_DAMPING * math.pi / 180}
                for name, value in values.items():
                    attr = prim.GetAttribute(f"physxJointAxis:{axis}:{name}")
                    if not attr or attr.IsCustom():
                        raise RuntimeError("Missing registered passive axis attribute")
                    attr.Set(value)
                authored.append({"joint": str(prim.GetPath()), "axis": axis, "usd_values": values})
        if len(set(children)) != 26 or set(children) != set(self.model["bodies"]) - {"/World/Core"}:
            raise ValueError("Original cube tree children changed")
        core = stage.GetPrimAtPath("/World/Core")
        if core.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError("Cube already has an articulation root")
        UsdPhysics.ArticulationRootAPI.Apply(core)
        articulation_props = PhysxSchema.PhysxArticulationAPI.Apply(core)
        articulation_props.CreateEnabledSelfCollisionsAttr(True)
        articulation_props.CreateSolverPositionIterationCountAttr(solver_position_iterations)
        articulation_props.CreateSolverVelocityIterationCountAttr(solver_velocity_iterations)
        for name in self.model["bodies"]:
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(name)).CreateThresholdAttr(0.0)
        self._art = Articulation(ArticulationCfg(prim_path="/World/Core", spawn=None, actuators={},
            joint_ordering=None, body_ordering=None,
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(position), joint_pos={".*": 0.0}, joint_vel={".*": 0.0})))
        self.data = _RootData(self)
        self.core = _CoreAdapter(self)
        self.model.update(runtime_model_version="passive_articulation_facade_draft_v1", angular_dofs=66,
            passive_articulation={"authored_axes": authored, "linear_damping_only": True,
                                  "omitted_quadratic_and_cubic_damping": [0.0005, 0.0005],
                                  "baseline_path": baseline["evidence_path"], "baseline_sha256": baseline["evidence_sha256"],
                                  "solver_iterations_requested": list(self._solver_iterations),
                                  "backend_verified": False})

    def _array(self, value):
        value = self._tensor(value)
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy().copy()
        return value.numpy().copy() if hasattr(value, "numpy") else self._np.asarray(value).copy()

    def verify_initialized(self):
        if self._verified:
            return
        # Existing wrappers run after the constructor. Recheck after the complete
        # wrapping chain, rather than assuming constructor-time metadata is final.
        from pxr import UsdPhysics, PhysxSchema
        if self.model.get("core_inertia_override") or self.model.get("passive_joint_damping"):
            raise ValueError("Late legacy inertia/damping wrapper is incompatible")
        joints = [self._stage.GetPrimAtPath(path) for path in self._cube_joint_paths]
        if (sum(p.IsA(UsdPhysics.RevoluteJoint) for p in joints) != 6
                or sum(p.IsA(UsdPhysics.SphericalJoint) for p in joints) != 20
                or any(any(a.startswith("PhysicsDriveAPI:") for a in p.GetAppliedSchemas()) for p in joints)):
            raise ValueError("Cube joint types/drives changed after passive factory construction")
        view = self._art.root_view
        if view.count != 1 or view.shared_metatype.fixed_base or self._art.num_joints != 66 or self._art.num_bodies != 27:
            raise RuntimeError("Expected one floating 27-link articulation with 66 angular DOFs")
        paths = list(view.link_paths[0])
        measured = validate_measured_mass_properties(self._baseline, paths, self._array(view.get_masses()),
                                                     self._array(view.get_inertias()), self._array(view.get_coms()))
        expected = {"armature": (view.get_dof_armatures, ARMATURE),
                    "friction": (view.get_dof_friction_properties, [DRY_FRICTION, DRY_FRICTION, LINEAR_DAMPING]),
                    "dof_types": (view.get_dof_types, 0),
                    "drive_stiffness": (view.get_dof_stiffnesses, 0), "drive_damping": (view.get_dof_dampings, 0),
                    "position_targets": (view.get_dof_position_targets, 0),
                    "velocity_targets": (view.get_dof_velocity_targets, 0), "actuation_forces": (view.get_dof_actuation_forces, 0)}
        reads = {}
        for name, (read, wanted) in expected.items():
            value = self._array(read())
            if not self._np.isfinite(value).all() or not self._np.allclose(value, wanted, rtol=1e-5, atol=1e-10):
                raise RuntimeError("Cube passive backend readback mismatch: " + name)
            reads[name] = value.tolist()
        names = [Path(path).name for path in paths]
        if len(set(names)) != 27:
            raise RuntimeError("Cubelet names are ambiguous")
        self._body_names = names
        properties = PhysxSchema.PhysxArticulationAPI(self._stage.GetPrimAtPath("/World/Core"))
        solver_values = (properties.GetSolverPositionIterationCountAttr().Get(),
                         properties.GetSolverVelocityIterationCountAttr().Get())
        if solver_values != self._solver_iterations:
            raise RuntimeError("Explicit cube solver iteration USD readback mismatch")
        self.model["passive_articulation"].update(backend_verified=True, measured_mass_properties=measured,
            backend_readback=reads, body_paths=paths, dof_names=list(view.shared_metatype.dof_names),
            dof_paths=view.dof_paths, solver_iterations={"requested": list(self._solver_iterations),
                "usd_readback": list(solver_values), "native_iteration_getter_available": False,
                "readback_scope": "USD articulation schema; installed tensor/binding API has no native iteration getter",
                "baseline_scope": "explicit 64/4 candidate and 64/4 development setting; no claim of original default solver equivalence"})
        self._verified = True

    def reset(self):
        self.verify_initialized()
        self._art.reset()

    def write_root_state_to_sim(self, state):
        if not self.allow_pose_reset:
            raise RuntimeError("Cube state writes are allowed only during episode reset")
        self.verify_initialized()
        value = self._tensor(state)
        if value.shape != (1, 13) or not self._torch.isfinite(value).all():
            raise ValueError("Expected finite native root state with shape (1,13)")
        q = self._torch.zeros((1, 66), dtype=value.dtype, device=value.device)
        self._art.write_root_link_pose_to_sim_index(root_pose=value[:, :7])
        self._art.write_root_com_velocity_to_sim_index(root_velocity=value[:, 7:13])
        self._art.write_joint_state_to_sim_index(position=q, velocity=q)
        observed_pose = self._array(self._art.root_view.get_root_transforms())
        observed_velocity = self._array(self._art.root_view.get_root_velocities())
        requested = self._array(value)
        positions_match = self._np.allclose(observed_pose[:, :3], requested[:, :3], atol=1e-6)
        quat_dot = self._np.abs(self._np.sum(observed_pose[:, 3:7] * requested[:, 3:7], axis=1))
        checks = {"root_pose": positions_match and self._np.allclose(quat_dot, 1., atol=1e-6),
                  "root_velocity": self._np.allclose(observed_velocity, requested[:, 7:13], atol=1e-6),
                  "joint_positions": self._np.allclose(self._array(self._art.root_view.get_dof_positions()), 0., atol=1e-7),
                  "joint_velocities": self._np.allclose(self._array(self._art.root_view.get_dof_velocities()), 0., atol=1e-7)}
        if not all(checks.values()):
            raise RuntimeError("Cube reset readback mismatch: " + str(checks))
        self.pose_reset_count += 1
        self.model["passive_articulation"]["last_reset_readback"] = {
            "count": self.pose_reset_count, "passed": True, "root_pose": observed_pose.tolist(),
            "root_com_velocity": observed_velocity.tolist(), "all_66_q_and_qd_zero": True}

    def write_data_to_sim(self):
        # No cube actuation, targets, external forces, or control-phase state writes.
        pass

    def update(self, dt):
        self._art.update(dt)

    def states(self):
        self.verify_initialized()
        poses = self._tensor(self._art.data.body_link_pose_w)[0]
        velocities = self._tensor(self._art.data.body_com_vel_w)[0]
        return {name: {"position_m": poses[i, :3].tolist(),
                       "quaternion_wxyz": self._from_native(poses[i, 3:7]).tolist(),
                       "linear_velocity_m_s": velocities[i, :3].tolist(),
                       "angular_velocity_rad_s": velocities[i, 3:].tolist()}
                for i, name in enumerate(self._body_names)}


@contextmanager
def passive_cube_articulation(baseline_path, baseline_sha256, *, module=None,
                              source_core_inertia=False, source_joint_damping=False,
                              solver_position_iterations=64, solver_velocity_iterations=4):
    """Outer factory context: enter BEFORE existing material/precision wrappers."""
    if source_core_inertia or source_joint_damping:
        raise ValueError("Passive articulation rejects both legacy source_core_inertia and source_joint_damping options")
    if module is None:
        import articulated_cube_env as module
    original = module.DynamicCubeHandle
    if not isinstance(original, type) or original.__name__ != "DynamicCubeHandle":
        raise ValueError("Enter passive_cube_articulation outside existing cube factory wrappers")
    baseline = load_baseline(baseline_path, baseline_sha256)
    import cube_core_inertia as legacy_core
    import cube_joint_damping as legacy_damping
    saved_core = legacy_core.apply_source_core_inertia
    saved_damping = legacy_damping.apply_source_joint_damping

    def reject_legacy(*args, **kwargs):
        # Inner legacy wrappers call these globals after constructing their handle.
        # Reject before their first USD mutation, even if a caller forgot CLI checks.
        raise ValueError("Legacy source Core inertia / D6 joint damping cannot be combined with passive articulation")

    def build(stage, path, position, friction, hand_friction=None):
        return PassiveCubeHandle(stage, path, position, friction, hand_friction,
                                 baseline=baseline, spawn=module.spawn_dynamic_cube,
                                 solver_position_iterations=solver_position_iterations,
                                 solver_velocity_iterations=solver_velocity_iterations)

    module.DynamicCubeHandle = build
    legacy_core.apply_source_core_inertia = reject_legacy
    legacy_damping.apply_source_joint_damping = reject_legacy
    try:
        yield
    finally:
        module.DynamicCubeHandle = original
        legacy_core.apply_source_core_inertia = saved_core
        legacy_damping.apply_source_joint_damping = saved_damping
