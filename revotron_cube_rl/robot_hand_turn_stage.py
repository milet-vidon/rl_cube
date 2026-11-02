"""Bilateral Revo3/Tianji contact-turn stage.

This is the first hand-contact teacher stage after the external-fixture gate.
It loads the actual 56-DOF Revo3 asset and the free-core 3x3 cube in one
PhysX scene, runs the measured bilateral pregrasp trajectory, then executes a
slow, contact-only wrist sweep.  No cube pose, layer joint, or torque is
written by the controller.  The result is kept as a teacher/diagnostic until
the contact gate passes on held-out seeds; it is intentionally not marked as
RL.
"""
from __future__ import annotations
from isaac_compat import prepare_single_world, normalize_windows_asset_paths
from isaac_compat import close_application

from isaac_compat import tensor, quaternion_to_native, quaternion_from_native, physx_config, simulation_config, runtime_versions, LAB3
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CUBE_ROOT = Path(os.environ.get("ISAAC_MUJOCO_CUBE_ROOT", str(ROOT.parent / "isaac_mujoco_cube"))).expanduser().resolve()
CUBE_USD = CUBE_ROOT / "cube_model.usda"

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="robot_hand_turn_stage")
ap.add_argument("--seed", type=int, default=2041)
ap.add_argument("--steps", type=int, default=840)
ap.add_argument("--capture-every", type=int, default=2)
ap.add_argument("--sweep-count", type=int, default=30)
args = ap.parse_args()

cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
run = ROOT / "runs" / args.name
run.mkdir(parents=True, exist_ok=True)

from stage_gate import require_staged_gate
require_staged_gate()
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args

app = AppLauncher(
    headless=True,
    enable_cameras=True,
    kit_args=platform_kit_args('--/app/vulkan=false --/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false'),
).app

from pxr import Gf, PhysxSchema, PhysicsSchemaTools, UsdGeom, UsdPhysics
import omni.physx
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import quat_apply

from kinematics import RobotKinematics
from traditional_controller import DownwardController
from joint_safety import soft_limits_from_hard, max_target_delta, limit_metadata, project_target


class _CubeShim:
    def __init__(self, xyz):
        self.data = type("Data", (), {"root_pos_w": torch.tensor([xyz], dtype=torch.float32)})()


def _set_translate(stage, path, xyz):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"missing free-core prim: {path}")
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*map(float, xyz)))


def _quaternion_angle(q0, q1):
    # USD quaternions expose real/imaginary components.
    a = np.array([q0.GetReal(), *q0.GetImaginary()], dtype=np.float64)
    b = np.array([q1.GetReal(), *q1.GetImaginary()], dtype=np.float64)
    dot = float(np.clip(abs(np.dot(a, b)), -1., 1.))
    return float(2. * math.degrees(math.acos(dot)))


class Scene:
    def __init__(self):
        if not CUBE_USD.is_file():
            raise FileNotFoundError(CUBE_USD)
        self.cfg = cfg
        self.sim = sim_utils.SimulationContext(
            simulation_config(
                dt=1.0 / cfg["physics_hz"], device="cpu", use_fabric=False,
                render_interval=cfg["physics_hz"] // cfg["control_hz"],
                physx=physx_config(solver_type=1, min_position_iteration_count=32,
                                          max_position_iteration_count=64,
                                          min_velocity_iteration_count=4),
            )
        )
        self.stage = self.sim.stage
        # The free-core asset already defines /World/Core, /World/Cube and all
        # layer joints.  A sublayer keeps those absolute joint paths intact.
        self.stage.GetRootLayer().subLayerPaths.append(str(CUBE_USD))
        target = cfg["cube_xyz"]
        _set_translate(self.stage, "/World/Core", target)
        _set_translate(self.stage, "/World/Cube", target)
        self._add_table()
        self._add_robot()
        self._add_cameras()
        self._contact_paths = {}
        self.contact_events = 0
        self.robot_cube_impulse = 0.0
        self.layer_contacts = 0
        self.sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
        self.asset_path_overrides = normalize_windows_asset_paths(self.stage)
        prepare_single_world(self.sim)
        self.sim.reset()
        self.ids, _ = self.robot.find_joints(cfg["joint_names"], preserve_order=True)
        if len(self.ids) != 56:
            raise RuntimeError(f"expected 56 joints, got {len(self.ids)}")
        self.home = torch.tensor([[cfg["home"][n] for n in cfg["joint_names"]]], dtype=torch.float32)
        self.hard_limits = tensor(self.robot.data.joint_pos_limits)[:, self.ids].clone()
        self.limits = soft_limits_from_hard(self.hard_limits, cfg["joint_names"], cfg)
        self.scales = torch.tensor([[.025 if n.startswith("Joint") else .06 for n in cfg["joint_names"]]])
        self.max_target_delta = max_target_delta(cfg["joint_names"], self.scales, cfg)
        self.targets = self.home.clone()
        self.limit_hit_count = 0
        self.robot.reset()
        q = tensor(self.robot.data.default_joint_pos).clone()
        q[:, self.ids] = self.home
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q)
        for _ in range(8):
            self.physics()
        self.kin = RobotKinematics()
        self.frames = {s: self._palm_frame(s) for s in ("left", "right")}
        self.initial_layer = self._layer_state()
        self.overview_frames = []
        self.chest_frames = []
        self.sim.render()
        self.camera.update(1.0 / cfg["control_hz"], force_recompute=True)
        self.overview.update(1.0 / cfg["control_hz"], force_recompute=True)

    def _add_table(self):
        table = sim_utils.CuboidCfg(
            size=(.8, .9, .04),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.32, .35, .4)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.7),
        )
        table.func("/World/Table", table, translation=(.60, 0., cfg["table_z"] - .02))
        ground = sim_utils.CuboidCfg(size=(5., 5., .02), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.17, .19, .22)))
        ground.func("/World/Ground", ground, translation=(0., 0., -.011))
        light = sim_utils.DomeLightCfg(intensity=1800.)
        light.func("/World/Light", light)

    def _add_robot(self):
        robot_path = Path(os.environ.get("REVOTRON_USD_PATH", str(ROOT.parent / "revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd"))).expanduser().resolve()
        arms = [n for n in cfg["joint_names"] if n.startswith("Joint")]
        fingers = [n for n in cfg["joint_names"] if not n.startswith("Joint")]
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(robot_path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True, solver_position_iteration_count=32, solver_velocity_iteration_count=4),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., cfg["robot_z"]), joint_pos=cfg["home"]),
            actuators={
                "arms": ImplicitActuatorCfg(joint_names_expr=arms, stiffness=cfg["arm_stiffness"], damping=1., effort_limit_sim=cfg["arm_effort"], velocity_limit_sim=3.1416 * cfg.get("arm_velocity_limit_scale", 1.0)),
                "fingers": ImplicitActuatorCfg(joint_names_expr=fingers, stiffness=30., damping=1., effort_limit_sim=5., velocity_limit_sim=10.),
            },
            soft_joint_pos_limit_factor=1.,
        )
        self.robot = Articulation(robot_cfg)
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).startswith("/World/Robot/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)

    def _add_cameras(self):
        eye = Gf.Vec3d(*cfg["camera_eye"])
        target = Gf.Vec3d(*cfg["camera_target"])
        m = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
        q = m.ExtractRotationQuat()
        self.camera = Camera(CameraCfg(
            prim_path="/World/Robot/world/ChestCamera", height=cfg["camera_resolution"], width=cfg["camera_resolution"],
            update_period=0., update_latest_camera_pose=True, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=16., horizontal_aperture=24., clipping_range=(.02, 5.)),
            offset=CameraCfg.OffsetCfg(pos=(eye[0], eye[1], eye[2] - cfg["robot_z"]), rot=quaternion_to_native((q.GetReal(), *q.GetImaginary())), convention="opengl"),
        ))
        overview_eye = Gf.Vec3d(1.9, -1.8, 1.65)
        overview_target = Gf.Vec3d(.43, 0., 1.03)
        om = Gf.Matrix4d().SetLookAt(overview_eye, overview_target, Gf.Vec3d(0, 0, 1)).GetInverse()
        oq = om.ExtractRotationQuat()
        self.overview = Camera(CameraCfg(
            prim_path="/World/Overview", height=512, width=640, update_latest_camera_pose=True, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=30., horizontal_aperture=24.),
            offset=CameraCfg.OffsetCfg(pos=tuple(overview_eye), rot=quaternion_to_native((oq.GetReal(), *oq.GetImaginary())), convention="opengl"),
        ))

    def _on_contact(self, headers, data):
        for h in headers:
            paths = []
            for x in (h.actor0, h.actor1):
                if x not in self._contact_paths:
                    self._contact_paths[x] = str(PhysicsSchemaTools.intToSdfPath(x))
                paths.append(self._contact_paths[x])
            if not any("/World/Cube/" in p for p in paths):
                continue
            self.contact_events += 1
            if any(p.startswith("/World/Robot/") for p in paths):
                self.layer_contacts += 1
                for item in data[h.contact_data_offset:h.contact_data_offset + h.num_contact_data]:
                    self.robot_cube_impulse += float(np.linalg.norm(np.asarray(item.impulse)))

    def _palm_frame(self, side):
        # Same geometric center as the pickup environment.
        name = side + "_palm_link"
        ids, _ = self.robot.find_bodies(name)
        return ids[0]

    def palm_position(self, side):
        idx = self.frames[side]
        center = torch.tensor([[0., 0., .09]], dtype=torch.float32)
        return tensor(self.robot.data.body_link_pos_w)[:, idx] + quat_apply(tensor(self.robot.data.body_link_quat_w)[:, idx], center)

    def physics(self):
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(1.0 / cfg["physics_hz"])

    def render(self, keep=True):
        omni.physx.get_physx_interface().update_transformations(False, True, True, False)
        self.sim.render()
        self.camera.update(1.0 / cfg["control_hz"], force_recompute=True)
        self.overview.update(1.0 / cfg["control_hz"], force_recompute=True)
        if keep:
            self.chest_frames.append(tensor(self.camera.data.output["rgb"])[0, :, :, :3].cpu().numpy().astype(np.uint8))
            self.overview_frames.append(tensor(self.overview.data.output["rgb"])[0, :, :, :3].cpu().numpy().astype(np.uint8))

    def command(self, target, render_frame=True):
        requested = torch.as_tensor(target, dtype=torch.float32).reshape(1, 56)
        velocity_limit=torch.tensor([[3.1416*cfg.get('arm_velocity_limit_scale',1.) if n.startswith('Joint') else 10. for n in cfg['joint_names']]])
        target, _ = project_target(self.targets, requested, self.limits, self.max_target_delta,
                                   velocity_limit, 1./cfg['control_hz'])
        self.limit_hit_count += int(((requested < self.limits[..., 0]) | (requested > self.limits[..., 1])).sum())
        self.targets = target.clone()
        self.robot.set_joint_position_target(target, joint_ids=self.ids)
        for _ in range(cfg["physics_hz"] // cfg["control_hz"]):
            self.physics()
        if render_frame:
            self.render()

    def _layer_state(self):
        cache = UsdGeom.XformCache()
        names = ["pZ", "pX_pZ", "nZ_pX", "nX_pZ", "pY_pZ", "nZ_pY", "nY_pZ", "nZ_pX_pY", "nY_nX_pZ", "nX_nY_pZ"]
        out = {}
        for name in names:
            prim = self.stage.GetPrimAtPath("/World/Cube/" + name)
            if not prim.IsValid():
                continue
            xf = cache.GetLocalToWorldTransform(prim)
            out[name] = {
                "position": [float(x) for x in xf.ExtractTranslation()],
                "quaternion": [float(xf.ExtractRotationQuat().GetReal()), *[float(x) for x in xf.ExtractRotationQuat().GetImaginary()]],
            }
        return out

    def save_artifacts(self):
        from PIL import Image
        from PIL import ImageDraw
        chest = [Image.fromarray(x) for x in self.chest_frames]
        overview = [Image.fromarray(x) for x in self.overview_frames]
        if overview:
            overview[0].save(run / "overview.gif", save_all=True, append_images=overview[1:], duration=1000 // cfg["control_hz"], loop=0)
            overview[-1].save(run / "overview_final.png")
        if chest:
            chest[0].save(run / "chest.gif", save_all=True, append_images=chest[1:], duration=1000 // cfg["control_hz"], loop=0)
            chest[-1].save(run / "chest_final.png")


def _make_controller(scene):
    shim = type("Env", (), {})()
    shim.cfg = cfg
    shim.cube = _CubeShim(cfg["cube_xyz"])
    shim.targets = scene.targets
    shim.scales = scene.scales
    controller = DownwardController(shim, width=.102, lift=.0)
    # Rebind after construction so each command uses the live scene target.
    controller.env = shim
    return controller, shim


def main():
    scene = Scene()
    controller, shim = _make_controller(scene)
    trajectory = []
    for step in range(min(args.steps, len(controller.phases) * 2)):
        shim.targets = scene.targets
        action = controller.action()
        scene.command(scene.targets + scene.scales * action, render_frame=(step % max(1, args.capture_every) == 0))
        trajectory.append({"step": step, "phase": controller.name, "palm_left": scene.palm_position("left")[0].tolist(), "palm_right": scene.palm_position("right")[0].tolist(), "layer_state": scene._layer_state() if step % max(1, args.capture_every) == 0 else None, "robot_cube_impulse": scene.robot_cube_impulse, "layer_contacts": scene.layer_contacts})

    # Contact-turn teacher: make slow, tangential wrist sweeps while fingers
    # stay closed.  The sweep is physical contact; no cube state is written.
    base = scene.targets.clone()
    q_sweeps = []
    # Keep this first probe cheap and reproducible: vary the two wrist yaw
    # joints around the measured pregrasp target while all finger joints stay
    # closed.  This is still a physical robot-link contact command; it is not
    # a direct layer-joint or cube-pose write.  Cartesian IK sweeps are added
    # after contact geometry is confirmed.
    wrist_indices = [cfg["joint_names"].index("Joint7_L"), cfg["joint_names"].index("Joint7_R")]
    for axis, sign in (("z", 1.), ("x", -1.)):
        for k in range(max(2, args.sweep_count)):
            a = math.radians(25.) * (k + 1) / max(2, args.sweep_count) * sign
            target = base.clone()
            target[0, wrist_indices[0]] += a
            target[0, wrist_indices[1]] -= a
            q_sweeps.append(target)
        for _ in range(max(2, args.sweep_count // 6)):
            q_sweeps.append(q_sweeps[-1].clone())

    for target in q_sweeps:
        scene.command(target, render_frame=(len(trajectory) % max(1, args.capture_every) == 0))
        trajectory.append({"step": len(trajectory), "phase": "contact_sweep", "palm_left": scene.palm_position("left")[0].tolist(), "palm_right": scene.palm_position("right")[0].tolist(), "layer_state": scene._layer_state() if len(trajectory) % max(1, args.capture_every) == 0 else None, "robot_cube_impulse": scene.robot_cube_impulse, "layer_contacts": scene.layer_contacts})

    scene.save_artifacts()
    final_state = scene._layer_state()
    initial = scene.initial_layer
    angles = []
    for name, row in final_state.items():
        if name in initial:
            q0 = type("Q", (), {"GetReal": lambda self: initial[name]["quaternion"][0], "GetImaginary": lambda self: Gf.Vec3d(*initial[name]["quaternion"][1:])})()
            q1 = type("Q", (), {"GetReal": lambda self: row["quaternion"][0], "GetImaginary": lambda self: Gf.Vec3d(*row["quaternion"][1:])})()
            angles.append(_quaternion_angle(q0, q1))
    result = {
        "mode": "robot_hand_contact_turn_teacher_probe",
        "status": "completed",
        "action_dim": 56,
        "seed": args.seed,
        "steps_total": len(trajectory),
        "physical_model": "Revo3/Tianji 56DOF + source cube asset with kinematic core",
        "cube_pose_or_joint_writes": False,
        "external_fixtures": True,
        "fixture_reason": "Source cube_model.usda has a kinematic Core; this legacy diagnostic is not a free-cube task",
        "robot_cube_contact_events": scene.layer_contacts,
        "robot_cube_impulse_sum": scene.robot_cube_impulse,
        "joint_limit_safety": limit_metadata(scene.hard_limits, scene.limits, cfg["joint_names"], cfg),
        "joint_limit_hit_count": scene.limit_hit_count,
        "max_layer_angle_deg": max(angles) if angles else 0.,
        "turn_target_deg": 30.,
        "robot_hand_turning_validated": False,
        "measurement_limitation": "World-frame cubelet orientation change cannot establish a signed 90-degree face turn relative to the cube core",
        "rl_turning_trained": False,
        "artifacts": {"overview_gif": "overview.gif", "overview_final": "overview_final.png", "chest_gif": "chest.gif", "chest_final": "chest_final.png", "trajectory": "trajectory.json"},
        "next_gate": "increase contact-turn teacher to 90 degrees and repeat on held-out seeds before PPO",
    }
    (run / "trajectory.json").write_text(json.dumps(trajectory, indent=2, ensure_ascii=False), encoding="utf-8")
    (run / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


code = 1
try:
    print("APP_READY", flush=True)
    code = main()
finally:
    try:
        close_application(app)
    except Exception:
        pass
raise SystemExit(code)
