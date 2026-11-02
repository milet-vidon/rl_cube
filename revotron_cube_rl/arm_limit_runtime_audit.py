"""Runtime USD/PhysX versus URDF arm-limit parity check."""
from __future__ import annotations
from isaac_compat import prepare_single_world, normalize_windows_asset_paths
from isaac_compat import close_application

from isaac_compat import tensor, quaternion_to_native, quaternion_from_native, physx_config, simulation_config, runtime_versions, LAB3
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
static = json.loads((ROOT / "arm_limit_audit.json").read_text(encoding="utf-8"))

from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
app = AppLauncher(headless=True, enable_cameras=False, kit_args=platform_kit_args('--/app/vulkan=false --/app/useFabricSceneDelegate=false')).app
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

sim = sim_utils.SimulationContext(simulation_config(dt=1.0 / cfg["physics_hz"], device="cpu", use_fabric=False, physx=physx_config(solver_type=1)))
robot_path = Path(os.environ.get("REVOTRON_USD_PATH", str(ROOT.parent / "revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd"))).expanduser().resolve()
arms = [n for n in cfg["joint_names"] if n.startswith("Joint")]
fingers = [n for n in cfg["joint_names"] if not n.startswith("Joint")]
robot = Articulation(ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=str(robot_path), rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True), articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., cfg["robot_z"]), joint_pos=cfg["home"]),
    actuators={
        "arms": ImplicitActuatorCfg(joint_names_expr=arms, stiffness=cfg["arm_stiffness"], damping=1., effort_limit_sim=cfg["arm_effort"], velocity_limit_sim=3.1416 * cfg.get("arm_velocity_limit_scale", 1.0)),
        "fingers": ImplicitActuatorCfg(joint_names_expr=fingers, stiffness=30., damping=1., effort_limit_sim=5., velocity_limit_sim=10.),
    },
    soft_joint_pos_limit_factor=1.0,
))
normalize_windows_asset_paths(sim.stage)
prepare_single_world(sim)
sim.reset()
ids, _ = robot.find_joints(cfg["joint_names"], preserve_order=True)
runtime = tensor(robot.data.joint_pos_limits)[0, ids].detach().cpu()
rows = []
max_error = 0.0
for name in arms:
    i = cfg["joint_names"].index(name)
    expected = next(r for r in static["joints"] if r["joint"] == name)
    actual = [float(runtime[i, 0]), float(runtime[i, 1])]
    err = max(abs(actual[0] - expected["hard_lower_rad"]), abs(actual[1] - expected["hard_upper_rad"]))
    max_error = max(max_error, err)
    rows.append({"joint": name, "urdf_hard_rad": [expected["hard_lower_rad"], expected["hard_upper_rad"]], "usd_physx_hard_rad": actual, "max_abs_error_rad": err, "passed": err <= 1e-4})
result = {"status": "passed" if max_error <= 1e-4 and len(rows) == 14 else "failed", "joint_count": len(rows), "max_abs_error_rad": max_error, "rows": rows, "soft_limits_applied_by_project": True, "note": "Isaac soft_joint_pos_limit_factor remains 1.0; project joint_safety.py applies the separate human-like software envelope before set_joint_position_target."}
out = ROOT / "runs" / "arm_limit_runtime_audit"
out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
sim.stop()
close_application(app)
raise SystemExit(0 if result["status"] == "passed" else 1)
