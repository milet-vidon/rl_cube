"""Create a reproducible Linux transfer archive for the Revo3 cube project.

The default archive contains code, configuration, scene snapshots, migration
checks and status records.  Use ``--run`` for selected checkpoints/datasets and
``--include-assets`` to add the adjacent Revo3 USD/URDF and mujoco_cube files.
It intentionally does not copy the whole historical ``runs`` directory by
default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = (
    "README.txt", "PROJECT.md", "FRAMEWORK_CHAIN.md", "STATUS.txt",
    "current_status.json", "config.json", "asset_contract.json",
    "camera_calibration.json", "curriculum.json", "locked_cube.usda",
    "training_scene.usda", "manifest.json", "LINUX_MIGRATION.md",
    "MIGRATION_AUDIT_20261102.md", "run_linux.sh", "linux_env_check.py", "isaac_runtime_probe.py", "runtime_platform.py", "isaac_compat.py", "rsl_compat.py", "test_migration_contract.py",
    "linux_migration_check.py", "prepare_linux_bundle.py",
    "environment.py", "assets.py", "kinematics.py", "joint_safety.py",
    "rewards.py", "policy.py", "absolute_policy.py", "position_policy.py",
    "train.py", "position_train.py", "warmstart.py", "collect_dagger.py",
    "iterative_training.py", "queue_continuation.py",
    "traditional_controller.py", "sweep_controller.py", "robot_hand_turn_stage.py",
    "test_reward_gates.py", "test_arm_safety.py", "arm_limit_audit.py",
    "arm_limit_runtime_audit.py",
    "acceptance.py", "stage_gate.py", "audit_teacher.py", "transfer_protocol.json",
    "METHOD_TRANSFER.txt", "pipeline_status.json", "cube_state.py", "dynamic_cube.py",
    "audit_dynamic_cube.py", "test_cube_state.py", "test_transfer_safety.py",
    "side_grasp.py", "side_control.py", "self_contact_audit.py", "side_gate.py",
    "side_experiment.py", "side_waypoints.py", "side_policy_evaluate.py", "side_video.py",
    "test_side_control.py", "SIDE_GRASP_TRAINING.txt",
    "state_learning.py", "state_policy.py", "test_state_policy.py", "STATE_TRAINING.txt",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add_file(archive: tarfile.TarFile, path: Path, arcname: str, included: list[dict]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    archive.add(path, arcname=arcname)
    included.append({"path": arcname, "bytes": path.stat().st_size, "sha256": sha256(path)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True, help="output .tar.gz path")
    ap.add_argument("--run", action="append", default=[], help="runs/<name> to include; repeatable")
    ap.add_argument("--include-assets", action="store_true", help="include adjacent robot and cube assets")
    args = ap.parse_args()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    included: list[dict] = []
    with tarfile.open(output, "w:gz") as archive:
        for rel in DEFAULT_FILES:
            path = ROOT / rel
            if path.is_file():
                add_file(archive, path, f"revotron_cube_rl/{rel}", included)
        for name in args.run:
            run = (ROOT / "runs" / name).resolve()
            if not run.is_dir() or ROOT.joinpath("runs").resolve() not in run.parents:
                raise ValueError(f"run must be a child of {ROOT / 'runs'}: {name}")
            for path in sorted(run.rglob("*")):
                if path.is_file():
                    rel = path.relative_to(ROOT).as_posix()
                    add_file(archive, path, f"revotron_cube_rl/{rel}", included)
        if args.include_assets:
            asset_root = ROOT.parent
            robot_root = asset_root / "revotron_revo3/source/revotron_system"
            for rel in (
                "usd/revotron_bimanual_revo3_56dof.usd",
                "urdf/revotron_bimanual_revo3_56dof.urdf",
            ):
                path = robot_root / rel
                add_file(archive, path, f"revotron_revo3/source/revotron_system/{rel}", included)
            cube_root = asset_root / "isaac_mujoco_cube/upstream/mujoco_cube-main"
            for rel in ("cube_3x3x3.xml", "assets/sticker.png"):
                path = cube_root / rel
                add_file(archive, path, f"isaac_mujoco_cube/upstream/mujoco_cube-main/{rel}", included)

    report = {
        "archive": str(output),
        "file_count": len(included),
        "include_assets": bool(args.include_assets),
        "runs": args.run,
        "files": included,
        "next_step": "extract on Linux, set ISAACLAB_LAUNCHER or ISAACLAB_PYTHON, then run ./revotron_cube_rl/run_linux.sh check",
    }
    report_path = output.with_suffix(output.suffix + ".manifest.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(output), "manifest": str(report_path), "file_count": len(included)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
