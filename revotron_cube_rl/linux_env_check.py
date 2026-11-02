"""Check the Linux Isaac runtime; --runtime boots an empty Isaac application.

The default mode is safe on a machine that has only the project files.  Pass
``--runtime`` on the Linux workstation to require the Isaac imports and CUDA.
The check reports actionable environment variables rather than silently using a
Windows path or a different Python installation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _nvidia_smi() -> dict[str, object]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except OSError as exc:
        return {"available": False, "reason": str(exc)}
    return {"available": proc.returncode == 0, "returncode": proc.returncode,
            "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="require Isaac Lab/Sim imports and a CUDA device")
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    parser.add_argument("--render", action="store_true", help="also probe RTX camera renderer (requires --runtime)")
    args = parser.parse_args()
    if args.render and not args.runtime:
        parser.error("--render requires --runtime")

    robot = _path_from_env("REVOTRON_USD_PATH", ROOT.parent / "revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd")
    cube = _path_from_env("ISAAC_MUJOCO_CUBE_ROOT", ROOT.parent / "isaac_mujoco_cube")
    required_files = {
        "robot_usd": robot,
        "cube_xml": cube / "upstream/mujoco_cube-main/cube_3x3x3.xml",
        "cube_texture": cube / "upstream/mujoco_cube-main/assets/sticker.png",
    }
    modules = {
        name: _module_available(name)
        for name in ("numpy", "scipy", "torch", "PIL", "tensordict", "rsl_rl", "isaaclab", "isaacsim", "pxr")
    }
    report: dict[str, object] = {
        "project_root": str(ROOT),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "asset_paths": {key: str(path) for key, path in required_files.items()},
        "asset_exists": {key: path.is_file() for key, path in required_files.items()},
        "modules": modules,
        "nvidia_smi": _nvidia_smi(),
        "environment": {
            "ISAACLAB_PYTHON": os.environ.get("ISAACLAB_PYTHON", ""),
            "REVOTRON_USD_PATH": os.environ.get("REVOTRON_USD_PATH", ""),
            "ISAAC_MUJOCO_CUBE_ROOT": os.environ.get("ISAAC_MUJOCO_CUBE_ROOT", ""),
        },
    }
    errors: list[str] = []
    if os.name != "posix":
        errors.append("this runtime check must be run on Linux/posix; use linux_migration_check.py for Windows static checks")
    for key, path in required_files.items():
        if not path.is_file():
            errors.append(f"missing {key}: {path}")
    for name in ("numpy", "scipy", "torch", "PIL", "tensordict", "rsl_rl"):
        if not modules[name]:
            errors.append(f"missing Python module: {name}")
    report["rendering_requested"] = args.render
    if args.runtime:
        # pxr is exposed by Kit extensions only after SimulationApp starts.
        if modules["isaaclab"] and modules["isaacsim"]:
            try:
                probe = subprocess.run(
                    [sys.executable, str(ROOT / "isaac_runtime_probe.py")] + (["--render"] if args.render else []),
                    capture_output=True, text=True, timeout=600, check=False,
                )
                log = ROOT.parent / "work" / "isaac_runtime_probe.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(probe.stdout + probe.stderr)
                report["isaac_probe_log"] = str(log)
                for line in probe.stdout.splitlines():
                    if line.startswith("ISAAC_RUNTIME_PROBE="):
                        result = json.loads(line.split("=", 1)[1])
                        modules["pxr"] = bool(result["pxr"]) and probe.returncode == 0
                        report["usd_version"] = result["usd_version"]
                report["isaac_probe_returncode"] = probe.returncode
                if probe.returncode != 0:
                    errors.append(f"Isaac application probe failed (exit {probe.returncode}); see {log}")
            except (OSError, subprocess.TimeoutExpired) as exc:
                report["isaac_probe_error"] = str(exc)
                errors.append(f"Isaac application probe did not complete: {exc}")
                modules["pxr"] = False
                if isinstance(exc, subprocess.TimeoutExpired):
                    log = ROOT.parent / "work" / "isaac_runtime_probe.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    def decoded(value):
                        return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
                    log.write_text(decoded(exc.stdout) + decoded(exc.stderr))
                    report["isaac_probe_log"] = str(log)
        for name in ("isaaclab", "isaacsim", "pxr"):
            if not modules[name]:
                errors.append(f"runtime mode requires Python module: {name}")
        cuda_ok = False
        if modules["torch"]:
            try:
                import torch  # type: ignore
                cuda_ok = bool(torch.cuda.is_available())
                report["torch_cuda_device_count"] = int(torch.cuda.device_count())
            except Exception as exc:  # pragma: no cover - depends on host runtime
                report["torch_cuda_error"] = repr(exc)
        if not cuda_ok:
            errors.append("runtime mode requires torch.cuda.is_available() == True")
    report["status"] = "passed" if not errors else "failed"
    report["errors"] = errors
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("LINUX_ENV_CHECK " + report["status"].upper())
        print("python=", report["python"])
        print("project_root=", report["project_root"])
        print("robot_usd=", required_files["robot_usd"])
        print("cube_root=", cube)
        print("isaaclab=", modules["isaaclab"], "isaacsim=", modules["isaacsim"], "cuda_required=", args.runtime)
        for error in errors:
            print(" -", error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
