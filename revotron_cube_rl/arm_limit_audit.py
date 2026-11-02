"""Audit Revo3 seven-axis hard limits and project soft motion envelope."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
URDF = ROOT.parent / "revotron_revo3/source/revotron_system/urdf/revotron_bimanual_revo3_56dof.urdf"

root = ET.parse(URDF).getroot()
hard = {}
for joint in root.findall("joint"):
    name = joint.get("name", "")
    if name.startswith("Joint") and name.endswith(("_L", "_R")):
        limit = joint.find("limit")
        hard[name] = {k: float(limit.get(k)) for k in ("lower", "upper", "velocity", "effort")}

margin = float(CFG.get("arm_soft_limit_margin_rad", 0.15))
explicit = CFG.get("arm_soft_limits", {})
soft = {}
rows = []
for name, values in hard.items():
    requested_lo, requested_hi = (map(float, explicit[name]) if name in explicit else (values["lower"] + margin, values["upper"] - margin))
    lo, hi = max(values["lower"], requested_lo), min(values["upper"], requested_hi)
    home = float(CFG["home"][name])
    row = {
        "joint": name,
        "hard_lower_rad": values["lower"],
        "hard_upper_rad": values["upper"],
        "soft_lower_rad": lo,
        "soft_upper_rad": hi,
        "requested_soft_lower_rad": requested_lo,
        "requested_soft_upper_rad": requested_hi,
        "home_rad": home,
        "home_inside_soft": lo <= home <= hi,
        "soft_margin_lower_rad": lo - values["lower"],
        "soft_margin_upper_rad": values["upper"] - hi,
        "velocity_limit_rad_s": values["velocity"],
        "effort_limit": values["effort"],
    }
    rows.append(row)
    soft[name] = [lo, hi]

result = {
    "status": "passed" if len(rows) == 14 and all(r["home_inside_soft"] for r in rows) else "failed",
    "joint_count": len(rows),
    "both_arms_seven_joints": len(rows) == 14,
    "urdf_path": str(URDF),
    "hard_limit_source": "Revo3 URDF; runtime USD parity still checked by Isaac articulation initialization",
    "soft_limit_source": "project software envelope; explicit user degree limits intersected with imported hard limits",
    "arm_soft_limit_margin_rad": margin,
    "arm_max_step_rad": float(CFG.get("arm_max_step_rad", 0.04)),
    "arm_velocity_limit_scale": float(CFG.get("arm_velocity_limit_scale", 0.5)),
    "joints": rows,
}
out = ROOT / "runs" / "arm_limit_audit"
out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
(ROOT / "arm_limit_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
raise SystemExit(0 if result["status"] == "passed" else 1)
