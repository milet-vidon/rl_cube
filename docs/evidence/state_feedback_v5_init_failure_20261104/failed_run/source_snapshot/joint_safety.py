"""Shared arm soft-limit and motion-rate safety helpers.

The imported URDF limits remain the hard articulation limits.  The training
controller uses a smaller, configurable software envelope so exploration does
not repeatedly drive the seven arm joints into end stops.  The same helper is
used by Isaac control, IK and the contact-turn probe.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def is_arm_joint(name: str) -> bool:
    return name.startswith("Joint") and name.endswith(("_L", "_R"))


def soft_limits_from_hard(hard_limits: torch.Tensor, joint_names: list[str], cfg: dict) -> torch.Tensor:
    """Return [1, n, 2] soft limits, preserving hard limits for fingers."""
    limits = hard_limits.clone()
    margin = float(cfg.get("arm_soft_limit_margin_rad", 0.15))
    explicit = cfg.get("arm_soft_limits", {})
    for i, name in enumerate(joint_names):
        if not is_arm_joint(name):
            continue
        lo = float(hard_limits[0, i, 0])
        hi = float(hard_limits[0, i, 1])
        if name in explicit:
            soft_lo, soft_hi = map(float, explicit[name])
        else:
            soft_lo, soft_hi = lo + margin, hi - margin
        # User-provided degree limits are intersected with the imported hard
        # limits to avoid a rounding difference commanding beyond PhysX.
        soft_lo, soft_hi = max(lo, soft_lo), min(hi, soft_hi)
        if not (lo - 1e-6 <= soft_lo < soft_hi <= hi + 1e-6):
            raise ValueError(f"invalid soft limit for {name}: hard=({lo},{hi}), soft=({soft_lo},{soft_hi})")
        limits[0, i, 0] = soft_lo
        limits[0, i, 1] = soft_hi
    return limits


def max_target_delta(joint_names: list[str], scales: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Per-control-step target change; arm joints use a human-like rate cap."""
    out = scales.clone()
    arm_step = float(cfg.get("arm_max_step_rad", 0.04))
    for i, name in enumerate(joint_names):
        if is_arm_joint(name):
            out[:, i] = min(float(out[:, i]), arm_step)
    return out


def limit_metadata(hard_limits: torch.Tensor, soft_limits: torch.Tensor, joint_names: list[str], cfg: dict) -> dict:
    return {
        "source": "Revo3 URDF/USD articulation hard limits plus project software soft envelope",
        "arm_soft_limit_margin_rad": float(cfg.get("arm_soft_limit_margin_rad", 0.15)),
        "arm_max_step_rad": float(cfg.get("arm_max_step_rad", 0.04)),
        "arm_velocity_limit_scale": float(cfg.get("arm_velocity_limit_scale", 0.5)),
        "hard_limits_rad": {n: [float(hard_limits[0, i, 0]), float(hard_limits[0, i, 1])] for i, n in enumerate(joint_names) if is_arm_joint(n)},
        "soft_limits_rad": {n: [float(soft_limits[0, i, 0]), float(soft_limits[0, i, 1])] for i, n in enumerate(joint_names) if is_arm_joint(n)},
        "home_inside_soft_limits": all(
            float(cfg["home"][n]) >= float(soft_limits[0, i, 0]) - 1e-6
            and float(cfg["home"][n]) <= float(soft_limits[0, i, 1]) + 1e-6
            for i, n in enumerate(joint_names) if is_arm_joint(n)
        ),
    }


def load_config(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    return json.loads((root / "config.json").read_text(encoding="utf-8"))


def project_target(previous, requested, soft_limits, step_limit, velocity_limit, dt):
    """One position-target boundary shared by teachers and policies (radians/seconds)."""
    if dt <= 0 or not torch.isfinite(requested).all():
        raise ValueError('Non-finite target or invalid control period')
    if torch.any(previous < soft_limits[..., 0]-1e-6) or torch.any(previous > soft_limits[..., 1]+1e-6):
        raise ValueError('Previous command is outside the safety envelope; reset required')
    soft = torch.clamp(requested, soft_limits[..., 0], soft_limits[..., 1])
    cap = torch.minimum(step_limit, velocity_limit * dt)
    if torch.any(cap <= 0):
        raise ValueError('Motion limits must be positive')
    applied = previous + torch.clamp(soft - previous, -cap, cap)
    return applied, (requested < soft_limits[..., 0]) | (requested > soft_limits[..., 1])
