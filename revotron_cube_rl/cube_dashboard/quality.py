"""Closed-loop outcomes from a campaign's validation summaries and accepted waves."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import NamedTuple


def _json(path, limit, digest=None):
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Expected a regular evidence file")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit or (digest is not None and hashlib.sha256(data).hexdigest() != digest):
        raise ValueError("Invalid evidence size or SHA256")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _safe(path, root):
    # Accepted destinations are absolute paths produced by the campaign.
    if not path.is_absolute() or path.resolve() != path or not path.is_relative_to(root):
        raise ValueError("Evidence outside the selected campaign")
    return path


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid duration")
    return value


def _summary(path):
    data = _json(path, 96 * 1024)
    n = data["envs"]
    counts = (data["held"], data["still_held_at_end"], data["whole_turn"])
    if type(n) is not int or n <= 0 or any(type(v) is not int or not 0 <= v <= n for v in counts):
        raise ValueError("Invalid validation counts")
    return dict(envs=n, held=counts[0], end_held=counts[1], whole_turn=counts[2],
                mean_hold_s=_seconds(data["mean_controlled_hold_s"]))


# Compare measured protocol fields, not campaign defaults: e.g. a tunable run's
# evaluation horizon can change. The model digest is intentionally different between rounds.
_HASH_FIELDS = ("scene_sha256", "scene_manifest_sha256", "grasp_sha256", "asset_sha256")
_NUM_FIELDS = ("requested_steps", "control_hz", "physics_hz", "action_dim")
_OTHER_FIELDS = ("solver", "physical_hold_definition", "code_sha256", "physics_device")
_PROTOCOL_FIELDS = (*_HASH_FIELDS, *_NUM_FIELDS, *_OTHER_FIELDS, "student_start_phase")


def _protocol(data):
    result = []
    for name in _HASH_FIELDS:
        v = data.get(name)
        result.append(v if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) else None)
    for name in _NUM_FIELDS:
        v = data.get(name)
        result.append(v if type(v) in (int,float) and v > 0 and math.isfinite(v) else None)
    for name in _OTHER_FIELDS:
        v = data.get(name)
        expected = dict if name in ("solver", "code_sha256") else str
        if not isinstance(v, expected) or not v:
            result.append(None)
        else:
            # Cache a small fingerprint even when the report has a long code manifest.
            try:
                canonical = json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)
                result.append(hashlib.sha256(canonical.encode()).hexdigest())
            except (ValueError, RecursionError):
                result.append(None)
    return tuple(result)


class _Lane(NamedTuple):
    seed: int
    ever_hold: bool
    end_held: bool
    whole_turn: bool
    reason: str
    hold_s: float
    protocol: tuple
    checkpoint: str | None


@lru_cache(maxsize=256)
def _wave(report_path, file_signature, report_sha, plan_sha, width):
    """Cache only per-lane outcome fields. The stat signature invalidates replaced reports."""
    data = _json(report_path, 12 * 1024**2, report_sha)
    lanes = data["per_lane"]
    if (data.get("role") != "validate" or data.get("cube_edge_m") != .09 or
            data.get("plan_sha256") != plan_sha or data.get("num_envs") != width or
            not isinstance(lanes, list) or len(lanes) != width):
        raise ValueError("Unexpected validation report")
    results = []
    protocol = _protocol(data)
    checkpoint = data.get("checkpoint")
    if not isinstance(checkpoint, str) or checkpoint != data.get("parent_sha") or not re.fullmatch(r"[0-9a-f]{64}", checkpoint):
        checkpoint = None
    start_phases = data.get("student_start_phase")
    if not isinstance(start_phases, list) or len(start_phases) != width:
        start_phases = [None] * width
    for lane, phase in zip(lanes, start_phases):
        seed = lane["seed"]
        flags = (lane["ever_hold"], lane["ending_hold"], lane["ever_whole_turn"])
        if (type(seed) is not int or any(type(flag) is not bool for flag in flags)
                or lane["teacher_executed_left_steps"] != 0):
            raise ValueError("Invalid student validation lane")
        reason = lane["end_reason"].split(":", 1)[0].strip()
        if not reason or len(reason) > 128:
            raise ValueError("Invalid termination reason")
        rotation = lane.get("physical_rotation") or {}
        seconds = _seconds(rotation.get("best_controlled_hold", {}).get("seconds", 0.0))
        results.append(_Lane(seed, *flags, reason, seconds, (*protocol, phase if type(phase) is int and phase >= 0 else None), checkpoint))
    if [row[0] for row in results] != data["seeds"]:
        raise ValueError("Lane seeds differ from report seeds")
    return tuple(results)


def _verified_lanes(rd, row):
    role = _safe(rd / "validate", rd)
    marks = sorted(role.glob("accepted_*.json"))
    if not marks:
        return None
    lanes = []
    for path in marks:
        offset = len(lanes)
        receipt = _json(_safe(path, role), 96 * 1024)
        width = receipt["width"]
        if (path.name != f"accepted_{offset:04d}.json" or
                type(width) is not int or not 0 < width <= row["envs"] - offset):
            raise ValueError("Validation waves do not cover distinct consecutive offsets")
        wave = role / f"wave_{offset:04d}_{width:04d}"
        dest = _safe(Path(receipt["dest"]), wave)
        if dest.parent != wave or not dest.name.startswith("sample_"):
            raise ValueError("Receipt does not point to this validation wave")
        report = _safe(dest / "report.json", wave)
        info = report.stat()
        signature = (info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_size)
        report_sha, plan_sha = receipt["report_sha"], receipt["plan_sha"]
        if not all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in (report_sha, plan_sha)):
            raise ValueError("Invalid validation receipt digests")
        lanes.extend(_wave(report, signature, report_sha, plan_sha, width))
    if len(lanes) != row["envs"]:
        return None
    if len({lane[0] for lane in lanes}) != len(lanes):
        raise ValueError("Validation seeds overlap")
    if (sum(lane[1] for lane in lanes) != row["held"] or
            sum(lane[2] for lane in lanes) != row["end_held"] or
            sum(lane[3] for lane in lanes) != row["whole_turn"] or
            not math.isclose(sum(lane[5] for lane in lanes) / len(lanes), row["mean_hold_s"],
                             rel_tol=1e-9, abs_tol=1e-9)):
        raise ValueError("Validation summary does not match accepted lanes")
    return lanes


def _detail(lanes):
    reasons = Counter(lane[4] for lane in lanes)
    endpoint_held = sum(lane[4] == "teacher_wrist_endpoint" and lane[2] for lane in lanes)
    joint_pass = sum(lane[2] and lane[3] and lane[4] == "teacher_wrist_endpoint" for lane in lanes)
    return dict(detail_status="verified", verified_envs=len(lanes), endpoint_held=endpoint_held, joint_pass=joint_pass,
                reason_counts=dict(sorted(reasons.items())))


_READ_ERRORS = (ValueError, OSError, TypeError, KeyError, AttributeError, OverflowError,
                RecursionError, RuntimeError)


def _read_round(home, number, directory):
    row = dict(round=number, envs=None, held=None, end_held=None, whole_turn=None,
               endpoint_held=None, joint_pass=None, mean_hold_s=None, reason_counts=None,
               verified_envs=0, detail_status="missing_summary")
    try:
        rd = _safe(Path(directory), home)
        if rd.parent != home or rd.name != f"round_{number:03d}":
            raise ValueError("Unexpected round directory")
        summary = _safe(rd / "validation_summary.json", rd)
        if not summary.exists():
            return row, None
        row.update(_summary(summary))
    except _READ_ERRORS:
        row["detail_status"] = "invalid_summary"
        return row, None
    try:
        lanes = _verified_lanes(rd, row)
        if lanes is not None:
            row.update(_detail(lanes))
            return row, lanes
        row["detail_status"] = "missing_receipt"
    except _READ_ERRORS:
        row["detail_status"] = "invalid_evidence"
    return row, None


def quality_history(home, rounds):
    home = Path(home).resolve()
    return [_read_round(home, number, directory)[0] for number, directory in rounds]


_PAIR_METRICS = ("joint_pass", "end_held", "whole_turn", "endpoint_held")


def _observation(lane):
    endpoint = lane.end_held and lane.reason == "teacher_wrist_endpoint"
    return dict(end_held=lane.end_held, whole_turn=lane.whole_turn, endpoint_held=endpoint,
                joint_pass=endpoint and lane.whole_turn, hold_s=lane.hold_s, reason=lane.reason)


def _read_model_panel(home, number):
    rd = home / f"round_{number:03d}"
    row, lanes = _read_round(home, number, rd)
    if lanes is not None:
        try:
            fit = _json(_safe(rd / "fit/result.json", rd), 96 * 1024)
            digest = fit["checkpoint_sha256"]
            if (fit.get("completed") is not True or fit.get("checkpoint_reload_verified") is not True
                    or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or any(lane.checkpoint != digest for lane in lanes)):
                raise ValueError("Accepted panel was not attributed to this round's published actor")
        except _READ_ERRORS:
            row["detail_status"] = "unverified_model"
            return row, None
    return row, lanes


def paired_validation(home, base, target):
    """Compare two accepted panels by seed. Missing or unpaired evidence is never a failure."""
    if (any(type(n) is not int or not 1 <= n <= 9999 for n in (base, target))
            or base == target):
        raise ValueError("请选择两个不同且有效的轮次。")
    home = Path(home).resolve()
    left, a = _read_model_panel(home, base)
    right, b = _read_model_panel(home, target)
    result = dict(id=home.name, base_round=base, target_round=target, status="unavailable",
                  base_status=left["detail_status"], target_status=right["detail_status"],
                  base_envs=left["envs"], target_envs=right["envs"], matched=None,
                  base_only=None, target_only=None, protocol_fields=[], metrics={}, rows=[])
    if a is None or b is None:
        return result
    before = {lane[0]: lane for lane in a}
    after = {lane[0]: lane for lane in b}
    common = sorted(before.keys() & after.keys())
    result.update(matched=len(common), base_only=len(before)-len(common),
                  target_only=len(after)-len(common))
    if not common:
        result["status"] = "no_overlap"
        return result
    different, unknown = set(), set()
    for seed in common:
        for field, x, y in zip(_PROTOCOL_FIELDS, before[seed].protocol, after[seed].protocol):
            if x is None or y is None:
                unknown.add(field)
            elif x != y:
                different.add(field)
    if different or unknown:
        return dict(result, status="protocol_mismatch" if different else "protocol_unknown",
                    protocol_fields=sorted(different | unknown))
    pairs = [dict(seed=seed, base=_observation(before[seed]), target=_observation(after[seed]))
             for seed in common]
    metrics = {}
    for key in _PAIR_METRICS:
        counts = Counter((r["base"][key], r["target"][key]) for r in pairs)
        stayed, gained, lost, missed = (counts[(True, True)], counts[(False, True)],
                                         counts[(True, False)], counts[(False, False)])
        metrics[key] = dict(gained=gained, lost=lost, both_pass=stayed, both_fail=missed,
                            base_pass=stayed+lost, target_pass=stayed+gained)
    return dict(result, status="ready", rows=pairs, metrics=metrics)
