"""Bounded, read-only access to the timing sidecar belonging to a sampler attempt."""
import math
import re

SCHEMA = "hold90_collection_perf_timing_v1"
NAME = re.compile(r"[A-Za-z0-9_.:>/-]{1,96}\Z")
REFILL_TICK = re.compile(r"^GPU90_STEP (\d+) active (\d+) completed (\d+) assigned (\d+) refills (\d+)\s*$")
REFILL_READY = re.compile(r"^GPU90_READY (\d+) logical_episodes (\d+) role collection\s*$")


def nonnegative(value):
    return type(value) in (int, float) and 0 <= value <= 1e15 and math.isfinite(value)


def counts(value):
    if not isinstance(value, dict) or len(value) > 256:
        return {}
    return {key: n for key, n in value.items()
            if isinstance(key, str) and len(key) <= 96 and type(n) is int and nonnegative(n)}


def snapshot(home, rounds, entry, current_log, load, within, regular):
    """Prefer the current collection attempt. In later stages use an accepted attempt."""
    stage = entry.get("stage")
    k = entry.get("active_round")
    directory = None
    if stage == "collection" and current_log is not None and type(k) is int:
        role_dir = home / f"round_{k:03d}" / "collection"
        if (current_log.suffix == ".log" and current_log.stem.startswith("sample_")
                and current_log.parent.parent == role_dir):
            directory = within(current_log.with_suffix(""), role_dir)
        scope = "current_collection"
    elif stage == "collection":
        scope = "current_collection"
    else:
        scope = "last_accepted_collection"
        for idx, rd in reversed(rounds):
            role_dir = rd / "collection"
            if within(role_dir, home) != role_dir:
                continue
            for mark in sorted(role_dir.glob("accepted_*.json"), reverse=True):
                dest = within(load(mark, home).get("dest"), role_dir)
                if dest and dest.parent.parent == role_dir and dest.name.startswith("sample_"):
                    directory, k = dest, idx
                    break
            if directory is not None:
                break
    result = {"available": False, "scope": scope, "round": k, "reason": "no_snapshot"}
    if directory is None:
        return result
    # Recheck a resolved sidecar against the worker directory (and its run).
    # This also avoids reading through an attempt directory or file pointing outside it.
    if within(directory, home) != directory:
        return result
    f = regular(directory / "perf_timing.json", directory)
    if not f:
        return result
    data = load(f, directory, 128 * 1024)
    counters = data.get("counters")
    if data.get("schema") != SCHEMA or not isinstance(counters, dict) or len(counters) > 192:
        return dict(result, reason="unreadable_snapshot")
    clean = {}
    for name, values in counters.items():
        if not NAME.fullmatch(name) or not isinstance(values, dict):
            return dict(result, reason="unreadable_snapshot")
        seconds, calls, parent = (values.get(key) for key in ("host_wall_s", "calls", "parent"))
        if (not nonnegative(seconds) or type(calls) is not int or not nonnegative(calls)
                or not (parent is None or isinstance(parent, str) and NAME.fullmatch(parent))):
            return dict(result, reason="unreadable_snapshot")
        clean[name] = {"host_wall_s": seconds, "calls": calls, "parent": parent}
    result.update(available=True, schema=SCHEMA, counters=clean)
    result.pop("reason")
    for name in ("status", "wall_time_utc"):
        value = data.get(name)
        if isinstance(value, str) and len(value) <= 96:
            result[name] = value
    result["snapshot_reason"] = [v for v in data.get("snapshot_reason", [])[:32] if isinstance(v,str) and len(v)<=96] if isinstance(data.get("snapshot_reason"),list) else []
    for name in ("elapsed_worker_s", "elapsed_rollout_s", "batch_control_steps", "rollout_batch_control_steps", "completed_episodes",
                 "assigned_episodes", "total_episodes", "physical_slots", "active_slots",
                 "episode_control_steps", "refills"):
        if nonnegative(data.get(name)):
            result[name] = data[name]
    for name in ("phase_counts", "phase_transition_counts"):
        result[name] = counts(data.get(name))
    return result


def round_metrics(home, rounds, load):
    """Expose only measured rates and scalar solver comparisons, never trajectory arrays."""
    rates = comparison = None
    rate_fields = ("effective_steps_per_worker_second", "entered_tilt_hold_per_worker_hour",
                   "wrist_endpoint_per_worker_hour")
    for k, rd in reversed(rounds):
        if rates is None:
            raw = load(rd / "collection_throughput.json", home)
            if all(nonnegative(raw.get(key)) for key in rate_fields):
                rates = dict(round=k, **{key:raw[key] for key in rate_fields})
        if comparison is None:
            raw = load(rd / "turn_path_benchmark.json", home, 4 * 1024**2)
            rows = raw.get("results")
            if isinstance(rows,list) and 0 < len(rows) <= 64 and raw.get("production_width_changed") is False:
                summary = []
                for row in rows:
                    if not isinstance(row,dict) or row.get("mode") not in ("preserve_actual_groups","cross_tick_component_only"):
                        continue
                    if row.get("status") != "measured" or type(row.get("width")) is not int or not nonnegative(row.get("median_seconds")):
                        continue
                    checks = row.get("numerics")
                    if not isinstance(checks,dict):
                        continue
                    error = checks.get("maximum_abs_delta_rad_on_common_frames")
                    exact = checks.get("all_valid_prefix_bits_and_lengths_and_accept_equal")
                    if not nonnegative(error) or type(exact) is not bool:
                        continue
                    summary.append(dict(mode=row["mode"],width=row["width"],median_seconds=row["median_seconds"],
                                        maximum_abs_delta_rad=error,bitwise_equal=exact))
                if summary:
                    comparison = dict(round=k,results=summary)
        if rates and comparison:
            break
    return {"throughput":rates,"turn_path_benchmark":comparison}
