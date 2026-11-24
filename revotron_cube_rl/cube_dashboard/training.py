"""Read-only snapshots of the native 90 mm campaigns. No Isaac, Torch or GPU calls."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import json
import math
import os
import re
import stat
from . import performance


_NAME = re.compile(r"contact_hold90_[a-zA-Z0-9_-]{1,120}\Z")
_DIAGNOSTIC = re.compile(r"(?:^|_)(?:smoke|pipeline_check|probe|test|check)(?:_|$)")
_WAVE = re.compile(r"wave_(\d+)_(\d+)(?:_slots(\d+))?$")
_TICK = re.compile(r"^GPU90_STEP (\d+) active (\d+) hold (\d+)\s*$")
_READY = re.compile(r"^GPU90_READY (\d+) role (?:validate|collection) IK accepted (\d+)")
_FIT = re.compile(r"^FIT90 (\d+) / (\d+) heldout_mse ([.\deE+-]+) best_step (\d+)")
_SETTINGS = ("rounds", "envs", "val_envs", "updates", "batch", "horizon", "lr", "replay_rounds")
_VAL = ("envs", "held", "hold_rate", "still_held_at_end", "whole_turn", "whole_turn_rate",
        "mean_controlled_hold_s", "mean_peak_height_m", "face_twist_trained")
_RES = ("ok", "gpu", "kind", "reason", "peak_device_mib", "total_mib", "elapsed_s", "exit_code",
        "min_host_available_mib", "peak_worker_rss_mib")
_LOG_BYTES = 128 * 1024


def _time(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _select(data, keys):
    return {key: data[key] for key in keys if key in data}


def _within(path, root):
    try:
        path = Path(path).resolve()
        return path if path.is_relative_to(root) else None
    except (OSError, TypeError, ValueError, RuntimeError):
        return None


def _file(path, root):
    p = _within(path, root)
    try:
        if p is not None and stat.S_ISREG(p.stat().st_mode): return p
    except OSError:
        pass
    return None


def _load(path, root, limit=96 * 1024):
    path = _file(path, root)
    if path is None: return {}
    try:
        with path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size > limit: return {}
            raw = stream.read(limit + 1)
        if len(raw) > limit: return {}
        data = json.loads(raw, parse_constant=lambda s: None)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, RecursionError):
        return {}


def _tail(path, root, limit=_LOG_BYTES):
    path = _file(path, root)
    if path is None: return [], None
    try:
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            offset = max(0, info.st_size - limit)
            stream.seek(offset)
            raw = stream.read(limit)
        lines = raw.decode("utf8", errors="replace").splitlines()
        return lines[1 if offset else 0:], _time(info.st_mtime)
    except OSError:
        return [], None


def _lock_status(file):
    """Inspect the kernel's flock table without opening/creating/locking a campaign file."""
    try:
        st = file.stat()
    except FileNotFoundError:
        return False, None
    except OSError:
        return None, None
    dev_ino = os.major(st.st_dev), os.minor(st.st_dev), st.st_ino
    try:
        with open("/proc/locks", encoding="ascii") as fp:
            for line in fp:
                bits = line.split()
                if len(bits) < 8 or bits[1:4] != ["FLOCK", "ADVISORY", "WRITE"]: continue
                maj, minor, inode = bits[5].split(":")
                if (int(maj, 16), int(minor, 16), int(inode)) == dev_ino:
                    return True, int(bits[4])
        # A restricted PID namespace cannot establish absence of a host-side flock.
        # On a desktop host (systemd/init), the /proc/locks table covers this service.
        if Path("/proc/1/comm").read_text().strip() not in ("systemd", "init"):
            return None, None
        return False, None
    except (OSError, ValueError):
        return None, None


def _live(raw, home):
    reported = raw.get("status")
    locked, pid = _lock_status(home / "campaign.lock")
    if reported == "completed": effective = "completed"
    elif locked is True: effective = "running"
    elif reported == "running": effective = "stale" if locked is False else "unknown"
    elif reported == "interrupted": effective = "interrupted"
    else: effective = "unknown"
    return effective, locked, pid


def _run_path(project, name):
    if not isinstance(name, str) or not _NAME.fullmatch(name): raise ValueError("Invalid campaign name")
    runs = (Path(project) / "runs").resolve()
    home = _within(runs / name, runs)
    if home is None or home.parent != runs or home.name != name or not home.is_dir():
        raise FileNotFoundError("No such campaign")
    return home


def _parts(home):
    settings = _load(home / "settings.json", home)
    if "envs" not in settings or "rounds" not in settings: raise FileNotFoundError("No campaign settings")
    raw = _load(home / "state.json", home)
    if settings.get("engine")=="contact_hold90_tunable_v1":
        k=raw.get("active_round") or raw.get("completed_round")
        if type(k) is int and 1 <= k <= 2000:
            card=_load(home/f"round_{k:03d}"/"control.json",home)
            # Read the parameters sealed by the trainer, never a draft/pending request.
            if card.get("round")==k and isinstance(card.get("params"),dict):
                settings.update(_select(card["params"],_SETTINGS))
        if raw.get("status")=="completed":
            # Lowering the round limit to the already active round is consumed
            # at the stop boundary: no further round receipt is created.
            head=_load(home/"control_state.json",home)
            limit=(head.get("params") or {}).get("rounds")
            if head.get("completed") is True and type(limit) is int:
                settings["rounds"]=limit
    status, held, pid = _live(raw, home)
    return settings, raw, dict(id=home.name, kind="diagnostic" if _DIAGNOSTIC.search(home.name) else "training",
        status=status, recorded_status=raw.get("status"), lock_held=held, pid=pid, stage=raw.get("stage"),
        completed_round=raw.get("completed_round",0), active_round=raw.get("active_round"),
        updated_at=raw.get("updated"), rounds=settings.get("rounds"))


def runs_catalog(project_root):
    """Active campaigns first, diagnostic fixtures last. Pure calibration/probe folders are excluded."""
    results = []
    for path in (Path(project_root) / "runs").glob("contact_hold90_*"):
        try:
            _, _, entry = _parts(_run_path(project_root, path.name))
            results.append(entry)
        except (ValueError, OSError):
            continue
    return sorted(results, key=lambda x:(x["kind"] == "training", x["status"] == "running",
                                          x.get("updated_at") or "", x["id"]), reverse=True)


def _val(home, rd):
    val = _load(rd / "validation_summary.json", home)
    return _select(val, _VAL) if val else None


def _fit_curve(lines):
    # fit.log is append-only on resume. Prefer the last emitted point for each step.
    pts = {}
    for line in lines:
        m = _FIT.match(line)
        if m:
            step, total, mse, best = m.groups()
            try: mse = float(mse)
            except ValueError: continue
            if math.isfinite(mse):
                pts[int(step)] = dict(update=int(step), total=int(total), mse=mse, best_step=int(best))
    return [pts[k] for k in sorted(pts)]


def _fit(home, rd):
    res = _load(rd / "fit/result.json", home)
    lines, _ = _tail(rd / "fit.log", home)
    curve = _fit_curve(lines)
    if not res and not curve: return None
    best = res.get("best_metrics", {})
    last = res.get("last_metrics", {})
    return dict(completed=res.get("completed",False), updates=res.get("updates",curve[-1]["update"] if curve else None),
                selected_step=res.get("selected_step"), mse=best.get("phase_balanced_normalized_mse"),
                mae_rad=best.get("phase_balanced_mae_rad"), last_mse=last.get("phase_balanced_normalized_mse"),
                last_mae_rad=last.get("phase_balanced_mae_rad"), curve=curve, phase_rows=res.get("phase_rows",{}),
                checkpoint_sha256=res.get("checkpoint_sha256"), last_sha256=res.get("last_sha256"),
                reload_verified=res.get("checkpoint_reload_verified"))


@lru_cache(maxsize=2048)
def _report_fields(path, root, mtime_ns, size):
    # Only cache the small aggregate, not the potentially large per-lane array.
    data = _load(path, root, 12 * 1024**2)
    if data.get("role") != "collection" or data.get("cube_edge_m") != .09: return {}
    metrics = data.get("metrics") or {}
    return dict(envs=data.get("num_envs"), samples=data.get("samples"), held=metrics.get("hold_count"),
                still_held_at_end=metrics.get("ending_hold_count"), returned=metrics.get("returned_to_table_count"))


def _collection(home, rd):
    if _within(rd/"collection", home) is None: return None
    counts = dict(envs=0,samples=0,held=0,still_held_at_end=0,returned=0)
    waves = 0
    elapsed = []; peaks = []
    for mark in (rd / "collection").glob("accepted_*.json"):
        raw = _load(mark, home)
        dest = _within(raw.get("dest"), (rd / "collection").resolve())
        if not dest: continue
        f = _file(dest / "report.json", home)
        if f is None: continue
        try: st = f.stat()
        except OSError: continue
        stat_row = _report_fields(f, home, st.st_mtime_ns, st.st_size)
        if not stat_row: continue
        waves += 1
        met = _load(dest.with_suffix(".resources.json"),home)
        if met.get("ok") is True:
            if isinstance(met.get("elapsed_s"), (int,float)): elapsed.append(met["elapsed_s"])
            if isinstance(met.get("peak_device_mib"), (int,float)): peaks.append(met["peak_device_mib"])
        for key in counts:
            v = stat_row.get(key)
            if isinstance(v, int): counts[key] += v
    if not waves: return None
    return dict(**counts, waves=waves, hold_rate=counts["held"]/counts["envs"] if counts["envs"] else None,
                peak_device_mib=max(peaks,default=None),elapsed_s=sum(elapsed) if elapsed else None)


def _checkpoint(home, path, kind, idx, sha, base=None):
    runs = home.parent
    # Include the known R15 source and this campaign only; never expose arbitrary external paths.
    scope = base if base is not None else home
    ck = _file(path, scope)
    if ck is None or ck.name not in ("best.pt","last.pt"): return None
    try: size=ck.stat().st_size
    except OSError: return None
    return dict(id=f"{home.name}:{kind}:{idx or 0}", path=str(ck.relative_to(runs.parent)), round=idx,
                kind=kind, sha256=sha, size=size)


def _last_telemetry(home, rounds):
    """A finished sampling process's resources, explicitly distinguished from live GPU usage."""
    for _,rd in reversed(rounds):
        most_recent = None
        for role in ("collection", "validate"):
            if _within(rd/role, home) is None: continue
            for f in (rd/role).glob("wave_*/*.resources.json"):
                f = _file(f,home)
                if not f: continue
                try: mt=f.stat().st_mtime
                except OSError: continue
                if most_recent is None or mt>most_recent[0]: most_recent=(mt,role,f)
        if most_recent:
            mt,role,p=most_recent; data=_load(p,home)
            if data: return dict(**_select(data,_RES),role=role,round=int(rd.name[6:]),updated_at=_time(mt),
                                 path=str(p.relative_to(home.parent.parent)))
    return None


def _selected_run(project_root, run_name):
    if run_name is None:
        entries = runs_catalog(project_root)
        if not entries: raise FileNotFoundError("No 90 mm campaigns found")
        run_name = entries[0]["id"]
    return _run_path(project_root, run_name)


def _round_dirs(home):
    rounds = []
    for d in home.glob("round_*"):
        if re.fullmatch(r"round_\d{3,4}", d.name) and _within(d, home) == d and d.is_dir():
            rounds.append((int(d.name[6:]), d))
    return sorted(rounds)


def quality_snapshot(project_root, run_name=None):
    from .quality import quality_history
    from .convergence import add_fit_trends
    home = _selected_run(project_root, run_name)
    settings, _, _ = _parts(home)
    dirs = _round_dirs(home)
    rows = quality_history(home, dirs)
    mode = add_fit_trends(home, dirs, rows, _load, settings)
    return {"id": home.name, "objective_mode": mode, "rounds": rows}


def comparison_snapshot(project_root, run_name, base, target):
    from .quality import paired_validation
    if not run_name:
        raise ValueError("请选择实验。")
    if not all(isinstance(v, str) and re.fullmatch(r"[1-9][0-9]{0,3}", v) for v in (base, target)):
        raise ValueError("请选择两个有效轮次。")
    return paired_validation(_run_path(project_root, run_name), int(base), int(target))


def training_snapshot(project_root, run_name=None):
    """Return JSON-compatible data without starting tools or changing the training files."""
    home = _selected_run(project_root, run_name)
    settings, raw, entry = _parts(home)
    current_log = _within(raw.get("log"),home)
    if current_log is not None and current_log.suffix != ".log": current_log=None
    lines, log_time = _tail(current_log,home)
    current_exists = log_time is not None
    if not current_exists:
        lines, log_time = _tail(home / "console.log",home)
    progress = {"envs": raw.get("current_envs"), "horizon": settings.get("horizon")}
    if type(raw.get("current_episodes")) is int:
        progress["total_episodes"] = raw["current_episodes"]
    if current_log and (w:=_WAVE.fullmatch(current_log.parent.name)):
        progress["offset"] = int(w[1])
        progress["envs"] = int(w[3] or w[2])
        if w[3]:
            progress["total_episodes"] = int(w[2])
    timeline = []
    # Fallback console.log spans runs/stages; never parse it as the current wave.
    if current_exists:
        for x in lines:
            if m:=performance.REFILL_TICK.match(x):
                tick = dict(zip(("step","active","completed","assigned","refills"), map(int, m.groups())))
                timeline.append(tick);progress.update(tick)
            elif m:=performance.REFILL_READY.match(x):
                progress.update(envs=int(m[1]),total_episodes=int(m[2]))
            elif m:=_TICK.match(x):
                tick=dict(step=int(m[1]),active=int(m[2]),hold=int(m[3]))
                timeline.append(tick);progress.update(tick)
            elif m:=_READY.match(x):
                progress.update(envs=int(m[1]),ik_accepted=int(m[2]))
        if entry["stage"]=="fit":
            curve=_fit_curve(lines)
            if curve: progress.update(update=curve[-1]["update"],update_total=curve[-1]["total"],mse=curve[-1]["mse"])
    rounds = _round_dirs(home)
    history, checkpoints = [], []
    source_fit = _within(settings.get("source_fit"),home.parent)
    if source_fit and source_fit==home.parent / "contact_hold_cuda_long_v1/round_015/fit":
        initial = _checkpoint(home,source_fit/"best.pt","warm_start",15,settings.get("source_weights_sha256"),base=source_fit)
        if initial:checkpoints.append(initial)
    for idx, rd in rounds:
        val = _val(home,rd)
        fit = _fit(home,rd)
        candidate = None
        if fit and fit["completed"]:
            candidate = _checkpoint(home,rd/"fit/best.pt","candidate",idx,fit["checkpoint_sha256"])
            if candidate:
                candidate["best"] = idx==raw.get("best_round")
                checkpoints.append(candidate)
        saved_last = _file(rd/"fit/last.pt",home)
        history.append(dict(round=idx,completed=bool(val),validation=val,collection=_collection(home,rd),fit=fit,
                            checkpoint=candidate,fit_resume_saved=(saved_last is not None)))
    concise=[]
    for x in lines:
        # The training log may include huge, one-line dumps; preserve meaningful milestones, warnings and errors.
        if not x.strip():continue
        concise.append(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]","",x)[:360])
    latest=next((row["validation"] for row in reversed(history) if row["validation"] is not None),None)
    return dict(**entry,settings=_select(settings,_SETTINGS),best_round=raw.get("best_round"),progress=progress,
                performance={**performance.snapshot(home,rounds,entry,current_log,_load,_within,_file),
                             **performance.round_metrics(home,rounds,_load)},
                telemetry=_last_telemetry(home,rounds), timeline=timeline, history=history, validation=latest,
                checkpoints=checkpoints, log=concise[-35:], log_updated_at=log_time, last_error=raw.get("last_error"),
                log_path=str(current_log.relative_to(home.parent.parent)) if current_exists else None)
