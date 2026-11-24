"""Read-only playback of attested physics episodes and CPU policy inference on those inputs.

Only local, allowlisted models and fixed recorded trajectories are addressable. Nothing here
launches Isaac, sends a command to a robot or reuses a replay as a new rollout.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import threading
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ARCHIVE = "contact_hold_size_r15_90mm_retarget_approach_20261112_084150_canonical_run"
REPORT_SHA = "fc5dfb1eac2570a49f523251c351b31fa61d6977340185ce0909e801a7d1ce3c"
R15_SHA = "13953a5c8bf4522fa9735b5807b082ae51cdfaac2044c80c30ee2f0839d116c2"
SEEDS = (21611, 21612)
ARCHIVE_MODEL = "r15"
TUNABLE_RUN = re.compile(r"contact_hold90_ui_[A-Za-z0-9_-]{1,100}\Z")
TUNABLE_ENGINE = "contact_hold90_tunable_v1"
FORMAL_RUN = re.compile(r"contact_hold90_long_v[0-9]+\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_IO_LOCK = threading.RLock()
_POLICY_LOCK = threading.Lock()


class InferenceNotFound(KeyError):
    """An ID was not one of the published episodes or models."""


class InferenceUnavailable(RuntimeError):
    """Missing, incomplete, changed, or incompatible local evidence."""


def _identity(path, limit):
    """Deny links (including intermediate directory links) and unbounded inputs."""
    try:
        path.relative_to(RUNS)
        if path.resolve() != path or not path.is_file():
            raise InferenceUnavailable("缺少可信的本地文件：" + path.name)
        st = path.stat()
        if not 0 < st.st_size <= limit:
            raise InferenceUnavailable("文件大小超出预期：" + path.name)
        return str(path), st.st_size, st.st_ino, st.st_mtime_ns, st.st_ctime_ns
    except (ValueError, OSError, RuntimeError) as err:
        if isinstance(err, InferenceUnavailable):
            raise
        raise InferenceUnavailable("无法读取本地文件：" + path.name) from err


def _read_json(path, limit=2_000_000):
    _identity(path, limit)
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError) as err:
        raise InferenceUnavailable("无法读取归档：" + path.name) from err


def _verified(path, expected, limit):
    _identity(path, limit)
    if not isinstance(expected, str) or not _DIGEST.fullmatch(expected):
        raise InferenceUnavailable("缺少来源哈希：" + path.name)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise InferenceUnavailable("来源校验失败：" + path.name)
    return raw


@lru_cache(maxsize=2)
def _report(identity):
    p = Path(identity[0])
    data = json.loads(_verified(p, REPORT_SHA, 1_000_000))
    if data.get("weights_sha256") != R15_SHA:
        raise InferenceUnavailable("归档模型和第 15 轮权重不匹配")
    return data


def _episode_cards():
    p = RUNS / ARCHIVE / "retarget_report.json"
    if not p.exists():
        return {}
    report = _report(_identity(p, 1_000_000))
    out = {}
    for seed in SEEDS:
        info = report["episodes"][str(seed)]["retargeted"]
        d = p.parent / "from_reset" / f"seed_{seed}"
        if not (d / "trajectory.jsonl").is_file():
            continue
        key = f"r15-canonical-{seed}"
        out[key] = ({"id": key, "label": f"90 mm · Seed {seed}", "seed": seed, "step_count": info["steps"],
                     "hz": 30, "edge_mm": 90, "origin_model_id": ARCHIVE_MODEL,
                     "controller": info["controller"], "source": str(d.relative_to(ROOT))}, d, info)
    return out


def _model_cards():
    out = {}
    def add(mid, run, index, expected, label):
        d = RUNS / run / f"round_{index:03d}" / "fit"
        try:
            receipt = _read_json(d / "result.json")
            digest = receipt["checkpoint_sha256"]
            if (not _DIGEST.fullmatch(digest) or receipt.get("completed") is not True
                    or receipt.get("checkpoint_reload_verified") is not True or (expected and digest != expected)):
                return
            _identity(d / "best.pt", 64 * 1024 ** 2)
        except (KeyError, TypeError, InferenceUnavailable):
            return
        out[mid] = ({"id": mid, "label": label, "round": index, "sha256": digest, "run": run}, d / "best.pt")

    add(ARCHIVE_MODEL, "contact_hold_cuda_long_v1", 15, R15_SHA, "第 15 轮 · 冻结基线")
    def collect(home, display):
        try:
            finished = _read_json(home / "state.json")["completed_round"]
            if type(finished) != int or not 0 <= finished <= 2000: return
        except (KeyError, TypeError, InferenceUnavailable):
            return
        # A fit result alone is not a completed physics round.
        for k in range(1, finished + 1):
            add(f"{home.name}-r{k}", home.name, k, None, f"90 mm · {display} · 第 {k} 轮")

    candidates = []
    for home in sorted(RUNS.glob("contact_hold90_*")):
        if home.is_symlink() or not (FORMAL_RUN.fullmatch(home.name) or TUNABLE_RUN.fullmatch(home.name)): continue
        try: settings = _read_json(home / "settings.json")
        except InferenceUnavailable: continue
        if not isinstance(settings,dict): continue
        if FORMAL_RUN.fullmatch(home.name):
            if settings.get("source_weights_sha256") == R15_SHA: collect(home,home.name.split('_')[-1])
        elif settings.get("engine") == TUNABLE_ENGINE and settings.get("warm_start_sha256") == R15_SHA:
            candidates.append((home,settings))
    # Accept forks only after the exact direct parent fit path and published digest
    # are known. A same-named external path, orphan, or a cycle cannot seed a fork.
    while candidates:
        known={(str(path.parent),card["sha256"]) for card,path in out.values()}
        accepted=[]
        for home,settings in candidates:
            pair=(settings.get("source_fit"),settings.get("source_weights_sha256"))
            if all(isinstance(v,str) for v in pair) and pair in known:
                collect(home,"可调 · "+home.name.removeprefix("contact_hold90_ui_"))
                accepted.append(home)
        if not accepted: break
        candidates=[(h,c) for h,c in candidates if h not in accepted]
    return out


def catalog():
    episodes = [v[0] for v in _episode_cards().values()]
    models = [v[0] for v in _model_cards().values()]
    return {"episodes": episodes, "models": models, "default_episode": episodes[0]["id"] if episodes else None,
            "default_model": models[0]["id"] if models else None}


def _vec(x, size):
    if not isinstance(x, list) or len(x) != size or not all(type(i) in (float, int) and math.isfinite(i) for i in x):
        raise InferenceUnavailable("归档里有无效的位姿或关节向量")
    return x


def _compact_frame(row, robot_z):
    hands, elbow = row["measured_hand_frames"], row["measured_elbow"]
    def world(v):
        x, y, z = _vec(v, 3)
        return [x, y, z + robot_z]
    def origin(k):
        return world([hands[k][i][3] for i in range(3)])
    impulses = _vec(row["contact_impulse"], 2)
    # One actual strongest contact per finger keeps the 3-D playback compact.
    # Do not synthesize fingertip positions during non-contact frames.
    strongest = {}
    for c in row["finger_contact_points"]:
        key = c["side"], c["finger"]
        if key not in strongest or strongest[key]["impulse_Ns"] < c["impulse_Ns"]:
            strongest[key] = c
    contacts = [{"side": side, "finger": finger, "xyz": _vec(c["position_world_m"], 3),
                 "impulse_Ns": c["impulse_Ns"]} for (side, finger), c in strongest.items()]
    return {"step": row["step"], "phase": row["phase"], "height_m": row["height_m"],
            "cube": _vec(row["cube_position"], 3), "quat": _vec(row["cube_quaternion_wxyz"], 4),
            "left_palm": origin("left_palm_link"), "right_palm": origin("right_palm_link"),
            "left_elbow": origin("Link4_L"), "right_elbow": origin("Link4_R"),
            "left_wrist": world(elbow["wrist_root_m"]), "left_shoulder": world(elbow["shoulder_root_m"]),
            "left_impulse": impulses[0], "right_impulse": impulses[1], "table_contact": row["cube_table_contact"],
            "face_angle_deg": row["layer_measurement"]["face_angle_deg"], "finger_contacts": contacts}


def _observation(row):
    return {**{k: _vec(row[k], n) for k, n in (("q",56), ("joint_velocity",56), ("applied_target",56),
                                                 ("cube_position",3), ("cube_quaternion_wxyz",4))},
            "finger_contact_points": [{k: p[k] for k in ("side","finger","impulse_Ns")}
                                      for p in row["finger_contact_points"]]}


def _episode_record(episode):
    if not isinstance(episode, str) or episode not in (t := _episode_cards()):
        raise InferenceNotFound("未知的回放轨迹")
    card, d, info = t[episode]
    # Key by identities so replaced artifacts cannot silently inherit a verified cached object.
    limits = {"artifact_manifest.json": 1_000_000, "config.json": 2_000_000, "trajectory.jsonl": 64*1024**2,
              "result.json": 2_000_000, "canonical_initial_observation.json": 1_000_000}
    key = tuple(_identity(d / n, lim) for n, lim in limits.items())
    with _IO_LOCK:
        return _load_episode(episode, key, info["manifest_sha256"])


@lru_cache(maxsize=2)
def _load_episode(episode, key, manifest_sha):
    card, d, info = _episode_cards()[episode]
    manifest = json.loads(_verified(d / "artifact_manifest.json", manifest_sha, 1_000_000))
    files = manifest["sha256"]
    if (not manifest["full_episode"] or manifest["seed"] != card["seed"] or files.get("evaluated_best.pt") != R15_SHA):
        raise InferenceUnavailable("未验证的轨迹或模型")
    def checked(n):
        return json.loads(_verified(d / n, files.get(n), 2_000_000))
    cfg = checked("config.json"); result = checked("result.json")
    if (result.get("checkpoint_sha256") != R15_SHA or result.get("steps") != card["step_count"]
            or result.get("execution_interrupted") is not False or result.get("failure") != info["end_reason"]):
        raise InferenceUnavailable("轨迹与来源报告不一致")
    previous = _observation(checked("canonical_initial_observation.json"))
    names = cfg["joint_names"]; left = [i for i,k in enumerate(names) if k.endswith("_L") or k.startswith("left_")]
    if len(names)!=56 or len(left)!=28 or cfg['control_hz'] != card['hz']:
        raise InferenceUnavailable("关节顺序或记录频率不匹配")
    frames, inputs = [], []
    raw = _verified(d / "trajectory.jsonl", files.get("trajectory.jsonl"), 64*1024**2)
    for i, line in enumerate(io.BytesIO(raw)):
        row = json.loads(line)
        if i != row["step"]:
            raise InferenceUnavailable("轨迹缺帧")
        frames.append(_compact_frame(row, cfg["robot_z"]))
        # Archive rows are post-action. Predict step i from the preceding measured state, not from row i.
        inputs.append((previous, row["phase_elapsed"], _vec(row["student_raw_left_target"],28),
                       [row["applied_target"][j] for j in left]))
        previous = _observation(row)
    if len(frames) != card["step_count"]:
        raise InferenceUnavailable("轨迹长度不匹配")
    h = info["hold"]
    summary = {"peak_height_m": info["peak_height_m"], "best_hold_s": h["best_controlled_hold"]["seconds"],
               "end_reason": info["end_reason"], "whole_turn": h["ever_hold_and_turn"],
               "face_turn": info["single_face"]["qualified"], "final_table_contact": info["final_table_contact"]}
    return {"episode": card, "table_z_m": cfg["table_z"], "summary": summary, "frames": frames,
            "checkpoint_sha256": R15_SHA, "trajectory_sha256": files["trajectory.jsonl"]}, inputs, names, left


def trajectory(episode):
    return _episode_record(episode)[0]


@lru_cache(maxsize=3)
def _load_model(identity, digest):
    try:
        import torch
        from contact_hold_policy import restored
    except ImportError as exc:
        raise InferenceUnavailable("CPU 推理需要项目的 rl_cube conda 环境") from exc
    contents = _verified(Path(identity[0]), digest, 64*1024**2)
    # Bounded model path, local digest, CPU-only deserialization and the restricted weights loader.
    ck = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    meta = ck["metadata"]
    if digest != R15_SHA and meta.get("warm_start_sha256") != R15_SHA:
        raise InferenceUnavailable("模型不属于同一训练谱系")
    for n, sha in meta["source_sha256"].items():
        if Path(n).name != n or not _DIGEST.fullmatch(sha) or (ROOT/n).is_symlink():
            raise InferenceUnavailable("非法模型来源")
        if not (ROOT/n).is_file() or hashlib.sha256((ROOT/n).read_bytes()).hexdigest() != sha:
            raise InferenceUnavailable("模型要求的源码已变更：" + n)
    torch.set_num_threads(1)
    return restored(ck).cpu().eval(), meta["joint_names"], meta["output_indices"]


def predict(model, episode, step):
    if type(step) != int or step < 0:
        raise ValueError("step 必须是有效的非负整数")
    if not isinstance(model, str) or model not in (m := _model_cards()):
        raise InferenceNotFound("未知或未完成的模型")
    ep, rows, names, ids = _episode_record(episode)
    if step >= len(rows): raise ValueError("step 超出该轨迹的范围")
    spec, path = m[model]
    with _POLICY_LOCK:
        network, ck_names, ck_ids = _load_model(_identity(path,64*1024**2), spec["sha256"])
        if ck_names != names or ck_ids != ids:
            raise InferenceUnavailable("模型关节顺序与归档不符")
        import torch
        from phase_contact_policy import features
        observation, clock, reference, applied = rows[step]
        phase = ep["frames"][step]["phase"]
        x = features(observation, phase, clock)
        with torch.inference_mode():
            prediction = network(torch.from_numpy(x).unsqueeze(0))[0].tolist()
    if not all(math.isfinite(v) for v in prediction):
        raise InferenceUnavailable("非有限的网络输出")
    differences = [abs(a-b) for a,b in zip(prediction,reference)]
    return {"model_id": model, "episode_id": episode, "step": step, "phase": phase, "device": "cpu", "unit":"rad",
            "names":[names[i] for i in ids], "predicted": prediction, "reference": reference, "applied": applied,
            "mae_rad": sum(differences)/len(differences), "max_abs_error_rad": max(differences),
            "checkpoint_sha256": spec["sha256"], "input_frame": "reset" if step == 0 else step-1,
            "input_kind":"physical_pre_action_198", "reference_kind":"archived_unretargeted_raw_r15",
            "applied_kind":"archived_projected_controller_target", "physics_stepped":False}
