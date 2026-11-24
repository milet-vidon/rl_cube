"""Read-only per-round fitting and actor-movement evidence for the dashboard."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import io
import math
from pathlib import Path
import threading

from . import inference


# This ContactHoldPolicy has four trainable Linear layers. The other tensors in
# model.state_dict() (normalization and limits) and Adam buffers are not weights.
_LAYER_KEYS = tuple(f"net.{i}.{k}" for i in (0, 2, 4, 6) for k in ("weight", "bias"))
_LIMIT = 16 * 1024 ** 2
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cube-actor-drift")
_RESULTS = OrderedDict()
_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _actor(identity, digest):
    import numpy as np
    import torch
    raw = inference._verified(Path(identity[0]), digest, _LIMIT)
    ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    meta, state = ck["metadata"], ck["model"]
    if (meta.get("schema") != "contact_hold_success_demo_state_bc_v1" or
            digest != inference.R15_SHA and meta.get("warm_start_sha256") != inference.R15_SHA or
            set(state) != set(_LAYER_KEYS) | {"mean", "std", "lower", "upper", "center", "scale"}):
        raise ValueError("Unsupported actor")
    tensors = [state[key] for key in _LAYER_KEYS]
    if any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32 or v.layout != torch.strided
           for v in tensors) or sum(v.numel() for v in tensors) > 1_000_000:
        raise ValueError("Unsupported parameter layout")
    shapes = tuple(tuple(v.shape) for v in tensors)
    params = np.concatenate([v.detach().numpy().reshape(-1) for v in tensors]).astype(np.float64)
    if not np.isfinite(params).all():
        raise ValueError("Non-finite actor")
    return meta.get("parent_checkpoint_sha256"), shapes, params


def _relative_change(parent, child):
    import numpy as np
    _, layout, before = _actor(*parent)
    actual_parent, after_layout, after = _actor(*child)
    if actual_parent != parent[1] or layout != after_layout:
        raise ValueError("Cannot compare unrelated checkpoints")
    norm = np.sum(np.square(before))
    if norm <= 0:
        raise ValueError("Undefined relative movement for a zero-norm parent")
    return float(100 * np.sqrt(np.sum(np.square(after - before)) / norm))


def _measure(parent, child):
    try:
        return _relative_change(parent, child), "ready"
    except Exception:
        # Keep neither tensor payloads nor exception traceback frames in the
        # result cache if Torch/weights/evidence is unavailable.
        return None, "unavailable"


def _drift(parent, child):
    # Stat identities invalidate replaced weights; digests attest their content.
    # Only one CPU worker; polling never loads checkpoints in the request thread.
    try:
        key = tuple((inference._identity(path, _LIMIT), sha) for path, sha in (parent, child))
    except inference.InferenceUnavailable:
        return None, "unavailable"
    with _LOCK:
        job = _RESULTS.get(key)
        if job is None:
            if sum(not v.done() for v in _RESULTS.values()) >= 32:
                return None, "pending"
            if len(_RESULTS) >= 2048:
                old = next((k for k, v in _RESULTS.items() if v.done()), None)
                if old is not None:
                    del _RESULTS[old]
            job = _RESULTS[key] = _POOL.submit(_measure, *key)
        _RESULTS.move_to_end(key)
    if not job.done():
        return None, "pending"
    return job.result()


def _metric(data, key):
    v = data.get(key) if isinstance(data, dict) else None
    return v if type(v) in (float, int) and math.isfinite(v) and v >= 0 else None


def add_fit_trends(home, rounds, rows, load, settings):
    """Add scalar errors; compare only pairs from the dashboard's published model catalog."""
    published = {}
    if (home.parent == inference.RUNS and
            (inference.FORMAL_RUN.fullmatch(home.name) or inference.TUNABLE_RUN.fullmatch(home.name))):
        published = {str(path): card["sha256"] for card, path in inference._model_cards().values()}
    for (_, rd), row in zip(rounds, rows):
        row.update(fit_mse=None, fit_last_mse=None, fit_mae_rad=None, parameter_delta_pct=None,
                   parameter_delta_status="unavailable")
        receipt = load(rd / "fit/result.json", home)
        if receipt.get("completed") is not True:
            continue
        best = receipt.get("best_metrics")
        row.update(fit_mse=_metric(best, "phase_balanced_normalized_mse"),
                   fit_last_mse=_metric(receipt.get("last_metrics"), "phase_balanced_normalized_mse"),
                   fit_mae_rad=_metric(best, "phase_balanced_mae_rad"))
        child = rd / "fit/best.pt"
        current_sha = published.get(str(child))
        if not current_sha or receipt.get("checkpoint_sha256") != current_sha:
            continue
        plan = load(rd / "fit/fit_plan.json", home)
        parent = plan.get("parent")
        parent_sha = published.get(parent) if isinstance(parent, str) else None
        if parent_sha and plan.get("parent_sha256") == parent_sha:
            row["parameter_delta_pct"], row["parameter_delta_status"] = _drift(
                (Path(parent), parent_sha), (child, current_sha))
    if settings.get("engine") == inference.TUNABLE_ENGINE:
        return "weighted_bc"
    if inference.FORMAL_RUN.fullmatch(home.name) and settings.get("source_weights_sha256") == inference.R15_SHA:
        return "legacy_bc"
    return "unknown"
