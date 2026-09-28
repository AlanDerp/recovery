"""Private acquisition transaction audit and candidate selection for v293.

The trace is deliberately outside the student-visible API stream.  It records
where an acquisition attempt stopped without turning simulator state, poses, or
IK internals into public controller inputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V293_ACQUISITION", "0") == "1"


def audit(owner: Any, event: str, **fields: Any) -> None:
    if not enabled():
        return
    root = os.environ.get("CAPX_REPRO_AUDIT_DIR", "").strip()
    if not root:
        raise RuntimeError("v293 acquisition requires CAPX_REPRO_AUDIT_DIR")
    path = Path(root) / "libero_g17_v293_acquisition" / "trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = int(getattr(owner, "_capx_v293_event_idx", 0))
    owner._capx_v293_event_idx = idx + 1
    row = {
        "schema": "capx_libero_g17_v293_acquisition_trace_v1",
        "event_idx": idx,
        "event": str(event),
        "replicate": int(os.environ.get("REPLICATE", "0")),
        "trial": int(os.environ.get("CAPX_ACTIVE_TRIAL", "1")),
        **fields,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def cloud_summary(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = pts[np.all(np.isfinite(pts), axis=1)]
    if len(finite) == 0:
        return {"point_count": 0}
    q05, q50, q95 = np.quantile(finite, (0.05, 0.50, 0.95), axis=0)
    return {
        "point_count": int(len(finite)),
        "centroid_xyz": np.mean(finite, axis=0).round(6).tolist(),
        "median_xyz": q50.round(6).tolist(),
        "q05_xyz": q05.round(6).tolist(),
        "q95_xyz": q95.round(6).tolist(),
        "robust_extent_xyz": (q95 - q05).round(6).tolist(),
    }


def begin_attempt(api: Any, object_name: str) -> int:
    attempt = int(getattr(api, "_capx_v293_attempt", 0)) + 1
    api._capx_v293_attempt = attempt
    api._capx_v293_active_object = str(object_name)
    api._capx_v293_capture_state = "acquisition_needed"
    if (
        os.environ.get("CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION", "0")
        == "1"
    ):
        # A new perceptual binding invalidates both halves of any older
        # approach/contact transaction.  Never let a later close retreat to a
        # pregrasp that belonged to another regeneration or object instance.
        api._capx_v312_acquisition_transaction = None
        api._capx_v304_ingress = None
    audit(api, "acquisition_needed", attempt=attempt, object_name=object_name)
    return attempt


def active_attempt(api: Any) -> tuple[int | None, str | None]:
    attempt = getattr(api, "_capx_v293_attempt", None)
    object_name = getattr(api, "_capx_v293_active_object", None)
    return (int(attempt) if attempt is not None else None, object_name)


def audit_attachment(api: Any, event: str, **fields: Any) -> str | None:
    """Record the private GT attachment label as an evaluator-only witness."""
    if not enabled():
        return None
    attempt, object_name = active_attempt(api)
    row: dict[str, Any] = {
        "attempt": attempt,
        "object_name": object_name,
        **fields,
    }
    try:
        from capx.verification.libero_g17_adapter import extract_libero_snapshot

        _, _, attachment = extract_libero_snapshot(api._env)
        row["attachment"] = attachment.kind
        row["held_movable_names"] = list(attachment.held_movable_names)
    except Exception as exc:
        row["attachment_error"] = repr(exc)
    audit(api, event, **row)
    return row.get("attachment")
