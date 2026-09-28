"""Freeze GraspNet outputs from a donor arm (v256) and replay by call index.

Enabled by ``CAPX_LIBERO_G17_GRASP_REPLAY=/path/to/grasp_replay_v1.json``.

Lookup key: ``(REPLICATE, CAPX_ACTIVE_TRIAL, call_idx)``.

Overflow (``CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW``):
* ``miss`` — raise ``GRASP_REPLAY_MISS`` when strict, else fall through to live GraspNet
* ``repeat_last`` — reuse last frozen pose for that trial (fair wash; never invent live)
* ``live`` — fall through to live GraspNet (non-strict legacy)

Default overflow: ``miss`` when strict else ``live``.
Fair v264 uses ``repeat_last`` so extra grasps do not invalidate the cell.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "capx_libero_g17_grasp_replay_v1"
ENV_PATH = "CAPX_LIBERO_G17_GRASP_REPLAY"
ENV_STRICT = "CAPX_LIBERO_G17_GRASP_REPLAY_STRICT"
ENV_OVERFLOW = "CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW"
ENV_VERBOSE = "CAPX_LIBERO_G17_GRASP_REPLAY_VERBOSE"

_PACKAGE: dict[str, Any] | None = None
_CURSOR: dict[tuple[int, int], int] = {}


@dataclass(frozen=True)
class GraspPose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    object_name: str
    call_idx: int


def enabled() -> bool:
    return bool(os.environ.get(ENV_PATH, "").strip())


def strict() -> bool:
    return os.environ.get(ENV_STRICT, "1") == "1"


def overflow_mode() -> str:
    raw = (os.environ.get(ENV_OVERFLOW) or "").strip().lower()
    if raw in ("miss", "repeat_last", "live"):
        return raw
    return "miss" if strict() else "live"


def verbose() -> bool:
    """Student-visible stdout for hit lines (default off — avoid LLM prompt pollution)."""
    return os.environ.get(ENV_VERBOSE, "0") == "1"


def package_path() -> Path:
    raw = os.environ.get(ENV_PATH, "").strip()
    if not raw:
        raise RuntimeError(f"{ENV_PATH} is not set")
    return Path(raw)


def _load() -> dict[str, Any]:
    global _PACKAGE
    if _PACKAGE is not None:
        return _PACKAGE
    path = package_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"unsupported grasp-replay schema {raw.get('schema')!r}")
    _PACKAGE = raw
    return _PACKAGE


def active_replicate() -> int:
    return int(os.environ.get("REPLICATE", "0"))


def active_trial() -> int:
    # runner.py sets CAPX_ACTIVE_TRIAL; fall back to 1-based trial env if present.
    for key in ("CAPX_ACTIVE_TRIAL", "CAPX_CURRENT_TRIAL", "TRIAL"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return int(raw)
    return 1


def reset_cursor(*, replicate: int | None = None, trial: int | None = None) -> None:
    """Optional: clear call index (normally auto-advances per trial)."""
    if replicate is None and trial is None:
        _CURSOR.clear()
        return
    rep = active_replicate() if replicate is None else int(replicate)
    tr = active_trial() if trial is None else int(trial)
    _CURSOR.pop((rep, tr), None)


def _pose_from_row(
    row: dict[str, Any], *, call_idx: int, object_name: str | None
) -> GraspPose:
    pos = np.asarray(row["position"], dtype=np.float64).reshape(3)
    quat = np.asarray(row["quaternion_wxyz"], dtype=np.float64).reshape(4)
    obj = str(row.get("object") or object_name or "")
    return GraspPose(
        position=pos,
        quaternion_wxyz=quat,
        object_name=obj,
        call_idx=call_idx,
    )


def _miss(msg: str) -> GraspPose | None:
    """Hard miss (missing rep / empty trial). Only ``live`` may fall through."""
    if overflow_mode() == "live":
        print(f"[grasp-replay] {msg}; live GraspNet", flush=True)
        return None
    raise RuntimeError(msg)


def next_pose(*, object_name: str | None = None) -> GraspPose | None:
    """Return next frozen grasp for (replicate, trial).

    On exhaustion: see ``overflow_mode()``. Returns None only for ``live`` fallthrough.
    """
    pkg = _load()
    rep = active_replicate()
    trial = active_trial()
    key = (rep, trial)
    idx = int(_CURSOR.get(key, 0))
    rep_entry = (pkg.get("reps") or {}).get(str(rep))
    if rep_entry is None:
        return _miss(f"GRASP_REPLAY_MISS: no replicate {rep} in {package_path()}")
    calls = (rep_entry.get("trials") or {}).get(str(trial)) or []
    if not calls:
        return _miss(
            f"GRASP_REPLAY_MISS: rep={rep} trial={trial} has zero frozen grasps"
        )
    if idx >= len(calls):
        mode = overflow_mode()
        if mode == "repeat_last":
            row = calls[-1]
            _CURSOR[key] = idx + 1
            pose = _pose_from_row(row, call_idx=idx, object_name=object_name)
            if verbose():
                print(
                    f"[grasp-replay] overflow repeat_last rep={rep} trial={trial} "
                    f"call_idx={idx} (n_frozen={len(calls)}) "
                    f"object={pose.object_name!r} pos={pose.position.tolist()}",
                    flush=True,
                )
            return pose
        msg = (
            f"GRASP_REPLAY_MISS: rep={rep} trial={trial} call_idx={idx} "
            f"exhausted (n_frozen={len(calls)})"
        )
        if mode == "miss":
            raise RuntimeError(msg)
        print(f"[grasp-replay] {msg}; live GraspNet", flush=True)
        return None

    row = calls[idx]
    _CURSOR[key] = idx + 1
    pose = _pose_from_row(row, call_idx=idx, object_name=object_name)
    if verbose():
        print(
            f"[grasp-replay] hit rep={rep} trial={trial} call_idx={idx} "
            f"object={pose.object_name!r} pos={pose.position.tolist()} "
            f"quat={pose.quaternion_wxyz.tolist()}",
            flush=True,
        )
    return pose
