"""Paired PT ablation of the v269 mechanical golden anchor.

Both arms execute this same controller and materialize GT, ideal-FG, and PT
axis bands at every action decision.  The only selectable variable is whether
the action band comes from ideal FG or PT (`get_object_pose`).  The v269
release predicate, shared-t schedule, tip-XY primitive, and Z-abort semantics
remain unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from capx.verification.libero_g17_public_iff import LIBERO_OBJECT_ONTOP_XY_M


_START_M = 0.15
_MAX_ROUNDS = 40
_Z_ABORT_M = 0.01


def enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V271_PT_PAIR", "0") == "1"


def symbol_source() -> str:
    source = os.environ.get("CAPX_LIBERO_G17_V271_SYMBOL_SOURCE", "").strip().lower()
    if source not in {"fg", "pt"}:
        raise ValueError(f"invalid v271 symbol source: {source!r}")
    return source


def _audit(api: Any, event: str, **fields: Any) -> None:
    root = os.environ.get("CAPX_REPRO_AUDIT_DIR", "").strip()
    if not root:
        raise RuntimeError("v271 requires CAPX_REPRO_AUDIT_DIR")
    path = Path(root) / "libero_g17_v271_pt_pair" / "trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = int(getattr(api, "_capx_v271_pt_event_idx", 0))
    api._capx_v271_pt_event_idx = idx + 1
    row = {
        "schema": "capx_libero_g17_v271_pt_pair_trace_v1",
        "event_idx": idx,
        "event": event,
        "replicate": int(os.environ.get("REPLICATE", "0")),
        "trial": int(os.environ.get("CAPX_ACTIVE_TRIAL", "1")),
        **fields,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _pt_axis_bands(api: Any, *, band_m: float) -> tuple[tuple[str, str], str]:
    from capx.verification.libero_g17_band_obligations import residual_to_band
    from capx.verification.libero_g17_nearmiss_osc import _student_axis_residual

    residual, src = _student_axis_residual(api)
    if residual is None:
        raise RuntimeError(f"PT_SYMBOL_UNAVAILABLE source={src}")
    residual = np.asarray(residual, dtype=np.float64).reshape(2)
    return (
        (
            residual_to_band(
                axis="x", residual=float(residual[0]), band_m=float(band_m)
            ),
            residual_to_band(
                axis="y", residual=float(residual[1]), band_m=float(band_m)
            ),
        ),
        src,
    )


def _all_bands(
    api: Any, snapshot: Any, gate: Any, *, band_m: float
) -> tuple[
    tuple[str, str],
    tuple[str, str],
    tuple[str, str],
    tuple[str, str],
    str,
    str,
]:
    """Materialize every source; select only FG or PT for action."""
    from capx.verification.libero_g17_nearmiss_osc import (
        _fg_axis_bands,
        _gt_axis_bands,
    )

    fg = _fg_axis_bands(snapshot, band_m=band_m)
    gt = _gt_axis_bands(gate, band_m=band_m)
    if fg is None or gt is None:
        raise RuntimeError(
            f"anchor symbol unavailable: fg={fg is not None} gt={gt is not None}"
        )
    fg_pair = (fg[0], fg[1])
    if fg_pair != gt:
        raise RuntimeError(f"GT_FG_BAND_MISMATCH gt={gt!r} fg={fg_pair!r}")
    pt_pair, pt_src = _pt_axis_bands(api, band_m=band_m)
    source = symbol_source()
    selected = fg_pair if source == "fg" else pt_pair
    return selected, gt, fg_pair, pt_pair, source, pt_src


def run(api: Any) -> bool:
    """Execute the exact v269/v256 mechanical law with FG or PT action bands."""
    from capx.integrations.franka.tip_xy_only import move_tip_xy
    from capx.verification.libero_g17_nearmiss_osc import (
        axis_delta_from_fg_band,
        axis_magnitude,
    )
    from capx.verification.libero_g17_release_ready import _read_live

    source = symbol_source()
    start = float(_START_M)
    step = float(LIBERO_OBJECT_ONTOP_XY_M)

    try:
        _binding, snapshot, gate = _read_live(api)
    except Exception as exc:
        _audit(api, "snapshot_failed", error=repr(exc), decision="refuse")
        return False
    if gate.relation != "on":
        _audit(api, "bypass_relation", relation=gate.relation, decision="allow")
        return True
    if not gate.acquired:
        _audit(api, "bypass_empty", acquired=False, decision="allow")
        return True
    # Keep the v269 release predicate fixed. PT ablates only action bands.
    if gate.release_ok_tipxy:
        _audit(api, "already_within", acquired=True, decision="allow")
        return True
    if not gate.needs_align_tipxy:
        _audit(api, "no_align_needed", decision="allow")
        return True
    if getattr(api, "_capx_release_ready_busy", False):
        _audit(api, "busy_reentry", decision="allow")
        return True

    _audit(
        api,
        "takeover",
        symbol_source=source,
        start_m=start,
        step_m=step,
        law="v256_shared_t",
        pt_scope="action_bands_only",
    )
    api._capx_release_ready_busy = True
    try:
        for t in range(_MAX_ROUNDS):
            _binding, snapshot, gate = _read_live(api)
            if not gate.acquired:
                _audit(api, "lost_grasp", t=t, decision="refuse")
                return False
            if gate.release_ok_tipxy:
                _audit(api, "converged", t=t, decision="allow")
                return True

            mag = axis_magnitude(start, step, t)
            if mag is None:
                ok = bool(gate.release_ok_tipxy)
                _audit(
                    api, "schedule_done", t=t, decision="allow" if ok else "refuse"
                )
                return ok
            try:
                selected, gt, fg, pt, selected_source, pt_src = _all_bands(
                    api, snapshot, gate, band_m=step
                )
            except Exception as exc:
                _audit(
                    api,
                    "symbol_unavailable",
                    t=t,
                    error=repr(exc),
                    decision="refuse",
                )
                return False

            # Preserve v256 shared-t and round-start XY snapshot exactly.
            for axis, band, name in (
                (0, selected[0], "x"),
                (1, selected[1], "y"),
            ):
                delta = axis_delta_from_fg_band(band, mag)
                dx = delta if axis == 0 else 0.0
                dy = delta if axis == 1 else 0.0
                _audit(
                    api,
                    "axis_decision",
                    t=t,
                    axis=name,
                    band=band,
                    gt_band=gt[axis],
                    fg_band=fg[axis],
                    pt_band=pt[axis],
                    pt_matches_fg=pt[axis] == fg[axis],
                    pt_source=pt_src,
                    symbol_source=selected_source,
                    magnitude_m=mag,
                    dx_m=dx,
                    dy_m=dy,
                    decision="skip" if abs(delta) < 1e-9 else "move",
                )
                if abs(delta) < 1e-9:
                    continue
                res = move_tip_xy(
                    api,
                    dx,
                    dy,
                    tol_m=0.0015,
                    max_step_m=step,
                    max_iters=20,
                    log=False,
                )
                _audit(
                    api,
                    "axis_result",
                    t=t,
                    axis=name,
                    ok=res.ok,
                    acceptance_ok=res.acceptance_ok,
                    d_tip_z_m=res.d_tip_z,
                    achieved_xy_m=res.achieved_xy,
                    final_xy_err_m=res.final_xy_err,
                )
                if abs(res.d_tip_z) > _Z_ABORT_M:
                    _audit(api, "z_abort", t=t, axis=name, decision="refuse")
                    return False
                _binding, snapshot, gate = _read_live(api)
                if not gate.acquired:
                    _audit(
                        api,
                        "lost_grasp_mid_round",
                        t=t,
                        axis=name,
                        decision="refuse",
                    )
                    return False
                if gate.release_ok_tipxy:
                    _audit(
                        api,
                        "converged_mid_round",
                        t=t,
                        axis=name,
                        decision="allow",
                    )
                    return True

        _binding, _snapshot, gate = _read_live(api)
        ok = bool(gate.release_ok_tipxy)
        _audit(api, "max_rounds", decision="allow" if ok else "refuse")
        return ok
    finally:
        api._capx_release_ready_busy = False
