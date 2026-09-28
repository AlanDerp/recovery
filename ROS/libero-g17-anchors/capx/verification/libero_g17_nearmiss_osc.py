"""Near-miss On placement: start/step oscillator on tip XY-only.

Enabled by ``CAPX_LIBERO_G17_NEARMISS_OSC=1`` (with ``RELEASE_READY=1``).

Modes
-----
* **v256** (``BANDWINDOW=1``): shared-t ``mag=start−t·step`` (15−3t);
  sign/skip from float residual axes; open via tipxy ``xy_within``.
* **v260 wash** (``FG_PREDICATE_WASH=1`` + bandwindow, no XY_DECOUPLE):
  same shared-t schedule as v256; sign+skip only via FG RelXY bands
  (WITHIN / ± closer-farther / left-right). Decision-isomorphic wash.
* **v261** (wash + ``INSCRIBED_SQUARE=1`` + ``NO_Z_ABORT=1``):
  圆中方 — axis WITHIN/skip band and start/step all ``/√2`` so L∞⊂L2 open disk
  (kills 方内圆外 deadlock); tip Z drift is logged but never aborts (measure
  success under XY-only; Z convergence deferred).
* **v258/v259** (``XY_DECOUPLE=1``): independent ``t_x/t_y``; student or FG c2c sign;
  open via predicate ``xy_within``.
* **No** GT residual-vector snap; **no** ``Δ = −r̂ · min(band, ‖r‖)`` (that is v254).

Legacy
------
* **v250**: full near-miss conjunct entry; may residual-snap (kept for history).
* **v257 fair-within**: student overlays within (``FAIR_RESIDUAL=1``).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from capx.verification.libero_g17_public_iff import LIBERO_OBJECT_ONTOP_XY_M

_DEFAULT_START_M = 0.08
_PROOF_START_M = 0.15  # 15 cm − t·3 cm schedule
_MAX_ROUNDS = 40
_Z_ABORT_M = 0.01
_V274_MAX_AXIS_ATTEMPTS = 2


def nearmiss_osc_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC", "0") == "1"


def nearmiss_osc_bandwindow_enabled() -> bool:
    """Tipxy-style entry/open window + schedule magnitude."""
    return os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW", "0") == "1"


def nearmiss_osc_fair_residual() -> bool:
    """v257: also overlay student within (ablation)."""
    return os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL", "0") == "1"


def nearmiss_osc_xy_decouple_enabled() -> bool:
    """Independent t_x/t_y; 15−3t; predicate within."""
    return os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE", "0") == "1"


def nearmiss_osc_student_sign() -> bool:
    """v258 ablation: sign from student get_object_pose instead of FG c2c."""
    return os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_STUDENT_SIGN", "0") == "1"


def nearmiss_osc_fg_predicate_wash_enabled() -> bool:
    """v260: v256 schedule/window, but sign+skip only via FG RelXY bands."""
    return os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH", "0") == "1"


def nearmiss_osc_v269_mech_enabled() -> bool:
    """Strict v256 mechanism replay with one shared GT/FG controller."""
    return os.environ.get("CAPX_LIBERO_G17_V269_MECH", "0") == "1"


def nearmiss_osc_v274_mech_contract_enabled() -> bool:
    """Held-out fix: inscribed completion band + accepted-motion schedule."""
    return os.environ.get("CAPX_LIBERO_G17_V274_MECH_CONTRACT", "0") == "1"


def nearmiss_osc_v275_trusted_tipxy_enabled() -> bool:
    """v274 RR law backed by transactional live-seeded tip XY."""
    return os.environ.get("CAPX_LIBERO_G17_V275_TRUSTED_TIPXY", "0") == "1"


def nearmiss_osc_v276_completion_ik_enabled() -> bool:
    """v274 RR law backed by completion-seeking hard-local tip XY."""
    return os.environ.get("CAPX_LIBERO_G17_V276_COMPLETION_IK", "0") == "1"


def nearmiss_osc_v286_final_fg_check_enabled() -> bool:
    """Re-observe both FG axes after the final nonzero schedule move."""
    return os.environ.get("CAPX_LIBERO_G17_V286_FINAL_FG_CHECK", "0") == "1"


def v270_live_isolation_enabled() -> bool:
    return (
        os.environ.get("CAPX_LIBERO_G17_V270_LIVE_ISOLATION", "0") == "1"
        or os.environ.get("CAPX_LIBERO_G17_V272_FAIR_LIVE", "0") == "1"
    )


def fg_event_stop_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V305_FG_EVENT_STOP", "0") == "1"


def v269_symbol_source() -> str:
    """Selected action source; the unselected source is always shadow-checked."""
    source = (
        os.environ.get("CAPX_LIBERO_G17_V269_SYMBOL_SOURCE", "fg")
        .strip()
        .lower()
    )
    if source not in {"gt", "fg"}:
        raise ValueError(f"invalid v269 symbol source: {source!r}")
    return source


def nearmiss_osc_inscribed_square_enabled() -> bool:
    """v261: 圆中方 — L∞ skip band = L2_open / √2 (square inscribed in open disk)."""
    return os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_INSCRIBED_SQUARE", "0") == "1"


def nearmiss_osc_no_z_abort_enabled() -> bool:
    """v261: log tip Z drift but never refuse open for |d_tip_z|."""
    return os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT", "0") == "1"


_SQRT2 = float(math.sqrt(2.0))


def schedule_scalars_m() -> tuple[float, float]:
    """Return (start_m, step_m) for shared-t schedule.

    Default: start=0.15, step=0.03 (15−3t).
    Inscribed-square: both /√2 so axis skip matches open disk.
    """
    start = float(_PROOF_START_M)
    step = float(LIBERO_OBJECT_ONTOP_XY_M)
    if nearmiss_osc_inscribed_square_enabled():
        start = start / _SQRT2
        step = step / _SQRT2
    return start, step


def axis_skip_band_m() -> float:
    """Per-axis WITHIN / skip half-width (L∞)."""
    band = float(LIBERO_OBJECT_ONTOP_XY_M)
    if nearmiss_osc_inscribed_square_enabled():
        return band / _SQRT2
    return band


def v274_axis_skip_band_m() -> float:
    """Axis completion square inscribed in the unchanged 3 cm release disk."""
    return float(LIBERO_OBJECT_ONTOP_XY_M) / _SQRT2


def v274_motion_accepted(result: Any) -> bool:
    """A commanded move may advance the schedule only after primitive acceptance."""
    return bool(getattr(result, "ok", False) and getattr(result, "acceptance_ok", False))


def axis_magnitude(start_m: float, step_m: float, t: int) -> float | None:
    """Return move magnitude for axis-local round ``t``, or None when exhausted."""
    mag = float(start_m) - float(t) * float(step_m)
    if mag <= 0.0:
        return None
    return mag


def axis_delta(residual_axis: float, magnitude: float, step_m: float) -> float:
    """Signed tip move for one axis; 0 if already inside ±step on that axis."""
    r = float(residual_axis)
    if abs(r) < float(step_m):
        return 0.0
    if abs(r) < 1e-12:
        return 0.0
    return float(-np.sign(r) * float(magnitude))


def axis_delta_from_fg_band(band: str, magnitude: float) -> float:
    """Map FG axis band → tip delta. Isomorphic to -sign(r)·mag with |r|<band skip.

    WITHIN_REFERENCE_BAND → 0 (same depth / laterally aligned)
    POSITIVE_OF_REFERENCE_BAND → -mag (farther / left when r>0)
    NEGATIVE_OF_REFERENCE_BAND → +mag (closer / right when r<0)
    """
    from capx.verification.libero_g17_band_obligations import (
        _NEG,
        _POS,
        _WITHIN,
    )

    if band == _WITHIN:
        return 0.0
    if band == _POS:
        return -float(magnitude)
    if band == _NEG:
        return float(magnitude)
    raise ValueError(f"unknown FG axis band: {band!r}")


def _fg_axis_bands(
    snapshot: Any, *, band_m: float | None = None
) -> tuple[str, str, str] | None:
    """Per-axis FG RelXY bands from the same c2c FG derives (predicate interface).

    ``band_m`` defaults to ontop XY (方中圆). v261 passes ontop/√2 (圆中方).
    Returns (band_x, band_y, src) or None.
    """
    from capx.verification.libero_g17_band_obligations import residual_to_band

    try:
        payload = np.asarray(snapshot.private_poses["obj01"][0], dtype=np.float64)
        receptacle = np.asarray(snapshot.private_poses["obj02"][0], dtype=np.float64)
    except Exception:
        return None
    rx = float(payload[0] - receptacle[0])
    ry = float(payload[1] - receptacle[1])
    if band_m is None:
        band_m = float(LIBERO_OBJECT_ONTOP_XY_M)
    bx = residual_to_band(axis="x", residual=rx, band_m=float(band_m))
    by = residual_to_band(axis="y", residual=ry, band_m=float(band_m))
    src = "fg_relxy_band_inscribed" if nearmiss_osc_inscribed_square_enabled() else "fg_relxy_band"
    return bx, by, src


def _gt_axis_bands(gate: Any, *, band_m: float) -> tuple[str, str] | None:
    """Quantize v256's private residual immediately; never expose the floats."""
    from capx.verification.libero_g17_band_obligations import residual_to_band

    if gate.residual_xy is None:
        return None
    rx, ry = (float(gate.residual_xy[0]), float(gate.residual_xy[1]))
    return (
        residual_to_band(axis="x", residual=rx, band_m=float(band_m)),
        residual_to_band(axis="y", residual=ry, band_m=float(band_m)),
    )


def _v269_audit(api: Any, event: str, **fields: Any) -> None:
    """Write privileged mechanism diagnostics outside captured stdout/stderr."""
    root = os.environ.get("CAPX_REPRO_AUDIT_DIR", "").strip()
    if not root:
        raise RuntimeError("v269 requires CAPX_REPRO_AUDIT_DIR")
    path = Path(root) / "libero_g17_v269_mech" / "trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    call_idx = int(getattr(api, "_capx_v269_mech_event_idx", 0))
    api._capx_v269_mech_event_idx = call_idx + 1
    row = {
        "schema": "capx_libero_g17_v269_mech_trace_v1",
        "event_idx": call_idx,
        "event": event,
        "replicate": int(os.environ.get("REPLICATE", "0")),
        "trial": int(os.environ.get("CAPX_ACTIVE_TRIAL", "1")),
        **fields,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _v269_bands(
    snapshot: Any, gate: Any, *, band_m: float
) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str], str]:
    """Materialize both sources and prove that the wash is action-equivalent."""
    fg = _fg_axis_bands(snapshot, band_m=band_m)
    gt = _gt_axis_bands(gate, band_m=band_m)
    if fg is None or gt is None:
        raise RuntimeError(
            f"symbol source unavailable: fg={fg is not None} gt={gt is not None}"
        )
    fg_pair = (fg[0], fg[1])
    if fg_pair != gt:
        raise RuntimeError(f"GT_FG_BAND_MISMATCH gt={gt!r} fg={fg_pair!r}")
    source = v269_symbol_source()
    selected = fg_pair if source == "fg" else gt
    return selected, gt, fg_pair, source


def receptacle_start_radius_m(api: Any) -> float:
    """Plate XY 'radius' = max projected half-extent (robot/base frame)."""
    try:
        from capx.verification import libero_g17_adapter as adapter
        from capx.verification.closed_gt import _projected_half_extents

        binding = adapter.bind_libero_g17_env(api._env)
        scene = adapter.extract_libero_raw_scene(api._env, binding)
        plate = scene.objects["obj02"]
        proj = _projected_half_extents(plate)
        start = float(max(float(proj[0]), float(proj[1])))
        if start > 1e-4:
            return start
    except Exception as exc:
        print(
            f"[nearmiss-osc] plate radius fallback ({exc!r}); "
            f"using {_DEFAULT_START_M}",
            flush=True,
        )
    env_override = os.environ.get("CAPX_LIBERO_G17_NEARMISS_START_M")
    if env_override:
        return float(env_override)
    return float(_DEFAULT_START_M)


def resolve_start_m(api: Any) -> float:
    """Bandwindow / xy-decouple / wash defaults to fixed 0.15; v250 uses plate radius."""
    env_override = os.environ.get("CAPX_LIBERO_G17_NEARMISS_START_M")
    if env_override:
        return float(env_override)
    if (
        nearmiss_osc_bandwindow_enabled()
        or nearmiss_osc_xy_decouple_enabled()
        or nearmiss_osc_fg_predicate_wash_enabled()
    ):
        if os.environ.get("CAPX_LIBERO_G17_NEARMISS_START_FROM_PLATE", "0") == "1":
            return receptacle_start_radius_m(api)
        return float(_PROOF_START_M)
    return receptacle_start_radius_m(api)


def _fg_c2c_axis_residual(snapshot: Any) -> tuple[np.ndarray | None, str]:
    """FG center-to-center axes: same payload−receptacle as FG placement facts."""
    try:
        payload = np.asarray(snapshot.private_poses["obj01"][0], dtype=np.float64)
        receptacle = np.asarray(snapshot.private_poses["obj02"][0], dtype=np.float64)
    except Exception as exc:
        return None, f"fg_c2c_unavailable:{exc!r}"
    residual = payload[:2] - receptacle[:2]
    return residual.astype(np.float64), "fg_c2c"


def _student_axis_residual(api: Any) -> tuple[np.ndarray | None, str]:
    """Ablation: student c2c = payload_xy − plate_xy."""
    from capx.verification.libero_g17_release_ready import student_c2c_residual_xy

    residual, _dist, src = student_c2c_residual_xy(api)
    if residual is None:
        return None, src
    return np.asarray(residual, dtype=np.float64).reshape(2), f"student:{src}"


def _axis_sign_residual(
    api: Any, snapshot: Any
) -> tuple[np.ndarray | None, str]:
    if nearmiss_osc_student_sign():
        return _student_axis_residual(api)
    return _fg_c2c_axis_residual(snapshot)


def _predicate_within(gate: Any) -> bool:
    """Legal FG predicate: WITHIN_ONTOP_XY_BAND."""
    return bool(gate.xy_within)


def run_nearmiss_osc(api: Any) -> bool:
    """True ⇒ allow open_gripper; False ⇒ refuse."""
    from capx.verification import libero_g17_v271_pt_mech as v271_pt

    if v271_pt.enabled():
        return v271_pt.run(api)
    if (
        nearmiss_osc_v276_completion_ik_enabled()
        or nearmiss_osc_v275_trusted_tipxy_enabled()
        or nearmiss_osc_v274_mech_contract_enabled()
    ):
        return _run_v274_mech_contract(api)
    if nearmiss_osc_v269_mech_enabled():
        return _run_v256_shared_mech(api)
    if (
        nearmiss_osc_fg_predicate_wash_enabled()
        and nearmiss_osc_bandwindow_enabled()
        and not nearmiss_osc_xy_decouple_enabled()
    ):
        return _run_v256_fg_predicate_wash(api)
    if nearmiss_osc_xy_decouple_enabled() and nearmiss_osc_bandwindow_enabled():
        return _run_xy_decouple_bandwindow(api)
    return _run_legacy_osc(api)


def _run_v256_shared_mech(api: Any) -> bool:
    """v269: exact v256 law, shared by GT/FG, with prompt-silent audit.

    This deliberately excludes every v265+ semantic change: no soft-held
    promotion, no near-miss router, no inscribed-square geometry and no Z
    fail-open.  The sole selectable variable is the already-discrete axis band.
    """
    from capx.integrations.franka.tip_xy_only import move_tip_xy
    from capx.verification.libero_g17_release_ready import _read_live

    source = v269_symbol_source()
    start = float(_PROOF_START_M)
    step = float(LIBERO_OBJECT_ONTOP_XY_M)

    try:
        _binding, snapshot, gate = _read_live(api)
    except Exception as exc:
        _v269_audit(api, "snapshot_failed", error=repr(exc), decision="refuse")
        return False
    if gate.relation != "on":
        _v269_audit(api, "bypass_relation", relation=gate.relation, decision="allow")
        return True
    if not gate.acquired:
        _v269_audit(api, "bypass_empty", acquired=False, decision="allow")
        return True
    if gate.release_ok_tipxy:
        _v269_audit(
            api, "already_within", acquired=True, xy_within=True, decision="allow"
        )
        return True
    if not gate.needs_align_tipxy:
        _v269_audit(api, "no_align_needed", decision="allow")
        return True
    if getattr(api, "_capx_release_ready_busy", False):
        decision = "refuse" if v270_live_isolation_enabled() else "allow"
        _v269_audit(api, "busy_reentry", decision=decision)
        return not v270_live_isolation_enabled()

    _v269_audit(
        api,
        "takeover",
        symbol_source=source,
        start_m=start,
        step_m=step,
        law="v256_shared_t",
    )
    api._capx_release_ready_busy = True
    try:
        for t in range(_MAX_ROUNDS):
            _binding, snapshot, gate = _read_live(api)
            if not gate.acquired:
                _v269_audit(api, "lost_grasp", t=t, decision="refuse")
                return False
            if gate.release_ok_tipxy:
                _v269_audit(api, "converged", t=t, xy_within=True, decision="allow")
                return True

            mag = axis_magnitude(start, step, t)
            if mag is None:
                ok = bool(gate.release_ok_tipxy)
                _v269_audit(
                    api, "schedule_done", t=t, decision="allow" if ok else "refuse"
                )
                return ok
            try:
                selected, gt_bands, fg_bands, selected_source = _v269_bands(
                    snapshot, gate, band_m=step
                )
            except Exception as exc:
                _v269_audit(
                    api,
                    "symbol_mismatch",
                    t=t,
                    error=repr(exc),
                    decision="refuse",
                )
                return False

            # Preserve v256 shared-t and its round-start XY snapshot exactly.
            for axis, band, name in (
                (0, selected[0], "x"),
                (1, selected[1], "y"),
            ):
                delta = axis_delta_from_fg_band(band, mag)
                dx = delta if axis == 0 else 0.0
                dy = delta if axis == 1 else 0.0
                _v269_audit(
                    api,
                    "axis_decision",
                    t=t,
                    axis=name,
                    band=band,
                    gt_band=gt_bands[axis],
                    fg_band=fg_bands[axis],
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
                _v269_audit(
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
                    _v269_audit(api, "z_abort", t=t, axis=name, decision="refuse")
                    return False
                _binding, snapshot, gate = _read_live(api)
                if not gate.acquired:
                    _v269_audit(
                        api,
                        "lost_grasp_mid_round",
                        t=t,
                        axis=name,
                        decision="refuse",
                    )
                    return False
                if gate.release_ok_tipxy:
                    _v269_audit(
                        api,
                        "converged_mid_round",
                        t=t,
                        axis=name,
                        decision="allow",
                    )
                    return True

        _binding, _snapshot, gate = _read_live(api)
        ok = bool(gate.release_ok_tipxy)
        _v269_audit(api, "max_rounds", decision="allow" if ok else "refuse")
        return ok
    finally:
        api._capx_release_ready_busy = False


def _run_v274_mech_contract(api: Any) -> bool:
    """v272 law with two held-out mechanical contract repairs only.

    The magnitude schedule remains exactly 15, 12, 9, 6, 3 cm.  Axis
    completion uses a 3/sqrt(2) cm half-width, so completing both axes implies
    membership in the unchanged 3 cm radial release disk.  A rejected tip-XY
    command cannot consume an axis or schedule round; the same (t, axis) is
    re-observed and retried with a bounded budget.
    """
    from capx.integrations.franka.tip_xy_only import move_tip_xy
    from capx.verification.libero_g17_release_ready import _read_live

    source = v269_symbol_source()
    start = float(_PROOF_START_M)
    step = float(LIBERO_OBJECT_ONTOP_XY_M)
    axis_band = v274_axis_skip_band_m()

    def _fg_release_ready_event() -> bool:
        """Event stop uses only the same discrete FG gate as RR."""
        _binding, _snapshot, live_gate = _read_live(api)
        return bool(live_gate.release_ok_tipxy)

    try:
        _binding, snapshot, gate = _read_live(api)
    except Exception as exc:
        _v269_audit(api, "snapshot_failed", error=repr(exc), decision="refuse")
        return False
    if gate.relation != "on":
        _v269_audit(api, "bypass_relation", relation=gate.relation, decision="allow")
        return True
    if not gate.acquired:
        _v269_audit(api, "bypass_empty", acquired=False, decision="allow")
        return True
    if gate.release_ok_tipxy:
        _v269_audit(
            api, "already_within", acquired=True, xy_within=True, decision="allow"
        )
        return True
    if not gate.needs_align_tipxy:
        _v269_audit(api, "no_align_needed", decision="allow")
        return True
    if getattr(api, "_capx_release_ready_busy", False):
        _v269_audit(api, "busy_reentry", decision="refuse")
        return False

    _v269_audit(
        api,
        "takeover",
        symbol_source=source,
        start_m=start,
        step_m=step,
        axis_band_m=axis_band,
        max_axis_attempts=_V274_MAX_AXIS_ATTEMPTS,
        law="v274_inscribed_acceptance_gated_shared_t",
    )
    api._capx_release_ready_busy = True
    try:
        for t in range(_MAX_ROUNDS):
            mag = axis_magnitude(start, step, t)
            if mag is None:
                _binding, final_snapshot, gate = _read_live(api)
                if nearmiss_osc_v286_final_fg_check_enabled():
                    if not gate.acquired:
                        _v269_audit(
                            api,
                            "final_fg_check",
                            t=t,
                            acquired=False,
                            decision="refuse",
                        )
                        return False
                    try:
                        selected, gt_bands, fg_bands, selected_source = _v269_bands(
                            final_snapshot, gate, band_m=axis_band
                        )
                    except Exception as exc:
                        _v269_audit(
                            api,
                            "final_fg_check",
                            t=t,
                            error=repr(exc),
                            decision="refuse",
                        )
                        return False
                    ok = all(
                        abs(axis_delta_from_fg_band(band, 1.0)) < 1e-9
                        for band in selected
                    )
                    _v269_audit(
                        api,
                        "final_fg_check",
                        t=t,
                        x_band=selected[0],
                        y_band=selected[1],
                        gt_bands=gt_bands,
                        fg_bands=fg_bands,
                        symbol_source=selected_source,
                        axis_band_m=axis_band,
                        decision="allow" if ok else "refuse",
                    )
                else:
                    ok = bool(gate.release_ok_tipxy)
                _v269_audit(
                    api, "schedule_done", t=t, decision="allow" if ok else "refuse"
                )
                return ok

            for axis, name in ((0, "x"), (1, "y")):
                attempt = 0
                while True:
                    _binding, snapshot, gate = _read_live(api)
                    if not gate.acquired:
                        _v269_audit(
                            api,
                            "lost_grasp",
                            t=t,
                            axis=name,
                            attempt=attempt,
                            decision="refuse",
                        )
                        return False
                    if gate.release_ok_tipxy:
                        _v269_audit(
                            api,
                            "converged",
                            t=t,
                            axis=name,
                            attempt=attempt,
                            xy_within=True,
                            decision="allow",
                        )
                        return True
                    try:
                        selected, gt_bands, fg_bands, selected_source = _v269_bands(
                            snapshot, gate, band_m=axis_band
                        )
                    except Exception as exc:
                        _v269_audit(
                            api,
                            "symbol_mismatch",
                            t=t,
                            axis=name,
                            attempt=attempt,
                            error=repr(exc),
                            decision="refuse",
                        )
                        return False

                    band = selected[axis]
                    delta = axis_delta_from_fg_band(band, mag)
                    dx = delta if axis == 0 else 0.0
                    dy = delta if axis == 1 else 0.0
                    _v269_audit(
                        api,
                        "axis_decision",
                        t=t,
                        axis=name,
                        attempt=attempt,
                        band=band,
                        axis_band_m=axis_band,
                        gt_band=gt_bands[axis],
                        fg_band=fg_bands[axis],
                        symbol_source=selected_source,
                        magnitude_m=mag,
                        dx_m=dx,
                        dy_m=dy,
                        decision="skip" if abs(delta) < 1e-9 else "move",
                    )
                    if abs(delta) < 1e-9:
                        _v269_audit(
                            api,
                            "axis_predicate_complete",
                            t=t,
                            axis=name,
                            attempt=attempt,
                            decision="advance_axis",
                        )
                        break

                    res = move_tip_xy(
                        api,
                        dx,
                        dy,
                        tol_m=0.0015,
                        max_step_m=step,
                        max_iters=20,
                        log=False,
                        stop_predicate=(
                            _fg_release_ready_event
                            if fg_event_stop_enabled()
                            else None
                        ),
                    )
                    accepted = v274_motion_accepted(res)
                    _v269_audit(
                        api,
                        "axis_result",
                        t=t,
                        axis=name,
                        attempt=attempt,
                        ok=res.ok,
                        acceptance_ok=res.acceptance_ok,
                        accepted=accepted,
                        d_tip_z_m=res.d_tip_z,
                        achieved_xy_m=res.achieved_xy,
                        final_xy_err_m=res.final_xy_err,
                        rejection_stage=getattr(res, "rejection_stage", None),
                        preflight_rejections=getattr(
                            res, "preflight_rejections", 0
                        ),
                        executed_steps=getattr(res, "executed_steps", 0),
                        max_arm_joint_delta=getattr(
                            res, "max_arm_joint_delta", None
                        ),
                        max_executed_arm_joint_delta=getattr(
                            res, "max_executed_arm_joint_delta", None
                        ),
                        motion_status=getattr(res, "status", None),
                        predicate_reached=getattr(
                            res, "predicate_reached", False
                        ),
                    )
                    if abs(res.d_tip_z) > _Z_ABORT_M:
                        if nearmiss_osc_v276_completion_ik_enabled():
                            _v269_audit(
                                api,
                                "z_diagnostic_continue",
                                t=t,
                                axis=name,
                                attempt=attempt,
                                d_tip_z_m=res.d_tip_z,
                                decision="check_holding_and_xy",
                            )
                        else:
                            _v269_audit(
                                api,
                                "z_abort",
                                t=t,
                                axis=name,
                                attempt=attempt,
                                decision="refuse",
                            )
                            return False

                    _binding, snapshot, gate = _read_live(api)
                    if not gate.acquired:
                        _v269_audit(
                            api,
                            "lost_grasp_mid_axis",
                            t=t,
                            axis=name,
                            attempt=attempt,
                            decision="refuse",
                        )
                        return False
                    if gate.release_ok_tipxy:
                        _v269_audit(
                            api,
                            "converged_mid_axis",
                            t=t,
                            axis=name,
                            attempt=attempt,
                            decision="allow",
                        )
                        return True
                    if accepted:
                        _v269_audit(
                            api,
                            "axis_motion_accepted",
                            t=t,
                            axis=name,
                            attempt=attempt,
                            decision="advance_axis",
                        )
                        break

                    attempt += 1
                    _v269_audit(
                        api,
                        "axis_rejected_no_advance",
                        t=t,
                        axis=name,
                        attempt=attempt,
                        decision="retry_same_axis",
                    )
                    if attempt >= _V274_MAX_AXIS_ATTEMPTS:
                        _v269_audit(
                            api,
                            "axis_retry_exhausted",
                            t=t,
                            axis=name,
                            attempt=attempt,
                            decision="refuse",
                        )
                        return False
        _binding, _snapshot, gate = _read_live(api)
        ok = bool(gate.release_ok_tipxy)
        _v269_audit(api, "max_rounds", decision="allow" if ok else "refuse")
        return ok
    finally:
        api._capx_release_ready_busy = False


def _run_v256_fg_predicate_wash(api: Any) -> bool:
    """v256 washed (+ optional v261): shared-t; FG RelXY bands; open = xy_within.

    v260: sign/skip via FG bands (isomorphic to float residual).
    v261: inscribed-square scalars (/√2) + no Z-abort (log Z, continue XY).
    """
    from capx.integrations.franka.tip_xy_only import move_tip_xy
    from capx.verification.libero_g17_release_ready import _read_live

    tag = "nearmiss-osc-fg-predicate-wash"
    try:
        _binding, snapshot, gate = _read_live(api)
    except Exception as exc:
        print(f"[{tag}] snapshot failed ({exc!r}); refuse open", flush=True)
        return False

    if gate.relation != "on":
        from capx.verification.libero_g17_release_ready import (
            release_ready_no_silent_allow,
        )

        if release_ready_no_silent_allow():
            print(
                f"[{tag}] relation={gate.relation!r}; open ok (not On gate)",
                flush=True,
            )
        return True
    if not gate.acquired:
        from capx.verification.libero_g17_release_ready import (
            promote_soft_held,
            release_ready_no_silent_allow,
        )

        gate, promoted = promote_soft_held(gate, snapshot)
        if promoted:
            print(
                f"[{tag}] soft-held without firm acquire; "
                "promote for tipxy (close silent-allow)",
                flush=True,
            )
        else:
            if release_ready_no_silent_allow():
                print(f"[{tag}] not acquired; open ok (empty)", flush=True)
            return True

    # v267: holding + not placement near-miss + not within → refuse (student goto).
    from capx.verification.libero_g17_release_ready import (
        is_placement_near_miss,
        nearmiss_outer_m,
        release_ready_nearmiss_gate_enabled,
    )

    if (
        release_ready_nearmiss_gate_enabled()
        and not gate.xy_within
        and not is_placement_near_miss(gate)
    ):
        print(
            f"[{tag}] holding but not near-miss "
            f"(contact={gate.contact} dist={gate.xy_distance_m} "
            f"outer={nearmiss_outer_m():.4f}m); refuse open "
            "(student goto/approach)",
            flush=True,
        )
        return False

    if gate.release_ok_tipxy:
        print(
            f"[{tag}] tipxy release_ok within={gate.xy_within} "
            f"contact={gate.contact}; open ok",
            flush=True,
        )
        return True
    if not gate.needs_align_tipxy:
        print(
            f"[{tag}] no-align-needed within={gate.xy_within} "
            f"contact={gate.contact}; open",
            flush=True,
        )
        return True

    # Env override still wins via resolve; inscribed scales the proof defaults.
    if os.environ.get("CAPX_LIBERO_G17_NEARMISS_START_M"):
        start = resolve_start_m(api)
        step = axis_skip_band_m()
    else:
        start, step = schedule_scalars_m()
    skip_band = axis_skip_band_m()
    no_z = nearmiss_osc_no_z_abort_enabled()
    geom = "circle-inscribed-square(/√2)" if nearmiss_osc_inscribed_square_enabled() else "square-circumscribed-circle"
    print(
        f"[{tag}] takeover start={start:.4f}m step={step:.4f}m "
        f"skip_band={skip_band:.4f}m geom={geom} no_z_abort={int(no_z)} "
        f"law=shared-t mag=start-t*step; sign+skip=FG RelXY bands; "
        f"open=predicate xy_within",
        flush=True,
    )

    if getattr(api, "_capx_release_ready_busy", False):
        print(f"[{tag}] busy re-entry; open ok (nested)", flush=True)
        return True
    api._capx_release_ready_busy = True
    try:
        for t in range(_MAX_ROUNDS):
            _b, snapshot, gate = _read_live(api)
            if not gate.acquired:
                from capx.verification.libero_g17_release_ready import (
                    promote_soft_held,
                )

                gate, promoted = promote_soft_held(gate, snapshot)
                if promoted:
                    print(
                        f"[{tag}] soft-held without firm acquire t={t}; "
                        "continue tipxy (close silent-allow)",
                        flush=True,
                    )
                else:
                    print(f"[{tag}] lost grasp t={t}; refuse open", flush=True)
                    return False
            if gate.release_ok_tipxy:
                print(
                    f"[{tag}] converged t={t} within={gate.xy_within}",
                    flush=True,
                )
                return True
            if not gate.contact:
                print(
                    f"[{tag}] contact=False t={t} (continue schedule; flat plate)",
                    flush=True,
                )

            bands = _fg_axis_bands(snapshot, band_m=skip_band)
            if bands is None:
                print(
                    f"[{tag}] FG axis bands unavailable t={t}; refuse open",
                    flush=True,
                )
                return False
            bx, by, src = bands

            mag = axis_magnitude(start, step, t)
            if mag is None:
                ok = bool(gate.release_ok_tipxy)
                print(
                    f"[{tag}] schedule done release_ok_tipxy={ok} "
                    f"(no residual-vector snap)",
                    flush=True,
                )
                return ok

            for axis, band, name in ((0, bx, "x"), (1, by, "y")):
                d = axis_delta_from_fg_band(band, mag)
                if abs(d) < 1e-9:
                    print(
                        f"[{tag}] t={t} axis={name} skip band={band} "
                        f"(WITHIN/align) src={src}",
                        flush=True,
                    )
                    continue
                dx = d if axis == 0 else 0.0
                dy = d if axis == 1 else 0.0
                print(
                    f"[{tag}] t={t} axis={name} mag={mag:.4f} band={band} "
                    f"d={d:+.4f} src={src}",
                    flush=True,
                )
                res = move_tip_xy(
                    api,
                    dx,
                    dy,
                    tol_m=0.0015,
                    max_step_m=float(step),
                    max_iters=20,
                    log=True,
                )
                if abs(res.d_tip_z) > _Z_ABORT_M:
                    if no_z:
                        print(
                            f"[{tag}] Z drift t={t} d_tip_z={res.d_tip_z:+.5f} "
                            f"(no-abort; continue XY)",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{tag}] Z abort t={t} d_tip_z={res.d_tip_z:+.5f}",
                            flush=True,
                        )
                        return False
                _b, snapshot, gate = _read_live(api)
                if not gate.acquired:
                    from capx.verification.libero_g17_release_ready import (
                        promote_soft_held,
                    )

                    gate, promoted = promote_soft_held(gate, snapshot)
                    if promoted:
                        print(
                            f"[{tag}] soft-held without firm acquire mid t={t}; "
                            "continue tipxy (close silent-allow)",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{tag}] lost grasp mid-round t={t}; refuse open",
                            flush=True,
                        )
                        return False
                if gate.release_ok_tipxy:
                    print(
                        f"[{tag}] converged mid-round t={t} "
                        f"within={gate.xy_within}",
                        flush=True,
                    )
                    return True
                bands = _fg_axis_bands(snapshot, band_m=skip_band)
                if bands is None:
                    print(
                        f"[{tag}] FG bands lost mid t={t}; refuse open",
                        flush=True,
                    )
                    return False
                bx, by, src = bands

        _b, _s, gate = _read_live(api)
        ok = bool(gate.release_ok_tipxy)
        print(f"[{tag}] max_rounds release_ok_tipxy={ok}", flush=True)
        return ok
    finally:
        api._capx_release_ready_busy = False


def _run_xy_decouple_bandwindow(api: Any) -> bool:
    """XY-decoupled 15−3t; FG c2c sign (or student-sign ablation); predicate within."""
    from capx.integrations.franka.tip_xy_only import move_tip_xy
    from capx.verification.libero_g17_release_ready import _read_live

    student_sign = nearmiss_osc_student_sign()
    tag = (
        "nearmiss-osc-xy-decouple-student-sign"
        if student_sign
        else "nearmiss-osc-xy-decouple-fg"
    )
    try:
        _binding, snapshot, gate = _read_live(api)
    except Exception as exc:
        print(f"[{tag}] snapshot failed ({exc!r}); refuse open", flush=True)
        return False

    if gate.relation != "on":
        return True
    if not gate.acquired:
        return True
    if _predicate_within(gate):
        print(
            f"[{tag}] predicate xy_within; open ok (acquired={gate.acquired})",
            flush=True,
        )
        return True

    start = resolve_start_m(api)
    step = float(LIBERO_OBJECT_ONTOP_XY_M)
    sign_law = "student_get_object_pose" if student_sign else "fg_c2c"
    print(
        f"[{tag}] takeover start={start:.4f}m step={step:.4f}m "
        f"law=mag(start-t*step) per-axis; sign={sign_law}; within=predicate FG",
        flush=True,
    )

    if getattr(api, "_capx_release_ready_busy", False):
        return True
    api._capx_release_ready_busy = True
    t_x = 0
    t_y = 0
    try:
        for round_i in range(_MAX_ROUNDS):
            _b, snapshot, gate = _read_live(api)
            if not gate.acquired:
                print(f"[{tag}] lost grasp round={round_i}; refuse open", flush=True)
                return False
            if _predicate_within(gate):
                print(
                    f"[{tag}] predicate within round={round_i} "
                    f"t_x={t_x} t_y={t_y}; open",
                    flush=True,
                )
                return True
            if not gate.contact:
                print(
                    f"[{tag}] contact=False round={round_i} "
                    "(continue schedule; flat plate)",
                    flush=True,
                )

            axis_r, src = _axis_sign_residual(api, snapshot)
            if axis_r is None:
                print(
                    f"[{tag}] sign residual unavailable ({src}); refuse open",
                    flush=True,
                )
                return False

            moved_any = False
            exhausted_both = True
            for axis, t_axis, name in ((0, t_x, "x"), (1, t_y, "y")):
                mag = axis_magnitude(start, step, t_axis)
                if mag is None:
                    continue
                exhausted_both = False
                r_a = float(axis_r[axis])
                d = axis_delta(r_a, mag, step)
                if abs(d) < 1e-9:
                    print(
                        f"[{tag}] round={round_i} axis={name} skip "
                        f"|r|={abs(r_a):.4f}<step t={t_axis} src={src}",
                        flush=True,
                    )
                    continue
                dx = d if axis == 0 else 0.0
                dy = d if axis == 1 else 0.0
                print(
                    f"[{tag}] round={round_i} axis={name} t={t_axis} "
                    f"mag={mag:.4f} sign={-int(np.sign(r_a)):+d} "
                    f"r={r_a:+.4f} d={d:+.4f} src={src}",
                    flush=True,
                )
                res = move_tip_xy(
                    api,
                    dx,
                    dy,
                    tol_m=0.0015,
                    max_step_m=float(step),
                    max_iters=20,
                    log=True,
                )
                if abs(res.d_tip_z) > _Z_ABORT_M:
                    print(
                        f"[{tag}] Z abort round={round_i} "
                        f"d_tip_z={res.d_tip_z:+.5f}",
                        flush=True,
                    )
                    return False
                moved_any = True
                if axis == 0:
                    t_x = t_axis + 1
                else:
                    t_y = t_axis + 1

                _b, snapshot, gate = _read_live(api)
                if not gate.acquired:
                    print(
                        f"[{tag}] lost grasp mid-axis={name}; refuse open",
                        flush=True,
                    )
                    return False
                if _predicate_within(gate):
                    print(
                        f"[{tag}] predicate within after axis={name} "
                        f"t_x={t_x} t_y={t_y}; open",
                        flush=True,
                    )
                    return True
                axis_r, src = _axis_sign_residual(api, snapshot)
                if axis_r is None:
                    print(
                        f"[{tag}] sign residual lost mid ({src}); refuse open",
                        flush=True,
                    )
                    return False

            if exhausted_both:
                ok = _predicate_within(gate)
                print(
                    f"[{tag}] both axis schedules exhausted "
                    f"predicate_within={ok} t_x={t_x} t_y={t_y} "
                    "(no GT residual snap)",
                    flush=True,
                )
                return ok
            if not moved_any:
                print(
                    f"[{tag}] no axis move (half-band) but "
                    f"predicate still out; refuse open t_x={t_x} t_y={t_y}",
                    flush=True,
                )
                return False

        _b, _s, gate = _read_live(api)
        ok = bool(gate.acquired and _predicate_within(gate))
        print(
            f"[{tag}] max_rounds predicate_within={ok} t_x={t_x} t_y={t_y}",
            flush=True,
        )
        return ok
    finally:
        api._capx_release_ready_busy = False


def _run_legacy_osc(api: Any) -> bool:
    """v250 / pre-v258 bandwindow (+ optional fair-within). Kept for history."""
    from capx.integrations.franka.tip_xy_only import move_tip_xy
    from capx.verification.libero_g17_release_ready import (
        _read_live,
        overlay_gate_student_residual,
    )

    bandwindow = nearmiss_osc_bandwindow_enabled()
    fair_r = nearmiss_osc_fair_residual()
    if fair_r and bandwindow:
        tag = "nearmiss-osc-bandwindow-fair"
    elif bandwindow:
        tag = "nearmiss-osc-bandwindow"
    else:
        tag = "nearmiss-osc"
    residual_src = "private_poses"

    def _apply(gate: Any) -> tuple[Any | None, str]:
        if not (bandwindow and fair_r):
            return gate, "private_poses"
        overlaid, src = overlay_gate_student_residual(gate, api)
        if overlaid is None:
            return None, src
        return overlaid, f"get_object_pose:{src}"

    try:
        _binding, _snapshot, gate = _read_live(api)
    except Exception as exc:
        print(f"[{tag}] snapshot failed ({exc!r}); refuse open", flush=True)
        return False

    if gate.relation != "on":
        return True
    if not gate.acquired:
        return True

    if bandwindow:
        gate, residual_src = _apply(gate)
        if gate is None:
            print(
                f"[{tag}] fair residual failed ({residual_src}); refuse open",
                flush=True,
            )
            return False
        if gate.release_ok_tipxy:
            print(
                f"[{tag}] tipxy release_ok dist={gate.xy_distance_m:.4f}m "
                f"residual_src={residual_src}; open ok",
                flush=True,
            )
            return True
        if not gate.needs_align_tipxy:
            print(
                f"[{tag}] no-align-needed within={gate.xy_within} "
                f"residual_src={residual_src}; open",
                flush=True,
            )
            return True
    else:
        if gate.release_ok:
            print(
                f"[{tag}] release_ok dist={gate.xy_distance_m:.4f}m; open ok",
                flush=True,
            )
            return True
        if not (
            gate.acquired and gate.contact and gate.surface_on and not gate.xy_within
        ):
            print(
                f"[{tag}] not nearmiss "
                f"acq={gate.acquired} contact={gate.contact} "
                f"surface={gate.surface_on} within={gate.xy_within}; refuse open",
                flush=True,
            )
            return False

    start = resolve_start_m(api)
    step = float(LIBERO_OBJECT_ONTOP_XY_M)
    print(
        f"[{tag}] takeover start={start:.4f}m step={step:.4f}m "
        f"dist={gate.xy_distance_m:.4f} residual={gate.residual_xy} "
        f"residual_src={residual_src} (legacy shared-t)",
        flush=True,
    )

    if getattr(api, "_capx_release_ready_busy", False):
        return True
    api._capx_release_ready_busy = True
    try:
        for t in range(_MAX_ROUNDS):
            _b, _s, gate = _read_live(api)
            if not gate.acquired:
                print(f"[{tag}] lost grasp t={t}; refuse open", flush=True)
                return False
            if bandwindow:
                gate, residual_src = _apply(gate)
                if gate is None:
                    print(
                        f"[{tag}] fair residual failed t={t} ({residual_src}); "
                        "refuse open",
                        flush=True,
                    )
                    return False
            if bandwindow and gate.release_ok_tipxy:
                print(
                    f"[{tag}] converged t={t} dist={gate.xy_distance_m:.4f}m "
                    f"residual_src={residual_src}",
                    flush=True,
                )
                return True
            if (not bandwindow) and gate.release_ok:
                print(
                    f"[{tag}] converged t={t} dist={gate.xy_distance_m:.4f}m",
                    flush=True,
                )
                return True
            if not bandwindow:
                if not (gate.contact and gate.surface_on):
                    print(
                        f"[{tag}] lost contact/surface t={t}; refuse open",
                        flush=True,
                    )
                    return False
            elif not gate.contact:
                print(
                    f"[{tag}] contact=False t={t} (continue schedule; flat plate)",
                    flush=True,
                )
            if gate.residual_xy is None:
                print(f"[{tag}] no residual t={t}; refuse open", flush=True)
                return False

            mag = axis_magnitude(start, step, t)
            rx, ry = float(gate.residual_xy[0]), float(gate.residual_xy[1])

            if mag is None:
                if bandwindow:
                    ok = bool(gate.release_ok_tipxy)
                    print(
                        f"[{tag}] schedule done release_ok_tipxy={ok} "
                        f"dist={gate.xy_distance_m} residual_src={residual_src} "
                        "(no residual snap)",
                        flush=True,
                    )
                    return ok
                if abs(rx) >= step or abs(ry) >= step or not gate.xy_within:
                    dx = -rx if abs(rx) >= 1e-6 else 0.0
                    dy = -ry if abs(ry) >= 1e-6 else 0.0
                    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                        res = move_tip_xy(api, dx, dy, log=True)
                        if abs(res.d_tip_z) > _Z_ABORT_M:
                            print(
                                f"[{tag}] Z abort final d_tip_z={res.d_tip_z:+.5f}",
                                flush=True,
                            )
                            return False
                _b, _s, gate = _read_live(api)
                ok = bool(gate.release_ok)
                print(
                    f"[{tag}] schedule done release_ok={ok} "
                    f"dist={gate.xy_distance_m}",
                    flush=True,
                )
                return ok

            for axis, r in ((0, rx), (1, ry)):
                d = axis_delta(r, mag, step)
                if abs(d) < 1e-9:
                    continue
                dx = d if axis == 0 else 0.0
                dy = d if axis == 1 else 0.0
                print(
                    f"[{tag}] t={t} axis={'xy'[axis]} mag={mag:.4f} "
                    f"r={r:+.4f} d={d:+.4f} residual_src={residual_src}",
                    flush=True,
                )
                res = move_tip_xy(
                    api,
                    dx,
                    dy,
                    tol_m=0.0015,
                    max_step_m=float(step),
                    max_iters=20,
                    log=True,
                )
                if abs(res.d_tip_z) > _Z_ABORT_M:
                    print(
                        f"[{tag}] Z abort t={t} d_tip_z={res.d_tip_z:+.5f}",
                        flush=True,
                    )
                    return False
                _b, _s, gate = _read_live(api)
                if not gate.acquired:
                    print(
                        f"[{tag}] lost grasp mid-round t={t}; refuse open",
                        flush=True,
                    )
                    return False
                if bandwindow:
                    gate, residual_src = _apply(gate)
                    if gate is None:
                        print(
                            f"[{tag}] fair residual failed mid t={t} "
                            f"({residual_src}); refuse open",
                            flush=True,
                        )
                        return False
                if bandwindow and gate.release_ok_tipxy:
                    print(
                        f"[{tag}] converged mid-round t={t} "
                        f"dist={gate.xy_distance_m:.4f}m "
                        f"residual_src={residual_src}",
                        flush=True,
                    )
                    return True
                if (not bandwindow) and gate.release_ok:
                    print(
                        f"[{tag}] converged mid-round t={t} "
                        f"dist={gate.xy_distance_m:.4f}m",
                        flush=True,
                    )
                    return True
                if gate.residual_xy is not None:
                    rx, ry = float(gate.residual_xy[0]), float(gate.residual_xy[1])

        _b, _s, gate = _read_live(api)
        if bandwindow:
            gate, residual_src = _apply(gate)
            if gate is None:
                print(
                    f"[{tag}] fair residual failed after max_rounds "
                    f"({residual_src}); refuse open",
                    flush=True,
                )
                return False
        ok = bool(gate.release_ok_tipxy if bandwindow else gate.release_ok)
        print(
            f"[{tag}] max_rounds release_ok={ok} dist={gate.xy_distance_m} "
            f"residual_src={residual_src}",
            flush=True,
        )
        return ok
    finally:
        api._capx_release_ready_busy = False
