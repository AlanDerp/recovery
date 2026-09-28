"""Near-miss placement fix relative to v238: tip_xy only, fail-open.

Flag: ``CAPX_LIBERO_G17_NEARMISS_FAILOPEN=1``

Contract (well-defined):
  * Default = v238: always allow ``open_gripper`` (never refuse).
  * Intervene only on clear near-miss:
        holding(acquired) ∧ plate_contact ∧ band ≤ ‖c2c‖ ≤ R
    where band = ``LIBERO_OBJECT_ONTOP_XY_M`` (0.03) and R = plate XY half-extent.
  * Action: small residual ``move_tip_xy`` steps toward c2c=0.
  * Then open regardless of whether within was reached (fail-open).

Does not use ``goto_pose``. Independent of ReleaseReady refuse-open paths.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from capx.verification.libero_g17_public_iff import LIBERO_OBJECT_ONTOP_XY_M

_STEP_M = 0.004
_MAX_ITERS = 20
_Z_ABORT_M = 0.008
_DEFAULT_R_M = 0.12
_TIP_TOL_M = 0.0015  # below step so micro-moves are not no-ops


def nearmiss_failopen_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_NEARMISS_FAILOPEN", "0") == "1"


def _plate_r_m(api: Any) -> float:
    try:
        from capx.verification import libero_g17_adapter as adapter
        from capx.verification.closed_gt import _projected_half_extents

        binding = adapter.bind_libero_g17_env(api._env)
        scene = adapter.extract_libero_raw_scene(api._env, binding)
        proj = _projected_half_extents(scene.objects["obj02"])
        r = float(max(float(proj[0]), float(proj[1])))
        if r > 1e-3:
            return r
    except Exception as exc:
        print(f"[nearmiss-fo] plate R fallback ({exc!r})", flush=True)
    ov = os.environ.get("CAPX_LIBERO_G17_NEARMISS_R_M")
    return float(ov) if ov else float(_DEFAULT_R_M)


def _read_gate(api: Any) -> Any:
    from capx.verification.libero_g17_release_ready import _read_live

    _b, _s, gate = _read_live(api)
    return gate


def is_nearmiss(gate: Any, *, r_m: float, band_m: float = LIBERO_OBJECT_ONTOP_XY_M) -> bool:
    """holding ∧ contact ∧ band ≤ dist ≤ R (on-plate near-miss)."""
    if gate.relation != "on":
        return False
    if not gate.acquired or not gate.contact:
        return False
    if gate.xy_distance_m is None or gate.residual_xy is None:
        return False
    d = float(gate.xy_distance_m)
    return bool(float(band_m) <= d <= float(r_m))


def run_nearmiss_tipxy_failopen(api: Any) -> bool:
    """Always returns True (fail-open). May tip_xy-nudge first when near-miss."""
    from capx.integrations.franka.tip_xy_only import move_tip_xy

    if getattr(api, "_capx_nearmiss_fo_busy", False):
        return True

    try:
        gate = _read_gate(api)
    except Exception as exc:
        print(f"[nearmiss-fo] snapshot failed ({exc!r}); open anyway", flush=True)
        return True

    if gate.relation != "on" or not gate.acquired:
        return True

    band = float(LIBERO_OBJECT_ONTOP_XY_M)
    if gate.xy_within and gate.contact:
        print(
            f"[nearmiss-fo] already within dist={gate.xy_distance_m:.4f}m; open",
            flush=True,
        )
        return True

    r_m = _plate_r_m(api)
    if not is_nearmiss(gate, r_m=r_m, band_m=band):
        print(
            f"[nearmiss-fo] skip (not nearmiss) "
            f"acq={gate.acquired} contact={gate.contact} "
            f"within={gate.xy_within} dist={gate.xy_distance_m} R={r_m:.3f}; open",
            flush=True,
        )
        return True

    print(
        f"[nearmiss-fo] takeover dist={gate.xy_distance_m:.4f}m "
        f"residual={gate.residual_xy} band={band} R={r_m:.3f}",
        flush=True,
    )
    api._capx_nearmiss_fo_busy = True
    try:
        for i in range(_MAX_ITERS):
            try:
                gate = _read_gate(api)
            except Exception as exc:
                print(f"[nearmiss-fo] mid snapshot fail ({exc!r}); open", flush=True)
                return True
            if not gate.acquired:
                print(f"[nearmiss-fo] lost grasp iter={i}; open anyway", flush=True)
                return True
            if gate.xy_within and gate.contact:
                print(
                    f"[nearmiss-fo] converged iter={i} dist={gate.xy_distance_m:.4f}m; open",
                    flush=True,
                )
                return True
            if not gate.contact:
                print(f"[nearmiss-fo] lost contact iter={i}; open anyway", flush=True)
                return True
            if gate.residual_xy is None or gate.xy_distance_m is None:
                print(f"[nearmiss-fo] no residual iter={i}; open anyway", flush=True)
                return True
            if float(gate.xy_distance_m) > r_m:
                print(
                    f"[nearmiss-fo] left plate band dist={gate.xy_distance_m:.4f}; open",
                    flush=True,
                )
                return True

            residual = np.asarray(gate.residual_xy, dtype=np.float64).reshape(2)
            dist = float(np.linalg.norm(residual))
            if dist < band:
                return True
            # Unit toward zero, capped step (always ≥ tip tol).
            step = min(_STEP_M, dist * 0.7)
            step = max(step, _TIP_TOL_M)
            delta = -residual / dist * step
            print(
                f"[nearmiss-fo] iter={i} dist={dist:.4f}m "
                f"dxy=({delta[0]:+.4f},{delta[1]:+.4f})",
                flush=True,
            )
            try:
                res = move_tip_xy(
                    api,
                    float(delta[0]),
                    float(delta[1]),
                    tol_m=_TIP_TOL_M,
                    max_step_m=_STEP_M,
                    log=True,
                )
            except Exception as exc:
                print(f"[nearmiss-fo] tip_xy failed ({exc!r}); open anyway", flush=True)
                return True
            if abs(res.d_tip_z) > _Z_ABORT_M:
                print(
                    f"[nearmiss-fo] Z abort d_tip_z={res.d_tip_z:+.5f}; open anyway",
                    flush=True,
                )
                return True

        print("[nearmiss-fo] max_iters; open anyway", flush=True)
        return True
    finally:
        api._capx_nearmiss_fo_busy = False
