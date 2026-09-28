"""Fair place-end servo: public ``get_object_pose`` + ``move_tip_xy`` (no GT).

Flag: ``CAPX_LIBERO_G17_FAIR_PLACE_SERVO=1``

Contract
--------
* Default path never refuses ``open_gripper`` (fail-open).
* Intervene only on a **tight** fair near-miss:
      gripper looks closed ∧ band ≤ ‖plate_xy − payload_xy‖ ≤ max_near
  with ``band = LIBERO_OBJECT_ONTOP_XY_M`` (0.03) and ``max_near`` default 0.05 m.
* Pose source is **only** the student-visible ``api.get_object_pose`` (same Molmo/SAM
  stack as baseline). No ``closed_gt``, adapter scene, or ReleaseReady gate.
* Payload fallback: if payload pose is None, use obs tip XY (proprio) as proxy.
* One vision query pair at start; then tip-closed-loop toward the implied tip goal
  (rigid-grasp assumption). Short budget + Z abort → open anyway.
* Also exposes student ``move_tip_xy`` when ``CAPX_LIBERO_TIP_XY_ONLY=1`` (arm script).

This is the root-cure shape for placement residual under fair perception — not a
GT open-hook salvage (v252).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from capx.verification.libero_g17_public_iff import LIBERO_OBJECT_ONTOP_XY_M

_STEP_M = 0.008
_MAX_ITERS = 8
_Z_ABORT_M = 0.008
_DEFAULT_MAX_NEAR_M = 0.05
_TIP_TOL_M = 0.0015
_GRIPPER_CLOSED_MAX = 0.45  # obs tip[7]: 0 closed … 1 open


def fair_place_servo_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_FAIR_PLACE_SERVO", "0") == "1"


def _payload_name() -> str:
    return os.environ.get("CAPX_LIBERO_G17_PLACE_PAYLOAD", "akita_black_bowl").strip()


def _receptacle_name() -> str:
    return os.environ.get("CAPX_LIBERO_G17_PLACE_RECEPTACLE", "plate").strip()


def _max_near_m() -> float:
    raw = os.environ.get("CAPX_LIBERO_G17_FAIR_PLACE_MAX_NEAR_M")
    return float(raw) if raw else float(_DEFAULT_MAX_NEAR_M)


def _read_tip_and_gripper(api: Any) -> tuple[np.ndarray, float]:
    from capx.integrations.franka.tip_xy_only import read_obs_tip

    tip_pos, _quat = read_obs_tip(api)
    obs = api.get_observation()
    cart = np.asarray(obs["robot_cartesian_pos"], dtype=np.float64).reshape(-1)
    grip = float(cart[7]) if cart.size >= 8 else 0.0
    return tip_pos, grip


def fair_xy_residual(
    api: Any,
    *,
    payload_name: str | None = None,
    receptacle_name: str | None = None,
) -> tuple[np.ndarray | None, str]:
    """Return residual plate_xy − payload_xy from public poses, or None + reason."""
    payload = payload_name or _payload_name()
    receptacle = receptacle_name or _receptacle_name()
    plate_pos, _ = api.get_object_pose(receptacle)
    if plate_pos is None:
        return None, f"no_pose:{receptacle}"
    plate_xy = np.asarray(plate_pos, dtype=np.float64).reshape(-1)[:2]

    payload_pos, _ = api.get_object_pose(payload)
    tip_pos, _grip = _read_tip_and_gripper(api)
    if payload_pos is None:
        payload_xy = tip_pos[:2].copy()
        src = "tip_proxy"
    else:
        payload_xy = np.asarray(payload_pos, dtype=np.float64).reshape(-1)[:2]
        src = "get_object_pose"

    residual = plate_xy - payload_xy
    return residual, src


def should_servo(
    dist_m: float,
    *,
    band_m: float = LIBERO_OBJECT_ONTOP_XY_M,
    max_near_m: float | None = None,
) -> bool:
    hi = float(_max_near_m() if max_near_m is None else max_near_m)
    return bool(float(band_m) <= float(dist_m) <= hi)


def run_fair_place_servo(api: Any) -> bool:
    """Always returns True (fail-open). May tip_xy-nudge first on fair near-miss."""
    from capx.integrations.franka.tip_xy_only import move_tip_xy

    if getattr(api, "_capx_fair_place_busy", False):
        return True

    try:
        tip0, grip = _read_tip_and_gripper(api)
    except Exception as exc:
        print(f"[fair-place] tip read failed ({exc!r}); open", flush=True)
        return True

    if grip > _GRIPPER_CLOSED_MAX:
        print(f"[fair-place] skip gripper_open grip={grip:.3f}; open", flush=True)
        return True

    try:
        residual, src = fair_xy_residual(api)
    except Exception as exc:
        print(f"[fair-place] pose failed ({exc!r}); open", flush=True)
        return True

    if residual is None:
        print(f"[fair-place] skip ({src}); open", flush=True)
        return True

    dist = float(np.linalg.norm(residual))
    band = float(LIBERO_OBJECT_ONTOP_XY_M)
    max_near = _max_near_m()
    if dist < band:
        print(
            f"[fair-place] already within dist={dist:.4f}m src={src}; open",
            flush=True,
        )
        return True
    if not should_servo(dist, band_m=band, max_near_m=max_near):
        print(
            f"[fair-place] skip dist={dist:.4f}m not in "
            f"[{band:.3f},{max_near:.3f}] src={src}; open",
            flush=True,
        )
        return True

    # Rigid-grasp: move tip by the fair residual (vision once → tip closed-loop).
    goal_tip_xy = tip0[:2] + residual
    print(
        f"[fair-place] servo dist={dist:.4f}m residual=({residual[0]:+.4f},{residual[1]:+.4f}) "
        f"src={src} band={band} max_near={max_near}",
        flush=True,
    )
    api._capx_fair_place_busy = True
    try:
        for i in range(_MAX_ITERS):
            tip, grip = _read_tip_and_gripper(api)
            if grip > _GRIPPER_CLOSED_MAX:
                print(f"[fair-place] gripper opened mid iter={i}; open", flush=True)
                return True
            err = goal_tip_xy - tip[:2]
            err_n = float(np.linalg.norm(err))
            if err_n < band * 0.5:
                print(
                    f"[fair-place] tip goal reached iter={i} err={err_n:.4f}m; open",
                    flush=True,
                )
                return True
            step = min(_STEP_M, err_n)
            step = max(step, _TIP_TOL_M)
            delta = err / err_n * step
            print(
                f"[fair-place] iter={i} tip_err={err_n:.4f}m "
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
                print(f"[fair-place] tip_xy failed ({exc!r}); open", flush=True)
                return True
            if abs(res.d_tip_z) > _Z_ABORT_M:
                print(
                    f"[fair-place] Z abort d_tip_z={res.d_tip_z:+.5f}; open",
                    flush=True,
                )
                return True
        print("[fair-place] max_iters; open", flush=True)
        return True
    finally:
        api._capx_fair_place_busy = False
