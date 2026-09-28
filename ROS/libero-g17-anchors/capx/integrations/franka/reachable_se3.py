"""Reachability recovery layered on the frozen v278 trusted-SE(3) primitive.

v278 guarantees that a bad IK branch is never executed.  v279 keeps that
transactional contract and adds completion behavior for two observed cases:

* a controller settles a few millimetres from an otherwise achieved pose;
* the straight simultaneous translation/rotation path reaches a joint limit.

For a large rotation, v279 preflights the complete straight path before any
command is sent.  A known dead end is replaced by a complete, preflighted
two-leg path through a neutral Panda configuration.  Every executed leg still
uses v278, so recovery cannot reintroduce an unseeded global IK jump.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from capx.integrations.franka.tip_xy_only import _live_actuated_cfg, fk_panda_hand
from capx.integrations.franka.trusted_se3 import (
    TrustedSe3Plan,
    TrustedSe3Result,
    move_hand_pose_trusted,
    plan_hand_pose_trusted,
)

_CONTROLLER_FLOOR_POSITION_M = 0.007
_CONTROLLER_FLOOR_ORIENTATION_RAD = 0.020
_TRANSIT_CLEARANCE_M = 0.10
_SECOND_TRANSIT_EXTRA_M = 0.08
_PREFERRED_TRANSIT_CEILING_M = 0.50
_FULL_PATH_PREFLIGHT_ORIENTATION_RAD = 0.60


def reachable_se3_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V279_REACHABLE_SE3", "0") == "1"


def reachable_se3_contract() -> dict[str, float | int | str]:
    return {
        "schema": "capx_libero_reachable_se3_contract_v1",
        "trusted_backend": "capx_libero_trusted_se3_contract_v1",
        "controller_floor_position_m": _CONTROLLER_FLOOR_POSITION_M,
        "controller_floor_orientation_rad": _CONTROLLER_FLOOR_ORIENTATION_RAD,
        "transit_clearance_m": _TRANSIT_CLEARANCE_M,
        "second_transit_extra_m": _SECOND_TRANSIT_EXTRA_M,
        "max_transit_routes": 2,
        "full_path_preflight_orientation_rad": _FULL_PATH_PREFLIGHT_ORIENTATION_RAD,
        "neutral_pose_candidates": 3,
    }


@dataclass(frozen=True)
class ReachableSe3Result:
    ok: bool
    status: str
    primary: TrustedSe3Result | None
    primary_plan: TrustedSe3Plan | None
    recovery_stages: tuple[tuple[str, TrustedSe3Result], ...]
    selected_route: str | None
    final_position_error_m: float
    final_orientation_error_rad: float
    message: str


def _neutral_cfg_candidates(api: Any, live_cfg: np.ndarray) -> tuple[np.ndarray, ...]:
    """Episode home plus two generic well-conditioned Panda configurations."""
    candidates: list[np.ndarray] = []
    home = getattr(getattr(api, "_env", None), "home_joint_position", None)
    if home is not None:
        raw = np.asarray(home, dtype=np.float64).reshape(-1)
        if raw.size >= 7 and np.all(np.isfinite(raw[:7])):
            cfg = live_cfg.copy()
            cfg[:7] = raw[:7]
            candidates.append(cfg)
    for arm in (
        np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
        np.array([0.0, -0.500, 0.0, -1.800, 0.0, 1.500, 0.785]),
    ):
        cfg = live_cfg.copy()
        cfg[:7] = arm
        if not any(np.allclose(cfg[:7], old[:7], atol=1e-6) for old in candidates):
            candidates.append(cfg)
    return tuple(candidates)


def _preflight_neutral_route(
    api: Any,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
    live_cfg: np.ndarray,
) -> tuple[str, np.ndarray, TrustedSe3Plan, TrustedSe3Plan] | None:
    """Find a complete two-leg path before committing its first waypoint."""
    feasible = []
    for index, neutral_cfg in enumerate(_neutral_cfg_candidates(api, live_cfg)):
        neutral_position, neutral_wxyz = fk_panda_hand(api, neutral_cfg)
        to_neutral = plan_hand_pose_trusted(
            api,
            neutral_position,
            neutral_wxyz,
            start_cfg=live_cfg,
        )
        if not to_neutral.ok:
            continue
        to_target = plan_hand_pose_trusted(
            api,
            target_position,
            target_wxyz,
            start_cfg=to_neutral.final_cfg,
        )
        if not to_target.ok:
            continue
        cost = len(to_neutral.waypoint_cfgs) + len(to_target.waypoint_cfgs)
        feasible.append(
            (
                cost,
                f"neutral_{index}",
                neutral_position,
                neutral_wxyz,
                to_neutral,
                to_target,
            )
        )
    if not feasible:
        return None
    _cost, name, position, wxyz, first, second = min(
        feasible, key=lambda row: row[0]
    )
    return name, np.concatenate([position, wxyz]), first, second


def _is_controller_floor(result: TrustedSe3Result) -> bool:
    return bool(
        result.status == "execution_nonprogress"
        and result.final_position_error_m <= _CONTROLLER_FLOOR_POSITION_M
        and result.final_orientation_error_rad <= _CONTROLLER_FLOOR_ORIENTATION_RAD
    )


def _transit_z(current_z: float, target_z: float, *, extra: float = 0.0) -> float:
    base = max(float(current_z), float(target_z))
    desired = base + _TRANSIT_CLEARANCE_M + float(extra)
    # Never lower a high target merely to obey a heuristic ceiling.
    return max(base, min(desired, _PREFERRED_TRANSIT_CEILING_M))


def _run_leg(
    api: Any,
    label: str,
    position: np.ndarray,
    wxyz: np.ndarray,
    *,
    log: bool,
) -> tuple[str, TrustedSe3Result, bool]:
    result = move_hand_pose_trusted(api, position, wxyz, log=log)
    accepted = result.ok or _is_controller_floor(result)
    if log and _is_controller_floor(result):
        print(
            f"[reachable-se3] accept controller floor stage={label} "
            f"position={result.final_position_error_m:.5f}m "
            f"orientation={result.final_orientation_error_rad:.5f}rad",
            flush=True,
        )
    return label, result, accepted


def _route_translate_then_rotate(
    api: Any,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
    *,
    extra_height: float,
    rotate_first: bool,
    route_name: str,
    log: bool,
) -> tuple[bool, list[tuple[str, TrustedSe3Result]]]:
    cfg = _live_actuated_cfg(api)
    current_position, current_wxyz = fk_panda_hand(api, cfg)
    z = _transit_z(current_position[2], target_position[2], extra=extra_height)
    current_high = current_position.copy()
    current_high[2] = z
    target_high = target_position.copy()
    target_high[2] = z
    if rotate_first:
        specs = (
            (f"{route_name}:lift", current_high, current_wxyz),
            (f"{route_name}:rotate", current_high, target_wxyz),
            (f"{route_name}:translate", target_high, target_wxyz),
            (f"{route_name}:goal", target_position, target_wxyz),
        )
    else:
        specs = (
            (f"{route_name}:lift", current_high, current_wxyz),
            (f"{route_name}:translate", target_high, current_wxyz),
            (f"{route_name}:rotate", target_high, target_wxyz),
            (f"{route_name}:goal", target_position, target_wxyz),
        )
    rows: list[tuple[str, TrustedSe3Result]] = []
    for label, position, wxyz in specs:
        stage, result, accepted = _run_leg(
            api, label, position, wxyz, log=log
        )
        rows.append((stage, result))
        if not accepted:
            return False, rows
    return True, rows


def move_hand_pose_reachable(
    api: Any,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
    *,
    log: bool = True,
) -> ReachableSe3Result:
    """Preflight large rotations, then use only v278-trusted execution legs."""
    target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
    target_wxyz = np.asarray(target_wxyz, dtype=np.float64).reshape(4)
    live_cfg = _live_actuated_cfg(api)
    _current_position, current_wxyz = fk_panda_hand(api, live_cfg)
    from capx.integrations.franka.tip_xy_only import _quaternion_angle_rad

    orientation_distance = _quaternion_angle_rad(current_wxyz, target_wxyz)
    primary_plan: TrustedSe3Plan | None = None
    if orientation_distance >= _FULL_PATH_PREFLIGHT_ORIENTATION_RAD:
        primary_plan = plan_hand_pose_trusted(
            api, target_position, target_wxyz, start_cfg=live_cfg
        )
        if log:
            print(
                f"[reachable-se3] straight preflight {primary_plan.message}",
                flush=True,
            )
        if not primary_plan.ok:
            neutral = _preflight_neutral_route(
                api, target_position, target_wxyz, live_cfg
            )
            if neutral is not None:
                route_name, packed_pose, first_plan, second_plan = neutral
                neutral_position = packed_pose[:3]
                neutral_wxyz = packed_pose[3:7]
                if log:
                    print(
                        f"[reachable-se3] selected preflight route={route_name} "
                        f"legs=({len(first_plan.waypoint_cfgs)},"
                        f"{len(second_plan.waypoint_cfgs)})",
                        flush=True,
                    )
                recovery_rows: list[tuple[str, TrustedSe3Result]] = []
                for label, position, wxyz in (
                    (f"{route_name}:neutral", neutral_position, neutral_wxyz),
                    (f"{route_name}:goal", target_position, target_wxyz),
                ):
                    stage, stage_result, accepted = _run_leg(
                        api, label, position, wxyz, log=log
                    )
                    recovery_rows.append((stage, stage_result))
                    if not accepted:
                        break
                if len(recovery_rows) == 2 and (
                    recovery_rows[-1][1].ok
                    or _is_controller_floor(recovery_rows[-1][1])
                ):
                    final = recovery_rows[-1][1]
                    message = (
                        f"status=recovered selected={route_name} "
                        f"preflight_legs=({len(first_plan.waypoint_cfgs)},"
                        f"{len(second_plan.waypoint_cfgs)}) final=("
                        f"{final.final_position_error_m:.5f}m,"
                        f"{final.final_orientation_error_rad:.5f}rad)"
                    )
                    result = ReachableSe3Result(
                        ok=True,
                        status="recovered",
                        primary=None,
                        primary_plan=primary_plan,
                        recovery_stages=tuple(recovery_rows),
                        selected_route=route_name,
                        final_position_error_m=final.final_position_error_m,
                        final_orientation_error_rad=final.final_orientation_error_rad,
                        message=message,
                    )
                    if log:
                        print(f"[reachable-se3] done ok=True {message}", flush=True)
                    return result
                final = recovery_rows[-1][1]
                message = (
                    f"status=recovery_execution_failed selected={route_name} "
                    f"stage={recovery_rows[-1][0]} final=("
                    f"{final.final_position_error_m:.5f}m,"
                    f"{final.final_orientation_error_rad:.5f}rad)"
                )
                result = ReachableSe3Result(
                    ok=False,
                    status="recovery_execution_failed",
                    primary=None,
                    primary_plan=primary_plan,
                    recovery_stages=tuple(recovery_rows),
                    selected_route=route_name,
                    final_position_error_m=final.final_position_error_m,
                    final_orientation_error_rad=final.final_orientation_error_rad,
                    message=message,
                )
                if log:
                    print(f"[reachable-se3] done ok=False {message}", flush=True)
                return result
            else:
                message = (
                    f"status=preflight_unreachable; straight={primary_plan.message}; "
                    "neutral_routes=exhausted; executed=0"
                )
                result = ReachableSe3Result(
                    ok=False,
                    status="preflight_unreachable",
                    primary=None,
                    primary_plan=primary_plan,
                    recovery_stages=(),
                    selected_route=None,
                    final_position_error_m=primary_plan.final_position_error_m,
                    final_orientation_error_rad=primary_plan.final_orientation_error_rad,
                    message=message,
                )
                if log:
                    print(f"[reachable-se3] done ok=False {message}", flush=True)
                return result

    primary = move_hand_pose_trusted(api, target_position, target_wxyz, log=log)
    if primary.ok or _is_controller_floor(primary):
        status = "completed" if primary.ok else "controller_floor_accepted"
        result = ReachableSe3Result(
            ok=True,
            status=status,
            primary=primary,
            primary_plan=primary_plan,
            recovery_stages=(),
            selected_route=None,
            final_position_error_m=primary.final_position_error_m,
            final_orientation_error_rad=primary.final_orientation_error_rad,
            message=f"status={status}; primary={primary.message}",
        )
        if log:
            print(f"[reachable-se3] done ok=True {result.message}", flush=True)
        return result

    all_stages: list[tuple[str, TrustedSe3Result]] = []
    routes = (
        ("transit_a", 0.0, False),
        ("transit_b", _SECOND_TRANSIT_EXTRA_M, True),
    )
    selected: str | None = None
    for route_name, extra_height, rotate_first in routes:
        if log:
            print(
                f"[reachable-se3] recovery start route={route_name} "
                f"primary_status={primary.status}",
                flush=True,
            )
        route_ok, rows = _route_translate_then_rotate(
            api,
            target_position,
            target_wxyz,
            extra_height=extra_height,
            rotate_first=rotate_first,
            route_name=route_name,
            log=log,
        )
        all_stages.extend(rows)
        if route_ok:
            selected = route_name
            break

    if all_stages:
        final = all_stages[-1][1]
        final_position_error = final.final_position_error_m
        final_orientation_error = final.final_orientation_error_rad
    else:
        final = primary
        final_position_error = primary.final_position_error_m
        final_orientation_error = primary.final_orientation_error_rad
    ok = selected is not None
    status = "recovered" if ok else "recovery_exhausted"
    message = (
        f"status={status} primary={primary.status} selected={selected} "
        f"stages={len(all_stages)} final=({final_position_error:.5f}m,"
        f"{final_orientation_error:.5f}rad)"
    )
    result = ReachableSe3Result(
        ok=ok,
        status=status,
        primary=primary,
        primary_plan=primary_plan,
        recovery_stages=tuple(all_stages),
        selected_route=selected,
        final_position_error_m=final_position_error,
        final_orientation_error_rad=final_orientation_error,
        message=message,
    )
    if log:
        print(f"[reachable-se3] done ok={ok} {message}", flush=True)
    return result
