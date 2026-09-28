"""Live-seeded, branch-continuous Cartesian motion for LIBERO Franka.

This is an execution backend, not a planner and not a new source of task truth.
It preserves the hand-pose target produced by ``goto_pose`` but reaches it as a
closed-loop sequence of small SE(3) waypoints.  Every waypoint is solved inside
a hard joint-space trust region around the measured arm configuration and is
FK-checked before it may be executed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from capx.integrations.franka.tip_xy_only import (
    _live_actuated_cfg,
    _quaternion_angle_rad,
    fk_panda_hand,
)

_MAX_TRANSLATION_STEP_M = 0.020
_MAX_ROTATION_STEP_RAD = 0.120
_MAX_ARM_JOINT_DELTA_RAD = 0.200
_FK_POSITION_TOL_M = 0.004
_FK_ORIENTATION_TOL_RAD = 0.050
_GOAL_POSITION_TOL_M = 0.003
_GOAL_ORIENTATION_TOL_RAD = 0.030
_MIN_WAYPOINT_FRACTION = 0.125
_MAX_PREFLIGHT_BACKOFFS = 4
_MAX_EXECUTED_WAYPOINTS = 96
_MAX_NONPROGRESS = 3
_MAX_CONTROLLER_STEPS = 40


def trusted_se3_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V278_TRUSTED_SE3", "0") == "1"


def trusted_se3_contract() -> dict[str, float | int | str]:
    return {
        "schema": "capx_libero_trusted_se3_contract_v1",
        "max_translation_step_m": _MAX_TRANSLATION_STEP_M,
        "max_rotation_step_rad": _MAX_ROTATION_STEP_RAD,
        "max_arm_joint_delta_rad": _MAX_ARM_JOINT_DELTA_RAD,
        "candidate_fk_position_tol_m": _FK_POSITION_TOL_M,
        "candidate_fk_orientation_tol_rad": _FK_ORIENTATION_TOL_RAD,
        "goal_position_tol_m": _GOAL_POSITION_TOL_M,
        "goal_orientation_tol_rad": _GOAL_ORIENTATION_TOL_RAD,
        "max_preflight_backoffs": _MAX_PREFLIGHT_BACKOFFS,
        "max_executed_waypoints": _MAX_EXECUTED_WAYPOINTS,
        "max_controller_steps": _MAX_CONTROLLER_STEPS,
    }


@dataclass(frozen=True)
class TrustedSe3Result:
    ok: bool
    status: str
    target_position: tuple[float, float, float]
    achieved_position: tuple[float, float, float]
    final_position_error_m: float
    final_orientation_error_rad: float
    executed_waypoints: int
    preflight_rejections: int
    max_candidate_arm_joint_delta_rad: float
    max_executed_arm_joint_delta_rad: float
    nonprogress_events: int
    message: str


@dataclass(frozen=True)
class TrustedSe3Plan:
    """Non-mutating proof that a complete local-IK path exists."""

    ok: bool
    status: str
    start_cfg: np.ndarray
    final_cfg: np.ndarray
    waypoint_cfgs: tuple[np.ndarray, ...]
    preflight_rejections: int
    max_arm_joint_delta_rad: float
    final_position_error_m: float
    final_orientation_error_rad: float
    message: str


def _unit_quaternion(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(out))
    if not np.all(np.isfinite(out)) or norm < 1e-8:
        raise ValueError("target quaternion is nonfinite or zero")
    return out / norm


def _slerp_shortest(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """WXYZ SLERP with antipodal canonicalization."""
    a = _unit_quaternion(q0)
    b = _unit_quaternion(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    t = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        return _unit_quaternion((1.0 - t) * a + t * b)
    theta = float(np.arccos(dot))
    denom = float(np.sin(theta))
    return _unit_quaternion(
        np.sin((1.0 - t) * theta) / denom * a
        + np.sin(t * theta) / denom * b
    )


def _waypoint_fraction(
    position_error_m: float, orientation_error_rad: float
) -> float:
    fractions = [1.0]
    if position_error_m > _MAX_TRANSLATION_STEP_M:
        fractions.append(_MAX_TRANSLATION_STEP_M / position_error_m)
    if orientation_error_rad > _MAX_ROTATION_STEP_RAD:
        fractions.append(_MAX_ROTATION_STEP_RAD / orientation_error_rad)
    return float(min(fractions))


def _candidate_reason(
    api: Any,
    live_cfg: np.ndarray,
    candidate: np.ndarray,
    waypoint_position: np.ndarray,
    waypoint_wxyz: np.ndarray,
) -> tuple[str | None, float, float, float]:
    if candidate.shape != live_cfg.shape or not np.all(np.isfinite(candidate)):
        return "candidate_nonfinite_or_wrong_shape", float("inf"), float("inf"), float("inf")
    arm_n = min(7, live_cfg.size)
    max_dq = float(np.max(np.abs(candidate[:arm_n] - live_cfg[:arm_n])))
    predicted_position, predicted_wxyz = fk_panda_hand(api, candidate)
    position_error = float(np.linalg.norm(predicted_position - waypoint_position))
    orientation_error = _quaternion_angle_rad(predicted_wxyz, waypoint_wxyz)
    reasons: list[str] = []
    lower = np.asarray(getattr(api._robot.joints, "lower_limits", []), dtype=np.float64).reshape(-1)
    upper = np.asarray(getattr(api._robot.joints, "upper_limits", []), dtype=np.float64).reshape(-1)
    if lower.size == candidate.size and upper.size == candidate.size:
        if np.any(candidate < lower - 1e-8) or np.any(candidate > upper + 1e-8):
            reasons.append("joint_limit")
    # The constrained optimizer can miss the boundary by a few solver epsilons.
    if max_dq > _MAX_ARM_JOINT_DELTA_RAD + 2e-4:
        reasons.append(f"joint_jump:{max_dq:.5f}")
    if position_error > _FK_POSITION_TOL_M:
        reasons.append(f"fk_position:{position_error:.5f}")
    if orientation_error > _FK_ORIENTATION_TOL_RAD:
        reasons.append(f"fk_orientation:{orientation_error:.5f}")
    return (",".join(reasons) if reasons else None), max_dq, position_error, orientation_error


def plan_hand_pose_trusted(
    api: Any,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
    *,
    start_cfg: np.ndarray | None = None,
) -> TrustedSe3Plan:
    """Solve and validate a complete path without executing any candidate.

    v278 validates one step before committing it.  v279 uses this full-path
    preflight to avoid entering a continuous branch that is known to dead-end
    at a joint limit.
    """
    target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(target_position)):
        raise ValueError("target position is nonfinite")
    target_wxyz = _unit_quaternion(target_wxyz)
    cfg0 = (
        _live_actuated_cfg(api)
        if start_cfg is None
        else np.asarray(start_cfg, dtype=np.float64).reshape(-1).copy()
    )
    cfg = cfg0.copy()
    rows: list[np.ndarray] = []
    rejected = 0
    max_dq_seen = 0.0
    status = "budget_exhausted"
    detail = "waypoint budget exhausted"

    for _ in range(_MAX_EXECUTED_WAYPOINTS + 1):
        position, wxyz = fk_panda_hand(api, cfg)
        pos_err = float(np.linalg.norm(target_position - position))
        ori_err = _quaternion_angle_rad(target_wxyz, wxyz)
        if pos_err <= _GOAL_POSITION_TOL_M and ori_err <= _GOAL_ORIENTATION_TOL_RAD:
            status = "completed"
            detail = ""
            break
        if len(rows) >= _MAX_EXECUTED_WAYPOINTS:
            break
        base_fraction = _waypoint_fraction(pos_err, ori_err)
        accepted: np.ndarray | None = None
        last_reason = "not_solved"
        for backoff in range(_MAX_PREFLIGHT_BACKOFFS + 1):
            scale = 0.5**backoff
            if scale < _MIN_WAYPOINT_FRACTION:
                break
            fraction = base_fraction * scale
            waypoint_position = position + fraction * (target_position - position)
            waypoint_wxyz = _slerp_shortest(wxyz, target_wxyz, fraction)
            try:
                candidate = np.asarray(
                    api._pks.solve_ik_local(
                        robot=api._robot,
                        target_link_name=api._target_link_name,
                        target_position=waypoint_position,
                        target_wxyz=waypoint_wxyz,
                        prev_cfg=cfg,
                        max_arm_delta=_MAX_ARM_JOINT_DELTA_RAD,
                    ),
                    dtype=np.float64,
                ).reshape(-1)
                reason, max_dq, _fk_p, _fk_r = _candidate_reason(
                    api, cfg, candidate, waypoint_position, waypoint_wxyz
                )
            except Exception as exc:
                max_dq = float("inf")
                reason = f"ik_or_fk_exception:{exc!r}"
            max_dq_seen = max(max_dq_seen, max_dq)
            if reason is None:
                accepted = candidate
                break
            rejected += 1
            last_reason = reason
        if accepted is None:
            status = "local_unreachable"
            detail = last_reason
            break
        cfg = accepted
        rows.append(cfg.copy())

    final_position, final_wxyz = fk_panda_hand(api, cfg)
    final_pos_err = float(np.linalg.norm(target_position - final_position))
    final_ori_err = _quaternion_angle_rad(target_wxyz, final_wxyz)
    ok = status == "completed"
    message = (
        f"status={status} waypoints={len(rows)} rejected={rejected} "
        f"final=({final_pos_err:.5f}m,{final_ori_err:.5f}rad) "
        f"max_dq={max_dq_seen:.5f} detail={detail}"
    )
    return TrustedSe3Plan(
        ok=ok,
        status=status,
        start_cfg=cfg0,
        final_cfg=cfg.copy(),
        waypoint_cfgs=tuple(rows),
        preflight_rejections=rejected,
        max_arm_joint_delta_rad=max_dq_seen,
        final_position_error_m=final_pos_err,
        final_orientation_error_rad=final_ori_err,
        message=message,
    )


def move_hand_pose_trusted(
    api: Any,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
    *,
    log: bool = True,
) -> TrustedSe3Result:
    """Reach one absolute panda-hand pose without executing an IK branch jump."""
    target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(target_position)):
        raise ValueError("target position is nonfinite")
    target_wxyz = _unit_quaternion(target_wxyz)
    executed = 0
    rejected = 0
    nonprogress = 0
    max_candidate_dq = 0.0
    max_executed_dq = 0.0
    status = "budget_exhausted"
    detail = ""

    for _ in range(_MAX_EXECUTED_WAYPOINTS + 1):
        live_cfg = _live_actuated_cfg(api)
        current_position, current_wxyz = fk_panda_hand(api, live_cfg)
        position_error = float(np.linalg.norm(target_position - current_position))
        orientation_error = _quaternion_angle_rad(target_wxyz, current_wxyz)
        if (
            position_error <= _GOAL_POSITION_TOL_M
            and orientation_error <= _GOAL_ORIENTATION_TOL_RAD
        ):
            status = "completed"
            break
        if executed >= _MAX_EXECUTED_WAYPOINTS:
            detail = "waypoint budget exhausted"
            break

        base_fraction = _waypoint_fraction(position_error, orientation_error)
        accepted: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
        last_reason = "not_solved"
        for backoff in range(_MAX_PREFLIGHT_BACKOFFS + 1):
            scale = 0.5**backoff
            fraction = base_fraction * scale
            if scale < _MIN_WAYPOINT_FRACTION:
                break
            waypoint_position = current_position + fraction * (
                target_position - current_position
            )
            waypoint_wxyz = _slerp_shortest(current_wxyz, target_wxyz, fraction)
            try:
                candidate = np.asarray(
                    api._pks.solve_ik_local(
                        robot=api._robot,
                        target_link_name=api._target_link_name,
                        target_position=waypoint_position,
                        target_wxyz=waypoint_wxyz,
                        prev_cfg=live_cfg,
                        max_arm_delta=_MAX_ARM_JOINT_DELTA_RAD,
                    ),
                    dtype=np.float64,
                ).reshape(-1)
                reason, max_dq, _fk_p, _fk_r = _candidate_reason(
                    api,
                    live_cfg,
                    candidate,
                    waypoint_position,
                    waypoint_wxyz,
                )
            except Exception as exc:
                candidate = np.empty(0)
                max_dq = float("inf")
                reason = f"ik_or_fk_exception:{exc!r}"
            max_candidate_dq = max(max_candidate_dq, max_dq)
            if reason is None:
                accepted = (candidate, waypoint_position, waypoint_wxyz, fraction)
                break
            rejected += 1
            last_reason = reason
            if log:
                print(
                    f"[trusted-se3] preflight reject backoff={backoff} "
                    f"fraction={fraction:.6f} reason={reason}",
                    flush=True,
                )
        if accepted is None:
            status = "local_unreachable"
            detail = last_reason
            break

        candidate, waypoint_position, _waypoint_wxyz, fraction = accepted
        arm_n = min(7, live_cfg.size)
        api._env.move_to_joints_blocking(
            candidate[:arm_n], max_steps=_MAX_CONTROLLER_STEPS
        )
        after_cfg = _live_actuated_cfg(api)
        api.cfg = after_cfg.copy()
        after_position, after_wxyz = fk_panda_hand(api, after_cfg)
        executed += 1
        actual_dq = float(np.max(np.abs(after_cfg[:arm_n] - live_cfg[:arm_n])))
        max_executed_dq = max(max_executed_dq, actual_dq)
        next_position_error = float(np.linalg.norm(target_position - after_position))
        next_orientation_error = _quaternion_angle_rad(target_wxyz, after_wxyz)
        made_progress = (
            next_position_error < position_error - 1e-4
            or position_error <= _GOAL_POSITION_TOL_M
        ) and (
            next_orientation_error < orientation_error - 1e-4
            or orientation_error <= _GOAL_ORIENTATION_TOL_RAD
        )
        if not made_progress:
            nonprogress += 1
        else:
            nonprogress = 0
        if log:
            direction = target_position - current_position
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            signed_progress = float(np.dot(after_position - current_position, direction))
            lateral = float(
                np.linalg.norm(
                    (after_position - current_position) - signed_progress * direction
                )
            )
            print(
                f"[trusted-se3] waypoint={executed} fraction={fraction:.6f} "
                f"dq={actual_dq:.5f} progress={signed_progress:+.5f} "
                f"lateral={lateral:.5f} remaining=({next_position_error:.5f}m," 
                f"{next_orientation_error:.5f}rad)",
                flush=True,
            )
        if nonprogress >= _MAX_NONPROGRESS:
            status = "execution_nonprogress"
            detail = "three consecutive measured SE(3) nonprogress events"
            break

    final_cfg = _live_actuated_cfg(api)
    final_position, final_wxyz = fk_panda_hand(api, final_cfg)
    final_position_error = float(np.linalg.norm(target_position - final_position))
    final_orientation_error = _quaternion_angle_rad(target_wxyz, final_wxyz)
    ok = status == "completed"
    message = (
        f"status={status} executed={executed} rejected={rejected} "
        f"final=({final_position_error:.5f}m,{final_orientation_error:.5f}rad) "
        f"max_candidate_dq={max_candidate_dq:.5f} "
        f"max_executed_dq={max_executed_dq:.5f} detail={detail}"
    )
    result = TrustedSe3Result(
        ok=ok,
        status=status,
        target_position=tuple(float(x) for x in target_position),
        achieved_position=tuple(float(x) for x in final_position),
        final_position_error_m=final_position_error,
        final_orientation_error_rad=final_orientation_error,
        executed_waypoints=executed,
        preflight_rejections=rejected,
        max_candidate_arm_joint_delta_rad=max_candidate_dq,
        max_executed_arm_joint_delta_rad=max_executed_dq,
        nonprogress_events=nonprogress,
        message=message,
    )
    if log:
        print(f"[trusted-se3] done ok={ok} {message}", flush=True)
    return result
