"""Command-faithful SE(3) execution built on v278's full-path IK proof.

The v279 failure was a representation mismatch: it proved a route through a
sequence of joint configurations, but executed only the Cartesian pose of the
neutral endpoint and solved IK again.  Panda redundancy means those operations
are not equivalent.  This module executes the proved joint homotopy itself.
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
from capx.integrations.franka.trusted_se3 import (
    TrustedSe3Plan,
    TrustedSe3Result,
    plan_hand_pose_trusted,
)

_MAX_COMMANDED_ARM_DELTA_RAD = 0.18
_MAX_MEASURED_ARM_DELTA_RAD = 0.2002
_GOAL_POSITION_TOL_M = 0.007
_GOAL_ORIENTATION_TOL_RAD = 0.030
_MAX_CONTROLLER_STEPS = 40
_MAX_SETTLE_RETRIES = 2
_MIN_COMMAND_BUDGET = 64


def planned_se3_enabled() -> bool:
    return bool(
        os.environ.get("CAPX_LIBERO_G17_V280_PLANNED_SE3", "0") == "1"
        or command_faithful_join_enabled()
    )


def command_faithful_join_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V281_JOINT_JOIN", "0") == "1"


def planned_se3_contract() -> dict[str, float | int | str]:
    return {
        "schema": "capx_libero_planned_se3_contract_v1",
        "planner": "capx_libero_trusted_se3_contract_v1",
        "execution_representation": "joint_waypoints",
        "max_commanded_arm_delta_rad": _MAX_COMMANDED_ARM_DELTA_RAD,
        "max_measured_arm_delta_rad": _MAX_MEASURED_ARM_DELTA_RAD,
        "goal_position_tol_m": _GOAL_POSITION_TOL_M,
        "goal_orientation_tol_rad": _GOAL_ORIENTATION_TOL_RAD,
        "measured_replan_after_neutral": 1,
        "rollback_on_second_leg_rejection": 1,
    }


def command_faithful_join_contract() -> dict[str, float | int | str]:
    contract = planned_se3_contract()
    contract.update(
        {
            "schema": "capx_libero_planned_se3_contract_v2",
            "neutral_second_leg": "preflighted_joint_waypoints",
            "measured_state_role": "bounded_joint_join_not_ik_resolve",
        }
    )
    return contract


@dataclass(frozen=True)
class PlannedSe3Result:
    ok: bool
    status: str
    direct_plan: TrustedSe3Plan
    selected_route: str | None
    stages: tuple[tuple[str, TrustedSe3Result], ...]
    rollback_performed: bool
    final_position_error_m: float
    final_orientation_error_rad: float
    message: str


def _neutral_cfg_candidates(live_cfg: np.ndarray) -> tuple[np.ndarray, ...]:
    rows = []
    for arm in (
        np.array([0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, np.pi / 4]),
        np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
        np.array([0.0, -0.500, 0.0, -1.800, 0.0, 1.500, 0.785]),
    ):
        cfg = live_cfg.copy()
        cfg[:7] = arm
        rows.append(cfg)
    return tuple(rows)


def _trusted_result(
    api: Any,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
    *,
    status: str,
    executed: int,
    max_commanded_dq: float,
    max_measured_dq: float,
    detail: str,
) -> TrustedSe3Result:
    cfg = _live_actuated_cfg(api)
    position, wxyz = fk_panda_hand(api, cfg)
    pos_err = float(np.linalg.norm(position - target_position))
    ori_err = _quaternion_angle_rad(wxyz, target_wxyz)
    ok = bool(
        status == "completed"
        and pos_err <= _GOAL_POSITION_TOL_M
        and ori_err <= _GOAL_ORIENTATION_TOL_RAD
    )
    if status == "completed" and not ok:
        status = "execution_residual"
    message = (
        f"status={status} executed={executed} final=({pos_err:.5f}m,"
        f"{ori_err:.5f}rad) max_commanded_dq={max_commanded_dq:.5f} "
        f"max_measured_dq={max_measured_dq:.5f} detail={detail}"
    )
    return TrustedSe3Result(
        ok=ok,
        status=status,
        target_position=tuple(float(x) for x in target_position),
        achieved_position=tuple(float(x) for x in position),
        final_position_error_m=pos_err,
        final_orientation_error_rad=ori_err,
        executed_waypoints=executed,
        preflight_rejections=0,
        max_candidate_arm_joint_delta_rad=max_commanded_dq,
        max_executed_arm_joint_delta_rad=max_measured_dq,
        nonprogress_events=0,
        message=message,
    )


def execute_trusted_plan(
    api: Any,
    plan: TrustedSe3Plan,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
    *,
    log: bool = True,
) -> TrustedSe3Result:
    """Execute the proved joint homotopy without invoking IK again."""
    target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
    target_wxyz = np.asarray(target_wxyz, dtype=np.float64).reshape(4)
    if not plan.ok:
        return _trusted_result(
            api,
            target_position,
            target_wxyz,
            status="plan_rejected",
            executed=0,
            max_commanded_dq=0.0,
            max_measured_dq=0.0,
            detail=plan.message,
        )

    executed = 0
    max_commanded = 0.0
    max_measured = 0.0
    detail = ""
    status = "completed"
    for waypoint_index, waypoint in enumerate(plan.waypoint_cfgs, start=1):
        goal = np.asarray(waypoint, dtype=np.float64).reshape(-1)
        while True:
            live = _live_actuated_cfg(api)
            delta = goal[:7] - live[:7]
            remaining = float(np.max(np.abs(delta)))
            if remaining <= 1e-5:
                break
            fraction = min(1.0, _MAX_COMMANDED_ARM_DELTA_RAD / remaining)
            command = live[:7] + fraction * delta
            commanded_dq = float(np.max(np.abs(command - live[:7])))
            api._env.move_to_joints_blocking(
                command, max_steps=_MAX_CONTROLLER_STEPS
            )
            after = _live_actuated_cfg(api)
            measured_dq = float(np.max(np.abs(after[:7] - live[:7])))
            executed += 1
            max_commanded = max(max_commanded, commanded_dq)
            max_measured = max(max_measured, measured_dq)
            if log:
                print(
                    f"[planned-se3] waypoint={waypoint_index}/"
                    f"{len(plan.waypoint_cfgs)} command={executed} "
                    f"dq_cmd={commanded_dq:.5f} dq_measured={measured_dq:.5f}",
                    flush=True,
                )
            if measured_dq > _MAX_MEASURED_ARM_DELTA_RAD:
                status = "measured_joint_jump"
                detail = f"measured_dq={measured_dq:.6f}"
                break
            # One bounded command per proved waypoint is sufficient.  Any
            # measured residual is absorbed by the next bounded command; only
            # the terminal waypoint receives explicit settle retries below.
            if fraction >= 1.0:
                break
            if executed > max(
                _MIN_COMMAND_BUDGET, 4 * max(1, len(plan.waypoint_cfgs))
            ):
                status = "controller_nonconvergence"
                detail = "joint waypoint command budget exhausted"
                break
        if status != "completed":
            break

    # Give the position controller a bounded chance to settle on the final
    # proved configuration; this does not change the joint homotopy.
    if status == "completed" and plan.waypoint_cfgs:
        final_goal = np.asarray(plan.waypoint_cfgs[-1], dtype=np.float64)
        for _ in range(_MAX_SETTLE_RETRIES):
            live = _live_actuated_cfg(api)
            if float(np.max(np.abs(final_goal[:7] - live[:7]))) <= 1e-4:
                break
            command = live[:7] + np.clip(
                final_goal[:7] - live[:7],
                -_MAX_COMMANDED_ARM_DELTA_RAD,
                _MAX_COMMANDED_ARM_DELTA_RAD,
            )
            commanded_dq = float(np.max(np.abs(command - live[:7])))
            api._env.move_to_joints_blocking(command, max_steps=_MAX_CONTROLLER_STEPS)
            after = _live_actuated_cfg(api)
            measured_dq = float(np.max(np.abs(after[:7] - live[:7])))
            executed += 1
            max_commanded = max(max_commanded, commanded_dq)
            max_measured = max(max_measured, measured_dq)
            if measured_dq > _MAX_MEASURED_ARM_DELTA_RAD:
                status = "measured_joint_jump"
                detail = f"settle measured_dq={measured_dq:.6f}"
                break

    result = _trusted_result(
        api,
        target_position,
        target_wxyz,
        status=status,
        executed=executed,
        max_commanded_dq=max_commanded,
        max_measured_dq=max_measured,
        detail=detail,
    )
    if log:
        print(f"[planned-se3] done ok={result.ok} {result.message}", flush=True)
    return result


def _rollback_cfg(api: Any, cfg: np.ndarray, *, log: bool) -> None:
    """Return to the exact pre-route joint branch with bounded joint commands."""
    live = _live_actuated_cfg(api)
    target_position, target_wxyz = fk_panda_hand(api, cfg)
    rollback = TrustedSe3Plan(
        ok=True,
        status="completed",
        start_cfg=live.copy(),
        final_cfg=cfg.copy(),
        waypoint_cfgs=(cfg.copy(),),
        preflight_rejections=0,
        max_arm_joint_delta_rad=float(np.max(np.abs(cfg[:7] - live[:7]))),
        final_position_error_m=0.0,
        final_orientation_error_rad=0.0,
        message="explicit bounded joint rollback",
    )
    execute_trusted_plan(
        api, rollback, target_position, target_wxyz, log=log
    )


def move_hand_pose_planned(
    api: Any,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
    *,
    log: bool = True,
) -> PlannedSe3Result:
    """Complete a command if fully proved, otherwise leave the arm unchanged."""
    target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
    target_wxyz = np.asarray(target_wxyz, dtype=np.float64).reshape(4)
    original_cfg = _live_actuated_cfg(api)
    direct = plan_hand_pose_trusted(
        api, target_position, target_wxyz, start_cfg=original_cfg
    )
    if direct.ok:
        stage = execute_trusted_plan(
            api, direct, target_position, target_wxyz, log=log
        )
        return PlannedSe3Result(
            ok=stage.ok,
            status=stage.status,
            direct_plan=direct,
            selected_route="direct",
            stages=(("direct", stage),),
            rollback_performed=False,
            final_position_error_m=stage.final_position_error_m,
            final_orientation_error_rad=stage.final_orientation_error_rad,
            message=f"route=direct {stage.message}",
        )

    attempts: list[tuple[str, TrustedSe3Result]] = []
    rollback_performed = False
    for index, neutral_cfg in enumerate(_neutral_cfg_candidates(original_cfg)):
        neutral_position, neutral_wxyz = fk_panda_hand(api, neutral_cfg)
        to_neutral = plan_hand_pose_trusted(
            api, neutral_position, neutral_wxyz, start_cfg=original_cfg
        )
        if not to_neutral.ok:
            continue
        ideal_second = plan_hand_pose_trusted(
            api, target_position, target_wxyz, start_cfg=to_neutral.final_cfg
        )
        if not ideal_second.ok:
            continue
        label = f"neutral_{index}"
        first = execute_trusted_plan(
            api, to_neutral, neutral_position, neutral_wxyz, log=log
        )
        attempts.append((f"{label}:neutral", first))
        if first.ok:
            measured = _live_actuated_cfg(api)
            measured_second = plan_hand_pose_trusted(
                api, target_position, target_wxyz, start_cfg=measured
            )
            if command_faithful_join_enabled():
                # v281: the second-leg joint homotopy was already proved from
                # the first plan's exact endpoint.  Controller residual changes
                # only the bounded join into its first waypoint; re-solving IK
                # here can create a false joint-limit rejection.
                second = execute_trusted_plan(
                    api, ideal_second, target_position, target_wxyz, log=log
                )
                attempts.append((f"{label}:goal_joint_join", second))
                if second.ok:
                    return PlannedSe3Result(
                        ok=True,
                        status="recovered",
                        direct_plan=direct,
                        selected_route=label,
                        stages=tuple(attempts),
                        rollback_performed=rollback_performed,
                        final_position_error_m=second.final_position_error_m,
                        final_orientation_error_rad=second.final_orientation_error_rad,
                        message=(
                            f"route={label} joint_join=accepted "
                            f"measured_replan={measured_second.status} "
                            f"{second.message}"
                        ),
                    )
            elif measured_second.ok:
                second = execute_trusted_plan(
                    api, measured_second, target_position, target_wxyz, log=log
                )
                attempts.append((f"{label}:goal", second))
                if second.ok:
                    return PlannedSe3Result(
                        ok=True,
                        status="recovered",
                        direct_plan=direct,
                        selected_route=label,
                        stages=tuple(attempts),
                        rollback_performed=rollback_performed,
                        final_position_error_m=second.final_position_error_m,
                        final_orientation_error_rad=second.final_orientation_error_rad,
                        message=(
                            f"route={label} measured_replan=accepted "
                            f"{second.message}"
                        ),
                    )
        _rollback_cfg(api, original_cfg, log=log)
        rollback_performed = True

    final_cfg = _live_actuated_cfg(api)
    final_position, final_wxyz = fk_panda_hand(api, final_cfg)
    pos_err = float(np.linalg.norm(final_position - target_position))
    ori_err = _quaternion_angle_rad(final_wxyz, target_wxyz)
    return PlannedSe3Result(
        ok=False,
        status="preflight_unreachable",
        direct_plan=direct,
        selected_route=None,
        stages=tuple(attempts),
        rollback_performed=rollback_performed,
        final_position_error_m=pos_err,
        final_orientation_error_rad=ori_err,
        message=(
            f"status=preflight_unreachable attempts={len(attempts)} "
            f"rollback={rollback_performed} final=({pos_err:.5f}m,"
            f"{ori_err:.5f}rad)"
        ),
    )
