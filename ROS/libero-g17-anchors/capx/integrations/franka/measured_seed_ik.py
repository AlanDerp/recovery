"""Native-speed IK with the optimizer anchored to the measured joint branch.

This fixes a state bug in the legacy LIBERO ``goto_pose`` path: the first IK
solve was unseeded, and later solves were referenced to the previous requested
IK configuration rather than the configuration actually reached by the
controller.  Panda redundancy makes either choice capable of selecting a
different branch.

Unlike the v278-v281 executors, this module does not add Cartesian waypoints,
joint-step gates, neutral routes, rollback, or terminal-pose acceptance gates.
Each requested stage is solved once and executed through the native blocking
controller once.
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


NATIVE_CONTROLLER_MAX_STEPS = 480
# Backward-compatible private alias for existing audit/tests.
_NATIVE_CONTROLLER_MAX_STEPS = NATIVE_CONTROLLER_MAX_STEPS
_ACCEPT_POSITION_M = 0.005
_ACCEPT_ORIENTATION_RAD = 0.030


def measured_seed_ik_enabled() -> bool:
    return bool(
        os.environ.get("CAPX_LIBERO_G17_V282_MEASURED_SEED_IK", "0") == "1"
        or tool_pi_symmetry_enabled()
    )


def tool_pi_symmetry_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY", "0") == "1"


def _quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64).reshape(4)
    out = np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    return out / np.linalg.norm(out)


def measured_seed_ik_contract() -> dict[str, int | str]:
    return {
        "schema": "capx_libero_measured_seed_ik_contract_v1",
        "ik_initialization": "measured_joint_state",
        "ik_fallback": "deterministic_multistart_nearest_pose_solution",
        "post_execution_cache": "measured_joint_state",
        "native_blocking_commands_per_stage": 1,
        "native_controller_max_steps": _NATIVE_CONTROLLER_MAX_STEPS,
        "cartesian_waypoints": 0,
        "joint_delta_gate": 0,
        "terminal_residual_gate": 0,
        "terminal_residual_report": 1,
        "neutral_route": 0,
        "rollback": 0,
        "tool_z_pi_symmetry": int(tool_pi_symmetry_enabled()),
    }


@dataclass(frozen=True)
class MeasuredSeedIkResult:
    ok: bool
    status: str
    candidate_arm_delta_rad: float
    measured_arm_delta_rad: float
    final_position_error_m: float
    final_orientation_error_rad: float
    execution_commands: int
    selected_seed: str
    candidate_count: int
    selected_orientation: str
    message: str
    pt_swept_path: dict[str, Any] | None = None


def _validate_candidate(api: Any, candidate: np.ndarray) -> None:
    n = int(api._robot.joints.num_actuated_joints)
    if candidate.shape != (n,) or not np.all(np.isfinite(candidate)):
        raise RuntimeError(
            f"measured-seed IK returned invalid cfg shape/value: {candidate.shape}"
        )
    lower = np.asarray(
        getattr(api._robot.joints, "lower_limits", []), dtype=np.float64
    ).reshape(-1)
    upper = np.asarray(
        getattr(api._robot.joints, "upper_limits", []), dtype=np.float64
    ).reshape(-1)
    if lower.size == n and upper.size == n:
        # jaxls' hard constraint can land a few float32 solver epsilons outside
        # the reported boundary.  This is numerical noise, not an invalid IK
        # branch, and must not become a new motion gate.
        if np.any(candidate < lower - 2e-4) or np.any(candidate > upper + 2e-4):
            raise RuntimeError("measured-seed IK returned a joint-limit violation")


def _fallback_seeds(measured_cfg: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
    rows = []
    for name, arm in (
        (
            "reset",
            np.array(
                [0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, np.pi / 4]
            ),
        ),
        ("neutral_1", np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])),
        ("neutral_2", np.array([0.0, -0.500, 0.0, -1.800, 0.0, 1.500, 0.785])),
    ):
        seed = measured_cfg.copy()
        seed[:7] = arm
        rows.append((name, seed))
    return tuple(rows)


def _candidate_metrics(
    api: Any,
    candidate: np.ndarray,
    measured_cfg: np.ndarray,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
) -> tuple[float, float, float, float]:
    position, wxyz = fk_panda_hand(api, candidate)
    position_error = float(np.linalg.norm(position - target_position))
    orientation_error = _quaternion_angle_rad(wxyz, target_wxyz)
    delta = candidate[:7] - measured_cfg[:7]
    return (
        position_error,
        orientation_error,
        float(np.max(np.abs(delta))),
        float(np.linalg.norm(delta)),
    )


def move_hand_pose_measured_seeded(
    api: Any,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
    *,
    log: bool = True,
    execute: bool = True,
) -> MeasuredSeedIkResult:
    """Solve from live q and optionally execute one native joint command.

    ``execute=False`` is a side-effect-free path preview used only to avoid an
    invalid approach waypoint poisoning an otherwise reachable contact goal.
    It is not a rejection gate on the goal itself.
    """
    target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
    target_wxyz = np.asarray(target_wxyz, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(target_position)) or not np.all(np.isfinite(target_wxyz)):
        raise ValueError("measured-seed IK target is nonfinite")
    quat_norm = float(np.linalg.norm(target_wxyz))
    if quat_norm < 1e-8:
        raise ValueError("measured-seed IK target quaternion is zero")
    target_wxyz = target_wxyz / quat_norm

    measured_before = _live_actuated_cfg(api)
    orientations = [("requested", target_wxyz)]
    if tool_pi_symmetry_enabled():
        orientations.append(
            (
                "tool_z_pi",
                _quat_multiply_wxyz(
                    target_wxyz, np.array([0.0, 0.0, 0.0, 1.0])
                ),
            )
        )
    candidates = []
    for orientation_name, orientation_wxyz in orientations:
        measured_candidate = np.asarray(
            api._pks.solve_ik_seeded(
                robot=api._robot,
                target_link_name=api._target_link_name,
                target_position=target_position,
                target_wxyz=orientation_wxyz,
                initial_cfg=measured_before,
            ),
            dtype=np.float64,
        ).reshape(-1)
        _validate_candidate(api, measured_candidate)
        candidates.append(
            (
                "measured",
                orientation_name,
                orientation_wxyz,
                measured_candidate,
                _candidate_metrics(
                    api,
                    measured_candidate,
                    measured_before,
                    target_position,
                    orientation_wxyz,
                ),
            )
        )
    # Most calls stop here.  Only a solver local minimum triggers deterministic
    # multistart.  Candidate quality chooses an IK result; it never rejects the
    # requested motion or introduces an execution-time gate.
    if (
        not any(row[4][0] <= 0.002 and row[4][1] <= 0.020 for row in candidates)
    ):
        for seed_name, seed in _fallback_seeds(measured_before):
            for orientation_name, orientation_wxyz in orientations:
                fallback = np.asarray(
                    api._pks.solve_ik_seeded(
                        robot=api._robot,
                        target_link_name=api._target_link_name,
                        target_position=target_position,
                        target_wxyz=orientation_wxyz,
                        initial_cfg=seed,
                    ),
                    dtype=np.float64,
                ).reshape(-1)
                _validate_candidate(api, fallback)
                candidates.append(
                    (
                        seed_name,
                        orientation_name,
                        orientation_wxyz,
                        fallback,
                        _candidate_metrics(
                            api,
                            fallback,
                            measured_before,
                            target_position,
                            orientation_wxyz,
                        ),
                    )
                )

    accurate = [
        row for row in candidates if row[4][0] <= 0.002 and row[4][1] <= 0.020
    ]
    if accurate:
        selected_seed, selected_orientation, selected_wxyz, candidate, metrics = min(
            accurate, key=lambda row: (row[4][2], row[4][3])
        )
    else:
        selected_seed, selected_orientation, selected_wxyz, candidate, metrics = min(
            candidates, key=lambda row: (row[4][0] + 0.05 * row[4][1], row[4][2])
        )

    arm_n = min(7, candidate.size)
    candidate_dq = float(
        np.max(np.abs(candidate[:arm_n] - measured_before[:arm_n]))
    )
    pt_swept_path = None
    if os.environ.get("CAPX_LIBERO_G17_V296A_PT_SWEPT_AUDIT", "0") == "1":
        pt_scene = getattr(api, "_capx_v296a_pt_scene", None)
        if pt_scene is not None:
            try:
                from capx.integrations.franka import perceptual_swept_path as _swept

                snapshot = "sample_grasp_pose_snapshot"
                target_points = getattr(api, "_capx_v296a_pt_target_points", None)
                if (
                    target_points is not None
                    and getattr(api, "_capx_v293_capture_state", None)
                    != "holding_payload"
                ):
                    pt_scene = _swept.prepare_scene_from_observation(
                        api, target_points, measured_before
                    )
                    snapshot = "exact_command_live_rgbd"

                pt_swept_path = _swept.audit_candidate_path(
                    api, pt_scene, measured_before, candidate
                )
                # This is the exact joint target selected by this function and
                # subsequently handed to the native controller.  Keeping it in
                # the private audit makes branch identity independently
                # checkable; it is never exposed to the policy.
                pt_swept_path["audit_subject"] = "exact_native_command_q_goal"
                pt_swept_path["scene_snapshot"] = snapshot
                pt_swept_path["command_goal_arm_q"] = (
                    np.asarray(candidate[:arm_n], dtype=np.float64)
                    .round(8)
                    .tolist()
                )
                pt_swept_path["command_start_arm_q"] = (
                    np.asarray(measured_before[:arm_n], dtype=np.float64)
                    .round(8)
                    .tolist()
                )
            except Exception as exc:
                pt_swept_path = {
                    "schema": "capx_libero_v296a_pt_swept_path_v1",
                    "behavior_effect": "audit_only",
                    "audit_subject": "exact_native_command_q_goal",
                    "error": repr(exc),
                }
    if not execute:
        position_error, orientation_error = metrics[:2]
        ok = bool(
            position_error <= _ACCEPT_POSITION_M
            and orientation_error <= _ACCEPT_ORIENTATION_RAD
        )
        message = (
            f"status=preview-{'accurate' if ok else 'inaccurate'} "
            f"seed={selected_seed} orientation={selected_orientation} "
            f"candidates={len(candidates)} commands=0 "
            f"candidate_dq={candidate_dq:.5f} "
            f"predicted=({position_error:.5f}m,{orientation_error:.5f}rad)"
        )
        if log:
            print(f"[measured-seed-ik] {message}", flush=True)
        return MeasuredSeedIkResult(
            ok=ok,
            status="preview_accurate" if ok else "preview_inaccurate",
            candidate_arm_delta_rad=candidate_dq,
            measured_arm_delta_rad=0.0,
            final_position_error_m=position_error,
            final_orientation_error_rad=orientation_error,
            execution_commands=0,
            selected_seed=selected_seed,
            candidate_count=len(candidates),
            selected_orientation=selected_orientation,
            message=message,
            pt_swept_path=pt_swept_path,
        )
    # Preserve the solver's non-arm coordinate while the physical command is
    # in flight; the subsequent live read always replaces all seven arm joints.
    api.cfg = candidate.copy()
    # ``move_to_joints_blocking`` exits early on tolerance.  Its legacy
    # 120-step cap silently returns partway through large but valid joint
    # targets (~0.96 rad observed on a 2.96 rad command), corrupting the next
    # IK seed.  A larger *maximum* keeps small motions byte-for-byte equivalent
    # in duration while allowing one continuous native command to finish.
    api._env.move_to_joints_blocking(
        candidate[:arm_n], max_steps=_NATIVE_CONTROLLER_MAX_STEPS
    )
    measured_after = _live_actuated_cfg(api)
    api.cfg = measured_after.copy()

    measured_dq = float(
        np.max(np.abs(measured_after[:arm_n] - measured_before[:arm_n]))
    )
    achieved_position, achieved_wxyz = fk_panda_hand(api, measured_after)
    position_error = float(np.linalg.norm(achieved_position - target_position))
    orientation_error = _quaternion_angle_rad(achieved_wxyz, selected_wxyz)
    ok = bool(
        position_error <= _ACCEPT_POSITION_M
        and orientation_error <= _ACCEPT_ORIENTATION_RAD
    )
    message = (
        f"status=executed-native-once-{'aligned' if ok else 'miss'} seed={selected_seed} "
        f"orientation={selected_orientation} "
        f"candidates={len(candidates)} cache=measured commands=1 "
        f"candidate_dq={candidate_dq:.5f} measured_dq={measured_dq:.5f} "
        f"final=({position_error:.5f}m,{orientation_error:.5f}rad)"
    )
    if log:
        print(f"[measured-seed-ik] {message}", flush=True)
    return MeasuredSeedIkResult(
        ok=ok,
        status="executed_native_once_aligned" if ok else "executed_native_once_miss",
        candidate_arm_delta_rad=candidate_dq,
        measured_arm_delta_rad=measured_dq,
        final_position_error_m=position_error,
        final_orientation_error_rad=orientation_error,
        execution_commands=1,
        selected_seed=selected_seed,
        candidate_count=len(candidates),
        selected_orientation=selected_orientation,
        message=message,
        pt_swept_path=pt_swept_path,
    )
