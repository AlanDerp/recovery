#!/usr/bin/env python3
"""Execute validated, hash-bound LLM cup-stack code on the R1 Lite."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel  # noqa: E402
from capx.integrations.r1lite.control import R1LiteControlApi  # noqa: E402
from capx.integrations.r1lite.cup_stack_llm import (  # noqa: E402
    ALLOWED_CALLS,
    execute_generated_code_dry_run,
    expected_operations,
    validate_generated_code,
)


class RequestedStopAfterFirstGrasp(Exception):
    """Controlled completion after the first verified grasp."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--expected-code-sha256", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--torso-hold-status", type=Path, required=True)
    parser.add_argument("--maximum-scene-drift-m", type=float, default=0.020)
    parser.add_argument(
        "--maximum-initial-orientation-delta-rad",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--maximum-initial-joint-delta-rad", type=float, default=0.03
    )
    parser.add_argument("--maximum-torso-drift-rad", type=float, default=0.03)
    parser.add_argument(
        "--maximum-planned-torso-delta-rad",
        type=float,
        default=0.01,
    )
    parser.add_argument("--stop-after-first-grasp", action="store_true")
    parser.add_argument("--resume-after-first-grasp", action="store_true")
    parser.add_argument("--resume-after-first-transfer", action="store_true")
    parser.add_argument("--resume-after-second-lift", action="store_true")
    parser.add_argument("--resume-after-second-transfer", action="store_true")
    parser.add_argument("--resume-hold-gripper-command", type=float)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close_gripper_until_contact(
    api: R1LiteControlApi,
    *,
    minimum_command: float,
    open_position: float = 60.0,
    step: float = 2.0,
    duration_s: float = 0.6,
    contact_gap: float = 2.5,
    stable_feedback_delta: float = 0.8,
    confirmations: int = 2,
    maximum_confirmation_samples: int = 8,
    minimum_observed_motion: float = 2.0,
) -> dict[str, Any]:
    """Close slowly; hold the first obstruction command while confirming contact."""
    trace = []
    for command in np.arange(
        float(open_position) - step,
        float(minimum_command) - step,
        -step,
    ):
        bounded_command = max(float(minimum_command), float(command))
        feedback = float(
            api.command_gripper(
                bounded_command,
                duration_s=float(duration_s),
                allow_contact=True,
            )
        )
        gap = feedback - bounded_command
        motion_observed = (
            float(open_position) - feedback >= float(minimum_observed_motion)
        )
        trace.append(
            {
                "command": bounded_command,
                "feedback": feedback,
                "gap": gap,
                "mode": "closing",
                "motion_observed": motion_observed,
            }
        )
        if gap >= float(contact_gap) and motion_observed:
            contact_count = 0
            previous_feedback = feedback
            for _ in range(int(maximum_confirmation_samples)):
                confirmed_feedback = float(
                    api.command_gripper(
                        bounded_command,
                        duration_s=float(duration_s),
                        allow_contact=True,
                    )
                )
                confirmed_gap = confirmed_feedback - bounded_command
                stable = (
                    abs(confirmed_feedback - previous_feedback)
                    <= float(stable_feedback_delta)
                )
                obstructed = confirmed_gap >= float(contact_gap) and stable
                contact_count = contact_count + 1 if obstructed else 0
                trace.append(
                    {
                        "command": bounded_command,
                        "feedback": confirmed_feedback,
                        "gap": confirmed_gap,
                        "mode": "confirming_contact",
                        "stable": stable,
                        "contact_count": contact_count,
                    }
                )
                if contact_count >= int(confirmations):
                    return {
                        "contact_detected": True,
                        "hold_command": bounded_command,
                        "feedback": confirmed_feedback,
                        "trace": trace,
                    }
                if confirmed_gap < float(contact_gap):
                    break
                previous_feedback = confirmed_feedback
        if bounded_command <= float(minimum_command):
            break
    raise RuntimeError(
        "adaptive gripper close reached its minimum without stable contact"
    )


def build_loaded_transfer_route_candidates(
    start_position: np.ndarray,
    above_support_position: np.ndarray,
    corridor_x_offsets_m: list[float],
) -> list[list[list[float]]]:
    """Build routes that rise over the source or outside the cup row."""
    start = np.asarray(start_position, dtype=float).reshape(3)
    target = np.asarray(above_support_position, dtype=float).reshape(3)
    outboard_y = start[1] + (-0.08 if start[1] < target[1] else 0.08)
    routes: list[list[list[float]]] = [
        [
            [float(target[0]), float(start[1]), float(start[2])],
            [float(target[0]), float(start[1]), float(target[2])],
            [float(target[0]), float(target[1]), float(target[2])],
        ]
    ]
    for offset in corridor_x_offsets_m:
        corridor_x = float(target[0] + float(offset))
        routes.append(
            [
                [corridor_x, float(start[1]), float(start[2])],
                [corridor_x, float(start[1]), float(target[2])],
                [corridor_x, float(target[1]), float(target[2])],
            ]
        )
        routes.append(
            [
                [float(start[0]), float(outboard_y), float(start[2])],
                [corridor_x, float(outboard_y), float(start[2])],
                [corridor_x, float(outboard_y), float(target[2])],
                [corridor_x, float(target[1]), float(target[2])],
            ]
        )
    return routes


def rotate_quaternion_about_base_axis(
    quaternion_xyzw: np.ndarray,
    axis_xyz: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """Rotate an end-effector orientation about a robot-base axis."""
    quaternion = np.asarray(quaternion_xyzw, dtype=float).reshape(4)
    quaternion /= np.linalg.norm(quaternion)
    axis = np.asarray(axis_xyz, dtype=float).reshape(3)
    axis /= np.linalg.norm(axis)
    half = 0.5 * float(angle_rad)
    delta = np.concatenate([axis * np.sin(half), [np.cos(half)]])
    x1, y1, z1, w1 = delta
    x2, y2, z2, w2 = quaternion
    result = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    return result / np.linalg.norm(result)


def build_free_transit_orientation_candidates(
    current_quaternion_xyzw: np.ndarray,
    topdown_quaternion_xyzw: np.ndarray,
) -> list[np.ndarray]:
    """Return ordered, de-duplicated orientations for one transport step."""
    current = np.asarray(current_quaternion_xyzw, dtype=float).reshape(4)
    current /= np.linalg.norm(current)
    topdown = np.asarray(topdown_quaternion_xyzw, dtype=float).reshape(4)
    topdown /= np.linalg.norm(topdown)
    candidates = [current, topdown]
    axes = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    for angle_deg in (15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0, 90.0, -90.0):
        angle_rad = np.deg2rad(angle_deg)
        for axis in axes:
            candidates.append(
                rotate_quaternion_about_base_axis(
                    topdown, axis, angle_rad
                )
            )
    unique: list[np.ndarray] = []
    for candidate in candidates:
        if not any(
            abs(float(np.dot(candidate, existing))) > 0.9999
            for existing in unique
        ):
            unique.append(candidate)
    return unique


def interpolate_cartesian_segment(
    start_position: np.ndarray,
    end_position: np.ndarray,
    *,
    maximum_step_m: float,
) -> list[list[float]]:
    start = np.asarray(start_position, dtype=float).reshape(3)
    end = np.asarray(end_position, dtype=float).reshape(3)
    steps = max(
        1,
        int(np.ceil(np.linalg.norm(end - start) / float(maximum_step_m))),
    )
    return [
        (start + (end - start) * (index / steps)).tolist()
        for index in range(1, steps + 1)
    ]


def load_robot(config_path: Path) -> R1LiteRealLowLevel:
    config = yaml.safe_load(config_path.read_text())
    low_config = dict(config["env"]["cfg"]["low_level"])
    low_config.pop("_target_", None)
    signature = inspect.signature(R1LiteRealLowLevel.__init__)
    kwargs = {
        name: value
        for name, value in low_config.items()
        if name in signature.parameters
    }
    kwargs["wrist_camera_enabled"] = False
    return R1LiteRealLowLevel(**kwargs)


def main() -> None:
    args = parse_args()
    if sha256_file(args.plan) != args.expected_plan_sha256:
        raise RuntimeError("approved LLM cup-stack plan hash mismatch")
    if sha256_file(args.code) != args.expected_code_sha256:
        raise RuntimeError("approved LLM cup-stack code hash mismatch")
    plan = json.loads(args.plan.read_text())
    code = args.code.read_text()
    torso_hold = json.loads(args.torso_hold_status.read_text())
    if torso_hold.get("violation"):
        raise RuntimeError("torso hold reported a startup drift violation")
    torso_hold_pid = int(torso_hold["pid"])
    os.kill(torso_hold_pid, 0)
    torso_reference = np.asarray(
        torso_hold["reference_position_rad"],
        dtype=float,
    ).reshape(3)
    planned_torso = np.asarray(
        plan["rgbd_scene"]["home_torso_joints_rad"],
        dtype=float,
    ).reshape(3)
    if float(np.max(np.abs(planned_torso - torso_reference))) > float(
        args.maximum_planned_torso_delta_rad
    ):
        raise RuntimeError(
            "planned torso pose differs from the active hold reference"
        )
    expected = expected_operations(
        plan,
        plan_sha256=args.expected_plan_sha256,
    )
    validate_generated_code(code, expected=expected)
    execute_generated_code_dry_run(code, expected=expected)

    phases = {phase["phase"]: phase for phase in plan["phases"]}
    ik_path_limits = plan["ik_path_limits"]
    initial_cups = plan["rgbd_scene"]["cups"]
    cup_height = float(plan["cup_height_m"])
    open_position = float(plan.get("gripper_open_position", 60.0))
    close_position = float(plan["gripper_close_position"])
    events: list[dict[str, Any]] = []
    completion_scope = "full_task"
    resume_active = bool(
        args.resume_after_first_grasp
        or args.resume_after_first_transfer
        or args.resume_after_second_lift
        or args.resume_after_second_transfer
    )
    state = {
        "bound": False,
        "closed": resume_active,
        "hold_gripper_command": (
            float(args.resume_hold_gripper_command)
            if args.resume_hold_gripper_command is not None
            else None
        ),
        "released_phases": (
            {plan["phase_order"][0]}
            if (
                args.resume_after_second_lift
                or args.resume_after_second_transfer
            )
            else set()
        ),
        "verified": False,
    }
    if resume_active and state["hold_gripper_command"] is None:
        raise ValueError(
            "--resume-hold-gripper-command is required when resuming"
        )
    robot = load_robot(args.config)
    api = R1LiteControlApi(robot)
    torso_monitor = {
        "reference_position_rad": torso_reference.tolist(),
        "latest_position_rad": None,
        "maximum_drift_rad": 0.0,
        "violation": False,
        "hold_process_alive": True,
    }
    monitor_stop = threading.Event()
    previous_sigusr1 = signal.getsignal(signal.SIGUSR1)

    def torso_abort_signal(_signum: int, _frame: Any) -> None:
        raise RuntimeError(
            "torso hold safety gate interrupted the arm task: "
            f"{torso_monitor}"
        )

    signal.signal(signal.SIGUSR1, torso_abort_signal)

    def monitor_torso() -> None:
        while not monitor_stop.wait(0.02):
            try:
                os.kill(torso_hold_pid, 0)
            except OSError:
                torso_monitor["hold_process_alive"] = False
                torso_monitor["violation"] = True
                os.kill(os.getpid(), signal.SIGUSR1)
                return
            current_torso = robot.get_torso_joints()
            drift = float(
                np.max(np.abs(current_torso - torso_reference))
            )
            torso_monitor["latest_position_rad"] = current_torso.tolist()
            torso_monitor["maximum_drift_rad"] = max(
                float(torso_monitor["maximum_drift_rad"]),
                drift,
            )
            if drift > float(args.maximum_torso_drift_rad):
                torso_monitor["violation"] = True
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    monitor_thread = threading.Thread(
        target=monitor_torso,
        name="r1lite-torso-drift-monitor",
        daemon=True,
    )
    try:
        monitor_thread.start()
        planned_home_joints = np.asarray(
            plan["rgbd_scene"]["home_arm_joints_rad"], dtype=float
        ).reshape(6)
        current_home_joints = robot.get_arm_joints()
        initial_joint_delta = float(
            np.max(np.abs(current_home_joints - planned_home_joints))
        )
        if (
            not resume_active
            and initial_joint_delta
            > float(args.maximum_initial_joint_delta_rad)
        ):
            raise RuntimeError(
                "right-arm joints changed after planning: "
                f"{initial_joint_delta:.4f}rad"
            )
        robot.enable_arm_joint_control_without_function_reset()
        enabled_home_joints = robot.get_arm_joints()
        enabled_joint_delta = float(
            np.max(np.abs(enabled_home_joints - planned_home_joints))
        )
        if (
            not resume_active
            and enabled_joint_delta
            > float(args.maximum_initial_joint_delta_rad)
        ):
            raise RuntimeError(
                "right arm moved while enabling joint control: "
                f"{enabled_joint_delta:.4f}rad"
            )
        events.append(
            {
                "event": "home_joint_and_existing_mode_gate",
                "before_enable_delta_rad": initial_joint_delta,
                "after_enable_delta_rad": enabled_joint_delta,
                "planned_home_joints_rad": planned_home_joints.tolist(),
            }
        )
        current_position, current_orientation = (
            robot.get_local_planning_fk_pose()
        )
        planned_orientation = np.asarray(
            (
                phases[plan["phase_order"][0]][
                    "orientation_quaternion_xyzw"
                ]
                if resume_active
                else plan["home_gripper_quaternion_xyzw"]
            ),
            dtype=float,
        )
        current_orientation /= np.linalg.norm(current_orientation)
        planned_orientation /= np.linalg.norm(planned_orientation)
        orientation_delta = float(
            2.0
            * np.arccos(
                np.clip(
                    abs(np.dot(current_orientation, planned_orientation)),
                    0.0,
                    1.0,
                )
            )
        )
        if orientation_delta > float(
            args.maximum_initial_orientation_delta_rad
        ):
            raise RuntimeError(
                "initial gripper orientation changed after planning: "
                f"{orientation_delta:.4f}rad"
            )
        events.append(
            {
                "event": (
                    "resume_orientation_gate"
                    if resume_active
                    else "home_orientation_gate"
                ),
                "orientation_delta_rad": orientation_delta,
                "planned_quaternion_xyzw": planned_orientation.tolist(),
            }
        )
        if resume_active:
            resume_phase_id = (
                plan["phase_order"][1]
                if (
                    args.resume_after_second_lift
                    or args.resume_after_second_transfer
                )
                else plan["phase_order"][0]
            )
            first_phase = phases[resume_phase_id]
            expected_lift = np.asarray(
                (
                    next(
                        item["tcp_position_base_m"]
                        for item in first_phase["waypoints"]
                        if item["id"] == "above_support"
                    )
                    if args.resume_after_second_transfer
                    else first_phase[
                        "loaded_transfer_tcp_position_base_m"
                    ]
                    if args.resume_after_first_transfer
                    else next(
                        item["tcp_position_base_m"]
                        for item in first_phase["waypoints"]
                        if item["id"] == "lift"
                    )
                ),
                dtype=float,
            )
            resume_position_error = float(
                np.linalg.norm(current_position - expected_lift)
            )
            if (
                args.resume_after_first_grasp
                or args.resume_after_second_lift
            ):
                expected_above_support = np.asarray(
                    next(
                        item["tcp_position_base_m"]
                        for item in first_phase["waypoints"]
                        if item["id"] == "above_support"
                    ),
                    dtype=float,
                )
                if first_phase.get("loaded_transfer_route_policy") == (
                    "mandatory_rear_retract_lift_rotate_then_advance"
                ):
                    rear_x = float(
                        expected_above_support[0]
                        + first_phase[
                            "loaded_transfer_rear_x_offset_m"
                        ]
                    )
                    transit_z = float(
                        first_phase["loaded_transfer_transit_tcp_z_m"]
                    )
                    resume_routes = [
                        [
                            [
                                rear_x,
                                float(expected_lift[1]),
                                float(expected_lift[2]),
                            ],
                            [rear_x, float(expected_lift[1]), transit_z],
                            [
                                rear_x,
                                float(expected_above_support[1]),
                                transit_z,
                            ],
                            [
                                float(expected_above_support[0]),
                                float(expected_above_support[1]),
                                transit_z,
                            ],
                        ]
                    ]
                else:
                    resume_routes = build_loaded_transfer_route_candidates(
                        expected_lift,
                        expected_above_support,
                        list(
                            first_phase[
                                "loaded_transfer_corridor_x_offsets_m"
                            ]
                        ),
                    )
                route_points = [
                    np.asarray(point, dtype=float)
                    for route in resume_routes
                    for point in route
                ]
                nearest_route_point = min(
                    route_points,
                    key=lambda point: float(
                        np.linalg.norm(current_position - point)
                    ),
                )
                route_position_error = float(
                    np.linalg.norm(current_position - nearest_route_point)
                )
                if route_position_error < resume_position_error:
                    resume_position_error = route_position_error
                    expected_lift = nearest_route_point
            if (
                args.resume_after_first_transfer
                or args.resume_after_second_transfer
            ):
                expected_release = np.asarray(
                    first_phase["release_hover_tcp_position_base_m"],
                    dtype=float,
                )
                release_position_error = float(
                    np.linalg.norm(current_position - expected_release)
                )
                if release_position_error < resume_position_error:
                    resume_position_error = release_position_error
                    expected_lift = expected_release
            maximum_resume_error = (
                0.08
                if (
                    args.resume_after_first_transfer
                    or args.resume_after_second_transfer
                )
                else 0.04
            )
            if resume_position_error > maximum_resume_error:
                raise RuntimeError(
                    "resume pose is not near the verified first-cup lift: "
                    f"{resume_position_error:.4f}m"
                )
            grasp_ok, grasp_details = api.verify_grasp(save_debug=True)
            if not grasp_ok:
                raise RuntimeError(
                    f"resume grasp-retention gate failed: {grasp_details}"
                )
            events.append(
                {
                    "event": "resume_after_first_grasp_gate",
                    "position_error_m": resume_position_error,
                    "current_tcp_base_m": current_position.tolist(),
                    "expected_lift_tcp_base_m": expected_lift.tolist(),
                    "hold_gripper_command": state["hold_gripper_command"],
                    "grasp_details": grasp_details,
                }
            )
        if plan.get("runtime_cup_reperception_enabled", True):
            current = api.locate_colored_cups_rgbd(
                cup_height_m=cup_height,
                save_debug=True,
                require_separation=True,
            )
            drift = {}
            for name in current:
                before = np.asarray(
                    initial_cups[name]["top_center_base_m"], dtype=float
                )
                now = np.asarray(
                    current[name]["top_center_base_m"], dtype=float
                )
                drift[name] = float(np.linalg.norm(now - before))
            if max(drift.values()) > float(args.maximum_scene_drift_m):
                raise RuntimeError(
                    f"RGB-D scene changed after LLM planning: {drift}"
                )
            events.append(
                {"event": "scene_repeatability_gate", "drift_m": drift}
            )
        else:
            events.append(
                {
                    "event": "runtime_cup_reperception_skipped",
                    "pose_source": plan["cup_pose_source"],
                }
            )

        def bind_cup_plan(*, sha256: str) -> None:
            if sha256 != args.expected_plan_sha256:
                raise RuntimeError("generated code bound the wrong plan")
            state["bound"] = True
            events.append({"event": "plan_bound", "sha256": sha256})

        def require_bound() -> None:
            if not state["bound"]:
                raise RuntimeError("cup plan must be bound before execution")

        def open_gripper() -> None:
            require_bound()
            feedback = api.command_gripper(open_position, duration_s=3.0)
            state["closed"] = False
            state["hold_gripper_command"] = None
            events.append(
                {"event": "open_gripper", "feedback": float(feedback)}
            )

        def close_gripper() -> None:
            require_bound()
            result = close_gripper_until_contact(
                api,
                minimum_command=close_position,
                open_position=open_position,
            )
            state["closed"] = True
            state["hold_gripper_command"] = float(result["hold_command"])
            events.append(
                {
                    "event": "adaptive_close_gripper",
                    **result,
                }
            )

        def execute_cup_stage(*, phase: str, stage_id: str) -> None:
            require_bound()
            phase_plan = phases[phase]
            if stage_id == "release_hover":
                target = np.asarray(
                    phase_plan["release_hover_tcp_position_base_m"],
                    dtype=float,
                )
            else:
                waypoint = next(
                    item
                    for item in phase_plan["waypoints"]
                    if item["id"] == stage_id
                )
                target = np.asarray(
                    waypoint["tcp_position_base_m"], dtype=float
                )
            quaternion = np.asarray(
                phase_plan["orientation_quaternion_xyzw"], dtype=float
            )
            if stage_id in {
                "lift",
                "above_support",
                "release_hover",
            } and not state["closed"]:
                raise RuntimeError(f"{stage_id} requires a retained closed gripper")
            gripper = (
                float(state["hold_gripper_command"])
                if state["closed"]
                else open_position
            )
            if stage_id == "grasp":
                start = next(
                    item
                    for item in phase_plan["waypoints"]
                    if item["id"] == "pregrasp"
                )
                result = api.descend_z_in_steps(
                    target,
                    quaternion,
                    start_z=float(start["tcp_position_base_m"][2]),
                    num_steps=int(phase_plan["grasp_descent_steps"]),
                    gripper=gripper,
                    check_final_cartesian_error_only=True,
                )
            elif (
                stage_id == "release_hover"
                and phase_plan.get("release_policy")
                == "gravity_drop_directly_from_above_support"
            ):
                measured_position, measured_quaternion = (
                    robot.get_local_planning_fk_pose()
                )
                xy_error = float(
                    np.linalg.norm(measured_position[:2] - target[:2])
                )
                maximum_xy_error = float(
                    phase_plan.get("release_maximum_xy_error_m", 0.01)
                )
                if xy_error >= maximum_xy_error:
                    raise RuntimeError(
                        "gravity-drop XY alignment is outside tolerance: "
                        f"error={xy_error:.4f}m, "
                        f"limit={maximum_xy_error:.4f}m"
                    )
                result = {
                    "success": True,
                    "release_policy": (
                        "gravity_drop_directly_from_above_support"
                    ),
                    "descent_skipped": True,
                    "release_xy_target_base_m": target[:2].tolist(),
                    "measured_position": measured_position.tolist(),
                    "measured_quaternion_xyzw": (
                        measured_quaternion.tolist()
                    ),
                    "xy_error_m": xy_error,
                    "maximum_xy_error_m": maximum_xy_error,
                }
            elif (
                stage_id == "release_hover"
                and stage_id
                in phase_plan.get("saved_stage_joint_positions_rad", {})
            ):
                saved_joints = np.asarray(
                    phase_plan["saved_stage_joint_positions_rad"][stage_id],
                    dtype=float,
                ).reshape(6)
                joint_result = api.move_joints_adaptive(
                    saved_joints,
                    waypoint_step_rad=0.015,
                    context=f"{phase}_{stage_id}_saved_joint_position",
                )
                measured_position, _ = robot.get_local_planning_fk_pose()
                position_error = float(
                    np.linalg.norm(measured_position - target)
                )
                if position_error > 0.025:
                    raise RuntimeError(
                        "saved release joint position missed its TCP target: "
                        f"{position_error:.4f}m"
                    )
                result = {
                    "success": True,
                    "saved_joint_position_used": True,
                    "saved_joint_position_rad": saved_joints.tolist(),
                    "joint_result": joint_result,
                    "target_position": target.tolist(),
                    "measured_position": measured_position.tolist(),
                    "position_error_m": position_error,
                }
            elif stage_id in {"lift", "release_hover"}:
                command_target = target.copy()
                correction_trace = []
                arrival_tolerance_m = (
                    0.015 if stage_id == "lift" else 0.025
                )
                for correction_index in range(3):
                    motion_result = api.goto_pose(
                        command_target,
                        quaternion,
                        gripper=gripper,
                        translate_then_orient=False,
                        max_cartesian_step_m=float(
                            ik_path_limits["max_cartesian_step_m"]
                        ),
                        max_orientation_step_rad=float(
                            ik_path_limits["max_orientation_step_rad"]
                        ),
                        check_cartesian_ik_error=True,
                        verify_motion=False,
                        require_pose_arrival=False,
                    )
                    measured_position, _ = (
                        robot.get_local_planning_fk_pose()
                    )
                    physical_error = target - measured_position
                    physical_error_norm = float(
                        np.linalg.norm(physical_error)
                    )
                    correction_trace.append(
                        {
                            "index": correction_index + 1,
                            "command_target_tcp_base_m": command_target.tolist(),
                            "measured_tcp_base_m": measured_position.tolist(),
                            "physical_error_m": physical_error.tolist(),
                            "physical_error_norm_m": physical_error_norm,
                            "motion_result": motion_result,
                        }
                    )
                    if physical_error_norm <= arrival_tolerance_m:
                        break
                    if float(np.linalg.norm(physical_error)) > 0.08:
                        raise RuntimeError(
                            "release feedback correction exceeded 0.08m"
                        )
                    command_target = command_target + physical_error
                    if stage_id == "release_hover" and command_target[2] < 0.95:
                        raise RuntimeError(
                            "release correction would violate TCP floor"
                        )
                else:
                    raise RuntimeError(
                        f"{stage_id} remained outside "
                        f"{arrival_tolerance_m:.3f}m after correction"
                    )
                result = {
                    "success": True,
                    "feedback_corrected_stage": stage_id,
                    "target_position": target.tolist(),
                    "measured_position": measured_position.tolist(),
                    "position_error_m": physical_error_norm,
                    "correction_trace": correction_trace,
                }
            elif stage_id == "above_support":
                if phase_plan.get("loaded_transfer_policy") != (
                    "obstacle_clearance_multi_via_search"
                ):
                    raise RuntimeError(
                        "low loaded-transfer fallback is forbidden"
                    )
                current_position, _ = robot.get_local_planning_fk_pose()
                if phase_plan.get("loaded_transfer_route_policy") == (
                    "mandatory_rear_retract_lift_rotate_then_advance"
                ):
                    rear_x = float(
                        target[0]
                        + float(
                            phase_plan[
                                "loaded_transfer_rear_x_offset_m"
                            ]
                        )
                    )
                    transit_z = float(
                        phase_plan["loaded_transfer_transit_tcp_z_m"]
                    )
                    route_candidates = [
                        [
                            [
                                rear_x,
                                float(current_position[1]),
                                float(current_position[2]),
                            ],
                            [
                                rear_x,
                                float(current_position[1]),
                                transit_z,
                            ],
                            [rear_x, float(target[1]), transit_z],
                            [
                                float(target[0]),
                                float(target[1]),
                                transit_z,
                            ],
                        ]
                    ]
                else:
                    route_candidates = build_loaded_transfer_route_candidates(
                        current_position,
                        target,
                        list(
                            phase_plan[
                                "loaded_transfer_corridor_x_offsets_m"
                            ]
                        ),
                    )
                transit_quaternion = quaternion
                if phase_plan.get("loaded_transfer_orientation_policy") == (
                    "tilted_in_transit_topdown_at_above_support"
                ):
                    transit_quaternion = rotate_quaternion_about_base_axis(
                        quaternion,
                        np.array([0.0, 1.0, 0.0]),
                        float(
                            phase_plan.get(
                                "loaded_transfer_tilt_angle_rad", 0.261799
                            )
                        ),
                    )
                flexible_transit_orientation = phase_plan.get(
                    "loaded_transfer_orientation_policy"
                ) == "free_per_waypoint_topdown_at_above_support"

                def execute_transport_point(
                    point: list[float],
                    preferred_quaternion: np.ndarray,
                    *,
                    force_topdown: bool = False,
                ) -> dict[str, Any]:
                    if flexible_transit_orientation and not force_topdown:
                        _, current_quaternion = (
                            robot.get_local_planning_fk_pose()
                        )
                        orientation_candidates = (
                            build_free_transit_orientation_candidates(
                                current_quaternion,
                                quaternion,
                            )
                        )
                    else:
                        orientation_candidates = [preferred_quaternion]
                    orientation_rejections = []
                    for orientation_candidate in orientation_candidates:
                        try:
                            candidate_plan = api.plan_pose_waypoints(
                                point,
                                orientation_candidate,
                                max_cartesian_step_m=0.03,
                                max_orientation_step_rad=float(
                                    ik_path_limits[
                                        "max_orientation_step_rad"
                                    ]
                                ),
                                check_cartesian_ik_error=False,
                            )
                            endpoint_error = float(
                                candidate_plan["waypoints"][-1][
                                    "fk_position_error_m"
                                ]
                            )
                            if endpoint_error > 0.025:
                                raise RuntimeError(
                                    "orientation candidate endpoint "
                                    f"error is {endpoint_error:.4f}m"
                                )
                            point_result = api.goto_pose(
                                point,
                                orientation_candidate,
                                gripper=gripper,
                                translate_then_orient=False,
                                max_cartesian_step_m=0.03,
                                max_orientation_step_rad=float(
                                    ik_path_limits[
                                        "max_orientation_step_rad"
                                    ]
                                ),
                                check_cartesian_ik_error=False,
                                verify_motion=True,
                                require_pose_arrival=True,
                                position_arrival_tolerance_m=0.025,
                            )
                            point_result[
                                "selected_transport_quaternion_xyzw"
                            ] = orientation_candidate.tolist()
                            point_result[
                                "orientation_candidates_rejected"
                            ] = orientation_rejections
                            return point_result
                        except Exception as exc:
                            orientation_rejections.append(repr(exc))
                    raise RuntimeError(
                        "no orientation reaches loaded transport "
                        f"waypoint {point}: {orientation_rejections}"
                    )
                valid_routes = []
                rejected_routes = []
                for route_index, route in enumerate(route_candidates):
                    if flexible_transit_orientation:
                        route_points = [
                            np.asarray(current_position, dtype=float),
                            *[
                                np.asarray(point, dtype=float)
                                for point in route
                            ],
                        ]
                        path_length = float(
                            sum(
                                np.linalg.norm(end - start)
                                for start, end in zip(
                                    route_points,
                                    route_points[1:],
                                )
                            )
                        )
                        valid_routes.append(
                            (
                                path_length,
                                route_index,
                                route,
                                {
                                    "planning_policy": (
                                        "online_per_waypoint_orientation_search"
                                    )
                                },
                            )
                        )
                        continue
                    try:
                        route_plan = api.plan_pose_waypoints(
                            target,
                            quaternion,
                            via_positions=route,
                            via_quaternions_xyzw=[
                                transit_quaternion.tolist()
                                for _ in route
                            ],
                            max_cartesian_step_m=float(
                                ik_path_limits["max_cartesian_step_m"]
                            ),
                            max_orientation_step_rad=float(
                                ik_path_limits["max_orientation_step_rad"]
                            ),
                            check_cartesian_ik_error=False,
                        )
                        endpoint_error = float(
                            route_plan["waypoints"][-1][
                                "fk_position_error_m"
                            ]
                        )
                        if endpoint_error > 0.025:
                            raise RuntimeError(
                                "route endpoint FK error exceeds 0.025m: "
                                f"{endpoint_error:.4f}m"
                            )
                        path_length = float(
                            route_plan["position_distance_m"]
                        )
                        valid_routes.append(
                            (path_length, route_index, route, route_plan)
                        )
                    except Exception as exc:
                        rejected_routes.append(
                            {
                                "route_index": route_index,
                                "route": route,
                                "error": repr(exc),
                            }
                        )
                if not valid_routes:
                    raise RuntimeError(
                        "no collision-clearance route reaches above_support"
                    )
                _, selected_index, selected_route, selected_plan = min(
                    valid_routes, key=lambda item: (item[0], item[1])
                )
                high_z = max(float(point[2]) for point in selected_route)
                high_index = next(
                    index
                    for index, point in enumerate(selected_route)
                    if np.isclose(point[2], high_z, atol=1e-6)
                )
                low_route = selected_route[:high_index]
                high_points = selected_route[high_index:]
                execution_trace = []
                first_high = np.asarray(high_points[0], dtype=float)
                if flexible_transit_orientation:
                    commanded_position = np.asarray(
                        current_position, dtype=float
                    )
                    for low_destination in [*low_route, first_high.tolist()]:
                        for low_point in interpolate_cartesian_segment(
                            commanded_position,
                            low_destination,
                            maximum_step_m=0.03,
                        ):
                            execution_trace.append(
                                execute_transport_point(
                                    low_point,
                                    transit_quaternion,
                                )
                            )
                        commanded_position = np.asarray(
                            low_destination, dtype=float
                        )
                else:
                    first_result = api.goto_pose(
                        first_high,
                        transit_quaternion,
                        gripper=gripper,
                        via_positions=low_route,
                        via_quaternions_xyzw=[
                            transit_quaternion.tolist() for _ in low_route
                        ],
                        translate_then_orient=False,
                        max_cartesian_step_m=float(
                            ik_path_limits["max_cartesian_step_m"]
                        ),
                        max_orientation_step_rad=float(
                            ik_path_limits["max_orientation_step_rad"]
                        ),
                        check_cartesian_ik_error=False,
                        verify_motion=False,
                        require_pose_arrival=False,
                    )
                    execution_trace.append(first_result)
                minimum_transport_z = float(
                    phase_plan["minimum_loaded_transport_tcp_z_m"]
                )
                measured_position, _ = (
                    robot.get_local_planning_fk_pose()
                )
                clearance_command_target = first_high.copy()
                for _ in range(2):
                    if float(measured_position[2]) >= minimum_transport_z:
                        break
                    clearance_command_target[2] += (
                        minimum_transport_z
                        - float(measured_position[2])
                        + 0.005
                    )
                    correction_result = api.goto_pose(
                        clearance_command_target,
                        transit_quaternion,
                        gripper=gripper,
                        translate_then_orient=False,
                        max_cartesian_step_m=0.03,
                        max_orientation_step_rad=float(
                            ik_path_limits["max_orientation_step_rad"]
                        ),
                        check_cartesian_ik_error=False,
                        verify_motion=False,
                        require_pose_arrival=False,
                    )
                    execution_trace.append(correction_result)
                    measured_position, _ = (
                        robot.get_local_planning_fk_pose()
                    )
                if float(measured_position[2]) < minimum_transport_z:
                    raise RuntimeError(
                        "loaded cup did not reach collision-clearance height"
                    )
                commanded_position = clearance_command_target
                oriented_high_destinations = [
                    (destination, transit_quaternion)
                    for destination in high_points[1:]
                ]
                oriented_high_destinations.append(
                    (target.tolist(), quaternion)
                )
                early_release_ready = False
                for destination_index, (
                    destination,
                    destination_quaternion,
                ) in enumerate(oriented_high_destinations):
                    final_topdown_step = destination_index == (
                        len(oriented_high_destinations) - 1
                    )
                    monitor_points = interpolate_cartesian_segment(
                        commanded_position,
                        destination,
                        maximum_step_m=0.03,
                    )
                    for high_point in monitor_points:
                        segment_result = execute_transport_point(
                            high_point,
                            destination_quaternion,
                            force_topdown=final_topdown_step,
                        )
                        execution_trace.append(segment_result)
                        measured_position, _ = (
                            robot.get_local_planning_fk_pose()
                        )
                        if float(measured_position[2]) < minimum_transport_z:
                            raise RuntimeError(
                                "loaded cup dropped below "
                                "collision-clearance height"
                            )
                        if phase_plan.get("release_trigger_policy") == (
                            "xy_aligned_before_above_support"
                        ):
                            release_xy_limit = float(
                                phase_plan[
                                    "release_maximum_xy_error_m"
                                ]
                            )
                            release_xy_error = float(
                                np.linalg.norm(
                                    measured_position[:2] - target[:2]
                                )
                            )
                            if release_xy_error < release_xy_limit:
                                topdown_target_position = (
                                    measured_position.copy()
                                )
                                topdown_result = api.goto_pose(
                                    topdown_target_position,
                                    quaternion,
                                    gripper=gripper,
                                    translate_then_orient=False,
                                    max_cartesian_step_m=0.03,
                                    max_orientation_step_rad=float(
                                        ik_path_limits[
                                            "max_orientation_step_rad"
                                        ]
                                    ),
                                    check_cartesian_ik_error=False,
                                    verify_motion=False,
                                    require_pose_arrival=False,
                                )
                                topdown_result[
                                    "early_release_topdown_alignment"
                                ] = True
                                execution_trace.append(topdown_result)
                                (
                                    measured_position,
                                    measured_quaternion,
                                ) = (
                                    robot.get_local_planning_fk_pose()
                                )
                                measured_quaternion /= np.linalg.norm(
                                    measured_quaternion
                                )
                                normalized_topdown = (
                                    quaternion
                                    / np.linalg.norm(quaternion)
                                )
                                topdown_error = float(
                                    2.0
                                    * np.arccos(
                                        np.clip(
                                            abs(
                                                np.dot(
                                                    measured_quaternion,
                                                    normalized_topdown,
                                                )
                                            ),
                                            0.0,
                                            1.0,
                                        )
                                    )
                                )
                                if topdown_error > 0.10:
                                    raise RuntimeError(
                                        "early-release top-down alignment "
                                        f"failed: {topdown_error:.4f}rad"
                                    )
                                release_xy_error = float(
                                    np.linalg.norm(
                                        measured_position[:2] - target[:2]
                                    )
                                )
                                for correction_index in range(3):
                                    if release_xy_error < release_xy_limit:
                                        break
                                    xy_correction_target = (
                                        measured_position.copy()
                                    )
                                    xy_correction_target[:2] = target[:2]
                                    xy_correction_result = api.goto_pose(
                                        xy_correction_target,
                                        quaternion,
                                        gripper=gripper,
                                        translate_then_orient=False,
                                        max_cartesian_step_m=0.02,
                                        max_orientation_step_rad=float(
                                            ik_path_limits[
                                                "max_orientation_step_rad"
                                            ]
                                        ),
                                        check_cartesian_ik_error=False,
                                        verify_motion=False,
                                        require_pose_arrival=False,
                                    )
                                    xy_correction_result[
                                        "early_release_xy_correction_index"
                                    ] = correction_index + 1
                                    execution_trace.append(
                                        xy_correction_result
                                    )
                                    (
                                        measured_position,
                                        measured_quaternion,
                                    ) = robot.get_local_planning_fk_pose()
                                    release_xy_error = float(
                                        np.linalg.norm(
                                            measured_position[:2]
                                            - target[:2]
                                        )
                                    )
                                if release_xy_error < release_xy_limit:
                                    early_release_ready = True
                                    break
                    if early_release_ready:
                        break
                    commanded_position = np.asarray(destination, dtype=float)
                saved_above_support = phase_plan.get(
                    "saved_stage_joint_positions_rad", {}
                ).get("above_support")
                if saved_above_support is not None:
                    saved_joints = np.asarray(
                        saved_above_support, dtype=float
                    ).reshape(6)
                    saved_joint_result = api.move_joints_adaptive(
                        saved_joints,
                        waypoint_step_rad=0.015,
                        context=(
                            f"{phase}_above_support_saved_joint_position"
                        ),
                    )
                    measured_position, _ = (
                        robot.get_local_planning_fk_pose()
                    )
                    saved_position_error = float(
                        np.linalg.norm(measured_position - target)
                    )
                    if float(measured_position[2]) < minimum_transport_z:
                        raise RuntimeError(
                            "saved above_support joint position is below "
                            "the loaded transport height"
                        )
                    if saved_position_error > 0.025:
                        raise RuntimeError(
                            "saved above_support joint position missed its "
                            f"TCP target: {saved_position_error:.4f}m"
                        )
                    execution_trace.append(
                        {
                            "saved_joint_position_used": True,
                            "saved_joint_position_rad": (
                                saved_joints.tolist()
                            ),
                            "joint_result": saved_joint_result,
                            "measured_position": measured_position.tolist(),
                            "position_error_m": saved_position_error,
                        }
                    )
                result = {
                    "success": True,
                    "selected_route_index": selected_index,
                    "selected_route": selected_route,
                    "candidate_count": len(route_candidates),
                    "rejected_routes": rejected_routes,
                    "preflight_plan": selected_plan,
                    "execution": execution_trace,
                    "target_position": target.tolist(),
                    "measured_position": measured_position.tolist(),
                    "position_error_m": float(
                        np.linalg.norm(measured_position - target)
                    ),
                    "early_release_ready": early_release_ready,
                }
            else:
                critical_endpoint = stage_id in {
                    "lift",
                }
                result = api.goto_pose(
                    target,
                    quaternion,
                    gripper=gripper,
                    translate_then_orient=False,
                    max_cartesian_step_m=float(
                        ik_path_limits["max_cartesian_step_m"]
                    ),
                    max_orientation_step_rad=float(
                        ik_path_limits["max_orientation_step_rad"]
                    ),
                    check_cartesian_ik_error=critical_endpoint,
                    verify_motion=critical_endpoint,
                    require_pose_arrival=critical_endpoint,
                )
            if stage_id == "retract":
                state["released_phases"].add(phase)
            events.append(
                {
                    "event": "execute_cup_stage",
                    "phase": phase,
                    "stage_id": stage_id,
                    "target_tcp_base_m": target.tolist(),
                    "result": result,
                }
            )

        def verify_grasp() -> None:
            nonlocal completion_scope
            require_bound()
            success, details = api.verify_grasp(save_debug=True)
            events.append(
                {
                    "event": "verify_grasp",
                    "success": bool(success),
                    "details": details,
                }
            )
            if not success:
                raise RuntimeError(f"cup grasp retention failed: {details}")
            if args.stop_after_first_grasp:
                completion_scope = "first_grasp_only"
                events.append(
                    {
                        "event": "requested_stop_after_first_grasp",
                        "held_phase": plan["phase_order"][0],
                    }
                )
                raise RequestedStopAfterFirstGrasp

        def verify_cup_stack() -> None:
            require_bound()
            if state["released_phases"] != set(plan["phase_order"]):
                raise RuntimeError("both cup phases must release before verification")
            verification = plan.get("verification", {})
            base_cup = str(verification.get("base_cup", "purple cup"))
            top_cup = str(verification.get("top_cup", "green cup"))
            expected_top = verification.get(
                "expected_top_center_base_m",
                initial_cups[base_cup]["top_center_base_m"],
            )
            if not plan.get("runtime_cup_reperception_enabled", True):
                state["verified"] = True
                events.append(
                    {
                        "event": "cup_stack_visual_verification_skipped",
                        "reason": "fixed_user_measurement_mode",
                        "physical_result_verified": False,
                        "base_cup": base_cup,
                        "top_cup": top_cup,
                        "expected_top_center_base_m": expected_top,
                    }
                )
                return
            success, details = api.verify_cup_stack(
                expected_top_center_base_m=expected_top,
                cup_height_m=cup_height,
                nested_occlusion_expected=True,
                nested_top_object_name=top_cup,
            )
            events.append(
                {
                    "event": "verify_cup_stack",
                    "success": bool(success),
                    "details": details,
                }
            )
            if not success:
                raise RuntimeError(f"cup-stack verification failed: {details}")
            state["verified"] = True

        def return_home() -> None:
            require_bound()
            if not state["verified"]:
                raise RuntimeError("cannot finish before final cup-stack verification")
            result = api.move_joints_adaptive(
                planned_home_joints,
                waypoint_step_rad=0.025,
                context="return_to_captured_cup_task_home",
            )
            events.append({"event": "return_home", "result": result})

        namespace = {
            "bind_cup_plan": bind_cup_plan,
            "open_gripper": open_gripper,
            "execute_cup_stage": execute_cup_stage,
            "close_gripper": close_gripper,
            "verify_grasp": verify_grasp,
            "verify_cup_stack": verify_cup_stack,
            "return_home": return_home,
        }
        if set(namespace) != ALLOWED_CALLS:
            raise AssertionError("generated-code runtime API is incomplete")
        executable_code = code
        if resume_active:
            verify_indices = [
                index
                for index, operation in enumerate(expected)
                if operation["call"] == "verify_grasp"
            ]
            first_verify_index = verify_indices[0]
            resume_start_index = first_verify_index + 1
            if args.resume_after_second_lift:
                resume_start_index = verify_indices[1]
            elif args.resume_after_second_transfer:
                resume_start_index = next(
                    index
                    for index in range(verify_indices[1] + 1, len(expected))
                    if (
                        expected[index].get("phase")
                        == plan["phase_order"][1]
                        and expected[index].get("stage_id")
                        == "above_support"
                    )
                ) + 1
            elif args.resume_after_first_transfer:
                resume_start_index = next(
                    index
                    for index in range(first_verify_index + 1, len(expected))
                    if expected[index].get("stage_id") == "above_support"
                ) + 1
            resume_operations = [expected[0], *expected[resume_start_index:]]
            executable_code = "\n".join(
                f"{operation['call']}("
                + ", ".join(
                    f"{key}={value!r}"
                    for key, value in operation.items()
                    if key != "call"
                )
                + ")"
                for operation in resume_operations
            )
            events.append(
                {
                    "event": "validated_resume_program_selected",
                    "skipped_through_operation": first_verify_index,
                }
            )
        exec(
            compile(
                executable_code,
                "<approved-r1lite-cup-stack-code>",
                "exec",
            ),
            {"__builtins__": {}, **namespace},
            {},
        )
        passed = True
        error = None
    except RequestedStopAfterFirstGrasp:
        passed = True
        error = None
    except Exception as exc:
        passed = False
        error = repr(exc)
        raise
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=1.0)
        signal.signal(signal.SIGUSR1, previous_sigusr1)
        robot.close()
        audit = {
            "schema": "capx.r1lite.three_cup_stack.execution_audit.v1",
            "generated_at_unix": time.time(),
            "plan_sha256": args.expected_plan_sha256,
            "code_sha256": args.expected_code_sha256,
            "passed": passed,
            "completion_scope": completion_scope,
            "error": error,
            "torso_monitor": torso_monitor,
            "events": events,
        }
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(
                audit,
                indent=2,
                ensure_ascii=False,
                default=lambda value: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else repr(value)
                ),
            )
        )


if __name__ == "__main__":
    main()
