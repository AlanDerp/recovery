#!/usr/bin/env python3
"""Perceive and place three requested objects into an open box."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.control import R1LiteControlApi  # noqa: E402
from capx.integrations.r1lite.cup_stack_llm import (  # noqa: E402
    nearest_top_down_quaternion,
)
from scripts.r1lite_execute_llm_cup_stack import (  # noqa: E402
    build_free_transit_orientation_candidates,
    close_gripper_until_contact,
    load_robot,
    rotate_quaternion_about_base_axis,
)

OBJECT_ALIASES = {
    "tennis": "tennis ball",
    "tennis-ball": "tennis ball",
    "tennis_ball": "tennis ball",
    "tennis ball": "tennis ball",
    "peach": "peach",
    "apple": "apple",
}
REQUIRED_OBJECTS = {"tennis ball", "peach", "apple"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("objects", nargs=3)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--torso-hold-status", type=Path)
    parser.add_argument("--left-arm-hold-status", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--perception-only", action="store_true")
    parser.add_argument("--maximum-initial-joint-delta-rad", type=float, default=0.05)
    parser.add_argument("--maximum-torso-drift-rad", type=float, default=0.03)
    return parser.parse_args()


def normalize_object_order(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(
        OBJECT_ALIASES.get(str(value).strip().lower(), "")
        for value in values
    )
    if len(normalized) != 3 or set(normalized) != REQUIRED_OBJECTS:
        raise ValueError(
            "object order must be one unique permutation of "
            "tennis, peach, apple"
        )
    return normalized


def quaternion_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1e-9:
        raise ValueError("quaternion is zero")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def predicted_held_object_center(
    tcp_position: np.ndarray,
    tcp_quaternion_xyzw: np.ndarray,
    held_offset_tool_m: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(tcp_position, dtype=float)
        + quaternion_matrix(tcp_quaternion_xyzw)
        @ np.asarray(held_offset_tool_m, dtype=float)
    )


def build_grasp_orientation_candidates(
    topdown_quaternion_xyzw: np.ndarray,
) -> list[np.ndarray]:
    """Return top-down-first grasp poses with at most 20 degrees of tilt."""
    topdown = np.asarray(topdown_quaternion_xyzw, dtype=float).reshape(4)
    topdown /= np.linalg.norm(topdown)
    candidates = [topdown]
    z_axis = np.array([0.0, 0.0, 1.0])
    for yaw_deg in (30.0, -30.0, 60.0, -60.0, 90.0, -90.0):
        candidates.append(
            rotate_quaternion_about_base_axis(
                topdown,
                z_axis,
                np.deg2rad(yaw_deg),
            )
        )
    for tilt_deg in (10.0, -10.0, 15.0, -15.0, 20.0, -20.0):
        for axis in (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ):
            candidates.append(
                rotate_quaternion_about_base_axis(
                    topdown,
                    axis,
                    np.deg2rad(tilt_deg),
                )
            )
    return candidates


def grasp_tcp_for_orientation(
    object_center: np.ndarray,
    quaternion_xyzw: np.ndarray,
    tcp_to_fingertip_m: float,
) -> np.ndarray:
    """Place the fingertip reference at the object center for any tilt."""
    fingertip_from_tcp_tool = np.array(
        [0.0, 0.0, float(tcp_to_fingertip_m)]
    )
    return (
        np.asarray(object_center, dtype=float).reshape(3)
        - quaternion_matrix(quaternion_xyzw) @ fingertip_from_tcp_tool
    )


def build_grasp_ik_seed_candidates(
    current_joints: np.ndarray,
) -> list[np.ndarray]:
    """Provide nearby elbow branches while preserving the measured start."""
    current = np.asarray(current_joints, dtype=float).reshape(6)
    offsets = (
        (0.0, 0.0),
        (-0.20, 0.0),
        (0.20, 0.0),
        (0.0, -0.20),
        (0.0, 0.20),
        (-0.20, -0.20),
        (-0.20, 0.20),
    )
    seeds = []
    for joint2_offset, joint4_offset in offsets:
        seed = current.copy()
        seed[1] = np.clip(seed[1] + joint2_offset, 0.05, 3.05)
        seed[3] = np.clip(seed[3] + joint4_offset, -1.45, 1.45)
        seeds.append(seed)
    return seeds


def converge_seeded_grasp_endpoint(
    robot: Any,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    *,
    maximum_attempts: int = 12,
    position_tolerance_m: float = 0.025,
) -> tuple[np.ndarray, float]:
    """Iterate RelaxedIK at a fixed endpoint instead of using its first frame."""
    best_joints: np.ndarray | None = None
    best_error = float("inf")
    torso_joints = robot.get_torso_joints()
    for _ in range(int(maximum_attempts)):
        joints = robot.get_relaxed_ik_candidate(
            target_position,
            target_quaternion,
            timeout_s=1.0,
            validate_joint_target=False,
        )
        endpoint_fk, _ = robot.get_local_planning_fk_pose(
            joints,
            torso_joints,
        )
        error = float(
            np.linalg.norm(
                endpoint_fk
                - np.asarray(target_position, dtype=float).reshape(3)
            )
        )
        if error < best_error:
            best_error = error
            best_joints = joints.copy()
        if error <= float(position_tolerance_m):
            robot.validate_arm_joint_feedback_target(
                joints,
                context="converged RelaxedIK grasp endpoint",
            )
            return joints, error
    raise RuntimeError(
        "seeded grasp endpoint did not converge: "
        f"best_error={best_error:.4f}m, attempts={maximum_attempts}, "
        f"best_joints={None if best_joints is None else best_joints.tolist()}"
    )


def execute_frozen_pose_plan(
    robot: Any,
    plan: dict[str, Any],
    target_position: np.ndarray,
    *,
    context: str,
    position_tolerance_m: float = 0.025,
) -> dict[str, Any]:
    """Execute exactly the joint waypoints that passed candidate validation."""
    execution = []
    waypoints = plan["waypoints"]
    for index, waypoint in enumerate(waypoints, start=1):
        execution.append(
            robot.move_joint_waypoints_adaptive(
                waypoint["joint_feedback"],
                waypoint_step_rad=float(
                    robot.goto_pose_joint_waypoint_step_rad
                ),
                context=(
                    f"{context}_{index:03d}_of_{len(waypoints):03d}"
                ),
            )
        )
    measured_position, measured_quaternion = (
        robot.get_local_planning_fk_pose()
    )
    position_error = float(
        np.linalg.norm(
            measured_position
            - np.asarray(target_position, dtype=float).reshape(3)
        )
    )
    if position_error > float(position_tolerance_m):
        raise RuntimeError(
            f"{context} missed frozen-plan endpoint: "
            f"position_error={position_error:.4f}m"
        )
    return {
        "success": True,
        "plan": plan,
        "execution": execution,
        "target_position": np.asarray(
            target_position,
            dtype=float,
        ).tolist(),
        "measured_position": measured_position.tolist(),
        "measured_quaternion_xyzw": measured_quaternion.tolist(),
        "position_error_m": position_error,
    }


def object_center_is_over_box(
    object_center: np.ndarray,
    box: dict[str, Any],
    *,
    radius_m: float,
) -> tuple[bool, dict[str, Any]]:
    center = np.asarray(object_center, dtype=float).reshape(3)
    opening_min = np.asarray(box["opening_xy_min_base_m"], dtype=float)
    opening_max = np.asarray(box["opening_xy_max_base_m"], dtype=float)
    rim_z = float(box["rim_z_base_m"])
    xy_inside = bool(
        np.all(center[:2] >= opening_min)
        and np.all(center[:2] <= opening_max)
    )
    bottom_clearance = float(center[2] - float(radius_m) - rim_z)
    safe_height = bool(0.015 <= bottom_clearance <= 0.30)
    return xy_inside and safe_height, {
        "predicted_object_center_base_m": center.tolist(),
        "opening_xy_min_base_m": opening_min.tolist(),
        "opening_xy_max_base_m": opening_max.tolist(),
        "box_rim_z_base_m": rim_z,
        "object_bottom_clearance_above_rim_m": bottom_clearance,
        "xy_inside": xy_inside,
        "safe_release_height": safe_height,
    }


def main() -> None:
    args = parse_args()
    object_order = normalize_object_order(args.objects)
    reference_plan = json.loads(args.reference_plan.read_text())
    planned_home_joints = np.asarray(
        reference_plan["rgbd_scene"]["home_arm_joints_rad"], dtype=float
    ).reshape(6)
    planned_home_quaternion = np.asarray(
        reference_plan["home_gripper_quaternion_xyzw"], dtype=float
    ).reshape(4)
    topdown_quaternion, _ = nearest_top_down_quaternion(
        planned_home_quaternion
    )
    torso_pid: int | None = None
    torso_reference: np.ndarray | None = None
    left_hold_pid: int | None = None
    if not args.perception_only:
        if args.torso_hold_status is None:
            raise ValueError("real execution requires --torso-hold-status")
        if args.left_arm_hold_status is None:
            raise ValueError("real execution requires --left-arm-hold-status")
        torso_status = json.loads(args.torso_hold_status.read_text())
        if torso_status.get("violation"):
            raise RuntimeError("torso hold reported a startup violation")
        torso_pid = int(torso_status["pid"])
        os.kill(torso_pid, 0)
        torso_reference = np.asarray(
            torso_status["reference_position_rad"], dtype=float
        ).reshape(3)
        left_status = json.loads(args.left_arm_hold_status.read_text())
        if left_status.get("violation"):
            raise RuntimeError("left-arm Home hold reported a startup violation")
        left_hold_pid = int(left_status["pid"])
        os.kill(left_hold_pid, 0)

    robot = load_robot(args.config)
    robot.object_prompts.update(
        {
            "tennis ball": "yellow tennis ball on the table",
            "peach": "red fruit",
            "apple": "green apple",
            "cardboard box": "open cardboard box on the table",
        }
    )
    api = R1LiteControlApi(robot)
    events: list[dict[str, Any]] = []
    passed = False
    error: str | None = None
    monitor_stop = threading.Event()
    torso_violation = threading.Event()
    left_hold_violation = threading.Event()
    safety_reason = ["right-arm task safety monitor aborted"]
    previous_sigusr1 = signal.getsignal(signal.SIGUSR1)

    def torso_abort(_signum: int, _frame: Any) -> None:
        raise RuntimeError(safety_reason[0])

    def monitor_torso() -> None:
        if torso_pid is None or torso_reference is None:
            return
        while not monitor_stop.wait(0.02):
            try:
                os.kill(torso_pid, 0)
                torso = robot.get_torso_joints()
                drift = float(np.max(np.abs(torso - torso_reference)))
                if drift > float(args.maximum_torso_drift_rad):
                    torso_violation.set()
                    safety_reason[0] = (
                        "torso drift exceeded the put-general safety limit"
                    )
                    os.kill(os.getpid(), signal.SIGUSR1)
                    return
                if left_hold_pid is not None:
                    os.kill(left_hold_pid, 0)
                    left_status = json.loads(
                        args.left_arm_hold_status.read_text()
                    )
                    if left_status.get("violation"):
                        left_hold_violation.set()
                        safety_reason[0] = (
                            "left-arm Home hold exceeded its safety limit"
                        )
                        os.kill(os.getpid(), signal.SIGUSR1)
                        return
            except OSError:
                torso_violation.set()
                safety_reason[0] = "a put-general hold process exited"
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    signal.signal(signal.SIGUSR1, torso_abort)
    monitor = threading.Thread(target=monitor_torso, daemon=True)
    try:
        if not args.perception_only:
            monitor.start()
        initial_joints = robot.get_arm_joints()
        initial_delta = float(
            np.max(np.abs(initial_joints - planned_home_joints))
        )
        events.append(
            {
                "event": "initial_task_home_check",
                "home_error_rad": initial_delta,
                "requires_realign": bool(
                    not args.perception_only
                    and initial_delta
                    > float(args.maximum_initial_joint_delta_rad)
                ),
            }
        )
        scene = api.locate_put_objects_head_rgbd(save_debug=True)
        events.append(
            {
                "event": "initial_head_rgbd_scene",
                "object_order": list(object_order),
                "detections": scene,
            }
        )
        if args.perception_only:
            passed = True
            return

        moca_startup_reference = robot.get_arm_joints()
        robot.enable_arm_joint_control_without_function_reset()
        moca_gripper_ready_deadline = time.monotonic() + 8.0
        moca_previous_joints = moca_startup_reference.copy()
        moca_maximum_startup_drift = 0.0
        startup_motion_limit_rad = float(
            robot.adaptive_max_measured_step_rad
        )
        while time.monotonic() < moca_gripper_ready_deadline:
            moca_current_joints = robot.get_arm_joints()
            startup_step = float(
                np.max(np.abs(moca_current_joints - moca_previous_joints))
            )
            startup_drift = float(
                np.max(
                    np.abs(
                        moca_current_joints
                        - moca_startup_reference
                    )
                )
            )
            moca_maximum_startup_drift = max(
                moca_maximum_startup_drift,
                startup_drift,
            )
            if (
                startup_step > startup_motion_limit_rad
                or startup_drift > startup_motion_limit_rad
            ):
                raise RuntimeError(
                    "right arm exceeded the MOCA startup safety envelope: "
                    f"step={startup_step:.4f}rad, "
                    f"drift={startup_drift:.4f}rad, "
                    f"limit={startup_motion_limit_rad:.4f}rad"
                )
            moca_previous_joints = moca_current_joints
            time.sleep(0.05)
        startup_home_joints = robot.get_arm_joints()
        startup_home_error = float(
            np.max(np.abs(startup_home_joints - planned_home_joints))
        )
        startup_realign = None
        if startup_home_error > float(args.maximum_initial_joint_delta_rad):
            startup_realign = api.move_joints_adaptive(
                planned_home_joints,
                waypoint_step_rad=0.100,
                context="put_general_moca_startup_realign_home",
            )
            startup_home_joints = robot.get_arm_joints()
            startup_home_error = float(
                np.max(np.abs(startup_home_joints - planned_home_joints))
            )
        if startup_home_error > float(args.maximum_initial_joint_delta_rad):
            raise RuntimeError(
                "right arm did not return to task Home after MOCA startup: "
                f"error={startup_home_error:.4f}rad"
            )
        events.append(
            {
                "event": "moca_startup_stabilized",
                "maximum_startup_drift_rad": moca_maximum_startup_drift,
                "home_error_rad": startup_home_error,
                "realign": startup_realign,
            }
        )
        left_deadline = time.monotonic() + 30.0
        while time.monotonic() < left_deadline:
            os.kill(int(left_hold_pid), 0)
            left_status = json.loads(args.left_arm_hold_status.read_text())
            if left_status.get("violation"):
                raise RuntimeError("left-arm Home hold violated during alignment")
            if left_status.get("arrived"):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("left arm did not reach Home before right-arm motion")
        if not robot.relaxed_ik_probe_ready():
            raise RuntimeError(
                "isolated right-arm RelaxedIK is unavailable before "
                "grasp candidate search"
            )
        tcp_to_fingertip_m = 0.078
        grasp_contact_height_offset_m = 0.012
        open_command = 100.0
        for phase_index, object_name in enumerate(object_order, start=1):
            detection = scene[object_name]
            object_center = np.asarray(
                detection["position_base_m"], dtype=float
            )
            radius = float(detection["estimated_radius_m"])
            box = scene["cardboard box"]
            grasp_orientation_rejections: list[str] = []
            grasp_quaternion: np.ndarray | None = None
            grasp_tcp: np.ndarray | None = None
            pregrasp_tcp: np.ndarray | None = None
            grasp_plan: dict[str, Any] | None = None
            selected_grasp_seed: np.ndarray | None = None
            current_tcp, current_quaternion = (
                robot.get_local_planning_fk_pose()
            )
            grasp_seed_candidates = build_grasp_ik_seed_candidates(
                robot.get_arm_joints()
            )
            for candidate_index, candidate in enumerate(
                build_grasp_orientation_candidates(topdown_quaternion)
            ):
                candidate_grasp_tcp = grasp_tcp_for_orientation(
                    object_center
                    + np.array([0.0, 0.0, grasp_contact_height_offset_m]),
                    candidate,
                    tcp_to_fingertip_m,
                )
                candidate_pregrasp_tcp = (
                    candidate_grasp_tcp + np.array([0.0, 0.0, 0.12])
                )
                safe_route_z = max(
                    float(current_tcp[2]) + 0.08,
                    float(candidate_pregrasp_tcp[2]) + 0.06,
                    1.08,
                )
                high_current = current_tcp.copy()
                high_current[2] = safe_route_z
                high_above_object = candidate_pregrasp_tcp.copy()
                high_above_object[2] = safe_route_z
                for seed_index, seed in enumerate(grasp_seed_candidates):
                    try:
                        robot.seed_relaxed_ik_probe(seed)
                        endpoint_joints, endpoint_error = (
                            converge_seeded_grasp_endpoint(
                                robot,
                                candidate_grasp_tcp,
                                candidate,
                            )
                        )
                        if not -1.52 <= float(endpoint_joints[3]) <= 1.52:
                            raise RuntimeError(
                                "seeded grasp endpoint approaches J4 limit: "
                                f"{endpoint_joints[3]:.4f}rad"
                            )
                        candidate_plan = api.plan_pose_waypoints(
                            candidate_grasp_tcp,
                            candidate,
                            via_positions=[
                                high_current.tolist(),
                                high_above_object.tolist(),
                                candidate_pregrasp_tcp.tolist(),
                            ],
                            via_quaternions_xyzw=[
                                current_quaternion.tolist(),
                                candidate.tolist(),
                                candidate.tolist(),
                            ],
                            ik_seed_joints=seed,
                            max_cartesian_step_m=0.02,
                            max_orientation_step_rad=0.25,
                            check_cartesian_ik_error=False,
                        )
                        planned_joints = np.asarray(
                            [
                                waypoint["joint_feedback"]
                                for waypoint in candidate_plan["waypoints"]
                            ],
                            dtype=float,
                        )
                        joint4_min = float(np.min(planned_joints[:, 3]))
                        joint4_max = float(np.max(planned_joints[:, 3]))
                        if joint4_min < -1.52 or joint4_max > 1.52:
                            raise RuntimeError(
                                "grasp path approaches J4 physical limit: "
                                f"range=[{joint4_min:.4f}, "
                                f"{joint4_max:.4f}]rad"
                            )
                        endpoint_error = float(
                            candidate_plan["waypoints"][-1][
                                "fk_position_error_m"
                            ]
                        )
                        if endpoint_error > 0.025:
                            raise RuntimeError(
                                "grasp endpoint error is "
                                f"{endpoint_error:.4f}m"
                            )
                        grasp_quaternion = candidate
                        grasp_tcp = candidate_grasp_tcp
                        pregrasp_tcp = candidate_pregrasp_tcp
                        grasp_plan = candidate_plan
                        selected_grasp_seed = seed
                        break
                    except Exception as exc:
                        grasp_orientation_rejections.append(
                            f"orientation={candidate_index}, "
                            f"seed={seed_index}: {exc!r}"
                        )
                if grasp_plan is not None:
                    break
            if (
                grasp_quaternion is None
                or grasp_tcp is None
                or pregrasp_tcp is None
                or grasp_plan is None
                or selected_grasp_seed is None
            ):
                raise RuntimeError(
                    f"no near-top-down grasp orientation for {object_name}: "
                    f"{grasp_orientation_rejections}"
                )
            api.command_gripper(open_command, duration_s=3.0)
            grasp_result = execute_frozen_pose_plan(
                robot,
                grasp_plan,
                grasp_tcp,
                context=f"put_general_phase_{phase_index}_frozen_grasp",
            )
            close_result = close_gripper_until_contact(
                api,
                minimum_command=0.0,
                open_position=open_command,
            )
            grasp_ok, grasp_details = api.verify_grasp(save_debug=True)
            if not grasp_ok:
                raise RuntimeError(
                    f"{object_name} grasp verification failed: {grasp_details}"
                )
            measured_tcp, measured_quaternion = (
                robot.get_local_planning_fk_pose()
            )
            held_offset_tool = (
                quaternion_matrix(measured_quaternion).T
                @ (object_center - measured_tcp)
            )
            hold_command = float(close_result["hold_command"])
            lift_tcp = measured_tcp + np.array([0.0, 0.0, 0.12])
            lift_result = api.goto_pose(
                lift_tcp,
                measured_quaternion,
                gripper=hold_command,
                max_cartesian_step_m=0.03,
                max_orientation_step_rad=0.25,
                check_cartesian_ik_error=False,
                position_arrival_tolerance_m=0.025,
            )
            rear_tcp = lift_tcp.copy()
            rear_tcp[0] = max(0.08, rear_tcp[0] - 0.12)
            rear_result = api.goto_pose(
                rear_tcp,
                measured_quaternion,
                gripper=hold_command,
                max_cartesian_step_m=0.03,
                max_orientation_step_rad=0.25,
                check_cartesian_ik_error=False,
                position_arrival_tolerance_m=0.025,
            )
            box_min = np.asarray(box["opening_xy_min_base_m"], dtype=float)
            box_max = np.asarray(box["opening_xy_max_base_m"], dtype=float)
            box_center_xy = 0.5 * (box_min + box_max)
            desired_object_center = np.array(
                [
                    box_center_xy[0],
                    box_center_xy[1],
                    float(box["rim_z_base_m"]) + radius + 0.08,
                ],
                dtype=float,
            )
            current_tcp, current_quaternion = (
                robot.get_local_planning_fk_pose()
            )
            high_tcp = current_tcp.copy()
            high_tcp[2] = max(
                float(current_tcp[2]),
                float(desired_object_center[2] + 0.08),
                1.03,
            )
            high_result = api.goto_pose(
                high_tcp,
                current_quaternion,
                gripper=hold_command,
                max_cartesian_step_m=0.03,
                max_orientation_step_rad=0.25,
                check_cartesian_ik_error=False,
                position_arrival_tolerance_m=0.025,
            )
            orientation_rejections: list[str] = []
            transfer_result: dict[str, Any] | None = None
            selected_quaternion: np.ndarray | None = None
            for candidate in build_free_transit_orientation_candidates(
                current_quaternion,
                topdown_quaternion,
            ):
                target_tcp = (
                    desired_object_center
                    - quaternion_matrix(candidate) @ held_offset_tool
                )
                try:
                    candidate_plan = api.plan_pose_waypoints(
                        target_tcp,
                        candidate,
                        max_cartesian_step_m=0.03,
                        max_orientation_step_rad=0.25,
                        check_cartesian_ik_error=False,
                    )
                    endpoint_error = float(
                        candidate_plan["waypoints"][-1][
                            "fk_position_error_m"
                        ]
                    )
                    if endpoint_error > 0.03:
                        raise RuntimeError(
                            f"endpoint error is {endpoint_error:.4f}m"
                        )
                    transfer_result = api.goto_pose(
                        target_tcp,
                        candidate,
                        gripper=hold_command,
                        translate_then_orient=False,
                        max_cartesian_step_m=0.03,
                        max_orientation_step_rad=0.25,
                        check_cartesian_ik_error=False,
                        position_arrival_tolerance_m=0.03,
                    )
                    selected_quaternion = candidate
                    break
                except Exception as exc:
                    orientation_rejections.append(repr(exc))
            if transfer_result is None or selected_quaternion is None:
                raise RuntimeError(
                    f"no transport orientation reaches the box: "
                    f"{orientation_rejections}"
                )
            release_checks = []
            for correction_index in range(4):
                actual_tcp, actual_quaternion = (
                    robot.get_local_planning_fk_pose()
                )
                predicted_center = predicted_held_object_center(
                    actual_tcp,
                    actual_quaternion,
                    held_offset_tool,
                )
                ready, release_details = object_center_is_over_box(
                    predicted_center,
                    box,
                    radius_m=radius,
                )
                release_details["correction_index"] = correction_index
                release_checks.append(release_details)
                if ready:
                    break
                if correction_index == 3:
                    continue
                correction = desired_object_center - predicted_center
                corrected_tcp = actual_tcp + correction
                api.goto_pose(
                    corrected_tcp,
                    selected_quaternion,
                    gripper=hold_command,
                    translate_then_orient=False,
                    max_cartesian_step_m=0.02,
                    max_orientation_step_rad=0.25,
                    check_cartesian_ik_error=False,
                    require_pose_arrival=False,
                )
            else:
                raise RuntimeError(
                    f"{object_name} is not safely above the box: "
                    f"{release_checks[-1]}"
                )
            open_feedback = api.command_gripper(
                open_command, duration_s=3.0
            )
            time.sleep(0.6)
            retract_tcp, retract_quaternion = (
                robot.get_local_planning_fk_pose()
            )
            retract_tcp[2] += 0.10
            retract_result = api.goto_pose(
                retract_tcp,
                retract_quaternion,
                gripper=open_command,
                max_cartesian_step_m=0.03,
                max_orientation_step_rad=0.25,
                check_cartesian_ik_error=False,
                require_pose_arrival=False,
            )
            home_result = api.move_joints_adaptive(
                planned_home_joints,
                waypoint_step_rad=0.100,
                context=f"put_general_phase_{phase_index}_return_home",
            )
            events.append(
                {
                    "event": "put_object",
                    "phase_index": phase_index,
                    "object": object_name,
                    "grasp_contact_height_offset_m": (
                        grasp_contact_height_offset_m
                    ),
                    "grasp_quaternion_xyzw": grasp_quaternion.tolist(),
                    "grasp_ik_seed_joints": selected_grasp_seed.tolist(),
                    "grasp_orientation_rejections": (
                        grasp_orientation_rejections
                    ),
                    "grasp": grasp_result,
                    "close": close_result,
                    "lift": lift_result,
                    "rear_retract": rear_result,
                    "high_transfer": high_result,
                    "transfer": transfer_result,
                    "orientation_rejections": orientation_rejections,
                    "held_offset_tool_m": held_offset_tool.tolist(),
                    "release_checks": release_checks,
                    "open_feedback": float(open_feedback),
                    "retract": retract_result,
                    "return_home": home_result,
                }
            )
        passed = True
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        monitor_stop.set()
        if monitor.is_alive():
            monitor.join(timeout=1.0)
        signal.signal(signal.SIGUSR1, previous_sigusr1)
        motion_fault = robot.motion_fault_record()
        robot.close()
        audit = {
            "schema": "capx.r1lite.put_general.execution_audit.v1",
            "generated_at_unix": time.time(),
            "object_order": list(object_order),
            "perception_only": bool(args.perception_only),
            "passed": passed,
            "error": error,
            "motion_fault": motion_fault,
            "torso_violation": torso_violation.is_set(),
            "left_hold_violation": left_hold_violation.is_set(),
            "events": events,
        }
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False, default=str)
            + "\n"
        )


if __name__ == "__main__":
    main()
