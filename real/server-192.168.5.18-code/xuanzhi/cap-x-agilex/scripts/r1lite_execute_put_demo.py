#!/usr/bin/env python3
"""Execute the fixed-position put-demo pipeline."""

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
    close_gripper_until_contact,
    load_robot,
)
from scripts.r1lite_execute_put_general import (  # noqa: E402
    grasp_tcp_for_orientation,
)

OBJECT_ALIASES = {
    "peach": "peach",
    "桃子": "peach",
    "tennis": "tennis ball",
    "tennis-ball": "tennis ball",
    "tennis_ball": "tennis ball",
    "tennis ball": "tennis ball",
    "网球": "tennis ball",
    "apple": "apple",
    "苹果": "apple",
}
REQUIRED_OBJECTS = {"peach", "tennis ball", "apple"}
OBJECT_TO_CUP = {
    "peach": "orange cup",
    "tennis ball": "purple cup",
    "apple": "green cup",
}
TCP_TO_FINGERTIP_M = 0.078
GRASP_BELOW_OBJECT_CENTER_M = 0.07
PREGRASP_CLEARANCE_M = 0.10
LIFT_DISTANCE_M = 0.10
FORWARD_RELEASE_DISTANCE_M = 0.15
BOX_CENTER_FORWARD_DISTANCE_M = 0.20
OPEN_GRIPPER_COMMAND = 100.0
BEIZI_ENTRY_CARTESIAN_STEP_M = 0.06
BEIZI_ENTRY_ORIENTATION_STEP_RAD = 0.25
BEIZI_GRASP_DESCENT_STEPS = 12
BEIZI_MOTION_PARAMETERS = {
    "r1lite_adaptive_joint_control_enabled": False,
    "adaptive_max_command_excursion_rad": 0.10,
    "adaptive_max_target_step_rad": 0.025,
    "adaptive_max_measured_step_rad": 0.08,
    "adaptive_settle_s": 1.0,
    "max_joint_step": 0.15,
    "goto_pose_joint_waypoint_step_rad": 0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("objects", nargs=3)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--torso-hold-status", type=Path, required=True)
    parser.add_argument("--left-arm-hold-status", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
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
            "peach, tennis, apple"
        )
    return normalized


def load_fixed_object_centers(
    reference_plan: dict[str, Any],
) -> dict[str, np.ndarray]:
    cups = reference_plan["rgbd_scene"]["cups"]
    return {
        object_name: np.asarray(
            cups[cup_name]["top_center_base_m"],
            dtype=float,
        ).reshape(3)
        for object_name, cup_name in OBJECT_TO_CUP.items()
    }


def build_put_demo_targets(
    object_center: np.ndarray,
    topdown_quaternion_xyzw: np.ndarray,
) -> dict[str, np.ndarray]:
    center = np.asarray(object_center, dtype=float).reshape(3)
    grasp_point = center - np.array(
        [0.0, 0.0, GRASP_BELOW_OBJECT_CENTER_M]
    )
    grasp_tcp = grasp_tcp_for_orientation(
        grasp_point,
        topdown_quaternion_xyzw,
        TCP_TO_FINGERTIP_M,
    )
    pregrasp_tcp = grasp_tcp + np.array([0.0, 0.0, PREGRASP_CLEARANCE_M])
    lift_tcp = grasp_tcp + np.array([0.0, 0.0, LIFT_DISTANCE_M])
    release_tcp = lift_tcp + np.array(
        [FORWARD_RELEASE_DISTANCE_M, 0.0, 0.0]
    )
    box_center = center + np.array(
        [BOX_CENTER_FORWARD_DISTANCE_M, 0.0, 0.0]
    )
    return {
        "object_center": center,
        "grasp_point": grasp_point,
        "grasp_tcp": grasp_tcp,
        "pregrasp_tcp": pregrasp_tcp,
        "lift_tcp": lift_tcp,
        "release_tcp": release_tcp,
        "box_center": box_center,
    }


def apply_beizi_motion_parameters(robot: Any) -> None:
    for name, value in BEIZI_MOTION_PARAMETERS.items():
        setattr(robot, name, value)


def execute_beizi_grasp_entry(
    api: R1LiteControlApi,
    targets: dict[str, np.ndarray],
    topdown_quaternion: np.ndarray,
) -> dict[str, Any]:
    """Use the beizi Home-to-pregrasp and stepped vertical descent policy."""
    pregrasp_result = api.goto_pose(
        targets["pregrasp_tcp"],
        topdown_quaternion,
        gripper=OPEN_GRIPPER_COMMAND,
        translate_then_orient=False,
        max_cartesian_step_m=BEIZI_ENTRY_CARTESIAN_STEP_M,
        max_orientation_step_rad=BEIZI_ENTRY_ORIENTATION_STEP_RAD,
        check_cartesian_ik_error=False,
        verify_motion=False,
        require_pose_arrival=False,
    )
    grasp_result = api.descend_z_in_steps(
        targets["grasp_tcp"],
        topdown_quaternion,
        start_z=float(targets["pregrasp_tcp"][2]),
        num_steps=BEIZI_GRASP_DESCENT_STEPS,
        gripper=OPEN_GRIPPER_COMMAND,
        check_final_cartesian_error_only=True,
    )
    return {
        "entry_policy": "beizi_direct_pregrasp_then_vertical_descent",
        "pregrasp": pregrasp_result,
        "grasp": grasp_result,
    }


def main() -> None:
    args = parse_args()
    object_order = normalize_object_order(args.objects)
    reference_plan = json.loads(args.reference_plan.read_text())
    object_centers = load_fixed_object_centers(reference_plan)
    planned_home_joints = np.asarray(
        reference_plan["rgbd_scene"]["home_arm_joints_rad"],
        dtype=float,
    ).reshape(6)
    planned_home_quaternion = np.asarray(
        reference_plan["home_gripper_quaternion_xyzw"],
        dtype=float,
    ).reshape(4)
    topdown_quaternion, _ = nearest_top_down_quaternion(
        planned_home_quaternion
    )

    torso_status = json.loads(args.torso_hold_status.read_text())
    if torso_status.get("violation"):
        raise RuntimeError("torso hold reported a startup violation")
    torso_pid = int(torso_status["pid"])
    os.kill(torso_pid, 0)
    torso_reference = np.asarray(
        torso_status["reference_position_rad"],
        dtype=float,
    ).reshape(3)
    left_status = json.loads(args.left_arm_hold_status.read_text())
    if left_status.get("violation"):
        raise RuntimeError("left-arm Home hold reported a startup violation")
    left_hold_pid = int(left_status["pid"])
    os.kill(left_hold_pid, 0)

    robot = load_robot(args.config)
    apply_beizi_motion_parameters(robot)
    api = R1LiteControlApi(robot)
    events: list[dict[str, Any]] = [
        {
            "event": "fixed_scene",
            "frame_id": "base_link",
            "object_order": list(object_order),
            "object_centers_base_m": {
                name: center.tolist()
                for name, center in object_centers.items()
            },
            "forward_axis": "+x",
            "forward_release_distance_m": FORWARD_RELEASE_DISTANCE_M,
            "box_center_forward_distance_m": (
                BOX_CENTER_FORWARD_DISTANCE_M
            ),
            "release_height_checked": False,
        }
    ]
    passed = False
    error: str | None = None
    monitor_stop = threading.Event()
    torso_violation = threading.Event()
    left_hold_violation = threading.Event()
    safety_reason = ["put-demo safety monitor aborted"]
    previous_sigusr1 = signal.getsignal(signal.SIGUSR1)

    def safety_abort(_signum: int, _frame: Any) -> None:
        raise RuntimeError(safety_reason[0])

    def monitor_holds() -> None:
        while not monitor_stop.wait(0.02):
            try:
                os.kill(torso_pid, 0)
                os.kill(left_hold_pid, 0)
                torso = robot.get_torso_joints()
                drift = float(np.max(np.abs(torso - torso_reference)))
                if drift > float(args.maximum_torso_drift_rad):
                    torso_violation.set()
                    safety_reason[0] = (
                        "torso drift exceeded the put-demo safety limit"
                    )
                    os.kill(os.getpid(), signal.SIGUSR1)
                    return
                status = json.loads(args.left_arm_hold_status.read_text())
                if status.get("violation"):
                    left_hold_violation.set()
                    safety_reason[0] = (
                        "left-arm Home hold exceeded its safety limit"
                    )
                    os.kill(os.getpid(), signal.SIGUSR1)
                    return
            except OSError:
                torso_violation.set()
                safety_reason[0] = "a put-demo hold process exited"
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    signal.signal(signal.SIGUSR1, safety_abort)
    monitor = threading.Thread(target=monitor_holds, daemon=True)
    try:
        monitor.start()
        initial_joints = robot.get_arm_joints()
        initial_home_error = float(
            np.max(np.abs(initial_joints - planned_home_joints))
        )
        events.append(
            {
                "event": "initial_task_home_check",
                "home_error_rad": initial_home_error,
                "requires_realign": bool(
                    initial_home_error
                    > float(args.maximum_initial_joint_delta_rad)
                ),
            }
        )

        moca_startup_reference = robot.get_arm_joints()
        robot.enable_arm_joint_control_without_function_reset()
        startup_deadline = time.monotonic() + 8.0
        previous_joints = moca_startup_reference.copy()
        maximum_startup_drift = 0.0
        startup_motion_limit = float(robot.adaptive_max_measured_step_rad)
        while time.monotonic() < startup_deadline:
            current_joints = robot.get_arm_joints()
            startup_step = float(
                np.max(np.abs(current_joints - previous_joints))
            )
            startup_drift = float(
                np.max(np.abs(current_joints - moca_startup_reference))
            )
            maximum_startup_drift = max(
                maximum_startup_drift,
                startup_drift,
            )
            if (
                startup_step > startup_motion_limit
                or startup_drift > startup_motion_limit
            ):
                raise RuntimeError(
                    "right arm exceeded the MOCA startup safety envelope: "
                    f"step={startup_step:.4f}rad, "
                    f"drift={startup_drift:.4f}rad, "
                    f"limit={startup_motion_limit:.4f}rad"
                )
            previous_joints = current_joints
            time.sleep(0.05)

        startup_home_error = float(
            np.max(
                np.abs(robot.get_arm_joints() - planned_home_joints)
            )
        )
        startup_realign = None
        if startup_home_error > float(args.maximum_initial_joint_delta_rad):
            startup_realign = api.move_joints_adaptive(
                planned_home_joints,
                waypoint_step_rad=0.10,
                context="put_demo_moca_startup_realign_home",
            )
            startup_home_error = float(
                np.max(
                    np.abs(robot.get_arm_joints() - planned_home_joints)
                )
            )
        if startup_home_error > float(args.maximum_initial_joint_delta_rad):
            raise RuntimeError(
                "right arm did not return to put-demo Home: "
                f"error={startup_home_error:.4f}rad"
            )
        events.append(
            {
                "event": "moca_startup_stabilized",
                "maximum_startup_drift_rad": maximum_startup_drift,
                "home_error_rad": startup_home_error,
                "realign": startup_realign,
            }
        )

        left_deadline = time.monotonic() + 30.0
        while time.monotonic() < left_deadline:
            os.kill(left_hold_pid, 0)
            status = json.loads(args.left_arm_hold_status.read_text())
            if status.get("violation"):
                raise RuntimeError(
                    "left-arm Home hold violated during alignment"
                )
            if status.get("arrived"):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("left arm did not reach Home before put-demo")
        if not robot.relaxed_ik_probe_ready():
            raise RuntimeError(
                "isolated right-arm RelaxedIK is unavailable before put-demo"
            )

        for phase_index, object_name in enumerate(object_order, start=1):
            targets = build_put_demo_targets(
                object_centers[object_name],
                topdown_quaternion,
            )
            api.command_gripper(OPEN_GRIPPER_COMMAND, duration_s=3.0)
            grasp_result = execute_beizi_grasp_entry(
                api,
                targets,
                topdown_quaternion,
            )
            close_result = close_gripper_until_contact(
                api,
                minimum_command=0.0,
                open_position=OPEN_GRIPPER_COMMAND,
            )
            grasp_ok, grasp_details = api.verify_grasp(save_debug=True)
            if not grasp_ok:
                raise RuntimeError(
                    f"{object_name} grasp verification failed: "
                    f"{grasp_details}"
                )
            hold_command = float(close_result["hold_command"])
            measured_grasp_tcp, measured_quaternion = (
                robot.get_local_planning_fk_pose()
            )
            actual_lift_tcp = measured_grasp_tcp + np.array(
                [0.0, 0.0, LIFT_DISTANCE_M]
            )
            lift_result = api.goto_pose(
                actual_lift_tcp,
                topdown_quaternion,
                gripper=hold_command,
                max_cartesian_step_m=0.02,
                max_orientation_step_rad=0.25,
                check_cartesian_ik_error=False,
                position_arrival_tolerance_m=0.025,
            )
            measured_lift_tcp, _ = robot.get_local_planning_fk_pose()
            actual_release_tcp = measured_lift_tcp + np.array(
                [FORWARD_RELEASE_DISTANCE_M, 0.0, 0.0]
            )
            transfer_result = api.goto_pose(
                actual_release_tcp,
                topdown_quaternion,
                gripper=hold_command,
                max_cartesian_step_m=0.02,
                max_orientation_step_rad=0.25,
                check_cartesian_ik_error=False,
                position_arrival_tolerance_m=0.025,
            )
            open_feedback = api.command_gripper(
                OPEN_GRIPPER_COMMAND,
                duration_s=3.0,
            )
            time.sleep(0.6)
            retract_tcp, retract_quaternion = (
                robot.get_local_planning_fk_pose()
            )
            retract_tcp[2] += 0.05
            retract_result = api.goto_pose(
                retract_tcp,
                retract_quaternion,
                gripper=OPEN_GRIPPER_COMMAND,
                max_cartesian_step_m=0.02,
                max_orientation_step_rad=0.25,
                check_cartesian_ik_error=False,
                require_pose_arrival=False,
            )
            home_result = api.move_joints_adaptive(
                planned_home_joints,
                waypoint_step_rad=0.10,
                context=f"put_demo_phase_{phase_index}_return_home",
            )
            events.append(
                {
                    "event": "put_object",
                    "phase_index": phase_index,
                    "object": object_name,
                    "source_center_base_m": (
                        targets["object_center"].tolist()
                    ),
                    "grasp_point_base_m": (
                        targets["grasp_point"].tolist()
                    ),
                    "grasp_tcp_base_m": targets["grasp_tcp"].tolist(),
                    "pregrasp_tcp_base_m": (
                        targets["pregrasp_tcp"].tolist()
                    ),
                    "planned_lift_tcp_base_m": (
                        targets["lift_tcp"].tolist()
                    ),
                    "planned_release_tcp_base_m": (
                        targets["release_tcp"].tolist()
                    ),
                    "actual_release_tcp_base_m": (
                        actual_release_tcp.tolist()
                    ),
                    "box_reference_center_base_m": (
                        targets["box_center"].tolist()
                    ),
                    "release_height_checked": False,
                    "grasp_quaternion_xyzw": (
                        topdown_quaternion.tolist()
                    ),
                    "grasp_entry_policy": (
                        "beizi_direct_pregrasp_then_vertical_descent"
                    ),
                    "grasp": grasp_result,
                    "close": close_result,
                    "grasp_verification": grasp_details,
                    "lift": lift_result,
                    "forward_transfer": transfer_result,
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
            "schema": "capx.r1lite.put_demo.execution_audit.v1",
            "generated_at_unix": time.time(),
            "object_order": list(object_order),
            "passed": passed,
            "error": error,
            "motion_fault": motion_fault,
            "torso_violation": torso_violation.is_set(),
            "left_hold_violation": left_hold_violation.is_set(),
            "events": events,
        }
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(
                audit,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
