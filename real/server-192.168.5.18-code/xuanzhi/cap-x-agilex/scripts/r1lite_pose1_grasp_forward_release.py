#!/usr/bin/env python3
"""Move to saved pose 1, grasp, translate +X, release, retract, and home."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.bimanual_kinematics import (  # noqa: E402
    R1LiteCombinedCollisionKinematics,
)
from capx.integrations.r1lite.bimanual_relaxed_ik import (  # noqa: E402
    IsolatedBimanualRelaxedIK,
)
from capx.integrations.r1lite.bimanual_runtime import (  # noqa: E402
    BimanualRuntime,
    BimanualRuntimeConfig,
    FeedbackStaleError,
    JointArrivalError,
)
from capx.integrations.r1lite.kinematics import (  # noqa: E402
    R1LiteCandidateKinematics,
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-1", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--forward-m", type=float, default=0.10)
    parser.add_argument("--lift-before-forward-m", type=float, default=0.0)
    parser.add_argument("--permissive-forward", action="store_true")
    parser.add_argument(
        "--no-pre-release-collision-scans", action="store_true"
    )
    parser.add_argument("--no-home-collision-scan", action="store_true")
    args = parser.parse_args()
    pose_1 = load(args.pose_1)
    plan = load(args.plan)
    bundle = load(args.bundle)
    torso = np.asarray(bundle["initial_state"]["torso_joints_rad"], dtype=float)
    saved = {
        side: np.asarray(
            pose_1["arms"][side]["joint_positions_rad"], dtype=float
        )
        for side in ("left", "right")
    }
    home = {
        side: np.asarray(
            bundle["initial_state"][f"{side}_joints_rad"], dtype=float
        )
        for side in ("left", "right")
    }
    audit: dict[str, Any] = {
        "schema": "capx.r1lite.pose1_grasp_forward_release.v1",
        "passed": False,
        "forward_m": float(args.forward_m),
        "robot_motion_commands_sent": False,
        "events": [],
    }
    controller = BimanualRuntime(
        config=BimanualRuntimeConfig(
            arrival_timeout_s=5.0,
            gripper_timeout_s=15.0,
            gripper_tolerance=8.0,
        )
    )
    try:
        controller.start()
        controller.wait_until_ready(timeout_s=10.0, require_tool_pose=True)
        collision = R1LiteCombinedCollisionKinematics(
            table_z_m=float(plan["safety"]["table_z_m"]),
            tcp_to_fingertip_m=float(plan["safety"]["tcp_to_fingertip_m"]),
        )
        solver = IsolatedBimanualRelaxedIK(controller, torso)
        local_fk = {
            side: R1LiteCandidateKinematics(side=side)
            for side in ("left", "right")
        }

        def poses() -> dict[str, Any]:
            deadline = time.monotonic() + 2.0
            while True:
                try:
                    return controller.get_tool_pose_pair()
                except FeedbackStaleError:
                    if time.monotonic() >= deadline:
                        raise
                    controller.refresh_feedback(0.02)

        def direct_joint(
            stage_id: str,
            target: dict[str, np.ndarray],
            *,
            fingers_open: bool,
            tolerance: float,
        ) -> None:
            current = controller.get_feedback_pair()
            delta = max(
                float(
                    np.max(
                        np.abs(
                            target[side]
                            - np.asarray(current[side], dtype=float)
                        )
                    )
                )
                for side in ("left", "right")
            )
            if delta > 3.0:
                raise RuntimeError(f"{stage_id} exceeds 3rad gate")
            if (
                args.no_pre_release_collision_scans
                and stage_id == "current_to_pose_1"
            ) or (
                args.no_home_collision_scan
                and stage_id == "retract_to_home"
            ):
                reports = []
            else:
                reports = collision.validate_interpolated_segment(
                    np.asarray(current["left"]),
                    np.asarray(current["right"]),
                    target["left"],
                    target["right"],
                    torso,
                    samples=50,
                    fingers_open=fingers_open,
                )
            arrival_exception = None
            try:
                controller.execute_joint_motion(
                    target["left"], target["right"], progress_steps=1
                )
            except JointArrivalError as exc:
                arrival_exception = repr(exc)
            audit["robot_motion_commands_sent"] = True
            arrived = controller.get_feedback_pair()
            errors = {
                side: float(
                    np.max(
                        np.abs(
                            np.asarray(arrived[side], dtype=float)
                            - target[side]
                        )
                    )
                )
                for side in ("left", "right")
            }
            audit["events"].append(
                {
                    "id": stage_id,
                    "collision_samples": len(reports),
                    "arrival_errors_rad": errors,
                    "arrival_exception": arrival_exception,
                }
            )
            if any(error > tolerance for error in errors.values()):
                raise RuntimeError(f"{stage_id} arrival failed: {errors}")

        def cartesian_pair(
            stage_id: str,
            target: dict[str, np.ndarray],
            quaternion: dict[str, np.ndarray],
            *,
            fingers_open: bool,
            apply_collision_gate: bool = True,
        ) -> None:
            expected_physical_relative = target["right"] - target["left"]
            for iteration in range(1, 21):
                current = controller.get_feedback_pair()
                current_local = {
                    side: local_fk[side]
                    .fk(np.asarray(current[side], dtype=float), torso)
                    .position_base_m
                    for side in ("left", "right")
                }
                expected_local_relative = (
                    current_local["right"] - current_local["left"]
                )
                candidates = {
                    side: solver.solve_candidates(
                        side=side,
                        target_position_base_m=target[side],
                        target_quaternion_xyzw=quaternion[side],
                        reference_joints_rad=current[side],
                        probes=5,
                    )
                    for side in ("left", "right")
                }
                valid = []
                for left_value, right_value in product(
                    candidates["left"], candidates["right"]
                ):
                    pair = {
                        "left": np.asarray(left_value, dtype=float),
                        "right": np.asarray(right_value, dtype=float),
                    }
                    if any(
                        np.max(
                            np.abs(
                                pair[side]
                                - np.asarray(current[side], dtype=float)
                            )
                        )
                        > 1.0
                        for side in ("left", "right")
                    ):
                        continue
                    if apply_collision_gate:
                        try:
                            reports = collision.validate_interpolated_segment(
                                np.asarray(current["left"]),
                                np.asarray(current["right"]),
                                pair["left"],
                                pair["right"],
                                torso,
                                samples=50,
                                fingers_open=fingers_open,
                            )
                        except ValueError:
                            continue
                    else:
                        reports = []
                    fk_position = {
                        side: local_fk[side]
                        .fk(pair[side], torso)
                        .position_base_m
                        for side in ("left", "right")
                    }
                    if (
                        np.linalg.norm(
                            (fk_position["right"] - fk_position["left"])
                            - expected_local_relative
                        )
                        > 0.005
                    ):
                        continue
                    errors = {
                        side: float(
                            np.linalg.norm(fk_position[side] - target[side])
                        )
                        for side in ("left", "right")
                    }
                    valid.append(
                        (
                            max(errors.values()),
                            sum(errors.values()),
                            pair,
                            len(reports),
                        )
                    )
                if not valid:
                    raise RuntimeError(
                        f"{stage_id} has no paired candidate passing gates"
                    )
                valid.sort(key=lambda item: item[:2])
                _, _, selected, samples = valid[0]
                controller.execute_joint_motion(
                    selected["left"],
                    selected["right"],
                    progress_steps=1,
                )
                audit["robot_motion_commands_sent"] = True
                physical = poses()
                actual = {
                    side: np.asarray(physical[side][0], dtype=float)
                    for side in ("left", "right")
                }
                errors = {
                    side: float(np.linalg.norm(actual[side] - target[side]))
                    for side in ("left", "right")
                }
                relative_error = float(
                    np.linalg.norm(
                        (actual["right"] - actual["left"])
                        - expected_physical_relative
                    )
                )
                audit["events"].append(
                    {
                        "id": stage_id,
                        "iteration": iteration,
                        "collision_samples": samples,
                        "target_errors_m": errors,
                        "relative_error_m": relative_error,
                    }
                )
                if (
                    all(error <= 0.015 for error in errors.values())
                    and relative_error <= 0.005
                ):
                    return
            raise RuntimeError(f"{stage_id} did not converge")

        controller.open_both_grippers()
        direct_joint(
            "current_to_pose_1",
            saved,
            fingers_open=True,
            tolerance=0.01,
        )
        controller.close_both_grippers()
        lift_start = poses()
        lift_step_count = max(
            0,
            int(
                np.ceil(
                    max(0.0, float(args.lift_before_forward_m)) / 0.01
                )
            ),
        )
        lift_stop_reason = "requested_height_reached"
        for lift_index in range(1, lift_step_count + 1):
            current_pose = poses()
            current_position = {
                side: np.asarray(current_pose[side][0], dtype=float)
                for side in ("left", "right")
            }
            current_quaternion = {
                side: np.asarray(current_pose[side][1], dtype=float)
                for side in ("left", "right")
            }
            achieved_lift = float(
                (
                    current_position["left"][2]
                    + current_position["right"][2]
                    - lift_start["left"][0][2]
                    - lift_start["right"][0][2]
                )
                / 2.0
            )
            remaining_lift = (
                float(args.lift_before_forward_m) - achieved_lift
            )
            step_lift = remaining_lift / float(
                lift_step_count - lift_index + 1
            )
            try:
                cartesian_pair(
                    f"lift_{lift_index}_of_{lift_step_count}",
                    {
                        side: current_position[side]
                        + np.array([0.0, 0.0, step_lift])
                        for side in ("left", "right")
                    },
                    current_quaternion,
                    fingers_open=False,
                    apply_collision_gate=(
                        not args.no_pre_release_collision_scans
                    ),
                )
            except RuntimeError as exc:
                lift_stop_reason = str(exc)
                break
        after_lift = poses()
        audit["lift_result"] = {
            "requested_lift_m": float(args.lift_before_forward_m),
            "achieved_lift_m": float(
                (
                    after_lift["left"][0][2]
                    + after_lift["right"][0][2]
                    - lift_start["left"][0][2]
                    - lift_start["right"][0][2]
                )
                / 2.0
            ),
            "stop_reason": lift_stop_reason,
        }
        initial_pose = poses()
        if args.permissive_forward:
            step_count = max(
                1, int(np.ceil(float(args.forward_m) / 0.01))
            )
        else:
            step_count = max(
                1, int(np.ceil(float(args.forward_m) / 0.02))
            )
        forward_stop_reason = "maximum_distance_reached"
        for index in range(1, step_count + 1):
            current_pose = poses()
            current_position = {
                side: np.asarray(current_pose[side][0], dtype=float)
                for side in ("left", "right")
            }
            current_quaternion = {
                side: np.asarray(current_pose[side][1], dtype=float)
                for side in ("left", "right")
            }
            achieved = float(
                (
                    current_position["left"][0]
                    + current_position["right"][0]
                    - initial_pose["left"][0][0]
                    - initial_pose["right"][0][0]
                )
                / 2.0
            )
            remaining = float(args.forward_m) - achieved
            step = remaining / float(step_count - index + 1)
            target = {
                side: current_position[side]
                + np.array([step, 0.0, 0.0])
                for side in ("left", "right")
            }
            if not args.permissive_forward:
                cartesian_pair(
                    f"forward_{index}_of_{step_count}",
                    target,
                    current_quaternion,
                    fingers_open=False,
                )
                continue
            current_joints = controller.get_feedback_pair()
            selected = {}
            for side in ("left", "right"):
                candidates = solver.solve_candidates(
                    side=side,
                    target_position_base_m=target[side],
                    target_quaternion_xyzw=current_quaternion[side],
                    reference_joints_rad=current_joints[side],
                    probes=5,
                )
                valid = []
                for value in candidates:
                    candidate = np.asarray(value, dtype=float)
                    joint_delta = float(
                        np.max(
                            np.abs(
                                candidate
                                - np.asarray(
                                    current_joints[side], dtype=float
                                )
                            )
                        )
                    )
                    if joint_delta > 1.0:
                        continue
                    fk_position = local_fk[side].fk(
                        candidate, torso
                    ).position_base_m
                    valid.append(
                        (
                            float(
                                np.linalg.norm(
                                    fk_position - target[side]
                                )
                            ),
                            joint_delta,
                            candidate,
                        )
                    )
                if not valid:
                    forward_stop_reason = (
                        f"{side}_has_no_joint_bounded_candidate"
                    )
                    selected = {}
                    break
                valid.sort(key=lambda item: item[:2])
                selected[side] = valid[0][2]
            if len(selected) != 2:
                break
            arrival_exception = None
            try:
                controller.execute_joint_motion(
                    selected["left"],
                    selected["right"],
                    progress_steps=1,
                )
            except JointArrivalError as exc:
                arrival_exception = repr(exc)
                forward_stop_reason = "joint_arrival_error"
            audit["robot_motion_commands_sent"] = True
            after_pose = poses()
            after_midpoint_x = float(
                (
                    after_pose["left"][0][0]
                    + after_pose["right"][0][0]
                )
                / 2.0
            )
            initial_midpoint_x = float(
                (
                    initial_pose["left"][0][0]
                    + initial_pose["right"][0][0]
                )
                / 2.0
            )
            achieved_after = after_midpoint_x - initial_midpoint_x
            audit["events"].append(
                {
                    "id": f"permissive_forward_{index}_of_{step_count}",
                    "collision_gate_applied": False,
                    "relative_vector_gate_applied": False,
                    "arrival_exception": arrival_exception,
                    "achieved_forward_m": achieved_after,
                }
            )
            if arrival_exception is not None:
                break
            if achieved_after - achieved < 0.001:
                forward_stop_reason = "forward_progress_below_1mm"
                break
        controller.open_both_grippers()
        final_pose = poses()
        audit["forward_result"] = {
            "stop_reason": forward_stop_reason,
            "achieved_forward_m": float(
                (
                    final_pose["left"][0][0]
                    + final_pose["right"][0][0]
                    - initial_pose["left"][0][0]
                    - initial_pose["right"][0][0]
                )
                / 2.0
            ),
            "grippers_opened": True,
        }
        direct_joint(
            "retract_to_home",
            home,
            fingers_open=True,
            tolerance=0.02,
        )
        audit["passed"] = True
    except Exception as exc:
        audit["error"] = repr(exc)
        try:
            controller.hold()
            audit["common_hold_attempted"] = True
        except Exception as hold_exc:
            audit["common_hold_error"] = repr(hold_exc)
        raise
    finally:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        controller.close()


if __name__ == "__main__":
    main()
