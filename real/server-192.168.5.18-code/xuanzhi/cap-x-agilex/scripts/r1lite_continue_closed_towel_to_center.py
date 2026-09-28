#!/usr/bin/env python3
"""Continue a closed dual-arm towel grasp to center with online certified pairs."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

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
)
from capx.integrations.r1lite.kinematics import (  # noqa: E402
    R1LiteCandidateKinematics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def main() -> None:
    args = parse_args()
    plan = load(args.plan)
    bundle = load(args.bundle)
    snapshot = load(args.snapshot)
    torso = np.asarray(bundle["initial_state"]["torso_joints_rad"], dtype=float)
    controller = BimanualRuntime(
        config=BimanualRuntimeConfig(arrival_timeout_s=5.0)
    )
    audit: dict[str, Any] = {
        "schema": "capx.r1lite.closed_grasp_center_continuation.v1",
        "passed": False,
        "robot_motion_commands_sent": False,
        "stages": [],
    }
    try:
        controller.start()
        controller.wait_until_ready(timeout_s=10.0, require_tool_pose=True)
        collision = R1LiteCombinedCollisionKinematics(
            table_z_m=float(plan["safety"]["table_z_m"]),
            tcp_to_fingertip_m=float(plan["safety"]["tcp_to_fingertip_m"]),
        )
        solver = IsolatedBimanualRelaxedIK(controller, torso)
        fk = {
            side: R1LiteCandidateKinematics(side=side)
            for side in ("left", "right")
        }

        def fresh_poses() -> dict[str, Any]:
            deadline = time.monotonic() + 2.0
            while True:
                try:
                    return controller.get_tool_pose_pair()
                except FeedbackStaleError:
                    if time.monotonic() >= deadline:
                        raise
                    controller.refresh_feedback(0.02)

        def move_cartesian_pair(
            stage_id: str,
            target: dict[str, np.ndarray],
            quaternion: dict[str, np.ndarray],
            *,
            fingers_open: bool,
        ) -> None:
            expected_physical_relative = target["right"] - target["left"]
            for iteration in range(1, 21):
                current = controller.get_feedback_pair()
                current_local_positions = {
                    side: fk[side]
                    .fk(np.asarray(current[side], dtype=float), torso)
                    .position_base_m
                    for side in ("left", "right")
                }
                expected_local_relative = (
                    current_local_positions["right"]
                    - current_local_positions["left"]
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
                    positions = {
                        side: fk[side].fk(pair[side], torso).position_base_m
                        for side in ("left", "right")
                    }
                    relative_error = float(
                        np.linalg.norm(
                            (positions["right"] - positions["left"])
                            - expected_local_relative
                        )
                    )
                    if relative_error > 0.005:
                        continue
                    errors = {
                        side: float(
                            np.linalg.norm(positions[side] - target[side])
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
                _, _, selected, sample_count = valid[0]
                controller.execute_joint_motion(
                    selected["left"],
                    selected["right"],
                    progress_steps=1,
                )
                poses = fresh_poses()
                actual = {
                    side: np.asarray(poses[side][0], dtype=float)
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
                audit["robot_motion_commands_sent"] = True
                audit["stages"].append(
                    {
                        "id": stage_id,
                        "iteration": iteration,
                        "target_errors_m": errors,
                        "relative_error_m": relative_error,
                        "collision_samples": sample_count,
                    }
                )
                if (
                    all(error <= 0.015 for error in errors.values())
                    and relative_error <= 0.005
                ):
                    return
            raise RuntimeError(f"{stage_id} did not converge in 20 iterations")

        poses = fresh_poses()
        current_position = {
            side: np.asarray(poses[side][0], dtype=float)
            for side in ("left", "right")
        }
        current_quaternion = {
            side: np.asarray(poses[side][1], dtype=float)
            for side in ("left", "right")
        }
        saved_mid_z = float(
            np.mean(
                [
                    snapshot["arms"][side]["tcp_position_base_m"][2]
                    for side in ("left", "right")
                ]
            )
        )
        current_mid_z = float(
            (current_position["left"][2] + current_position["right"][2]) / 2.0
        )
        remaining_lift = saved_mid_z + 0.08 - current_mid_z
        achieved_lift_at_start = current_mid_z - saved_mid_z
        if achieved_lift_at_start >= 0.04:
            lift_steps = 0
            audit["lift_terminated_at_local_reach_limit"] = {
                "completed_substeps": 0,
                "requested_substeps": 0,
                "achieved_lift_m": achieved_lift_at_start,
                "reason": "continuing from previously verified partial lift",
            }
        else:
            lift_steps = max(
                1, int(np.ceil(abs(remaining_lift) / 0.02))
            )
        for lift_index in range(1, lift_steps + 1):
            poses = fresh_poses()
            step_position = {
                side: np.asarray(poses[side][0], dtype=float)
                for side in ("left", "right")
            }
            step_quaternion = {
                side: np.asarray(poses[side][1], dtype=float)
                for side in ("left", "right")
            }
            step_mid_z = float(
                (step_position["left"][2] + step_position["right"][2]) / 2.0
            )
            step_dz = (
                saved_mid_z + 0.08 - step_mid_z
            ) / float(lift_steps - lift_index + 1)
            try:
                move_cartesian_pair(
                    f"lift_to_8cm_{lift_index}_of_{lift_steps}",
                    {
                        side: step_position[side]
                        + np.array([0.0, 0.0, step_dz])
                        for side in ("left", "right")
                    },
                    step_quaternion,
                    fingers_open=False,
                )
            except RuntimeError as exc:
                achieved_lift = step_mid_z - saved_mid_z
                if lift_index == 1 and achieved_lift < 0.04:
                    raise
                audit["lift_terminated_at_local_reach_limit"] = {
                    "completed_substeps": lift_index - 1,
                    "requested_substeps": lift_steps,
                    "achieved_lift_m": achieved_lift,
                    "reason": str(exc),
                }
                break

        poses = fresh_poses()
        lifted = {
            side: np.asarray(poses[side][0], dtype=float)
            for side in ("left", "right")
        }
        lifted_quaternion = {
            side: np.asarray(poses[side][1], dtype=float)
            for side in ("left", "right")
        }
        for tilt_index in range(1, 3):
            poses = fresh_poses()
            tilt_position = {
                side: np.asarray(poses[side][0], dtype=float)
                for side in ("left", "right")
            }
            tilt_quaternion = {
                side: (
                    Rotation.from_rotvec(
                        np.radians(-15.0) * np.array([0.0, 1.0, 0.0])
                    )
                    * Rotation.from_quat(
                        np.asarray(poses[side][1], dtype=float)
                    )
                ).as_quat()
                for side in ("left", "right")
            }
            move_cartesian_pair(
                f"tilt_for_translation_{tilt_index}_of_2",
                tilt_position,
                tilt_quaternion,
                fingers_open=False,
            )
        poses = fresh_poses()
        lifted = {
            side: np.asarray(poses[side][0], dtype=float)
            for side in ("left", "right")
        }
        lifted_quaternion = {
            side: np.asarray(poses[side][1], dtype=float)
            for side in ("left", "right")
        }
        midpoint = (lifted["left"] + lifted["right"]) / 2.0
        center = np.asarray(plan["first_center_base_m"], dtype=float)
        translation = np.array(
            [center[0] - midpoint[0], center[1] - midpoint[1], 0.0]
        )
        translation_steps = max(
            1, int(np.ceil(np.linalg.norm(translation[:2]) / 0.02))
        )
        for translation_index in range(1, translation_steps + 1):
            poses = fresh_poses()
            step_position = {
                side: np.asarray(poses[side][0], dtype=float)
                for side in ("left", "right")
            }
            step_quaternion = {
                side: np.asarray(poses[side][1], dtype=float)
                for side in ("left", "right")
            }
            step_midpoint = (
                step_position["left"] + step_position["right"]
            ) / 2.0
            remaining = np.array(
                [
                    center[0] - step_midpoint[0],
                    center[1] - step_midpoint[1],
                    0.0,
                ]
            )
            step_translation = remaining / float(
                translation_steps - translation_index + 1
            )
            move_cartesian_pair(
                (
                    "translate_midpoint_to_center_"
                    f"{translation_index}_of_{translation_steps}"
                ),
                {
                    side: step_position[side] + step_translation
                    for side in ("left", "right")
                },
                step_quaternion,
                fingers_open=False,
            )

        controller.open_both_grippers()
        poses = fresh_poses()
        released = {
            side: np.asarray(poses[side][0], dtype=float)
            for side in ("left", "right")
        }
        released_quaternion = {
            side: np.asarray(poses[side][1], dtype=float)
            for side in ("left", "right")
        }
        retract_target = {
            side: released[side] + np.array([0.0, 0.0, 0.10])
            for side in ("left", "right")
        }
        move_cartesian_pair(
            "release_retract",
            retract_target,
            released_quaternion,
            fingers_open=True,
        )

        current = controller.get_feedback_pair()
        home = {
            side: np.asarray(
                bundle["initial_state"][f"{side}_joints_rad"], dtype=float
            )
            for side in ("left", "right")
        }
        collision.validate_interpolated_segment(
            np.asarray(current["left"]),
            np.asarray(current["right"]),
            home["left"],
            home["right"],
            torso,
            samples=50,
            fingers_open=True,
        )
        controller.execute_joint_motion(
            home["left"], home["right"], progress_steps=1
        )
        arrived = controller.get_feedback_pair()
        home_errors = {
            side: float(
                np.max(
                    np.abs(
                        np.asarray(arrived[side], dtype=float) - home[side]
                    )
                )
            )
            for side in ("left", "right")
        }
        audit["robot_motion_commands_sent"] = True
        audit["home_joint_errors_rad"] = home_errors
        if any(error > 0.01 for error in home_errors.values()):
            raise RuntimeError(f"home arrival failed: {home_errors}")
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
