#!/usr/bin/env python3
"""Replay pose 1, grasp, move to pose 2, release, and home."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.bimanual_kinematics import (  # noqa: E402
    R1LiteCombinedCollisionKinematics,
)
from capx.integrations.r1lite.bimanual_runtime import (  # noqa: E402
    BimanualRuntime,
    BimanualRuntimeConfig,
    JointArrivalError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-1", type=Path, required=True)
    parser.add_argument("--pose-2", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def main() -> None:
    args = parse_args()
    pose_1 = load(args.pose_1)
    pose_2 = load(args.pose_2)
    plan = load(args.plan)
    bundle = load(args.bundle)
    torso = np.asarray(bundle["initial_state"]["torso_joints_rad"], dtype=float)
    targets = {
        "pose_1": {
            side: np.asarray(
                pose_1["arms"][side]["joint_positions_rad"], dtype=float
            )
            for side in ("left", "right")
        },
        "pose_2": {
            side: np.asarray(
                pose_2["arms"][side]["joint_positions_rad"], dtype=float
            )
            for side in ("left", "right")
        },
        "home": {
            side: np.asarray(
                bundle["initial_state"][f"{side}_joints_rad"], dtype=float
            )
            for side in ("left", "right")
        },
    }
    audit: dict[str, Any] = {
        "schema": "capx.r1lite.pose1_grasp_pose2_replay.v1",
        "passed": False,
        "robot_motion_commands_sent": False,
        "segments": [],
    }
    controller = BimanualRuntime(
        config=BimanualRuntimeConfig(arrival_timeout_s=5.0)
    )
    try:
        controller.start()
        controller.wait_until_ready(timeout_s=10.0, require_tool_pose=True)
        collision = R1LiteCombinedCollisionKinematics(
            table_z_m=float(plan["safety"]["table_z_m"]),
            tcp_to_fingertip_m=float(plan["safety"]["tcp_to_fingertip_m"]),
        )

        def move(
            segment_id: str,
            target: dict[str, np.ndarray],
            *,
            fingers_open: bool,
        ) -> None:
            current = controller.get_feedback_pair()
            maximum_delta = max(
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
            if maximum_delta > 3.0:
                raise RuntimeError(
                    f"{segment_id} exceeds 3.0rad direct-motion gate"
                )
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
                if segment_id != "pose_2_to_home":
                    raise
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
            audit["segments"].append(
                {
                    "id": segment_id,
                    "collision_samples": len(reports),
                    "maximum_joint_delta_rad": maximum_delta,
                    "arrival_errors_rad": errors,
                    "fingers_open": fingers_open,
                    "arrival_exception": arrival_exception,
                }
            )
            tolerance = 0.02 if segment_id == "pose_2_to_home" else 0.01
            if any(error > tolerance for error in errors.values()):
                raise RuntimeError(
                    f"{segment_id} joint arrival gate failed: {errors}"
                )

        controller.open_both_grippers()
        move("current_to_pose_1", targets["pose_1"], fingers_open=True)
        controller.close_both_grippers()
        move("pose_1_to_pose_2", targets["pose_2"], fingers_open=False)
        controller.open_both_grippers()
        move("pose_2_to_home", targets["home"], fingers_open=True)
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
