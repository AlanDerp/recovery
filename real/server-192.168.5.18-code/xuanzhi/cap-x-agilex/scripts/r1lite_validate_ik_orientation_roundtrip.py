#!/usr/bin/env python3
"""Audit current-pose orientation round trips through isolated RelaxedIK."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.kinematics import (  # noqa: E402
    R1LiteCandidateKinematics,
)
from scripts.r1lite_validate_towel_swept_collision import (  # noqa: E402
    make_robot,
    orientation_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def variants(side: str, tool_quaternion: np.ndarray) -> dict[str, np.ndarray]:
    tool = Rotation.from_quat(tool_quaternion)
    rotations = {
        "identity_tool_quaternion": tool,
        "post_y_minus_90": tool * Rotation.from_euler("y", -np.pi / 2.0),
        "post_y_plus_90": tool * Rotation.from_euler("y", np.pi / 2.0),
        "pre_y_minus_90": Rotation.from_euler("y", -np.pi / 2.0) * tool,
        "pre_y_plus_90": Rotation.from_euler("y", np.pi / 2.0) * tool,
    }
    if side == "right":
        # Keep all variants for diagnosis, even though identity is the declared
        # right_gripper_tool_z contract.
        pass
    return {name: rotation.as_quat() for name, rotation in rotations.items()}


def main() -> None:
    args = parse_args()
    robots = {
        side: make_robot(args.config, side) for side in ("left", "right")
    }
    audit = {
        "experiment": "isolated_relaxedik_current_pose_orientation_roundtrip",
        "robot_motion_commands_sent": False,
        "sides": {},
    }
    try:
        torso = robots["right"].get_torso_joints()
        for side in ("left", "right"):
            robot = robots[side]
            joints = robot.get_arm_joints()
            kinematics = R1LiteCandidateKinematics(side=side)
            target_fk = kinematics.fk(joints, torso)
            results = []
            for name, solver_quaternion in variants(
                side,
                target_fk.quaternion_xyzw,
            ).items():
                try:
                    robot.seed_relaxed_ik_probe(joints, duration_s=0.15)
                    candidate = robot.get_relaxed_ik_candidate(
                        target_fk.position_base_m,
                        solver_quaternion,
                        timeout_s=1.2,
                        reference_joints=joints,
                        republish_interval_s=None,
                    )
                    candidate_fk = kinematics.fk(candidate, torso)
                    results.append(
                        {
                            "variant": name,
                            "candidate_joints_rad": candidate.tolist(),
                            "max_joint_roundtrip_error_rad": float(
                                np.max(np.abs(candidate - joints))
                            ),
                            "position_roundtrip_error_m": float(
                                np.linalg.norm(
                                    candidate_fk.position_base_m
                                    - target_fk.position_base_m
                                )
                            ),
                            "tool_orientation_roundtrip_error_rad": (
                                orientation_error(
                                    candidate_fk.quaternion_xyzw,
                                    target_fk.quaternion_xyzw,
                                )
                            ),
                        }
                    )
                except Exception as exc:
                    results.append({"variant": name, "error": repr(exc)})
            successful = [item for item in results if "error" not in item]
            best = min(
                successful,
                key=lambda item: (
                    item["tool_orientation_roundtrip_error_rad"],
                    item["position_roundtrip_error_m"],
                    item["max_joint_roundtrip_error_rad"],
                ),
            )
            audit["sides"][side] = {
                "current_joints_rad": joints.tolist(),
                "current_tool_position_base_m": (
                    target_fk.position_base_m.tolist()
                ),
                "current_tool_quaternion_xyzw": (
                    target_fk.quaternion_xyzw.tolist()
                ),
                "variants": results,
                "best_variant": best["variant"],
            }
        audit["passed"] = True
    except Exception as exc:
        audit["passed"] = False
        audit["error"] = repr(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
        for robot in robots.values():
            robot.close()
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
