#!/usr/bin/env python3
"""Scan nearby high-transfer XY targets without commanding robot motion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.r1lite_execute_llm_cup_stack import load_robot  # noqa: E402
from capx.integrations.r1lite.control import R1LiteControlApi  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-correction-m",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    phase = plan["phases"][0]
    target = np.asarray(
        phase.get(
            "loaded_transfer_tcp_position_base_m",
            next(
                item["tcp_position_base_m"]
                for item in phase["waypoints"]
                if item["id"] == "above_support"
            ),
        ),
        dtype=float,
    ) + np.asarray(args.target_correction_m, dtype=float)
    quaternion = np.asarray(phase["orientation_quaternion_xyzw"], dtype=float)
    offsets = [
        (0.0, 0.0),
        (-0.03, 0.0),
        (-0.06, 0.0),
        (-0.09, 0.0),
        (-0.12, 0.0),
        (0.03, 0.0),
        (0.06, 0.0),
        (0.0, -0.03),
        (0.0, -0.06),
        (0.0, 0.03),
        (0.0, 0.06),
        (-0.03, -0.03),
        (-0.03, 0.03),
        (-0.06, -0.03),
        (-0.06, 0.03),
        (-0.09, -0.03),
        (-0.09, 0.03),
    ]
    robot = load_robot(args.config)
    api = R1LiteControlApi(robot)
    results = []
    try:
        for dx, dy in offsets:
            candidate = target + np.array([dx, dy, 0.0])
            try:
                candidate_plan = api.plan_pose_waypoints(
                    candidate,
                    quaternion,
                    max_cartesian_step_m=float(
                        plan["ik_path_limits"]["max_cartesian_step_m"]
                    ),
                    max_orientation_step_rad=float(
                        plan["ik_path_limits"]["max_orientation_step_rad"]
                    ),
                    check_cartesian_ik_error=True,
                )
                endpoint = candidate_plan["waypoints"][-1]
                results.append(
                    {
                        "target_tcp_base_m": candidate.tolist(),
                        "offset_xy_m": [dx, dy],
                        "path_valid": True,
                        "fk_position_error_m": endpoint[
                            "fk_position_error_m"
                        ],
                        "fk_orientation_error_rad": endpoint[
                            "fk_orientation_error_rad"
                        ],
                        "joint_feedback": endpoint["joint_feedback"],
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "target_tcp_base_m": candidate.tolist(),
                        "offset_xy_m": [dx, dy],
                        "path_valid": False,
                        "error": repr(exc),
                    }
                )
    finally:
        robot.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    valid = [
        result
        for result in results
        if result["path_valid"]
        and result["fk_position_error_m"] <= 0.025
    ]
    print(json.dumps(valid, indent=2))


if __name__ == "__main__":
    main()
