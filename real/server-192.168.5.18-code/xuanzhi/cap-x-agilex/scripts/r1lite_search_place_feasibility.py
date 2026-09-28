#!/usr/bin/env python3
"""Solver-only staged feasibility search for the box placement target."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import yaml


REPO = Path("/home/nvidia/xuanzhi/cap-x-agilex")
sys.path.insert(0, str(REPO))

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel
from scripts.r1lite_audit_tcp_frames import (
    FULL_URDF,
    RELAXED_URDF,
    UrdfChain,
    arm_joint_map,
)


CONFIG = REPO / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml"
SOURCE_PLAN = (
    REPO
    / "outputs/r1lite_tennis_ball_in_box/"
    "offline_ik_pick_place_joint_poses_3d.json"
)
MAX_POSITION_ERROR_M = 0.01
MAX_ORIENTATION_ERROR_RAD = 0.05
MAX_POSITION_STEP_M = 0.01
MAX_ORIENTATION_STEP_RAD = 0.05
MAX_JOINT_STEP_RAD = 0.2


def load_low_level() -> R1LiteRealLowLevel:
    config = yaml.safe_load(CONFIG.read_text())
    values = dict(config["env"]["cfg"]["low_level"])
    values.pop("_target_", None)
    signature = inspect.signature(R1LiteRealLowLevel.__init__)
    kwargs = {key: value for key, value in values.items() if key in signature.parameters}
    return R1LiteRealLowLevel(**kwargs)


def arm_limits() -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(FULL_URDF).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    lower = []
    upper = []
    for index in range(1, 7):
        limit = joints[f"right_arm_joint{index}"].find("limit")
        assert limit is not None
        lower.append(float(limit.attrib["lower"]))
        upper.append(float(limit.attrib["upper"]))
    return np.asarray(lower), np.asarray(upper)


class Solver:
    def __init__(self, torso: np.ndarray) -> None:
        self.chain = UrdfChain(RELAXED_URDF)
        full = UrdfChain(FULL_URDF)
        torso_map = {
            f"torso_joint{index + 1}": float(value)
            for index, value in enumerate(torso)
        }
        self.base_from_torso = full.fk(
            "base_link",
            "torso_link3",
            torso_map,
        )
        self.lower, self.upper = arm_limits()

    def fk(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.base_from_torso @ self.chain.fk(
            "torso_link3",
            "right_gripper_tool_z",
            arm_joint_map(joints),
        )
        return (
            matrix[:3, 3].copy(),
            Rotation.from_matrix(matrix[:3, :3]).as_quat(),
        )

    def solve(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        seeds: list[np.ndarray],
        *,
        previous: np.ndarray | None = None,
    ) -> dict[str, object]:
        lower = self.lower.copy()
        upper = self.upper.copy()
        if previous is not None:
            lower = np.maximum(lower, previous - (MAX_JOINT_STEP_RAD - 1e-6))
            upper = np.minimum(upper, previous + (MAX_JOINT_STEP_RAD - 1e-6))
        target_rotation = Rotation.from_quat(quaternion)
        attempts = []
        for weight in (0.4, 0.7, 1.0, 1.5):
            for seed in seeds:
                x0 = np.clip(seed, lower + 1e-8, upper - 1e-8)

                def objective(joints: np.ndarray) -> np.ndarray:
                    actual_position, actual_quaternion = self.fk(joints)
                    position_residual = (
                        actual_position - position
                    ) / MAX_POSITION_ERROR_M
                    orientation_residual = (
                        target_rotation.inv()
                        * Rotation.from_quat(actual_quaternion)
                    ).as_rotvec() / (MAX_ORIENTATION_ERROR_RAD * weight)
                    return np.concatenate(
                        [position_residual, orientation_residual]
                    )

                solution = least_squares(
                    objective,
                    x0,
                    bounds=(lower, upper),
                    max_nfev=500,
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12,
                ).x
                fk_position, fk_quaternion = self.fk(solution)
                position_error = float(np.linalg.norm(fk_position - position))
                orientation_error = float(
                    (
                        target_rotation.inv()
                        * Rotation.from_quat(fk_quaternion)
                    ).magnitude()
                )
                attempts.append(
                    {
                        "solution": solution,
                        "fk_position": fk_position,
                        "fk_quaternion": fk_quaternion,
                        "position_error_m": position_error,
                        "orientation_error_rad": orientation_error,
                        "score": (
                            position_error / MAX_POSITION_ERROR_M
                            + orientation_error / MAX_ORIENTATION_ERROR_RAD
                        ),
                    }
                )
        attempts.sort(key=lambda row: row["score"])
        best = attempts[0]
        return {
            **best,
            "passed": (
                best["position_error_m"] < MAX_POSITION_ERROR_M
                and best["orientation_error_rad"]
                < MAX_ORIENTATION_ERROR_RAD
            ),
        }


def evaluate_path(
    solver: Solver,
    start_position: np.ndarray,
    start_quaternion: np.ndarray,
    start_joints: np.ndarray,
    legs: list[tuple[np.ndarray, np.ndarray, str]],
) -> dict[str, object]:
    previous_position = start_position.copy()
    previous_quaternion = start_quaternion.copy()
    previous_joints = start_joints.copy()
    rows = []
    for target_position, target_quaternion, route in legs:
        if float(np.dot(previous_quaternion, target_quaternion)) < 0.0:
            target_quaternion = -target_quaternion
        position_distance = float(
            np.linalg.norm(target_position - previous_position)
        )
        orientation_distance = float(
            (
                Rotation.from_quat(previous_quaternion).inv()
                * Rotation.from_quat(target_quaternion)
            ).magnitude()
        )
        count = max(
            1,
            int(np.ceil(position_distance / MAX_POSITION_STEP_M)),
            int(np.ceil(orientation_distance / MAX_ORIENTATION_STEP_RAD)),
        )
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_quat([previous_quaternion, target_quaternion]),
        )
        leg_start_position = previous_position.copy()
        for index in range(1, count + 1):
            fraction = index / count
            position = (
                leg_start_position
                + (target_position - leg_start_position) * fraction
            )
            quaternion = slerp([fraction]).as_quat()[0]
            solved = solver.solve(
                position,
                quaternion,
                [previous_joints],
                previous=previous_joints,
            )
            rows.append(
                {
                    "route": route,
                    "step": index,
                    "step_count": count,
                    "position_error_m": solved["position_error_m"],
                    "orientation_error_rad": solved[
                        "orientation_error_rad"
                    ],
                    "passed": solved["passed"],
                }
            )
            if not solved["passed"]:
                return {
                    "passed": False,
                    "failed_at": rows[-1],
                    "steps": rows,
                }
            previous_joints = solved["solution"]
        previous_position = target_position.copy()
        previous_quaternion = target_quaternion.copy()
    return {
        "passed": True,
        "steps": rows,
        "final_solution_feedback_joints_rad": previous_joints.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Output exists; use a new run ID")
    source = json.loads(SOURCE_PLAN.read_text())
    place = next(
        row
        for row in source["waypoints"]
        if row["id"] == "place_02_above_box"
    )
    target_position = np.asarray(
        place["target_ee_position_base_m"],
        dtype=float,
    )
    base_quaternion = np.asarray(
        place["target_quaternion_xyzw"],
        dtype=float,
    )
    base_quaternion /= np.linalg.norm(base_quaternion)

    r1 = load_low_level()
    try:
        start_position, start_quaternion = r1.get_physical_tool_z_pose()
        start_joints = r1.get_arm_joints()
        torso = r1._base_from_torso_link3()[1]
    finally:
        r1.close()
    solver = Solver(torso)
    midpoint = (solver.lower + solver.upper) / 2.0
    seed_set = [start_joints, midpoint, np.clip(np.zeros(6), solver.lower, solver.upper)]

    orientation_stages = []
    feasible = []
    for yaw in np.linspace(-np.pi, np.pi, 25):
        quaternion = (
            Rotation.from_quat(base_quaternion)
            * Rotation.from_euler("z", float(yaw))
        ).as_quat()
        solved = solver.solve(target_position, quaternion, seed_set)
        orientation_stages.append(
            {
                "stage": "yaw_only",
                "yaw_rad": float(yaw),
                "position_error_m": solved["position_error_m"],
                "orientation_error_rad": solved["orientation_error_rad"],
                "passed": solved["passed"],
            }
        )
        if solved["passed"]:
            feasible.append((solved["score"], quaternion, solved))

    feasible.sort(key=lambda row: row[0])
    best_yaw_quaternion = feasible[0][1] if feasible else base_quaternion
    for axis in ("x", "y"):
        stage_feasible = []
        for tilt in np.linspace(-0.35, 0.35, 15):
            quaternion = (
                Rotation.from_quat(best_yaw_quaternion)
                * Rotation.from_euler(axis, float(tilt))
            ).as_quat()
            solved = solver.solve(target_position, quaternion, seed_set)
            orientation_stages.append(
                {
                    "stage": f"tilt_{axis}_with_fixed_yaw",
                    "tilt_rad": float(tilt),
                    "position_error_m": solved["position_error_m"],
                    "orientation_error_rad": solved[
                        "orientation_error_rad"
                    ],
                    "passed": solved["passed"],
                }
            )
            if solved["passed"]:
                stage_feasible.append((solved["score"], quaternion, solved))
        if stage_feasible:
            stage_feasible.sort(key=lambda row: row[0])
            feasible.extend(stage_feasible)

    path_results = {}
    chosen = None
    if feasible:
        feasible.sort(key=lambda row: row[0])
        chosen = feasible[0]
        target_quaternion = chosen[1]
        clearance_z = max(start_position[2], target_position[2])
        routes = {
            "straight": [(target_position, target_quaternion, "straight")],
            "vertical_then_translate": [
                (
                    np.array(
                        [start_position[0], start_position[1], clearance_z]
                    ),
                    start_quaternion,
                    "vertical",
                ),
                (
                    np.array(
                        [target_position[0], target_position[1], clearance_z]
                    ),
                    start_quaternion,
                    "translate_xy",
                ),
                (target_position, target_quaternion, "orient_and_descend"),
            ],
            "split_axes": [
                (
                    np.array(
                        [target_position[0], start_position[1], clearance_z]
                    ),
                    start_quaternion,
                    "x",
                ),
                (
                    np.array(
                        [target_position[0], target_position[1], clearance_z]
                    ),
                    start_quaternion,
                    "y",
                ),
                (target_position, target_quaternion, "orient_and_z"),
            ],
        }
        for name, legs in routes.items():
            path_results[name] = evaluate_path(
                solver,
                start_position,
                start_quaternion,
                start_joints,
                legs,
            )

    strict_paths = [
        name for name, value in path_results.items() if value["passed"]
    ]
    report = {
        "status": (
            "strict_place_path_found"
            if strict_paths
            else "place_target_or_path_rejected"
        ),
        "run_id": args.run_id,
        "generated_at_unix": time.time(),
        "robot_motion_commands_sent": False,
        "target_position_base_m": target_position.tolist(),
        "base_target_tool_z_quaternion_xyzw": base_quaternion.tolist(),
        "initial_arm_feedback_rad": start_joints.tolist(),
        "initial_torso_feedback_rad": torso.tolist(),
        "variable_search_order": [
            "yaw_only",
            "tilt_x_with_fixed_yaw",
            "tilt_y_with_fixed_yaw",
            "path_order_after_fixed_feasible_pose",
        ],
        "orientation_search": orientation_stages,
        "point_pose_feasible_count": len(feasible),
        "selected_target_tool_z_quaternion_xyzw": (
            chosen[1].tolist() if chosen else None
        ),
        "path_results": path_results,
        "strict_paths": strict_paths,
        "conclusion": (
            "At least one strict solver-only placement path is feasible."
            if strict_paths
            else "No strict path was demonstrated; adjust task target or "
            "route constraints before execution."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(report["status"])
    return 0 if strict_paths else 2


if __name__ == "__main__":
    raise SystemExit(main())
