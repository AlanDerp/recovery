#!/usr/bin/env python3
"""Regenerate every nominal waypoint with explicit tool-Z semantics.

The solver is advanced through small Cartesian/orientation steps so distant
anchor poses do not force an unobserved IK branch jump. This script only talks
to the isolated RelaxedIK solver and never publishes robot motion commands.
"""

from __future__ import annotations

import argparse
import hashlib
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
    UrdfChain,
    arm_joint_map,
    full_joint_map,
    pose_matrix,
)


CONFIG = REPO / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml"
SOURCE_PLAN = (
    REPO
    / "outputs/r1lite_tennis_ball_in_box/"
    "offline_ik_pick_place_joint_poses_3d.json"
)
SETTINGS = Path(
    "/home/nvidia/galaxea/install/mobiman/lib/mobiman/configs/"
    "R1_Lite/settings_right.yaml"
)
RELAXED_URDF = Path(
    "/home/nvidia/galaxea/install/mobiman/lib/mobiman/configs/urdfs/"
    "r1_lite_floating_right.urdf"
)
FULL_URDF = Path(
    "/home/nvidia/galaxea/install/r1lite_urdf/share/r1lite_urdf/urdf/"
    "r1lite_2026.urdf"
)
MAX_POSITION_STEP_M = 0.01
MAX_ORIENTATION_STEP_RAD = 0.05
MAX_INCREMENTAL_JOINT_DELTA_RAD = 0.20
MAX_FK_POSITION_ERROR_M = 0.01
MAX_FK_ORIENTATION_ERROR_RAD = 0.05
MAX_SOLVER_REFINEMENTS = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        (
            Rotation.from_quat(np.asarray(a, dtype=float)).inv()
            * Rotation.from_quat(np.asarray(b, dtype=float))
        ).magnitude()
    )


def nearest_tool_z_twist(
    target_quaternion: np.ndarray,
    reference_quaternion: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Choose the equivalent ball-grasp wrist twist nearest the current pose."""
    target = Rotation.from_quat(target_quaternion)
    candidates = np.linspace(-np.pi, np.pi, 361)
    best_quaternion = target_quaternion
    best_angle = 0.0
    best_distance = float("inf")
    for angle in candidates:
        candidate = (
            target * Rotation.from_euler("z", float(angle))
        ).as_quat()
        distance = rotation_error(reference_quaternion, candidate)
        if distance < best_distance:
            best_quaternion = candidate
            best_angle = float(angle)
            best_distance = distance
    return best_quaternion, best_angle


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


def refine_candidate(
    *,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    initial: np.ndarray,
    previous: np.ndarray,
    torso: np.ndarray,
    relaxed_chain: UrdfChain,
    full_chain: UrdfChain,
    lower: np.ndarray,
    upper: np.ndarray,
    max_delta_rad: float = MAX_INCREMENTAL_JOINT_DELTA_RAD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torso_map = {
        f"torso_joint{index + 1}": float(value)
        for index, value in enumerate(torso)
    }
    base_from_torso = full_chain.fk(
        "base_link",
        "torso_link3",
        torso_map,
    )
    local_lower = np.maximum(
        lower,
        previous - (max_delta_rad - 1e-6),
    )
    local_upper = np.minimum(
        upper,
        previous + (max_delta_rad - 1e-6),
    )
    x0 = np.clip(initial, local_lower + 1e-8, local_upper - 1e-8)
    target_rotation = Rotation.from_quat(target_quaternion)

    def fk(joints: np.ndarray) -> np.ndarray:
        return base_from_torso @ relaxed_chain.fk(
            "torso_link3",
            "right_gripper_tool_z",
            arm_joint_map(joints),
        )

    def objective(joints: np.ndarray) -> np.ndarray:
        matrix = fk(joints)
        position_residual = (
            matrix[:3, 3] - target_position
        ) / MAX_FK_POSITION_ERROR_M
        rotation_residual = (
            target_rotation.inv()
            * Rotation.from_matrix(matrix[:3, :3])
        ).as_rotvec() / (MAX_FK_ORIENTATION_ERROR_RAD * 0.8)
        return np.concatenate([position_residual, rotation_residual])

    optimized = least_squares(
        objective,
        x0,
        bounds=(local_lower, local_upper),
        max_nfev=400,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    ).x
    matrix = fk(optimized)
    return (
        optimized,
        matrix[:3, 3].copy(),
        Rotation.from_matrix(matrix[:3, :3]).as_quat(),
    )


def load_low_level() -> R1LiteRealLowLevel:
    config = yaml.safe_load(CONFIG.read_text())
    values = dict(config["env"]["cfg"]["low_level"])
    values.pop("_target_", None)
    values["r1lite_motion_validation_enabled"] = True
    signature = inspect.signature(R1LiteRealLowLevel.__init__)
    kwargs = {key: value for key, value in values.items() if key in signature.parameters}
    return R1LiteRealLowLevel(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, default=SOURCE_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--scope",
        choices=("current", "pick", "full"),
        default="pick",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Output exists; use a new run ID")
    original = json.loads(args.source_plan.read_text())
    r1 = load_low_level()
    try:
        start_position, start_quaternion = r1.get_physical_tool_z_pose()
        start_joints = r1.get_arm_joints()
        torso_joints = r1._base_from_torso_link3()[1]
        r1.reset_relaxed_ik_probe()
        relaxed_chain = UrdfChain(RELAXED_URDF)
        full_chain = UrdfChain(FULL_URDF)
        lower_limits, upper_limits = arm_limits()

        waypoints = []
        frame0 = dict(original["waypoints"][0])
        frame0.update(
            {
                "index": 0,
                "target_ee_position_base_m": start_position.tolist(),
                "target_quaternion_xyzw": start_quaternion.tolist(),
                "pose_semantic": "T_base_right_gripper_tool_z",
                "quaternion_order": "xyzw",
                "joint_feedback_solution_rad": start_joints.tolist(),
                "solver_seed_joints_rad": start_joints.tolist(),
                "fk_position_base_m": start_position.tolist(),
                "fk_quaternion_xyzw_base": start_quaternion.tolist(),
                "fk_position_error_m": 0.0,
                "fk_orientation_error_rad": 0.0,
                "joint_path_from_previous": [],
            }
        )
        waypoints.append(frame0)

        previous_position = start_position.copy()
        previous_quaternion = start_quaternion.copy()
        previous_joints = start_joints.copy()

        source_waypoints = original["waypoints"][1:]
        if args.scope == "current":
            source_waypoints = []
        elif args.scope == "pick":
            source_waypoints = source_waypoints[:5]
        for frame_index, source in enumerate(source_waypoints, start=1):
            if frame_index == 1:
                target_position = start_position + np.array([0.0, 0.0, 0.03])
                target_quaternion = start_quaternion.copy()
            else:
                target_position = np.asarray(
                    source["target_ee_position_base_m"],
                    dtype=float,
                )
                target_quaternion = np.asarray(
                    source["target_quaternion_xyzw"],
                    dtype=float,
                )
            target_quaternion /= np.linalg.norm(target_quaternion)
            tool_z_twist_rad = 0.0
            if source["id"].startswith("pick_0") and frame_index >= 2:
                target_quaternion, tool_z_twist_rad = nearest_tool_z_twist(
                    target_quaternion,
                    previous_quaternion,
                )
            if float(np.dot(previous_quaternion, target_quaternion)) < 0.0:
                target_quaternion = -target_quaternion

            path = []
            segment_start_position = previous_position.copy()
            segment_start_joints = previous_joints.copy()
            route = []
            if source["id"] == "pick_02_above_ball_high":
                route.append(
                    (
                        target_position.copy(),
                        previous_quaternion.copy(),
                        "translate_before_orientation",
                    )
                )
            if source["id"] == "place_02_above_box":
                route.append(
                    (
                        np.array(
                            [
                                previous_position[0],
                                previous_position[1],
                                target_position[2],
                            ]
                        ),
                        previous_quaternion.copy(),
                        "vertical_lift",
                    )
                )
                route.append(
                    (
                        np.array(
                            [
                                previous_position[0],
                                previous_position[1],
                                target_position[2],
                            ]
                        ),
                        previous_quaternion.copy(),
                        "hold_after_lift",
                    )
                )
                route.append(
                    (
                        np.array(
                            [
                                target_position[0],
                                previous_position[1],
                                target_position[2],
                            ]
                        ),
                        previous_quaternion.copy(),
                        "shift_x_at_clearance",
                    )
                )
            route.append(
                (target_position, target_quaternion, "anchor_target")
            )
            total_step_index = 0
            for route_index, (
                route_position,
                route_quaternion,
                route_name,
            ) in enumerate(route):
                leg_start_position = previous_position.copy()
                leg_start_quaternion = previous_quaternion.copy()
                if (
                    source["id"] == "pick_02_above_ball_high"
                    and route_name == "anchor_target"
                ):
                    feasible_twists = []
                    base_rotation = Rotation.from_quat(route_quaternion)
                    for twist in np.linspace(-np.pi, np.pi, 73):
                        trial_quaternion = (
                            base_rotation
                            * Rotation.from_euler("z", float(twist))
                        ).as_quat()
                        (
                            trial_joints,
                            trial_fk_position,
                            trial_fk_quaternion,
                        ) = refine_candidate(
                            target_position=route_position,
                            target_quaternion=trial_quaternion,
                            initial=previous_joints,
                            previous=previous_joints,
                            torso=torso_joints,
                            relaxed_chain=relaxed_chain,
                            full_chain=full_chain,
                            lower=lower_limits,
                            upper=upper_limits,
                            max_delta_rad=1.5,
                        )
                        trial_position_error = float(
                            np.linalg.norm(
                                trial_fk_position - route_position
                            )
                        )
                        trial_orientation_error = rotation_error(
                            trial_quaternion,
                            trial_fk_quaternion,
                        )
                        if (
                            trial_position_error
                            < MAX_FK_POSITION_ERROR_M
                            and trial_orientation_error
                            < MAX_FK_ORIENTATION_ERROR_RAD
                        ):
                            feasible_twists.append(
                                (
                                    rotation_error(
                                        leg_start_quaternion,
                                        trial_quaternion,
                                    ),
                                    trial_position_error,
                                    trial_orientation_error,
                                    trial_quaternion,
                                )
                            )
                    if not feasible_twists:
                        raise RuntimeError(
                            "pick_02 has no strict tool-Z-twist-equivalent "
                            "orientation at the requested position"
                        )
                    feasible_twists.sort(key=lambda row: row[:3])
                    route_quaternion = feasible_twists[0][3]
                    target_quaternion = route_quaternion.copy()
                if (
                    float(
                        np.dot(leg_start_quaternion, route_quaternion)
                    )
                    < 0.0
                ):
                    route_quaternion = -route_quaternion
                position_distance = float(
                    np.linalg.norm(
                        route_position - leg_start_position
                    )
                )
                orientation_distance = rotation_error(
                    leg_start_quaternion,
                    route_quaternion,
                )
                step_count = max(
                    1,
                    int(
                        np.ceil(
                            position_distance / MAX_POSITION_STEP_M
                        )
                    ),
                    int(
                        np.ceil(
                            orientation_distance
                            / MAX_ORIENTATION_STEP_RAD
                        )
                    ),
                )
                slerp = Slerp(
                    [0.0, 1.0],
                    Rotation.from_quat(
                        [leg_start_quaternion, route_quaternion]
                    ),
                )
                for step_index in range(1, step_count + 1):
                    total_step_index += 1
                    fraction = step_index / step_count
                    step_position = (
                        leg_start_position
                        + (route_position - leg_start_position)
                        * fraction
                    )
                    step_quaternion = slerp([fraction]).as_quat()[0]
                    refinement_reference = previous_joints.copy()
                    for refinement_index in range(
                        1,
                        MAX_SOLVER_REFINEMENTS + 1,
                    ):
                        candidate = r1.get_relaxed_ik_candidate(
                            step_position,
                            step_quaternion,
                            timeout_s=3.0,
                            reference_joints=refinement_reference,
                        )
                        fk_position, fk_quaternion = (
                            r1.get_relaxed_ik_fk_pose_base()
                        )
                        candidate, fk_position, fk_quaternion = (
                            refine_candidate(
                                target_position=step_position,
                                target_quaternion=step_quaternion,
                                initial=candidate,
                                previous=previous_joints,
                                torso=torso_joints,
                                relaxed_chain=relaxed_chain,
                                full_chain=full_chain,
                                lower=lower_limits,
                                upper=upper_limits,
                            )
                        )
                        position_error = float(
                            np.linalg.norm(
                                fk_position - step_position
                            )
                        )
                        orientation_error = rotation_error(
                            step_quaternion,
                            fk_quaternion,
                        )
                        if (
                            position_error
                            <= MAX_FK_POSITION_ERROR_M
                            and orientation_error
                            <= MAX_FK_ORIENTATION_ERROR_RAD
                        ):
                            break
                        refinement_reference = candidate
                    joint_delta = float(
                        np.max(
                            np.abs(candidate - previous_joints)
                        )
                    )
                    if (
                        position_error > MAX_FK_POSITION_ERROR_M
                        or orientation_error
                        > MAX_FK_ORIENTATION_ERROR_RAD
                        or joint_delta
                        >= MAX_INCREMENTAL_JOINT_DELTA_RAD
                    ):
                        raise RuntimeError(
                            f"{source['id']} route {route_name} "
                            f"step {step_index}/{step_count} "
                            "failed FK validation: "
                            f"position_error={position_error:.4f}m, "
                            f"orientation_error="
                            f"{orientation_error:.4f}rad, "
                            f"joint_delta={joint_delta:.4f}rad"
                        )
                    path.append(
                        {
                            "step_index": total_step_index,
                            "route_index": route_index,
                            "route_name": route_name,
                            "route_step_index": step_index,
                            "route_step_count": step_count,
                            "fraction": fraction,
                            "solver_refinement_count": (
                                refinement_index
                            ),
                            "target_position_base_m": (
                                step_position.tolist()
                            ),
                            "target_quaternion_tool_z_xyzw": (
                                step_quaternion.tolist()
                            ),
                            "solver_seed_joints_rad": (
                                previous_joints.tolist()
                            ),
                            "solution_joints_rad": candidate.tolist(),
                            "max_incremental_joint_delta_rad": (
                                joint_delta
                            ),
                            "fk_position_base_m": (
                                fk_position.tolist()
                            ),
                            "fk_quaternion_tool_z_xyzw": (
                                fk_quaternion.tolist()
                            ),
                            "fk_position_error_m": position_error,
                            "fk_orientation_error_rad": (
                                orientation_error
                            ),
                        }
                    )
                    previous_joints = candidate
                previous_position = route_position.copy()
                previous_quaternion = route_quaternion.copy()

            final = path[-1]
            waypoint = dict(source)
            waypoint.update(
                {
                    "index": frame_index,
                    "pose_semantic": "T_base_right_gripper_tool_z",
                    "quaternion_order": "xyzw",
                    "target_ee_position_base_m": target_position.tolist(),
                    "target_quaternion_xyzw": target_quaternion.tolist(),
                    "task_equivalent_tool_z_twist_rad": tool_z_twist_rad,
                    "solver_seed_joints_rad": segment_start_joints.tolist(),
                    "joint_feedback_solution_rad": previous_joints.tolist(),
                    "fk_position_base_m": final["fk_position_base_m"],
                    "fk_quaternion_xyzw_base": final[
                        "fk_quaternion_tool_z_xyzw"
                    ],
                    "fk_position_error_m": final[
                        "fk_position_error_m"
                    ],
                    "fk_orientation_error_rad": final[
                        "fk_orientation_error_rad"
                    ],
                    "joint_path_from_previous": path,
                    "segment_position_distance_m": float(
                        np.linalg.norm(
                            target_position - segment_start_position
                        )
                    ),
                }
            )
            waypoints.append(waypoint)
            previous_position = target_position
            previous_quaternion = target_quaternion

        anchors = []
        for waypoint in waypoints:
            microsteps = [
                {
                    "step_index": step["step_index"],
                    "fraction": step["fraction"],
                    "route_name": step["route_name"],
                    "target_position_base_m": step[
                        "target_position_base_m"
                    ],
                    "target_tool_z_quaternion_xyzw": step[
                        "target_quaternion_tool_z_xyzw"
                    ],
                    "solver_seed_feedback_joints_rad": step[
                        "solver_seed_joints_rad"
                    ],
                    "solution_feedback_joints_rad": step[
                        "solution_joints_rad"
                    ],
                    "generated_fk_position_base_m": step[
                        "fk_position_base_m"
                    ],
                    "generated_fk_tool_z_quaternion_xyzw": step[
                        "fk_quaternion_tool_z_xyzw"
                    ],
                    "generated_fk_position_error_m": step[
                        "fk_position_error_m"
                    ],
                    "generated_fk_orientation_error_rad": step[
                        "fk_orientation_error_rad"
                    ],
                    "incremental_joint_delta_rad": step[
                        "max_incremental_joint_delta_rad"
                    ],
                    "solver_refinement_count": step[
                        "solver_refinement_count"
                    ],
                }
                for step in waypoint["joint_path_from_previous"]
            ]
            anchors.append(
                {
                    "index": waypoint["index"],
                    "id": waypoint["id"],
                    "task_label": waypoint.get(
                        "description",
                        waypoint["id"],
                    ),
                    "target_position_base_m": waypoint[
                        "target_ee_position_base_m"
                    ],
                    "target_tool_z_quaternion_xyzw": waypoint[
                        "target_quaternion_xyzw"
                    ],
                    "task_equivalent_tool_z_twist_rad": waypoint.get(
                        "task_equivalent_tool_z_twist_rad",
                        0.0,
                    ),
                    "task_invariance": (
                        "tennis_ball_grasp_is_invariant_to_rotation_"
                        "about_the_tool_z_approach_axis"
                    ),
                    "segment_seed_feedback_joints_rad": waypoint[
                        "solver_seed_joints_rad"
                    ],
                    "solution_feedback_joints_rad": waypoint[
                        "joint_feedback_solution_rad"
                    ],
                    "generated_fk_position_base_m": waypoint[
                        "fk_position_base_m"
                    ],
                    "generated_fk_tool_z_quaternion_xyzw": waypoint[
                        "fk_quaternion_xyzw_base"
                    ],
                    "generated_fk_position_error_m": waypoint[
                        "fk_position_error_m"
                    ],
                    "generated_fk_orientation_error_rad": waypoint[
                        "fk_orientation_error_rad"
                    ],
                    "microsteps": microsteps,
                }
            )
        regenerated = {
            "schema_version": "r1lite.nominal_plan.v2",
            "plan_id": f"{args.run_id}_{args.scope}",
            "generated_at_unix": time.time(),
            "generation_run_id": args.run_id,
            "status": "candidate_unverified",
            "scope": args.scope,
            "execution": "solver_only_no_robot_motion_commands_sent",
            "pose_semantic": "T_base_right_gripper_tool_z",
            "quaternion_order": "xyzw",
            "solver_base_link": "torso_link3",
            "solver_ee_link": "right_gripper_tool_z",
            "initial_arm_feedback_rad": start_joints.tolist(),
            "initial_torso_feedback_rad": torso_joints.tolist(),
            "source_paths": {
                "source_targets": str(args.source_plan),
                "relaxed_ik_settings": str(SETTINGS),
                "relaxed_ik_urdf": str(RELAXED_URDF),
                "full_r1lite_urdf": str(FULL_URDF),
                "low_level_config": str(CONFIG),
            },
            "source_sha256": {
                "source_targets": sha256(args.source_plan),
                "relaxed_ik_settings": sha256(SETTINGS),
                "relaxed_ik_urdf": sha256(RELAXED_URDF),
                "full_r1lite_urdf": sha256(FULL_URDF),
                "low_level_config": sha256(CONFIG),
            },
            "validation_limits": {
                "max_position_step_m": MAX_POSITION_STEP_M,
                "max_orientation_step_rad": MAX_ORIENTATION_STEP_RAD,
                "max_incremental_joint_delta_rad": (
                    MAX_INCREMENTAL_JOINT_DELTA_RAD
                ),
                "max_fk_position_error_m": MAX_FK_POSITION_ERROR_M,
                "max_fk_orientation_error_rad": (
                    MAX_FK_ORIENTATION_ERROR_RAD
                ),
                "max_solver_refinements": MAX_SOLVER_REFINEMENTS,
            },
            "anchors": anchors,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(regenerated, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"[wrote] {args.output}")
        for waypoint in waypoints:
            print(
                waypoint["index"],
                waypoint["id"],
                f"steps={len(waypoint['joint_path_from_previous'])}",
                f"fk_pos_err={waypoint['fk_position_error_m']:.4f}",
                f"fk_ori_err={waypoint['fk_orientation_error_rad']:.4f}",
            )
    except Exception as exc:
        failure_path = args.output.with_name(
            f"{args.output.stem}_generation_failure.json"
        )
        if failure_path.exists():
            raise FileExistsError(
                "Failure report exists; use a new output path"
            ) from exc
        failure_path.write_text(
            json.dumps(
                {
                    "status": "generation_rejected",
                    "generation_run_id": args.run_id,
                    "scope": args.scope,
                    "source_plan": str(args.source_plan),
                    "source_plan_sha256": sha256(args.source_plan),
                    "robot_motion_commands_sent": False,
                    "validation_limits": {
                        "max_position_step_m": MAX_POSITION_STEP_M,
                        "max_orientation_step_rad": (
                            MAX_ORIENTATION_STEP_RAD
                        ),
                        "max_incremental_joint_delta_rad": (
                            MAX_INCREMENTAL_JOINT_DELTA_RAD
                        ),
                        "max_fk_position_error_m": (
                            MAX_FK_POSITION_ERROR_M
                        ),
                        "max_fk_orientation_error_rad": (
                            MAX_FK_ORIENTATION_ERROR_RAD
                        ),
                    },
                    "error": repr(exc),
                    "conclusion": (
                        "No candidate plan was written and execution remains "
                        "prohibited. The task target or path constraints must "
                        "change; validation thresholds were not relaxed."
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"[rejected] {failure_path}")
        print(repr(exc))
        return 2
    finally:
        r1.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
