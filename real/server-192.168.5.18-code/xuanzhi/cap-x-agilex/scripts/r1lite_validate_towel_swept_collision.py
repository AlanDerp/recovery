#!/usr/bin/env python3
"""Solve an approved towel plan with isolated IK and check swept capsules."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel  # noqa: E402
from capx.integrations.r1lite.bimanual_kinematics import (  # noqa: E402
    R1LiteBimanualKinematics,
)
from capx.integrations.r1lite.execution_bundle import (  # noqa: E402
    build_execution_bundle,
    canonical_json_bytes,
    sha256_file,
)
from capx.integrations.r1lite.kinematics import (  # noqa: E402
    FULL_URDF,
    LEFT_RELAXED_URDF,
    RELAXED_URDF,
    R1LiteCandidateKinematics,
)
from capx.integrations.r1lite.towel_plan import (  # noqa: E402
    MINIMUM_GRIPPER_CENTER_SEPARATION_WITH_MARGIN_M,
    validate_towel_fold_plan,
)


FK_POSITION_TOL_M = 0.030
FK_ORIENTATION_TOL_RAD = 0.25
MAX_STAGE_JOINT_DELTA_RAD = 1.0
SWEEP_SAMPLES_PER_SEGMENT = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-bundle-output", type=Path)
    return parser.parse_args()


def load_kwargs(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text())
    low_cfg = dict(cfg["env"]["cfg"]["low_level"])
    low_cfg.pop("_target_", None)
    signature = inspect.signature(R1LiteRealLowLevel.__init__)
    result = {
        name: value
        for name, value in low_cfg.items()
        if name in signature.parameters
    }
    result.update(
        {
            "wrist_camera_enabled": False,
            "relaxed_ik_probe_enabled": True,
            "r1lite_motion_validation_enabled": False,
        }
    )
    return result


def make_robot(config_path: Path, side: str) -> R1LiteRealLowLevel:
    kwargs = load_kwargs(config_path)
    if side == "left":
        kwargs.update(
            {
                "arm_name": "left_arm",
                "gripper_name": "left_gripper",
                "arm_feedback_topic": "/hdas/feedback_arm_left",
                "gripper_feedback_topic": "/hdas/feedback_gripper_left",
                "arm_target_topic": "/relaxed_ik/never_execute_arm_left",
                "arm_control_topic": "/motion_control/control_arm_left",
                "gripper_target_topic": "/relaxed_ik/never_execute_gripper_left",
                "ee_feedback_topic": "/motion_control/pose_ee_arm_left",
                "relaxed_ik_target_topic": "/capx/relaxed_ik/target_pose_arm_left",
                "relaxed_ik_solution_topic": "/relaxed_ik/joint_angle_solutions_left",
                "relaxed_ik_reset_topic": "/relaxed_ik/reset_left",
                "relaxed_ik_fk_topic": "/relaxed_ik/ik_ee_pose_left",
                "relaxed_ik_ee_feedback_topic": (
                    "/relaxed_ik/motion_control/pose_ee_arm_left"
                ),
            }
        )
    else:
        kwargs.update(
            {
                "arm_target_topic": "/relaxed_ik/never_execute_arm_right",
                "gripper_target_topic": "/relaxed_ik/never_execute_gripper_right",
                "relaxed_ik_target_topic": "/capx/relaxed_ik/target_pose_arm_right",
            }
        )
    return R1LiteRealLowLevel(**kwargs)


def topdown_quaternion(yaw_deg: float) -> np.ndarray:
    return (
        Rotation.from_euler("z", np.deg2rad(yaw_deg))
        * Rotation.from_euler("y", np.pi)
    ).as_quat()


def tilted_quaternion(
    lateral: np.ndarray,
    *,
    side: str,
    tilt_deg: float,
    roll_deg: float,
) -> np.ndarray:
    lateral = np.asarray(lateral, dtype=float).reshape(3)
    lateral[2] = 0.0
    lateral /= np.linalg.norm(lateral)
    sign = 1.0 if side == "left" else -1.0
    tilt = np.deg2rad(float(tilt_deg))
    z_axis = sign * lateral * np.sin(tilt) + np.array(
        [0.0, 0.0, -np.cos(tilt)]
    )
    y_axis = np.array([-lateral[1], lateral[0], 0.0])
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    base = Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))
    return (base * Rotation.from_rotvec([0.0, 0.0, np.deg2rad(roll_deg)])).as_quat()


def directed_tilted_quaternion(
    direction_xy: np.ndarray,
    *,
    tilt_deg: float,
    roll_deg: float,
) -> np.ndarray:
    direction = np.asarray(direction_xy, dtype=float).reshape(3).copy()
    direction[2] = 0.0
    direction /= np.linalg.norm(direction)
    tilt = np.deg2rad(float(tilt_deg))
    z_axis = direction * np.sin(tilt) + np.array(
        [0.0, 0.0, -np.cos(tilt)]
    )
    y_axis = np.array([-direction[1], direction[0], 0.0])
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    base = Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))
    return (
        base * Rotation.from_rotvec([0.0, 0.0, np.deg2rad(roll_deg)])
    ).as_quat()


def left_tool_to_solver_quaternion(tool_quaternion: np.ndarray) -> np.ndarray:
    return (
        Rotation.from_quat(tool_quaternion)
        * Rotation.from_euler("y", -np.pi / 2.0)
    ).as_quat()


def orientation_error(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def solve_candidates(
    robot: R1LiteRealLowLevel,
    kinematics: R1LiteCandidateKinematics,
    *,
    side: str,
    tool_targets: list[tuple[str, np.ndarray, np.ndarray]],
    reference_joints: np.ndarray,
    torso_joints: np.ndarray,
    max_joint_delta_rad: float,
    max_qualified: int | None = None,
    position_xy_tolerance_m: float | None = None,
    position_z_tolerance_m: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qualified = []
    rejected = []
    for label, target_position, tool_quaternion in tool_targets:
        solver_quaternion = (
            left_tool_to_solver_quaternion(tool_quaternion)
            if side == "left"
            else tool_quaternion
        )
        try:
            robot.seed_relaxed_ik_probe(reference_joints, duration_s=0.12)
            samples = []
            accepted = None
            for iteration in range(1, 21):
                joints = robot.get_relaxed_ik_candidate(
                    target_position,
                    solver_quaternion,
                    timeout_s=0.6,
                    reference_joints=reference_joints,
                    republish_interval_s=None,
                )
                fk = kinematics.fk(joints, torso_joints)
                position_error = float(
                    np.linalg.norm(fk.position_base_m - target_position)
                )
                rotation_error = orientation_error(
                    fk.quaternion_xyzw,
                    tool_quaternion,
                )
                item = {
                    "label": label,
                    "ik_iteration": iteration,
                    "target_position_base_m": target_position.tolist(),
                    "target_quaternion_xyzw": tool_quaternion.tolist(),
                    "joints_rad": joints.tolist(),
                    "fk_position_base_m": fk.position_base_m.tolist(),
                    "fk_quaternion_xyzw": fk.quaternion_xyzw.tolist(),
                    "fk_position_error_m": position_error,
                    "fk_orientation_error_rad": rotation_error,
                    "max_joint_delta_rad": float(
                        np.max(np.abs(joints - reference_joints))
                    ),
                }
                samples.append(item)
                difference = fk.position_base_m - target_position
                axis_tolerances_pass = (
                    position_xy_tolerance_m is None
                    or float(np.linalg.norm(difference[:2]))
                    <= position_xy_tolerance_m
                ) and (
                    position_z_tolerance_m is None
                    or float(abs(difference[2])) <= position_z_tolerance_m
                )
                if (
                    position_error <= FK_POSITION_TOL_M
                    and rotation_error <= FK_ORIENTATION_TOL_RAD
                    and item["max_joint_delta_rad"] <= max_joint_delta_rad
                    and axis_tolerances_pass
                ):
                    accepted = item
                    break
            if accepted is not None:
                qualified.append(accepted)
                if (
                    max_qualified is not None
                    and len(qualified) >= max_qualified
                ):
                    break
            else:
                best = min(
                    samples,
                    key=lambda item: (
                        max(
                            0.0,
                            item["max_joint_delta_rad"]
                            - max_joint_delta_rad,
                        ),
                        item["fk_position_error_m"],
                        item["fk_orientation_error_rad"],
                    ),
                )
                rejected.append(
                    {
                        **best,
                        "iterations_tested": len(samples),
                        "reason": "joint_or_fk_gate",
                    }
                )
        except Exception as exc:
            rejected.append({"label": label, "reason": repr(exc)})
    qualified.sort(
        key=lambda item: (
            item["max_joint_delta_rad"],
            item["fk_position_error_m"],
            item["fk_orientation_error_rad"],
        )
    )
    return qualified, rejected


def solve_stage_transit(
    *,
    stage_id: str,
    robots: dict[str, R1LiteRealLowLevel],
    local_fk: dict[str, R1LiteCandidateKinematics],
    safety: R1LiteBimanualKinematics,
    torso: np.ndarray,
    initial_references: dict[str, np.ndarray],
    final_targets: dict[str, tuple[str, np.ndarray, np.ndarray]],
    maximum_cartesian_step_m: float = 0.08,
    maximum_orientation_step_deg: float = 15.0,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[list[float]]]]:
    references = {
        side: np.asarray(value, dtype=float).copy()
        for side, value in initial_references.items()
    }
    starts = {side: local_fk[side].fk(references[side], torso) for side in ("left", "right")}
    step_count = 1
    for side in ("left", "right"):
        _label, final_position, final_quaternion = final_targets[side]
        distance = float(
            np.linalg.norm(final_position - starts[side].position_base_m)
        )
        angle = orientation_error(
            starts[side].quaternion_xyzw,
            final_quaternion,
        )
        step_count = max(
            step_count,
            int(np.ceil(distance / maximum_cartesian_step_m)),
            int(np.ceil(np.rad2deg(angle) / maximum_orientation_step_deg)),
        )
    waypoints = {"left": [], "right": []}
    selected: dict[str, dict[str, Any]] = {}
    for step_index in range(1, step_count + 1):
        alpha = step_index / step_count
        results = {}
        for side in ("left", "right"):
            label, final_position, final_quaternion = final_targets[side]
            start_fk = starts[side]
            target_position = (
                start_fk.position_base_m * (1.0 - alpha)
                + final_position * alpha
            )
            rotations = Rotation.from_quat(
                np.vstack(
                    [start_fk.quaternion_xyzw, final_quaternion]
                )
            )
            target_quaternion = Slerp([0.0, 1.0], rotations)([alpha]).as_quat()[0]
            qualified, rejected = solve_candidates(
                robots[side],
                local_fk[side],
                side=side,
                tool_targets=[
                    (
                        f"transit_{step_index:02d}_of_{step_count:02d}_{label}",
                        target_position,
                        target_quaternion,
                    )
                ],
                reference_joints=references[side],
                torso_joints=torso,
                max_joint_delta_rad=MAX_STAGE_JOINT_DELTA_RAD,
            )
            if not qualified and step_index < step_count:
                current_fk = local_fk[side].fk(references[side], torso)
                current_final_error = float(
                    np.linalg.norm(
                        current_fk.position_base_m - final_position
                    )
                )
                progress_candidates = []
                for item in rejected:
                    if (
                        "fk_position_base_m" not in item
                        or item["max_joint_delta_rad"]
                        > MAX_STAGE_JOINT_DELTA_RAD
                        or item["fk_orientation_error_rad"]
                        > FK_ORIENTATION_TOL_RAD
                    ):
                        continue
                    final_error = float(
                        np.linalg.norm(
                            np.asarray(item["fk_position_base_m"], dtype=float)
                            - final_position
                        )
                    )
                    improvement = current_final_error - final_error
                    if improvement >= 0.001:
                        progress_candidates.append(
                            (improvement, final_error, item)
                        )
                if progress_candidates:
                    improvement, _, progress_item = max(
                        progress_candidates,
                        key=lambda value: (value[0], -value[1]),
                    )
                    qualified = [
                        {
                            **progress_item,
                            "accepted_via": "predicted_final_improvement",
                            "predicted_final_improvement_m": float(improvement),
                        }
                    ]
            if not qualified:
                raise RuntimeError(
                    f"{stage_id} transit {step_index}/{step_count} {side} "
                    f"has no qualified IK candidate: {rejected}"
                )
            results[side] = qualified
        pairs = []
        for left_item in results["left"]:
            for right_item in results["right"]:
                left_joints = np.asarray(left_item["joints_rad"], dtype=float)
                right_joints = np.asarray(right_item["joints_rad"], dtype=float)
                try:
                    reports = safety.validate_interpolated_segment(
                        references["left"],
                        references["right"],
                        left_joints,
                        right_joints,
                        torso,
                        samples=SWEEP_SAMPLES_PER_SEGMENT,
                    )
                except ValueError:
                    continue
                pairs.append(
                    (
                        max(
                            left_item["max_joint_delta_rad"],
                            right_item["max_joint_delta_rad"],
                        ),
                        min(report.estimated_link_clearance_m for report in reports),
                        left_item,
                        right_item,
                    )
                )
        if not pairs:
            raise RuntimeError(
                f"{stage_id} transit {step_index}/{step_count} has no safe pair"
            )
        pairs.sort(key=lambda item: (item[0], -item[1]))
        _, _clearance, left_best, right_best = pairs[0]
        selected = {"left": left_best, "right": right_best}
        for side, item in selected.items():
            references[side] = np.asarray(item["joints_rad"], dtype=float)
            waypoints[side].append(item["joints_rad"])
    return selected, waypoints


def main() -> None:
    args = parse_args()
    if args.execution_bundle_output is not None:
        args.execution_bundle_output.unlink(missing_ok=True)
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    validate_towel_fold_plan(plan)
    left_robot = make_robot(args.config, "left")
    right_robot = make_robot(args.config, "right")
    audit: dict[str, Any] = {
        "experiment": "pipeline7_isolated_ik_swept_collision",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "robot_motion_commands_sent": False,
        "isolated_ik_topics": True,
        "cartesian_refinement_policy": plan["safety"][
            "cartesian_refinement_policy"
        ],
        "passed": False,
        "stages": [],
        "segments": [],
    }
    try:
        torso = right_robot.get_torso_joints()
        references = {
            "left": left_robot.get_arm_joints(),
            "right": right_robot.get_arm_joints(),
        }
        initial = {side: value.copy() for side, value in references.items()}
        local_fk = {
            "left": R1LiteCandidateKinematics(side="left"),
            "right": R1LiteCandidateKinematics(side="right"),
        }
        safety = R1LiteBimanualKinematics(
            minimum_tcp_separation_m=(
                MINIMUM_GRIPPER_CENTER_SEPARATION_WITH_MARGIN_M
            )
        )
        lateral = (
            np.asarray(
                plan["stages"][-1]["right_fingertip_position_base_m"],
                dtype=float,
            )
            - np.asarray(
                plan["stages"][-1]["left_fingertip_position_base_m"],
                dtype=float,
            )
        )
        chosen: list[dict[str, Any]] = []
        for stage in plan["stages"]:
            orientation_sets = {}
            for side in ("left", "right"):
                sign = -1.0 if side == "left" else 1.0
                outward = sign * lateral
                outward[2] = 0.0
                forward = np.array([1.0, 0.0, 0.0])
                diagonal = forward + outward / np.linalg.norm(outward)
                diagonal /= np.linalg.norm(diagonal)
                directions = (
                    [
                        ("forward_outward", diagonal),
                        ("outward", outward),
                    ]
                    if stage["id"] == "fold_center"
                    else [
                        ("forward", forward),
                        ("forward_outward", diagonal),
                        ("outward", outward),
                    ]
                )
                orientation_sets[side] = []
                for direction_name, direction in directions:
                    for tilt in (
                        [60.0, 45.0]
                        if stage["id"] == "fold_center"
                        else list(reversed(stage["tilt_search_deg"]))
                    ):
                        for roll in (0.0, -90.0, 90.0, 180.0):
                            quaternion = directed_tilted_quaternion(
                                direction,
                                tilt_deg=tilt,
                                roll_deg=roll,
                            )
                            position = np.asarray(
                                stage[f"{side}_fingertip_position_base_m"],
                                dtype=float,
                            ) - float(
                                plan["safety"]["tcp_to_fingertip_m"]
                            ) * Rotation.from_quat(quaternion).as_matrix()[:, 2]
                            orientation_sets[side].append(
                                (
                                    f"dir{direction_name}_tilt{tilt:.0f}_"
                                    f"roll{roll:+.0f}",
                                    position,
                                    quaternion,
                                )
                            )
            if stage["id"] in ("pregrasp", "fold_center"):
                by_label = {
                    side: {item[0]: item for item in orientation_sets[side]}
                    for side in ("left", "right")
                }
                common_labels = set(by_label["left"]) & set(by_label["right"])
                if stage["id"] == "fold_center":
                    priority = sorted(
                        common_labels,
                        key=lambda label: (
                            0
                            if "dirforward_outward_tilt60" in label
                            else 1,
                            label,
                        ),
                    )
                else:
                    priority = sorted(
                        common_labels,
                        key=lambda label: (
                            0 if "dirforward_tilt45" in label else 1,
                            0
                            if "dirforward_outward_tilt45" in label
                            else 1,
                            0 if "dirforward_" in label else 1,
                            label,
                        ),
                    )
                transit_errors = []
                transit_selected = None
                transit_waypoints = None
                for label in priority:
                    try:
                        transit_selected, transit_waypoints = solve_stage_transit(
                            stage_id=stage["id"],
                            robots={"left": left_robot, "right": right_robot},
                            local_fk=local_fk,
                            safety=safety,
                            torso=torso,
                            initial_references=references,
                            final_targets={
                                side: by_label[side][label]
                                for side in ("left", "right")
                            },
                        )
                        break
                    except Exception as exc:
                        transit_errors.append(
                            {"orientation_label": label, "error": repr(exc)}
                        )
                if transit_selected is None or transit_waypoints is None:
                    audit["stages"].append(
                        {
                            "id": stage["id"],
                            "transit_attempts": transit_errors,
                            "pair_gate_passed": False,
                        }
                    )
                    raise RuntimeError(
                        f"stage {stage['id']} has no collision-safe "
                        "interpolated IK transit"
                    )
                references = {
                    side: np.asarray(
                        transit_selected[side]["joints_rad"], dtype=float
                    )
                    for side in ("left", "right")
                }
                pair_report = safety.validate(
                    references["left"],
                    references["right"],
                    torso,
                )
                chosen.append(
                    {
                        "id": stage["id"],
                        "left": transit_selected["left"],
                        "right": transit_selected["right"],
                        "joint_waypoints_rad": transit_waypoints,
                    }
                )
                audit["stages"].append(
                    {
                        "id": stage["id"],
                        "transit_waypoint_count": len(
                            transit_waypoints["left"]
                        ),
                        "selected": transit_selected,
                        "pair_safety": pair_report.to_dict(),
                        "pair_gate_passed": True,
                    }
                )
                continue
            results = {}
            rejected = {}
            for side, robot in (
                ("left", left_robot),
                ("right", right_robot),
            ):
                results[side], rejected[side] = solve_candidates(
                    robot,
                    local_fk[side],
                    side=side,
                    tool_targets=orientation_sets[side],
                    reference_joints=references[side],
                    torso_joints=torso,
                    max_joint_delta_rad=MAX_STAGE_JOINT_DELTA_RAD,
                    max_qualified=4,
                )
            valid_pairs = []
            for left_item in results["left"]:
                for right_item in results["right"]:
                    try:
                        report = safety.validate(
                            np.asarray(left_item["joints_rad"]),
                            np.asarray(right_item["joints_rad"]),
                            torso,
                        )
                    except Exception:
                        continue
                    valid_pairs.append(
                        (
                            left_item["max_joint_delta_rad"]
                            + right_item["max_joint_delta_rad"],
                            left_item,
                            right_item,
                            report,
                        )
                    )
            if not valid_pairs:
                audit["stages"].append(
                    {
                        "id": stage["id"],
                        "qualified_counts": {
                            side: len(results[side]) for side in results
                        },
                        "rejected": rejected,
                        "pair_gate_passed": False,
                    }
                )
                raise RuntimeError(
                    f"stage {stage['id']} has no collision-safe paired IK solution"
                )
            _score, left_best, right_best, pair_report = min(
                valid_pairs,
                key=lambda item: item[0],
            )
            references = {
                "left": np.asarray(left_best["joints_rad"], dtype=float),
                "right": np.asarray(right_best["joints_rad"], dtype=float),
            }
            chosen.append(
                {
                    "id": stage["id"],
                    "left": left_best,
                    "right": right_best,
                    "joint_waypoints_rad": {
                        "left": [left_best["joints_rad"]],
                        "right": [right_best["joints_rad"]],
                    },
                }
            )
            audit["stages"].append(
                {
                    "id": stage["id"],
                    "qualified_counts": {
                        side: len(results[side]) for side in results
                    },
                    "selected": {
                        "left": left_best,
                        "right": right_best,
                    },
                    "pair_safety": pair_report.to_dict(),
                    "pair_gate_passed": True,
                }
            )
        previous = initial
        minimum_clearance = float("inf")
        minimum_tcp_separation = float("inf")
        for selected in chosen:
            current = {
                "left": np.asarray(selected["left"]["joints_rad"], dtype=float),
                "right": np.asarray(selected["right"]["joints_rad"], dtype=float),
            }
            reports = safety.validate_interpolated_segment(
                previous["left"],
                previous["right"],
                current["left"],
                current["right"],
                torso,
                samples=SWEEP_SAMPLES_PER_SEGMENT,
            )
            segment_clearance = min(
                item.estimated_link_clearance_m for item in reports
            )
            segment_tcp = min(item.tcp_separation_m for item in reports)
            minimum_clearance = min(minimum_clearance, segment_clearance)
            minimum_tcp_separation = min(minimum_tcp_separation, segment_tcp)
            audit["segments"].append(
                {
                    "to_stage": selected["id"],
                    "samples": len(reports),
                    "minimum_estimated_link_clearance_m": segment_clearance,
                    "minimum_tcp_separation_m": segment_tcp,
                    "passed": True,
                }
            )
            previous = current
        audit.update(
            {
                "passed": True,
                "minimum_swept_link_clearance_m": minimum_clearance,
                "minimum_swept_tcp_separation_m": minimum_tcp_separation,
                "sweep_samples_per_segment": SWEEP_SAMPLES_PER_SEGMENT,
                "collision_model": (
                    "arm_link_capsules_plus_gripper_TCP_mesh_envelope"
                ),
            }
        )
        if args.execution_bundle_output is not None:
            collision_model = {
                "name": audit["collision_model"],
                "link_capsule_radius_m": safety.link_capsule_radius_m,
                "minimum_link_clearance_m": safety.minimum_link_clearance_m,
                "minimum_tcp_separation_m": safety.minimum_tcp_separation_m,
                "sweep_interpolation": "linear_joint_space",
                "sweep_samples_per_segment": SWEEP_SAMPLES_PER_SEGMENT,
            }
            bundle_stages = []
            for selected in chosen:
                bundle_stage = {"id": selected["id"]}
                for side in ("left", "right"):
                    waypoint_values = selected["joint_waypoints_rad"][side]
                    previous_joint_values = (
                        initial[side]
                        if not bundle_stages
                        else np.asarray(
                            bundle_stages[-1][side]["joints_rad"], dtype=float
                        )
                    )
                    maximum_waypoint_delta = 0.0
                    for waypoint in waypoint_values:
                        waypoint_array = np.asarray(waypoint, dtype=float)
                        maximum_waypoint_delta = max(
                            maximum_waypoint_delta,
                            float(
                                np.max(
                                    np.abs(
                                        waypoint_array - previous_joint_values
                                    )
                                )
                            ),
                        )
                        previous_joint_values = waypoint_array
                    bundle_stage[side] = {
                        "joints_rad": selected[side]["joints_rad"],
                        "joint_waypoints_rad": waypoint_values,
                        "target_position_base_m": selected[side][
                            "target_position_base_m"
                        ],
                        "target_quaternion_xyzw": selected[side][
                            "target_quaternion_xyzw"
                        ],
                        "fk_position_base_m": selected[side]["fk_position_base_m"],
                        "fk_quaternion_xyzw": selected[side][
                            "fk_quaternion_xyzw"
                        ],
                        "fk_position_error_m": selected[side][
                            "fk_position_error_m"
                        ],
                        "fk_orientation_error_rad": selected[side][
                            "fk_orientation_error_rad"
                        ],
                        "maximum_joint_delta_rad": maximum_waypoint_delta,
                    }
                bundle_stages.append(bundle_stage)
            bundle = build_execution_bundle(
                plan_sha256=audit["plan_sha256"],
                config_sha256=sha256_file(args.config),
                full_urdf_sha256=sha256_file(FULL_URDF),
                left_relaxed_urdf_sha256=sha256_file(LEFT_RELAXED_URDF),
                right_relaxed_urdf_sha256=sha256_file(RELAXED_URDF),
                tcp_definition=plan["tcp"],
                collision_model=collision_model,
                initial_left_joints_rad=initial["left"],
                initial_right_joints_rad=initial["right"],
                torso_joints_rad=torso,
                stages=bundle_stages,
                sweeps={
                    "samples_per_segment": SWEEP_SAMPLES_PER_SEGMENT,
                    "segments": audit["segments"],
                    "minimum_swept_link_clearance_m": minimum_clearance,
                    "minimum_swept_tcp_separation_m": minimum_tcp_separation,
                },
                maximum_stage_joint_delta_rad=MAX_STAGE_JOINT_DELTA_RAD,
                fk_position_tolerance_m=FK_POSITION_TOL_M,
                fk_orientation_tolerance_rad=FK_ORIENTATION_TOL_RAD,
            )
            args.execution_bundle_output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            args.execution_bundle_output.write_bytes(canonical_json_bytes(bundle))
    except Exception as exc:
        audit["error"] = repr(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
        left_robot.close()
        right_robot.close()
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
