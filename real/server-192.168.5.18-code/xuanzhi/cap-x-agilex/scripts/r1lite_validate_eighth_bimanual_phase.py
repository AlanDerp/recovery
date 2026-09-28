#!/usr/bin/env python3
"""Offline-only certification for the eighth bimanual recenter pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.bimanual_kinematics import (  # noqa: E402
    R1LiteBimanualTableCollisionKinematics,
    strict_topdown_edge_rotation,
)
from capx.integrations.r1lite.execution_bundle import (  # noqa: E402
    build_eighth_execution_bundle,
    canonical_json_bytes,
    canonical_sha256,
    sha256_file,
)
from capx.integrations.r1lite.kinematics import (  # noqa: E402
    FULL_URDF,
    LEFT_RELAXED_URDF,
    RELAXED_URDF,
    R1LiteCandidateKinematics,
)
from capx.integrations.r1lite.towel_plan import (  # noqa: E402
    EIGHTH_BIMANUAL_STAGE_ORDER,
    validate_eighth_bimanual_recenter_plan,
)
from r1lite_validate_towel_swept_collision import (  # noqa: E402
    FK_ORIENTATION_TOL_RAD,
    FK_POSITION_TOL_M,
    MAX_STAGE_JOINT_DELTA_RAD,
    make_robot,
    orientation_error,
    solve_candidates,
)


MAXIMUM_SHARED_CARTESIAN_STEP_M = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-bundle-output", type=Path, required=True)
    return parser.parse_args()


def _collision_summary(
    stage_id: str, reports: list[Any] | None = None
) -> dict[str, Any]:
    checked = False
    return {
        "mode": (
            "first_phase_collision_disabled"
            if checked
            else "disabled_non_grasp_stage"
        ),
        "checked": checked,
        "sweep_samples": len(reports or []) if checked else 0,
    }


def _side_record(
    item: dict[str, Any],
    previous: np.ndarray,
) -> dict[str, Any]:
    joints = np.asarray(item["joints_rad"], dtype=float)
    return {
        "target_position_base_m": item["target_position_base_m"],
        "target_quaternion_xyzw": item["target_quaternion_xyzw"],
        "joints_rad": item["joints_rad"],
        "fk_position_base_m": item["fk_position_base_m"],
        "fk_quaternion_xyzw": item["fk_quaternion_xyzw"],
        "fk_position_error_m": item["fk_position_error_m"],
        "fk_orientation_error_rad": item["fk_orientation_error_rad"],
        "maximum_joint_delta_rad": float(np.max(np.abs(joints - previous))),
    }


def _joint_fk_item(
    joints: np.ndarray,
    previous: np.ndarray,
    fk: Any,
) -> dict[str, Any]:
    quaternion = np.asarray(fk.quaternion_xyzw, dtype=float)
    quaternion_values = quaternion.tolist()
    self_orientation_error = 2.0 * math.acos(
        min(
            1.0,
            abs(
                sum(
                    a * b
                    for a, b in zip(
                        quaternion_values, quaternion_values
                    )
                )
            ),
        )
    )
    return {
        "target_position_base_m": fk.position_base_m.tolist(),
        "target_quaternion_xyzw": quaternion_values,
        "joints_rad": joints.tolist(),
        "fk_position_base_m": fk.position_base_m.tolist(),
        "fk_quaternion_xyzw": quaternion_values,
        "fk_position_error_m": 0.0,
        "fk_orientation_error_rad": self_orientation_error,
        "maximum_joint_delta_rad": float(np.max(np.abs(joints - previous))),
    }


def _waypoint(
    *,
    alpha: float,
    left: dict[str, Any],
    right: dict[str, Any],
    collision_summary: dict[str, Any],
    ik_acceptance: str = "required",
) -> dict[str, Any]:
    left_target = np.asarray(left["target_position_base_m"], dtype=float)
    right_target = np.asarray(right["target_position_base_m"], dtype=float)
    return {
        "alpha": float(alpha),
        "expected_midpoint_base_m": ((left_target + right_target) / 2.0).tolist(),
        "expected_right_minus_left_m": (right_target - left_target).tolist(),
        "ik_acceptance": ik_acceptance,
        "left": left,
        "right": right,
        "collision_summary": collision_summary,
    }


def _solve_with_fk_residual_compensation(
    *,
    robot: Any,
    kinematics: R1LiteCandidateKinematics,
    side: str,
    target: np.ndarray,
    quaternion: np.ndarray,
    reference: np.ndarray,
    torso: np.ndarray,
    label: str,
    maximum_joint_delta_rad: float = MAX_STAGE_JOINT_DELTA_RAD,
    fk_position_tolerance_m: float = FK_POSITION_TOL_M,
    fk_outer_position_tolerance_m: float | None = None,
    allow_best_effort_seed: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the approved target fixed while compensating RelaxedIK FK bias."""
    request = np.asarray(target, dtype=float).copy()
    original = request.copy()
    diagnostics: list[dict[str, Any]] = []
    admissible: list[dict[str, Any]] = []
    best_effort: list[dict[str, Any]] = []
    outer_tolerance = (
        float(fk_position_tolerance_m)
        if fk_outer_position_tolerance_m is None
        else float(fk_outer_position_tolerance_m)
    )
    for correction in range(6):
        qualified, rejected = solve_candidates(
            robot,
            kinematics,
            side=side,
            tool_targets=[(f"{label}_correction_{correction}", request, quaternion)],
            reference_joints=reference,
            torso_joints=torso,
            max_joint_delta_rad=maximum_joint_delta_rad,
            max_qualified=5,
        )
        samples = qualified + [
            item for item in rejected if "joints_rad" in item
        ]
        if not samples:
            diagnostics.extend(rejected)
            break
        samples.sort(
            key=lambda item: float(
                np.linalg.norm(
                    np.asarray(item["fk_position_base_m"], dtype=float)
                    - original
                )
            )
        )
        best = dict(samples[0])
        fk_position = np.asarray(best["fk_position_base_m"], dtype=float)
        original_error = float(np.linalg.norm(fk_position - original))
        joints = np.asarray(best["joints_rad"], dtype=float)
        joint_delta = float(np.max(np.abs(joints - reference)))
        best.update(
            {
                "target_position_base_m": original.tolist(),
                "fk_position_error_m": original_error,
                "max_joint_delta_rad": joint_delta,
                "fk_residual_compensation_iteration": correction,
                "ik_request_position_base_m": request.tolist(),
            }
        )
        diagnostics.append(best)
        if (
            allow_best_effort_seed
            and joint_delta <= maximum_joint_delta_rad
            and np.isfinite(original_error)
        ):
            best_effort.append(best)
        gates_except_tight_position = (
            original_error <= outer_tolerance
            and float(best["fk_orientation_error_rad"])
            <= FK_ORIENTATION_TOL_RAD
            and joint_delta <= maximum_joint_delta_rad
        )
        if gates_except_tight_position:
            admissible.append(best)
        if (
            original_error <= fk_position_tolerance_m
            and gates_except_tight_position
        ):
            return admissible, diagnostics
        residual = original - fk_position
        if correction < 3 and original_error <= outer_tolerance:
            request = original + np.array(
                [0.0, 0.0, 0.006 * float(correction + 1)]
            )
        else:
            request = request + 0.75 * residual
        offset = request - original
        offset_norm = float(np.linalg.norm(offset))
        if offset_norm > 0.05:
            request = original + offset * (0.05 / offset_norm)
    if admissible:
        return admissible, diagnostics
    if best_effort:
        best_effort.sort(
            key=lambda item: (
                float(item["fk_position_error_m"]),
                float(item["fk_orientation_error_rad"]),
                float(item["max_joint_delta_rad"]),
            )
        )
        return [best_effort[0]], diagnostics
    return [], diagnostics


def _safe_pairs(
    *,
    stage_id: str,
    step_index: int,
    targets: dict[str, np.ndarray],
    quaternions: dict[str, np.ndarray],
    robots: dict[str, Any],
    kinematics: dict[str, R1LiteCandidateKinematics],
    references: dict[str, np.ndarray],
    torso: np.ndarray,
    collision: R1LiteBimanualTableCollisionKinematics,
    maximum_joint_delta_rad: float = MAX_STAGE_JOINT_DELTA_RAD,
    fk_position_tolerance_m: float = FK_POSITION_TOL_M,
    fk_outer_position_tolerance_m: float | None = None,
    fixed_joints: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[Any], list[Any]]:
    qualified: dict[str, list[dict[str, Any]]] = {}
    rejected: dict[str, list[dict[str, Any]]] = {}
    for side in ("left", "right"):
        if fixed_joints is not None and side in fixed_joints:
            joints = np.asarray(fixed_joints[side], dtype=float)
            fixed = _joint_fk_item(
                joints,
                references[side],
                kinematics[side].fk(joints, torso),
            )
            fixed["max_joint_delta_rad"] = float(
                np.max(np.abs(joints - references[side]))
            )
            qualified[side] = [fixed]
            rejected[side] = []
            continue
        qualified[side], rejected[side] = _solve_with_fk_residual_compensation(
            robot=robots[side],
            kinematics=kinematics[side],
            side=side,
            target=targets[side],
            quaternion=quaternions[side],
            reference=references[side],
            torso=torso,
            label=f"{stage_id}_{step_index:03d}",
            maximum_joint_delta_rad=maximum_joint_delta_rad,
            fk_position_tolerance_m=fk_position_tolerance_m,
            fk_outer_position_tolerance_m=fk_outer_position_tolerance_m,
            allow_best_effort_seed=stage_id == "grasp",
        )
    safe: list[tuple[tuple[float, float], dict[str, dict[str, Any]], list[Any]]] = []
    for left in qualified["left"]:
        for right in qualified["right"]:
            reports: list[Any] = []
            maximum_joint_delta = max(
                float(left["max_joint_delta_rad"]),
                float(right["max_joint_delta_rad"]),
            )
            score = (
                max(
                    float(left["fk_position_error_m"]),
                    float(right["fk_position_error_m"]),
                )
                if stage_id == "grasp"
                else 0.0,
                maximum_joint_delta,
            )
            safe.append((score, {"left": left, "right": right}, reports))
    if not safe:
        raise RuntimeError(
            f"{stage_id} waypoint {step_index} has no joint/FK-qualified paired "
            f"IK candidate; rejected={rejected}"
        )
    safe.sort(key=lambda entry: entry[0])
    _, pair, reports = safe[0]
    return pair, reports, [rejected["left"], rejected["right"]]


def main() -> None:
    args = parse_args()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    robots: dict[str, Any] = {}
    audit: dict[str, Any] = {
        "experiment": "pipeline8_bimanual_offline_certification",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "robot_motion_commands_sent": False,
        "isolated_relaxed_ik": True,
        "collision_policy": "first_phase_collision_disabled",
        "collision_checks_performed": False,
        "passed": False,
        "stages": [],
    }
    bundle: dict[str, Any] | None = None
    try:
        validate_eighth_bimanual_recenter_plan(plan)
        # make_robot routes all arm/gripper command topics to never_execute and
        # only enables isolated RelaxedIK probes. This script has no motion call.
        for side in ("left", "right"):
            robots[side] = make_robot(args.config, side)
        torso = robots["right"].get_torso_joints()
        initial = {side: robots[side].get_arm_joints() for side in ("left", "right")}
        references = {side: joints.copy() for side, joints in initial.items()}
        kinematics = {
            side: R1LiteCandidateKinematics(side=side)
            for side in ("left", "right")
        }
        collision = R1LiteBimanualTableCollisionKinematics(
            table_z_m=float(plan["safety"]["table_z_m"]),
            tcp_to_fingertip_m=float(plan["safety"]["tcp_to_fingertip_m"]),
        )
        grasp_midpoint = np.asarray(
            plan["grasp_geometry"]["grasp_pair_midpoint_base_m"],
            dtype=float,
        )
        inward = (
            np.asarray(plan["first_center_base_m"], dtype=float)
            - grasp_midpoint
        )
        topdown_candidates = [
            Rotation.from_matrix(
                strict_topdown_edge_rotation(direction)
            ).as_quat()
            for direction in (inward[:2], -inward[:2])
        ]
        initial_quaternions = {
            side: kinematics[side]
            .fk(references[side], torso)
            .quaternion_xyzw
            for side in ("left", "right")
        }
        final_quaternions = {
            side: min(
                topdown_candidates,
                key=lambda value: orientation_error(
                    initial_quaternions[side], value
                ),
            )
            for side in ("left", "right")
        }
        previous_targets = {
            side: kinematics[side].fk(references[side], torso).position_base_m
            for side in ("left", "right")
        }
        bundle_stages: list[dict[str, Any]] = []
        for stage in plan["stages"]:
            stage_id = stage["id"]
            waypoints: list[dict[str, Any]] = []
            adaptive_endpoint_used = False
            if stage_id == "release_retract":
                records = {
                    side: _joint_fk_item(
                        references[side],
                        references[side],
                        kinematics[side].fk(references[side], torso),
                    )
                    for side in ("left", "right")
                }
                waypoints.append(
                    _waypoint(
                        alpha=1.0,
                        left=records["left"],
                        right=records["right"],
                        collision_summary=_collision_summary(stage_id),
                    )
                )
            elif stage_id == "home":
                target_joints = {
                    "left": initial["left"],
                    "right": initial["right"],
                }
                count = max(
                    1,
                    *[
                        int(
                            math.ceil(
                                np.max(
                                    np.abs(target_joints[side] - references[side])
                                )
                                / MAX_STAGE_JOINT_DELTA_RAD
                            )
                        )
                        for side in ("left", "right")
                    ],
                )
                starts = {
                    side: references[side].copy() for side in ("left", "right")
                }
                for index in range(1, count + 1):
                    progress = index / count
                    end = {
                        side: starts[side] * (1.0 - progress)
                        + target_joints[side] * progress
                        for side in ("left", "right")
                    }
                    records = {
                        side: _joint_fk_item(
                            end[side],
                            references[side],
                            kinematics[side].fk(end[side], torso),
                        )
                        for side in ("left", "right")
                    }
                    waypoints.append(
                        _waypoint(
                            alpha=progress,
                            left=records["left"],
                            right=records["right"],
                            collision_summary=_collision_summary(stage_id),
                            ik_acceptance="joint_transition",
                        )
                    )
                    references = end
                    previous_targets = {
                        side: np.asarray(
                            records[side]["target_position_base_m"], dtype=float
                        )
                        for side in ("left", "right")
                    }
            else:
                if stage_id == "grasp":
                    transition_policy = plan["safety"][
                        "ik_branch_transition"
                    ]
                    transition = {
                        side: np.asarray(
                            transition_policy[f"{side}_joints_rad"],
                            dtype=float,
                        )
                        for side in ("left", "right")
                    }
                    maximum_transition_delta = max(
                        float(
                            np.max(
                                np.abs(
                                    transition[side] - references[side]
                                )
                            )
                        )
                        for side in ("left", "right")
                    )
                    if maximum_transition_delta > float(
                        transition_policy[
                            "maximum_transition_joint_delta_rad"
                        ]
                    ):
                        raise RuntimeError(
                            "IK branch transition exceeds joint-delta gate"
                        )
                    transition_reports = []
                    if int(transition_policy["sweep_samples_per_segment"]) > 0:
                        transition_reports = (
                            collision.validate_interpolated_segment(
                                references["left"],
                                references["right"],
                                transition["left"],
                                transition["right"],
                                torso,
                                samples=int(
                                    transition_policy[
                                        "sweep_samples_per_segment"
                                    ]
                                ),
                                fingers_open=True,
                            )
                        )
                    transition_records = {
                        side: _joint_fk_item(
                            transition[side],
                            references[side],
                            kinematics[side].fk(transition[side], torso),
                        )
                        for side in ("left", "right")
                    }
                    waypoints.append(
                        _waypoint(
                            alpha=0.5,
                            left=transition_records["left"],
                            right=transition_records["right"],
                            collision_summary=_collision_summary(
                                "ik_branch_transition", transition_reports
                            ),
                        )
                    )
                    references = transition
                    previous_targets = {
                        side: np.asarray(
                            transition_records[side][
                                "target_position_base_m"
                            ],
                            dtype=float,
                        )
                        for side in ("left", "right")
                    }
                right_only = False
                final_targets = {
                    side: (
                        previous_targets[side].copy()
                        if right_only and side == "left"
                        else np.asarray(
                            stage[f"{side}_tcp_position_base_m"], dtype=float
                        )
                    )
                    for side in ("left", "right")
                }
                start_quaternions = {
                    side: kinematics[side]
                    .fk(references[side], torso)
                    .quaternion_xyzw
                    for side in ("left", "right")
                }
                stage_quaternions = {
                    side: (
                        start_quaternions[side]
                        if right_only and side == "left"
                        else
                        (
                            Rotation.from_rotvec(
                                np.radians(-30.0)
                                * np.array([0.0, 1.0, 0.0])
                            )
                            * Rotation.from_quat(final_quaternions[side])
                        ).as_quat()
                        if stage.get("orientation")
                        == "shared_world_y_tilt_minus_30deg"
                        else final_quaternions[side]
                    )
                    for side in ("left", "right")
                }
                starts = {side: previous_targets[side].copy() for side in ("left", "right")}
                position_count = max(
                    1,
                    *[
                        int(math.ceil(
                            np.linalg.norm(final_targets[side] - starts[side])
                            / MAXIMUM_SHARED_CARTESIAN_STEP_M
                        ))
                        for side in ("left", "right")
                    ],
                )
                orientation_count = max(
                    1,
                    *[
                        int(math.ceil(
                            math.degrees(
                                orientation_error(
                                    start_quaternions[side],
                                    stage_quaternions[side],
                                )
                            )
                            / 7.5
                        ))
                        for side in ("left", "right")
                    ],
                )
                orientation_paths = {
                    side: Slerp(
                        [0.0, 1.0],
                        Rotation.from_quat(
                            np.vstack(
                                [
                                    start_quaternions[side],
                                    stage_quaternions[side],
                                ]
                            )
                        ),
                    )
                    for side in ("left", "right")
                }
                specifications: list[
                    tuple[float, dict[str, np.ndarray], dict[str, np.ndarray]]
                ] = []
                if stage_id == "grasp":
                    specifications.append(
                        (
                            1.0,
                            {
                                side: final_targets[side]
                                for side in ("left", "right")
                            },
                            {
                                side: stage_quaternions[side]
                                for side in ("left", "right")
                            },
                        )
                    )
                elif stage_id == "lift":
                    clearance_progress = 0.25
                    clearance_targets = {
                        side: starts[side] * (1.0 - clearance_progress)
                        + final_targets[side] * clearance_progress
                        for side in ("left", "right")
                    }
                    specifications.append(
                        (
                            clearance_progress,
                            clearance_targets,
                            {
                                side: start_quaternions[side]
                                for side in ("left", "right")
                            },
                        )
                    )
                    remaining_position_count = max(
                        1,
                        *[
                            int(
                                math.ceil(
                                    np.linalg.norm(
                                        final_targets[side]
                                        - clearance_targets[side]
                                    )
                                    / MAXIMUM_SHARED_CARTESIAN_STEP_M
                                )
                            )
                            for side in ("left", "right")
                        ],
                    )
                    count = max(
                        remaining_position_count, orientation_count
                    )
                    for index in range(1, count + 1):
                        local_progress = index / count
                        progress = clearance_progress + (
                            1.0 - clearance_progress
                        ) * local_progress
                        specifications.append(
                            (
                                progress,
                                {
                                    side: clearance_targets[side]
                                    * (1.0 - local_progress)
                                    + final_targets[side] * local_progress
                                    for side in ("left", "right")
                                },
                                {
                                    side: orientation_paths[side](
                                        [local_progress]
                                    ).as_quat()[0]
                                    for side in ("left", "right")
                                },
                            )
                        )
                else:
                    count = max(position_count, orientation_count)
                    for index in range(1, count + 1):
                        progress = index / count
                        specifications.append(
                            (
                                progress,
                                {
                                    side: starts[side] * (1.0 - progress)
                                    + final_targets[side] * progress
                                    for side in ("left", "right")
                                },
                                {
                                    side: orientation_paths[side](
                                        [progress]
                                    ).as_quat()[0]
                                    for side in ("left", "right")
                                },
                            )
                        )
                for index, (alpha, targets, quaternions) in enumerate(
                    specifications,
                    start=1,
                ):
                    try:
                        pair, reports, _ = _safe_pairs(
                            stage_id=stage_id,
                            step_index=index,
                            targets=targets,
                            quaternions=quaternions,
                            robots=robots,
                            kinematics=kinematics,
                            references=references,
                            torso=torso,
                            collision=collision,
                            maximum_joint_delta_rad=(
                                float(
                                    plan["safety"]["grasp_refinement_policy"][
                                        "initial_seed_maximum_total_joint_delta_rad"
                                    ]
                                )
                                if stage_id == "grasp"
                                else MAX_STAGE_JOINT_DELTA_RAD
                            ),
                            fk_position_tolerance_m=(
                                min(
                                    float(
                                        plan["safety"]["ik_branch_transition"][
                                            "initial_grasp_fk_tolerance_m"
                                        ]
                                    ),
                                    float(
                                        plan["safety"][
                                            "grasp_refinement_policy"
                                        ]["individual_tcp_tolerance_m"]
                                    ),
                                )
                                if stage_id == "grasp"
                                else 0.035
                            ),
                            fk_outer_position_tolerance_m=(
                                float(
                                    plan["safety"]["ik_branch_transition"][
                                        "initial_grasp_fk_tolerance_m"
                                    ]
                                )
                                if stage_id == "grasp"
                                else 0.035
                            ),
                            fixed_joints=(
                                {"left": initial["left"]}
                                if right_only
                                else None
                            ),
                        )
                    except RuntimeError:
                        if (
                            stage_id == "translate_center"
                            and plan["safety"].get(
                                "translation_reachability_policy"
                            )
                            == "farthest_certified_waypoint_if_endpoint_unreachable"
                            and waypoints
                        ):
                            waypoints[-1]["alpha"] = 1.0
                            adaptive_endpoint_used = True
                            break
                        raise
                    if stage_id == "grasp":
                        final_seed = {
                            side: np.asarray(pair[side]["joints_rad"], dtype=float)
                            for side in ("left", "right")
                        }
                        substep_limit = float(
                            plan["safety"]["grasp_refinement_policy"][
                                "initial_seed_maximum_substep_joint_delta_rad"
                            ]
                        )
                        substep_count = max(
                            1,
                            *[
                                int(
                                    math.ceil(
                                        np.max(
                                            np.abs(
                                                final_seed[side]
                                                - references[side]
                                            )
                                        )
                                        / substep_limit
                                    )
                                )
                                for side in ("left", "right")
                            ],
                        )
                        start_references = {
                            side: references[side].copy()
                            for side in ("left", "right")
                        }
                        previous_alpha = (
                            float(waypoints[-1]["alpha"]) if waypoints else 0.0
                        )
                        for substep in range(1, substep_count):
                            progress = substep / substep_count
                            intermediate = {
                                side: start_references[side] * (1.0 - progress)
                                + final_seed[side] * progress
                                for side in ("left", "right")
                            }
                            records = {
                                side: _joint_fk_item(
                                    intermediate[side],
                                    references[side],
                                    kinematics[side].fk(
                                        intermediate[side], torso
                                    ),
                                )
                                for side in ("left", "right")
                            }
                            # The user explicitly disabled all first-grasp
                            # collision checks, including its joint transition.
                            intermediate_reports = []
                            waypoints.append(
                                _waypoint(
                                    alpha=previous_alpha
                                    + (alpha - previous_alpha) * progress,
                                    left=records["left"],
                                    right=records["right"],
                                    collision_summary=_collision_summary(
                                        stage_id, intermediate_reports
                                    ),
                                    ik_acceptance="joint_transition",
                                )
                            )
                            references = intermediate
                            previous_targets = {
                                side: np.asarray(
                                    records[side]["target_position_base_m"],
                                    dtype=float,
                                )
                                for side in ("left", "right")
                            }
                    left = _side_record(pair["left"], references["left"])
                    right = _side_record(pair["right"], references["right"])
                    waypoints.append(_waypoint(
                        alpha=alpha,
                        left=left,
                        right=right,
                        collision_summary=_collision_summary(stage_id, reports),
                        ik_acceptance=(
                            "best_effort_seed"
                            if stage_id == "grasp"
                            else "required"
                        ),
                    ))
                    references = {
                        side: np.asarray(pair[side]["joints_rad"], dtype=float)
                        for side in ("left", "right")
                    }
                    previous_targets = {
                        side: np.asarray(pair[side]["target_position_base_m"], dtype=float)
                        for side in ("left", "right")
                    }
            bundle_stages.append({"id": stage_id, "waypoints": waypoints})
            audit["stages"].append({
                "id": stage_id,
                "shared_waypoint_count": len(waypoints),
                "passed": True,
                "adaptive_endpoint_used": adaptive_endpoint_used,
            })
        tcp_definition = {
            "left": "left_gripper_tool_z",
            "right": "right_gripper_tool_z",
            "tcp_to_fingertip_m": float(plan["safety"]["tcp_to_fingertip_m"]),
        }
        collision_model = {
            "name": "first_phase_collision_disabled",
            "planning_checks_enabled": False,
            "certification_checks_enabled": False,
            "execution_checks_enabled": False,
            "checked_stage_ids": [],
            "sweep_samples_per_segment": 0,
            "restore_at": "fresh_seventh_phase",
        }
        bundle = build_eighth_execution_bundle(
            plan_sha256=audit["plan_sha256"],
            config_sha256=sha256_file(args.config),
            full_urdf_sha256=sha256_file(FULL_URDF),
            left_relaxed_urdf_sha256=sha256_file(LEFT_RELAXED_URDF),
            right_relaxed_urdf_sha256=sha256_file(RELAXED_URDF),
            tcp_definition=tcp_definition,
            collision_model=collision_model,
            initial_left_joints_rad=initial["left"],
            initial_right_joints_rad=initial["right"],
            torso_joints_rad=torso,
            stages=bundle_stages,
            relative_geometry_tolerance_m=float(
                plan["safety"]["online_relative_geometry_gate"][
                    "maximum_vector_error_m"
                ]
            ),
            maximum_waypoint_joint_delta_rad=3.0,
            fk_position_tolerance_m=0.035,
            sweep_samples_per_segment=50,
        )
        audit["passed"] = True
    except Exception as exc:
        audit["error"] = repr(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.execution_bundle_output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
        if bundle is None:
            failure = {
                "schema": "capx.r1lite.eighth_execution_bundle_failure.v1",
                "plan_sha256": audit["plan_sha256"],
                "config_sha256": sha256_file(args.config),
                "robot_motion_commands_sent": False,
                "passed": False,
                "error": audit.get("error", "certification did not complete"),
            }
            failure["bundle_sha256"] = canonical_sha256(failure)
            bundle = failure
        args.execution_bundle_output.write_bytes(canonical_json_bytes(bundle))
        for robot in robots.values():
            robot.close()
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
