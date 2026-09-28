"""Deterministic dual-arm towel plan compiled from validated perception."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from capx.integrations.r1lite.bimanual_kinematics import (
    R1LiteTableCollisionKinematics,
)


STAGE_ORDER = ("pregrasp", "grasp", "lift", "fold_center")
SINGLE_ARM_STAGE_ORDER = (
    "pregrasp",
    "align_grasp",
    "grasp",
    "lift",
    "place_center",
    "release_retract",
    "home",
)
SEQUENTIAL_SINGLE_ARM_SCHEMA = "capx.r1lite.seventh_sequential_single_arm.v34"
EIGHTH_BIMANUAL_RECENTER_SCHEMA = "capx.r1lite.eighth_bimanual_recenter.v33"
EIGHTH_BIMANUAL_STAGE_ORDER = (
    "grasp",
    "lift",
    "pullback",
    "translate_center",
    "release_retract",
    "home",
)
MINIMUM_GRIPPER_CENTER_SEPARATION_WITH_MARGIN_M = 0.12552056969962836


def _point(geometry: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(
        geometry["keypoints"][name]["base_xyz_m"],
        dtype=np.float64,
    ).reshape(3)


def build_towel_fold_plan(
    geometry: dict[str, Any],
    *,
    calibration_sha256: str,
    first_image_sha256: str,
    second_image_sha256: str,
    keypoint_repeatability_m: float,
    table_z_m: float,
    final_fingertip_center_separation_m: float = 0.020,
    tcp_to_fingertip_m: float = 0.078,
    towel_thickness_m: float = 0.003,
    pregrasp_clearance_m: float = 0.10,
    lift_height_m: float = 0.12,
    fold_step_m: float = 0.02,
    fold_fingertip_clearance_m: float = 0.02,
    nominal_outward_tilt_deg: float = 45.0,
    refinement_target_tolerance_m: float = 0.015,
    refinement_max_iterations: int = 20,
    refinement_ik_probes_per_iteration: int = 3,
    refinement_minimum_improvement_m: float = 0.001,
    refinement_local_optimum_stall_iterations: int = 2,
) -> dict[str, Any]:
    """Build the exact four-stage motion requested by the user."""
    if abs(float(final_fingertip_center_separation_m) - 0.020) > 1e-9:
        raise ValueError("seventh pipeline final fingertip separation is fixed at 2 cm")
    center = _point(geometry, "center")
    left_grasp = _point(geometry, "left_edge_midpoint")
    right_grasp = _point(geometry, "right_edge_midpoint")
    lateral = right_grasp - left_grasp
    lateral[2] = 0.0
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm < final_fingertip_center_separation_m:
        raise ValueError("towel is narrower than the requested final 2 cm separation")
    lateral /= lateral_norm

    grasp_fingertip_z = float(table_z_m) + float(towel_thickness_m)
    grasp_tcp_z = grasp_fingertip_z + float(tcp_to_fingertip_m)
    left_grasp_fingertip = left_grasp.copy()
    right_grasp_fingertip = right_grasp.copy()
    left_grasp_fingertip[2] = grasp_fingertip_z
    right_grasp_fingertip[2] = grasp_fingertip_z
    left_grasp[2] = grasp_tcp_z
    right_grasp[2] = grasp_tcp_z
    left_pregrasp = left_grasp + np.array([0.0, 0.0, pregrasp_clearance_m])
    right_pregrasp = right_grasp + np.array([0.0, 0.0, pregrasp_clearance_m])
    left_lift = left_grasp + np.array([0.0, 0.0, lift_height_m])
    right_lift = right_grasp + np.array([0.0, 0.0, lift_height_m])
    half_gap = float(final_fingertip_center_separation_m) / 2.0
    if float(fold_fingertip_clearance_m) <= 0.0:
        raise ValueError("fold fingertip clearance must be positive")
    fingertip_fold_z = float(table_z_m) + float(fold_fingertip_clearance_m)
    left_fold_fingertip = center.copy()
    right_fold_fingertip = center.copy()
    left_fold_fingertip[:2] = center[:2] - lateral[:2] * half_gap
    right_fold_fingertip[:2] = center[:2] + lateral[:2] * half_gap
    left_fold_fingertip[2] = fingertip_fold_z
    right_fold_fingertip[2] = fingertip_fold_z
    tilt_rad = np.deg2rad(float(nominal_outward_tilt_deg))
    horizontal_tcp_offset = float(tcp_to_fingertip_m) * np.sin(tilt_rad)
    vertical_tcp_offset = float(tcp_to_fingertip_m) * np.cos(tilt_rad)
    left_fold = left_fold_fingertip.copy()
    right_fold = right_fold_fingertip.copy()
    left_fold[:2] -= lateral[:2] * horizontal_tcp_offset
    right_fold[:2] += lateral[:2] * horizontal_tcp_offset
    left_fold[2] += vertical_tcp_offset
    right_fold[2] += vertical_tcp_offset
    nominal_fold_tcp_separation = float(
        np.linalg.norm(left_fold - right_fold)
    )
    minimum_tilt_deg = float(
        np.rad2deg(
            np.arcsin(
                np.clip(
                    (
                        MINIMUM_GRIPPER_CENTER_SEPARATION_WITH_MARGIN_M
                        - final_fingertip_center_separation_m
                    )
                    / (2.0 * tcp_to_fingertip_m),
                    -1.0,
                    1.0,
                )
            )
        )
    )

    def stage(
        stage_id: str,
        left: np.ndarray,
        right: np.ndarray,
        *,
        speed: str,
    ) -> dict[str, Any]:
        return {
            "id": stage_id,
            "left_tcp_position_base_m": left.tolist(),
            "right_tcp_position_base_m": right.tolist(),
            "orientation": "strict_topdown",
            "synchronization": "shared_progress_barrier",
            "speed": speed,
        }

    stages = [
        {
            **stage("pregrasp", left_pregrasp, right_pregrasp, speed="slow"),
            "orientation": "topdown_or_multi_direction_tilt",
            "tilt_search_deg": [0.0, 15.0, 30.0, 45.0],
            "left_fingertip_position_base_m": (
                left_grasp_fingertip + np.array([0.0, 0.0, pregrasp_clearance_m])
            ).tolist(),
            "right_fingertip_position_base_m": (
                right_grasp_fingertip + np.array([0.0, 0.0, pregrasp_clearance_m])
            ).tolist(),
        },
        {
            **stage("grasp", left_grasp, right_grasp, speed="very_slow"),
            "orientation": "topdown_or_multi_direction_tilt",
            "tilt_search_deg": [0.0, 15.0, 30.0, 45.0],
            "left_fingertip_position_base_m": left_grasp_fingertip.tolist(),
            "right_fingertip_position_base_m": right_grasp_fingertip.tolist(),
        },
        {
            **stage("lift", left_lift, right_lift, speed="very_slow"),
            "orientation": "topdown_or_multi_direction_tilt",
            "tilt_search_deg": [0.0, 15.0, 30.0, 45.0],
            "left_fingertip_position_base_m": (
                left_grasp_fingertip + np.array([0.0, 0.0, lift_height_m])
            ).tolist(),
            "right_fingertip_position_base_m": (
                right_grasp_fingertip + np.array([0.0, 0.0, lift_height_m])
            ).tolist(),
        },
        {
            **stage("fold_center", left_fold, right_fold, speed="very_slow"),
            "orientation": "mirrored_outward_tilt",
            "nominal_outward_tilt_deg": float(nominal_outward_tilt_deg),
            "grasp_tilt_search_deg": [0.0, 15.0, 30.0, 45.0],
            "tilt_search_deg": [45.0, 50.0, 55.0, 60.0],
            "left_fingertip_position_base_m": left_fold_fingertip.tolist(),
            "right_fingertip_position_base_m": right_fold_fingertip.tolist(),
        },
    ]
    plan = {
        "schema": "capx.r1lite.seventh_towel_fold.v1",
        "pipeline": 7,
        "task": "dual_arm_towel_center_fold",
        "tcp": {
            "left": "left_gripper_tool_z",
            "right": "right_gripper_tool_z",
            "left_definition": {
                "source_link": "left_gripper_link",
                "xyz_m": [0.0, 0.0, 0.0],
                "rpy_rad": [0.0, float(np.pi / 2.0), 0.0],
                "tool_positive_z": "gripper_physical_forward",
                "geometry_policy": "identical_to_right_and_mirrored",
            },
        },
        "perception": {
            "method": "head_left_sam3_towel_mask_plus_table_homography",
            "edge_midpoint_method": geometry.get("edge_midpoint_method"),
            "contour_projection_distance_px": geometry.get(
                "contour_projection_distance_px"
            ),
            "calibration_sha256": calibration_sha256,
            "first_image_sha256": first_image_sha256,
            "second_image_sha256": second_image_sha256,
            "keypoint_repeatability_m": float(keypoint_repeatability_m),
            "maximum_keypoint_extrapolation_m": float(
                geometry["maximum_extrapolation_m"]
            ),
            "mask_must_be_inside_calibration_polygon": False,
            "keypoints": {
                name: {
                    key: value
                    for key, value in item.items()
                    if key != "selected_mask"
                }
                for name, item in geometry["keypoints"].items()
            },
        },
        "assignment": {
            "left_arm": "camera_left_edge_midpoint",
            "right_arm": "camera_right_edge_midpoint",
        },
        "stages": stages,
        "gripper_events": [
            {"after": "grasp", "action": "close_both_grippers"},
            {"after": "fold_center", "action": "open_both_grippers"},
        ],
        "safety": {
            "final_fingertip_center_separation_m": float(
                final_fingertip_center_separation_m
            ),
            "nominal_fold_tcp_center_separation_m": nominal_fold_tcp_separation,
            "minimum_outward_tilt_deg": minimum_tilt_deg,
            "nominal_outward_tilt_deg": float(nominal_outward_tilt_deg),
            "minimum_gripper_center_separation_with_margin_m": (
                MINIMUM_GRIPPER_CENTER_SEPARATION_WITH_MARGIN_M
            ),
            "static_gripper_envelope_gate_passed": (
                nominal_fold_tcp_separation
                >= MINIMUM_GRIPPER_CENTER_SEPARATION_WITH_MARGIN_M
            ),
            "static_gripper_envelope_gate_reason": (
                "2cm fingertip spacing uses mirrored outward tilt so the "
                "gripper TCPs remain outside the 12.552cm mesh-envelope minimum"
            ),
            "tcp_to_fingertip_m": float(tcp_to_fingertip_m),
            "towel_thickness_m": float(towel_thickness_m),
            "pregrasp_clearance_m": float(pregrasp_clearance_m),
            "lift_height_m": float(lift_height_m),
            "fold_fingertip_clearance_above_table_m": float(
                fold_fingertip_clearance_m
            ),
            "maximum_fold_cartesian_step_m": float(fold_step_m),
            "cartesian_refinement_policy": {
                "source": "pipeline6_vertex_refinement",
                "applies_to": "both_arms_all_four_stages",
                "feedback_source": "physical_tool_z_pose_after_each_motion",
                "target_tolerance_m": float(refinement_target_tolerance_m),
                "max_iterations": int(refinement_max_iterations),
                "ik_probes_per_iteration": int(
                    refinement_ik_probes_per_iteration
                ),
                "minimum_improvement_m": float(
                    refinement_minimum_improvement_m
                ),
                "local_optimum_stall_iterations": int(
                    refinement_local_optimum_stall_iterations
                ),
                "max_incremental_joint_delta_rad": 1.0,
                "joint_arrival_tolerance_rad": 0.01,
                "joint_arrival_max_retries": 2,
                "joint_hard_safety_limit_rad": 0.05,
                "final_arrival_policy": (
                    "paired_cartesian_pose_after_joint_retries"
                ),
                "paired_swept_collision_gate_each_iteration": True,
                "shared_progress_barrier_each_iteration": True,
            },
            "dual_arm_collision_gate_required": True,
            "left_tcp_model": "constructed_from_right_symmetric_geometry",
            "left_tcp_physical_pivot_calibration_required": False,
            "execution_default": "preview_only",
        },
    }
    validate_towel_fold_plan(plan)
    return plan


def validate_towel_fold_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") == SEQUENTIAL_SINGLE_ARM_SCHEMA:
        validate_sequential_single_arm_plan(plan)
        return
    if plan.get("schema") != "capx.r1lite.seventh_towel_fold.v1":
        raise ValueError("unexpected seventh-pipeline plan schema")
    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise ValueError("seventh-pipeline plan stages must be a list")
    if tuple(str(item.get("id")) for item in stages) != STAGE_ORDER:
        raise ValueError("seventh-pipeline stage order is immutable")
    for item in stages:
        for side in ("left", "right"):
            point = np.asarray(
                item[f"{side}_tcp_position_base_m"],
                dtype=np.float64,
            ).reshape(3)
            if not np.isfinite(point).all():
                raise ValueError(f"{item['id']} {side} target is non-finite")
        expected_orientation = (
            "mirrored_outward_tilt"
            if item["id"] == "fold_center"
            else "topdown_or_multi_direction_tilt"
        )
        if item.get("orientation") != expected_orientation:
            raise ValueError(
                f"{item['id']} has unexpected orientation policy"
            )
        if item.get("synchronization") != "shared_progress_barrier":
            raise ValueError("all seventh-pipeline stages must be synchronized")
    fold = stages[-1]
    separation = float(
        np.linalg.norm(
            np.asarray(fold["left_fingertip_position_base_m"])
            - np.asarray(fold["right_fingertip_position_base_m"])
        )
    )
    expected = float(plan["safety"]["final_fingertip_center_separation_m"])
    if abs(separation - expected) > 1e-6:
        raise ValueError(
            "fold-center fingertip separation changed: "
            f"expected={expected}, actual={separation}"
        )
    if not bool(plan["safety"]["static_gripper_envelope_gate_passed"]):
        raise ValueError("fold-center tilted TCP separation fails mesh envelope gate")
    refinement = plan["safety"].get("cartesian_refinement_policy", {})
    required_refinement = {
        "applies_to": "both_arms_all_four_stages",
        "feedback_source": "physical_tool_z_pose_after_each_motion",
        "target_tolerance_m": 0.015,
        "max_iterations": 20,
        "ik_probes_per_iteration": 3,
        "minimum_improvement_m": 0.001,
        "local_optimum_stall_iterations": 2,
        "max_incremental_joint_delta_rad": 1.0,
        "joint_arrival_tolerance_rad": 0.01,
        "joint_arrival_max_retries": 2,
        "joint_hard_safety_limit_rad": 0.05,
        "paired_swept_collision_gate_each_iteration": True,
        "shared_progress_barrier_each_iteration": True,
    }
    for key, expected_value in required_refinement.items():
        if refinement.get(key) != expected_value:
            raise ValueError(
                f"unexpected pipeline6-style refinement policy for {key}"
            )


def build_sequential_single_arm_plan(
    geometry: dict[str, Any],
    *,
    side: str,
    original_center_base_m: list[float],
    calibration_sha256: str,
    first_image_sha256: str,
    second_image_sha256: str,
    keypoint_repeatability_m: float,
    table_z_m: float,
    tcp_to_fingertip_m: float = 0.078,
    towel_thickness_m: float = 0.00002,
    open_finger_table_margin_m: float = 0.001,
    pregrasp_clearance_m: float = 0.10,
    grasp_alignment_clearance_m: float = 0.05,
    lift_height_m: float = 0.05,
    place_clearance_m: float = 0.05,
    grasp_candidate: dict[str, Any] | None = None,
    selected_grasp_tilt_deg: float | None = None,
) -> dict[str, Any]:
    """Compile one immutable phase of the right-then-left towel fold."""
    if side not in ("left", "right"):
        raise ValueError("single-arm towel phase side must be left or right")
    center = np.asarray(original_center_base_m, dtype=float).reshape(3)
    if not np.isfinite(center).all():
        raise ValueError("original center must be finite")
    keypoint_name = f"{side}_edge_midpoint"
    candidates = [
        dict(item) for item in geometry.get("right_side_grasp_candidates", [])
    ]
    if not candidates:
        fallback = geometry["keypoints"][keypoint_name]
        fallback_contour = np.asarray(fallback["base_xyz_m"], dtype=float).reshape(3)
        fallback_inward = _point(geometry, "center")[:2] - fallback_contour[:2]
        fallback_inward_norm = float(np.linalg.norm(fallback_inward))
        if fallback_inward_norm <= 1e-6:
            raise ValueError("fallback edge point and towel center are degenerate")
        fallback_inward /= fallback_inward_norm
        fallback_contact = fallback_contour.copy()
        fallback_contact[:2] += fallback_inward * 0.005
        candidates = [
            {
                "id": f"{side}_edge_midpoint_fallback",
                "pixel_uv": list(fallback.get("pixel_uv", [0.0, 0.0])),
                "base_xyz_m": fallback_contact.tolist(),
                "contour_base_xyz_m": fallback_contour.tolist(),
                "inward_normal_base_xy": fallback_inward.tolist(),
                "contact_center_inset_m": 0.005,
                "arc_distance_from_first_corner_m": 0.0,
                "extrapolation_m": float(fallback.get("extrapolation_m", 0.0)),
                "two_frame_repeatability_m": float(keypoint_repeatability_m),
                "matched_first_candidate_id": "fallback",
            }
        ]
    selected_candidate = dict(grasp_candidate or candidates[len(candidates) // 2])
    if selected_candidate.get("id") not in {item.get("id") for item in candidates}:
        raise ValueError("selected grasp candidate is not in the right-side contour set")
    grasp_fingertip = np.asarray(
        selected_candidate["base_xyz_m"], dtype=float
    ).reshape(3)
    nominal_grasp_fingertip_z = (
        float(table_z_m)
        + float(towel_thickness_m)
        + float(open_finger_table_margin_m)
    )
    grasp_fingertip[2] = nominal_grasp_fingertip_z
    detected_center = _point(geometry, "center")
    inward_normal_xy = detected_center[:2] - grasp_fingertip[:2]
    inward_norm = float(np.linalg.norm(inward_normal_xy))
    if inward_norm <= 1e-6:
        raise ValueError("detected towel center and grasp edge midpoint are degenerate")
    inward_normal_xy /= inward_norm
    place_fingertip = center.copy()
    place_fingertip[2] = float(table_z_m) + float(place_clearance_m)
    grasp_tilt_rad = math.radians(float(selected_grasp_tilt_deg or 0.0))
    strict_tool_z = np.array([0.0, 0.0, -1.0])
    strict_tool_y = np.array(
        [inward_normal_xy[0], inward_normal_xy[1], 0.0]
    )
    strict_tool_x = np.cross(
        strict_tool_y,
        strict_tool_z,
    )
    strict_rotation_base_tool = np.column_stack(
        [strict_tool_x, strict_tool_y, strict_tool_z]
    )
    local_tilt_rotation = np.array(
        [
            [math.cos(grasp_tilt_rad), 0.0, math.sin(grasp_tilt_rad)],
            [0.0, 1.0, 0.0],
            [-math.sin(grasp_tilt_rad), 0.0, math.cos(grasp_tilt_rad)],
        ]
    )
    tilted_rotation_base_tool = (
        strict_rotation_base_tool @ local_tilt_rotation
    )
    tilted_tool_z = (
        tilted_rotation_base_tool[:, 2]
    )
    topdown_lowest_offset_z = (
        R1LiteTableCollisionKinematics.finger_minimum_z_offset_from_tcp(
            strict_rotation_base_tool,
            fingers_open=True,
        )
    )
    tilted_lowest_offset_z = (
        R1LiteTableCollisionKinematics.finger_minimum_z_offset_from_tcp(
            tilted_rotation_base_tool,
            fingers_open=True,
        )
    )
    tilted_finger_offsets_z = (
        R1LiteTableCollisionKinematics.finger_minimum_z_offsets_from_tcp(
            tilted_rotation_base_tool,
            fingers_open=True,
        )
    )
    designated_contact_link = (
        "finger_link1"
        if tilted_finger_offsets_z[0] <= tilted_finger_offsets_z[1]
        else "finger_link2"
    )
    reference_lowest_point_z = (
        nominal_grasp_fingertip_z
        - strict_tool_z[2] * float(tcp_to_fingertip_m)
        + topdown_lowest_offset_z
    )
    compensated_fingertip_z = (
        reference_lowest_point_z
        + tilted_tool_z[2] * float(tcp_to_fingertip_m)
        - tilted_lowest_offset_z
    )
    fingertip_height_compensation_m = (
        compensated_fingertip_z - nominal_grasp_fingertip_z
    )
    grasp_fingertip[2] = compensated_fingertip_z
    grasp_tcp = (
        grasp_fingertip
        - tilted_tool_z * float(tcp_to_fingertip_m)
    )
    pregrasp_tcp = grasp_tcp + np.array([0.0, 0.0, pregrasp_clearance_m])
    align_grasp_fingertip = (
        grasp_fingertip - tilted_tool_z * grasp_alignment_clearance_m
    )
    align_grasp_tcp = (
        align_grasp_fingertip
        - tilted_tool_z * float(tcp_to_fingertip_m)
    )
    tool_offset = np.array([0.0, 0.0, float(tcp_to_fingertip_m)])
    lift_tcp = grasp_tcp + np.array([0.0, 0.0, lift_height_m])
    place_tcp = place_fingertip + tool_offset
    retract_tcp = place_tcp + np.array([0.0, 0.0, pregrasp_clearance_m])

    def motion_stage(
        stage_id: str,
        tcp: np.ndarray,
        fingertip: np.ndarray,
        *,
        speed: str,
        strict_grasp_alignment: bool = False,
    ) -> dict[str, Any]:
        stage = {
            "id": stage_id,
            "active_side": side,
            "active_tcp_position_base_m": tcp.tolist(),
            "active_fingertip_position_base_m": fingertip.tolist(),
            "inactive_arm_policy": "hold_session_initial_joints",
            "orientation": (
                "edge_normal_constrained_tilt"
                if strict_grasp_alignment
                else "topdown_or_multi_direction_tilt"
            ),
            "tilt_search_deg": (
                (
                    [float(selected_grasp_tilt_deg)]
                    if selected_grasp_tilt_deg is not None
                    else [0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0]
                )
                if strict_grasp_alignment
                else [0.0, 15.0, 30.0, 45.0, 60.0]
            ),
            "speed": speed,
        }
        if strict_grasp_alignment:
            stage.update(
                {
                    "jaw_closing_axis_tool": "tool_y",
                    "jaw_closing_axis_base_xy": inward_normal_xy.tolist(),
                    "jaw_closing_axis_max_vertical_deg": 3.0,
                    "maximum_tool_z_tilt_from_down_deg": 45.0,
                }
            )
        return stage

    stages = [
        motion_stage(
            "pregrasp",
            pregrasp_tcp,
            grasp_fingertip + np.array([0.0, 0.0, pregrasp_clearance_m]),
            speed="slow",
        ),
        motion_stage(
            "align_grasp",
            align_grasp_tcp,
            align_grasp_fingertip,
            speed="very_slow",
            strict_grasp_alignment=True,
        ),
        motion_stage(
            "grasp",
            grasp_tcp,
            grasp_fingertip,
            speed="very_slow",
            strict_grasp_alignment=True,
        ),
        motion_stage(
            "lift",
            lift_tcp,
            grasp_fingertip + np.array([0.0, 0.0, lift_height_m]),
            speed="very_slow",
        ),
        motion_stage(
            "place_center",
            place_tcp,
            place_fingertip,
            speed="very_slow",
        ),
        motion_stage(
            "release_retract",
            retract_tcp,
            place_fingertip + np.array([0.0, 0.0, pregrasp_clearance_m]),
            speed="very_slow",
        ),
        {
            "id": "home",
            "active_side": side,
            "inactive_arm_policy": "hold_session_initial_joints",
            "target": "session_initial_joint_state",
            "speed": "slow",
        },
    ]
    plan = {
        "schema": SEQUENTIAL_SINGLE_ARM_SCHEMA,
        "pipeline": 7,
        "task": "sequential_single_arm_towel_center_fold",
        "phase_side": side,
        "original_center_base_m": center.tolist(),
        "tcp": {
            "left": "left_gripper_tool_z",
            "right": "right_gripper_tool_z",
        },
        "perception": {
            "method": "head_left_sam3_towel_mask_plus_table_homography",
            "edge_midpoint_method": geometry.get("edge_midpoint_method"),
            "contour_projection_distance_px": geometry.get(
                "contour_projection_distance_px"
            ),
            "calibration_sha256": calibration_sha256,
            "first_image_sha256": first_image_sha256,
            "second_image_sha256": second_image_sha256,
            "keypoint_repeatability_m": float(keypoint_repeatability_m),
            "selected_keypoint": selected_candidate["id"],
            "selected_keypoint_base_m": np.asarray(
                selected_candidate["base_xyz_m"], dtype=float
            ).tolist(),
            "detected_center_base_m": detected_center.tolist(),
            "right_side_contour_policy": dict(
                geometry.get(
                    "right_side_contour_policy",
                    {
                        "definition": (
                            "fitted_right_side_contour_with_inset_finger_contact_center"
                        ),
                        "corner_exclusion_m": 0.01,
                        "sample_spacing_m": 0.01,
                        "maximum_candidates": 24,
                        "finger_contact_width_m": 0.01,
                        "contact_center_inset_m": 0.005,
                    },
                )
            ),
            "right_side_grasp_candidates": candidates,
        },
        "grasp_geometry": {
            "edge_inward_normal_base_xy": inward_normal_xy.tolist(),
            "jaw_closing_axis_tool": "tool_y",
            "orientation_policy": "edge_normal_constrained_tilt_about_tool_y",
            "tilt_candidates_deg": [
                0.0,
                -15.0,
                15.0,
                -30.0,
                30.0,
                -45.0,
                45.0,
            ],
            "selected_tilt_deg": (
                None
                if selected_grasp_tilt_deg is None
                else float(selected_grasp_tilt_deg)
            ),
            "jaw_closing_axis_max_vertical_deg": 3.0,
            "maximum_tool_z_tilt_from_down_deg": 45.0,
            "contact_reference": "lowest_open_finger_mesh_point",
            "reference_topdown_lowest_point_z_m": float(
                reference_lowest_point_z
            ),
            "selected_fingertip_center_height_compensation_m": float(
                fingertip_height_compensation_m
            ),
            "alignment_clearance_m": float(grasp_alignment_clearance_m),
            "selected_candidate_id": selected_candidate["id"],
        },
        "stages": stages,
        "gripper_events": [
            {"before": "pregrasp", "side": side, "action": "open_gripper"},
            {"after": "grasp", "side": side, "action": "close_gripper"},
            {"after": "place_center", "side": side, "action": "open_gripper"},
        ],
        "safety": {
            "tcp_to_fingertip_m": float(tcp_to_fingertip_m),
            "table_z_m": float(table_z_m),
            "towel_thickness_m": float(towel_thickness_m),
            "place_release_clearance_m": float(place_clearance_m),
            "lift_height_m": float(lift_height_m),
            "inactive_arm_policy": "hold_session_initial_joints",
            "original_center_is_immutable": True,
            "home_joint_tolerance_rad": 0.01,
            "collision_policy": "active_arm_table_only",
            "post_grasp_closed_lift_collision_policy": (
                "skip_table_collision_if_gripper_close_succeeded"
            ),
            "post_grasp_close_dwell_s": 1.0,
            "minimum_table_clearance_m": 0.0,
            "distal_link_capsule_radius_m": 0.025,
            "gripper_table_collision_model": "two_fully_open_finger_mesh_aabbs",
            "gripper_finger_open_travel_m": 0.05,
            "grasp_open_finger_table_margin_m": float(
                open_finger_table_margin_m
            ),
            "grasp_allowed_table_contact_link": designated_contact_link,
            "grasp_designated_contact_min_clearance_m": -0.0002,
            "grasp_noncontact_min_clearance_m": 0.001,
            "grasp_contact_stage_only": True,
            "dual_arm_collision_gate_required": False,
            "cartesian_refinement_policy": {
                "source": "pipeline6_vertex_refinement",
                "applies_to": "active_arm_motion_stages",
                "feedback_source": "physical_tool_z_pose_after_each_motion",
                "target_tolerance_m": 0.015,
                "grasp_xy_tolerance_m": 0.010,
                "grasp_z_tolerance_m": 0.0005,
                "max_iterations": 20,
                "ik_probes_per_iteration": 3,
                "minimum_improvement_m": 0.001,
                "local_optimum_stall_iterations": 2,
                "max_incremental_joint_delta_rad": 1.0,
                "joint_arrival_tolerance_rad": 0.01,
                "joint_arrival_max_retries": 2,
                "joint_hard_safety_limit_rad": 0.05,
                "joint_command_progress_steps": 1,
                "paired_swept_collision_gate_each_iteration": True,
            },
            "grasp_effort_contact_policy": {
                "enabled": True,
                "feedback_source": "hdas_joint_state_effort",
                "reference_capture": (
                    "after_unsegmented_initial_10mm_descent"
                ),
                "joint_indices_zero_based": [1, 2],
                "sum_abs_residual_threshold_nm": 2.5,
                "sample_count": 40,
                "motion_monitor_sample_count": 5,
                "maximum_median_abs_velocity_rad_s": 0.05,
                "require_settled_velocity_during_motion": False,
                "maximum_tcp_z_above_target_m": 0.005,
                "arrival_logic": (
                    "continuous_fixed_baseline_effort_during_post_arrival_probe"
                ),
                "calibration_id": "right_grasp_table_20260817_fixed_baseline",
            },
            "grasp_post_arrival_probe_policy": {
                "enabled": True,
                "start_condition": (
                    "grasp_xy_within_10mm_and_z_error_below_5mm"
                ),
                "maximum_entry_z_error_m": 0.005,
                "direction_base": [0.0, 0.0, -1.0],
                "initial_descent_m": 0.010,
                "absolute_height_refinement_iterations": 20,
                "absolute_contact_tcp_z_base_m": float(
                    grasp_tcp[2] - (reference_lowest_point_z - float(table_z_m))
                ),
                "absolute_contact_z_tolerance_m": 0.0005,
                "skip_collision_during_initial_descent": True,
                "check_collision_during_absolute_height_refinement": True,
                "forbid_upward_tcp_motion": True,
                "maximum_xy_shift_per_descent_m": 0.010,
                "stop_condition": (
                    "stable_effort_contact_or_absolute_height_refinement"
                    "_or_descent_condition_unavailable"
                ),
                "require_stable_contact_before_close": False,
                "close_after_probe_commands_without_contact": True,
                "close_if_descent_condition_unavailable": True,
                "designated_contact_min_clearance_m": -0.0152,
                "noncontact_min_clearance_m": 0.001,
            },
        },
    }
    validate_sequential_single_arm_plan(plan)
    return plan


def rebuild_sequential_single_arm_plan_for_candidate(
    plan: dict[str, Any],
    *,
    candidate_id: str,
    selected_grasp_tilt_deg: float | None = None,
) -> dict[str, Any]:
    """Recompile every grasp-dependent stage for one immutable contour point."""
    validate_sequential_single_arm_plan(plan)
    candidates = plan["perception"]["right_side_grasp_candidates"]
    selected = next(
        (dict(item) for item in candidates if item.get("id") == candidate_id),
        None,
    )
    if selected is None:
        raise ValueError("unknown right-side grasp candidate")
    perception = plan["perception"]
    selected_plan = build_sequential_single_arm_plan(
        {
            "edge_midpoint_method": perception["edge_midpoint_method"],
            "right_side_contour_policy": perception["right_side_contour_policy"],
            "right_side_grasp_candidates": candidates,
            "keypoints": {
                "center": {
                    "base_xyz_m": perception["detected_center_base_m"],
                },
                "right_edge_midpoint": {
                    "base_xyz_m": perception["selected_keypoint_base_m"],
                },
            },
        },
        side=str(plan["phase_side"]),
        original_center_base_m=list(plan["original_center_base_m"]),
        calibration_sha256=str(perception["calibration_sha256"]),
        first_image_sha256=str(perception["first_image_sha256"]),
        second_image_sha256=str(perception["second_image_sha256"]),
        keypoint_repeatability_m=float(perception["keypoint_repeatability_m"]),
        table_z_m=float(plan["safety"]["table_z_m"]),
        tcp_to_fingertip_m=float(plan["safety"]["tcp_to_fingertip_m"]),
        towel_thickness_m=float(plan["safety"]["towel_thickness_m"]),
        open_finger_table_margin_m=float(
            plan["safety"]["grasp_open_finger_table_margin_m"]
        ),
        grasp_alignment_clearance_m=float(
            plan["grasp_geometry"]["alignment_clearance_m"]
        ),
        place_clearance_m=float(plan["safety"]["place_release_clearance_m"]),
        grasp_candidate=selected,
        selected_grasp_tilt_deg=selected_grasp_tilt_deg,
    )
    for optional in ("sam3",):
        if optional in perception:
            selected_plan["perception"][optional] = dict(perception[optional])
    selected_plan["candidate_selection"] = {
        "source_plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "selected_candidate_id": candidate_id,
        "selected_tilt_deg": selected_grasp_tilt_deg,
        "selection_status": "selected_before_full_certification",
    }
    selected_plan["execution_readiness"] = {
        "ready": False,
        "reason": "selected candidate still requires full certification",
        "left_tcp_model_accepted": True,
        "robot_motion_commands_sent": False,
    }
    validate_sequential_single_arm_plan(selected_plan)
    return selected_plan


def validate_sequential_single_arm_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != SEQUENTIAL_SINGLE_ARM_SCHEMA:
        raise ValueError("unexpected sequential single-arm plan schema")
    side = plan.get("phase_side")
    if side not in ("left", "right"):
        raise ValueError("sequential phase side must be left or right")
    stages = plan.get("stages")
    if not isinstance(stages, list) or tuple(
        item.get("id") for item in stages
    ) != SINGLE_ARM_STAGE_ORDER:
        raise ValueError("sequential single-arm stage order is immutable")
    center = np.asarray(plan.get("original_center_base_m"), dtype=float).reshape(3)
    if not np.isfinite(center).all():
        raise ValueError("sequential original center must be finite")
    detected_center = np.asarray(
        plan.get("perception", {}).get("detected_center_base_m"), dtype=float
    ).reshape(3)
    if (
        plan.get("perception", {}).get("edge_midpoint_method")
        != "rectangle_direction_then_nearest_contour_projection"
    ):
        raise ValueError("sequential grasp edge must be projected to the real contour")
    selected_edge = np.asarray(
        plan.get("perception", {}).get("selected_keypoint_base_m"), dtype=float
    ).reshape(3)
    candidates = plan.get("perception", {}).get("right_side_grasp_candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 24:
        raise ValueError("right-side contour candidate set must contain 1-24 points")
    candidate_ids = [item.get("id") for item in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("right-side contour candidate IDs must be unique")
    selected_candidate_id = plan.get("grasp_geometry", {}).get(
        "selected_candidate_id"
    )
    if selected_candidate_id not in candidate_ids:
        raise ValueError("selected grasp candidate is not bound to the plan")
    selected_candidate = candidates[candidate_ids.index(selected_candidate_id)]
    if not np.allclose(
        selected_edge,
        np.asarray(selected_candidate.get("base_xyz_m"), dtype=float).reshape(3),
        atol=1e-9,
    ):
        raise ValueError("selected grasp point differs from its contour candidate")
    contour_policy = plan.get("perception", {}).get("right_side_contour_policy", {})
    if (
        contour_policy.get("definition")
        != "fitted_right_side_contour_with_inset_finger_contact_center"
        or not np.isclose(
            float(contour_policy.get("corner_exclusion_m")), 0.01, atol=1e-9
        )
        or not np.isclose(
            float(contour_policy.get("sample_spacing_m")), 0.01, atol=1e-9
        )
        or int(contour_policy.get("maximum_candidates")) != 24
        or not np.isclose(
            float(contour_policy.get("finger_contact_width_m")), 0.01, atol=1e-9
        )
        or not np.isclose(
            float(contour_policy.get("contact_center_inset_m")), 0.005, atol=1e-9
        )
    ):
        raise ValueError("unexpected right-side contour sampling policy")
    expected_normal = detected_center[:2] - selected_edge[:2]
    expected_normal_norm = float(np.linalg.norm(expected_normal))
    if expected_normal_norm <= 1e-6:
        raise ValueError("perceived towel center and selected edge are degenerate")
    expected_normal /= expected_normal_norm
    grasp_geometry = plan.get("grasp_geometry", {})
    inward_normal = np.asarray(
        grasp_geometry.get("edge_inward_normal_base_xy"), dtype=float
    ).reshape(2)
    if not np.allclose(inward_normal, expected_normal, atol=1e-9):
        raise ValueError("grasp inward normal does not match the perceived towel edge")
    if grasp_geometry.get("jaw_closing_axis_tool") != "tool_y":
        raise ValueError("towel edge grasp must use tool_y as the jaw-closing axis")
    tilt_candidates = [0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0]
    if (
        grasp_geometry.get("orientation_policy")
        != "edge_normal_constrained_tilt_about_tool_y"
        or grasp_geometry.get("tilt_candidates_deg") != tilt_candidates
        or not np.isclose(
            float(grasp_geometry.get("jaw_closing_axis_max_vertical_deg")),
            3.0,
            atol=1e-9,
        )
        or not np.isclose(
            float(grasp_geometry.get("maximum_tool_z_tilt_from_down_deg")),
            45.0,
            atol=1e-9,
        )
    ):
        raise ValueError("unexpected edge-normal constrained grasp orientation policy")
    selected_tilt = grasp_geometry.get("selected_tilt_deg")
    if selected_tilt is not None and float(selected_tilt) not in tilt_candidates:
        raise ValueError("selected grasp tilt is outside the approved search set")
    for stage in stages:
        if stage.get("active_side") != side:
            raise ValueError("all sequential stages must use the phase side")
        if stage.get("inactive_arm_policy") != "hold_session_initial_joints":
            raise ValueError("inactive arm must hold the session initial state")
        if stage["id"] != "home":
            point = np.asarray(
                stage.get("active_tcp_position_base_m"), dtype=float
            ).reshape(3)
            if not np.isfinite(point).all():
                raise ValueError(f"{stage['id']} active target is non-finite")
            strict = stage["id"] in {"align_grasp", "grasp"}
            expected_orientation = (
                "edge_normal_constrained_tilt"
                if strict
                else "topdown_or_multi_direction_tilt"
            )
            if stage.get("orientation") != expected_orientation:
                raise ValueError(f"{stage['id']} has unexpected orientation policy")
            if strict:
                expected_tilts = (
                    tilt_candidates
                    if selected_tilt is None
                    else [float(selected_tilt)]
                )
                if stage.get("tilt_search_deg") != expected_tilts:
                    raise ValueError(f"{stage['id']} has an unbound tilt search")
                if stage.get("jaw_closing_axis_tool") != "tool_y":
                    raise ValueError(f"{stage['id']} must align the gripper jaw axis")
                if not np.allclose(
                    np.asarray(stage.get("jaw_closing_axis_base_xy"), dtype=float),
                    inward_normal,
                    atol=1e-9,
                ):
                    raise ValueError(f"{stage['id']} jaw axis misses the towel edge normal")
    align_stage = stages[SINGLE_ARM_STAGE_ORDER.index("align_grasp")]
    grasp_stage = stages[SINGLE_ARM_STAGE_ORDER.index("grasp")]
    align_fingertip = np.asarray(
        align_stage["active_fingertip_position_base_m"], dtype=float
    )
    grasp_fingertip = np.asarray(
        grasp_stage["active_fingertip_position_base_m"], dtype=float
    )
    alignment_clearance = float(grasp_geometry.get("alignment_clearance_m"))
    if not 0.04 <= alignment_clearance <= 0.05:
        raise ValueError("grasp alignment clearance must remain within 4-5cm")
    tilt_rad = math.radians(float(selected_tilt or 0.0))
    tool_y = np.array([inward_normal[0], inward_normal[1], 0.0])
    strict_tool_z = np.array([0.0, 0.0, -1.0])
    strict_tool_x = np.cross(tool_y, strict_tool_z)
    expected_tool_z = (
        strict_tool_x * math.sin(tilt_rad)
        + strict_tool_z * math.cos(tilt_rad)
    )
    strict_rotation = np.column_stack([strict_tool_x, tool_y, strict_tool_z])
    local_tilt = np.array(
        [
            [math.cos(tilt_rad), 0.0, math.sin(tilt_rad)],
            [0.0, 1.0, 0.0],
            [-math.sin(tilt_rad), 0.0, math.cos(tilt_rad)],
        ]
    )
    selected_rotation = strict_rotation @ local_tilt
    selected_lowest_offset_z = (
        R1LiteTableCollisionKinematics.finger_minimum_z_offset_from_tcp(
            selected_rotation,
            fingers_open=True,
        )
    )
    tcp_to_fingertip_m = float(plan["safety"]["tcp_to_fingertip_m"])
    actual_lowest_point_z = (
        grasp_fingertip[2]
        - expected_tool_z[2] * tcp_to_fingertip_m
        + selected_lowest_offset_z
    )
    reference_lowest_point_z = float(
        grasp_geometry.get("reference_topdown_lowest_point_z_m")
    )
    nominal_fingertip_z = (
        float(plan["safety"]["table_z_m"])
        + float(plan["safety"]["towel_thickness_m"])
        + float(plan["safety"]["grasp_open_finger_table_margin_m"])
    )
    expected_compensation = grasp_fingertip[2] - nominal_fingertip_z
    if (
        grasp_geometry.get("contact_reference")
        != "lowest_open_finger_mesh_point"
        or not np.isclose(
            actual_lowest_point_z, reference_lowest_point_z, atol=1e-9
        )
        or not np.isclose(
            float(
                grasp_geometry.get(
                    "selected_fingertip_center_height_compensation_m"
                )
            ),
            expected_compensation,
            atol=1e-9,
        )
    ):
        raise ValueError("tilted grasp does not preserve physical finger clearance")
    if not np.allclose(
        align_fingertip - grasp_fingertip,
        -expected_tool_z * alignment_clearance,
        atol=1e-9,
    ):
        raise ValueError("final grasp approach must follow the selected tool_z axis")
    safety = plan.get("safety", {})
    place = stages[SINGLE_ARM_STAGE_ORDER.index("place_center")]
    grasp = stages[SINGLE_ARM_STAGE_ORDER.index("grasp")]
    lift = stages[SINGLE_ARM_STAGE_ORDER.index("lift")]
    if not np.isclose(float(safety.get("lift_height_m")), 0.05, atol=1e-9):
        raise ValueError("sequential lift height must be exactly 5cm")
    if not np.allclose(
        np.asarray(lift["active_tcp_position_base_m"], dtype=float)
        - np.asarray(grasp["active_tcp_position_base_m"], dtype=float),
        [0.0, 0.0, 0.05],
        atol=1e-9,
    ):
        raise ValueError("sequential lift TCP must be exactly 5cm above grasp")
    place_fingertip = np.asarray(
        place["active_fingertip_position_base_m"], dtype=float
    )
    if not np.allclose(place_fingertip[:2], center[:2], atol=1e-9):
        raise ValueError("place_center must preserve the immutable original center")
    place_clearance = float(safety.get("place_release_clearance_m"))
    post_close_z_policy = safety.get("post_close_tcp_z_policy", {})
    if (
        post_close_z_policy.get("enabled") is not True
        and not np.isclose(place_clearance, 0.05, atol=1e-9)
    ):
        raise ValueError("sequential release clearance must be exactly 5cm")
    if post_close_z_policy.get("enabled") is True:
        if place_clearance < 0.05:
            raise ValueError("Z-monotonic release clearance cannot be below 5cm")
        if post_close_z_policy.get("side") != plan["phase_side"]:
            raise ValueError("post-close TCP Z policy side mismatch")
        if post_close_z_policy.get("protected_stage_ids") != [
            "lift",
            "place_center",
        ]:
            raise ValueError("unexpected post-close TCP Z protected stages")
        if float(place["active_tcp_position_base_m"][2]) < float(
            lift["active_tcp_position_base_m"][2]
        ):
            raise ValueError("place_center TCP Z cannot be below lift TCP Z")
    if not np.isclose(
        place_fingertip[2] - float(safety["table_z_m"]),
        place_clearance,
        atol=1e-9,
    ):
        raise ValueError("place_center fingertip height mismatches release clearance")
    if stages[-1].get("target") != "session_initial_joint_state":
        raise ValueError("home must target the session initial joint state")
    if safety.get("collision_policy") != "active_arm_table_only":
        raise ValueError("sequential phases must use the table-only collision gate")
    if (
        safety.get("post_grasp_closed_lift_collision_policy")
        != "skip_table_collision_if_gripper_close_succeeded"
    ):
        raise ValueError(
            "closed-gripper lift must skip table collision after successful close"
        )
    if not np.isclose(
        float(safety.get("post_grasp_close_dwell_s")), 1.0, atol=1e-9
    ):
        raise ValueError("sequential grasp close dwell must be exactly 1 second")
    if (
        safety.get("gripper_table_collision_model")
        != "two_fully_open_finger_mesh_aabbs"
    ):
        raise ValueError("sequential phases must check both open gripper fingers")
    if not np.isclose(
        float(safety.get("gripper_finger_open_travel_m")),
        0.05,
        atol=1e-9,
    ):
        raise ValueError("gripper table gate must assume the full 5cm finger travel")
    if not np.isclose(
        float(safety.get("grasp_open_finger_table_margin_m")),
        0.001,
        atol=1e-9,
    ):
        raise ValueError("grasp must preserve the 1mm open-finger table margin")
    expected_contact_link = (
        "finger_link1"
        if R1LiteTableCollisionKinematics.finger_minimum_z_offsets_from_tcp(
            selected_rotation,
            fingers_open=True,
        )[0]
        <= R1LiteTableCollisionKinematics.finger_minimum_z_offsets_from_tcp(
            selected_rotation,
            fingers_open=True,
        )[1]
        else "finger_link2"
    )
    if safety.get("grasp_allowed_table_contact_link") != expected_contact_link:
        raise ValueError("designated contact must be the lower open finger mesh")
    if not np.isclose(
        float(safety.get("grasp_designated_contact_min_clearance_m")),
        -0.0002,
        atol=1e-9,
    ):
        raise ValueError("designated grasp finger penetration bound must be 0.2mm")
    if not np.isclose(
        float(safety.get("grasp_noncontact_min_clearance_m")),
        0.001,
        atol=1e-9,
    ):
        raise ValueError("non-contact grasp geometry must preserve 1mm clearance")
    if safety.get("grasp_contact_stage_only") is not True:
        raise ValueError("designated finger contact must be limited to grasp")
    if safety.get("dual_arm_collision_gate_required") is not False:
        raise ValueError("sequential phases must not enable dual-arm collision gates")
    refinement = safety.get("cartesian_refinement_policy", {})
    if not np.isclose(
        float(refinement.get("grasp_xy_tolerance_m")),
        0.010,
        atol=1e-9,
    ):
        raise ValueError("grasp XY arrival tolerance must be exactly 10mm")
    if not np.isclose(
        float(refinement.get("grasp_z_tolerance_m")),
        0.0005,
        atol=1e-9,
    ):
        raise ValueError("grasp Z arrival tolerance must be exactly 0.5mm")
    effort_contact = safety.get("grasp_effort_contact_policy", {})
    expected_effort_contact = {
        "enabled": True,
        "feedback_source": "hdas_joint_state_effort",
        "reference_capture": "after_unsegmented_initial_10mm_descent",
        "joint_indices_zero_based": [1, 2],
        "sum_abs_residual_threshold_nm": 2.5,
        "sample_count": 40,
        "motion_monitor_sample_count": 5,
        "maximum_median_abs_velocity_rad_s": 0.05,
        "require_settled_velocity_during_motion": False,
        "maximum_tcp_z_above_target_m": 0.005,
        "arrival_logic": (
            "continuous_fixed_baseline_effort_during_post_arrival_probe"
        ),
        "calibration_id": "right_grasp_table_20260817_fixed_baseline",
    }
    if effort_contact != expected_effort_contact:
        raise ValueError("grasp effort-contact policy mismatches calibration")
    grasp = stages[SINGLE_ARM_STAGE_ORDER.index("grasp")]
    post_arrival_probe = safety.get("grasp_post_arrival_probe_policy", {})
    if post_arrival_probe != {
        "enabled": True,
        "start_condition": "grasp_xy_within_10mm_and_z_error_below_5mm",
        "maximum_entry_z_error_m": 0.005,
        "direction_base": [0.0, 0.0, -1.0],
        "initial_descent_m": 0.010,
        "absolute_height_refinement_iterations": 20,
        "absolute_contact_tcp_z_base_m": float(
            grasp["active_tcp_position_base_m"][2]
            - (
                float(plan["grasp_geometry"]["reference_topdown_lowest_point_z_m"])
                - float(safety["table_z_m"])
            )
        ),
        "absolute_contact_z_tolerance_m": 0.0005,
        "skip_collision_during_initial_descent": True,
        "check_collision_during_absolute_height_refinement": True,
        "forbid_upward_tcp_motion": True,
        "maximum_xy_shift_per_descent_m": 0.010,
        "stop_condition": (
            "stable_effort_contact_or_absolute_height_refinement"
            "_or_descent_condition_unavailable"
        ),
        "require_stable_contact_before_close": False,
        "close_after_probe_commands_without_contact": True,
        "close_if_descent_condition_unavailable": True,
        "designated_contact_min_clearance_m": -0.0152,
        "noncontact_min_clearance_m": 0.001,
    }:
        raise ValueError("sequential post-arrival grasp probe policy mismatch")


def build_eighth_bimanual_recenter_plan(
    geometry: dict[str, Any],
    *,
    table_z_m: float,
    first_center_base_m: list[float] | None = None,
    tcp_to_fingertip_m: float = 0.078,
    towel_thickness_m: float = 0.00002,
    grasp_table_margin_m: float = 0.001,
    pregrasp_clearance_m: float = 0.10,
    grasp_alignment_clearance_m: float = 0.05,
    lift_height_m: float = 0.08,
    shared_cartesian_step_m: float = 0.05,
    relative_geometry_tolerance_m: float = 0.005,
) -> dict[str, Any]:
    """Build the first, perception-to-plan part of pipeline eight."""
    if geometry.get("grasp_point_method") != (
        "nearest_robot_two_projected_rectangle_corners"
    ):
        raise ValueError("eighth pipeline requires the two nearest towel corners")
    if geometry.get("points_on_real_external_contour") is not True:
        raise ValueError("eighth grasp points must lie on the real external contour")
    left_pixel = np.asarray(
        geometry["keypoints"]["left_grasp"]["pixel_uv"], dtype=float
    ).reshape(2)
    right_pixel = np.asarray(
        geometry["keypoints"]["right_grasp"]["pixel_uv"], dtype=float
    ).reshape(2)
    if not (
        np.isfinite(left_pixel).all()
        and np.isfinite(right_pixel).all()
        and left_pixel[0] < right_pixel[0]
    ):
        raise ValueError("eighth grasp corner pixels must be finite and ordered")
    if not np.isclose(float(lift_height_m), 0.08, atol=1e-12):
        raise ValueError("eighth pipeline lift height is fixed at 0.08m")
    if not np.isclose(float(shared_cartesian_step_m), 0.05, atol=1e-12):
        raise ValueError("eighth shared Cartesian step is fixed at 0.05m")
    if float(relative_geometry_tolerance_m) <= 0.0:
        raise ValueError("relative geometry tolerance must be positive")

    detected_center = _point(geometry, "center")
    first_center = (
        detected_center.copy()
        if first_center_base_m is None
        else np.asarray(first_center_base_m, dtype=np.float64).reshape(3)
    )
    if not np.isfinite(first_center).all():
        raise ValueError("first center must be finite")
    left_fingertip = _point(geometry, "left_grasp")
    right_fingertip = _point(geometry, "right_grasp")
    detected_left_fingertip = left_fingertip.copy()
    detected_right_fingertip = right_fingertip.copy()
    detected_center_for_offset = _point(geometry, "center")
    center_y_delta = float(
        detected_center_for_offset[1] - left_fingertip[1]
    )
    if abs(center_y_delta) <= 1e-6:
        raise ValueError("left grasp point and towel center have degenerate Y")
    left_fingertip[1] += math.copysign(0.015, center_y_delta)
    right_fingertip[0] -= 0.005
    if not np.isclose(float(towel_thickness_m), 0.00002, atol=1e-12):
        raise ValueError("eighth towel thickness is fixed at 0.00002m")
    if not np.isclose(float(grasp_table_margin_m), 0.001, atol=1e-12):
        raise ValueError("eighth grasp table margin is fixed at 0.001m")
    grasp_z = (
        float(table_z_m)
        + float(towel_thickness_m)
        + float(grasp_table_margin_m)
    )
    left_fingertip[2] = grasp_z
    right_fingertip[2] = grasp_z
    initial_relative_vector = right_fingertip - left_fingertip
    if float(np.linalg.norm(initial_relative_vector[:2])) <= 1e-6:
        raise ValueError("eighth grasp pair is degenerate")

    tool_offset = np.array([0.0, 0.0, float(tcp_to_fingertip_m)])
    left_grasp_tcp = left_fingertip + tool_offset
    right_grasp_tcp = right_fingertip + tool_offset
    lift_delta = np.array([0.0, 0.0, float(lift_height_m)])
    left_lift_fingertip = left_fingertip + lift_delta
    right_lift_fingertip = right_fingertip + lift_delta
    pullback_delta = np.array([-0.02, 0.0, 0.0])
    left_pullback_fingertip = left_lift_fingertip + pullback_delta
    right_pullback_fingertip = right_lift_fingertip + pullback_delta
    grasp_midpoint = (left_fingertip + right_fingertip) / 2.0
    pullback_midpoint = (
        left_pullback_fingertip + right_pullback_fingertip
    ) / 2.0
    centerward_translation = np.array(
        [
            first_center[0] - pullback_midpoint[0],
            first_center[1] - pullback_midpoint[1],
            0.0,
        ]
    )
    translation = centerward_translation
    left_center_fingertip = left_pullback_fingertip + translation
    right_center_fingertip = right_pullback_fingertip + translation
    def motion_stage(
        stage_id: str,
        left_fingertip_position: np.ndarray,
        right_fingertip_position: np.ndarray,
        *,
        speed: str,
        post_grasp_tilt: bool = False,
    ) -> dict[str, Any]:
        tool_z = (
            np.array([0.5, 0.0, -math.sqrt(3.0) / 2.0])
            if post_grasp_tilt
            else np.array([0.0, 0.0, -1.0])
        )
        stage_tool_offset = -tool_z * float(tcp_to_fingertip_m)
        return {
            "id": stage_id,
            "left_tcp_position_base_m": (
                left_fingertip_position + stage_tool_offset
            ).tolist(),
            "right_tcp_position_base_m": (
                right_fingertip_position + stage_tool_offset
            ).tolist(),
            "left_fingertip_position_base_m": left_fingertip_position.tolist(),
            "right_fingertip_position_base_m": right_fingertip_position.tolist(),
            "orientation": (
                "shared_world_y_tilt_minus_30deg"
                if post_grasp_tilt
                else "strict_topdown"
            ),
            "tool_z_base": tool_z.tolist(),
            "synchronization": "shared_cartesian_progress",
            "speed": speed,
        }

    stages = [
        motion_stage(
            "grasp",
            left_fingertip,
            right_fingertip,
            speed="very_slow",
        ),
        motion_stage(
            "lift",
            left_lift_fingertip,
            right_lift_fingertip,
            speed="very_slow",
            post_grasp_tilt=True,
        ),
        motion_stage(
            "pullback",
            left_pullback_fingertip,
            right_pullback_fingertip,
            speed="very_slow",
            post_grasp_tilt=True,
        ),
        motion_stage(
            "translate_center",
            left_center_fingertip,
            right_center_fingertip,
            speed="very_slow",
            post_grasp_tilt=True,
        ),
        {
            **motion_stage(
                "release_retract",
                left_center_fingertip,
                right_center_fingertip,
                speed="very_slow",
                post_grasp_tilt=True,
            ),
            "motion_policy": "skip_after_release_direct_home",
            "release_left_fingertip_position_base_m": (
                left_center_fingertip.tolist()
            ),
            "release_right_fingertip_position_base_m": (
                right_center_fingertip.tolist()
            ),
        },
        {
            "id": "home",
            "target": "session_initial_joint_state",
            "synchronization": "shared_joint_progress",
            "speed": "slow",
        },
    ]
    quality = geometry.get("quality")
    if not isinstance(quality, dict):
        raise ValueError("eighth perception quality metrics are required")
    plan = {
        "schema": EIGHTH_BIMANUAL_RECENTER_SCHEMA,
        "pipeline": 8,
        "task": "bimanual_towel_recenter",
        "first_center_base_m": first_center.tolist(),
        "perception": {
            "source": geometry.get("source"),
            "grasp_point_method": geometry.get("grasp_point_method"),
            "same_scanline": geometry.get("same_scanline"),
            "points_on_real_external_contour": geometry.get(
                "points_on_real_external_contour"
            ),
            "corner_candidates": geometry.get("corner_candidates"),
            "robot_reference_base_xyz_m": geometry.get(
                "robot_reference_base_xyz_m"
            ),
            "pixel_order": geometry.get("pixel_order"),
            "quality": dict(quality),
            "external_contour_pixel_uv": geometry.get(
                "external_contour_pixel_uv"
            ),
            "external_contour_base_xyz_m": geometry.get(
                "external_contour_base_xyz_m"
            ),
            "external_contour_sample_stride": geometry.get(
                "external_contour_sample_stride"
            ),
            "keypoints": {
                name: dict(geometry["keypoints"][name])
                for name in ("center", "left_grasp", "right_grasp")
            },
        },
        "grasp_geometry": {
            "initial_full_relative_vector_right_minus_left_m": (
                initial_relative_vector.tolist()
            ),
            "grasp_pair_midpoint_base_m": grasp_midpoint.tolist(),
            "center_translation_base_m": translation.tolist(),
            "translation_destination": "original_towel_center",
            "post_lift_pullback_base_m": pullback_delta.tolist(),
            "translation_start_midpoint_base_m": pullback_midpoint.tolist(),
            "translation_target_midpoint_base_m": first_center.tolist(),
            "required_tool_z_base": [0.0, 0.0, -1.0],
            "detected_left_grasp_base_m": detected_left_fingertip.tolist(),
            "left_grasp_centerward_y_offset_m": 0.015,
            "detected_right_grasp_base_m": detected_right_fingertip.tolist(),
            "right_grasp_robotward_x_offset_m": -0.005,
        },
        "stages": stages,
        "gripper_events": [
            {
                "before": "grasp",
                "action": "set_both_grippers",
                "position": 40.0,
            },
            {"after": "contact_probe", "action": "close_both_grippers"},
            {"before": "release_retract", "action": "open_both_grippers"},
        ],
        "safety": {
            "table_z_m": float(table_z_m),
            "tcp_to_fingertip_m": float(tcp_to_fingertip_m),
            "towel_thickness_m": float(towel_thickness_m),
            "grasp_table_margin_m": float(grasp_table_margin_m),
            "left_grasp_centerward_y_offset_m": 0.015,
            "right_grasp_robotward_x_offset_m": -0.005,
            "pregrasp_gripper_position": 40.0,
            "grasp_x_tolerance_m": 0.005,
            "grasp_maximum_inward_x_error_m": 0.005,
            "post_grasp_world_y_tilt_deg": -30.0,
            "lift_height_m": float(lift_height_m),
            "post_lift_pullback_m": 0.02,
            "post_lift_pullback_direction_base": "negative_x_robotward",
            "release_height_above_grasp_m": float(lift_height_m),
            "release_destination": "original_towel_center",
            "translation_reachability_policy": (
                "farthest_certified_waypoint_if_endpoint_unreachable"
            ),
            "translation_control_policy": {
                "mode": "midpoint_plus_fixed_grasp_relative_vector",
                "reference_vector_source": (
                    "initial_full_relative_vector_right_minus_left_m"
                ),
            },
            "post_release_policy": "skip_retract_direct_home",
            "maximum_shared_cartesian_step_m": float(shared_cartesian_step_m),
            "joint_command_progress_steps": 5,
            "collision_policy": "first_phase_collision_disabled",
            "planning_collision_checks_enabled": False,
            "certification_collision_checks_enabled": False,
            "execution_collision_checks_enabled": False,
            "collision_checked_stage_ids": [],
            "dual_arm_collision_gate_required": False,
            "second_phase_perception_policy": "fresh_head_camera_reperception",
            "grasp_refinement_policy": {
                "enabled": True,
                "maximum_iterations": 20,
                "individual_tcp_tolerance_m": 0.015,
                "initial_seed_policy": (
                    "best_joint_bounded_candidate_without_ik_fk_acceptance"
                ),
                "initial_seed_ik_fk_acceptance_required": False,
                "initial_seed_maximum_total_joint_delta_rad": 3.0,
                "initial_seed_maximum_substep_joint_delta_rad": 1.0,
                "gripper_close_requires_both_tcp_within_tolerance": False,
                "completion_policy": "force_continue_after_maximum_iterations",
                "post_refinement_tcp_error_gate_required": False,
                "no_local_optimum_early_exit": True,
                "strategy": "independent_preclose_worst_error_first",
                "synchronized_pair_only": False,
                "inactive_arm_holds": True,
                "final_relative_geometry_gate_required": False,
            },
            "contact_probe_policy": {
                "enabled": True,
                "trigger": "after_grasp_before_gripper_close",
                "probe_order": ["right", "left"],
                "axis_base": [0.0, 0.0, -1.0],
                "step_m": 0.0005,
                "maximum_descent_m": 0.04,
                "sample_count": 40,
                "joint_indices_zero_based": [1, 2],
                "residual_reference_mode": "probe_start_fixed",
                "sum_abs_residual_threshold_nm": 6.0,
                "single_joint_emergency_threshold_nm": 5.0,
                "maximum_median_abs_velocity_rad_s": 0.05,
                "confirmation_samples": 2,
                "freeze_on_first_contact_indication": True,
                "stationary_confirmation_maximum_attempts": 12,
                "ik_probes_per_step": 8,
                "maximum_fk_position_error_m": 0.015,
                "maximum_incremental_joint_delta_rad": 0.25,
                "both_arms_contact_required_before_close": True,
                "failure_policy": "retract_both_to_preprobe_and_abort",
                "table_collision_policy": "bounded_intentional_contact_probe",
                "execution_grasp_table_gate_bypassed": True,
                "lock_xy_to_grasp_target": True,
                "threshold_source": "restored_dual_arm_fixed_baseline_20260812",
            },
            "ik_branch_transition": {
                "enabled": True,
                "source": "historical_certified_pregrasp_branch_seed",
                "left_joints_rad": [
                    0.06677660969649797,
                    2.9091607378448114,
                    -2.4480134142278853,
                    1.322660292429798,
                    0.013784489037664228,
                    0.14381353497903884,
                ],
                "right_joints_rad": [
                    -0.09255347056790428,
                    2.2279198527217137,
                    -1.0554818120342575,
                    0.17235419688951864,
                    -0.05216138924052844,
                    0.07816264354899514,
                ],
                "maximum_transition_joint_delta_rad": 3.0,
                "maximum_transition_to_grasp_joint_delta_rad": 1.7,
                "initial_grasp_fk_tolerance_m": 0.035,
                "sweep_samples_per_segment": 0,
            },
            "online_relative_geometry_gate": {
                "enabled": True,
                "check_each_shared_cartesian_step": True,
                "preserve_full_xyz_relative_vector": True,
                "reference": "initial_full_relative_vector_right_minus_left_m",
                "maximum_vector_error_m": float(relative_geometry_tolerance_m),
                "failure_policy": "abort_both_arms",
            },
            "execution_default": "preview_only",
        },
    }
    validate_eighth_bimanual_recenter_plan(plan)
    return plan


def validate_eighth_bimanual_recenter_plan(plan: dict[str, Any]) -> None:
    """Strictly validate immutable geometry and safety invariants for pipeline 8."""
    if plan.get("schema") != EIGHTH_BIMANUAL_RECENTER_SCHEMA:
        raise ValueError("unexpected eighth bimanual recenter schema")
    if plan.get("pipeline") != 8 or plan.get("task") != "bimanual_towel_recenter":
        raise ValueError("unexpected eighth pipeline identity")
    stages = plan.get("stages")
    if not isinstance(stages, list) or tuple(
        stage.get("id") for stage in stages
    ) != EIGHTH_BIMANUAL_STAGE_ORDER:
        raise ValueError("eighth bimanual stage order is immutable")
    perception = plan.get("perception", {})
    if (
        perception.get("source") != "sam3_largest_external_contour"
        or perception.get("grasp_point_method")
        != "nearest_robot_two_projected_rectangle_corners"
        or perception.get("points_on_real_external_contour") is not True
        or perception.get("pixel_order") != "camera_left_to_camera_right"
    ):
        raise ValueError("eighth perception provenance or contour invariants changed")
    contour_pixels = np.asarray(
        perception.get("external_contour_pixel_uv"), dtype=float
    )
    contour_base = np.asarray(
        perception.get("external_contour_base_xyz_m"), dtype=float
    )
    contour_stride = perception.get("external_contour_sample_stride")
    if (
        contour_pixels.ndim != 2
        or contour_pixels.shape[1:] != (2,)
        or contour_pixels.shape[0] < 4
        or contour_base.shape != (contour_pixels.shape[0], 3)
        or not np.isfinite(contour_pixels).all()
        or not np.isfinite(contour_base).all()
        or isinstance(contour_stride, bool)
        or not isinstance(contour_stride, int)
        or contour_stride < 1
    ):
        raise ValueError("eighth external contour audit geometry is invalid")
    keypoints = perception.get("keypoints", {})
    left_pixel = np.asarray(
        keypoints.get("left_grasp", {}).get("pixel_uv"), dtype=float
    ).reshape(2)
    right_pixel = np.asarray(
        keypoints.get("right_grasp", {}).get("pixel_uv"), dtype=float
    ).reshape(2)
    if not (
        np.isfinite(left_pixel).all()
        and np.isfinite(right_pixel).all()
        and left_pixel[0] < right_pixel[0]
    ):
        raise ValueError("eighth corner pixels are not ordered")
    quality = perception.get("quality")
    required_quality = {
        "pair_separation_px",
        "rectangularity",
        "selection_margin_to_third_corner_m",
        "maximum_corner_projection_distance_px",
        "left_contour_distance_px",
        "right_contour_distance_px",
        "quality_score",
    }
    if not isinstance(quality, dict) or not required_quality.issubset(quality):
        raise ValueError("eighth perception quality metrics are incomplete")
    quality_values = np.asarray(
        [quality[name] for name in required_quality], dtype=float
    )
    if not np.isfinite(quality_values).all():
        raise ValueError("eighth perception quality metrics must be finite")
    if (
        not np.isclose(
            float(quality["pair_separation_px"]),
            float(np.linalg.norm(right_pixel - left_pixel)),
            atol=1e-9,
        )
        or not 0.5 <= float(quality["rectangularity"]) <= 1.01
        or float(quality["selection_margin_to_third_corner_m"]) < 0.0
        or float(quality["maximum_corner_projection_distance_px"]) < 0.0
        or float(quality["left_contour_distance_px"]) != 0.0
        or float(quality["right_contour_distance_px"]) != 0.0
        or not 0.0 <= float(quality["quality_score"]) <= 1.0
    ):
        raise ValueError("eighth perception quality metrics are inconsistent")
    corner_candidates = perception.get("corner_candidates")
    if not isinstance(corner_candidates, list) or len(corner_candidates) != 4:
        raise ValueError("eighth perception must provide four towel corners")
    corner_distances = np.asarray(
        [item.get("robot_xy_distance_m") for item in corner_candidates],
        dtype=float,
    )
    selected_corner_indices = [
        index
        for index, item in enumerate(corner_candidates)
        if item.get("selected") is True
    ]
    nearest_indices = sorted(
        range(4), key=lambda index: (corner_distances[index], index)
    )[:2]
    if (
        not np.isfinite(corner_distances).all()
        or np.any(corner_distances < 0.0)
        or selected_corner_indices != [
            index for index in range(4) if index in nearest_indices
        ]
    ):
        raise ValueError("eighth selected corners are not the two nearest robot corners")

    bimanual_motion_ids = {
        "grasp",
        "lift",
        "pullback",
        "translate_center",
        "release_retract",
    }
    tcp_distance = float(
        plan.get("safety", {}).get("tcp_to_fingertip_m")
    )
    if not np.isfinite(tcp_distance) or tcp_distance <= 0.0:
        raise ValueError("eighth TCP-to-fingertip offset is invalid")
    for stage in stages:
        if stage["id"] not in bimanual_motion_ids:
            continue
        tilted = stage.get("id") in {
            "lift",
            "pullback",
            "translate_center",
            "release_retract",
        }
        expected_orientation = (
            "shared_world_y_tilt_minus_30deg"
            if tilted
            else "strict_topdown"
        )
        expected_tool_z = (
            [0.5, 0.0, -math.sqrt(3.0) / 2.0]
            if tilted
            else [0.0, 0.0, -1.0]
        )
        if (
            stage.get("orientation") != expected_orientation
            or not np.allclose(
                stage.get("tool_z_base"), expected_tool_z, atol=1e-12
            )
            or (
                stage["id"] in bimanual_motion_ids
                and stage.get("synchronization") != "shared_cartesian_progress"
            )
        ):
            raise ValueError(f"{stage.get('id')} violates approved motion policy")
    for stage in stages:
        stage_id = stage["id"]
        if stage_id not in bimanual_motion_ids:
            continue
        for side in ("left", "right"):
            fingertip = np.asarray(
                stage.get(f"{side}_fingertip_position_base_m"), dtype=float
            ).reshape(3)
            tcp = np.asarray(
                stage.get(f"{side}_tcp_position_base_m"), dtype=float
            ).reshape(3)
            if not np.isfinite(fingertip).all() or not np.isfinite(tcp).all():
                raise ValueError(f"{stage['id']} {side} target is non-finite")
            expected_offset = (
                -np.asarray(stage["tool_z_base"], dtype=float) * tcp_distance
            )
            if not np.allclose(tcp - fingertip, expected_offset, atol=1e-9):
                raise ValueError(f"{stage['id']} {side} TCP offset changed")

    by_id = {stage["id"]: stage for stage in stages}

    def pair(stage_id: str) -> tuple[np.ndarray, np.ndarray]:
        stage = by_id[stage_id]
        return (
            np.asarray(stage["left_fingertip_position_base_m"], dtype=float),
            np.asarray(stage["right_fingertip_position_base_m"], dtype=float),
        )

    grasp_left, grasp_right = pair("grasp")
    lift_left, lift_right = pair("lift")
    pullback_left, pullback_right = pair("pullback")
    center_left, center_right = pair("translate_center")
    table_z = float(plan.get("safety", {}).get("table_z_m"))
    towel_thickness = float(plan.get("safety", {}).get("towel_thickness_m"))
    grasp_table_margin = float(
        plan.get("safety", {}).get("grasp_table_margin_m")
    )
    if (
        not np.isfinite(table_z)
        or not np.isfinite(towel_thickness)
        or not np.isclose(towel_thickness, 0.00002, atol=1e-12)
        or not np.isclose(grasp_table_margin, 0.001, atol=1e-12)
    ):
        raise ValueError("eighth table or towel geometry is non-finite")
    if not np.isclose(
        float(
            plan.get("safety", {}).get(
                "post_grasp_world_y_tilt_deg"
            )
        ),
        -30.0,
        atol=1e-12,
    ):
        raise ValueError("eighth post-grasp tilt must remain -30deg")
    perceived_left = np.asarray(
        keypoints.get("left_grasp", {}).get("base_xyz_m"), dtype=float
    ).reshape(3)
    perceived_right = np.asarray(
        keypoints.get("right_grasp", {}).get("base_xyz_m"), dtype=float
    ).reshape(3)
    perceived_center = np.asarray(
        keypoints.get("center", {}).get("base_xyz_m"), dtype=float
    ).reshape(3)
    center_y_delta = float(perceived_center[1] - perceived_left[1])
    expected_left_xy = perceived_left[:2].copy()
    expected_left_xy[1] += math.copysign(0.015, center_y_delta)
    expected_right_xy = perceived_right[:2].copy()
    expected_right_xy[0] -= 0.005
    if not (
        np.allclose(grasp_left[:2], expected_left_xy, atol=1e-9)
        and np.allclose(grasp_right[:2], expected_right_xy, atol=1e-9)
        and np.isclose(
            grasp_left[2],
            table_z + towel_thickness + grasp_table_margin,
            atol=1e-9,
        )
        and np.isclose(
            grasp_right[2],
            table_z + towel_thickness + grasp_table_margin,
            atol=1e-9,
        )
    ):
        raise ValueError("grasp targets no longer match perceived contour points")
    if not np.isclose(
        float(
            plan.get("safety", {}).get(
                "left_grasp_centerward_y_offset_m"
            )
        ),
        0.015,
        atol=1e-12,
    ):
        raise ValueError(
            "eighth left grasp centerward Y offset must remain 15mm"
        )
    if not np.isclose(
        float(
            plan.get("safety", {}).get(
                "right_grasp_robotward_x_offset_m"
            )
        ),
        -0.005,
        atol=1e-12,
    ):
        raise ValueError(
            "eighth right grasp robotward X offset must remain -5mm"
        )
    reference = np.asarray(
        plan.get("grasp_geometry", {}).get(
            "initial_full_relative_vector_right_minus_left_m"
        ),
        dtype=float,
    ).reshape(3)
    if not np.allclose(grasp_right - grasp_left, reference, atol=1e-9):
        raise ValueError("grasp pair does not match its full relative vector")
    lift_height = float(plan.get("safety", {}).get("lift_height_m"))
    if not np.isclose(lift_height, 0.08, atol=1e-12):
        raise ValueError("eighth lift height must remain exactly 0.08m")
    expected_lift_delta = np.array([0.0, 0.0, lift_height])
    if not (
        np.allclose(lift_left - grasp_left, expected_lift_delta, atol=1e-9)
        and np.allclose(lift_right - grasp_right, expected_lift_delta, atol=1e-9)
    ):
        raise ValueError("both grasp points must lift vertically by 0.08m")
    expected_pullback_delta = np.array([-0.02, 0.0, 0.0])
    if not (
        np.allclose(
            pullback_left - lift_left, expected_pullback_delta, atol=1e-9
        )
        and np.allclose(
            pullback_right - lift_right, expected_pullback_delta, atol=1e-9
        )
        and np.isclose(
            float(plan["safety"].get("post_lift_pullback_m")),
            0.02,
            atol=1e-12,
        )
        and plan["safety"].get("post_lift_pullback_direction_base")
        == "negative_x_robotward"
    ):
        raise ValueError("both arms must pull back 20mm after the first lift")
    left_translation = center_left - pullback_left
    right_translation = center_right - pullback_right
    if not np.allclose(left_translation, right_translation, atol=1e-9):
        raise ValueError("center translation must move both arms by one vector")
    if not np.isclose(left_translation[2], 0.0, atol=1e-9):
        raise ValueError("center translation must retain the lifted release height")
    if not (
        np.allclose(lift_right - lift_left, reference, atol=1e-9)
        and np.allclose(
            pullback_right - pullback_left, reference, atol=1e-9
        )
        and np.allclose(center_right - center_left, reference, atol=1e-9)
    ):
        raise ValueError("translation must preserve the complete relative vector")
    first_center = np.asarray(plan.get("first_center_base_m"), dtype=float).reshape(3)
    if not np.isfinite(first_center).all() or not np.allclose(
        ((center_left + center_right) / 2.0)[:2],
        first_center[:2],
        atol=1e-9,
    ):
        raise ValueError("translated pair midpoint must reach the original center")
    if (
        plan.get("grasp_geometry", {}).get("translation_destination")
        != "original_towel_center"
        or plan.get("safety", {}).get("release_destination")
        != "original_towel_center"
        or plan.get("safety", {}).get("translation_reachability_policy")
        != "farthest_certified_waypoint_if_endpoint_unreachable"
    ):
        raise ValueError("eighth release destination policy is immutable")
    release_stage = by_id["release_retract"]
    release_left = np.asarray(
        release_stage.get("release_left_fingertip_position_base_m"), dtype=float
    ).reshape(3)
    release_right = np.asarray(
        release_stage.get("release_right_fingertip_position_base_m"), dtype=float
    ).reshape(3)
    if not (
        np.allclose(release_left, center_left, atol=1e-9)
        and np.allclose(release_right, center_right, atol=1e-9)
    ):
        raise ValueError("release must occur at the lifted center-translation targets")
    retract_left, retract_right = pair("release_retract")
    if not (
        np.allclose(retract_left, release_left, atol=1e-9)
        and np.allclose(retract_right, release_right, atol=1e-9)
        and release_stage.get("motion_policy") == "skip_after_release_direct_home"
        and plan.get("safety", {}).get("post_release_policy")
        == "skip_retract_direct_home"
    ):
        raise ValueError("release must skip retraction and enter home")
    home = by_id["home"]
    if (
        home.get("target") != "session_initial_joint_state"
        or home.get("synchronization") != "shared_joint_progress"
    ):
        raise ValueError("both-arm home policy changed")
    recorded_translation = np.asarray(
        plan.get("grasp_geometry", {}).get("center_translation_base_m"),
        dtype=float,
    ).reshape(3)
    recorded_midpoint = np.asarray(
        plan.get("grasp_geometry", {}).get("grasp_pair_midpoint_base_m"),
        dtype=float,
    ).reshape(3)
    if not (
        np.allclose(recorded_translation, left_translation, atol=1e-9)
        and np.allclose(recorded_midpoint, (grasp_left + grasp_right) / 2.0, atol=1e-9)
    ):
        raise ValueError("recorded center translation geometry is inconsistent")
    safety = plan.get("safety", {})
    if not np.isclose(
        float(safety.get("release_height_above_grasp_m")), 0.08, atol=1e-12
    ):
        raise ValueError("release height must remain exactly 0.08m above grasp")
    if not np.isclose(
        float(safety.get("maximum_shared_cartesian_step_m")), 0.05, atol=1e-12
    ):
        raise ValueError("shared Cartesian step must remain exactly 0.05m")
    gate = safety.get("online_relative_geometry_gate", {})
    required_gate = {
        "enabled": True,
        "check_each_shared_cartesian_step": True,
        "preserve_full_xyz_relative_vector": True,
        "reference": "initial_full_relative_vector_right_minus_left_m",
        "failure_policy": "abort_both_arms",
    }
    if any(gate.get(key) != value for key, value in required_gate.items()):
        raise ValueError("online relative geometry gate is incomplete")
    tolerance = float(gate.get("maximum_vector_error_m"))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("online relative geometry gate tolerance is invalid")
    collision_policy = {
        "collision_policy": "first_phase_collision_disabled",
        "planning_collision_checks_enabled": False,
        "certification_collision_checks_enabled": False,
        "execution_collision_checks_enabled": False,
        "collision_checked_stage_ids": [],
        "dual_arm_collision_gate_required": False,
        "second_phase_perception_policy": "fresh_head_camera_reperception",
    }
    if any(safety.get(key) != value for key, value in collision_policy.items()):
        raise ValueError(
            "eighth first phase collision policy changed"
        )
    if safety.get("joint_command_progress_steps") != 5:
        raise ValueError("eighth joint commands must use exactly 5 progress steps")
    refinement = safety.get("grasp_refinement_policy", {})
    if refinement != {
        "enabled": True,
        "maximum_iterations": 20,
        "individual_tcp_tolerance_m": 0.015,
        "initial_seed_policy": (
            "best_joint_bounded_candidate_without_ik_fk_acceptance"
        ),
        "initial_seed_ik_fk_acceptance_required": False,
        "initial_seed_maximum_total_joint_delta_rad": 3.0,
        "initial_seed_maximum_substep_joint_delta_rad": 1.0,
        "gripper_close_requires_both_tcp_within_tolerance": False,
        "completion_policy": "force_continue_after_maximum_iterations",
        "post_refinement_tcp_error_gate_required": False,
        "no_local_optimum_early_exit": True,
        "strategy": "independent_preclose_worst_error_first",
        "synchronized_pair_only": False,
        "inactive_arm_holds": True,
        "final_relative_geometry_gate_required": False,
    }:
        raise ValueError("eighth grasp refinement policy is immutable")
    if safety.get("contact_probe_policy") != {
        "enabled": True,
        "trigger": "after_grasp_before_gripper_close",
        "probe_order": ["right", "left"],
        "axis_base": [0.0, 0.0, -1.0],
        "step_m": 0.0005,
        "maximum_descent_m": 0.04,
        "sample_count": 40,
        "joint_indices_zero_based": [1, 2],
        "residual_reference_mode": "probe_start_fixed",
        "sum_abs_residual_threshold_nm": 6.0,
        "single_joint_emergency_threshold_nm": 5.0,
        "maximum_median_abs_velocity_rad_s": 0.05,
        "confirmation_samples": 2,
        "freeze_on_first_contact_indication": True,
        "stationary_confirmation_maximum_attempts": 12,
        "ik_probes_per_step": 8,
        "maximum_fk_position_error_m": 0.015,
        "maximum_incremental_joint_delta_rad": 0.25,
        "both_arms_contact_required_before_close": True,
        "failure_policy": "retract_both_to_preprobe_and_abort",
        "table_collision_policy": "bounded_intentional_contact_probe",
        "execution_grasp_table_gate_bypassed": True,
        "lock_xy_to_grasp_target": True,
        "threshold_source": "restored_dual_arm_fixed_baseline_20260812",
    }:
        raise ValueError("eighth contact probe policy is immutable")
    transition = safety.get("ik_branch_transition", {})
    if (
        transition.get("enabled") is not True
        or transition.get("source")
        != "historical_certified_pregrasp_branch_seed"
        or len(transition.get("left_joints_rad", [])) != 6
        or len(transition.get("right_joints_rad", [])) != 6
        or float(transition.get("maximum_transition_joint_delta_rad", 0.0))
        != 3.0
        or float(
            transition.get(
                "maximum_transition_to_grasp_joint_delta_rad", 0.0
            )
        )
        != 1.7
        or float(transition.get("initial_grasp_fk_tolerance_m", 0.0))
        != 0.035
        or transition.get("sweep_samples_per_segment") != 0
    ):
        raise ValueError("eighth IK branch transition policy is immutable")
    for side in ("left", "right"):
        joints = np.asarray(
            transition[f"{side}_joints_rad"], dtype=float
        ).reshape(6)
        if not np.isfinite(joints).all():
            raise ValueError("eighth IK branch transition joints are invalid")
    if safety.get("execution_default") != "preview_only":
        raise ValueError("eighth pipeline must default to preview only")
    expected_events = [
        {
            "before": "grasp",
            "action": "set_both_grippers",
            "position": 40.0,
        },
        {"after": "contact_probe", "action": "close_both_grippers"},
        {"before": "release_retract", "action": "open_both_grippers"},
    ]
    if plan.get("gripper_events") != expected_events:
        raise ValueError("eighth gripper event order is immutable")
    if plan.get("safety", {}).get("pregrasp_gripper_position") != 40.0:
        raise ValueError("eighth pregrasp gripper position must remain 40")
    if (
        plan.get("safety", {}).get("grasp_x_tolerance_m") != 0.005
        or plan.get("safety", {}).get("grasp_maximum_inward_x_error_m")
        != 0.005
        or plan.get("safety", {})
        .get("contact_probe_policy", {})
        .get("lock_xy_to_grasp_target")
        is not True
    ):
        raise ValueError("eighth grasp X lock policy is immutable")
    if plan.get("safety", {}).get("translation_control_policy") != {
        "mode": "midpoint_plus_fixed_grasp_relative_vector",
        "reference_vector_source": (
            "initial_full_relative_vector_right_minus_left_m"
        ),
    }:
        raise ValueError("eighth translation midpoint control is immutable")


def build_sequential_session_manifest(
    *,
    original_center_base_m: list[float],
    initial_left_joints_rad: list[float],
    initial_right_joints_rad: list[float],
    torso_joints_rad: list[float],
    right_phase: dict[str, str],
    left_phase: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a canonical self-hashing audit manifest for both phases."""
    manifest: dict[str, Any] = {
        "schema": "capx.r1lite.seventh_sequential_session.v1",
        "original_center_base_m": [float(v) for v in original_center_base_m],
        "initial_state": {
            "left_joints_rad": [float(v) for v in initial_left_joints_rad],
            "right_joints_rad": [float(v) for v in initial_right_joints_rad],
            "torso_joints_rad": [float(v) for v in torso_joints_rad],
        },
        "phase_order": ["right", "left"],
        "right_phase": dict(right_phase),
        "left_phase": None if left_phase is None else dict(left_phase),
    }
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def validate_sequential_session_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "capx.r1lite.seventh_sequential_session.v1":
        raise ValueError("unexpected sequential session manifest schema")
    if manifest.get("phase_order") != ["right", "left"]:
        raise ValueError("sequential session phase order changed")
    recorded = manifest.get("manifest_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("sequential session manifest hash is invalid")
    payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    expected = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if recorded != expected:
        raise ValueError("sequential session manifest hash mismatch")
