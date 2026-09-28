import cv2
import numpy as np
import pytest

from capx.integrations.r1lite.towel_perception import (
    extract_nearest_towel_corner_grasp_points,
)
from capx.integrations.r1lite.towel_plan import (
    EIGHTH_BIMANUAL_RECENTER_SCHEMA,
    EIGHTH_BIMANUAL_STAGE_ORDER,
    SEQUENTIAL_SINGLE_ARM_SCHEMA,
    build_eighth_bimanual_recenter_plan,
    validate_eighth_bimanual_recenter_plan,
)


def calibration():
    return {
        "plane_z": 0.80,
        "image_to_base_xy_homography": [
            [0.001, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "points": [
            {"pixel": [0, 0], "base_xy": [0.0, 0.0]},
            {"pixel": [999, 0], "base_xy": [0.999, 0.0]},
            {"pixel": [999, 999], "base_xy": [0.999, 0.999]},
            {"pixel": [0, 999], "base_xy": [0.0, 0.999]},
        ],
    }


def irregular_towel_mask():
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    polygon = np.array(
        [[180, 180], [820, 210], [790, 790], [540, 850], [200, 780]]
    )
    cv2.fillPoly(mask, [polygon], 1)
    mask[600:690, 650:791] = 0
    return mask.astype(bool)


def geometry():
    return extract_nearest_towel_corner_grasp_points(
        irregular_towel_mask(),
        calibration(),
    )


def build():
    return build_eighth_bimanual_recenter_plan(
        geometry(),
        table_z_m=0.80,
    )


def test_two_nearest_real_towel_corners_are_selected_and_ordered():
    result = geometry()
    left = np.asarray(result["keypoints"]["left_grasp"]["pixel_uv"])
    right = np.asarray(result["keypoints"]["right_grasp"]["pixel_uv"])

    contours, _ = cv2.findContours(
        result["selected_mask"].astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    contour_pixels = {
        tuple(point) for point in max(contours, key=cv2.contourArea).reshape(-1, 2)
    }
    assert left[0] < right[0]
    assert tuple(left.astype(int)) in contour_pixels
    assert tuple(right.astype(int)) in contour_pixels
    assert result["points_on_real_external_contour"] is True
    assert result["grasp_point_method"] == (
        "nearest_robot_two_projected_rectangle_corners"
    )
    corners = result["corner_candidates"]
    assert len(corners) == 4
    selected = [item for item in corners if item["selected"]]
    assert len(selected) == 2
    distances = sorted(item["robot_xy_distance_m"] for item in corners)
    assert sorted(item["robot_xy_distance_m"] for item in selected) == pytest.approx(
        distances[:2]
    )
    contour_pixels = np.asarray(result["external_contour_pixel_uv"])
    contour_base = np.asarray(result["external_contour_base_xyz_m"])
    assert contour_pixels.ndim == 2 and contour_pixels.shape[1] == 2
    assert contour_base.shape == (len(contour_pixels), 3)
    assert len(contour_pixels) <= 500

    quality = result["quality"]
    assert quality["pair_separation_px"] == pytest.approx(
        np.linalg.norm(right - left)
    )
    assert quality["rectangularity"] >= 0.50
    assert quality["selection_margin_to_third_corner_m"] >= 0.0
    assert quality["left_contour_distance_px"] == 0.0
    assert quality["right_contour_distance_px"] == 0.0
    assert 0.0 <= quality["quality_score"] <= 1.0


def test_recenter_plan_preserves_full_pair_vector_and_releases_at_lift_height():
    result = build()
    assert result["schema"] == EIGHTH_BIMANUAL_RECENTER_SCHEMA
    assert len(result["perception"]["external_contour_base_xyz_m"]) >= 4
    assert [stage["id"] for stage in result["stages"]] == list(
        EIGHTH_BIMANUAL_STAGE_ORDER
    )
    by_id = {stage["id"]: stage for stage in result["stages"]}

    def pair(stage_id):
        stage = by_id[stage_id]
        return (
            np.asarray(stage["left_fingertip_position_base_m"]),
            np.asarray(stage["right_fingertip_position_base_m"]),
        )

    grasp_left, grasp_right = pair("grasp")
    detected_left = np.asarray(
        result["perception"]["keypoints"]["left_grasp"]["base_xyz_m"]
    )
    detected_center = np.asarray(
        result["perception"]["keypoints"]["center"]["base_xyz_m"]
    )
    detected_right = np.asarray(
        result["perception"]["keypoints"]["right_grasp"]["base_xyz_m"]
    )
    assert grasp_left[0] == pytest.approx(detected_left[0])
    assert grasp_left[1] - detected_left[1] == pytest.approx(
        np.copysign(0.015, detected_center[1] - detected_left[1])
    )
    assert result["safety"]["left_grasp_centerward_y_offset_m"] == pytest.approx(
        0.015
    )
    assert grasp_right[0] - detected_right[0] == pytest.approx(-0.005)
    assert grasp_right[1] == pytest.approx(detected_right[1])
    assert result["safety"]["right_grasp_robotward_x_offset_m"] == pytest.approx(
        -0.005
    )
    lift_left, lift_right = pair("lift")
    pullback_left, pullback_right = pair("pullback")
    center_left, center_right = pair("translate_center")
    reference = grasp_right - grasp_left
    assert lift_left - grasp_left == pytest.approx([0.0, 0.0, 0.08])
    assert lift_right - grasp_right == pytest.approx([0.0, 0.0, 0.08])
    assert lift_right - lift_left == pytest.approx(reference)
    assert pullback_left - lift_left == pytest.approx([-0.02, 0.0, 0.0])
    assert pullback_right - lift_right == pytest.approx([-0.02, 0.0, 0.0])
    assert pullback_right - pullback_left == pytest.approx(reference)
    assert center_right - center_left == pytest.approx(reference)
    assert center_left - pullback_left == pytest.approx(
        center_right - pullback_right
    )
    assert ((center_left + center_right) / 2.0)[:2] == pytest.approx(
        np.asarray(result["first_center_base_m"])[:2]
    )
    assert result["safety"]["post_lift_pullback_m"] == pytest.approx(0.02)
    assert result["safety"]["post_lift_pullback_direction_base"] == (
        "negative_x_robotward"
    )
    assert result["safety"]["release_destination"] == "original_towel_center"
    release = by_id["release_retract"]
    assert release["release_left_fingertip_position_base_m"] == pytest.approx(
        center_left
    )
    assert release["release_right_fingertip_position_base_m"] == pytest.approx(
        center_right
    )
    assert release["left_fingertip_position_base_m"] == pytest.approx(center_left)
    assert release["right_fingertip_position_base_m"] == pytest.approx(center_right)
    assert release["motion_policy"] == "skip_after_release_direct_home"
    assert result["safety"]["post_release_policy"] == "skip_retract_direct_home"
    assert result["safety"]["lift_height_m"] == pytest.approx(0.08)
    assert result["safety"]["maximum_shared_cartesian_step_m"] == pytest.approx(0.05)
    assert result["safety"]["joint_command_progress_steps"] == 5
    assert result["safety"]["collision_policy"] == (
        "first_phase_collision_disabled"
    )
    assert result["safety"]["planning_collision_checks_enabled"] is False
    assert result["safety"]["certification_collision_checks_enabled"] is False
    assert result["safety"]["execution_collision_checks_enabled"] is False
    assert result["safety"]["collision_checked_stage_ids"] == []
    assert result["safety"]["dual_arm_collision_gate_required"] is False
    assert result["safety"]["second_phase_perception_policy"] == (
        "fresh_head_camera_reperception"
    )
    assert result["safety"]["grasp_refinement_policy"][
        "initial_seed_policy"
    ] == "best_joint_bounded_candidate_without_ik_fk_acceptance"
    assert result["safety"]["grasp_refinement_policy"][
        "initial_seed_ik_fk_acceptance_required"
    ] is False
    assert result["safety"]["grasp_refinement_policy"][
        "initial_seed_maximum_total_joint_delta_rad"
    ] == pytest.approx(3.0)
    assert result["safety"]["grasp_refinement_policy"][
        "initial_seed_maximum_substep_joint_delta_rad"
    ] == pytest.approx(1.0)
    assert result["safety"]["grasp_refinement_policy"][
        "gripper_close_requires_both_tcp_within_tolerance"
    ] is False
    assert result["safety"]["grasp_refinement_policy"][
        "completion_policy"
    ] == "force_continue_after_maximum_iterations"
    assert result["safety"]["grasp_refinement_policy"][
        "post_refinement_tcp_error_gate_required"
    ] is False
    assert result["safety"]["grasp_refinement_policy"][
        "final_relative_geometry_gate_required"
    ] is False
    assert result["safety"]["online_relative_geometry_gate"] == {
        "enabled": True,
        "check_each_shared_cartesian_step": True,
        "preserve_full_xyz_relative_vector": True,
        "reference": "initial_full_relative_vector_right_minus_left_m",
        "maximum_vector_error_m": 0.005,
        "failure_policy": "abort_both_arms",
    }
    assert by_id["grasp"]["orientation"] == "strict_topdown"
    for stage_id in ("lift", "pullback", "translate_center", "release_retract"):
        assert (
            by_id[stage_id]["orientation"]
            == "shared_world_y_tilt_minus_30deg"
        )
    assert by_id["home"]["target"] == "session_initial_joint_state"
    assert by_id["home"]["synchronization"] == "shared_joint_progress"


def test_eighth_validator_fails_closed_without_changing_seventh_v34():
    assert SEQUENTIAL_SINGLE_ARM_SCHEMA == (
        "capx.r1lite.seventh_sequential_single_arm.v34"
    )
    result = build()
    next(
        stage for stage in result["stages"]
        if stage["id"] == "translate_center"
    )["right_fingertip_position_base_m"][2] += 0.01
    with pytest.raises(ValueError):
        validate_eighth_bimanual_recenter_plan(result)

    result = build()
    result["safety"]["maximum_shared_cartesian_step_m"] = 0.051
    with pytest.raises(ValueError, match="Cartesian step"):
        validate_eighth_bimanual_recenter_plan(result)

    result = build()
    result["perception"]["keypoints"]["left_grasp"]["pixel_uv"][0] = (
        result["perception"]["keypoints"]["right_grasp"]["pixel_uv"][0] + 1.0
    )
    with pytest.raises(ValueError, match="ordered"):
        validate_eighth_bimanual_recenter_plan(result)
