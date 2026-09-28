import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from capx.integrations.r1lite.bimanual_kinematics import _segment_distance
from capx.integrations.r1lite.generated_bimanual_policy import (
    dispatch_program,
    execute_dry_run,
    expected_operations,
    validate_program,
)
from capx.integrations.r1lite.kinematics import R1LiteCandidateKinematics
from capx.integrations.r1lite.kinematics import (
    GRIPPER_LINK_TO_TOOL_Z_RPY_RAD,
    GRIPPER_LINK_TO_TOOL_Z_XYZ_M,
)
from capx.integrations.r1lite.towel_perception import (
    extract_towel_geometry,
    keypoint_repeatability_error_m,
)
from capx.integrations.r1lite.towel_plan import build_towel_fold_plan


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/r1lite_seventh_dual_arm_towel_fold.py"
LAUNCHER = REPO / "scripts/run_r1lite_seventh_dual_arm_towel_fold_pipeline.sh"
GENERATOR = REPO / "scripts/r1lite_generate_llm_towel_fold_code.py"
RENDERER = REPO / "scripts/render_r1lite_seventh_towel_fold_html.py"


def calibration(*, left_px=100):
    return {
        "plane_z": 0.80,
        "image_to_base_xy_homography": [
            [0.001, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "points": [
            {"pixel": [left_px, 100], "base_xy": [left_px / 1000, 0.1]},
            {"pixel": [900, 100], "base_xy": [0.9, 0.1]},
            {"pixel": [900, 900], "base_xy": [0.9, 0.9]},
            {"pixel": [left_px, 900], "base_xy": [left_px / 1000, 0.9]},
        ],
        "image_size": [1000, 1000],
    }


def towel_mask():
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.rectangle(mask, (20, 250), (780, 750), 1, thickness=-1)
    return mask.astype(bool)


def geometry():
    return extract_towel_geometry(towel_mask(), calibration())


def plan():
    return build_towel_fold_plan(
        geometry(),
        calibration_sha256="c" * 64,
        first_image_sha256="1" * 64,
        second_image_sha256="2" * 64,
        keypoint_repeatability_m=0.002,
        table_z_m=0.80,
    )


def test_towel_mask_may_cross_calibration_hull_with_bounded_keypoints():
    result = geometry()

    assert result["mask_must_be_inside_calibration_polygon"] is False
    assert result["keypoints"]["left_edge_midpoint"]["extrapolation_m"] == pytest.approx(
        0.08,
        abs=0.003,
    )
    assert result["keypoints"]["center"]["extrapolation_m"] == 0.0
    assert result["keypoints"]["right_edge_midpoint"]["extrapolation_m"] == 0.0

    with pytest.raises(ValueError, match="bounded homography extrapolation"):
        extract_towel_geometry(
            towel_mask(),
            calibration(left_px=300),
            maximum_extrapolation_m=0.15,
        )


def test_rectangle_direction_is_projected_to_real_towel_contour():
    mask = towel_mask().astype(np.uint8)
    mask[400:601, 700:781] = 0
    result = extract_towel_geometry(mask.astype(bool), calibration())

    rectangle_right = np.asarray(
        result["rectangle_edge_midpoints_pixel_uv"]["right"]
    )
    projected_right = np.asarray(
        result["keypoints"]["right_edge_midpoint"]["pixel_uv"]
    )
    assert result["edge_midpoint_method"] == (
        "rectangle_direction_then_nearest_contour_projection"
    )
    assert rectangle_right[0] == pytest.approx(780.0, abs=1.0)
    assert projected_right[0] <= 700.0
    assert result["contour_projection_distance_px"]["right"] >= 79.0


def test_right_side_candidates_cover_only_real_side_and_exclude_corners():
    result = geometry()
    candidates = result["right_side_grasp_candidates"]
    pixels = np.asarray([item["pixel_uv"] for item in candidates], dtype=float)

    assert len(candidates) == 24
    assert np.allclose(pixels[:, 0], 780.0, atol=1.0)
    assert float(np.min(pixels[:, 1])) >= 260.0 - 1.0
    assert float(np.max(pixels[:, 1])) <= 740.0 + 1.0
    contour_points = np.asarray(
        [item["contour_base_xyz_m"] for item in candidates], dtype=float
    )
    contact_points = np.asarray(
        [item["base_xyz_m"] for item in candidates], dtype=float
    )
    assert np.linalg.norm(contact_points[:, :2] - contour_points[:, :2], axis=1) == pytest.approx(
        np.full(len(candidates), 0.005)
    )
    assert result["right_side_contour_policy"] == {
        "definition": (
            "fitted_right_side_contour_with_inset_finger_contact_center"
        ),
        "corner_exclusion_m": 0.01,
        "sample_spacing_m": 0.01,
        "maximum_candidates": 24,
        "side_length_m": pytest.approx(0.5),
        "finger_contact_width_m": 0.01,
        "contact_center_inset_m": 0.005,
    }


def test_towel_geometry_repeatability_uses_all_three_keypoints():
    first = geometry()
    second = geometry()
    second["keypoints"]["right_edge_midpoint"]["base_xyz_m"][0] += 0.012

    assert keypoint_repeatability_error_m(first, second) == pytest.approx(0.012)


def test_plan_uses_two_cm_fingertip_gap_with_mirrored_outward_tilt():
    result = plan()

    assert [item["id"] for item in result["stages"]] == [
        "pregrasp",
        "grasp",
        "lift",
        "fold_center",
    ]
    assert all(
        item["orientation"] == "topdown_or_multi_direction_tilt"
        for item in result["stages"][:3]
    )
    assert all(
        item["tilt_search_deg"] == [0.0, 15.0, 30.0, 45.0]
        for item in result["stages"][:3]
    )
    assert all(
        "left_fingertip_position_base_m" in item
        and "right_fingertip_position_base_m" in item
        for item in result["stages"][:3]
    )
    assert result["stages"][-1]["orientation"] == "mirrored_outward_tilt"
    assert result["stages"][-1]["nominal_outward_tilt_deg"] == pytest.approx(45.0)
    assert result["stages"][-1]["tilt_search_deg"] == [45.0, 50.0, 55.0, 60.0]
    assert all(
        item["synchronization"] == "shared_progress_barrier"
        for item in result["stages"]
    )
    fold = result["stages"][-1]
    fingertip_separation = np.linalg.norm(
        np.asarray(fold["left_fingertip_position_base_m"])
        - np.asarray(fold["right_fingertip_position_base_m"])
    )
    tcp_separation = np.linalg.norm(
        np.asarray(fold["left_tcp_position_base_m"])
        - np.asarray(fold["right_tcp_position_base_m"])
    )
    assert fingertip_separation == pytest.approx(0.02)
    assert fold["left_fingertip_position_base_m"][2] == pytest.approx(0.82)
    assert fold["right_fingertip_position_base_m"][2] == pytest.approx(0.82)
    assert tcp_separation == pytest.approx(0.13030865786510142)
    assert result["tcp"]["left"] == "left_gripper_tool_z"
    assert result["tcp"]["left_definition"]["xyz_m"] == [0.0, 0.0, 0.0]
    assert result["tcp"]["left_definition"]["rpy_rad"] == pytest.approx(
        [0.0, np.pi / 2.0, 0.0]
    )
    assert result["tcp"]["left_definition"]["tool_positive_z"] == (
        "gripper_physical_forward"
    )
    assert result["safety"]["left_tcp_physical_pivot_calibration_required"] is False
    assert result["safety"]["static_gripper_envelope_gate_passed"] is True
    assert result["safety"]["minimum_outward_tilt_deg"] == pytest.approx(
        42.563,
        abs=0.01,
    )
    assert result["safety"]["minimum_gripper_center_separation_with_margin_m"] == (
        pytest.approx(0.12552056969962836)
    )
    assert result["safety"]["execution_default"] == "preview_only"
    assert result["safety"]["fold_fingertip_clearance_above_table_m"] == (
        pytest.approx(0.02)
    )
    refinement = result["safety"]["cartesian_refinement_policy"]
    assert refinement["source"] == "pipeline6_vertex_refinement"
    assert refinement["applies_to"] == "both_arms_all_four_stages"
    assert refinement["feedback_source"] == (
        "physical_tool_z_pose_after_each_motion"
    )
    assert refinement["target_tolerance_m"] == pytest.approx(0.015)
    assert refinement["max_iterations"] == 20
    assert refinement["ik_probes_per_iteration"] == 3
    assert refinement["minimum_improvement_m"] == pytest.approx(0.001)
    assert refinement["local_optimum_stall_iterations"] == 2
    assert refinement["joint_arrival_max_retries"] == 2
    assert refinement["paired_swept_collision_gate_each_iteration"] is True
    assert refinement["shared_progress_barrier_each_iteration"] is True


def test_llm_program_is_complete_plan_bound_and_restricted():
    immutable_plan = plan()
    plan_sha = hashlib.sha256(repr(immutable_plan).encode()).hexdigest()
    expected = expected_operations(immutable_plan, plan_sha256=plan_sha)
    code = "\n".join(
        f"{item['call']}("
        + ", ".join(
            f"{key}={value!r}" for key, value in item.items() if key != "call"
        )
        + ")"
        for item in expected
    )

    assert execute_dry_run(code, expected=expected) == expected
    trace = []

    class Runtime:
        def __getattr__(self, name):
            def record(**kwargs):
                trace.append({"call": name, **kwargs})

            return record

    assert dispatch_program(code, expected=expected, runtime=Runtime()) == expected
    assert trace == expected
    with pytest.raises(ValueError):
        validate_program("import os\n" + code, expected=expected)
    with pytest.raises(ValueError):
        validate_program(code.replace("fold_center", "unsafe"), expected=expected)


def test_left_fk_is_parameterized_and_joint_limits_are_enforced():
    left = R1LiteCandidateKinematics(side="left", enforce_joint_limits=True)
    result = left.fk(
        np.array([0.0, 0.1, -0.1, 0.0, 0.0, 0.0]),
        np.zeros(3),
    )

    assert result.position_base_m.shape == (3,)
    assert result.dual_position_error_m < 1e-9
    assert result.dual_orientation_error_rad < 1e-9
    with pytest.raises(ValueError, match="joint limit"):
        left.fk(np.array([9.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3))


def test_left_tool_z_matches_right_transform_and_points_gripper_forward():
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler(
        "xyz",
        GRIPPER_LINK_TO_TOOL_Z_RPY_RAD,
    ).as_matrix()

    assert GRIPPER_LINK_TO_TOOL_Z_XYZ_M == pytest.approx([0.0, 0.0, 0.0])
    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx(
        [1.0, 0.0, 0.0],
        abs=1e-9,
    )


def test_segment_distance_detects_crossing_and_separated_links():
    assert _segment_distance(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.5, -1.0, 0.0]),
        np.array([0.5, 1.0, 0.0]),
    ) == pytest.approx(0.0)
    assert _segment_distance(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.2, 0.0]),
        np.array([1.0, 0.2, 0.0]),
    ) == pytest.approx(0.2)


def test_launcher_requires_certified_immutable_artifacts_for_execution():
    launcher = LAUNCHER.read_text()
    script = SCRIPT.read_text()
    generator = GENERATOR.read_text()
    renderer = RENDERER.read_text()

    assert 'SEVENTH_PIPELINE_EXECUTE_ROBOT="${SEVENTH_PIPELINE_EXECUTE_ROBOT:-0}"' in launcher
    assert "APPROVED_BUNDLE_PATH" in launcher
    assert "CAPABILITY_CERTIFICATE_PATH" in launcher
    assert "--expected-capability-certificate-sha256" in launcher
    assert "maximum_keypoint_extrapolation_m=0.15" in launcher
    assert "final_fingertip_center_separation_m=0.02" in launcher
    assert "mirrored_outward_tilt_search_45_to_60_deg" in launcher
    assert 'env_cfg["r1lite_motion_validation_enabled"] = True' in launcher
    assert "robot_motion_commands_sent" in script
    assert "left_tcp_model_accepted" in script
    assert "execute_approved_bimanual_program" in script
    assert "dispatch_program" in script
    assert "generated_bimanual_policy" in generator
    assert "complete task-level Python program" in generator
    assert "render_r1lite_seventh_towel_fold_html.py" in launcher
    assert 'trajectory("Left TCP target"' in renderer
    assert 'trajectory("Right TCP target"' in renderer
