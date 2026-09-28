import hashlib
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from capx.integrations.r1lite.bimanual_kinematics import (
    R1LiteTableCollisionKinematics,
    strict_topdown_edge_rotation,
    tcp_position_for_fingertip,
)
from capx.integrations.r1lite.generated_bimanual_policy import (
    execute_dry_run,
    expected_operations,
    validate_program,
)
from capx.integrations.r1lite.towel_plan import (
    SEQUENTIAL_SINGLE_ARM_SCHEMA,
    SINGLE_ARM_STAGE_ORDER,
    build_sequential_session_manifest,
    build_sequential_single_arm_plan,
    rebuild_sequential_single_arm_plan_for_candidate,
    validate_sequential_session_manifest,
    validate_sequential_single_arm_plan,
)


def geometry(left=(0.45, 0.20, 0.80), right=(0.45, -0.20, 0.80)):
    return {
        "edge_midpoint_method": (
            "rectangle_direction_then_nearest_contour_projection"
        ),
        "contour_projection_distance_px": {"left": 0.0, "right": 0.0},
        "keypoints": {
            "center": {"base_xyz_m": [0.45, 0.0, 0.80]},
            "left_edge_midpoint": {"base_xyz_m": list(left)},
            "right_edge_midpoint": {"base_xyz_m": list(right)},
        }
    }


def build(side, *, center=(0.45, 0.0, 0.80), value=None):
    data = geometry() if value is None else value
    return build_sequential_single_arm_plan(
        data,
        side=side,
        original_center_base_m=list(center),
        calibration_sha256="c" * 64,
        first_image_sha256="1" * 64,
        second_image_sha256="2" * 64,
        keypoint_repeatability_m=0.001,
        table_z_m=0.80,
    )


def test_right_then_redetected_left_preserves_original_center():
    right = build("right")
    moved = geometry(left=(0.43, 0.17, 0.80))
    left = build(
        "left",
        center=tuple(right["original_center_base_m"]),
        value=moved,
    )

    assert [stage["id"] for stage in right["stages"]] == list(
        SINGLE_ARM_STAGE_ORDER
    )
    assert left["perception"]["selected_keypoint_base_m"] == pytest.approx(
        [0.430584206237837, 0.16503424697838562, 0.80]
    )
    assert left["original_center_base_m"] == right["original_center_base_m"]
    place = next(stage for stage in left["stages"] if stage["id"] == "place_center")
    assert place["active_fingertip_position_base_m"][:2] == pytest.approx(
        right["original_center_base_m"][:2]
    )
    assert place["active_fingertip_position_base_m"][2] == pytest.approx(0.85)
    assert left["safety"]["place_release_clearance_m"] == pytest.approx(0.05)
    assert all(
        stage["inactive_arm_policy"] == "hold_session_initial_joints"
        for stage in left["stages"]
    )
    assert left["safety"]["collision_policy"] == "active_arm_table_only"
    assert left["safety"]["post_grasp_closed_lift_collision_policy"] == (
        "skip_table_collision_if_gripper_close_succeeded"
    )
    assert left["safety"]["post_grasp_close_dwell_s"] == pytest.approx(1.0)
    assert left["safety"]["lift_height_m"] == pytest.approx(0.05)
    grasp = next(stage for stage in left["stages"] if stage["id"] == "grasp")
    lift = next(stage for stage in left["stages"] if stage["id"] == "lift")
    assert (
        np.asarray(lift["active_tcp_position_base_m"])
        - np.asarray(grasp["active_tcp_position_base_m"])
    ) == pytest.approx([0.0, 0.0, 0.05])
    assert (
        left["safety"]["gripper_table_collision_model"]
        == "two_fully_open_finger_mesh_aabbs"
    )
    assert left["safety"]["gripper_finger_open_travel_m"] == pytest.approx(0.05)
    assert left["safety"]["grasp_open_finger_table_margin_m"] == pytest.approx(
        0.001
    )
    assert left["safety"]["grasp_allowed_table_contact_link"] in (
        "finger_link1",
        "finger_link2",
    )
    assert left["safety"]["grasp_designated_contact_min_clearance_m"] == (
        pytest.approx(-0.0002)
    )
    assert left["safety"]["grasp_noncontact_min_clearance_m"] == pytest.approx(
        0.001
    )
    assert left["safety"]["grasp_contact_stage_only"] is True
    assert left["safety"]["grasp_effort_contact_policy"] == {
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
    assert left["safety"]["grasp_post_arrival_probe_policy"] == {
        "enabled": True,
        "start_condition": "grasp_xy_within_10mm_and_z_error_below_5mm",
        "maximum_entry_z_error_m": 0.005,
        "direction_base": [0.0, 0.0, -1.0],
        "initial_descent_m": 0.010,
        "absolute_height_refinement_iterations": 20,
        "absolute_contact_tcp_z_base_m": pytest.approx(
            left["stages"][2]["active_tcp_position_base_m"][2]
            - (
                left["grasp_geometry"]["reference_topdown_lowest_point_z_m"]
                - left["safety"]["table_z_m"]
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
    }
    refinement = left["safety"]["cartesian_refinement_policy"]
    assert refinement["grasp_xy_tolerance_m"] == pytest.approx(0.010)
    assert refinement["grasp_z_tolerance_m"] == pytest.approx(0.0005)
    assert left["safety"]["dual_arm_collision_gate_required"] is False
    assert right["schema"] == SEQUENTIAL_SINGLE_ARM_SCHEMA
    assert SINGLE_ARM_STAGE_ORDER[:3] == ("pregrasp", "align_grasp", "grasp")


def test_grasp_alignment_preserves_edge_normal_while_searching_tilt():
    plan = build("right")
    normal = np.asarray(plan["grasp_geometry"]["edge_inward_normal_base_xy"])
    assert normal == pytest.approx([0.0, 1.0])
    strict_stages = {}
    for stage_id in ("align_grasp", "grasp"):
        stage = next(stage for stage in plan["stages"] if stage["id"] == stage_id)
        strict_stages[stage_id] = stage
        assert stage["orientation"] == "edge_normal_constrained_tilt"
        assert stage["tilt_search_deg"] == [
            0.0,
            -15.0,
            15.0,
            -30.0,
            30.0,
            -45.0,
            45.0,
        ]
        assert stage["jaw_closing_axis_tool"] == "tool_y"
        assert stage["jaw_closing_axis_base_xy"] == pytest.approx(normal)
    align = np.asarray(strict_stages["align_grasp"]["active_fingertip_position_base_m"])
    grasp = np.asarray(strict_stages["grasp"]["active_fingertip_position_base_m"])
    assert align[:2] == pytest.approx(grasp[:2])
    assert align[2] - grasp[2] == pytest.approx(0.05)

    tampered = build("right")
    next(
        stage for stage in tampered["stages"] if stage["id"] == "grasp"
    )["tilt_search_deg"] = [60.0]
    with pytest.raises(ValueError, match="unbound tilt search"):
        validate_sequential_single_arm_plan(tampered)


def test_selected_contour_candidate_rebuilds_all_grasp_dependent_stages():
    value = geometry()
    value["right_side_contour_policy"] = {
        "definition": (
            "fitted_right_side_contour_with_inset_finger_contact_center"
        ),
        "corner_exclusion_m": 0.01,
        "sample_spacing_m": 0.01,
        "maximum_candidates": 24,
        "finger_contact_width_m": 0.01,
        "contact_center_inset_m": 0.005,
    }
    value["right_side_grasp_candidates"] = [
        {
            "id": "right_side_00",
            "pixel_uv": [700.0, 300.0],
            "base_xyz_m": [0.44, -0.24, 0.80],
            "arc_distance_from_first_corner_m": 0.01,
            "extrapolation_m": 0.0,
            "two_frame_repeatability_m": 0.001,
        },
        {
            "id": "right_side_01",
            "pixel_uv": [700.0, 400.0],
            "base_xyz_m": [0.48, -0.22, 0.80],
            "arc_distance_from_first_corner_m": 0.02,
            "extrapolation_m": 0.0,
            "two_frame_repeatability_m": 0.001,
        },
    ]
    source = build("right", value=value)
    selected = rebuild_sequential_single_arm_plan_for_candidate(
        source,
        candidate_id="right_side_00",
        selected_grasp_tilt_deg=30.0,
    )

    assert selected["grasp_geometry"]["selected_candidate_id"] == "right_side_00"
    assert selected["grasp_geometry"]["selected_tilt_deg"] == 30.0
    assert (
        selected["grasp_geometry"][
            "selected_fingertip_center_height_compensation_m"
        ]
        > 0.005
    )
    assert selected["grasp_geometry"]["reference_topdown_lowest_point_z_m"] == (
        pytest.approx(
            source["grasp_geometry"]["reference_topdown_lowest_point_z_m"]
        )
    )
    assert next(
        stage for stage in selected["stages"] if stage["id"] == "grasp"
    )["tilt_search_deg"] == [30.0]
    for stage_id in ("pregrasp", "grasp", "lift"):
        stage = next(item for item in selected["stages"] if item["id"] == stage_id)
        assert stage["active_fingertip_position_base_m"][:2] == pytest.approx(
            [0.44, -0.24]
        )
    validate_sequential_single_arm_plan(selected)


def test_tcp_to_fingertip_offset_rotates_with_tool_orientation():
    fingertip = np.array([0.45, -0.20, 0.803])
    length = 0.078
    topdown = np.array([0.0, 1.0, 0.0, 0.0])
    tcp = tcp_position_for_fingertip(fingertip, topdown, length)
    assert tcp == pytest.approx([0.45, -0.20, 0.881])

    tilted_tool_z_along_x = np.array([0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)])
    tilted_tcp = tcp_position_for_fingertip(
        fingertip, tilted_tool_z_along_x, length
    )
    assert tilted_tcp == pytest.approx([0.372, -0.20, 0.803])


def test_strict_topdown_frame_aligns_tool_y_with_towel_edge_normal():
    inward = np.array([0.6, 0.8])
    rotation = strict_topdown_edge_rotation(inward)
    assert rotation[:, 1] == pytest.approx([0.6, 0.8, 0.0])
    assert rotation[:, 2] == pytest.approx([0.0, 0.0, -1.0])
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_old_sequential_schema_is_rejected():
    for version in (
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
        "v9",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v30",
        "v31",
        "v32",
        "v33",
    ):
        plan = build("right")
        plan["schema"] = f"capx.r1lite.seventh_sequential_single_arm.{version}"
        with pytest.raises(ValueError, match="unexpected"):
            validate_sequential_single_arm_plan(plan)


def test_open_finger_mesh_detects_lower_finger_when_gripper_is_tilted():
    safety = R1LiteTableCollisionKinematics(
        active_side="right",
        table_z_m=0.0,
    )
    topdown = Rotation.from_euler("y", 180.0, degrees=True)
    tilted = Rotation.from_euler("x", 30.0, degrees=True) * topdown
    tcp = np.array([0.0, 0.0, 1.0])
    fake_fk = SimpleNamespace(
        position_base_m=tcp,
        quaternion_xyzw=tilted.as_quat(),
    )
    finger_points = safety._open_finger_points_base(fake_fk)
    center_fingertip = (
        tcp + tilted.as_matrix()[:, 2] * safety.tcp_to_fingertip_m
    )
    assert np.min(finger_points[:, 2]) < center_fingertip[2] - 0.02
    assert np.ptp(finger_points[:, 2]) > 0.05


def test_designated_finger_contact_keeps_other_finger_and_arm_clear(monkeypatch):
    from capx.integrations.r1lite.bimanual_kinematics import (
        R1LiteBimanualKinematics,
    )

    safety = R1LiteTableCollisionKinematics(
        active_side="right",
        table_z_m=0.0,
    )
    rotation = (
        Rotation.from_euler("x", -45.0, degrees=True)
        * Rotation.from_euler("y", 180.0, degrees=True)
    )
    orientation = rotation.as_quat()
    probe_fk = SimpleNamespace(
        position_base_m=np.zeros(3),
        quaternion_xyzw=orientation,
    )
    offsets = [
        float(np.min(points[:, 2]))
        for points in safety._finger_points_by_link_base(
            probe_fk, fingers_open=True
        )
    ]
    contact_index = int(np.argmin(offsets))
    contact_link = f"finger_link{contact_index + 1}"
    tcp_z = -offsets[contact_index] - 0.0001
    fake_fk = SimpleNamespace(
        position_base_m=np.array([0.0, 0.0, tcp_z]),
        quaternion_xyzw=orientation,
    )
    safety.kinematics = SimpleNamespace(fk=lambda *_args: fake_fk)
    safety.left = SimpleNamespace(fk=lambda *_args: fake_fk)
    safety.right = SimpleNamespace(fk=lambda *_args: fake_fk)
    monkeypatch.setattr(
        R1LiteBimanualKinematics,
        "_arm_points",
        staticmethod(
            lambda *_args: [np.array([0.0, 0.0, 0.10])] * 7
        ),
    )
    joints = np.zeros(6)
    torso = np.zeros(3)
    with pytest.raises(ValueError, match="table collision"):
        safety.validate(joints, joints, torso)
    report = safety.validate(
        joints,
        joints,
        torso,
        allowed_contact_finger_link=contact_link,
        minimum_contact_clearance_m=-0.0002,
        minimum_noncontact_clearance_m=0.001,
    )
    assert report.estimated_link_clearance_m == pytest.approx(-0.0001)


def test_sequential_llm_policy_is_side_bound_and_home_is_mandatory():
    plan = build("right")
    digest = hashlib.sha256(repr(plan).encode()).hexdigest()
    expected = expected_operations(plan, plan_sha256=digest)
    code = "\n".join(
        f"{item['call']}("
        + ", ".join(
            f"{key}={value!r}" for key, value in item.items() if key != "call"
        )
        + ")"
        for item in expected
    )

    assert expected[-1] == {
        "call": "execute_single_arm_stage",
        "side": "right",
        "stage_id": "home",
    }
    align_index = expected.index(
        {
            "call": "execute_single_arm_stage",
            "side": "right",
            "stage_id": "align_grasp",
        }
    )
    grasp_index = expected.index(
        {
            "call": "execute_single_arm_stage",
            "side": "right",
            "stage_id": "grasp",
        }
    )
    assert align_index < grasp_index
    assert execute_dry_run(code, expected=expected) == expected
    with pytest.raises(ValueError):
        validate_program(code.replace("side='right'", "side='left'", 1), expected=expected)
    with pytest.raises(ValueError):
        validate_program("\n".join(code.splitlines()[:-1]), expected=expected)


def test_plan_and_session_manifest_fail_closed_on_center_or_hash_tampering():
    plan = build("left")
    next(
        stage for stage in plan["stages"] if stage["id"] == "place_center"
    )["active_fingertip_position_base_m"][0] += 0.01
    with pytest.raises(ValueError, match="original center"):
        validate_sequential_single_arm_plan(plan)

    manifest = build_sequential_session_manifest(
        original_center_base_m=[0.45, 0.0, 0.80],
        initial_left_joints_rad=[0.0] * 6,
        initial_right_joints_rad=[0.0] * 6,
        torso_joints_rad=[0.0] * 3,
        right_phase={"plan_sha256": "a" * 64},
        left_phase={"plan_sha256": "b" * 64},
    )
    validate_sequential_session_manifest(manifest)
    manifest["original_center_base_m"][0] += 0.01
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_sequential_session_manifest(manifest)


def test_align_transit_may_waive_fk_position_but_grasp_may_not(monkeypatch):
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/r1lite_validate_sequential_towel_phase.py"
    )
    spec = importlib.util.spec_from_file_location("sequential_validator", script)
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(script.parent))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    candidate = {
        "label": "candidate",
        "joints_rad": [0.1] * 6,
        "target_position_base_m": [0.01, 0.0, 0.0],
        "target_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "fk_position_base_m": [0.06, 0.0, 0.0],
        "fk_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "fk_position_error_m": 0.05,
        "fk_orientation_error_rad": 0.3,
        "max_joint_delta_rad": 0.1,
    }
    monkeypatch.setattr(
        module,
        "solve_candidates",
        lambda *args, **kwargs: ([], [candidate]),
    )
    local_fk = SimpleNamespace(
        fk=lambda joints, torso: SimpleNamespace(
            position_base_m=np.zeros(3),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
    )
    safety = SimpleNamespace(
        validate_interpolated_segment=lambda *args, **kwargs: [
            SimpleNamespace(
                estimated_link_clearance_m=0.1,
                tcp_separation_m=0.2,
            )
        ]
    )
    common = {
        "side": "right",
        "robot": object(),
        "local_fk": local_fk,
        "safety": safety,
        "torso": np.zeros(3),
        "reference": np.zeros(6),
        "inactive_side": "left",
        "inactive_joints": np.zeros(6),
        "final_target": (
            "strict",
            np.array([0.01, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ),
    }

    selected, _, _ = module._solve_active_transit(
        stage_id="align_grasp", **common
    )
    assert selected["accepted_via"] == "pre_final_grasp_fk_pose_waiver"
    with pytest.raises(RuntimeError, match="grasp active transit"):
        module._solve_active_transit(stage_id="grasp", **common)


def test_pipeline_seven_orchestrator_is_right_arm_only():
    repository = __file__
    assert repository
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    orchestrator = (
        root / "scripts/r1lite_run_sequential_towel_fold.py"
    ).read_text()
    validator = (
        root / "scripts/r1lite_validate_sequential_towel_phase.py"
    ).read_text()

    assert "right_execute_audit.json" in orchestrator
    assert "left_perception_audit.json" not in orchestrator
    assert "left_capability_certificate.json" not in orchestrator
    assert "left_phase=None" in orchestrator
    assert "home_completed" in orchestrator
    assert "robot_motion_commands_sent" in validator
    assert 'execution_mode="sequential_single_arm"' in validator
    assert "R1LiteTableCollisionKinematics" in validator
    assert "R1LiteBimanualKinematics()" not in validator
    assert "select_and_certify_candidate_plan" in validator
    assert "candidate_terminal_reports" in validator
    assert "fk_position_error_m" in validator
    assert "estimated_link_clearance_m" in validator


def test_table_only_gate_ignores_arm_separation_but_rejects_table_penetration():
    joints = np.zeros(6)
    torso = np.zeros(3)
    permissive = R1LiteTableCollisionKinematics(
        active_side="right",
        table_z_m=-1.0,
    )
    report = permissive.validate(joints, joints, torso)
    assert report.required_tcp_separation_m == 0.0

    blocked = R1LiteTableCollisionKinematics(
        active_side="right",
        table_z_m=2.0,
    )
    with pytest.raises(ValueError, match="table collision"):
        blocked.validate(joints, joints, torso)
