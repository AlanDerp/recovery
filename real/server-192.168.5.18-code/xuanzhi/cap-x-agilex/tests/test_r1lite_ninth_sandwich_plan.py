from copy import deepcopy

import pytest

from capx.integrations.r1lite.sandwich_perception import SCENE_SCHEMA
from capx.integrations.r1lite.sandwich_plan import (
    STAGE_ORDER,
    apply_grasp_orientation_branch,
    build_policy,
    build_sandwich_intent,
    build_session_manifest,
    select_arm,
    validate_policy,
    validate_sandwich_intent,
    validate_sandwich_plan,
    validate_session_manifest,
)


def object_geometry(center, *, width, length, height, major=(1.0, 0.0)):
    minor = (-major[1], major[0])
    return {
        "center_base_m": list(center),
        "table_z_at_center_m": 0.80,
        "top_z_base_m": 0.80 + height,
        "height_above_table_m": height,
        "thickness_m": height,
        "major_axis_base_xy": list(major),
        "minor_axis_base_xy": list(minor),
        "width_m": width,
        "length_m": length,
        "yaw_rad": 0.0,
    }


def scene():
    return {
        "schema": SCENE_SCHEMA,
        "frame_id": "head_left",
        "stamp_ns": 123,
        "image_shape": [720, 1280],
        "table_plane": {
            "normal_base": [0.0, 0.0, 1.0],
            "d": -0.80,
            "rmse_m": 0.001,
            "inlier_count": 1000,
            "candidate_count": 1200,
        },
        "objects": {
            "brown_bread": object_geometry(
                (0.46, -0.10, 0.81),
                width=0.070,
                length=0.090,
                height=0.020,
                major=(0.0, 1.0),
            ),
            "donut": object_geometry(
                (0.40, 0.10, 0.815),
                width=0.065,
                length=0.070,
                height=0.030,
            ),
            "mixed_bread": object_geometry(
                (0.52, 0.12, 0.81),
                width=0.068,
                length=0.088,
                height=0.020,
            ),
        },
    }


def build(phase="donut_on_brown"):
    return build_sandwich_intent(
        scene(),
        phase=phase,
        first_image_sha256="1" * 64,
        second_image_sha256="2" * 64,
        repeatability_m=0.002,
    )


def test_intent_has_strict_order_target_height_and_no_force_continue():
    intent = build()

    assert tuple(stage["id"] for stage in intent["stages"]) == STAGE_ORDER
    place = next(
        stage for stage in intent["stages"] if stage["id"] == "place_on_target"
    )
    assert place["active_fingertip_position_base_m"][:2] == pytest.approx(
        scene()["objects"]["brown_bread"]["center_base_m"][:2]
    )
    assert place["active_fingertip_position_base_m"][2] == pytest.approx(0.824)
    assert intent["safety"]["collision_policy"] == (
        "combined_interarm_table_and_open_gripper"
    )
    assert (
        intent["safety"]["cartesian_refinement_policy"][
            "force_continue_after_maximum_iterations"
        ]
        is False
    )


def test_auto_arm_selection_binds_every_stage_and_deterministic_policy():
    intent = build()
    plan = select_arm(intent, "left")
    validate_sandwich_plan(plan)
    assert plan["selected_side"] == "left"
    assert all(stage["active_side"] == "left" for stage in plan["stages"])

    policy = build_policy(plan, plan_sha256="a" * 64)
    validate_policy(policy, plan=plan, plan_sha256="a" * 64)
    assert policy["operations"][-1] == {
        "call": "execute_single_arm_stage",
        "side": "left",
        "stage_id": "home",
    }
    tampered = deepcopy(policy)
    tampered["operations"].pop()
    with pytest.raises(ValueError, match="deterministic"):
        validate_policy(tampered, plan=plan, plan_sha256="a" * 64)


def test_reachable_grasp_branch_rotates_source_and_target_yaw_independently():
    plan = select_arm(build(), "right")
    branched = apply_grasp_orientation_branch(
        plan,
        source_yaw_offset_deg=90.0,
        target_yaw_offset_deg=180.0,
    )

    stages = {stage["id"]: stage for stage in branched["stages"]}
    assert stages["grasp"]["jaw_closing_axis_base_xy"] == pytest.approx(
        [-1.0, 0.0]
    )
    assert stages["place_on_target"]["jaw_closing_axis_base_xy"] == pytest.approx(
        [1.0, 0.0]
    )
    assert stages["grasp"]["tool_z_base"] == [0.0, 0.0, -1.0]
    assert branched["grasp_orientation_branch"] == {
        "method": "offline_reachable_topdown_yaw_sampling",
        "source_yaw_offset_deg": 90.0,
        "target_yaw_offset_deg": 180.0,
    }
    validate_sandwich_plan(branched)


def test_top_phase_aligns_to_brown_bread_and_uses_fresh_stack_height():
    intent = build("mixed_bread_on_stack")
    place = next(
        stage for stage in intent["stages"] if stage["id"] == "place_on_target"
    )
    assert place["jaw_closing_axis_base_xy"] == pytest.approx([-1.0, 0.0])
    assert place["active_fingertip_position_base_m"][2] == pytest.approx(0.834)
    assert intent["phase_definition"]["support"] == "donut"


def test_span_schema_and_manifest_tampering_are_rejected():
    oversized = scene()
    oversized["objects"]["donut"]["width_m"] = 0.11
    with pytest.raises(ValueError, match="exceeds"):
        build_sandwich_intent(
            oversized,
            phase="donut_on_brown",
            first_image_sha256="1" * 64,
            second_image_sha256="2" * 64,
            repeatability_m=0.002,
        )

    intent = build()
    intent["safety"]["cartesian_refinement_policy"][
        "force_continue_after_maximum_iterations"
    ] = True
    with pytest.raises(ValueError, match="force-continue"):
        validate_sandwich_intent(intent)

    manifest = build_session_manifest(
        donut_phase={"plan_sha256": "a" * 64},
        top_phase={"plan_sha256": "b" * 64},
    )
    validate_session_manifest(manifest)
    manifest["donut_on_brown"]["plan_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_session_manifest(manifest)
