import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from capx.integrations.r1lite.towel_plan import build_eighth_bimanual_recenter_plan


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/render_r1lite_eighth_towel_execution_html.py"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "render_r1lite_eighth_towel_execution_html", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan():
    geometry = {
        "source": "sam3_largest_external_contour",
        "grasp_point_method": "nearest_robot_two_projected_rectangle_corners",
        "same_scanline": True,
        "points_on_real_external_contour": True,
        "pixel_order": "camera_left_to_camera_right",
        "robot_reference_base_xyz_m": [0.0, 0.0, 0.0],
        "corner_candidates": [
            {
                "pixel_uv": [250.0, 300.0],
                "base_xyz_m": [0.25, 0.30, 0.80],
                "robot_xy_distance_m": 1.0,
                "selected": False,
            },
            {
                "pixel_uv": [750.0, 300.0],
                "base_xyz_m": [0.75, 0.30, 0.80],
                "robot_xy_distance_m": 1.1,
                "selected": False,
            },
            {
                "pixel_uv": [700.0, 700.0],
                "base_xyz_m": [0.70, 0.70, 0.80],
                "robot_xy_distance_m": 0.8,
                "selected": True,
            },
            {
                "pixel_uv": [300.0, 700.0],
                "base_xyz_m": [0.30, 0.70, 0.80],
                "robot_xy_distance_m": 0.7,
                "selected": True,
            },
        ],
        "external_contour_pixel_uv": [
            [250.0, 300.0],
            [750.0, 300.0],
            [700.0, 700.0],
            [300.0, 700.0],
        ],
        "external_contour_base_xyz_m": [
            [0.25, 0.30, 0.80],
            [0.75, 0.30, 0.80],
            [0.70, 0.70, 0.80],
            [0.30, 0.70, 0.80],
        ],
        "external_contour_sample_stride": 1,
        "quality": {
            "pair_separation_px": 400.0,
            "rectangularity": 1.0,
            "selection_margin_to_third_corner_m": 0.2,
            "maximum_corner_projection_distance_px": 0.0,
            "left_contour_distance_px": 0.0,
            "right_contour_distance_px": 0.0,
            "quality_score": 1.0,
        },
        "keypoints": {
            "center": {
                "pixel_uv": [500.0, 450.0],
                "base_xyz_m": [0.50, 0.45, 0.80],
            },
            "left_grasp": {
                "pixel_uv": [300.0, 700.0],
                "base_xyz_m": [0.30, 0.70, 0.80],
            },
            "right_grasp": {
                "pixel_uv": [700.0, 700.0],
                "base_xyz_m": [0.70, 0.70, 0.80],
            },
        },
    }
    return build_eighth_bimanual_recenter_plan(geometry, table_z_m=0.80)


def fake_bundle():
    def side(target, fk):
        return {
            "target_position_base_m": target,
            "target_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "fk_position_base_m": fk,
            "fk_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }

    return {
        "stages": [
            {
                "id": "grasp",
                "waypoints": [
                    {
                        "left": side([0.30, 0.70, 0.881], [0.301, 0.70, 0.881]),
                        "right": side([0.70, 0.70, 0.881], [0.699, 0.70, 0.881]),
                        "expected_midpoint_base_m": [0.50, 0.70, 0.881],
                        "expected_right_minus_left_m": [0.40, 0.0, 0.0],
                        "collision_summary": {
                            "estimated_link_clearance_m": 0.020,
                            "left_table_clearance_m": 0.010,
                            "right_table_clearance_m": 0.012,
                            "gripper_envelope_separation_m": 0.030,
                        },
                    }
                ],
            }
        ]
    }


def audit_events():
    return {
        "events": [
            {"event": "ignored"},
            {
                "event": "actual_tcp_sample",
                "side": "left",
                "stage_id": "grasp",
                "phase": "eighth_certified_waypoint",
                "sample_index": 1,
                "position_base_m": [0.301, 0.700, 0.881],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "event": "actual_tcp_sample",
                "side": "right",
                "stage_id": "grasp",
                "phase": "eighth_certified_waypoint",
                "sample_index": 1,
                "position_base_m": [0.699, 0.700, 0.881],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        ]
    }


def test_pure_trace_extraction_and_pair_errors():
    module = load_script()
    bundle_traces = module.extract_bundle_traces(
        fake_bundle(), tcp_to_fingertip_m=0.078
    )
    actual = module.extract_actual_tcp_traces(
        audit_events(), tcp_to_fingertip_m=0.078
    )
    errors = module.actual_pair_errors(actual, plan(), fake_bundle())

    assert bundle_traces["target"]["left"][0]["position_base_m"] == [
        0.30,
        0.70,
        0.881,
    ]
    assert bundle_traces["fk"]["right"][0]["position_base_m"] == [
        0.699,
        0.70,
        0.881,
    ]
    assert bundle_traces["clearance"][0]["clearance_m"] == pytest.approx(0.010)
    assert len(actual["left"]) == len(actual["right"]) == 1
    assert errors[0]["relative_vector_error_m"] == pytest.approx(
        0.015297058540778369
    )
    assert errors[0]["midpoint_error_m"] == pytest.approx(0.0)


def test_audit_artifact_hash_mismatches_are_rejected():
    module = load_script()
    plan_raw = json.dumps(plan(), sort_keys=True).encode()
    plan_sha = hashlib.sha256(plan_raw).hexdigest()
    bundle_sha = "2" * 64
    audit = {
        "pipeline": 8,
        "mode": "execute",
        "events": [],
        "plan_sha256": plan_sha,
        "policy_sha256": "1" * 64,
        "execution_bundle_sha256": bundle_sha,
        "capability_certificate_sha256": "3" * 64,
    }
    module.validate_audit_bindings(
        audit,
        plan_file_sha256=plan_sha,
        bundle_file_sha256=bundle_sha,
    )

    wrong_plan = dict(audit, plan_sha256="9" * 64)
    with pytest.raises(ValueError, match="plan_sha256 binding mismatch"):
        module.validate_audit_bindings(
            wrong_plan,
            plan_file_sha256=plan_sha,
            bundle_file_sha256=bundle_sha,
        )

    wrong_bundle = dict(audit, execution_bundle_sha256="8" * 64)
    with pytest.raises(ValueError, match="execution_bundle_sha256 binding mismatch"):
        module.validate_audit_bindings(
            wrong_bundle,
            plan_file_sha256=plan_sha,
            bundle_file_sha256=bundle_sha,
        )


def test_html_contains_distinct_legends_geometry_and_error_curves():
    module = load_script()
    execution_audit = {"passed": True, **audit_events()}
    payload = module.build_payload(plan(), fake_bundle(), execution_audit)
    html = module.render_html(payload, "/* embedded plotly */")

    for legend in (
        "Left target fingertip",
        "Right target fingertip",
        "Left IK/FK fingertip",
        "Right IK/FK fingertip",
        "Left actual fingertip",
        "Right actual fingertip",
        "Actual pair relative-vector error",
        "Actual pair midpoint error",
        "Bundle collision minimum clearance",
        "First center to translate-center endpoint",
        "Perceived towel contour",
    ):
        assert legend in html
    for geometry_label in (
        "first_towel_center",
        "left_lower_contour_grasp",
        "right_lower_contour_grasp",
        "grasp_pair_midpoint",
        "translate_center_endpoint",
    ):
        assert geometry_label in html
