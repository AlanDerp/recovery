import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from capx.integrations.r1lite.execution_bundle import (
    build_execution_bundle,
    canonical_json_bytes,
    canonical_sha256,
    load_execution_bundle,
    sha256_bytes,
    validate_execution_bundle,
)

REPO = Path(__file__).resolve().parents[1]


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def tcp_definition():
    return {
        "left": "left_gripper_tool_z",
        "right": "right_gripper_tool_z",
        "left_definition": {
            "source_link": "left_gripper_link",
            "xyz_m": [0.0, 0.0, 0.0],
            "rpy_rad": [0.0, 1.5707963267948966, 0.0],
        },
    }


def collision_model():
    return {
        "name": "arm_link_capsules_plus_gripper_TCP_mesh_envelope",
        "link_capsule_radius_m": 0.025,
        "minimum_link_clearance_m": 0.005,
        "minimum_tcp_separation_m": 0.125,
        "sweep_interpolation": "linear_joint_space",
        "sweep_samples_per_segment": 50,
    }


def stage(stage_id, joint_value, maximum_delta):
    return {
        "id": stage_id,
        "left": {
            "joints_rad": [joint_value] * 6,
            "joint_waypoints_rad": [[joint_value] * 6],
            "target_position_base_m": [0.4, 0.2, 0.8],
            "target_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "fk_position_base_m": [0.4, 0.2, 0.8],
            "fk_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "fk_position_error_m": 0.0,
            "fk_orientation_error_rad": 0.0,
            "maximum_joint_delta_rad": maximum_delta,
        },
        "right": {
            "joints_rad": [-joint_value] * 6,
            "joint_waypoints_rad": [[-joint_value] * 6],
            "target_position_base_m": [0.4, -0.2, 0.8],
            "target_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "fk_position_base_m": [0.4, -0.2, 0.8],
            "fk_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "fk_position_error_m": 0.0,
            "fk_orientation_error_rad": 0.0,
            "maximum_joint_delta_rad": maximum_delta,
        },
    }


def build_bundle():
    return build_execution_bundle(
        plan_sha256=digest("plan"),
        config_sha256=digest("config"),
        full_urdf_sha256=digest("full"),
        left_relaxed_urdf_sha256=digest("left"),
        right_relaxed_urdf_sha256=digest("right"),
        tcp_definition=tcp_definition(),
        collision_model=collision_model(),
        initial_left_joints_rad=[0.0] * 6,
        initial_right_joints_rad=[0.0] * 6,
        torso_joints_rad=[0.0, 0.1, 0.2],
        stages=[
            stage("pregrasp", 0.25, 0.25),
            stage("grasp", 0.5, 0.25),
            stage("lift", 0.75, 0.25),
            stage("fold_center", 1.0, 0.25),
        ],
        sweeps={
            "samples_per_segment": 50,
            "segments": [
                {
                    "to_stage": "pregrasp",
                    "samples": 50,
                    "minimum_estimated_link_clearance_m": 0.04,
                    "minimum_tcp_separation_m": 0.20,
                    "passed": True,
                },
                {
                    "to_stage": "grasp",
                    "samples": 50,
                    "minimum_estimated_link_clearance_m": 0.03,
                    "minimum_tcp_separation_m": 0.18,
                    "passed": True,
                },
                {
                    "to_stage": "lift",
                    "samples": 50,
                    "minimum_estimated_link_clearance_m": 0.035,
                    "minimum_tcp_separation_m": 0.19,
                    "passed": True,
                },
                {
                    "to_stage": "fold_center",
                    "samples": 50,
                    "minimum_estimated_link_clearance_m": 0.025,
                    "minimum_tcp_separation_m": 0.17,
                    "passed": True,
                },
            ],
            "minimum_swept_link_clearance_m": 0.025,
            "minimum_swept_tcp_separation_m": 0.17,
        },
        maximum_stage_joint_delta_rad=1.0,
        fk_position_tolerance_m=0.015,
        fk_orientation_tolerance_rad=0.25,
    )


def test_build_is_canonical_and_load_validates_all_bindings(tmp_path):
    bundle = build_bundle()
    output = tmp_path / "bundle.json"
    raw = canonical_json_bytes(bundle)
    output.write_bytes(raw)

    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256(
        {"a": 1, "b": 2}
    )
    assert bundle["bundle_sha256"] == canonical_sha256(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    )
    assert load_execution_bundle(
        output,
        expected_file_sha256=sha256_bytes(raw),
        plan_sha256=digest("plan"),
        config_sha256=digest("config"),
        full_urdf_sha256=digest("full"),
        left_relaxed_urdf_sha256=digest("left"),
        right_relaxed_urdf_sha256=digest("right"),
        tcp_definition=tcp_definition(),
        collision_model=collision_model(),
        initial_left_joints_rad=[0.0] * 6,
        initial_right_joints_rad=[0.0] * 6,
        torso_joints_rad=[0.0, 0.1, 0.2],
    ) == bundle


def test_sequential_bundle_binds_declared_stage_order():
    order = ["pregrasp", "home"]
    bundle = build_execution_bundle(
        plan_sha256=digest("sequential-plan"),
        config_sha256=digest("config"),
        full_urdf_sha256=digest("full"),
        left_relaxed_urdf_sha256=digest("left"),
        right_relaxed_urdf_sha256=digest("right"),
        tcp_definition=tcp_definition(),
        collision_model=collision_model(),
        initial_left_joints_rad=[0.0] * 6,
        initial_right_joints_rad=[0.0] * 6,
        torso_joints_rad=[0.0, 0.1, 0.2],
        stages=[
            stage("pregrasp", 0.25, 0.25),
            stage("home", 0.0, 0.25),
        ],
        sweeps={
            "samples_per_segment": 50,
            "segments": [
                {
                    "to_stage": stage_id,
                    "samples": 50,
                    "minimum_estimated_link_clearance_m": 0.03,
                    "minimum_tcp_separation_m": 0.18,
                    "passed": True,
                }
                for stage_id in order
            ],
            "minimum_swept_link_clearance_m": 0.03,
            "minimum_swept_tcp_separation_m": 0.18,
        },
        maximum_stage_joint_delta_rad=1.0,
        fk_position_tolerance_m=0.015,
        fk_orientation_tolerance_rad=0.25,
        execution_mode="sequential_single_arm",
        stage_order=order,
    )

    assert bundle["execution_mode"] == "sequential_single_arm"
    assert bundle["stage_order"] == order
    assert validate_execution_bundle(bundle) == bundle


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.pop("sweeps"), "fields mismatch"),
        (lambda value: value.update({"unexpected": True}), "fields mismatch"),
        (
            lambda value: value["initial_state"]["left_joints_rad"].__setitem__(
                0, float("nan")
            ),
            "non-finite",
        ),
        (
            lambda value: value["stages"][0]["left"].update({"extra": 1}),
            "fields mismatch",
        ),
        (
            lambda value: value["sweeps"]["segments"][0].__setitem__(
                "to_stage", "grasp"
            ),
            "stage binding mismatch",
        ),
        (
            lambda value: value["sweeps"].__setitem__(
                "minimum_swept_tcp_separation_m", 0.19
            ),
            "summary mismatch",
        ),
    ],
)
def test_strict_schema_rejects_nan_missing_extra_and_cross_link_mismatch(
    mutation, message
):
    bundle = deepcopy(build_bundle())
    mutation(bundle)

    with pytest.raises(ValueError, match=message):
        validate_execution_bundle(bundle)


def test_tampering_and_external_hash_mismatches_fail_closed(tmp_path):
    bundle = build_bundle()
    tampered = deepcopy(bundle)
    tampered["stages"][0]["left"]["target_position_base_m"][0] = 0.41
    tampered["stages"][0]["left"]["fk_position_base_m"][0] = 0.41
    with pytest.raises(ValueError, match="canonical SHA-256 mismatch"):
        validate_execution_bundle(tampered)

    with pytest.raises(ValueError, match="plan_sha256 binding mismatch"):
        validate_execution_bundle(bundle, plan_sha256=digest("other-plan"))
    with pytest.raises(ValueError, match="tcp_definition_sha256 binding mismatch"):
        validate_execution_bundle(bundle, tcp_definition={"changed": True})
    with pytest.raises(ValueError, match="left_joints_rad binding mismatch"):
        validate_execution_bundle(bundle, initial_left_joints_rad=[0.1] * 6)

    output = tmp_path / "bundle.json"
    output.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        load_execution_bundle(output, expected_file_sha256=digest("wrong-file"))


def test_strict_json_loader_rejects_duplicate_keys_and_nan(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"first","schema":"second"}')
    with pytest.raises(ValueError, match="strict JSON"):
        load_execution_bundle(duplicate)

    non_finite = tmp_path / "nan.json"
    non_finite.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="strict JSON"):
        load_execution_bundle(non_finite)


def test_swept_validator_has_optional_bundle_output_and_uniform_joint_limit():
    script = (
        REPO / "scripts/r1lite_validate_towel_swept_collision.py"
    ).read_text()

    assert '--execution-bundle-output", type=Path' in script
    assert "max_joint_delta_rad=MAX_STAGE_JOINT_DELTA_RAD" in script
    assert "solve_stage_transit" in script
    assert "FK_POSITION_TOL_M = 0.030" in script
    assert "max_joint_delta_rad=3.0" not in script
