import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from capx.integrations.r1lite.bimanual_kinematics import (
    R1LiteCombinedCollisionKinematics,
)
from capx.integrations.r1lite.execution_bundle import (
    EIGHTH_STAGE_ORDER,
    build_eighth_execution_bundle,
    canonical_sha256,
    validate_eighth_execution_bundle,
)


REPO = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class FakeGate:
    def __init__(self, report):
        self.report = report
        self.calls = 0

    def validate(self, left, right, torso):
        self.calls += 1
        return self.report


class FakeReport:
    def __init__(self, *, tcp=0.2, clearance=0.03):
        self.tcp_separation_m = tcp
        self.estimated_link_clearance_m = clearance

    def to_dict(self):
        return {
            "tcp_separation_m": self.tcp_separation_m,
            "estimated_link_clearance_m": self.estimated_link_clearance_m,
        }


def test_combined_gate_checks_interarm_both_tables_and_gripper_envelope():
    pair = FakeGate(FakeReport(tcp=0.20, clearance=0.03))
    left_table = FakeGate(FakeReport(clearance=0.01))
    right_table = FakeGate(FakeReport(clearance=0.02))
    gate = R1LiteCombinedCollisionKinematics(
        table_z_m=0.0,
        minimum_gripper_envelope_separation_m=0.125,
        bimanual_gate=pair,
        left_table_gate=left_table,
        right_table_gate=right_table,
    )

    report = gate.validate(np.zeros(6), np.zeros(6), np.zeros(3))

    assert (pair.calls, left_table.calls, right_table.calls) == (1, 1, 1)
    assert report.estimated_link_clearance_m == 0.01
    assert report.gripper_envelope_separation_m == 0.20

    pair.report = FakeReport(tcp=0.12)
    with pytest.raises(ValueError, match="gripper conservative envelope"):
        gate.validate(np.zeros(6), np.zeros(6), np.zeros(3))
    # The envelope failure cannot bypass either arm's open-finger table gate.
    assert (left_table.calls, right_table.calls) == (2, 2)


def side_item(target, joint, previous):
    return {
        "target_position_base_m": target,
        "target_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "joints_rad": [joint] * 6,
        "fk_position_base_m": target,
        "fk_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "fk_position_error_m": 0.0,
        "fk_orientation_error_rad": 0.0,
        "maximum_joint_delta_rad": abs(joint - previous),
    }


def build_stages():
    stages = []
    previous_left = 0.0
    previous_right = 0.0
    previous_targets = {
        "left": [0.30, 0.10, 0.70],
        "right": [0.50, 0.10, 0.70],
    }
    for index, stage_id in enumerate(EIGHTH_STAGE_ORDER):
        if stage_id == "home":
            left_joint, right_joint = 0.0, 0.0
        else:
            left_joint, right_joint = previous_left + 0.1, previous_right - 0.1
        if stage_id == "home":
            left_target = list(previous_targets["left"])
            right_target = list(previous_targets["right"])
        else:
            left_target = [
                previous_targets["left"][0] + 0.01,
                0.10,
                0.70,
            ]
            right_target = [previous_targets["right"][0] + 0.01, 0.10, 0.70]
        left = side_item(left_target, left_joint, previous_left)
        right = side_item(right_target, right_joint, previous_right)
        stages.append(
            {
                "id": stage_id,
                "waypoints": [
                    {
                        "alpha": 1.0,
                        "ik_acceptance": (
                            "best_effort_seed"
                            if stage_id == "grasp"
                            else "joint_transition"
                            if stage_id == "home"
                            else "required"
                        ),
                        "expected_midpoint_base_m": (
                            (np.asarray(left_target) + right_target) / 2.0
                        ).tolist(),
                        "expected_right_minus_left_m": (
                            np.asarray(right_target) - left_target
                        ).tolist(),
                        "left": left,
                        "right": right,
                        "collision_summary": {
                            "mode": "disabled_non_grasp_stage",
                            "checked": False,
                            "sweep_samples": 0,
                        },
                    }
                ],
            }
        )
        previous_left, previous_right = left_joint, right_joint
        previous_targets = {"left": left_target, "right": right_target}
    return stages


def build_bundle():
    return build_eighth_execution_bundle(
        plan_sha256=digest("plan"),
        config_sha256=digest("config"),
        full_urdf_sha256=digest("full"),
        left_relaxed_urdf_sha256=digest("left"),
        right_relaxed_urdf_sha256=digest("right"),
        tcp_definition={"left": "tool", "right": "tool"},
        collision_model={
            "name": "first_phase_collision_disabled"
        },
        initial_left_joints_rad=[0.0] * 6,
        initial_right_joints_rad=[0.0] * 6,
        torso_joints_rad=[0.0] * 3,
        stages=build_stages(),
    )


def test_eighth_bundle_is_fixed_order_strictly_bound_and_self_hashed():
    bundle = build_bundle()

    assert bundle["stage_order"] == list(EIGHTH_STAGE_ORDER)
    assert bundle["bundle_sha256"] == canonical_sha256(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    )
    assert validate_eighth_execution_bundle(
        bundle,
        plan_sha256=digest("plan"),
        tcp_definition={"left": "tool", "right": "tool"},
        collision_model={
            "name": "first_phase_collision_disabled"
        },
    ) == bundle


def test_final_grasp_seed_may_exceed_fk_tolerance_before_online_refinement():
    bundle = deepcopy(build_bundle())
    grasp = bundle["stages"][0]["waypoints"][-1]
    fk_position = list(grasp["left"]["fk_position_base_m"])
    fk_position[0] += 0.20
    grasp["left"]["fk_position_base_m"] = fk_position
    grasp["left"]["fk_position_error_m"] = float(
        np.linalg.norm(
            np.asarray(grasp["left"]["fk_position_base_m"])
            - np.asarray(grasp["left"]["target_position_base_m"])
        )
    )
    bundle["bundle_sha256"] = canonical_sha256(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    )

    assert validate_eighth_execution_bundle(bundle) == bundle


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda value: value["stage_order"].reverse(),
            "stage order is immutable",
        ),
        (
            lambda value: value["stages"][-1]["waypoints"][0][
                "collision_summary"
                ].__setitem__("checked", True),
            "grasp-only table policy",
        ),
        (
            lambda value: value["stages"][1]["waypoints"][0]["left"][
                "target_position_base_m"
            ].__setitem__(0, 0.50),
            "Cartesian step exceeds limit",
        ),
        (
            lambda value: value["stages"][2]["waypoints"][0].update(
                {"unexpected": True}
            ),
            "fields mismatch",
        ),
    ],
)
def test_eighth_bundle_rejects_unsafe_or_noncanonical_content(mutation, message):
    bundle = deepcopy(build_bundle())
    mutation(bundle)
    with pytest.raises(ValueError, match=message):
        validate_eighth_execution_bundle(bundle)


def test_eighth_script_is_offline_and_writes_failure_bundle():
    script = (
        REPO / "scripts/r1lite_validate_eighth_bimanual_phase.py"
    ).read_text()
    assert '"collision_checks_performed": False' in script
    assert "validate_interpolated_segment" in script
    assert "robot_motion_commands_sent" in script
    assert "never_execute" in script
    assert ".move_" not in script
    assert ".set_arm" not in script
    assert "eighth_execution_bundle_failure.v1" in script
