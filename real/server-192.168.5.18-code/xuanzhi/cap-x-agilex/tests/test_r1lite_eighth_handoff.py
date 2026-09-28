import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/r1lite_eighth_towel_recenter_then_right.py"


def load_script():
    spec = importlib.util.spec_from_file_location("r1lite_eighth_handoff", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_policy_is_plan_bound_and_has_exact_event_order():
    module = load_script()
    plan_sha = "a" * 64
    policy = module.build_fixed_policy(plan_sha)

    assert [(item["kind"], item["id"]) for item in policy["operations"]] == [
        ("gripper", "pregrasp_open_40"),
        ("stage", "grasp"),
        ("contact_probe", "both"),
        ("gripper", "close_both"),
        ("stage", "lift"),
        ("stage", "pullback"),
        ("stage", "translate_center"),
        ("gripper", "open_both"),
        ("stage", "release_retract"),
        ("stage", "home"),
    ]
    assert module.validate_fixed_policy(policy, plan_sha256=plan_sha) == policy

    tampered = deepcopy(policy)
    tampered["operations"][-1]["id"] = "not_home"
    with pytest.raises(ValueError, match="fixed plan-bound policy"):
        module.validate_fixed_policy(tampered, plan_sha256=plan_sha)
    with pytest.raises(ValueError, match="fixed plan-bound policy"):
        module.validate_fixed_policy(policy, plan_sha256="b" * 64)


def test_handoff_requires_passed_execution_and_explicit_home_gate():
    module = load_script()
    hashes = {
        "plan_sha256": "1" * 64,
        "policy_sha256": "2" * 64,
        "execution_bundle_sha256": "3" * 64,
        "capability_certificate_sha256": "4" * 64,
    }
    audit = {
        "pipeline": 8,
        "mode": "execute",
        "passed": True,
        "home_completed": True,
        **hashes,
        "events": [
            {
                "event": "home_joint_gate",
                "stage_id": "home",
                "passed": True,
            },
            {"event": "eighth_stage_completed", "stage_id": "home"},
        ],
    }
    module.validate_eighth_execution_audit(audit, expected_hashes=hashes)

    for mutation in (
        lambda value: value.__setitem__("passed", False),
        lambda value: value.__setitem__("home_completed", False),
        lambda value: value.__setitem__("events", []),
        lambda value: value.__setitem__("policy_sha256", "9" * 64),
    ):
        invalid = deepcopy(audit)
        mutation(invalid)
        with pytest.raises(ValueError):
            module.validate_eighth_execution_audit(invalid, expected_hashes=hashes)


def test_manifest_binds_both_phases_but_never_authorizes_right_execution():
    module = load_script()
    artifacts = {
        "plan_sha256": "1" * 64,
        "policy_sha256": "2" * 64,
        "execution_bundle_sha256": "3" * 64,
        "capability_certificate_sha256": "4" * 64,
    }
    manifest = module.build_handoff_manifest(
        eighth_artifact_hashes=artifacts,
        eighth_execution_audit_sha256="5" * 64,
        seventh_plan_sha256="6" * 64,
        seventh_perception_audit_sha256="7" * 64,
    )

    assert manifest["eighth"]["artifact_sha256"] == artifacts
    assert manifest["eighth"]["execution_audit_sha256"] == "5" * 64
    assert manifest["seventh_reperception"]["plan_sha256"] == "6" * 64
    assert manifest["seventh_reperception"]["perception_audit_sha256"] == "7" * 64
    assert manifest["seventh_reperception"]["first_eighth_images_reused"] is False
    assert manifest["right_execution_authorized"] is False


def test_completed_manifest_binds_certification_execution_and_home():
    module = load_script()
    pending = module.build_handoff_manifest(
        eighth_artifact_hashes={
            "plan_sha256": "1" * 64,
            "policy_sha256": "2" * 64,
            "execution_bundle_sha256": "3" * 64,
            "capability_certificate_sha256": "4" * 64,
        },
        eighth_execution_audit_sha256="5" * 64,
        seventh_plan_sha256="6" * 64,
        seventh_perception_audit_sha256="7" * 64,
    )
    completed = module.build_completed_handoff_manifest(
        pending_manifest=pending,
        right_code_sha256="8" * 64,
        right_execution_bundle_sha256="9" * 64,
        right_capability_certificate_sha256="a" * 64,
        right_collision_audit_sha256="b" * 64,
        right_execution_audit_sha256="c" * 64,
    )

    assert completed["schema"] == module.COMPLETED_HANDOFF_SCHEMA
    assert completed["right_certification"] == {
        "selected_plan_sha256": "6" * 64,
        "code_sha256": "8" * 64,
        "execution_bundle_sha256": "9" * 64,
        "capability_certificate_sha256": "a" * 64,
        "collision_audit_sha256": "b" * 64,
        "passed": True,
    }
    assert completed["right_candidate_selection"] == {
        "audit_sha256": None,
        "passed": True,
    }
    assert completed["right_execution"] == {
        "audit_sha256": "c" * 64,
        "passed": True,
        "home_completed": True,
    }
    assert completed["right_execution_authorized"] is True


def test_right_execution_audit_requires_hashes_and_home_evidence():
    module = load_script()
    expected = {
        "plan_sha256": "1" * 64,
        "code_sha256": "2" * 64,
        "execution_bundle_sha256": "3" * 64,
    }
    audit = {
        "pipeline": 7,
        "mode": "execute",
        "passed": True,
        **expected,
        "events": [{"event": "home_completed"}],
    }
    module.validate_right_execution_audit(audit, **expected)

    no_home = deepcopy(audit)
    no_home["events"] = []
    with pytest.raises(ValueError, match="home completion"):
        module.validate_right_execution_audit(no_home, **expected)
    wrong_bundle = deepcopy(audit)
    wrong_bundle["execution_bundle_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="binding mismatch"):
        module.validate_right_execution_audit(wrong_bundle, **expected)


def test_sources_restore_fresh_seventh_handoff_after_home():
    source = SCRIPT.read_text()
    runner = (REPO / "scripts/run_r1lite_eighth_towel_pipeline.sh").read_text()

    assert "load_camera_robot" in source
    assert "segment_towel" in source
    assert "load_eighth_execution_bundle" in source
    assert "execute_eighth_stage" in source
    assert '"handoff"' in source
    assert "fresh_head_camera_reperception" in source
    assert "prepare-eighth" in runner
    assert "execute-eighth" in runner
    assert "prepare-right" in runner
    assert "execute-right" in runner
    assert "run_r1lite_seventh_dual_arm_towel_fold_pipeline.sh" in runner
    assert "fresh_right" in runner


def test_runner_can_handoff_directly_to_saved_pipeline7_replay():
    runner = (
        REPO / "scripts/run_r1lite_eighth_towel_pipeline.sh"
    ).read_text()

    assert "execute-eighth-then-saved-right" in runner
    assert "r1lite_replay_saved_ik_sequence.py" in runner
    assert "--saved-grasp-joint-sequence" in runner
    assert "saved_grasp_joint_sequence_20260815_1920.json" in runner
    assert "--feedback-downward-step-m 0.005" in runner
    assert "--feedback-downward-step-candidates-m" in runner
    assert "0.005 0.006 0.007 0.008 0.009 0.01" in runner
    assert "--feedback-downward-refinement-iterations 1" in runner
    assert "--stop-on-effort-contact" in runner
    assert "--ignore-table-height-clamp" in runner
    assert "--ignore-local-fk-interpolation-z-gate" in runner
    replay = (
        REPO / "scripts/r1lite_replay_saved_ik_sequence.py"
    ).read_text()
    assert "replaced_by_feedback_table_refinement" in replay
    assert "fingertip_at_table_or_maximum_iterations" in replay
    assert "right_gripper_close_stabilized" in replay
    assert "right gripper did not reach a stable closed/contact position" in replay
    assert "execute_right_descent_with_contact_monitor" in replay
    assert "--right-gripper-open-position" in replay
    assert "controller.config.gripper_open_position" in replay
    assert "residual_threshold_nm=2.5" in replay
    assert "fixed_reference_effort = reference_effort.copy()" in replay
    assert "require_settled_velocity=False" in replay
    assert "stop_requested=contact_stop_requested" in replay
    assert "after_position[2] - before_position[2] >= 0.0001" in replay
    assert '"capture": "after_unsegmented_initial_10mm_descent"' in replay
    assert '"fresh_perception_after_eighth": False' in runner
    assert '"first_phase_home_completed"' in runner
