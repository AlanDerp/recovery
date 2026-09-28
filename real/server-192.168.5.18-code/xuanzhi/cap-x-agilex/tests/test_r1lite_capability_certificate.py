from datetime import datetime, timezone

import pytest

from capx.integrations.r1lite.capability_certificate import (
    SCHEMA,
    build_capability_certificate,
    validate_capability_certificate,
)


def certificate():
    return {
        "schema": SCHEMA,
        "passed": True,
        "robot_id": "r1lite-lab",
        "issued_at": "2026-08-02T08:00:00Z",
        "expires_at": "2026-08-03T08:00:00Z",
        "dual_arm_control_source_exclusive": True,
        "dual_arm_stop_behavior_verified": True,
        "dual_arm_collision_gate_verified": True,
        "dual_arm_feedback_stale_interlock_verified": True,
        "dual_arm_shared_progress_verified": True,
        "artifact_bindings": {
            "plan_sha256": "p",
            "code_sha256": "c",
            "execution_bundle_sha256": "b",
        },
    }


def test_capability_certificate_binds_all_artifacts_and_expiry():
    value = certificate()
    now = datetime(2026, 8, 2, 9, tzinfo=timezone.utc)
    assert validate_capability_certificate(
        value,
        plan_sha256="p",
        code_sha256="c",
        execution_bundle_sha256="b",
        now=now,
    ) == value

    value["artifact_bindings"]["plan_sha256"] = "changed"
    with pytest.raises(ValueError, match="bindings mismatch"):
        validate_capability_certificate(
            value,
            plan_sha256="p",
            code_sha256="c",
            execution_bundle_sha256="b",
            now=now,
        )


def test_capability_certificate_fails_closed_on_missing_pass_or_expiry():
    value = certificate()
    value["dual_arm_stop_behavior_verified"] = False
    with pytest.raises(ValueError, match="required pass"):
        validate_capability_certificate(
            value,
            plan_sha256="p",
            code_sha256="c",
            execution_bundle_sha256="b",
            now=datetime(2026, 8, 2, 9, tzinfo=timezone.utc),
        )

    value = certificate()
    with pytest.raises(ValueError, match="not currently valid"):
        validate_capability_certificate(
            value,
            plan_sha256="p",
            code_sha256="c",
            execution_bundle_sha256="b",
            now=datetime(2026, 8, 4, tzinfo=timezone.utc),
        )


def test_builder_issues_short_lived_artifact_bound_certificate():
    issued = datetime(2026, 8, 2, 9, tzinfo=timezone.utc)
    value = build_capability_certificate(
        robot_id="r1lite-lab",
        plan_sha256="p",
        code_sha256="c",
        execution_bundle_sha256="b",
        evidence_sha256={"collision": "e"},
        validity_hours=2,
        issued_at=issued,
    )
    assert value["expires_at"] == "2026-08-02T11:00:00Z"
    assert value["evidence_sha256"] == {"collision": "e"}
