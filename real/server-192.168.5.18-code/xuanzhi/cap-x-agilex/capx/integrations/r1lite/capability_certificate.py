"""Fail-closed validation for R1 Lite dual-arm capability certificates."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCHEMA = "capx.r1lite.bimanual_capability.v1"
REQUIRED_PASSES = (
    "dual_arm_control_source_exclusive",
    "dual_arm_stop_behavior_verified",
    "dual_arm_collision_gate_verified",
    "dual_arm_feedback_stale_interlock_verified",
    "dual_arm_shared_progress_verified",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid capability certificate {field}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"capability certificate {field} must include timezone")
    return parsed.astimezone(timezone.utc)


def validate_capability_certificate(
    certificate: dict[str, Any],
    *,
    plan_sha256: str,
    code_sha256: str,
    execution_bundle_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    if certificate.get("schema") != SCHEMA:
        raise ValueError("unexpected bimanual capability certificate schema")
    if certificate.get("passed") is not True:
        raise ValueError("bimanual capability certificate did not pass")
    if not isinstance(certificate.get("robot_id"), str) or not certificate["robot_id"]:
        raise ValueError("capability certificate robot_id is required")
    for key in REQUIRED_PASSES:
        if certificate.get(key) is not True:
            raise ValueError(f"capability certificate missing required pass {key}")
    bindings = certificate.get("artifact_bindings")
    expected = {
        "plan_sha256": plan_sha256,
        "code_sha256": code_sha256,
        "execution_bundle_sha256": execution_bundle_sha256,
    }
    if not isinstance(bindings, dict) or bindings != expected:
        raise ValueError("capability certificate artifact bindings mismatch")
    issued_at = _utc_timestamp(certificate.get("issued_at"), field="issued_at")
    expires_at = _utc_timestamp(certificate.get("expires_at"), field="expires_at")
    current = now or datetime.now(timezone.utc)
    if issued_at > current or expires_at <= current or expires_at <= issued_at:
        raise ValueError("capability certificate is not currently valid")
    return certificate


def build_capability_certificate(
    *,
    robot_id: str,
    plan_sha256: str,
    code_sha256: str,
    execution_bundle_sha256: str,
    evidence_sha256: dict[str, str],
    validity_hours: float = 8.0,
    issued_at: datetime | None = None,
) -> dict[str, Any]:
    if not robot_id:
        raise ValueError("robot_id is required")
    issued = (issued_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = issued + timedelta(hours=float(validity_hours))
    if validity_hours <= 0:
        raise ValueError("capability certificate validity must be positive")
    certificate = {
        "schema": SCHEMA,
        "passed": True,
        "robot_id": robot_id,
        "issued_at": issued.isoformat().replace("+00:00", "Z"),
        "expires_at": expires.isoformat().replace("+00:00", "Z"),
        "dual_arm_control_source_exclusive": True,
        "dual_arm_stop_behavior_verified": True,
        "dual_arm_collision_gate_verified": True,
        "dual_arm_feedback_stale_interlock_verified": True,
        "dual_arm_shared_progress_verified": True,
        "artifact_bindings": {
            "plan_sha256": plan_sha256,
            "code_sha256": code_sha256,
            "execution_bundle_sha256": execution_bundle_sha256,
        },
        "evidence_sha256": dict(evidence_sha256),
    }
    return certificate


def load_capability_certificate(
    path: Path,
    *,
    expected_sha256: str,
    plan_sha256: str,
    code_sha256: str,
    execution_bundle_sha256: str,
) -> dict[str, Any]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError("capability certificate SHA-256 mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("capability certificate must be a JSON object")
    return validate_capability_certificate(
        value,
        plan_sha256=plan_sha256,
        code_sha256=code_sha256,
        execution_bundle_sha256=execution_bundle_sha256,
    )
