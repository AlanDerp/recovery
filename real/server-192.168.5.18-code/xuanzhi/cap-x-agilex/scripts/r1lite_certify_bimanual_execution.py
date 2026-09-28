#!/usr/bin/env python3
"""Issue a short-lived certificate only from passing dual-arm evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.capability_certificate import (  # noqa: E402
    build_capability_certificate,
    sha256_bytes,
)
from capx.integrations.r1lite.execution_bundle import (  # noqa: E402
    EIGHTH_SCHEMA,
    load_eighth_execution_bundle,
    load_execution_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument("--code", type=Path)
    artifact.add_argument(
        "--policy",
        type=Path,
        help="fixed pipeline-eight policy artifact (bound as code_sha256)",
    )
    parser.add_argument("--execution-bundle", type=Path, required=True)
    parser.add_argument("--static-sync-audit", type=Path, required=True)
    parser.add_argument("--motion-sync-audit", type=Path, required=True)
    parser.add_argument("--interlock-audit", type=Path, required=True)
    parser.add_argument("--collision-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validity-hours", type=float, default=8.0)
    return parser.parse_args()


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def main() -> None:
    args = parse_args()
    code_or_policy = args.policy or args.code
    assert code_or_policy is not None
    static = load_object(args.static_sync_audit)
    motion = load_object(args.motion_sync_audit)
    interlock = load_object(args.interlock_audit)
    collision = load_object(args.collision_audit)
    if static.get("result", {}).get("command_exclusivity_passed") is not True:
        raise RuntimeError("control-source exclusivity evidence did not pass")
    if static.get("result", {}).get("static_feedback_sync_passed") is not True:
        raise RuntimeError("static dual-arm feedback synchronization did not pass")
    if motion.get("passed") is not True:
        raise RuntimeError("minimal dual-arm motion synchronization did not pass")
    if interlock.get("passed") is not True or interlock.get(
        "interlock_action"
    ) != "hold_both":
        raise RuntimeError("dual-arm stale-feedback common hold did not pass")
    if collision.get("passed") is not True:
        raise RuntimeError("dual-arm IK and swept collision evidence did not pass")
    plan_sha = sha256_bytes(args.plan.read_bytes())
    code_sha = sha256_bytes(code_or_policy.read_bytes())
    bundle_raw = args.execution_bundle.read_bytes()
    bundle_header = load_object(args.execution_bundle)
    if bundle_header.get("schema") == EIGHTH_SCHEMA:
        load_eighth_execution_bundle(args.execution_bundle, plan_sha256=plan_sha)
    else:
        load_execution_bundle(args.execution_bundle, plan_sha256=plan_sha)
    evidence_paths = {
        "static_sync_audit": args.static_sync_audit,
        "motion_sync_audit": args.motion_sync_audit,
        "interlock_audit": args.interlock_audit,
        "collision_audit": args.collision_audit,
    }
    certificate = build_capability_certificate(
        robot_id=args.robot_id,
        plan_sha256=plan_sha,
        code_sha256=code_sha,
        execution_bundle_sha256=sha256_bytes(bundle_raw),
        evidence_sha256={
            name: sha256_bytes(path.read_bytes())
            for name, path in evidence_paths.items()
        },
        validity_hours=args.validity_hours,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, ensure_ascii=False))
    print(json.dumps(certificate, ensure_ascii=False))


if __name__ == "__main__":
    main()
