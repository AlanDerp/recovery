#!/usr/bin/env python3
"""Fail-closed right-arm-only orchestration for pipeline seven."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.execution_bundle import (  # noqa: E402
    load_execution_bundle,
    sha256_file,
)
from capx.integrations.r1lite.towel_plan import (  # noqa: E402
    build_sequential_session_manifest,
    validate_sequential_single_arm_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--right-plan", type=Path, required=True)
    parser.add_argument("--right-code", type=Path, required=True)
    parser.add_argument("--right-bundle", type=Path, required=True)
    parser.add_argument("--right-certificate", type=Path, required=True)
    parser.add_argument("--static-sync-audit", type=Path, required=True)
    parser.add_argument("--motion-sync-audit", type=Path, required=True)
    parser.add_argument("--interlock-audit", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8110/chat/completions",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def execute_phase(
    *,
    args: argparse.Namespace,
    plan: Path,
    code: Path,
    bundle: Path,
    certificate: Path,
    audit: Path,
) -> None:
    run(
        [
            sys.executable,
            "scripts/r1lite_seventh_dual_arm_towel_fold.py",
            "--mode",
            "execute",
            "--config",
            str(args.config),
            "--calibration",
            str(args.calibration),
            "--audit-output",
            str(audit),
            "--approved-plan",
            str(plan),
            "--expected-plan-sha256",
            sha256_file(plan),
            "--generated-code",
            str(code),
            "--expected-code-sha256",
            sha256_file(code),
            "--execution-bundle",
            str(bundle),
            "--expected-execution-bundle-sha256",
            sha256_file(bundle),
            "--capability-certificate",
            str(certificate),
            "--expected-capability-certificate-sha256",
            sha256_file(certificate),
        ]
    )
    result = json.loads(audit.read_text())
    if result.get("passed") is not True:
        raise RuntimeError(f"phase execution failed: {audit}")
    if not any(
        event.get("event") == "home_completed"
        for event in result.get("events", [])
    ):
        raise RuntimeError("phase did not pass the certified home gate")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    right_plan = json.loads(args.right_plan.read_text())
    validate_sequential_single_arm_plan(right_plan)
    if right_plan["phase_side"] != "right":
        raise ValueError("the first approved phase must be right-arm")
    right_bundle = load_execution_bundle(
        args.right_bundle,
        plan_sha256=sha256_file(args.right_plan),
    )
    right_audit = args.output_dir / "right_execute_audit.json"
    execute_phase(
        args=args,
        plan=args.right_plan,
        code=args.right_code,
        bundle=args.right_bundle,
        certificate=args.right_certificate,
        audit=right_audit,
    )

    original_center = right_plan["original_center_base_m"]
    manifest = build_sequential_session_manifest(
        original_center_base_m=original_center,
        initial_left_joints_rad=right_bundle["initial_state"][
            "left_joints_rad"
        ],
        initial_right_joints_rad=right_bundle["initial_state"][
            "right_joints_rad"
        ],
        torso_joints_rad=right_bundle["initial_state"]["torso_joints_rad"],
        right_phase={
            "plan_sha256": sha256_file(args.right_plan),
            "code_sha256": sha256_file(args.right_code),
            "bundle_sha256": sha256_file(args.right_bundle),
            "certificate_sha256": sha256_file(args.right_certificate),
            "execute_audit_sha256": sha256_file(right_audit),
        },
        left_phase=None,
    )
    manifest_path = args.output_dir / "sequential_session_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "passed": True,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
