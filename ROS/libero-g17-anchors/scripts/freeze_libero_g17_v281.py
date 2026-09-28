#!/usr/bin/env python3
"""Seal the v281 focused 2/2 anchor before the K=5 live validation."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


ROOT = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/"
    "libero_g17_fair_fg_v281_joint_join_20260819"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = ROOT / "manifests/SOURCE_SHA256.json"
    snapshot = ROOT / "manifests/SOURCE_SNAPSHOT.zip"
    audit = ROOT / "evidence/v272_trial2_lagged_joint_join.json"
    tests = ROOT / "evidence/unit_tests.txt"
    summary = ROOT / "arms/v281/task-00/rep-01/SUMMARY_job-523702.json"
    submit = ROOT / "manifests/LATEST_SUBMIT.json"
    required = (source, snapshot, audit, tests, summary, submit)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"v281 freeze missing artifacts: {missing}")

    source_data = json.loads(source.read_text())
    for raw_path, expected in source_data["files"].items():
        path = Path(raw_path)
        actual = sha(path)
        if actual != expected:
            raise SystemExit(f"v281 source drift: {path} {actual} != {expected}")

    summary_data = json.loads(summary.read_text())
    outcomes = summary_data.get("treatment_outcomes", {})
    if summary_data.get("validity") != "valid_prefix_arm":
        raise SystemExit("v281 focused summary is invalid")
    if sorted(int(k) for k in outcomes) != [1, 2]:
        raise SystemExit(f"v281 focused outcome keys mismatch: {sorted(outcomes)}")
    if not all(int(row.get("completed", 0)) == 1 for row in outcomes.values()):
        raise SystemExit("v281 focused anchor is not 2/2")

    audit_data = json.loads(audit.read_text())
    if (
        audit_data.get("schema") != "capx_v281_lagged_joint_join_audit_v1"
        or not audit_data.get("planned", {}).get("ok")
    ):
        raise SystemExit("v281 lagged joint-join audit invalid")
    if audit_data["planned"]["max_commanded_arm_delta_rad"] > 0.180001:
        raise SystemExit("v281 audit command bound failed")
    if audit_data["planned"]["max_measured_arm_delta_rad"] > 0.2002:
        raise SystemExit("v281 audit measured bound failed")

    data = {
        "schema": "capx_libero_g17_v281_focused_frozen_anchor_v1",
        "status": "frozen_read_only",
        "score": {"successes": 2, "trials": 2},
        "job_id": "523702",
        "task_id": 0,
        "replicate": 1,
        "boundary": (
            "command-faithful joint homotopy with bounded measured-state join; "
            "K=5 live stability remains unvalidated"
        ),
        "artifacts": {str(path): sha(path) for path in required},
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = ROOT / "manifests/FROZEN_FOCUSED_ANCHOR.json"
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
