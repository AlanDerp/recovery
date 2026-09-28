#!/usr/bin/env python3
"""Seal the validated v278 artifacts before v279 development starts."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


ROOT = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/"
    "libero_g17_fair_fg_v278_trusted_se3_20260819"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = ROOT / "manifests/SOURCE_SHA256.json"
    snapshot = ROOT / "manifests/SOURCE_SNAPSHOT.zip"
    audit = ROOT / "evidence/v272_first_goto_trusted_se3.json"
    tests = ROOT / "evidence/unit_tests.txt"
    summaries = {
        "task0_rep1": ROOT / "arms/v278/task-00/rep-01/LATEST_SUMMARY.json",
        "task5_rep0": ROOT / "arms/v278/task-05/rep-00/LATEST_SUMMARY.json",
    }
    required = [source, snapshot, audit, tests, *summaries.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"v278 freeze missing artifacts: {missing}")

    cells = {}
    for name, path in summaries.items():
        summary = json.loads(path.read_text())
        outcomes = summary["treatment_outcomes"]
        if summary["validity"] != "valid_prefix_arm" or not summary["infra_ok"]:
            raise SystemExit(f"v278 invalid summary: {path}")
        if sorted(row["completed"] for row in outcomes.values()) != [1, 1]:
            raise SystemExit(f"v278 non-2/2 summary: {path}")
        cells[name] = {
            "job": summary["treatment_job"],
            "task_id": summary["task_id"],
            "replicate": summary["replicate"],
            "outcomes": outcomes,
            "summary": str(path),
            "summary_sha256": sha256(path),
        }

    exact = json.loads(audit.read_text())
    if not exact["trusted"]["ok"]:
        raise SystemExit("v278 exact replay is not contract-ok")
    record = {
        "schema": "capx_libero_g17_v278_frozen_anchor_v1",
        "status": "frozen_read_only",
        "claim": "trusted SE3 removes the v272 catastrophic IK branch jump",
        "scope_boundary": (
            "does not claim arbitrary commanded SE3 reachability; local joint-limit "
            "and controller-plateau recovery are deferred to v279"
        ),
        "aggregate": {"completed": 4, "trials": 4},
        "cells": cells,
        "source_manifest": str(source),
        "source_manifest_sha256": sha256(source),
        "source_snapshot": str(snapshot),
        "source_snapshot_sha256": sha256(snapshot),
        "exact_replay": str(audit),
        "exact_replay_sha256": sha256(audit),
        "unit_tests": str(tests),
        "unit_tests_sha256": sha256(tests),
        "exact_replay_metrics": {
            "legacy_max_arm_joint_delta_rad": exact["legacy"][
                "max_arm_joint_delta_rad"
            ],
            "legacy_endpoint_fk_position_error_m": exact["legacy"][
                "endpoint_fk_position_error_m"
            ],
            "trusted_waypoint_count": exact["trusted"]["waypoint_count"],
            "trusted_max_arm_joint_delta_rad": exact["trusted"][
                "max_arm_joint_delta_rad"
            ],
            "trusted_min_signed_progress_m": exact["trusted"][
                "min_signed_progress_m"
            ],
            "trusted_final_position_error_m": exact["trusted"][
                "final_position_error_m"
            ],
        },
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = ROOT / "manifests/FROZEN_ANCHOR.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
