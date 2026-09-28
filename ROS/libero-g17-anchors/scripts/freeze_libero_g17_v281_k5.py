#!/usr/bin/env python3
"""Seal the v281 fair live K=5 result and its mechanical evidence."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path


ROOT = Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/"
    "libero_g17_fair_fg_v281_joint_join_20260819"
)
JOBS = {0: "523727", 1: "523702", 2: "523728", 3: "523729", 4: "523730"}
MEASURED_DQ_LIMIT = 0.2002


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = ROOT / "manifests/SOURCE_SHA256.json"
    snapshot = ROOT / "manifests/SOURCE_SNAPSHOT.zip"
    focused = ROOT / "manifests/FROZEN_FOCUSED_ANCHOR.json"
    submit = ROOT / "manifests/LATEST_SUBMIT_K5.json"
    required = [source, snapshot, focused, submit]
    rows = []

    source_data = json.loads(source.read_text())
    for raw_path, expected in source_data["files"].items():
        path = Path(raw_path)
        actual = sha(path)
        if actual != expected:
            raise SystemExit(f"v281 source drift: {path} {actual} != {expected}")

    for rep, job in JOBS.items():
        summary = ROOT / f"arms/v281/task-00/rep-{rep:02d}/SUMMARY_job-{job}.json"
        logs = sorted((ROOT / "logs").glob(f"*-{job}.out"))
        if not summary.is_file() or len(logs) != 1:
            raise SystemExit(f"missing or ambiguous artifacts for rep={rep} job={job}")
        log = logs[0]
        required.extend((summary, log))

        summary_data = json.loads(summary.read_text())
        outcomes = summary_data.get("treatment_outcomes", {})
        if summary_data.get("validity") != "valid_prefix_arm":
            raise SystemExit(f"invalid summary for rep={rep} job={job}")
        if sorted(int(key) for key in outcomes) != [1, 2]:
            raise SystemExit(f"incomplete outcomes for rep={rep} job={job}")

        text = log.read_text(errors="replace")
        measured = [
            float(value)
            for value in re.findall(r"max_measured_dq=([0-9]+(?:\.[0-9]+)?)", text)
        ]
        max_measured = max(measured, default=0.0)
        jump_hits = text.count("measured_joint_jump")
        if jump_hits:
            raise SystemExit(f"measured joint jump in rep={rep} job={job}")
        if max_measured > MEASURED_DQ_LIMIT:
            raise SystemExit(
                f"measured dq bound failed in rep={rep}: "
                f"{max_measured} > {MEASURED_DQ_LIMIT}"
            )

        rows.append(
            {
                "replicate": rep,
                "job_id": job,
                "successes": sum(
                    int(outcome.get("completed", 0)) for outcome in outcomes.values()
                ),
                "trials": 2,
                "validity": summary_data["validity"],
                "max_measured_dq_rad": max_measured,
                "measured_joint_jump_hits": jump_hits,
                "controlled_stop_hits": text.count("goto_pose stopped"),
                "release_refusal_hits": len(
                    re.findall(r"(?:release-ready refused|open_gripper refused)", text)
                ),
            }
        )

    successes = sum(row["successes"] for row in rows)
    if successes < 9:
        raise SystemExit(f"v281 K=5 acceptance failed: {successes}/10 < 9/10")

    data = {
        "schema": "capx_libero_g17_v281_k5_frozen_anchor_v1",
        "status": "frozen_read_only",
        "task_id": 0,
        "score": {"successes": successes, "trials": 10},
        "acceptance": {
            "performance": "9-10/10",
            "zero_measured_joint_jumps": True,
            "max_measured_dq_limit_rad": MEASURED_DQ_LIMIT,
        },
        "mechanical_result": {
            "max_measured_dq_rad": max(row["max_measured_dq_rad"] for row in rows),
            "measured_joint_jump_hits": sum(
                row["measured_joint_jump_hits"] for row in rows
            ),
            "controlled_stop_hits": sum(row["controlled_stop_hits"] for row in rows),
            "release_refusal_hits": sum(row["release_refusal_hits"] for row in rows),
        },
        "replicates": rows,
        "boundary": (
            "IK branch-jump acceptance passed on this K=5 live set; the sole failure "
            "is a post-release placement_xy miss, so RR placement reliability remains "
            "the next optimization target."
        ),
        "artifacts": {str(path): sha(path) for path in required},
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = ROOT / "manifests/FROZEN_K5_ANCHOR.json"
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
