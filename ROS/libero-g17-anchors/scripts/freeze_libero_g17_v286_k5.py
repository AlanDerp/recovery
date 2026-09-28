#!/usr/bin/env python3
"""Validate and freeze the completed v286 task-0 K=5 live result."""

import hashlib
import json
import re
import time
from pathlib import Path


ROOT = Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v286_final_fg_20260819")
JOBS = {0: "523874", 1: "523865", 2: "523875", 3: "523876", 4: "523877"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


source = ROOT / "manifests/SOURCE_SHA256.json"
snapshot = ROOT / "manifests/SOURCE_SNAPSHOT.zip"
focused = ROOT / "manifests/FROZEN_FOCUSED_ANCHOR.json"
submit = ROOT / "manifests/LATEST_SUBMIT_K5.json"
for path in (source, snapshot, focused, submit):
    if not path.is_file():
        raise SystemExit(f"missing K5 provenance artifact: {path}")

source_data = json.loads(source.read_text())
for raw, expected in source_data["files"].items():
    path = Path(raw)
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"v286 source drift: {path}: {actual} != {expected}")

artifacts = {str(path): sha256(path) for path in (source, snapshot, focused, submit)}
replicates = {}
total_successes = 0
all_candidate_dq = []
all_trial_seconds = []
for replicate, job in JOBS.items():
    summary = ROOT / f"arms/v286/task-00/rep-{replicate:02d}/SUMMARY_job-{job}.json"
    log = ROOT / f"logs/libero-g17-v286-final-fg-t00-r{replicate:02d}-{job}.out"
    run = ROOT / f"runs/libero_spatial/arm-v286/rep-{replicate:02d}/job-{job}/libero_spatial_0"
    trace = run / "repro_audit/libero_g17_v270_live_isolation/trace.jsonl"
    for path in (summary, log, trace):
        if not path.is_file():
            raise SystemExit(f"missing rep{replicate} artifact: {path}")

    data = json.loads(summary.read_text())
    outcomes = data.get("treatment_outcomes", {})
    successes = sum(int(row.get("completed", 0)) for row in outcomes.values())
    if data.get("validity") != "valid_prefix_arm" or len(outcomes) != 2 or successes != 2:
        raise SystemExit(f"rep{replicate} is not valid 2/2")
    if any("v2.86" not in schema or "final-fg-axis-check" not in schema
           for schema in data.get("schemas", {}).values()):
        raise SystemExit(f"rep{replicate} schema mismatch")

    log_text = log.read_text(errors="replace")
    candidate_dq = [float(value) for value in re.findall(r"candidate_dq=([0-9.]+)", log_text)]
    trial_seconds = [float(value) for value in re.findall(r"Trial [12] took ([0-9.]+) seconds", log_text)]
    witnesses = re.findall(r"terminal_witness[^\n]+released_goal=1", log_text)
    if not candidate_dq or len(trial_seconds) != 2 or len(witnesses) != 2:
        raise SystemExit(f"rep{replicate} mechanical/release evidence incomplete")
    for forbidden in ("goto_pose stopped", "rollback=True", "measured_joint_jump"):
        if forbidden in log_text:
            raise SystemExit(f"rep{replicate} forbidden behavior: {forbidden}")

    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    deferred = [event for event in events if event.get("event") == "release_success_deferred"]
    latched = [event for event in events if event.get("event") == "terminal_latched"]
    if len(deferred) != 2 or len(latched) != 2:
        raise SystemExit(f"rep{replicate} release counts mismatch")
    if any(event.get("source") != "release_window_complete" for event in latched):
        raise SystemExit(f"rep{replicate} terminal latch preceded full release window")

    videos = []
    for trial in (1, 2):
        video = run / "qwen3.7-plus/eval" / (
            f"trial_{trial:02d}_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined.mp4"
        )
        if not video.is_file() or video.stat().st_size <= 0:
            raise SystemExit(f"rep{replicate}/trial{trial} video missing")
        videos.append(video)

    for path in (summary, log, trace, *videos):
        artifacts[str(path)] = sha256(path)
    total_successes += successes
    all_candidate_dq.extend(candidate_dq)
    all_trial_seconds.extend(trial_seconds)
    replicates[str(replicate)] = {
        "job_id": job,
        "successes": successes,
        "trials": 2,
        "trial_seconds": trial_seconds,
        "max_candidate_arm_delta_rad": max(candidate_dq),
        "release_success_deferred": len(deferred),
        "terminal_latched_after_full_release_window": len(latched),
        "released_goal_terminal_witnesses": len(witnesses),
    }

if total_successes != 10 or max(all_candidate_dq) > 1.25:
    raise SystemExit(f"K5 acceptance failed: successes={total_successes}, max_dq={max(all_candidate_dq)}")

result = {
    "schema": "capx_libero_g17_v286_k5_frozen_result_v1",
    "status": "frozen_read_only",
    "task_id": 0,
    "score": {"successes": total_successes, "trials": 10, "replicates": 5},
    "replicates": replicates,
    "mechanical": {
        "max_candidate_arm_delta_rad": max(all_candidate_dq),
        "candidate_arm_delta_acceptance_rad": 1.25,
        "cartesian_waypoints": 0,
        "joint_delta_gate": 0,
        "terminal_residual_gate": 0,
        "rollback": 0,
        "native_commands_per_stage": 1,
    },
    "release_semantics": {
        "release_success_deferred": 10,
        "terminal_latched_after_full_release_window": 10,
        "released_goal_terminal_witnesses": 10,
    },
    "artifacts": artifacts,
    "boundary": (
        "task0 K=5 live anchor only; visible small-step approach jitter remains an observed, "
        "unquantified quality issue and was not changed or claimed solved in this run"
    ),
    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
output = ROOT / "manifests/FROZEN_K5_RESULT.json"
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
