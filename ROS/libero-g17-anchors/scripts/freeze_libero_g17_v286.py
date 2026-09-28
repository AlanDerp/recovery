#!/usr/bin/env python3
"""Freeze the v286 focused 2/2 anchor before K=5 expansion."""

import hashlib
import json
import re
import time
from pathlib import Path


ROOT = Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v286_final_fg_20260819")
JOB = "523865"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


source = ROOT / "manifests/SOURCE_SHA256.json"
snapshot = ROOT / "manifests/SOURCE_SNAPSHOT.zip"
summary = ROOT / f"arms/v286/task-00/rep-01/SUMMARY_job-{JOB}.json"
log = ROOT / f"logs/libero-g17-v286-final-fg-t00-r01-{JOB}.out"
audit = ROOT / "evidence/v272_trial2_tool_pi_ik.json"
tests = ROOT / "evidence/unit_tests.txt"
run = ROOT / f"runs/libero_spatial/arm-v286/rep-01/job-{JOB}/libero_spatial_0"
trace = run / "repro_audit/libero_g17_v270_live_isolation/trace.jsonl"

required = (source, snapshot, summary, log, audit, tests, trace)
for path in required:
    if not path.is_file():
        raise SystemExit(f"missing focused artifact: {path}")

source_data = json.loads(source.read_text())
if source_data.get("schema") != "capx_libero_g17_v286_final_fg_source_v1":
    raise SystemExit("wrong v286 source schema")
for raw, expected in source_data["files"].items():
    path = Path(raw)
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"source drift: {path}: {actual} != {expected}")

summary_data = json.loads(summary.read_text())
outcomes = summary_data.get("treatment_outcomes", {})
successes = sum(int(row.get("completed", 0)) for row in outcomes.values())
if summary_data.get("validity") != "valid_prefix_arm" or successes != 2 or len(outcomes) != 2:
    raise SystemExit("v286 focused result is not a valid 2/2")
schemas = set(summary_data.get("schemas", {}).values())
if len(schemas) != 1 or not all("v2.86" in s and "final-fg-axis-check" in s for s in schemas):
    raise SystemExit(f"unexpected v286 schemas: {schemas}")

log_text = log.read_text(errors="replace")
candidate_dq = [float(value) for value in re.findall(r"candidate_dq=([0-9.]+)", log_text)]
trial_seconds = [float(value) for value in re.findall(r"Trial [12] took ([0-9.]+) seconds", log_text)]
if not candidate_dq or len(trial_seconds) != 2:
    raise SystemExit("focused mechanical evidence is incomplete")
for forbidden in ("goto_pose stopped", "rollback=True", "measured_joint_jump"):
    if forbidden in log_text:
        raise SystemExit(f"forbidden live behavior: {forbidden}")
terminal_witnesses = re.findall(r"terminal_witness[^\n]+released_goal=1", log_text)
if len(terminal_witnesses) != 2:
    raise SystemExit(f"expected two released-goal terminal witnesses, got {len(terminal_witnesses)}")

events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
deferred = [event for event in events if event.get("event") == "release_success_deferred"]
latched = [event for event in events if event.get("event") == "terminal_latched"]
if len(deferred) != 2 or len(latched) != 2:
    raise SystemExit(f"release evidence mismatch: deferred={len(deferred)} latched={len(latched)}")
if any(event.get("source") != "release_window_complete" for event in latched):
    raise SystemExit(f"terminal latched before full release window: {latched}")

videos = []
for trial in (1, 2):
    matches = list((run / "qwen3.7-plus/eval").glob(
        f"trial_{trial:02d}_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined.mp4"
    ))
    if len(matches) != 1 or matches[0].stat().st_size <= 0:
        raise SystemExit(f"missing successful trial {trial} video")
    videos.append(matches[0])

data = {
    "schema": "capx_libero_g17_v286_focused_frozen_anchor_v1",
    "status": "frozen_read_only",
    "score": {"successes": 2, "trials": 2},
    "job_id": JOB,
    "task_id": 0,
    "replicate": 1,
    "release_semantics": {
        "release_success_deferred": len(deferred),
        "terminal_latched": len(latched),
        "terminal_latch_source": "release_window_complete",
        "released_goal_terminal_witnesses": len(terminal_witnesses),
    },
    "mechanical": {
        "max_candidate_arm_delta_rad": max(candidate_dq),
        "trial_seconds": trial_seconds,
        "cartesian_waypoints": 0,
        "joint_delta_gate": 0,
        "terminal_residual_gate": 0,
        "rollback": 0,
        "native_commands_per_stage": 1,
    },
    "artifacts": {str(path): sha256(path) for path in required + tuple(videos)},
    "boundary": (
        "focused task0/rep1 2/2 anchor; the final-FG fallback is regression-tested but was not "
        "causally exercised in the successful live trace; K=5 and jitter generality remain unvalidated"
    ),
    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
output = ROOT / "manifests/FROZEN_FOCUSED_ANCHOR.json"
output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(json.dumps(data, indent=2, sort_keys=True))
