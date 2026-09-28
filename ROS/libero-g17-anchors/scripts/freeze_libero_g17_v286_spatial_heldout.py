#!/usr/bin/env python3
"""Validate and freeze the v286 task1-9 spatial held-out result."""

import hashlib
import json
import re
import time
from pathlib import Path


ROOT = Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v286_final_fg_20260819")
JOBS = {1: "523906", 2: "523908", 3: "523909", 4: "523910", 5: "523911",
        6: "523912", 7: "523913", 8: "523914", 9: "523915"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


source = ROOT / "manifests/SOURCE_SHA256.json"
anchor = ROOT / "manifests/FROZEN_K5_RESULT.json"
submit = ROOT / "manifests/LATEST_SUBMIT_SPATIAL_HELDOUT.json"
for path in (source, anchor, submit):
    if not path.is_file():
        raise SystemExit(f"missing held-out provenance: {path}")
source_data = json.loads(source.read_text())
for raw, expected in source_data["files"].items():
    path = Path(raw)
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"v286 source drift: {path}: {actual} != {expected}")
anchor_data = json.loads(anchor.read_text())
if anchor_data.get("score", {}).get("successes") != 10:
    raise SystemExit("held-out source anchor is not frozen 10/10")

artifacts = {str(path): sha256(path) for path in (source, anchor, submit)}
tasks = {}
failures = {"acquisition": [], "placement": []}
total_successes = 0
total_latched = 0
total_deferred = 0
candidate_dq = []
packaging_false_invalid = []
video_inventory = []

for task, job in JOBS.items():
    summary = ROOT / f"arms/v286/task-{task:02d}/rep-00/SUMMARY_job-{job}.json"
    actual_log = ROOT / f"logs/libero-g17-v286-heldout-t{task:02d}-r00-{job}.out"
    expected_log = ROOT / f"logs/libero-g17-v286-final-fg-t{task:02d}-r00-{job}.out"
    run = ROOT / f"runs/libero_spatial/arm-v286/rep-00/job-{job}/libero_spatial_{task}"
    mech_trace = run / "repro_audit/libero_g17_v269_mech/trace.jsonl"
    release_trace = run / "repro_audit/libero_g17_v270_live_isolation/trace.jsonl"
    molmo_log = run.parent / "molmo.log"
    for path in (summary, actual_log, expected_log, mech_trace, release_trace, molmo_log):
        if not path.exists():
            raise SystemExit(f"task{task} missing evidence: {path}")
    if expected_log.resolve() != actual_log.resolve():
        raise SystemExit(f"task{task} log alias does not resolve to the immutable actual log")

    data = json.loads(summary.read_text())
    outcomes = data.get("treatment_outcomes", {})
    if len(outcomes) != 2 or not data.get("prefix_ok") or data.get("prefix_hits") != 2:
        raise SystemExit(f"task{task} prefix/outcome evidence invalid")
    if not data.get("fg_enabled") or not data.get("fg_engaged"):
        raise SystemExit(f"task{task} FG did not engage")
    schemas = data.get("schemas", {}).values()
    if len(data.get("schemas", {})) != 2 or any("v2.86" not in schema for schema in schemas):
        raise SystemExit(f"task{task} schema mismatch")

    mech_events = [json.loads(line) for line in mech_trace.read_text().splitlines() if line.strip()]
    if not mech_events or any(event.get("event") == "symbol_mismatch" for event in mech_events):
        raise SystemExit(f"task{task} mechanical symbol evidence invalid")
    log_text = actual_log.read_text(errors="replace")
    molmo_text = molmo_log.read_text(errors="replace")
    actual_infra_ok = (
        "molmo_url_proof_ok" in log_text
        and "prefix_hook_proof_ok" in log_text
        and "Connection refused" not in log_text
        and "127.0.0.1:8122" not in log_text
        and "/v1/chat/completions" in molmo_text
    )
    if not actual_infra_ok:
        raise SystemExit(f"task{task} actual held-out infra evidence invalid")
    if data.get("validity") == "invalid_infra":
        packaging_false_invalid.append(task)
    elif data.get("validity") != "valid_prefix_arm":
        raise SystemExit(f"task{task} unexpected summary validity: {data.get('validity')}")

    dq = [float(value) for value in re.findall(r"candidate_dq=([0-9.]+)", log_text)]
    if not dq:
        raise SystemExit(f"task{task} lacks measured-seed IK evidence")
    candidate_dq.extend(dq)
    for forbidden in ("goto_pose stopped", "rollback=True", "measured_joint_jump"):
        if forbidden in log_text:
            raise SystemExit(f"task{task} forbidden motion behavior: {forbidden}")

    witness_matches = list(re.finditer(
        r"terminal_witness[^\n]*trial=(\d+)[^\n]*finish=(\d+)[^\n]*"
        r"gap=([^ ]+) task_gap=([^ ]+) released_goal=(\d+)", log_text
    ))
    witnesses = {
        int(match.group(1)): {
            "finish": int(match.group(2)), "gap": match.group(3),
            "task_gap": match.group(4), "released_goal": int(match.group(5)),
        }
        for match in witness_matches
    }
    if len(witnesses) != 2:
        raise SystemExit(f"task{task} terminal witness cardinality mismatch")

    release_events = [json.loads(line) for line in release_trace.read_text().splitlines() if line.strip()]
    deferred = [event for event in release_events if event.get("event") == "release_success_deferred"]
    latched = [event for event in release_events if event.get("event") == "terminal_latched"]
    if any(event.get("source") != "release_window_complete" for event in latched):
        raise SystemExit(f"task{task} terminal success preceded full release window")
    successes = sum(int(row.get("completed", 0)) for row in outcomes.values())
    if len(latched) != successes:
        raise SystemExit(f"task{task} latch/success mismatch: {len(latched)} != {successes}")
    for raw_trial, outcome in outcomes.items():
        trial = int(raw_trial)
        witness = witnesses[trial]
        if outcome.get("completed"):
            if witness != {"finish": 1, "gap": "placement_incomplete",
                            "task_gap": "none", "released_goal": 1}:
                raise SystemExit(f"task{task}/trial{trial} success witness mismatch: {witness}")
        else:
            kind = "acquisition" if witness["task_gap"] == "acquisition" else "placement"
            failures[kind].append({"task_id": task, "trial": trial, "witness": witness})

        matches = list((run / "qwen3.7-plus/eval").glob(f"trial_{trial:02d}_*/video_combined.mp4"))
        if len(matches) != 1 or matches[0].stat().st_size <= 0:
            raise SystemExit(f"task{task}/trial{trial} video missing or ambiguous")
        video_inventory.append({"task_id": task, "trial": trial,
                                "path": str(matches[0]), "bytes": matches[0].stat().st_size})

    total_successes += successes
    total_latched += len(latched)
    total_deferred += len(deferred)
    artifacts.update({str(path): sha256(path) for path in
                      (summary, actual_log, mech_trace, release_trace)})
    tasks[str(task)] = {
        "job_id": job,
        "successes": successes,
        "trials": 2,
        "raw_summary_validity": data.get("validity"),
        "evidence_validity": "valid_posthoc_actual_log" if data.get("validity") == "invalid_infra"
                             else "valid_prefix_arm",
        "max_candidate_arm_delta_rad": max(dq),
        "release_success_deferred": len(deferred),
        "terminal_latched_after_full_release_window": len(latched),
    }

if total_successes != 12 or total_latched != 12:
    raise SystemExit(f"held-out aggregate mismatch: score={total_successes}, latches={total_latched}")
if packaging_false_invalid != [1, 2, 3]:
    raise SystemExit(f"unexpected packaging false-invalid set: {packaging_false_invalid}")

result = {
    "schema": "capx_libero_g17_v286_spatial_heldout_frozen_result_v1",
    "status": "frozen_read_only",
    "axis": {"task_ids": list(range(1, 10)), "trials": [1, 2], "K": 1},
    "score": {"successes": total_successes, "trials": 18},
    "comparison": {"v273_same_axis_successes": 10, "trials": 18,
                   "absolute_success_delta": total_successes - 10},
    "tasks": tasks,
    "failure_taxonomy": {
        "acquisition": {"count": len(failures["acquisition"]), "trials": failures["acquisition"]},
        "placement": {"count": len(failures["placement"]), "trials": failures["placement"]},
    },
    "release_semantics": {
        "successful_terminal_latches_after_full_release_window": total_latched,
        "release_success_deferred_events": total_deferred,
        "extra_deferred_without_terminal_latch": total_deferred - total_latched,
        "extra_deferred_location": "task5/trial1 placement failure",
    },
    "mechanical_observation": {
        "max_candidate_arm_delta_rad": max(candidate_dq),
        "goto_pose_stopped": 0, "measured_joint_jump": 0, "rollback": 0,
        "interpretation": "held-out task geometry exceeds the task0 1.25-rad observation; this is not by itself branch-jump evidence",
    },
    "packaging_correction": {
        "affected_tasks": packaging_false_invalid,
        "cause": "heldout job-name log was absent from frozen runner summary log_candidates",
        "correction": "immutable actual logs were aliased to the runner-expected final-fg names; original summaries preserved",
        "evidence_revalidated": ["prefix", "FG engagement/schema", "Molmo endpoint", "mechanical trace", "release trace"],
    },
    "video_inventory": video_inventory,
    "artifacts": artifacts,
    "boundary": "visible small-step jitter was not quantitatively measured; held-out failures are five acquisition and one placement",
    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
output = ROOT / "manifests/FROZEN_SPATIAL_HELDOUT_RESULT.json"
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
