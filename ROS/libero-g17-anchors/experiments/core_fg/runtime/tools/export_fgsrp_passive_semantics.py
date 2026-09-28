#!/usr/bin/env python3
"""Export action-invisible A1 GT transitions for M3 replay certification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TRACE_DIR = "fgsrp_passive_traces"
STATE_FILE = "fgsrp_state.json"
FGSRP_SCHEMA = "capx-fgsrp-v2.0-ir-alignment"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def semantic_evidence_sha256(record: dict[str, Any]) -> str:
    evidence = record.get("private_temporal_evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("Passive trace lacks private temporal evidence")
    vector: dict[str, dict[str, Any]] = {}
    for evidence_id, item in sorted(evidence.items()):
        if not isinstance(item, dict):
            raise ValueError(f"Malformed semantic evidence {evidence_id!r}")
        if evidence_id.startswith("transition."):
            vector[evidence_id] = {
                "before_value": item.get("before_value"),
                "after_value": item.get("after_value"),
                "changed": item.get("changed"),
            }
        else:
            vector[evidence_id] = {"value": item.get("value")}
    return _sha256_json(vector)


def export(
    *,
    run_root: Path,
    output: Path,
    task: str,
    total_trials: int,
    global_seed: int,
) -> dict[str, Any]:
    roots = sorted(run_root.glob(f"**/{TRACE_DIR}/trial_*"))
    if len(roots) != total_trials:
        raise ValueError(
            f"Expected {total_trials} passive traces, found {len(roots)}"
        )
    semantic_hashes: dict[str, dict[str, str]] = {}
    transition_count = 0
    for root in roots:
        trial = int(root.name.split("_")[-1])
        if trial not in range(1, total_trials + 1) or str(trial) in semantic_hashes:
            raise ValueError(f"Invalid or duplicate passive trial: {root}")
        state = json.loads((root / STATE_FILE).read_text(encoding="utf-8"))
        snapshots = state.get("snapshots")
        if (
            state.get("schema_version") != FGSRP_SCHEMA
            or state.get("mode") != "passive_action_invisible"
            or state.get("task") != task
            or state.get("trial") != trial
            or not isinstance(snapshots, list)
            or not snapshots
        ):
            raise ValueError(f"Malformed passive state: {root}")
        records = []
        for path in sorted(root.glob("turn_*/transition.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            records.append(record)
        if len(snapshots) != len(records) + 1:
            raise ValueError(f"Passive snapshots are not turn-synchronous: {root}")
        semantic_hashes[str(trial)] = {}
        for expected_turn, record in enumerate(records, 1):
            if (
                record.get("schema_version") != FGSRP_SCHEMA
                or record.get("mode") != "passive_action_invisible"
                or record.get("task") != task
                or record.get("trial") != trial
                or record.get("turn") != expected_turn
                or record.get("action_visible_gt_packet_injected") is not False
                or record.get("action_visible_bridge_injected") is not False
                or record.get("action_request_changed") is not False
            ):
                raise ValueError(f"Malformed passive transition: {root}")
            actual_temporal_hash = _sha256_json(record["private_temporal_evidence"])
            if record.get("temporal_evidence_sha256") != actual_temporal_hash:
                raise ValueError(f"Passive temporal evidence hash mismatch: {root}")
            semantic_hashes[str(trial)][str(expected_turn)] = (
                semantic_evidence_sha256(record)
            )
            transition_count += 1
    if sorted(map(int, semantic_hashes)) != list(range(1, total_trials + 1)):
        raise ValueError("Passive trace trial coverage is incomplete")
    package = {
        "schema": "capx_fgsrp_passive_semantic_reference_v1",
        "task": task,
        "native_mode": "M3",
        "global_seed": global_seed,
        "total_trials": total_trials,
        "source_run_root": str(run_root.resolve()),
        "capture_mode": "passive_action_invisible",
        "student_prompt_changed": False,
        "student_action_request_changed": False,
        "transition_count": transition_count,
        "m3_semantic_transition_sha256": semantic_hashes,
    }
    output.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    return package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--total-trials", type=int, required=True)
    parser.add_argument("--global-seed", type=int, required=True)
    args = parser.parse_args()
    package = export(
        run_root=args.run_root,
        output=args.output,
        task=args.task,
        total_trials=args.total_trials,
        global_seed=args.global_seed,
    )
    print(json.dumps(package, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
