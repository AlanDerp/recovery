#!/usr/bin/env python3
"""Register a validated A0 arm as a complete frozen model trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from register_replay_visual_fingerprints import register as register_visuals
from validate_paired_run import (
    CHAIN_BLOCKERS,
    _initial_state_records,
    _semantic_config_sha256,
    _summary_result,
    _trial_results,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            result[key] = value
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern!r} under {root}, found {len(matches)}")
    return matches[0]


def _chain_blockers(run_root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for path in sorted(run_root.rglob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in CHAIN_BLOCKERS.items():
            if pattern.search(text):
                hits.append({"blocker": name, "path": str(path)})
    return hits


def _semantic_hashes(reference: dict[str, Any]) -> tuple[dict[str, Any], str]:
    schema = reference.get("schema")
    if schema == "capx_fgsrp_passive_semantic_reference_v1":
        key = "m3_semantic_transition_sha256"
    elif schema == "capx_closed_gt_takeover_passive_reference_v2.1":
        key = "closed_gt_turn_semantics_sha256"
    else:
        raise ValueError(f"Unsupported M3 passive semantic schema: {schema!r}")
    hashes = reference.get(key)
    if not isinstance(hashes, dict):
        raise ValueError("M3 passive semantic hashes are missing")
    return hashes, key


def register(args: argparse.Namespace) -> dict[str, Any]:
    pair_root = args.pair_root.resolve()
    manifest = _manifest(pair_root / "PAIR_MANIFEST.txt")
    task = manifest.get("task")
    native_mode = manifest.get("native_mode")
    trials = int(manifest.get("trials", "0"))
    global_seed = int(manifest.get("global_seed", "0"))
    if task not in {"cube_stack", "cube_restack"}:
        raise ValueError(f"Unsupported A0 task: {task!r}")
    if native_mode not in {"M2", "M3"}:
        raise ValueError(f"Unsupported A0 native mode: {native_mode!r}")
    if (
        manifest.get("git_status") != "clean"
        or manifest.get("temperature") != "0"
        or manifest.get("num_workers") != "1"
        or trials < 1
    ):
        raise ValueError("A0 source does not satisfy clean deterministic controls")

    run_root = Path(manifest["a1_run_root"]).resolve()
    audit_path = run_root / "qwen_proxy_audit.jsonl"
    audit = _jsonl(audit_path)
    if not audit or any(record.get("ok") is not True for record in audit):
        raise ValueError("A0 source has missing or failed model calls")
    blockers = _chain_blockers(run_root)
    if blockers:
        raise ValueError(f"A0 source has chain blockers: {blockers}")

    exporter = Path(__file__).with_name("export_request_hash_replay.py")
    subprocess.run(
        [
            sys.executable,
            str(exporter),
            "--run-root",
            str(run_root),
            "--output",
            str(args.output_replay),
        ],
        check=True,
    )
    register_visuals(
        args.output_replay,
        task,
        global_seed,
        native_mode=native_mode,
        total_trials=trials,
    )
    replay = json.loads(args.output_replay.read_text(encoding="utf-8"))
    summary_path = _one(run_root, "**/summaries.txt")
    state_path = run_root / "repro_audit" / "initial_states.jsonl"
    states = _initial_state_records(state_path)
    trial_results = _trial_results(run_root / f"{task}.log")
    config_suffix = "vdm" if native_mode == "M3" else "vf"
    config_path = _one(
        run_root / "materialized_configs",
        f"**/franka_robosuite_{task}_multiturn_{config_suffix}.yaml",
    )
    infra_path = pair_root / "INFRA_FINGERPRINT.json"
    infra = json.loads(infra_path.read_text(encoding="utf-8"))
    if (
        len(states) != trials
        or len(trial_results) != trials
        or infra.get("lock_matched") is not True
        or replay.get("missing_count") != 0
    ):
        raise ValueError("A0 source artifact is incomplete")

    pair_validation_path = pair_root / "PAIR_VALIDATION.json"
    if not pair_validation_path.is_file():
        raise ValueError("A0 source lacks PAIR_VALIDATION.json")
    pair_validation_status = json.loads(
        pair_validation_path.read_text(encoding="utf-8")
    ).get("status", "missing")
    if pair_validation_status != "valid":
        raise ValueError("A0 source pair validation is not valid")

    semantic_reference: dict[str, Any] | None = None
    semantic_reference_path = pair_root / "a1_m3_semantic_reference.json"
    if native_mode == "M3":
        if not semantic_reference_path.is_file():
            raise ValueError("M3 A0 source lacks passive semantic transitions")
        semantic_reference = json.loads(
            semantic_reference_path.read_text(encoding="utf-8")
        )
        expected_semantic_identity = {
            "task": task,
            "native_mode": native_mode,
            "global_seed": global_seed,
            "total_trials": trials,
            "source_run_root": str(run_root),
            "capture_mode": "passive_action_invisible",
            "student_prompt_changed": False,
            "student_action_request_changed": False,
        }
        if any(
            semantic_reference.get(key) != value
            for key, value in expected_semantic_identity.items()
        ):
            raise ValueError("M3 passive semantic reference identity mismatch")
        hashes, semantic_hash_key = _semantic_hashes(semantic_reference)
        if (
            sorted(map(int, hashes)) != list(range(1, trials + 1))
            or any(not isinstance(turns, dict) or not turns for turns in hashes.values())
        ):
            raise ValueError("M3 passive semantic transition coverage is incomplete")
        semantic_transition_count = sum(len(turns) for turns in hashes.values())

    initial_offset = 1 if native_mode == "M3" else 0
    required_initial_seeds = {
        global_seed + trial * 1000 + initial_offset
        for trial in range(1, trials + 1)
    }
    initial_records = [
        record
        for record in audit
        if record.get("effective_request_seed") in required_initial_seeds
    ]
    if (
        len(initial_records) != trials
        or {record.get("effective_request_seed") for record in initial_records}
        != required_initial_seeds
    ):
        raise ValueError("A0 initial request trajectory is incomplete")

    source_validation = {
        "schema": "capx_frozen_a0_source_validation_v1",
        "status": "valid",
        "task": task,
        "native_mode": native_mode,
        "global_seed": global_seed,
        "total_trials": trials,
        "source_pair_root": str(pair_root),
        "source_pair_validation": str(pair_validation_path),
        "source_pair_validation_sha256": _sha256(pair_validation_path),
        "source_a0_run_root": str(run_root),
        "source_git_rev": manifest["git_rev"],
        "source_pair_validation_status": pair_validation_status,
        "source_pair_used_as_a0_evidence": False,
        "a0_arm_independently_validated": True,
        "a0_qwen_call_count": len(audit),
        "a0_qwen_failed_call_count": 0,
        "chain_blocker_count": 0,
        "initial_state_record_sha256": [
            record.get("record_sha256") for record in states
        ],
        "materialized_config_semantic_sha256": _semantic_config_sha256(config_path),
        "infra_bundle_sha256": infra["infra_bundle_sha256"],
        "python_packages_sha256": infra["python_packages_sha256"],
        "expected_summary": _summary_result(summary_path),
        "trials": trial_results,
    }
    if semantic_reference is not None:
        source_validation.update(
            {
                "m3_passive_semantic_reference": str(semantic_reference_path),
                "m3_passive_semantic_reference_sha256": _sha256(
                    semantic_reference_path
                ),
                "m3_passive_semantic_schema": semantic_reference["schema"],
                "m3_passive_semantic_hash_key": semantic_hash_key,
                "m3_passive_transition_count": semantic_transition_count,
            }
        )
    args.output_source_validation.parent.mkdir(parents=True, exist_ok=True)
    args.output_source_validation.write_text(
        json.dumps(source_validation, indent=2, sort_keys=True) + "\n"
    )
    reference = {
        "schema": "capx_frozen_a0_reference_v1",
        "task": task,
        "native_mode": native_mode,
        "global_seed": global_seed,
        "total_trials": trials,
        "source_git_rev": manifest["git_rev"],
        "source_pair_job_id": manifest["slurm_job_id"],
        "source_pair_validation_sha256": _sha256(pair_validation_path),
        "source_a0_validation_sha256": _sha256(args.output_source_validation),
        "full_replay_sha256": _sha256(args.output_replay),
        "full_replay_entries": replay["entries_count"],
        "expected_summary": source_validation["expected_summary"],
        "initial_state_record_sha256": source_validation[
            "initial_state_record_sha256"
        ],
        "trials": trial_results,
    }
    if semantic_reference is not None:
        semantic_hashes, semantic_hash_key = _semantic_hashes(semantic_reference)
        reference.update(
            {
                # Generic runner discovery key. Values are copied exactly from
                # the source schema; no semantic projection is performed.
                "m3_semantic_transition_sha256": semantic_hashes,
                "m3_semantic_source_capture_mode": semantic_reference[
                    "capture_mode"
                ],
                "m3_semantic_source_schema": semantic_reference["schema"],
                "m3_semantic_source_hash_key": semantic_hash_key,
                "m3_semantic_source_transition_count": sum(
                    len(turns) for turns in semantic_hashes.values()
                ),
                "m3_semantic_source_reference_sha256": _sha256(
                    semantic_reference_path
                ),
            }
        )
        if semantic_hash_key == "closed_gt_turn_semantics_sha256":
            reference[semantic_hash_key] = semantic_hashes
    args.output_reference.parent.mkdir(parents=True, exist_ok=True)
    args.output_reference.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n"
    )
    return {
        "reference": reference,
        "source_validation": source_validation,
        "replay": str(args.output_replay),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output-replay", type=Path, required=True)
    parser.add_argument("--output-reference", type=Path, required=True)
    parser.add_argument("--output-source-validation", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(register(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
