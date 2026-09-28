#!/usr/bin/env python3
"""Register one valid live FGSRP trajectory for exact replay verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from validate_paired_run import _initial_state_records, _summary_result, _trial_results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key] = value
    return fields


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object audit record in {path}")
            records.append(value)
    return records


def _one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern!r} under {root}, found {len(matches)}")
    return matches[0]


def _trace_signatures(run_root: Path) -> dict[str, list[dict[str, Any]]]:
    signatures: dict[str, list[dict[str, Any]]] = {}
    for trial_root in sorted(run_root.glob("**/fgsrp_traces/trial_*")):
        trial = str(int(trial_root.name.split("_")[-1]))
        turns: list[dict[str, Any]] = []
        for path in sorted(trial_root.glob("turn_*/replacement.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            turns.append(
                {
                    "turn": record.get("turn"),
                    "native_prompt_sha256": record.get("native_prompt_sha256"),
                    "corrected_prompt_sha256": record.get("corrected_prompt_sha256"),
                    "repair_count": record.get("repair_count"),
                    "before_evidence_sha256": record.get("before_evidence_sha256"),
                    "after_evidence_sha256": record.get("after_evidence_sha256"),
                    "temporal_evidence_sha256": record.get("temporal_evidence_sha256"),
                }
            )
        signatures[trial] = turns
    return signatures


def register(args: argparse.Namespace) -> dict[str, Any]:
    pair_root = args.pair_root.resolve()
    manifest_path = pair_root / "PAIR_MANIFEST.txt"
    validation_path = pair_root / "PAIR_VALIDATION.json"
    manifest = _manifest(manifest_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "valid" or validation.get("treatment") != (
        "FGSRP-v2-semantic-ir-alignment"
    ):
        raise ValueError("Only a valid FGSRP-v2 pair may be registered")
    if manifest.get("treatment_replay_mode", "registration_live") != "registration_live":
        raise ValueError("A replay-verification run cannot be registered as a live source")

    run_root = Path(manifest["treatment_run_root"]).resolve()
    audit_path = run_root / "qwen_proxy_audit.jsonl"
    audit = _jsonl(audit_path)
    if not audit or any(record.get("ok") is not True for record in audit):
        raise ValueError("Treatment source has missing or failed model calls")

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
    replay = json.loads(args.output_replay.read_text(encoding="utf-8"))
    summary_path = _one(run_root, "**/summaries.txt")
    state_path = run_root / "repro_audit" / "initial_states.jsonl"
    task = manifest["task"]
    trials = int(manifest["trials"])
    trial_results = _trial_results(run_root / f"{task}.log")
    if len(trial_results) != trials:
        raise ValueError("Treatment source trial-result count is incomplete")

    request_response_sequence = [
        {
            "request_sha256": record.get("request_sha256"),
            "request_normalized_sha256": record.get("request_normalized_sha256"),
            "response_content_sha256": record.get("response_content_sha256"),
            "effective_request_seed": record.get("effective_request_seed"),
        }
        for record in audit
    ]
    reference = {
        "schema": "capx_fgsrp_treatment_reference_v1",
        "task": task,
        "carrier": manifest["fgsrp_carrier"],
        "native_mode": manifest["native_mode"],
        "global_seed": int(manifest["global_seed"]),
        "fixed_episode_seed": manifest.get("fixed_episode_seed", "none"),
        "total_trials": trials,
        "source_git_rev": manifest["git_rev"],
        "source_pair_job_id": manifest["slurm_job_id"],
        "source_pair_root": str(pair_root),
        "source_pair_validation_sha256": _sha256(validation_path),
        "source_treatment_run_root": str(run_root),
        "full_replay_sha256": _sha256(args.output_replay),
        "full_replay_entries": replay["entries_count"],
        "model_call_count": len(audit),
        "request_response_sequence": request_response_sequence,
        "expected_summary": _summary_result(summary_path),
        "initial_state_record_sha256": [
            record.get("record_sha256") for record in _initial_state_records(state_path)
        ],
        "trials": trial_results,
        "trace_signatures": _trace_signatures(run_root),
        "treatment_implementation_sha256": manifest["treatment_implementation_sha256"],
        "model": manifest["model"],
        "qwen_base_url": manifest["qwen_base_url"],
        "materialized_config_semantic_sha256": validation[
            "materialized_config_semantic_sha256"
        ],
        "frozen_a0_reference_sha256": manifest.get(
            "frozen_a0_reference_sha256", "not_applicable"
        ),
        "verifier_prompt_sha256": manifest["verifier_prompt_sha256"],
        "verifier_schema_sha256": manifest["verifier_schema_sha256"],
        "adjudicator_prompt_sha256": manifest["adjudicator_prompt_sha256"],
        "adjudicator_schema_sha256": manifest["adjudicator_schema_sha256"],
        "infra_bundle_sha256": validation["infra_bundle_sha256"],
        "python_packages_sha256": validation["python_packages_sha256"],
    }
    args.output_reference.parent.mkdir(parents=True, exist_ok=True)
    args.output_reference.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output-replay", type=Path, required=True)
    parser.add_argument("--output-reference", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(register(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
