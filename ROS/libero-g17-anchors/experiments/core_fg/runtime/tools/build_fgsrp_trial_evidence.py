#!/usr/bin/env python3
"""Build fail-closed, seed-level FGSRP attribution evidence packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from validate_paired_run import _trial_results


def _manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            result[key] = value
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _redact_images(value: Any) -> Any:
    if isinstance(value, dict):
        result = {key: _redact_images(child) for key, child in value.items()}
        image = result.get("image_url")
        if isinstance(image, dict) and isinstance(image.get("url"), str):
            url = image["url"]
            image["url"] = {
                "redacted": True,
                "encoded_sha256": hashlib.sha256(url.encode()).hexdigest(),
            }
        return result
    if isinstance(value, list):
        return [_redact_images(item) for item in value]
    return value


def _audit_by_message_hash(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        canonical = record.get("request_canonical")
        if not isinstance(canonical, str):
            continue
        request = json.loads(canonical)
        messages = request.get("messages")
        if isinstance(messages, list):
            result[_sha256_json(messages)] = record
    return result


def _final_trial_dir(run_root: Path, trial: int) -> Path | None:
    matches = list(run_root.glob(f"**/trial_{trial:02d}_*/video_combined.mp4"))
    if len(matches) == 1:
        return matches[0].parent
    return None


def _execution_evidence(run_root: Path, trial: int) -> dict[str, Any]:
    trial_dir = _final_trial_dir(run_root, trial)
    if trial_dir is None:
        return {"final_trial_dir": None, "summary_available": False}
    summary_path = trial_dir / "summary.txt"
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, Any] = {
        "final_trial_dir": str(trial_dir),
        "summary": str(summary_path),
        "video": str(trial_dir / "video_combined.mp4"),
        "summary_available": True,
    }
    patterns = {
        "terminated": r"Terminated:\s*(True|False)",
        "truncated": r"Truncated:\s*(True|False)",
        "regenerations": r"Num Regenerations:\s*(\d+)",
        "code_blocks": r"Num Code Blocks:\s*(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            raw = match.group(1)
            values[key] = raw == "True" if raw in {"True", "False"} else int(raw)
    return values


def build(args: argparse.Namespace) -> dict[str, Any]:
    pair_root = args.pair_root.resolve()
    manifest = _manifest(pair_root / "PAIR_MANIFEST.txt")
    validation = json.loads((pair_root / "PAIR_VALIDATION.json").read_text())
    if validation.get("status") != "valid":
        raise ValueError("P1-P6 attribution is forbidden for a P0-invalid pair")
    a1_root = Path(manifest["a1_run_root"])
    treatment_root = Path(manifest["treatment_run_root"])
    task = manifest["task"]
    trials = int(manifest["trials"])
    a1_results = _trial_results(a1_root / f"{task}.log")
    treatment_results = _trial_results(treatment_root / f"{task}.log")
    audit = _jsonl(treatment_root / "qwen_proxy_audit.jsonl")
    by_message_hash = _audit_by_message_hash(audit)
    packets: list[dict[str, Any]] = []

    for trial in range(1, trials + 1):
        trace_root = next(
            iter(treatment_root.glob(f"**/fgsrp_traces/trial_{trial:02d}")), None
        )
        if trace_root is None:
            raise ValueError(f"Missing FGSRP trace for trial {trial}")
        turns: list[dict[str, Any]] = []
        first_repair_turn: int | None = None
        for path in sorted(trace_root.glob("turn_*/replacement.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("repair_count", 0) and first_repair_turn is None:
                first_repair_turn = int(record["turn"])
            action = by_message_hash.get(str(record["corrected_prompt_sha256"]))
            action_request = None
            if action is not None and isinstance(action.get("request_canonical"), str):
                action_request = _redact_images(json.loads(action["request_canonical"]))
            verifier = record["verification_protocol"]["verifier"]
            claims = [
                claim
                for sentence in verifier["claim_ledger"]
                for claim in sentence["claims"]
            ]
            turns.append(
                {
                    "turn": record["turn"],
                    "native_prompt_sha256": record["native_prompt_sha256"],
                    "corrected_prompt_sha256": record["corrected_prompt_sha256"],
                    "repair_count": record["repair_count"],
                    "repairs": record["repairs"],
                    "claim_dispositions": {
                        name: sum(claim.get("disposition") == name for claim in claims)
                        for name in (
                            "preserve_supported",
                            "preserve_compatible_or_undecidable",
                            "preserve_non_editable",
                            "replace_contradicted",
                        )
                    },
                    "known_false_claims": record["verification_protocol"][
                        "known_false_claims"
                    ],
                    "private_temporal_evidence": record["private_temporal_evidence"],
                    "action_api": (
                        None
                        if action is None
                        else {
                            "request_sha256": action.get("request_sha256"),
                            "response_content_sha256": action.get(
                                "response_content_sha256"
                            ),
                            "response_content": action.get("response_content"),
                            "source": action.get("source", "live_upstream"),
                            "request_without_raw_images": action_request,
                        }
                    ),
                }
            )
        a1 = a1_results[str(trial)]
        treatment = treatment_results[str(trial)]
        failed = treatment["task_completed"] is not True
        packet = {
            "trial": trial,
            "a1_result": a1,
            "treatment_result": treatment,
            "comparison": (
                "gain"
                if treatment["task_completed"] and not a1["task_completed"]
                else "harm"
                if a1["task_completed"] and not treatment["task_completed"]
                else "tie"
            ),
            "first_report_repair_turn": first_repair_turn,
            "turns": turns,
            "execution_and_budget_evidence": _execution_evidence(
                treatment_root, trial
            ),
            "failure_attribution": {
                "required": failed,
                "primary": None if not failed else "unresolved",
                "machine_certified": [
                    "P0 chain/provenance validity",
                    "P2 structural replacement invariants",
                ],
                "review_order": (
                    []
                    if not failed
                    else ["P1", "P2-semantic", "P3", "P4", "P5", "P7", "P6"]
                ),
                "p6_admissible": False,
                "rule": (
                    "P6 is inadmissible until GT semantics, replacement semantics, "
                    "report completeness, logical planning, physical execution, and "
                    "budget exhaustion have each been excluded with cited turn evidence."
                ),
            },
        }
        packets.append(packet)

    return {
        "schema": "capx_fgsrp_trial_evidence_v1",
        "pair_root": str(pair_root),
        "pair_validation": str(pair_root / "PAIR_VALIDATION.json"),
        "pair_status": "valid",
        "task": task,
        "trials": trials,
        "treatment_replay_mode": manifest.get(
            "treatment_replay_mode", "registration_live"
        ),
        "attribution_complete": all(
            not item["failure_attribution"]["required"]
            or item["failure_attribution"]["primary"] != "unresolved"
            for item in packets
        ),
        "trial_evidence": packets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "attribution_complete": result["attribution_complete"],
        "trials": result["trials"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
