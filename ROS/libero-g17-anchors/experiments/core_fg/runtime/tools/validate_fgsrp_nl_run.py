#!/usr/bin/env python3
"""Fail-closed validation for an FGSRP-NL paired run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from export_fgsrp_passive_semantics import semantic_evidence_sha256
from validate_fgsrp_run import _jsonl, _manifest, _message_hashes, _sha256_file
from validate_paired_run import validate as validate_pair

TRACE_DIR = "fgsrp_nl_traces"
SCHEMA_VERSION = "capx-fgsrp-nl-v1"
TREATMENT_LABEL = "FGSRP-NL-v1"


def _validate_trace(
    root: Path,
    *,
    carrier: str,
    action_message_hashes: Counter[str],
) -> dict[str, int]:
    state = json.loads((root / "fgsrp_nl_state.json").read_text(encoding="utf-8"))
    initialization = json.loads(
        (root / "initialization.json").read_text(encoding="utf-8")
    )
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"FGSRP-NL state schema mismatch: {root}")
    if initialization.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"FGSRP-NL initialization schema mismatch: {root}")
    if initialization.get("action_visible_gt_packet_injected") is not False:
        raise ValueError("FGSRP-NL initialization exposed GT")
    if initialization.get("action_visible_bridge_injected") is not False:
        raise ValueError("FGSRP-NL initialization exposed a bridge")

    paths = sorted(root.glob("turn_*/replacement.json"))
    snapshots = state.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != len(paths) + 1:
        raise ValueError("FGSRP-NL state is not turn-synchronous")
    totals = Counter()
    for expected_turn, path in enumerate(paths, 1):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"FGSRP-NL replacement schema mismatch: {path}")
        if record.get("turn") != expected_turn or record.get("carrier") != carrier:
            raise ValueError(f"FGSRP-NL carrier/turn mismatch: {path}")
        if record.get("non_text_structure_identical") is not True:
            raise ValueError(f"FGSRP-NL changed request structure: {path}")
        for field in (
            "action_visible_gt_packet_injected",
            "action_visible_bridge_injected",
            "action_visible_verifier_protocol_injected",
            "raw_image_replaced",
            "console_stdout_replaced",
            "console_stderr_replaced",
            "task_or_api_text_replaced",
        ):
            if record.get(field) is not False:
                raise ValueError(f"FGSRP-NL immutable-surface violation {field}: {path}")
        corrected_hash = record.get("corrected_prompt_sha256")
        if not isinstance(corrected_hash, str) or action_message_hashes[corrected_hash] < 1:
            raise ValueError(f"FGSRP-NL corrected request was not sent: {path}")
        repairs = record.get("repairs")
        patches = record.get("applied_patches")
        if not isinstance(repairs, list) or not isinstance(patches, list):
            raise ValueError(f"Malformed FGSRP-NL repair audit: {path}")
        if len(repairs) != len(patches) or record.get("repair_count") != len(repairs):
            raise ValueError(f"FGSRP-NL repair/patch count mismatch: {path}")
        protocol = record.get("verification_protocol")
        if not isinstance(protocol, dict):
            raise ValueError(f"Missing FGSRP-NL protocol: {path}")
        if protocol.get("status") == "completed":
            if protocol.get("host_semantic_judgment") is not False:
                raise ValueError(f"FGSRP-NL host performed semantic judgment: {path}")
            if protocol.get("host_alias_matching") is not False:
                raise ValueError(f"FGSRP-NL host performed alias matching: {path}")
            if protocol.get("host_predicate_matching") is not False:
                raise ValueError(f"FGSRP-NL host performed predicate matching: {path}")
        statements = record.get("private_gt_natural_language_observations")
        if not isinstance(statements, list) or not statements:
            raise ValueError(f"FGSRP-NL lacks private NL evidence: {path}")
        statement_set = set(statements)
        for repair in repairs:
            quotes = repair.get("evidence_quotes")
            if not isinstance(quotes, list) or not quotes:
                raise ValueError(f"FGSRP-NL repair lacks evidence quotes: {path}")
            if any(quote not in statement_set for quote in quotes):
                raise ValueError(f"FGSRP-NL repair cites foreign evidence: {path}")
        if not repairs:
            if record.get("native_prompt_sha256") != corrected_hash:
                raise ValueError(f"FGSRP-NL no-op changed the prompt: {path}")
            if record.get("no_op_prompt_byte_equivalent") is not True:
                raise ValueError(f"FGSRP-NL no-op lacks byte certificate: {path}")
            totals["no_op_count"] += 1
        totals["repair_count"] += len(repairs)
        totals["turn_count"] += 1
    return dict(totals)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _manifest(args.pair_root / "PAIR_MANIFEST.txt")
    if manifest.get("treatment") != "fgsrp_nl":
        raise ValueError("Pair manifest did not select FGSRP-NL")
    if manifest.get("treatment_label") != TREATMENT_LABEL:
        raise ValueError("Pair manifest has the wrong FGSRP-NL label")
    if manifest.get("control_runtime_git_rev") != manifest.get("git_rev"):
        raise ValueError("FGSRP-NL control/runtime revision mismatch")
    carrier = manifest.get("fgsrp_carrier")
    if carrier not in {"m2_comments", "m3_visual_difference"}:
        raise ValueError(f"Invalid FGSRP-NL carrier: {carrier!r}")
    expected_mode = "M2" if carrier == "m2_comments" else "M3"
    if manifest.get("native_mode") != expected_mode:
        raise ValueError("FGSRP-NL native mode/carrier mismatch")
    offset = 0 if expected_mode == "M2" else 1
    fixed_text = manifest.get("fixed_episode_seed", "none")
    frozen_reference = args.frozen_a0_reference
    semantic_reference = args.m3_semantic_reference or frozen_reference
    semantic_value = (
        json.loads(semantic_reference.read_text(encoding="utf-8"))
        if semantic_reference is not None
        else {}
    )
    has_semantic_reference = isinstance(
        semantic_value.get("m3_semantic_transition_sha256"), dict
    )
    base = validate_pair(
        SimpleNamespace(
            pair_root=args.pair_root,
            a1_run_root=args.a1_run_root,
            core_run_root=args.treatment_run_root,
            task=args.task,
            trials=args.trials,
            global_seed=args.global_seed,
            initial_query_seed_offset=offset,
            config_suffix="vf" if expected_mode == "M2" else "vdm",
            frozen_a0_reference=frozen_reference,
            frozen_a0_reference_only=bool(frozen_reference),
            fixed_episode_seed=None if fixed_text == "none" else int(fixed_text),
            allow_deferred_semantic_replay=(
                expected_mode == "M3" and has_semantic_reference
            ),
        )
    )
    roots = sorted(args.treatment_run_root.glob(f"**/{TRACE_DIR}/trial_*"))
    if len(roots) != args.trials:
        raise ValueError(f"Expected {args.trials} FGSRP-NL traces, found {len(roots)}")
    request_hashes = _message_hashes(
        args.treatment_run_root / "qwen_proxy_audit.jsonl"
    )
    totals = Counter()
    semantic_hashes: dict[str, dict[str, str]] = {}
    for root in roots:
        totals.update(
            _validate_trace(
                root,
                carrier=(
                    "m2_code_comments"
                    if carrier == "m2_comments"
                    else "m3_visual_difference_report"
                ),
                action_message_hashes=request_hashes,
            )
        )
        trial = str(int(root.name.split("_")[-1]))
        semantic_hashes[trial] = {}
        for path in sorted(root.glob("turn_*/replacement.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            semantic_hashes[trial][str(int(record["turn"]))] = (
                semantic_evidence_sha256(record)
            )

    certified: list[dict[str, Any]] = []
    deferred = base.get("deferred_semantic_replay_seeds") or []
    if deferred:
        expected_hashes = semantic_value.get("m3_semantic_transition_sha256")
        if not isinstance(expected_hashes, dict):
            raise ValueError("Deferred M3 replay lacks semantic hashes")
        records = _jsonl(args.treatment_run_root / "qwen_proxy_audit.jsonl")
        positions = {
            int(item["effective_request_seed"]): (
                str(int(item["replay_trial"])),
                str(int(item["replay_turn"]) // 2),
            )
            for item in records
            if item.get("normalized_replay_visual_match_deferred_to_semantics") is True
        }
        for seed in deferred:
            position = positions.get(int(seed))
            if position is None:
                raise ValueError(f"Missing deferred M3 position: {seed}")
            trial, turn = position
            actual = semantic_hashes.get(trial, {}).get(turn)
            expected = expected_hashes.get(trial, {}).get(turn)
            if not actual or actual != expected:
                raise ValueError(f"Deferred M3 semantic mismatch: seed={seed}")
            certified.append({"seed": seed, "trial": trial, "turn": turn})

    return {
        **base,
        "schema": "capx_fgsrp_nl_pair_validation_v1",
        "status": "valid",
        "treatment": TREATMENT_LABEL,
        "carrier": carrier,
        "native_mode": expected_mode,
        "fgsrp_nl_trial_trace_count": len(roots),
        "fgsrp_nl_turn_count": totals["turn_count"],
        "fgsrp_nl_repair_count": totals["repair_count"],
        "fgsrp_nl_no_op_count": totals["no_op_count"],
        "m3_deferred_semantic_replay_certificates": certified,
        "m3_semantic_reference_sha256": (
            _sha256_file(semantic_reference)
            if semantic_reference is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--a1-run-root", type=Path, required=True)
    parser.add_argument("--treatment-run-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--global-seed", type=int, required=True)
    parser.add_argument("--frozen-a0-reference", type=Path)
    parser.add_argument("--m3-semantic-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
