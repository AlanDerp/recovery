#!/usr/bin/env python3
"""Fail-closed validation for an FGSRP paired run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from export_fgsrp_passive_semantics import (
    semantic_evidence_sha256 as _canonical_semantic_evidence_sha256,
)
from validate_paired_run import validate as validate_pair

TRACE_DIR = "fgsrp_traces"
SCHEMA_VERSION = "capx-fgsrp-v2.0-ir-alignment"
TREATMENT_LABEL = "FGSRP-v2-semantic-ir-alignment"


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields[key] = value
    return fields


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trace_signatures(trace_roots: list[Path]) -> dict[str, list[dict[str, Any]]]:
    signatures: dict[str, list[dict[str, Any]]] = {}
    for root in trace_roots:
        trial = str(int(root.name.split("_")[-1]))
        signatures[trial] = []
        for path in sorted(root.glob("turn_*/replacement.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            signatures[trial].append(
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
    return signatures


def _validate_frozen_treatment(
    *,
    reference_path: Path,
    replay_path: Path,
    source_validation_path: Path,
    manifest: dict[str, str],
    treatment_run_root: Path,
    audit: list[dict[str, Any]],
    trace_roots: list[Path],
    task: str,
    trials: int,
    global_seed: int,
    base_validation: dict[str, Any],
) -> dict[str, Any]:
    from validate_paired_run import _initial_state_records, _summary_result, _trial_results

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    source_validation = json.loads(
        source_validation_path.read_text(encoding="utf-8")
    )
    if reference.get("schema") != "capx_fgsrp_treatment_reference_v1":
        raise ValueError("Unexpected frozen treatment reference schema")
    expected_identity = {
        "task": task,
        "carrier": manifest.get("fgsrp_carrier"),
        "native_mode": manifest.get("native_mode"),
        "global_seed": global_seed,
        "fixed_episode_seed": manifest.get("fixed_episode_seed", "none"),
        "total_trials": trials,
        "source_git_rev": manifest.get("git_rev"),
        "model": manifest.get("model"),
        "qwen_base_url": manifest.get("qwen_base_url"),
        "materialized_config_semantic_sha256": base_validation.get(
            "materialized_config_semantic_sha256"
        ),
        "frozen_a0_reference_sha256": manifest.get(
            "frozen_a0_reference_sha256", "not_applicable"
        ),
        "infra_bundle_sha256": base_validation.get("infra_bundle_sha256"),
        "python_packages_sha256": base_validation.get("python_packages_sha256"),
        "treatment_implementation_sha256": manifest.get(
            "treatment_implementation_sha256"
        ),
        "verifier_prompt_sha256": manifest.get("verifier_prompt_sha256"),
        "verifier_schema_sha256": manifest.get("verifier_schema_sha256"),
        "adjudicator_prompt_sha256": manifest.get("adjudicator_prompt_sha256"),
        "adjudicator_schema_sha256": manifest.get("adjudicator_schema_sha256"),
    }
    mismatches = {
        key: {"expected": value, "actual": reference.get(key)}
        for key, value in expected_identity.items()
        if reference.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen treatment identity mismatch: {mismatches}")
    if (
        source_validation.get("status") != "valid"
        or _sha256_file(source_validation_path)
        != reference.get("source_pair_validation_sha256")
    ):
        raise ValueError("Frozen treatment source validation is not registered")
    if _sha256_file(replay_path) != reference.get("full_replay_sha256"):
        raise ValueError("Frozen treatment replay hash mismatch")
    entries = replay.get("entries")
    if (
        replay.get("schema") != "capx_request_hash_replay_v1"
        or replay.get("missing_count") != 0
        or not isinstance(entries, dict)
        or len(entries) != reference.get("full_replay_entries")
    ):
        raise ValueError("Frozen treatment replay package is incomplete")
    replay_sources = {
        "request_hash_replay",
        "request_normalized_hash_replay",
        "request_legacy_runtime_normalized_hash_replay",
    }
    if any(
        record.get("ok") is not True or record.get("source") not in replay_sources
        for record in audit
    ):
        raise ValueError("Frozen treatment made a live or failed model call")
    expected_sequence = reference.get("request_response_sequence")
    actual_sequence = [
        {
            "request_sha256": record.get("replay_request_sha256"),
            "request_normalized_sha256": record.get(
                "replay_request_normalized_sha256"
            ),
            "response_content_sha256": record.get("response_content_sha256"),
            "effective_request_seed": record.get("effective_request_seed"),
        }
        for record in audit
    ]
    if actual_sequence != expected_sequence:
        raise ValueError("Frozen treatment request/response sequence drifted")
    summary = next(iter(treatment_run_root.glob("**/summaries.txt")), None)
    if summary is None or _summary_result(summary) != reference.get("expected_summary"):
        raise ValueError("Frozen treatment aggregate result drifted")
    actual_trials = _trial_results(treatment_run_root / f"{task}.log")
    if actual_trials != reference.get("trials"):
        raise ValueError("Frozen treatment per-trial result trajectory drifted")
    states = _initial_state_records(
        treatment_run_root / "repro_audit" / "initial_states.jsonl"
    )
    if [item.get("record_sha256") for item in states] != reference.get(
        "initial_state_record_sha256"
    ):
        raise ValueError("Frozen treatment initial-state trajectory drifted")
    if _trace_signatures(trace_roots) != reference.get("trace_signatures"):
        raise ValueError("Frozen treatment FGSRP trace trajectory drifted")
    return {
        "frozen_treatment": True,
        "frozen_treatment_reference": str(reference_path),
        "frozen_treatment_replay": str(replay_path),
        "frozen_treatment_source_validation": str(source_validation_path),
        "frozen_treatment_source_pair_job_id": reference["source_pair_job_id"],
        "frozen_treatment_model_calls": len(audit),
        "frozen_treatment_no_live_model_calls": True,
        "frozen_treatment_request_response_sequence_equal": True,
        "frozen_treatment_per_trial_results_equal": True,
        "frozen_treatment_trace_signatures_equal": True,
    }


def _message_hashes(path: Path) -> Counter[str]:
    hashes: Counter[str] = Counter()
    for record in _jsonl(path):
        request = record.get("request_canonical")
        if isinstance(request, str):
            request = json.loads(request)
        if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
            continue
        hashes[_sha256_json(request["messages"])] += 1
    return hashes


def _semantic_evidence_sha256(record: dict[str, Any]) -> str:
    return _canonical_semantic_evidence_sha256(record)


def _validate_trace(
    trial_root: Path,
    *,
    carrier: str,
    action_message_hashes: Counter[str],
) -> dict[str, int]:
    state_path = trial_root / "fgsrp_state.json"
    initialization_path = trial_root / "initialization.json"
    if not state_path.is_file() or not initialization_path.is_file():
        raise ValueError(f"Missing FGSRP trial state under {trial_root}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"FGSRP state schema mismatch: {state_path}")
    if initialization.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"FGSRP initialization schema mismatch: {initialization_path}")
    if initialization.get("action_visible_gt_packet_injected") is not False:
        raise ValueError("FGSRP initialization exposed a GT packet")
    if initialization.get("action_visible_bridge_injected") is not False:
        raise ValueError("FGSRP initialization exposed a bridge")

    turn_paths = sorted(trial_root.glob("turn_*/replacement.json"))
    snapshots = state.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != len(turn_paths) + 1:
        raise ValueError("FGSRP snapshot history is not turn-synchronous")
    expected_turns = list(range(1, len(turn_paths) + 1))
    if [item.get("turn") for item in snapshots] != [0, *expected_turns]:
        raise ValueError("FGSRP snapshot turn ordering is invalid")

    repair_count = 0
    no_op_count = 0
    for expected_turn, path in zip(expected_turns, turn_paths, strict=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"FGSRP replacement schema mismatch: {path}")
        if record.get("turn") != expected_turn or record.get("carrier") != carrier:
            raise ValueError(f"FGSRP carrier/turn mismatch: {path}")
        if record.get("non_text_structure_identical") is not True:
            raise ValueError(f"FGSRP changed request structure: {path}")
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
                raise ValueError(f"FGSRP immutable-surface violation {field}: {path}")
        corrected_hash = record.get("corrected_prompt_sha256")
        if not isinstance(corrected_hash, str) or action_message_hashes[corrected_hash] < 1:
            raise ValueError(f"FGSRP corrected request was not sent to the action API: {path}")

        repairs = record.get("repairs")
        patches = record.get("applied_patches")
        if not isinstance(repairs, list) or not isinstance(patches, list):
            raise ValueError(f"Malformed FGSRP repair audit: {path}")
        if len(repairs) != len(patches) or record.get("repair_count") != len(repairs):
            raise ValueError(f"FGSRP repair/patch count mismatch: {path}")
        protocol = record.get("verification_protocol")
        if not isinstance(protocol, dict):
            raise ValueError(f"Missing FGSRP verification protocol: {path}")
        verifier = protocol.get("verifier")
        claim_ledger = (
            verifier.get("claim_ledger") if isinstance(verifier, dict) else None
        )
        verifier_audit = (
            verifier.get("protocol") if isinstance(verifier, dict) else None
        )
        if not isinstance(claim_ledger, list) or not claim_ledger:
            raise ValueError(f"Missing FGSRP exhaustive claim ledger: {path}")
        if not isinstance(verifier_audit, dict):
            raise ValueError(f"Missing FGSRP claim-ledger audit: {path}")
        claim_count = 0
        contradicted_count = 0
        for sentence in claim_ledger:
            claims = sentence.get("claims") if isinstance(sentence, dict) else None
            if not isinstance(claims, list) or not claims:
                raise ValueError(f"Malformed FGSRP claim-ledger sentence: {path}")
            claim_count += len(claims)
            contradicted_count += sum(
                item.get("disposition") == "replace_contradicted"
                for item in claims
                if isinstance(item, dict)
            )
        if verifier_audit.get("sentence_count") != len(claim_ledger):
            raise ValueError(f"FGSRP claim-ledger sentence count mismatch: {path}")
        if verifier_audit.get("claim_count") != claim_count:
            raise ValueError(f"FGSRP claim-ledger claim count mismatch: {path}")
        proposals = verifier.get("proposals")
        if not isinstance(proposals, list) or len(proposals) != contradicted_count:
            raise ValueError(f"FGSRP claim-ledger repair coverage mismatch: {path}")
        known_false = protocol.get("known_false_claims")
        if not isinstance(known_false, list):
            raise ValueError(f"Missing FGSRP known-false audit: {path}")
        unresolved = [
            item
            for item in known_false
            if not isinstance(item, dict) or item.get("resolved") is not True
        ]
        if unresolved:
            raise ValueError(f"FGSRP left a known-false claim unresolved: {path}")
        if repairs:
            gate = protocol.get("adjudicator")
            decisions = gate.get("decisions") if isinstance(gate, dict) else None
            if not isinstance(decisions, list):
                raise ValueError(f"Missing FGSRP contradiction-gate decisions: {path}")
            accepted = [item for item in decisions if item.get("accepted") is True]
            host_decisions = protocol.get("deterministic_gate")
            if not isinstance(host_decisions, list):
                raise ValueError(f"Missing FGSRP deterministic-gate decisions: {path}")
            host_accepted = [
                item
                for item in host_decisions
                if item.get("accepted") is True
                and (
                    item.get("effective_accepted") is True
                    or item.get("llm_gate_accepted") is True
                )
            ]
            refinement = protocol.get("atomic_refinement")
            refined_decisions = []
            refined_certificates = []
            if isinstance(refinement, dict):
                refinement_gate = refinement.get("gate")
                if isinstance(refinement_gate, dict):
                    value = refinement_gate.get("decisions")
                    if isinstance(value, list):
                        refined_decisions = value
                value = refinement.get("host_certificates")
                if isinstance(value, list):
                    refined_certificates = value
            refined_accepted = [
                item for item in refined_decisions if item.get("accepted") is True
            ]
            refined_host_accepted = [
                refined_certificates[item["repair_index"]]
                for item in refined_accepted
                if (
                    isinstance(item.get("repair_index"), int)
                    and 0 <= item["repair_index"] < len(refined_certificates)
                    and refined_certificates[item["repair_index"]].get("accepted")
                    is True
                )
            ]
            if (
                len(accepted) != len(host_accepted)
                or len(refined_accepted) != len(refined_host_accepted)
                or len(accepted) + len(refined_accepted) != len(repairs)
            ):
                raise ValueError(f"FGSRP accepted-decision count mismatch: {path}")
        repair_count += len(repairs)
        if record.get("active_repair_count") == 0:
            if (
                record.get("native_prompt_sha256") != corrected_hash
                or record.get("no_op_prompt_byte_equivalent") is not True
            ):
                raise ValueError(f"FGSRP no-op was not byte-equivalent: {path}")
            no_op_count += 1
    return {
        "turn_count": len(turn_paths),
        "repair_count": repair_count,
        "no_op_count": no_op_count,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.pair_root / "PAIR_MANIFEST.txt"
    if not manifest_path.is_file():
        raise ValueError(f"Missing pair manifest: {manifest_path}")
    manifest = _manifest(manifest_path)
    if (
        manifest.get("treatment") != "fgsrp"
        or manifest.get("treatment_label") != TREATMENT_LABEL
    ):
        raise ValueError("Pair manifest did not select semantic-IR FGSRP-v2")
    if manifest.get("control_runtime_kind") != "patched_clean_runtime":
        raise ValueError("FGSRP control runtime kind is not explicit")
    if manifest.get("control_runtime_git_rev") != manifest.get("git_rev"):
        raise ValueError("FGSRP control runtime revision does not match run revision")
    carrier = manifest.get("fgsrp_carrier")
    if carrier not in {"m2_comments", "m3_visual_difference"}:
        raise ValueError(f"Invalid FGSRP carrier in manifest: {carrier!r}")
    expected_mode = "M2" if carrier == "m2_comments" else "M3"
    if manifest.get("native_mode") != expected_mode:
        raise ValueError("FGSRP native mode does not match its carrier")
    initial_query_seed_offset = 0 if expected_mode == "M2" else 1
    config_suffix = "vf" if expected_mode == "M2" else "vdm"
    if manifest.get("initial_action_query_seed_offset") != str(
        initial_query_seed_offset
    ):
        raise ValueError("FGSRP manifest has the wrong initial action seed offset")
    fixed_episode_seed_text = manifest.get("fixed_episode_seed", "none")
    fixed_episode_seed = (
        None
        if fixed_episode_seed_text == "none"
        else int(fixed_episode_seed_text)
    )

    frozen_reference = args.frozen_a0_reference
    frozen_reference_value = (
        json.loads(frozen_reference.read_text(encoding="utf-8"))
        if frozen_reference is not None
        else {}
    )
    semantic_reference = args.m3_semantic_reference or frozen_reference
    semantic_reference_value = (
        json.loads(semantic_reference.read_text(encoding="utf-8"))
        if semantic_reference is not None
        else {}
    )
    has_semantic_reference = isinstance(
        semantic_reference_value.get("m3_semantic_transition_sha256"), dict
    )
    if args.m3_semantic_reference is not None:
        expected_semantic_identity = {
            "schema": "capx_fgsrp_passive_semantic_reference_v1",
            "task": args.task,
            "native_mode": "M3",
            "global_seed": args.global_seed,
            "total_trials": args.trials,
            "capture_mode": "passive_action_invisible",
            "student_prompt_changed": False,
            "student_action_request_changed": False,
        }
        observed_semantic_identity = {
            key: semantic_reference_value.get(key)
            for key in expected_semantic_identity
        }
        if observed_semantic_identity != expected_semantic_identity:
            raise ValueError("M3 passive semantic reference identity mismatch")
        if Path(semantic_reference_value.get("source_run_root", "")).resolve() != (
            args.a1_run_root.resolve()
        ):
            raise ValueError("M3 passive semantic reference is not bound to this A1 run")
    base = validate_pair(
        SimpleNamespace(
            pair_root=args.pair_root,
            a1_run_root=args.a1_run_root,
            core_run_root=args.treatment_run_root,
            task=args.task,
            trials=args.trials,
            global_seed=args.global_seed,
            initial_query_seed_offset=initial_query_seed_offset,
            config_suffix=config_suffix,
            frozen_a0_reference=frozen_reference,
            frozen_a0_reference_only=bool(frozen_reference),
            fixed_episode_seed=fixed_episode_seed,
            allow_deferred_semantic_replay=(
                expected_mode == "M3" and has_semantic_reference
            ),
        )
    )
    trace_roots = sorted(args.treatment_run_root.glob(f"**/{TRACE_DIR}/trial_*"))
    if len(trace_roots) != args.trials:
        raise ValueError(
            f"Expected {args.trials} FGSRP traces, found {len(trace_roots)}"
        )
    request_hashes = _message_hashes(args.treatment_run_root / "qwen_proxy_audit.jsonl")
    treatment_qwen_records = _jsonl(
        args.treatment_run_root / "qwen_proxy_audit.jsonl"
    )
    totals = Counter()
    semantic_hashes: dict[str, dict[str, str]] = {}
    for root in trace_roots:
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
                _semantic_evidence_sha256(record)
            )

    deferred_seeds = base.get("deferred_semantic_replay_seeds") or []
    if deferred_seeds:
        if semantic_reference is None:
            raise ValueError("Deferred semantic replay lacks a semantic reference")
        reference = semantic_reference_value
        expected_hashes = reference.get("m3_semantic_transition_sha256")
        if not isinstance(expected_hashes, dict):
            raise ValueError("M3 reference lacks semantic transition hashes")
        certified: list[dict[str, Any]] = []
        deferred_positions = {
            int(record["effective_request_seed"]): (
                int(record["replay_trial"]),
                int(record["replay_turn"]),
            )
            for record in treatment_qwen_records
            if record.get("normalized_replay_visual_match_deferred_to_semantics")
            is True
        }
        for seed in deferred_seeds:
            position = deferred_positions.get(int(seed))
            if position is None:
                raise ValueError(f"Deferred M3 replay lacks an audited position: {seed}")
            trial, query_turn = position
            if query_turn <= 0 or query_turn % 2:
                raise ValueError(f"Invalid deferred M3 report seed: {seed}")
            turn = query_turn // 2
            expected = expected_hashes.get(str(trial), {}).get(str(turn))
            actual = semantic_hashes.get(str(trial), {}).get(str(turn))
            if not expected or actual != expected:
                raise ValueError(
                    "Deferred M3 report failed GT semantic equivalence: "
                    f"seed={seed} expected={expected!r} actual={actual!r}"
                )
            certified.append(
                {"seed": seed, "trial": trial, "turn": turn, "sha256": actual}
            )
    else:
        certified = []

    frozen_treatment: dict[str, Any] = {}
    if args.frozen_treatment_reference is not None:
        if (
            args.frozen_treatment_replay is None
            or args.frozen_treatment_source_validation is None
        ):
            raise ValueError("Frozen treatment reference lacks replay/source validation")
        if manifest.get("treatment_replay_mode") != "registered_exact":
            raise ValueError("Frozen treatment did not declare registered_exact mode")
        frozen_treatment = _validate_frozen_treatment(
            reference_path=args.frozen_treatment_reference,
            replay_path=args.frozen_treatment_replay,
            source_validation_path=args.frozen_treatment_source_validation,
            manifest=manifest,
            treatment_run_root=args.treatment_run_root,
            audit=treatment_qwen_records,
            trace_roots=trace_roots,
            task=args.task,
            trials=args.trials,
            global_seed=args.global_seed,
            base_validation=base,
        )
    elif manifest.get("treatment_replay_mode", "registration_live") != "registration_live":
        raise ValueError("Unregistered treatment replay mode")

    prefix_divergences: list[dict[str, int]] = []
    if (
        expected_mode == "M3"
        and semantic_reference is not None
        and not frozen_treatment
    ):
        native_records = [
            record
            for record in treatment_qwen_records
            if isinstance(record.get("replay_trial"), int)
            and isinstance(record.get("replay_turn"), int)
        ]
        diverged_trials: set[int] = set()
        divergence_turn_by_trial: dict[int, int] = {}
        replay_sources = {
            "request_hash_replay",
            "request_normalized_hash_replay",
            "request_legacy_runtime_normalized_hash_replay",
        }
        if not native_records or any(
            record.get("m3_prefix_replay") is not True
            for record in native_records
        ):
            raise ValueError("Frozen M3 run did not enforce prefix replay")
        for record in native_records:
            trial = int(record["replay_trial"])
            turn = int(record["replay_turn"])
            if trial in diverged_trials and record.get("source") in replay_sources:
                raise ValueError(
                    "M3 replay resumed after the policy trajectory diverged: "
                    f"trial={trial} query_turn={turn}"
                )
            if record.get("m3_prefix_diverged_on_request") is True:
                if trial in diverged_trials:
                    raise ValueError(f"M3 trial {trial} diverged more than once")
                diverged_trials.add(trial)
                divergence_turn_by_trial[trial] = turn
                prefix_divergences.append(
                    {"trial": trial, "query_turn": turn}
                )
            if (
                trial in divergence_turn_by_trial
                and turn > divergence_turn_by_trial[trial]
            ):
                if record.get("m3_prefix_replay_blocked") is not True:
                    raise ValueError(
                        "M3 post-divergence native request was not replay-blocked: "
                        f"trial={trial} query_turn={turn}"
                    )
    return {
        **base,
        "schema": "capx_fgsrp_pair_validation_v1",
        "status": "valid",
        "treatment": TREATMENT_LABEL,
        "carrier": carrier,
        "native_mode": expected_mode,
        "initial_action_query_seed_offset": initial_query_seed_offset,
        "fgsrp_trial_trace_count": len(trace_roots),
        "fgsrp_turn_count": totals["turn_count"],
        "fgsrp_repair_count": totals["repair_count"],
        "fgsrp_no_op_count": totals["no_op_count"],
        "m3_deferred_semantic_replay_certificates": certified,
        "m3_semantic_reference_sha256": (
            _sha256_file(semantic_reference)
            if semantic_reference is not None
            else None
        ),
        "m3_prefix_replay_divergences": prefix_divergences,
        **frozen_treatment,
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
    parser.add_argument("--frozen-treatment-reference", type=Path)
    parser.add_argument("--frozen-treatment-replay", type=Path)
    parser.add_argument("--frozen-treatment-source-validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
