#!/usr/bin/env python3
"""Fail-closed validation for a paired Ideal-TK run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from validate_closed_gt_takeover_run import (
    REPLAY_SOURCES,
    _jsonl,
    _manifest,
    _message_hashes,
)
from validate_paired_run import validate as validate_pair

TRACE_DIR = "ideal_tk_traces"
SCHEMA_VERSION = "capx-ideal-tk-v3.3-public-lossless"
TREATMENT_LABEL = "Ideal-TK-v3.3-public-lossless"


def _validate_trace(
    root: Path, *, action_message_hashes: Counter[str]
) -> tuple[int, dict[str, str]]:
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    ledger = state.get("evidence_ledger")
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("continuous_state_persisted") is not False
        or state.get("sim_step_observer_installed") is not True
        or not isinstance(state.get("wrapped_api_names"), list)
        or not isinstance(ledger, list)
        or len(ledger) > 2
    ):
        raise ValueError(f"Invalid Ideal-TK trial state: {root}")
    paths = sorted(root.glob("turn_*/takeover.json"))
    if state.get("turns_completed") != len(paths):
        raise ValueError(f"Ideal-TK trace is not turn-synchronous: {root}")

    semantic_hashes: dict[str, str] = {}
    for expected_turn, path in enumerate(paths, 1):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("schema_version") != SCHEMA_VERSION
            or record.get("turn") != expected_turn
            or record.get("carriers")
            != [
                "initial_visual_pt_suffix",
                "execution_context",
                "m3_visual_difference_report",
            ]
            or record.get("initial_visual_pt_suffix_absent") is not True
            or not isinstance(record.get("initial_visual_pt_suffix_removed"), bool)
            or record.get("latest_block_only_binding") is not True
            or record.get("non_text_structure_identical") is not True
            or len(record.get("full_prior_state", {})) != 51
            or len(record.get("full_current_state", {})) != 51
            or len(record.get("full_transition", {})) != 57
            or len(record.get("public_current_state", {})) != 51
            or record.get("public_current_state_field_count") != 51
            or record.get("public_current_state_roundtrip_equal") is not True
            or record.get("public_prior_state_roundtrip_equal") is not True
            or len(record.get("public_transition", {})) != 57
            or record.get("public_transition_field_count") != 57
            or record.get("public_transition_roundtrip_equal") is not True
            or record.get("public_full_closure_visible") is not True
            or len(record.get("public_interval_grasp_outcomes", {})) != 2
            or record.get("private_goal_iff_audit", {}).get("iff_holds") is not True
            or record.get("public_goal_iff_audit", {}).get("iff_holds") is not True
            or record.get("public_goal_iff_audit", {}).get("goal_definition_visible") is not False
            or record.get("public_goal_iff_audit", {}).get("private_poses_used") is not False
            or not isinstance(record.get("public_turn_semantics_sha256"), str)
            or not isinstance(record.get("report_word_count"), int)
            or not isinstance(record.get("report_attention_target"), int)
            or record.get("report_attention_target_exceeded")
            is not (
                record.get("report_word_count")
                > record.get("report_attention_target")
            )
        ):
            raise ValueError(f"Malformed or incomplete Ideal-TK record: {path}")
        if (
            record.get("public_current_state") != record.get("full_current_state")
            or record.get("public_prior_state") != record.get("full_prior_state")
            or record.get("public_transition") != record.get("full_transition")
        ):
            raise ValueError(f"Ideal-TK public lossless audit mismatch: {path}")
        for field in (
            "continuous_state_persisted",
            "raw_image_replaced",
            "executed_code_execution_changed",
            "console_stdout_content_changed",
            "console_stderr_content_changed",
            "task_or_api_text_replaced",
            "completion_or_reward_visible",
            "next_move_visible",
            "numeric_state_visible",
        ):
            if record.get(field) is not False:
                raise ValueError(f"Ideal-TK immutable-surface violation: {path}")
        if record.get("execution_scope_labels_changed") is not True:
            raise ValueError(f"Ideal-TK did not correct execution scope: {path}")
        for field in (
            "task_contract_sha256",
            "native_initial_context_sha256",
            "latest_executed_code_sha256",
            "latest_action_code_sha256",
        ):
            if not isinstance(record.get(field), str):
                raise ValueError(f"Ideal-TK PT-free envelope audit is incomplete: {path}")
        if (
            not isinstance(record.get("student_comment_count_removed"), int)
            or record.get("student_comment_count_removed") != 0
            or record.get("earlier_code_block_count_omitted") != expected_turn - 1
            or record.get("latest_executed_code_sha256")
            != record.get("latest_action_code_sha256")
            or not isinstance(
                record.get("executed_code_prompt_representation_changed"), bool
            )
        ):
            raise ValueError(f"Ideal-TK action evidence audit is malformed: {path}")
        fields = record.get("canonical_current_fields")
        if (
            not isinstance(fields, list)
            or len(fields) != 51
            or len(fields) != len(set(fields))
            or record.get("canonical_current_field_count") != len(fields)
        ):
            raise ValueError(f"Ideal-TK public closure is malformed: {path}")
        event_batches = record.get("public_temporal_event_batches")
        if not isinstance(event_batches, list):
            raise ValueError(f"Ideal-TK event trace is malformed: {path}")
        for batch in event_batches:
            changes = batch.get("changes") if isinstance(batch, dict) else None
            if not isinstance(changes, list) or not changes:
                raise ValueError(f"Ideal-TK event batch is malformed: {path}")
            for change in changes:
                if (
                    not isinstance(change, list)
                    or len(change) != 3
                    or not all(isinstance(value, str) for value in change)
                    or change[1] == change[2]
                ):
                    raise ValueError(f"Ideal-TK event change is malformed: {path}")
        calls = record.get("api_calls")
        if not isinstance(calls, list):
            raise ValueError(f"Ideal-TK API-call trace is malformed: {path}")
        for call in calls:
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("name"), str)
                or call.get("status") not in {"returned", "raised"}
                or not isinstance(call.get("target_ids"), list)
                or not isinstance(call.get("before_state_sha256"), str)
                or not isinstance(call.get("after_state_sha256"), str)
            ):
                raise ValueError(f"Ideal-TK API-call entry is malformed: {path}")
        prompt_hash = record.get("takeover_prompt_sha256")
        if not isinstance(prompt_hash, str) or action_message_hashes[prompt_hash] < 1:
            raise ValueError(f"Ideal-TK prompt was not sent to the action API: {path}")
        semantic_hashes[str(expected_turn)] = record["public_turn_semantics_sha256"]
    return len(paths), semantic_hashes


def validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _manifest(args.pair_root / "PAIR_MANIFEST.txt")
    if (
        manifest.get("treatment") != "ideal_tk"
        or manifest.get("treatment_label") != TREATMENT_LABEL
        or manifest.get("native_mode") != "M3"
        or manifest.get("initial_action_query_seed_offset") != "1"
    ):
        raise ValueError("Pair manifest did not select audited Ideal TK")

    semantic_reference = (
        json.loads(args.m3_semantic_reference.read_text(encoding="utf-8"))
        if args.m3_semantic_reference is not None
        else {}
    )
    base = validate_pair(
        SimpleNamespace(
            pair_root=args.pair_root,
            a1_run_root=args.a1_run_root,
            core_run_root=args.treatment_run_root,
            task=args.task,
            trials=args.trials,
            global_seed=args.global_seed,
            initial_query_seed_offset=1,
            config_suffix="vdm",
            frozen_a0_reference=args.frozen_a0_reference,
            frozen_a0_reference_only=bool(args.frozen_a0_reference),
            fixed_episode_seed=args.fixed_episode_seed,
            allow_deferred_semantic_replay=(args.m3_semantic_reference is not None),
        )
    )
    roots = sorted(args.treatment_run_root.glob(f"**/{TRACE_DIR}/trial_*"))
    if len(roots) != args.trials:
        raise ValueError(f"Expected {args.trials} Ideal-TK traces, found {len(roots)}")
    audit_path = args.treatment_run_root / "qwen_proxy_audit.jsonl"
    audit = _jsonl(audit_path)
    action_hashes = _message_hashes(audit_path)
    actual_hashes: dict[str, dict[str, str]] = {}
    turn_count = 0
    for root in roots:
        trial = str(int(root.name.split("_")[-1]))
        count, hashes = _validate_trace(root, action_message_hashes=action_hashes)
        turn_count += count
        actual_hashes[trial] = hashes

    expected_hashes = semantic_reference.get("closed_gt_turn_semantics_sha256", {})
    deferred = {
        int(record["effective_request_seed"]): (
            int(record["replay_trial"]),
            int(record["replay_turn"]),
        )
        for record in audit
        if record.get("normalized_replay_visual_match_deferred_to_semantics") is True
    }
    certificates: list[dict[str, Any]] = []
    for seed in base.get("deferred_semantic_replay_seeds", []):
        trial, query_turn = deferred[int(seed)]
        if query_turn <= 0 or query_turn % 2:
            raise ValueError(f"Invalid deferred M3 report position for seed {seed}")
        turn = query_turn // 2
        expected = expected_hashes.get(str(trial), {}).get(str(turn))
        actual = actual_hashes.get(str(trial), {}).get(str(turn))
        if not expected or actual != expected:
            raise ValueError(f"Deferred M3 replay failed GT equivalence for seed {seed}")
        certificates.append(
            {"seed": seed, "trial": trial, "turn": turn, "sha256": actual}
        )

    diverged: dict[int, int] = {}
    divergences: list[dict[str, int]] = []
    for record in audit:
        if not isinstance(record.get("replay_trial"), int):
            continue
        trial = int(record["replay_trial"])
        query_turn = int(record["replay_turn"])
        if record.get("m3_prefix_replay") is not True:
            raise ValueError("Ideal-TK did not enforce M3 prefix replay")
        if trial in diverged and record.get("source") in REPLAY_SOURCES:
            raise ValueError("M3 replay resumed after Ideal-TK divergence")
        if record.get("m3_prefix_diverged_on_request") is True:
            if trial in diverged:
                raise ValueError(f"M3 trial {trial} diverged more than once")
            diverged[trial] = query_turn
            divergences.append({"trial": trial, "query_turn": query_turn})
        if trial in diverged and query_turn > diverged[trial]:
            if record.get("m3_prefix_replay_blocked") is not True:
                raise ValueError("M3 post-divergence request was not replay-blocked")

    return {
        **base,
        "schema": "capx_ideal_tk_pair_validation_v3_3_public_lossless",
        "status": "valid",
        "treatment": TREATMENT_LABEL,
        "native_mode": "M3",
        "takeover_trial_trace_count": len(roots),
        "takeover_turn_count": turn_count,
        "m3_deferred_semantic_replay_certificates": certificates,
        "m3_prefix_replay_divergences": divergences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--a1-run-root", type=Path, required=True)
    parser.add_argument("--treatment-run-root", type=Path, required=True)
    parser.add_argument("--task", choices=("cube_stack", "cube_restack"), required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--global-seed", type=int, required=True)
    parser.add_argument("--fixed-episode-seed", type=int)
    parser.add_argument("--frozen-a0-reference", type=Path)
    parser.add_argument("--m3-semantic-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args)
    except Exception as exc:
        result = {
            "schema": "capx_ideal_tk_pair_validation_v3_3_public_lossless",
            "status": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
        }
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
