#!/usr/bin/env python3
"""Fail-closed validation for a paired closed-GT whole-report takeover run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from validate_paired_run import validate as validate_pair

TRACE_DIR = "closed_gt_takeover_traces"
SCHEMA_VERSION = "capx-closed-gt-takeover-v2.1"
TREATMENT_LABEL = "Closed-GT-Takeover-v2.1"
REPLAY_SOURCES = {
    "request_hash_replay",
    "request_normalized_hash_replay",
    "request_legacy_runtime_normalized_hash_replay",
}
INTERVAL_GRASP_KEYS = {
    "executor_object.exe01.obj01.grasp_contact_state",
    "executor_object.exe01.obj02.grasp_contact_state",
}
INTERVAL_GRASP_VALUES = {
    "NO_EXECUTOR_CONTACT_OBSERVED",
    "CONTACT_WITHOUT_BILATERAL_GRASP_OBSERVED",
    "BILATERAL_GRASP_WITHOUT_LIFT_THEN_LOST",
    "BILATERAL_GRASP_WITHOUT_LIFT_RETAINED",
    "LIFT_CONFIRMED_THEN_LOST",
    "LIFT_CONFIRMED_AND_RETAINED",
}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            result[key] = value
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _message_hashes(path: Path) -> Counter[str]:
    hashes: Counter[str] = Counter()
    for record in _jsonl(path):
        request = record.get("request_canonical")
        if isinstance(request, str):
            request = json.loads(request)
        if isinstance(request, dict) and isinstance(request.get("messages"), list):
            hashes[_sha256_json(request["messages"])] += 1
    return hashes


def _validate_trace(
    root: Path, *, action_message_hashes: Counter[str]
) -> tuple[int, dict[str, str]]:
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("continuous_state_persisted") is not False
        or state.get("action_visible_initial_gt") is not False
        or state.get("sim_step_observer_installed") is not True
    ):
        raise ValueError(f"Invalid takeover trial state: {root}")
    paths = sorted(root.glob("turn_*/takeover.json"))
    if state.get("turns_completed") != len(paths):
        raise ValueError(f"Takeover trace is not turn-synchronous: {root}")
    transition_hashes: dict[str, str] = {}
    for expected_turn, path in enumerate(paths, 1):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("schema_version") != SCHEMA_VERSION
            or record.get("turn") != expected_turn
            or record.get("carrier")
            != "m3_visual_difference_report_whole_takeover"
            or record.get("non_text_structure_identical") is not True
            or record.get("current_state_field_count") != 44
            or record.get("changed_state_field_count")
            + record.get("unchanged_state_field_count")
            != 44
            or len(record.get("public_current_state", {})) != 51
            or len(record.get("public_transition", {})) != 57
            or len(record.get("pose_transition_fields", [])) != 3
            or record.get("interval_grasp_outcome_count") != 2
            or len(record.get("public_interval_grasp_outcomes", {})) != 2
            or record.get("temporal_event_field_count")
            != len(record.get("temporal_event_fields", []))
            or not isinstance(record.get("public_turn_semantics_sha256"), str)
            or record.get("private_goal_iff_audit", {}).get("iff_holds") is not True
        ):
            raise ValueError(f"Malformed or incomplete takeover record: {path}")
        interval_outcomes = record["public_interval_grasp_outcomes"]
        if (
            set(interval_outcomes) != INTERVAL_GRASP_KEYS
            or not set(interval_outcomes.values()).issubset(INTERVAL_GRASP_VALUES)
        ):
            raise ValueError(f"Takeover interval grasp partition is invalid: {path}")
        immutable_false = (
            "continuous_state_persisted",
            "raw_image_replaced",
            "executed_code_replaced",
            "console_stdout_replaced",
            "console_stderr_replaced",
            "task_or_api_text_replaced",
            "completion_or_reward_visible",
            "next_move_visible",
            "numeric_state_visible",
        )
        if any(record.get(field) is not False for field in immutable_false):
            raise ValueError(f"Takeover immutable-surface violation: {path}")
        current = record["current_state_fields"]
        changed = record["changed_state_fields"]
        unchanged = record["unchanged_state_fields"]
        if (
            len(current) != len(set(current))
            or set(changed).intersection(unchanged)
            or set(changed).union(unchanged) != set(current)
        ):
            raise ValueError(f"Takeover closure partition is invalid: {path}")
        event_batches = record.get("public_temporal_event_batches")
        if not isinstance(event_batches, list):
            raise ValueError(f"Takeover temporal events are malformed: {path}")
        event_fields: list[str] = []
        for batch in event_batches:
            changes = batch.get("changes") if isinstance(batch, dict) else None
            if not isinstance(changes, list) or not changes:
                raise ValueError(f"Takeover temporal event batch is malformed: {path}")
            for change in changes:
                if (
                    not isinstance(change, list)
                    or len(change) != 3
                    or not all(isinstance(value, str) for value in change)
                    or change[1] == change[2]
                ):
                    raise ValueError(f"Takeover temporal event change is malformed: {path}")
                event_fields.append(change[0])
        if event_fields != record.get("temporal_event_fields"):
            raise ValueError(f"Takeover temporal event field order mismatch: {path}")
        turn_semantics = {
            "public_transition": record["public_transition"],
            "public_interval_grasp_outcomes": record[
                "public_interval_grasp_outcomes"
            ],
            "public_temporal_event_batches": event_batches,
        }
        if record["public_turn_semantics_sha256"] != _sha256_json(turn_semantics):
            raise ValueError(f"Takeover temporal semantics hash mismatch: {path}")
        prompt_hash = record.get("takeover_prompt_sha256")
        if not isinstance(prompt_hash, str) or action_message_hashes[prompt_hash] < 1:
            raise ValueError(f"Takeover prompt was not sent to the action API: {path}")
        transition_hashes[str(expected_turn)] = record[
            "public_turn_semantics_sha256"
        ]
    return len(paths), transition_hashes


def validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _manifest(args.pair_root / "PAIR_MANIFEST.txt")
    if (
        manifest.get("treatment") != "closed_gt_takeover"
        or manifest.get("treatment_label") != TREATMENT_LABEL
        or manifest.get("native_mode") != "M3"
        or manifest.get("initial_action_query_seed_offset") != "1"
    ):
        raise ValueError("Pair manifest did not select audited closed-GT takeover")
    semantic_reference = (
        json.loads(args.m3_semantic_reference.read_text(encoding="utf-8"))
        if args.m3_semantic_reference is not None
        else {}
    )
    if args.m3_semantic_reference is not None:
        expected = {
            "schema": "capx_closed_gt_takeover_passive_reference_v2.1",
            "task": args.task,
            "native_mode": "M3",
            "global_seed": args.global_seed,
            "total_trials": args.trials,
            "capture_mode": "passive_action_invisible",
            "student_prompt_changed": False,
            "student_action_request_changed": False,
        }
        if {key: semantic_reference.get(key) for key in expected} != expected:
            raise ValueError("Closed-GT passive reference identity mismatch")
        if Path(semantic_reference.get("source_run_root", "")).resolve() != (
            args.a1_run_root.resolve()
        ):
            raise ValueError("Closed-GT passive reference is not bound to this A1 run")

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
        raise ValueError(f"Expected {args.trials} takeover traces, found {len(roots)}")
    audit_path = args.treatment_run_root / "qwen_proxy_audit.jsonl"
    action_hashes = _message_hashes(audit_path)
    audit = _jsonl(audit_path)
    semantic_hashes: dict[str, dict[str, str]] = {}
    turn_count = 0
    for root in roots:
        trial = str(int(root.name.split("_")[-1]))
        count, hashes = _validate_trace(root, action_message_hashes=action_hashes)
        turn_count += count
        semantic_hashes[trial] = hashes

    expected_hashes = semantic_reference.get("closed_gt_turn_semantics_sha256", {})
    certified: list[dict[str, Any]] = []
    deferred = {
        int(record["effective_request_seed"]): (
            int(record["replay_trial"]),
            int(record["replay_turn"]),
        )
        for record in audit
        if record.get("normalized_replay_visual_match_deferred_to_semantics") is True
    }
    for seed in base.get("deferred_semantic_replay_seeds", []):
        trial, query_turn = deferred[int(seed)]
        if query_turn <= 0 or query_turn % 2:
            raise ValueError(f"Invalid deferred M3 report position for seed {seed}")
        turn = query_turn // 2
        expected = expected_hashes.get(str(trial), {}).get(str(turn))
        actual = semantic_hashes.get(str(trial), {}).get(str(turn))
        if not expected or actual != expected:
            raise ValueError(
                f"Deferred M3 replay failed closed-GT equivalence for seed {seed}"
            )
        certified.append(
            {"seed": seed, "trial": trial, "turn": turn, "sha256": actual}
        )

    diverged_trials: set[int] = set()
    divergence_turn_by_trial: dict[int, int] = {}
    prefix_divergences: list[dict[str, int]] = []
    for record in audit:
        if not isinstance(record.get("replay_trial"), int):
            continue
        trial = int(record["replay_trial"])
        turn = int(record["replay_turn"])
        if record.get("m3_prefix_replay") is not True:
            raise ValueError("M3 takeover did not enforce prefix replay")
        if trial in diverged_trials and record.get("source") in REPLAY_SOURCES:
            raise ValueError("M3 replay resumed after takeover trajectory divergence")
        if record.get("m3_prefix_diverged_on_request") is True:
            if trial in diverged_trials:
                raise ValueError(f"M3 trial {trial} diverged more than once")
            diverged_trials.add(trial)
            divergence_turn_by_trial[trial] = turn
            prefix_divergences.append({"trial": trial, "query_turn": turn})
        if trial in diverged_trials and turn > divergence_turn_by_trial[trial]:
            if record.get("m3_prefix_replay_blocked") is not True:
                raise ValueError("M3 post-divergence request was not replay-blocked")

    return {
        **base,
        "schema": "capx_closed_gt_takeover_pair_validation_v2.1",
        "status": "valid",
        "treatment": TREATMENT_LABEL,
        "native_mode": "M3",
        "takeover_trial_trace_count": len(roots),
        "takeover_turn_count": turn_count,
        "m3_deferred_semantic_replay_certificates": certified,
        "m3_prefix_replay_divergences": prefix_divergences,
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
        args.output.write_text(
            json.dumps(
                {
                    "schema": "capx_closed_gt_takeover_pair_validation_v2.1",
                    "status": "invalid",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
