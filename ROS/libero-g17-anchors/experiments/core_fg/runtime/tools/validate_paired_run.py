#!/usr/bin/env python3
"""Fail-closed validation for a completed A1/Core-FG paired run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPLAY_SOURCES = {
    "request_hash_replay",
    "request_normalized_hash_replay",
    "request_legacy_runtime_normalized_hash_replay",
}
NORMALIZED_REPLAY_SOURCES = REPLAY_SOURCES - {"request_hash_replay"}
SUMMARY_RESULT = re.compile(
    r"Code generation success rate / Average reward / Task completed:\s*"
    r"(\d+\.\d{3}/-?\d+\.\d{3}/\d+)"
)
TRIAL_RESULT = re.compile(
    r"Code saved to .*/trial_(\d+)_sandboxrc_(\d+)_"
    r"reward_([-\d.]+)_taskcompleted_(\d+)/code\.py"
)
CHAIN_BLOCKERS = {
    "assertion_error": re.compile(r"AssertionError"),
    "mask_depth_mismatch": re.compile(
        r"(?:mask-depth index mismatch|"
        r"IndexError:[^\n]*out of bounds for axis 0[^\n]*size 0)",
        re.IGNORECASE,
    ),
    "invalid_depth_points": re.compile(
        r"(?:no valid depth points|insufficient valid depth points)", re.IGNORECASE
    ),
    "empty_grasp_candidates": re.compile(
        r"ValueError:\s*attempt to get argmax of an empty sequence",
        re.IGNORECASE,
    ),
    "open3d_obb": re.compile(r"(?:Open3D.*OBB|OBB.*Open3D)", re.IGNORECASE),
    "terminated_episode_action": re.compile(
        r"executing action in terminated episode", re.IGNORECASE
    ),
    "replay_miss": re.compile(
        r"(?:required request-hash replay miss|replay miss)", re.IGNORECASE
    ),
    "trial_child_process_failed": re.compile(
        r"Trial\s+\d+\s+child process failed", re.IGNORECASE
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_config_sha256(path: Path) -> str:
    """Hash task semantics while excluding per-job localhost port bindings."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Materialized task config is not a mapping: {path}")
    api_servers = value.get("api_servers", [])
    if not isinstance(api_servers, list):
        raise ValueError(f"Materialized task config api_servers is not a list: {path}")
    for index, server in enumerate(api_servers):
        if not isinstance(server, dict):
            raise ValueError(
                f"Materialized task config api_servers[{index}] is not a mapping: {path}"
            )
        server.pop("port", None)
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _require_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {pattern!r} under {root}, found {len(matches)}"
        )
    return matches[0]


def _initial_state_records(path: Path) -> list[dict[str, Any]]:
    return [record for record in _jsonl(path) if "binding" in record]


def _validate_episode_seed_mapping(
    records: list[dict[str, Any]],
    *,
    trials: int,
    fixed_episode_seed: int | None,
    episode_seed_base: int | None,
) -> list[int]:
    if fixed_episode_seed is not None and episode_seed_base is not None:
        raise ValueError("fixed_episode_seed and episode_seed_base are mutually exclusive")
    if fixed_episode_seed is not None and trials != 1:
        raise ValueError("A fixed episode seed is valid only for one-trial smoke")

    expected_effective = [
        (
            int(fixed_episode_seed)
            if fixed_episode_seed is not None
            else int(episode_seed_base) + trial
            if episode_seed_base is not None
            else trial
        )
        for trial in range(1, trials + 1)
    ]
    actual_effective: list[int] = []
    for trial, (record, expected) in enumerate(
        zip(records, expected_effective, strict=True), 1
    ):
        requested = record.get("requested_seed")
        effective = record.get("effective_seed")
        recorded_base = record.get("episode_seed_base")
        if requested != trial:
            raise ValueError(
                f"Initial-state trial {trial} requested_seed mismatch: {requested!r}"
            )
        if effective != expected or record.get("seed") != expected:
            raise ValueError(
                f"Initial-state trial {trial} effective seed mismatch: "
                f"expected={expected} effective_seed={effective!r} "
                f"seed={record.get('seed')!r}"
            )
        if recorded_base != episode_seed_base:
            raise ValueError(
                f"Initial-state trial {trial} episode_seed_base mismatch: "
                f"expected={episode_seed_base!r} actual={recorded_base!r}"
            )
        binding = record.get("binding")
        if not isinstance(binding, dict) or binding.get("seed") != expected:
            raise ValueError(
                f"Initial-state trial {trial} RNG binding seed mismatch"
            )
        actual_effective.append(int(effective))
    return actual_effective


def _audit_by_seed(
    records: list[dict[str, Any]],
    required_seeds: set[int],
    *,
    replay_only: bool,
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for record in records:
        seed = record.get("effective_request_seed")
        if not isinstance(seed, int) or seed not in required_seeds:
            continue
        if replay_only and record.get("source") not in REPLAY_SOURCES:
            continue
        if seed in selected:
            raise ValueError(f"Duplicate required initial-query audit seed: {seed}")
        if record.get("ok") is not True:
            raise ValueError(f"Required initial-query audit seed {seed} is not OK")
        selected[seed] = record
    missing = sorted(required_seeds - selected.keys())
    if missing:
        raise ValueError(f"Missing required initial-query audit seeds: {missing}")
    return selected


def _summary_result(path: Path) -> str:
    match = SUMMARY_RESULT.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"Could not parse summary result: {path}")
    return match.group(1)


def _trial_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for match in TRIAL_RESULT.finditer(
        path.read_text(encoding="utf-8", errors="replace")
    ):
        trial, sandbox_rc, reward, task_completed = match.groups()
        trial = str(int(trial))
        if trial in results:
            raise ValueError(f"Duplicate final trial result {trial} in {path}")
        results[trial] = {
            "sandbox_rc": int(sandbox_rc),
            "reward": float(reward),
            "task_completed": bool(int(task_completed)),
        }
    return results


def validate(args: argparse.Namespace) -> dict[str, Any]:
    pair_root = args.pair_root.resolve()
    a1_root = args.a1_run_root.resolve()
    core_root = args.core_run_root.resolve()
    initial_query_seed_offset = int(
        getattr(args, "initial_query_seed_offset", 0)
    )
    if initial_query_seed_offset < 0 or initial_query_seed_offset >= 1000:
        raise ValueError("initial_query_seed_offset must be in [0, 1000)")
    fixed_episode_seed = getattr(args, "fixed_episode_seed", None)
    episode_seed_base = getattr(args, "episode_seed_base", None)
    if episode_seed_base is not None:
        episode_seed_base = int(episode_seed_base)
    if fixed_episode_seed is not None and episode_seed_base is not None:
        raise ValueError("fixed_episode_seed and episode_seed_base are mutually exclusive")
    if fixed_episode_seed is not None:
        if args.trials != 1:
            raise ValueError("A fixed episode seed is valid only for one-trial smoke")
        required_seeds = {
            int(fixed_episode_seed) * 1000 + initial_query_seed_offset
        }
    else:
        required_seeds = {
            args.global_seed + trial * 1000 + initial_query_seed_offset
            for trial in range(1, args.trials + 1)
        }

    required_files = [
        pair_root / "INFRA_FINGERPRINT.json",
        a1_root / "qwen_proxy_audit.jsonl",
        core_root / "qwen_proxy_audit.jsonl",
        a1_root / "repro_audit" / "initial_states.jsonl",
        core_root / "repro_audit" / "initial_states.jsonl",
        pair_root / "a1_initial_request_hash_replay.json",
    ]
    for path in required_files:
        if not path.is_file():
            raise ValueError(f"Missing required paired-run artifact: {path}")

    a1_summary = _require_one(a1_root, "**/summaries.txt")
    core_summary = _require_one(core_root, "**/summaries.txt")

    infra_fingerprint = json.loads(required_files[0].read_text(encoding="utf-8"))
    if (
        infra_fingerprint.get("schema") != "capx_cube_infra_fingerprint_v1"
        or infra_fingerprint.get("lock_matched") is not True
        or infra_fingerprint.get("lock_mismatches") != {}
    ):
        raise ValueError("Cube experiment infrastructure fingerprint is not locked")

    replay_package = json.loads(required_files[5].read_text(encoding="utf-8"))
    if replay_package.get("schema") != "capx_request_hash_replay_v1":
        raise ValueError("Unexpected replay package schema")
    if replay_package.get("missing_count") != 0:
        raise ValueError(
            f"Replay export has missing_count={replay_package.get('missing_count')!r}"
        )

    a1_audit_records = _jsonl(required_files[1])
    core_audit_records = _jsonl(required_files[2])
    for arm, records in (
        ("A1", a1_audit_records),
        ("treatment", core_audit_records),
    ):
        failed_calls = [
            {
                "status_code": record.get("status_code"),
                "source": record.get("source"),
                "effective_request_seed": record.get("effective_request_seed"),
                "error": record.get("error"),
            }
            for record in records
            if record.get("ok") is not True
        ]
        if failed_calls:
            raise ValueError(f"{arm} Qwen audit contains failed calls: {failed_calls}")
    deferred_semantic_replays = [
        record
        for record in core_audit_records
        if record.get("normalized_replay_visual_match_deferred_to_semantics")
        is True
    ]
    allow_deferred_semantic_replay = bool(
        getattr(args, "allow_deferred_semantic_replay", False)
    )
    unverified_normalized_replays = [
        {
            "source": record.get("source"),
            "effective_request_seed": record.get("effective_request_seed"),
        }
        for record in core_audit_records
        if (
            record.get("source") in NORMALIZED_REPLAY_SOURCES
            and record.get("normalized_replay_visual_match_verified") is not True
            and not (
                allow_deferred_semantic_replay
                and record.get(
                    "normalized_replay_visual_match_deferred_to_semantics"
                )
                is True
            )
        )
    ]
    if unverified_normalized_replays:
        raise ValueError(
            "Treatment used normalized replay without verified source images: "
            f"{unverified_normalized_replays}"
        )
    a1_audit = _audit_by_seed(
        a1_audit_records, required_seeds, replay_only=False
    )
    core_audit = _audit_by_seed(
        core_audit_records, required_seeds, replay_only=True
    )
    for seed in sorted(required_seeds):
        a1_response = a1_audit[seed].get("response_content_sha256")
        core_response = core_audit[seed].get("response_content_sha256")
        if not a1_response or a1_response != core_response:
            raise ValueError(
                f"Initial response hash mismatch for seed {seed}: "
                f"A1={a1_response!r} Core-FG={core_response!r}"
            )

    a1_states = _initial_state_records(required_files[3])
    core_states = _initial_state_records(required_files[4])
    if len(a1_states) != args.trials or len(core_states) != args.trials:
        raise ValueError(
            "Initial-state record count mismatch: "
            f"A1={len(a1_states)} Core-FG={len(core_states)} trials={args.trials}"
        )
    a1_effective_episode_seeds = _validate_episode_seed_mapping(
        a1_states,
        trials=args.trials,
        fixed_episode_seed=fixed_episode_seed,
        episode_seed_base=episode_seed_base,
    )
    core_effective_episode_seeds = _validate_episode_seed_mapping(
        core_states,
        trials=args.trials,
        fixed_episode_seed=fixed_episode_seed,
        episode_seed_base=episode_seed_base,
    )
    if a1_effective_episode_seeds != core_effective_episode_seeds:
        raise ValueError("A1/Core-FG effective episode seeds differ")
    a1_state_hashes = [record.get("record_sha256") for record in a1_states]
    core_state_hashes = [record.get("record_sha256") for record in core_states]
    if a1_state_hashes != core_state_hashes:
        raise ValueError("A1/Core-FG initial-state hashes differ")

    frozen_a0: dict[str, Any] | None = None
    frozen_a0_reference = getattr(args, "frozen_a0_reference", None)
    frozen_a0_reference_only = bool(
        getattr(args, "frozen_a0_reference_only", False)
    )
    if frozen_a0_reference is not None:
        reference_path = Path(frozen_a0_reference)
        frozen_a0 = json.loads(reference_path.read_text(encoding="utf-8"))
        if (
            frozen_a0.get("schema") != "capx_frozen_a0_reference_v1"
            or frozen_a0.get("task") != args.task
            or frozen_a0.get("global_seed") != args.global_seed
            or frozen_a0.get("total_trials") != args.trials
        ):
            raise ValueError("Frozen A0 reference does not match this experiment")
        if _sha256(required_files[5]) != frozen_a0.get("full_replay_sha256"):
            raise ValueError("Frozen A0 replay package hash mismatch")
        replay_entries = replay_package.get("entries")
        if (
            not isinstance(replay_entries, dict)
            or len(replay_entries) != frozen_a0.get("full_replay_entries")
        ):
            raise ValueError("Frozen A0 replay entry count mismatch")
        if a1_state_hashes != frozen_a0.get("initial_state_record_sha256"):
            raise ValueError("Frozen A0 initial-state trajectory mismatch")
        if frozen_a0_reference_only:
            if any(record.get("ok") is not True for record in a1_audit_records):
                raise ValueError("Frozen A0 source audit contains a failed model call")
        else:
            if any(
                record.get("ok") is not True
                or record.get("source") not in REPLAY_SOURCES
                for record in a1_audit_records
            ):
                raise ValueError("Frozen A0 audit contains a live or failed model call")
            consumed_entries = [
                record.get("replay_request_sha256") for record in a1_audit_records
            ]
            if Counter(consumed_entries) != Counter(replay_entries.keys()):
                raise ValueError(
                    "Frozen A0 did not consume the complete reference model trajectory"
                )
        if _summary_result(a1_summary) != frozen_a0.get("expected_summary"):
            raise ValueError("Frozen A0 aggregate result drifted")
        a1_log = a1_root / f"{args.task}.log"
        if not a1_log.is_file():
            raise ValueError(f"Missing frozen A0 task log: {a1_log}")
        actual_trials = _trial_results(a1_log)
        if actual_trials != frozen_a0.get("trials"):
            raise ValueError("Frozen A0 per-trial result trajectory drifted")

    config_suffix = str(getattr(args, "config_suffix", "vf"))
    if config_suffix not in {"vf", "vdm"}:
        raise ValueError(f"Unsupported config suffix: {config_suffix!r}")
    config_name = f"franka_robosuite_{args.task}_multiturn_{config_suffix}.yaml"
    a1_config = _require_one(a1_root / "materialized_configs", f"**/{config_name}")
    core_config = _require_one(core_root / "materialized_configs", f"**/{config_name}")
    a1_config_raw_sha = _sha256(a1_config)
    core_config_raw_sha = _sha256(core_config)
    a1_config_semantic_sha = _semantic_config_sha256(a1_config)
    core_config_semantic_sha = _semantic_config_sha256(core_config)
    if a1_config_semantic_sha != core_config_semantic_sha:
        raise ValueError("A1/Core-FG materialized task config semantics differ")

    blocker_hits: list[dict[str, str]] = []
    for run_root in (a1_root, core_root):
        for log_path in sorted(run_root.rglob("*.log")):
            text = log_path.read_text(encoding="utf-8", errors="replace")
            for name, pattern in CHAIN_BLOCKERS.items():
                if pattern.search(text):
                    blocker_hits.append({"blocker": name, "path": str(log_path)})
    if blocker_hits:
        raise ValueError(f"Known chain blockers found: {blocker_hits}")

    result = {
        "schema": "capx_core_fg_pair_validation_v1",
        "status": "valid",
        "task": args.task,
        "trials": args.trials,
        "required_initial_query_seeds": sorted(required_seeds),
        "initial_query_seed_offset": initial_query_seed_offset,
        "fixed_episode_seed": fixed_episode_seed,
        "episode_seed_base": episode_seed_base,
        "effective_episode_seeds": a1_effective_episode_seeds,
        "episode_seed_mapping_verified": True,
        "config_suffix": config_suffix,
        "replay_hits": len(core_audit),
        "initial_response_hashes_equal": True,
        "initial_state_hashes_equal": True,
        "initial_state_record_count": len(a1_states),
        "materialized_config_sha256": a1_config_semantic_sha,
        "materialized_config_semantic_sha256": a1_config_semantic_sha,
        "materialized_config_raw_sha256": {
            "a1": a1_config_raw_sha,
            "treatment": core_config_raw_sha,
        },
        "known_chain_blocker_count": 0,
        "infra_fingerprint": str(required_files[0]),
        "infra_bundle_sha256": infra_fingerprint["infra_bundle_sha256"],
        "python_packages_sha256": infra_fingerprint["python_packages_sha256"],
        "infra_lock_matched": True,
        "a1_summary": str(a1_summary),
        "core_summary": str(core_summary),
        "a1_qwen_audit": str(required_files[1]),
        "core_qwen_audit": str(required_files[2]),
        "replay_file": str(required_files[5]),
        "deferred_semantic_replay_seeds": [
            record.get("effective_request_seed")
            for record in deferred_semantic_replays
        ],
    }
    if frozen_a0 is not None:
        result.update(
            {
                "frozen_a0": True,
                "frozen_a0_reference": str(frozen_a0_reference),
                "frozen_a0_source_git_rev": frozen_a0["source_git_rev"],
                "frozen_a0_source_pair_job_id": frozen_a0["source_pair_job_id"],
                "frozen_a0_expected_summary": frozen_a0["expected_summary"],
                "frozen_a0_reference_only": frozen_a0_reference_only,
                "frozen_a0_replay_entries_consumed": (
                    None
                    if frozen_a0_reference_only
                    else len(a1_audit_records)
                ),
                "frozen_a0_no_live_model_calls": (
                    None if frozen_a0_reference_only else True
                ),
                "frozen_a0_source_no_failed_model_calls": True,
                "frozen_a0_source_contains_live_model_calls": any(
                    record.get("source") not in REPLAY_SOURCES
                    for record in a1_audit_records
                ),
                "frozen_a0_current_run_model_calls": (
                    0 if frozen_a0_reference_only else len(a1_audit_records)
                ),
                "frozen_a0_reexecuted_in_current_run": (
                    not frozen_a0_reference_only
                ),
                "frozen_a0_per_trial_results_equal": True,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", required=True, type=Path)
    parser.add_argument("--a1-run-root", required=True, type=Path)
    parser.add_argument("--core-run-root", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("cube_stack", "cube_restack"))
    parser.add_argument("--trials", required=True, type=int)
    parser.add_argument("--global-seed", required=True, type=int)
    parser.add_argument("--initial-query-seed-offset", type=int, default=0)
    parser.add_argument("--fixed-episode-seed", type=int)
    parser.add_argument("--episode-seed-base", type=int)
    parser.add_argument("--config-suffix", choices=("vf", "vdm"), default="vf")
    parser.add_argument("--frozen-a0-reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        result = validate(args)
    except Exception as exc:
        result = {
            "schema": "capx_core_fg_pair_validation_v1",
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
