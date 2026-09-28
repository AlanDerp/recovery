#!/usr/bin/env python3
"""Fail-closed validation for A1/envelope-replacement paired runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from validate_paired_run import validate as validate_pair


SCHEMA_VERSION = "capx-envelope-replacement-v1"
TRACE_DIR = "envelope_replacement_traces"
OLD_VISIBLE_MARKERS = (
    "GT_BACKED_FACTUAL_CONTEXT:",
    "GT_BACKED_VERIFIER_FACTS:",
    "PGSVP MATERIALIZED STATE CONTRACT",
    "PGSVP PERCEIVED-TRUTH WORKSHEET",
    "PGSVP_PT_GT_CORRECTIONS:",
)
FORBIDDEN_REPLACEMENT_TEXT = (
    "reward",
    "task_completed",
    "success label",
    "goto_pose",
    "open_gripper",
    "close_gripper",
    "sample_grasp_pose",
    "next action",
    "repair plan",
)
_CUBE_EXTENT_RE = re.compile(
    r"Object extent for (?:red|green) cube:\s*\[([^\]]+)\]"
)
_MIN_CUBE_FULL_EXTENT_METERS = 0.005
_MAX_CUBE_FULL_EXTENT_METERS = 0.20
_MAX_SILENT_IMAGE_CHANGED_CHANNELS = 32


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        records.append(value)
    return records


def _manifest_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip():
            raise ValueError(f"Malformed manifest line {line_number}: {path}")
        fields[key.strip()] = value.strip()
    return fields


def _unique_visual_feedback_path(
    run_root: Path,
    task: str,
    trial: int,
    index: int,
) -> Path:
    name = f"visual_feedback_{index:02d}.png"
    paths = sorted(run_root.glob(f"**/{task}/trial_{trial:02d}_*/{name}"))
    if not paths:
        raise ValueError(f"Missing visual feedback {name} under {run_root}")
    hashes = {_sha256_bytes(path.read_bytes()) for path in paths}
    if len(hashes) != 1:
        raise ValueError(
            f"Visual feedback archive has conflicting copies of {name}: {run_root}"
        )
    return paths[0]


def _validate_silent_visual_equivalence(
    *,
    a1_run_root: Path,
    treatment_run_root: Path,
    task: str,
    trial: int,
    turn: int,
) -> dict[str, Any]:
    a1_path = _unique_visual_feedback_path(a1_run_root, task, trial, turn)
    treatment_path = _unique_visual_feedback_path(
        treatment_run_root, task, trial, turn
    )
    with Image.open(a1_path) as a1_image, Image.open(treatment_path) as treatment_image:
        a1_rgb = a1_image.convert("RGB")
        treatment_rgb = treatment_image.convert("RGB")
        if a1_rgb.size != treatment_rgb.size:
            raise ValueError("Silent A1/treatment visual feedback shapes differ")
        a1_bytes = a1_rgb.tobytes()
        treatment_bytes = treatment_rgb.tobytes()
    differences = [
        abs(first - second)
        for first, second in zip(a1_bytes, treatment_bytes, strict=True)
        if first != second
    ]
    changed_channels = len(differences)
    if changed_channels > _MAX_SILENT_IMAGE_CHANGED_CHANNELS:
        raise ValueError(
            "Normalized silent replay exceeded the visual-equivalence bound: "
            f"turn={turn}, changed_channels={changed_channels}, "
            f"limit={_MAX_SILENT_IMAGE_CHANGED_CHANNELS}"
        )
    return {
        "turn": turn,
        "trial": trial,
        "a1_image_sha256": _sha256_bytes(a1_path.read_bytes()),
        "treatment_image_sha256": _sha256_bytes(treatment_path.read_bytes()),
        "pixel_shape": [a1_rgb.height, a1_rgb.width, 3],
        "changed_channels": changed_channels,
        "total_channels": len(a1_bytes),
        "max_absolute_channel_difference": max(differences, default=0),
        "bound": _MAX_SILENT_IMAGE_CHANGED_CHANNELS,
    }


def _validate_a0_execution_provenance(
    *,
    pair_manifest: dict[str, str],
    frozen_a0_reference: Path | None,
    task: str,
    trials: int,
) -> str:
    mode = pair_manifest.get(
        "a0_execution_mode",
        "replayed_in_current_allocation",
    )
    if mode == "frozen_reference_only":
        source_validation = Path(
            pair_manifest.get("frozen_a0_source_pair_validation", "")
        )
        expected_sha = pair_manifest.get(
            "frozen_a0_source_pair_validation_sha256"
        )
        if (
            pair_manifest.get("a1_reexecuted") != "false"
            or pair_manifest.get("same_slurm_allocation") != "false"
            or not source_validation.is_file()
            or not isinstance(expected_sha, str)
            or _sha256_bytes(source_validation.read_bytes()) != expected_sha
        ):
            raise ValueError("Frozen A1 provenance is invalid")
        if frozen_a0_reference is None or not frozen_a0_reference.is_file():
            raise ValueError("Frozen A1 mode requires its registered reference")
        reference = json.loads(
            frozen_a0_reference.read_text(encoding="utf-8")
        )
        if reference.get("source_pair_validation_sha256") != expected_sha:
            raise ValueError("Frozen A1 source validation is not registered")
        source_record = json.loads(
            source_validation.read_text(encoding="utf-8")
        )
        if (
            source_record.get("status") != "valid"
            or source_record.get("task") != task
            or source_record.get("trials") != trials
        ):
            raise ValueError("Frozen A1 validation is incompatible")
    elif mode not in {"replayed_in_current_allocation", "live_then_export"}:
        raise ValueError(f"Unknown A1 execution mode: {mode}")
    return mode


def _validate_runner_snapshot(
    *,
    pair_root: Path,
    pair_manifest: dict[str, str],
) -> None:
    runner = Path(pair_manifest.get("submission_runner_snapshot", ""))
    expected_sha = pair_manifest.get("submission_runner_sha256")
    if (
        pair_manifest.get("submission_runner_read_only") != "true"
        or pair_manifest.get("submission_git_rev")
        != pair_manifest.get("git_rev")
        or pair_manifest.get("submission_runner_git_rev")
        != pair_manifest.get("git_rev")
        or runner.parent.resolve() != pair_root.resolve()
        or runner.name != "SUBMISSION_RUNNER.sbatch"
        or not runner.is_file()
        or not isinstance(expected_sha, str)
        or _sha256_bytes(runner.read_bytes()) != expected_sha
        or runner.stat().st_mode & 0o222
    ):
        raise ValueError("Top-level runner snapshot provenance is invalid")


def _contains_marker(value: Any, marker: str) -> bool:
    if isinstance(value, str):
        return marker in value
    if isinstance(value, dict):
        return any(_contains_marker(item, marker) for item in value.values())
    if isinstance(value, list):
        return any(_contains_marker(item, marker) for item in value)
    return False


def _validate_initial_and_saved_prompts(run_root: Path) -> None:
    files = sorted(run_root.glob("**/all_responses.json"))
    if not files:
        raise ValueError(f"No saved response artifacts under {run_root}")
    for path in files:
        responses = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(responses, list):
            raise ValueError(f"Malformed response artifact: {path}")
        for response in responses:
            if not isinstance(response, dict):
                raise ValueError(f"Malformed response record: {path}")
            for field in ("initial_prompt", "multi_turn_prompt"):
                prompt = response.get(field)
                if prompt is None:
                    continue
                for marker in OLD_VISIBLE_MARKERS:
                    if _contains_marker(prompt, marker):
                        raise ValueError(
                            f"Old visible FG marker {marker!r} leaked at {path}"
                        )


def _action_message_hashes(qwen_audit: Path) -> set[str]:
    hashes: set[str] = set()
    records = _jsonl(qwen_audit)
    if not records:
        raise ValueError("Treatment Qwen audit is empty")
    for record in records:
        request = record.get("request_canonical")
        if isinstance(request, str):
            try:
                request = json.loads(request)
            except json.JSONDecodeError:
                continue
        if not isinstance(request, dict):
            continue
        messages = request.get("messages")
        if isinstance(messages, list):
            hashes.add(_sha256_json(messages))
    if not hashes:
        raise ValueError("Treatment Qwen audit has no parseable raw requests")
    return hashes


def _request_record_for_seed(
    records: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if record.get("effective_request_seed") == seed
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one action request for seed {seed}, found {len(matches)}"
        )
    return matches[0]


def _message_hash_for_request_record(record: dict[str, Any]) -> str:
    request = record.get("request_canonical")
    if isinstance(request, str):
        request = json.loads(request)
    if not isinstance(request, dict) or not isinstance(
        request.get("messages"),
        list,
    ):
        raise ValueError("Action request has no canonical messages")
    return _sha256_json(request["messages"])


def _validate_cube_extent_semantics(run_root: Path, task: str) -> int:
    """Reject finite-but-semantic OBB failures hidden behind a successful API call."""
    task_log = run_root / f"{task}.log"
    if not task_log.is_file():
        raise ValueError(f"Missing task log for OBB validation: {task_log}")
    checked = 0
    for match in _CUBE_EXTENT_RE.finditer(
        task_log.read_text(encoding="utf-8", errors="replace")
    ):
        values = [float(token) for token in match.group(1).split()]
        if len(values) != 3:
            raise ValueError(f"Malformed cube OBB extent: {match.group(0)}")
        checked += 1
        if any(
            not (_MIN_CUBE_FULL_EXTENT_METERS <= value <= _MAX_CUBE_FULL_EXTENT_METERS)
            for value in values
        ):
            raise ValueError(
                "Silent cube OBB semantic corruption: "
                f"{match.group(0)}; required each full extent in "
                f"[{_MIN_CUBE_FULL_EXTENT_METERS}, "
                f"{_MAX_CUBE_FULL_EXTENT_METERS}] meters"
            )
    return checked


def validate(args: argparse.Namespace) -> dict[str, Any]:
    pair_manifest_path = args.pair_root / "PAIR_MANIFEST.txt"
    if not pair_manifest_path.is_file():
        raise ValueError(f"Missing pair manifest: {pair_manifest_path}")
    manifest = _manifest_fields(pair_manifest_path)
    if (
        manifest.get("treatment") != "envelope_replacement"
        or manifest.get("treatment_label") != "Envelope-Replacement-v1"
    ):
        raise ValueError("Pair manifest selected the wrong treatment")
    _validate_runner_snapshot(
        pair_root=args.pair_root,
        pair_manifest=manifest,
    )
    frozen_reference = getattr(args, "frozen_a0_reference", None)
    frozen_reference_path = (
        Path(frozen_reference) if frozen_reference is not None else None
    )
    a0_mode = _validate_a0_execution_provenance(
        pair_manifest=manifest,
        frozen_a0_reference=frozen_reference_path,
        task=args.task,
        trials=args.trials,
    )
    base = validate_pair(
        SimpleNamespace(
            pair_root=args.pair_root,
            a1_run_root=args.a1_run_root,
            core_run_root=args.treatment_run_root,
            task=args.task,
            trials=args.trials,
            global_seed=args.global_seed,
            frozen_a0_reference=frozen_reference_path,
            frozen_a0_reference_only=(a0_mode == "frozen_reference_only"),
        )
    )

    _validate_initial_and_saved_prompts(args.treatment_run_root)
    treatment_qwen_audit = (
        args.treatment_run_root / "qwen_proxy_audit.jsonl"
    )
    action_hashes = _action_message_hashes(treatment_qwen_audit)
    treatment_qwen_records = _jsonl(treatment_qwen_audit)
    treatment_replay = Path(
        manifest.get("treatment_request_hash_replay_file", "")
    )
    runtime_replay_sha = manifest.get(
        "treatment_request_hash_replay_runtime_sha256"
    )
    if (
        not treatment_replay.is_file()
        or not isinstance(runtime_replay_sha, str)
        or _sha256_bytes(treatment_replay.read_bytes()) != runtime_replay_sha
    ):
        raise ValueError("Treatment request-hash replay provenance is invalid")
    trial_roots = sorted(
        args.treatment_run_root.glob(f"**/{TRACE_DIR}/trial_*")
    )
    if len(trial_roots) != args.trials:
        raise ValueError(
            f"Expected {args.trials} envelope trial traces, found "
            f"{len(trial_roots)}"
        )
    treatment_log = args.treatment_run_root / f"{args.task}.log"
    if not treatment_log.is_file():
        raise ValueError(f"Missing treatment task log: {treatment_log}")
    treatment_log_text = treatment_log.read_text(
        encoding="utf-8",
        errors="replace",
    )
    if (
        "child process failed" in treatment_log_text
        and "capx/verification/envelope_replacement.py" in treatment_log_text
    ):
        raise ValueError("Envelope verifier caused a child-process failure")
    treatment_cube_extent_count = _validate_cube_extent_semantics(
        args.treatment_run_root,
        args.task,
    )
    a1_cube_extent_count = 0
    if a0_mode != "frozen_reference_only":
        a1_cube_extent_count = _validate_cube_extent_semantics(
            args.a1_run_root,
            args.task,
        )

    turn_count = 0
    repair_count = 0
    no_op_count = 0
    silent_replay_count = 0
    silent_exact_request_replay_count = 0
    silent_normalized_request_replay_count = 0
    silent_legacy_runtime_replay_count = 0
    silent_visual_equivalence: list[dict[str, Any]] = []
    grounded_live_count = 0
    verifier_prompt_hashes: set[str] = set()
    verifier_schema_hashes: set[str] = set()
    adjudicator_prompt_hashes: set[str] = set()
    adjudicator_schema_hashes: set[str] = set()
    for trial_root in trial_roots:
        initialization_path = trial_root / "initialization.json"
        if not initialization_path.is_file():
            raise ValueError(f"Missing envelope initialization: {trial_root}")
        initialization = json.loads(
            initialization_path.read_text(encoding="utf-8")
        )
        if (
            initialization.get("schema_version") != SCHEMA_VERSION
            or initialization.get("task") != args.task
            or initialization.get("stage") != "INITIALIZED_BEFORE_ACTION"
            or initialization.get("action_visible_gt_packet_injected") is not False
            or initialization.get("action_visible_bridge_injected") is not False
            or initialization.get("student_pt_call_performed") is not False
        ):
            raise ValueError(
                f"Invalid envelope treatment initialization: {trial_root}"
            )
        records = sorted(trial_root.glob("turn_*/replacement.json"))
        incomplete_turns = [
            turn_root
            for turn_root in sorted(trial_root.glob("turn_*"))
            if not (turn_root / "replacement.json").is_file()
        ]
        if incomplete_turns:
            raise ValueError(
                f"Incomplete envelope replacement turns: {incomplete_turns}"
            )
        if not records:
            successful_initial = sorted(
                args.treatment_run_root.glob(
                    f"**/{args.task}/{trial_root.name}_*taskcompleted_1"
                )
            )
            if not successful_initial:
                raise ValueError(
                    "Trial ended without replacement turn or successful initial "
                    f"policy: {trial_root}"
                )
            continue
        for path in records:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                record.get("schema_version") != SCHEMA_VERSION
                or record.get("task") != args.task
            ):
                raise ValueError(f"Envelope trace schema/task mismatch: {path}")
            required_true = (
                "non_text_structure_identical",
                "only_candidate_sources_mutated",
                "action_message_count_unchanged",
            )
            required_false = (
                "action_visible_gt_packet_injected",
                "action_visible_bridge_injected",
                "action_visible_verifier_protocol_injected",
                "student_pt_call_performed",
                "raw_image_replaced",
                "runtime_stderr_replaced",
                "task_or_api_text_replaced",
            )
            if any(record.get(field) is not True for field in required_true):
                raise ValueError(f"Envelope positive audit failed: {path}")
            if any(record.get(field) is not False for field in required_false):
                raise ValueError(f"Envelope isolation audit failed: {path}")
            candidates = record.get("candidate_chunks")
            proposed_repairs = record.get("proposed_repairs")
            adjudications = record.get("adjudications")
            repairs = record.get("repairs")
            patches = record.get("applied_patches")
            if (
                not isinstance(candidates, list)
                or not isinstance(proposed_repairs, list)
                or not isinstance(adjudications, list)
                or not isinstance(repairs, list)
                or not isinstance(patches, list)
                or record.get("candidate_count") != len(candidates)
                or record.get("repair_count") != len(repairs)
                or len(repairs) != len(patches)
                or len(proposed_repairs) != len(adjudications)
            ):
                raise ValueError(f"Envelope count audit failed: {path}")
            candidate_map = {
                candidate.get("chunk_id"): candidate
                for candidate in candidates
                if isinstance(candidate, dict)
            }
            if len(candidate_map) != len(candidates):
                raise ValueError(f"Envelope candidate IDs are invalid: {path}")
            if any(
                candidate.get("source") != "executed_code_comment"
                for candidate in candidates
            ):
                raise ValueError(
                    f"Non-predicate evidence was exposed as repairable: {path}"
                )
            if (
                record.get("repaired_console_stdout_sha256")
                != record.get("console_stdout_sha256")
            ):
                raise ValueError(f"Envelope treatment changed stdout: {path}")
            accepted_repairs = []
            for repair_index, (proposal, decision) in enumerate(
                zip(proposed_repairs, adjudications, strict=True)
            ):
                if (
                    decision.get("repair_index") != repair_index
                    or decision.get("chunk_id") != proposal.get("chunk_id")
                    or decision.get("original_claim")
                    != proposal.get("original_claim")
                    or decision.get("replacement_claim")
                    != proposal.get("replacement_claim")
                ):
                    raise ValueError(
                        f"Contradiction-gate provenance failed: {path}"
                    )
                expected_acceptance = (
                    decision.get("relation") == "contradicted"
                    and decision.get("span_is_smallest_complete_clause") is True
                    and decision.get("selected_facts_entail_claim_negation") is True
                    and decision.get(
                        "replacement_claim_entailed_by_selected_facts"
                    )
                    is True
                    and decision.get("replacement_preserves_all_other_meaning")
                    is True
                )
                if decision.get("accepted") is not expected_acceptance:
                    raise ValueError(
                        f"Contradiction-gate acceptance audit failed: {path}"
                    )
                if expected_acceptance:
                    accepted_repairs.append(proposal)
            if accepted_repairs != repairs:
                raise ValueError(
                    f"Applied repairs bypassed the contradiction gate: {path}"
                )
            fact_catalog = record.get("private_fact_catalog")
            if not isinstance(fact_catalog, dict):
                raise ValueError(f"Missing private fact catalog: {path}")
            for repair, patch in zip(repairs, patches):
                chunk_id = repair.get("chunk_id")
                candidate = candidate_map.get(chunk_id)
                if (
                    not isinstance(candidate, dict)
                    or repair.get("original_sha256")
                    != candidate.get("sha256")
                    or patch.get("chunk_id") != chunk_id
                    or patch.get("candidate_sha256")
                    != candidate.get("sha256")
                    or repair.get("original_claim")
                    != patch.get("original_claim")
                    or repair.get("original_claim_sha256")
                    != patch.get("original_claim_sha256")
                    or repair.get("replacement_claim")
                    != patch.get("replacement_claim")
                    or repair.get("replacement_claim_sha256")
                    != patch.get("replacement_sha256")
                    or _sha256_text(str(patch.get("original_claim", "")))
                    != patch.get("original_claim_sha256")
                    or _sha256_text(str(patch.get("replacement_text", "")))
                    != patch.get("replacement_sha256")
                ):
                    raise ValueError(f"Envelope patch provenance failed: {path}")
                if not isinstance(repair.get("replacement_fact_ids"), list):
                    raise ValueError(f"Envelope repair has no GT facts: {path}")
                replacement_text = str(patch.get("replacement_text", ""))
                replacement = replacement_text.lower()
                if replacement_text != repair.get("replacement_claim"):
                    raise ValueError(
                        f"Envelope replacement is not the gated local edit: {path}"
                    )
                if re.search(r"\d", replacement):
                    raise ValueError(
                        f"Envelope replacement exposes numeric data: {path}"
                    )
                if any(term in replacement for term in FORBIDDEN_REPLACEMENT_TEXT):
                    raise ValueError(
                        f"Envelope replacement contains forbidden content: {path}"
                    )
            native_hash = record.get("native_prompt_sha256")
            corrected_hash = record.get("corrected_prompt_sha256")
            if not isinstance(corrected_hash, str) or corrected_hash not in action_hashes:
                raise ValueError(
                    f"Corrected action prompt was not sent to the API: {path}"
                )
            active_repairs = record.get("active_persisted_repair_count")
            if (
                not isinstance(active_repairs, int)
                or active_repairs < len(repairs)
            ):
                raise ValueError(f"Envelope history repair count failed: {path}")
            if active_repairs:
                if corrected_hash == native_hash:
                    raise ValueError(f"Envelope repair was a prompt no-op: {path}")
                grounded_live_count += 1
            else:
                no_op_count += 1
                if (
                    corrected_hash != native_hash
                    or record.get("no_op_prompt_byte_equivalent") is not True
                ):
                    raise ValueError(f"Envelope no-op changed A1 prompt: {path}")
                silent_replay_count += 1
            action_seed = (
                args.global_seed
                + int(record["trial"]) * 1000
                + int(record["turn"])
            )
            action_request = _request_record_for_seed(
                treatment_qwen_records,
                action_seed,
            )
            if _message_hash_for_request_record(action_request) != corrected_hash:
                raise ValueError(
                    f"Envelope turn is not bound to its action request: {path}"
                )
            replay_source = action_request.get("source")
            replayed = replay_source in {
                "request_hash_replay",
                "request_normalized_hash_replay",
                "request_legacy_runtime_normalized_hash_replay",
            }
            if active_repairs == 0:
                if replay_source == "request_hash_replay":
                    silent_exact_request_replay_count += 1
                elif replay_source == "request_normalized_hash_replay":
                    silent_normalized_request_replay_count += 1
                    visual_audit = _validate_silent_visual_equivalence(
                        a1_run_root=args.a1_run_root,
                        treatment_run_root=args.treatment_run_root,
                        task=args.task,
                        trial=int(record["trial"]),
                        turn=int(record["turn"]),
                    )
                    silent_visual_equivalence.append(visual_audit)
                elif replay_source == "request_legacy_runtime_normalized_hash_replay":
                    if action_request.get("replay_legacy_runtime_root") != (
                        "/home/qshou/capx_pgsvp_v9_verify_clean"
                    ):
                        raise ValueError(
                            f"Unregistered legacy runtime replay root: {path}"
                        )
                    silent_legacy_runtime_replay_count += 1
                else:
                    raise ValueError(
                        f"Silent envelope turn did not replay A1: {path}"
                    )
            if active_repairs > 0 and replayed:
                raise ValueError(
                    f"Grounded envelope turn unexpectedly replayed A1: {path}"
                )
            verifier_prompt_hashes.add(str(record.get("verifier_prompt_sha256")))
            verifier_schema_hashes.add(str(record.get("verifier_schema_sha256")))
            adjudicator_prompt_hashes.add(
                str(record.get("adjudicator_prompt_sha256"))
            )
            adjudicator_schema_hashes.add(
                str(record.get("adjudicator_schema_sha256"))
            )
            repair_count += len(repairs)
            turn_count += 1
        history_path = trial_root / "verified_code_history.json"
        if not history_path.is_file():
            raise ValueError(f"Missing verified code history: {trial_root}")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if (
            history.get("schema_version") != SCHEMA_VERSION
            or history.get("task") != args.task
            or not isinstance(history.get("blocks"), list)
            or len(history["blocks"]) != len(records)
        ):
            raise ValueError(f"Verified code history is incomplete: {history_path}")
        for block_index, (path, block) in enumerate(
            zip(records, history["blocks"], strict=True)
        ):
            record = json.loads(path.read_text(encoding="utf-8"))
            prefix = {
                **history,
                "blocks": history["blocks"][: block_index + 1],
            }
            if (
                block.get("block_index") != block_index
                or block.get("verified_after_turn") != block_index + 1
                or record.get("latest_code_block_index") != block_index
                or record.get("verified_history_sha256")
                != _sha256_json(prefix)
            ):
                raise ValueError(
                    f"Verified code history provenance failed: {history_path}"
                )

    if turn_count:
        if len(verifier_prompt_hashes) != 1 or "" in verifier_prompt_hashes:
            raise ValueError("Envelope verifier prompt changed within the run")
        if len(verifier_schema_hashes) != 1 or "" in verifier_schema_hashes:
            raise ValueError("Envelope verifier schema changed within the run")
        verifier_prompt_sha = next(iter(verifier_prompt_hashes))
        verifier_schema_sha = next(iter(verifier_schema_hashes))
        if (
            len(adjudicator_prompt_hashes) != 1
            or "" in adjudicator_prompt_hashes
        ):
            raise ValueError("Envelope adjudicator prompt changed within the run")
        if (
            len(adjudicator_schema_hashes) != 1
            or "" in adjudicator_schema_hashes
        ):
            raise ValueError("Envelope adjudicator schema changed within the run")
        adjudicator_prompt_sha = next(iter(adjudicator_prompt_hashes))
        adjudicator_schema_sha = next(iter(adjudicator_schema_hashes))
    else:
        verifier_prompt_sha = manifest.get("verifier_prompt_sha256")
        verifier_schema_sha = manifest.get("verifier_schema_sha256")
        if not verifier_prompt_sha or not verifier_schema_sha:
            raise ValueError("Envelope verifier asset hashes are absent")
        adjudicator_prompt_sha = manifest.get("adjudicator_prompt_sha256")
        adjudicator_schema_sha = manifest.get("adjudicator_schema_sha256")
        if not adjudicator_prompt_sha or not adjudicator_schema_sha:
            raise ValueError("Envelope adjudicator asset hashes are absent")
    base.update(
        {
            "schema": "capx_envelope_replacement_pair_validation_v1",
            "treatment": "Envelope-Replacement-v1",
            "envelope_trial_trace_count": len(trial_roots),
            "envelope_turn_count": turn_count,
            "envelope_repair_count": repair_count,
            "envelope_no_op_turn_count": no_op_count,
            "silent_a1_regeneration_replay_count": silent_replay_count,
            "silent_exact_request_replay_count": (
                silent_exact_request_replay_count
            ),
            "silent_normalized_request_replay_count": (
                silent_normalized_request_replay_count
            ),
            "silent_legacy_runtime_replay_count": (
                silent_legacy_runtime_replay_count
            ),
            "silent_normalized_replay_visual_audit": (
                silent_visual_equivalence
            ),
            "grounded_live_regeneration_count": grounded_live_count,
            "silent_turns_replayed_exact_a1_responses": True,
            "temporal_repairs_bound_to_latest_block": True,
            "verified_code_history_persisted": True,
            "no_op_prompts_byte_identical": True,
            "corrected_prompts_matched_raw_api_requests": True,
            "old_visible_fg_markers_absent": True,
            "student_pt_calls": 0,
            "action_visible_full_gt_packets": 0,
            "action_visible_bridges": 0,
            "raw_images_replaced": 0,
            "stderr_spans_replaced": 0,
            "a1_cube_obb_extent_count": a1_cube_extent_count,
            "treatment_cube_obb_extent_count": treatment_cube_extent_count,
            "silent_cube_obb_semantic_corruption_count": 0,
            "verifier_prompt_sha256": verifier_prompt_sha,
            "verifier_schema_sha256": verifier_schema_sha,
            "adjudicator_prompt_sha256": adjudicator_prompt_sha,
            "adjudicator_schema_sha256": adjudicator_schema_sha,
            "a0_execution_mode": a0_mode,
            "a1_reexecuted": a0_mode != "frozen_reference_only",
        }
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", required=True, type=Path)
    parser.add_argument("--a1-run-root", required=True, type=Path)
    parser.add_argument("--treatment-run-root", required=True, type=Path)
    parser.add_argument(
        "--task",
        required=True,
        choices=("cube_stack", "cube_restack"),
    )
    parser.add_argument("--trials", required=True, type=int)
    parser.add_argument("--global-seed", required=True, type=int)
    parser.add_argument("--frozen-a0-reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = validate(args)
    except Exception as exc:
        result = {
            "schema": "capx_envelope_replacement_pair_validation_v1",
            "status": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
        }
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
