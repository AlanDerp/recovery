#!/usr/bin/env python3
"""Fail-closed validation for a paired Ideal V4 canonical-delta run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from validate_paired_run import validate as validate_pair

TRACE_DIR = "ideal_tk_v4_traces"
SCHEMA_VERSION = "capx-ideal-tk-v4-observation-integrity"
SCHEMA_VERSIONS = {
    "capx-ideal-tk-v4-semantic-gain-gate",
    SCHEMA_VERSION,
}
TREATMENT_LABEL = "Ideal-TK-v4-canonical-delta"
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
INLINE_API_PROVENANCE = re.compile(
    r", during public API `([A-Za-z_][A-Za-z0-9_]*)`"
)
EFFECT_ALIGNMENT_HEADERS = (
    "Verified operator-effect contradictions:",
    "Verified operator-effect prefix:",
    "Verified compact operator outcome:",
    "Verified operator-effect ledger:",
)
EFFECT_ALIGNMENT_SEPARATOR = f"\n\n{EFFECT_ALIGNMENT_HEADERS[0]}\n"
OBSERVATION_INTEGRITY_HEADER = "Verified native-observation integrity:"
OBSERVATION_INTEGRITY_SEPARATOR = f"\n\n{OBSERVATION_INTEGRITY_HEADER}\n"
TRIAL_CHAIN_FAILURE = re.compile(
    r"^Trial \d+ (?:"
    r"timed out after \d+ seconds\."
    r"|exceeded process watchdog timeout \(\d+ seconds\)(?:; terminating child process)?"
    r"|child process failed\."
    r"|child process exited without a summary\."
    r")$",
    re.MULTILINE,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_observation_integrity_text(object_ids: list[str]) -> str:
    names = {"obj01": "red cube", "obj02": "green cube"}
    facts = "\n".join(
        "- At least one extent returned by executed `get_object_pose` for "
        f"{names[object_id]} was inconsistent with the simulator-backed "
        "rigid geometry of that same object."
        for object_id in object_ids
    )
    return f"{OBSERVATION_INTEGRITY_HEADER}\n{facts}"


def _strip_observation_integrity(
    report: str,
    *,
    applied: bool,
    object_ids: list[str],
    expected_text_sha256: str | None,
    path: Path,
) -> str:
    separators = report.count(OBSERVATION_INTEGRITY_SEPARATOR)
    if not applied:
        if separators or OBSERVATION_INTEGRITY_HEADER in report:
            raise ValueError(f"Silent takeover contains observation integrity: {path}")
        if expected_text_sha256 not in {None, _sha256_json(None)}:
            raise ValueError(f"Silent observation-integrity hash is invalid: {path}")
        return report

    if separators != 1 or object_ids != sorted(set(object_ids)) or not object_ids:
        raise ValueError(f"Observation-integrity appendix is malformed: {path}")
    g17_report, body = report.rsplit(OBSERVATION_INTEGRITY_SEPARATOR, 1)
    text = f"{OBSERVATION_INTEGRITY_HEADER}\n{body}"
    if (
        text != _expected_observation_integrity_text(object_ids)
        or expected_text_sha256 != _sha256_json(text)
    ):
        raise ValueError(f"Observation-integrity appendix is invalid: {path}")
    return g17_report


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


def _assert_no_trial_chain_failures(run_root: Path) -> int:
    """Reject runner-level failures without interpreting student content."""
    summaries = sorted(run_root.glob("**/trial_*/summary.txt"))
    failures: list[str] = []
    for path in summaries:
        text = path.read_text(encoding="utf-8", errors="replace")
        match = TRIAL_CHAIN_FAILURE.search(text)
        if match is not None:
            failures.append(f"{path}: {match.group(0)}")
    if failures:
        raise ValueError(
            "Treatment contains runner-level trial chain failures:\n"
            + "\n".join(failures)
        )
    return len(summaries)


def _validate_visible_temporal_fields(
    *,
    path: Path,
    audited_event_fields: list[str],
    visible_event_fields: Any,
    ledger_applied: bool,
) -> None:
    if not isinstance(visible_event_fields, list):
        raise ValueError(f"Takeover visible temporal fields are malformed: {path}")
    if ledger_applied:
        if visible_event_fields:
            raise ValueError(
                f"Ledger takeover leaked ordered-event fields into its carrier: {path}"
            )
    elif audited_event_fields != visible_event_fields:
        raise ValueError(f"Takeover temporal event field order mismatch: {path}")


def _validate_response_frozen_audit(
    manifest: dict[str, str], audit: list[dict[str, Any]]
) -> dict[str, Any]:
    mode = manifest.get("treatment_replay_mode", "registration_live")
    if mode == "registration_live":
        return {}
    if mode != "response_frozen_audit":
        raise ValueError(f"Unsupported Ideal V4 treatment replay mode: {mode}")

    replay_path = Path(manifest.get("treatment_request_hash_replay_file", ""))
    if not replay_path.is_file():
        raise ValueError("Response-frozen audit replay package is missing")
    expected_sha = manifest.get("treatment_request_hash_replay_sha256")
    if not expected_sha or _sha256_file(replay_path) != expected_sha:
        raise ValueError("Response-frozen audit replay hash mismatch")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    entries = replay.get("entries")
    if (
        replay.get("schema") != "capx_request_hash_replay_v1"
        or replay.get("missing_count") != 0
        or replay.get("duplicate_request_conflicts")
        or not isinstance(entries, dict)
        or not entries
    ):
        raise ValueError("Response-frozen audit replay package is incomplete")
    if not audit:
        raise ValueError("Response-frozen audit contains no model calls")

    consumed: set[str] = set()
    for record in audit:
        if record.get("ok") is not True or record.get("source") not in REPLAY_SOURCES:
            raise ValueError("Response-frozen audit made a live or failed model call")
        replay_request = record.get("replay_request_sha256")
        entry = entries.get(replay_request) if isinstance(replay_request, str) else None
        if not isinstance(entry, dict):
            raise ValueError("Response-frozen audit consumed an unregistered request")
        if entry.get("response_sha256") != record.get("response_content_sha256"):
            raise ValueError("Response-frozen audit response hash drifted")
        consumed.add(replay_request)

    return {
        "response_frozen_audit": True,
        "response_frozen_replay": str(replay_path),
        "response_frozen_replay_sha256": expected_sha,
        "response_frozen_source_run_root": replay.get("source_run_root"),
        "response_frozen_model_calls": len(audit),
        "response_frozen_replay_entries": len(entries),
        "response_frozen_replay_entries_consumed": len(consumed),
        "response_frozen_no_live_model_calls": True,
        "response_frozen_response_hashes_equal": True,
    }


def _expected_semantic_hashes(
    reference: dict[str, Any],
    *,
    task: str,
    global_seed: int,
    trials: int,
    a1_run_root: Path,
) -> dict[str, dict[str, str]]:
    """Decode either supported carrier for passive A1 semantics."""
    identity = {
        "task": task,
        "native_mode": "M3",
        "global_seed": global_seed,
        "total_trials": trials,
    }
    if any(reference.get(key) != value for key, value in identity.items()):
        raise ValueError("Ideal V4 semantic reference identity mismatch")

    if reference.get("schema") == "capx_closed_gt_takeover_passive_reference_v2.1":
        if (
            reference.get("capture_mode") != "passive_action_invisible"
            or reference.get("student_prompt_changed") is not False
            or reference.get("student_action_request_changed") is not False
            or Path(reference.get("source_run_root", "")).resolve()
            != a1_run_root.resolve()
        ):
            raise ValueError("Ideal V4 direct passive reference is invalid")
        hashes = reference.get("closed_gt_turn_semantics_sha256")
    elif reference.get("schema") == "capx_frozen_a0_reference_v1":
        if (
            reference.get("m3_semantic_source_capture_mode")
            != "passive_action_invisible"
            or not isinstance(reference.get("m3_semantic_source_reference_sha256"), str)
            or not isinstance(reference.get("source_pair_validation_sha256"), str)
        ):
            raise ValueError("Ideal V4 frozen A0 semantic reference is invalid")
        hashes = reference.get("m3_semantic_transition_sha256")
    else:
        raise ValueError("Ideal V4 semantic reference schema is unsupported")

    if not isinstance(hashes, dict):
        raise ValueError("Ideal V4 semantic transition hashes are missing")
    return hashes


def _validate_trace(
    root: Path, *, action_message_hashes: Counter[str]
) -> tuple[int, dict[str, str]]:
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    if (
        state.get("schema_version") not in SCHEMA_VERSIONS
        or state.get("continuous_state_persisted") is not False
        or state.get("action_visible_initial_gt") is not False
        or state.get("sim_step_observer_installed") is not True
        or not isinstance(state.get("wrapped_api_names"), list)
        or not state.get("wrapped_api_names")
    ):
        raise ValueError(f"Invalid takeover trial state: {root}")
    paths = sorted(root.glob("turn_*/takeover.json"))
    if state.get("turns_completed") != len(paths):
        raise ValueError(f"Takeover trace is not turn-synchronous: {root}")
    transition_hashes: dict[str, str] = {}
    for expected_turn, path in enumerate(paths, 1):
        record = json.loads(path.read_text(encoding="utf-8"))
        effect_enabled = record.get("effect_alignment_enabled")
        effect_mode = record.get("effect_alignment_mode", "contradictions")
        effect_status = record.get("effect_alignment_status")
        effect_expected = record.get("effect_alignment_expected_effects")
        effect_contradicted = record.get("effect_alignment_contradicted_effects")
        effect_visible = effect_status == "mismatch"
        release_enabled = record.get("release_bound_placement_enabled")
        release_applied = record.get("release_bound_placement_applied")
        release_outcome = record.get("release_bound_placement")
        release_candidate = record.get("release_bound_placement_candidate")
        k2_enabled = record.get("k2_release_delta_enabled")
        k2_applied = record.get("k2_release_delta_applied")
        k2_delta = record.get("k2_release_delta")
        ledger_enabled = record.get("api_transition_ledger_enabled")
        ledger_mode = record.get("api_transition_ledger_mode")
        ledger_applied = record.get("api_transition_ledger_applied")
        ledger = record.get("api_transition_ledger")
        ledger_candidate = record.get("api_transition_ledger_candidate")
        gain_enabled = record.get("semantic_gain_gate_enabled")
        gain_status = record.get("semantic_gain_status")
        gain_facts = record.get("semantic_gain_novel_fact_ids")
        gain_extension_applied = record.get("semantic_gain_extension_applied")
        parent_preserved = record.get("parent_prompt_preserved")
        integrity_enabled = record.get("observation_integrity_enabled", False)
        integrity_applied = record.get("observation_integrity_applied", False)
        integrity_objects = record.get(
            "observation_integrity_inconsistent_objects", []
        )
        integrity_audit = record.get("private_extent_integrity_audit", [])
        is_integrity_schema = record.get("schema_version") == SCHEMA_VERSION
        g17_parent_preserved = record.get(
            "g17_parent_prompt_preserved", not integrity_applied
        )
        projection_extension = release_applied or k2_applied or ledger_applied
        if (
            record.get("schema_version") != state.get("schema_version")
            or record.get("turn") != expected_turn
            or record.get("carrier")
            != "m3_visual_difference_report_exactly_once"
            or record.get("non_text_structure_identical") is not True
            or record.get("non_report_prompt_roundtrip_equal") is not True
            or release_enabled not in {True, False}
            or release_applied not in {True, False}
            or (release_applied and not release_enabled)
            or (release_candidate is not None and not release_enabled)
            or (release_applied and release_outcome != release_candidate)
            or k2_enabled not in {True, False}
            or k2_applied not in {True, False}
            or (k2_applied and (not k2_enabled or not release_applied))
            or ledger_enabled not in {True, False}
            or ledger_mode not in {"full", "phase_bound_sparse"}
            or ledger_applied not in {True, False}
            or (ledger_applied and not ledger_enabled)
            or (ledger_candidate is not None and not ledger_enabled)
            or (ledger_applied and ledger != ledger_candidate)
            or record.get("canonical_v2_1_projection_equal")
            is not (not projection_extension)
            or record.get("release_bound_placement_sha256")
            != _sha256_json(release_outcome)
            or record.get("release_bound_placement_candidate_sha256")
            != _sha256_json(release_candidate)
            or record.get("k2_release_delta_sha256") != _sha256_json(k2_delta)
            or record.get("api_transition_ledger_sha256")
            != _sha256_json(ledger)
            or record.get("api_transition_ledger_candidate_sha256")
            != _sha256_json(ledger_candidate)
            or gain_enabled not in {True, False}
            or gain_status
            not in {
                "disabled",
                "no_bound_release",
                "parent_complete",
                "novel_release_facts",
            }
            or not isinstance(gain_facts, list)
            or gain_extension_applied not in {True, False}
            or gain_extension_applied is not projection_extension
            or parent_preserved not in {True, False}
            or (parent_preserved is projection_extension)
            or record.get("parent_report_sha256") is None
            or record.get("parent_prompt_sha256") is None
            or (
                not is_integrity_schema
                and parent_preserved
                and record.get("parent_report_sha256")
                != record.get("takeover_report_sha256")
            )
            or (
                not is_integrity_schema
                and parent_preserved
                and record.get("parent_prompt_sha256")
                != record.get("takeover_prompt_sha256")
            )
            or integrity_enabled not in {True, False}
            or integrity_applied not in {True, False}
            or not isinstance(integrity_objects, list)
            or any(object_id not in {"obj01", "obj02"} for object_id in integrity_objects)
            or not isinstance(integrity_audit, list)
            or record.get(
                "private_extent_integrity_audit_sha256", _sha256_json([])
            )
            != _sha256_json(integrity_audit)
            or integrity_applied is not (
                integrity_enabled and bool(integrity_objects)
            )
            or (
                is_integrity_schema
                and integrity_enabled
                and (
                    g17_parent_preserved not in {True, False}
                    or g17_parent_preserved is integrity_applied
                    or record.get("g17_parent_report_sha256") is None
                    or record.get("g17_parent_prompt_sha256") is None
                )
            )
            or (
                gain_enabled
                and gain_status == "novel_release_facts"
                and (not gain_facts or not projection_extension)
            )
            or (
                gain_enabled
                and gain_status != "novel_release_facts"
                and (gain_facts or projection_extension)
            )
            or (not gain_enabled and gain_status != "disabled")
            or record.get("inline_api_provenance_only") is not True
            or effect_enabled not in {True, False}
            or effect_mode not in {
                "contradictions",
                "first_failure",
                "verified_prefix",
                "release_support",
                "grasp_loss_only",
                "instance_bound_grasp_loss_only",
                "instance_bound_grasp_outcome_only",
                "instance_bound_transport_failure_only",
                "instance_bound_off_table_lift_regression_only",
                "instance_bound_grasp_loss_plus_off_table_lift_regression",
                "post_release_support_only",
                "compact_outcome",
                "effect_ledger",
            }
            or effect_status not in {"disabled", "silent", "mismatch"}
            or not isinstance(effect_expected, list)
            or not isinstance(effect_contradicted, list)
            or not set(effect_contradicted).issubset(effect_expected)
            or record.get("standalone_api_effect_section_visible") is not effect_visible
            or record.get("canonical_a_star_prompt_unchanged") is effect_visible
            or record.get("v2_1_projection_equal")
            is not (not effect_visible and not projection_extension)
            or (not effect_enabled and effect_status != "disabled")
            or (effect_enabled and effect_status == "disabled")
            or (effect_visible and not effect_contradicted)
            or (
                not effect_visible
                and effect_contradicted
                and effect_mode
                not in {
                    "grasp_loss_only",
                    "instance_bound_grasp_loss_only",
                    "instance_bound_grasp_outcome_only",
                    "instance_bound_transport_failure_only",
                    "instance_bound_off_table_lift_regression_only",
                    "instance_bound_grasp_loss_plus_off_table_lift_regression",
                    "post_release_support_only",
                }
            )
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
        integrity_keys = {
            "api_call_index",
            "object_id",
            "status",
            "observed_signature_sha256",
            "expected_signature_sha256",
        }
        digest = re.compile(r"[0-9a-f]{64}")
        if any(
            not isinstance(audit, dict)
            or set(audit) != integrity_keys
            or not isinstance(audit["api_call_index"], int)
            or isinstance(audit["api_call_index"], bool)
            or audit["api_call_index"] < 0
            or audit["object_id"] not in {"obj01", "obj02"}
            or audit["status"]
            not in {"CONSISTENT", "INCONSISTENT", "UNAVAILABLE"}
            or (
                audit["observed_signature_sha256"] is not None
                and digest.fullmatch(audit["observed_signature_sha256"]) is None
            )
            or digest.fullmatch(audit["expected_signature_sha256"]) is None
            for audit in integrity_audit
        ):
            raise ValueError(f"Extent-integrity audit is malformed: {path}")
        expected_integrity_objects = sorted(
            {
                audit["object_id"]
                for audit in integrity_audit
                if audit["status"] == "INCONSISTENT"
            }
        )
        if integrity_objects != expected_integrity_objects:
            raise ValueError(f"Extent-integrity object projection is invalid: {path}")
        if release_applied:
            release_keys = {
                "subject_id",
                "reference_id",
                "release_call_index",
                "relative_x",
                "relative_y",
                "relative_z",
                "support_region",
                "footprint",
                "reference_orientation",
                "reference_stability",
                "support_at_contact_loss",
                "support_first_established",
                "support_at_endpoint",
            }
            if (
                not isinstance(release_outcome, dict)
                or set(release_outcome) != release_keys
                or release_outcome["subject_id"] not in {"obj01", "obj02"}
                or release_outcome["reference_id"] not in {"obj01", "obj02"}
                or release_outcome["subject_id"] == release_outcome["reference_id"]
                or release_outcome["relative_x"]
                not in {
                    "NEGATIVE_OF_REFERENCE_BAND",
                    "WITHIN_REFERENCE_BAND",
                    "POSITIVE_OF_REFERENCE_BAND",
                }
                or release_outcome["relative_y"]
                not in {
                    "NEGATIVE_OF_REFERENCE_BAND",
                    "WITHIN_REFERENCE_BAND",
                    "POSITIVE_OF_REFERENCE_BAND",
                }
                or release_outcome["relative_z"]
                not in {
                    "NEGATIVE_OF_REFERENCE_BAND",
                    "WITHIN_REFERENCE_BAND",
                    "POSITIVE_OF_REFERENCE_BAND",
                }
                or release_outcome["support_region"]
                not in {"OUTSIDE", "EDGE_REGION", "CENTER_REGION"}
                or release_outcome["footprint"] not in {"OUTSIDE", "INSIDE"}
                or release_outcome["reference_orientation"]
                not in {
                    "LOCAL_Z_ALIGNED_WITH_TABLE_NORMAL",
                    "LOCAL_Z_OBLIQUE_TO_TABLE_NORMAL",
                    "LOCAL_Z_OPPOSED_TO_TABLE_NORMAL",
                }
                or not isinstance(release_outcome["reference_stability"], str)
                or release_outcome["support_at_contact_loss"] not in {True, False}
                or release_outcome["support_at_endpoint"] not in {True, False}
                or release_outcome["support_first_established"]
                not in {
                    "AT_CONTACT_LOSS",
                    "DURING_RELEASE",
                    "AFTER_RELEASE",
                    "NEVER",
                }
            ):
                raise ValueError(f"Release-bound placement audit is invalid: {path}")
        elif release_outcome is not None:
            raise ValueError(f"Silent release-bound placement has payload: {path}")
        if k2_applied:
            if (
                not isinstance(k2_delta, dict)
                or set(k2_delta)
                != {"subject_id", "reference_id", "comparisons"}
                or k2_delta["subject_id"] != release_outcome["subject_id"]
                or k2_delta["reference_id"] != release_outcome["reference_id"]
                or not isinstance(k2_delta["comparisons"], list)
                or len(k2_delta["comparisons"]) != 10
                or any(
                    not isinstance(item, list)
                    or len(item) != 3
                    or not all(isinstance(value, str) for value in item)
                    for item in k2_delta["comparisons"]
                )
            ):
                raise ValueError(f"K2 release delta audit is invalid: {path}")
        elif k2_delta is not None:
            raise ValueError(f"Silent K2 release delta has payload: {path}")
        if ledger_applied:
            if not isinstance(ledger, list) or not ledger:
                raise ValueError(f"API transition ledger is empty: {path}")
            expected_indices = []
            for entry in ledger:
                if (
                    not isinstance(entry, dict)
                    or set(entry)
                    != {
                        "api_name",
                        "api_call_index",
                        "endpoint_changes",
                        "endpoint_unchanged",
                        "transient_sequences",
                    }
                    or not isinstance(entry["api_name"], str)
                    or not isinstance(entry["api_call_index"], int)
                    or isinstance(entry["api_call_index"], bool)
                    or not isinstance(entry["endpoint_changes"], list)
                    or not isinstance(entry["endpoint_unchanged"], list)
                    or not isinstance(entry["transient_sequences"], list)
                ):
                    raise ValueError(f"API transition ledger entry is invalid: {path}")
                expected_indices.append(entry["api_call_index"])
            if expected_indices != sorted(expected_indices):
                raise ValueError(f"API transition ledger order is invalid: {path}")
        elif ledger is not None:
            raise ValueError(f"Silent API transition ledger has payload: {path}")
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
        _validate_visible_temporal_fields(
            path=path,
            audited_event_fields=event_fields,
            visible_event_fields=record.get("temporal_event_fields"),
            ledger_applied=ledger_applied,
        )
        provenance = record.get("public_event_api_provenance")
        call_indices = record.get("public_event_api_call_indices")
        executed_api_calls = record.get("public_executed_api_calls")
        if (
            not isinstance(provenance, list)
            or len(provenance) != len(event_batches)
            or any(value is not None and not isinstance(value, str) for value in provenance)
            or record.get("public_event_api_provenance_sha256")
            != _sha256_json(provenance)
        ):
            raise ValueError(f"Takeover event API provenance is malformed: {path}")
        if call_indices is not None and (
            not isinstance(call_indices, list)
            or len(call_indices) != len(event_batches)
            or any(
                value is not None
                and (not isinstance(value, int) or isinstance(value, bool))
                for value in call_indices
            )
            or record.get("public_event_api_call_indices_sha256")
            != _sha256_json(call_indices)
        ):
            raise ValueError(f"Takeover event call indices are malformed: {path}")
        if not isinstance(executed_api_calls, list) or any(
            not isinstance(value, str) for value in executed_api_calls
        ):
            raise ValueError(f"Takeover executed API sequence is malformed: {path}")
        if any(
            audit["api_call_index"] >= len(executed_api_calls)
            or executed_api_calls[audit["api_call_index"]] != "get_object_pose"
            for audit in integrity_audit
        ):
            raise ValueError(f"Extent-integrity API binding is invalid: {path}")
        if ledger_applied and any(
            entry["api_call_index"] < 0
            or entry["api_call_index"] >= len(executed_api_calls)
            or executed_api_calls[entry["api_call_index"]] != entry["api_name"]
            for entry in ledger
        ):
            raise ValueError(f"API transition ledger binding is invalid: {path}")
        if release_applied:
            release_index = release_outcome["release_call_index"]
            pair_key = (
                f"placement_outcome.{release_outcome['subject_id']}."
                f"{release_outcome['reference_id']}"
            )
            placement_outcomes = record.get("public_interval_placement_outcomes", {})
            at_contact = release_outcome["support_at_contact_loss"]
            established = release_outcome["support_first_established"]
            if (
                not isinstance(release_index, int)
                or isinstance(release_index, bool)
                or release_index < 0
                or release_index >= len(executed_api_calls)
                or re.fullmatch(
                    r"open_gripper(?:_arm\d+|_both)?",
                    executed_api_calls[release_index],
                )
                is None
                or set(placement_outcomes) != {pair_key}
                or at_contact is not (established == "AT_CONTACT_LOSS")
                or (
                    release_outcome["support_region"] == "OUTSIDE"
                    and release_outcome["footprint"] != "OUTSIDE"
                )
                or (
                    release_outcome["support_region"] != "OUTSIDE"
                    and release_outcome["footprint"] != "INSIDE"
                )
            ):
                raise ValueError(f"Release-bound placement binding is invalid: {path}")
        if call_indices is not None and any(
            index is not None
            and (
                index < 0
                or index >= len(executed_api_calls)
                or executed_api_calls[index] != api_name
            )
            for index, api_name in zip(call_indices, provenance, strict=True)
        ):
            raise ValueError(f"Takeover event call-index binding is invalid: {path}")
        if effect_mode in {
            "instance_bound_grasp_loss_only",
            "instance_bound_grasp_outcome_only",
            "instance_bound_transport_failure_only",
            "instance_bound_off_table_lift_regression_only",
            "instance_bound_grasp_loss_plus_off_table_lift_regression",
        }:
            binding_status = record.get(
                "effect_alignment_grasp_episode_binding_status"
            )
            binding_indices = [
                record.get("effect_alignment_grasp_episode_start_call_index"),
                record.get("effect_alignment_grasp_episode_release_call_index"),
                record.get("effect_alignment_grasp_episode_acquisition_call_index"),
                record.get("effect_alignment_grasp_episode_lift_call_index"),
                record.get("effect_alignment_grasp_episode_loss_call_index"),
            ]
            acquisition_endpoint = record.get(
                "effect_alignment_grasp_episode_acquisition_endpoint", {}
            )
            if (
                call_indices is None
                or binding_status
                not in {
                    "not_evaluated",
                    "no_executed_acquisition_window",
                    "ambiguous_unindexed_event",
                    "ambiguous_executor",
                    "bound_no_grasp",
                    "bound",
                }
                or any(
                    value is not None
                    and (
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < 0
                        or value >= len(executed_api_calls)
                    )
                    for value in binding_indices
                )
                or not isinstance(acquisition_endpoint, dict)
                or (
                    effect_visible
                    and effect_mode == "instance_bound_grasp_loss_only"
                    and (
                        binding_status != "bound"
                        or binding_indices[-1] is None
                    )
                )
                or (
                    effect_visible
                    and effect_mode == "instance_bound_grasp_outcome_only"
                    and binding_status not in {"bound", "bound_no_grasp"}
                )
                or (
                    effect_visible
                    and effect_mode == "instance_bound_transport_failure_only"
                    and binding_status != "bound"
                )
                or (
                    effect_visible
                    and effect_mode
                    == "instance_bound_off_table_lift_regression_only"
                    and binding_status != "bound"
                )
                or (
                    effect_visible
                    and effect_mode
                    == "instance_bound_grasp_loss_plus_off_table_lift_regression"
                    and binding_status != "bound"
                )
                or (
                    binding_indices[0] is not None
                    and binding_indices[2] is not None
                    and binding_indices[2] < binding_indices[0]
                )
                or (
                    binding_indices[0] is not None
                    and binding_indices[3] is not None
                    and binding_indices[3] < binding_indices[0]
                )
                or (
                    binding_indices[0] is not None
                    and binding_indices[4] is not None
                    and binding_indices[4] <= binding_indices[0]
                )
                or (
                    binding_indices[1] is not None
                    and binding_indices[3] is not None
                    and binding_indices[3] >= binding_indices[1]
                )
                or (
                    binding_indices[1] is not None
                    and binding_indices[4] is not None
                    and binding_indices[4] >= binding_indices[1]
                )
            ):
                raise ValueError(f"Takeover grasp-episode binding is invalid: {path}")
            if effect_mode == "instance_bound_grasp_outcome_only":
                contradicted_set = set(effect_contradicted)
                endpoint_suffixes = {
                    key.rsplit(".", 1)[-1] for key in acquisition_endpoint
                }
                endpoint_prefixes = {
                    key.rsplit(".", 1)[0] for key in acquisition_endpoint
                }
                endpoint_shape_valid = (
                    len(acquisition_endpoint) == 4
                    and len(endpoint_prefixes) == 1
                    and endpoint_suffixes
                    == {"contact", "relative_x", "relative_y", "relative_z"}
                    and next(iter(endpoint_prefixes), "").startswith(
                        "executor_object."
                    )
                    and acquisition_endpoint.get(
                        next(iter(endpoint_prefixes), "") + ".contact"
                    )
                    in {"FALSE", "TRUE"}
                    and {
                        acquisition_endpoint.get(
                            next(iter(endpoint_prefixes), "") + f".relative_{axis}"
                        )
                        for axis in "xyz"
                    }
                    <= {
                        "NEGATIVE_OF_REFERENCE_BAND",
                        "WITHIN_REFERENCE_BAND",
                        "POSITIVE_OF_REFERENCE_BAND",
                    }
                )
                if (
                    (
                        binding_status == "bound_no_grasp"
                        and "BILATERAL_GRASP_ESTABLISHED" not in contradicted_set
                    )
                    or (
                        binding_status == "bound"
                        and "BILATERAL_GRASP_ESTABLISHED" in contradicted_set
                    )
                    or (
                        binding_indices[3] is not None
                        and "LIFT_ESTABLISHED_BEFORE_RELEASE" in contradicted_set
                    )
                    or (
                        binding_indices[4] is None
                        and "GRASP_RETAINED_DURING_POST_GRASP_MOTION"
                        in contradicted_set
                    )
                    or (
                        "BILATERAL_GRASP_ESTABLISHED" in contradicted_set
                        and not endpoint_shape_valid
                    )
                    or (
                        "BILATERAL_GRASP_ESTABLISHED" not in contradicted_set
                        and acquisition_endpoint
                    )
                ):
                    raise ValueError(
                        f"Takeover grasp-outcome contradiction is invalid: {path}"
                    )
            if effect_mode == "instance_bound_transport_failure_only":
                visible_transport_failures = {
                    "GRASP_RETAINED_DURING_POST_GRASP_MOTION",
                    "LIFT_ESTABLISHED_BEFORE_RELEASE",
                } & set(effect_contradicted)
                if (
                    acquisition_endpoint
                    or (effect_visible and not visible_transport_failures)
                    or (
                        "GRASP_RETAINED_DURING_POST_GRASP_MOTION"
                        in visible_transport_failures
                        and binding_indices[4] is None
                    )
                    or (
                        "LIFT_ESTABLISHED_BEFORE_RELEASE"
                        in visible_transport_failures
                        and binding_indices[3] is not None
                    )
                ):
                    raise ValueError(
                        f"Takeover transport-failure contradiction is invalid: {path}"
                    )
            if effect_mode == "instance_bound_off_table_lift_regression_only":
                regression_effect = "LIFT_CONFIRMED_STATE_RETAINED_AT_ENDPOINT"
                if (
                    acquisition_endpoint
                    or (effect_visible and regression_effect not in effect_contradicted)
                    or (
                        regression_effect in effect_contradicted
                        and binding_indices[4] is not None
                    )
                ):
                    raise ValueError(
                        f"Takeover off-table lift-regression fact is invalid: {path}"
                    )
            if (
                effect_mode
                == "instance_bound_grasp_loss_plus_off_table_lift_regression"
            ):
                loss_effect = "GRASP_RETAINED_DURING_POST_GRASP_MOTION"
                regression_effect = "LIFT_CONFIRMED_STATE_RETAINED_AT_ENDPOINT"
                visible_effects = {loss_effect, regression_effect} & set(
                    effect_contradicted
                )
                if (
                    acquisition_endpoint
                    or (effect_visible and not visible_effects)
                    or (
                        effect_visible
                        and loss_effect in visible_effects
                        and binding_indices[4] is None
                    )
                    or (
                        effect_visible
                        and loss_effect not in visible_effects
                        and regression_effect in visible_effects
                        and binding_indices[4] is not None
                    )
                ):
                    raise ValueError(
                        f"Takeover additive transport fact is invalid: {path}"
                    )
        report = path.with_name("takeover_report.txt").read_text(encoding="utf-8")
        g17_report = _strip_observation_integrity(
            report,
            applied=integrity_applied,
            object_ids=integrity_objects,
            expected_text_sha256=record.get("observation_integrity_text_sha256"),
            path=path,
        )
        if (
            is_integrity_schema
            and integrity_enabled
            and _sha256_text(g17_report)
            != record.get("g17_parent_report_sha256")
        ):
            raise ValueError(f"G17 parent report hash mismatch: {path}")
        if "API-bound effects" in g17_report:
            raise ValueError(f"Takeover contains a standalone API-effect section: {path}")
        if effect_visible:
            separators = [
                f"\n\n{header}\n"
                for header in EFFECT_ALIGNMENT_HEADERS
                if f"\n\n{header}\n" in g17_report
            ]
            if len(separators) != 1 or g17_report.count(separators[0]) != 1:
                raise ValueError(f"Takeover effect appendix is malformed: {path}")
            canonical_report, effect_appendix = g17_report.split(
                separators[0], 1
            )
            if "CONTRADICTED:" not in effect_appendix:
                raise ValueError(f"Takeover effect appendix has no contradiction: {path}")
        else:
            if any(
                f"\n\n{header}\n" in g17_report
                for header in EFFECT_ALIGNMENT_HEADERS
            ):
                raise ValueError(f"Silent takeover contains an effect appendix: {path}")
            canonical_report = g17_report
        if _sha256_text(report) != record.get("takeover_report_sha256"):
            raise ValueError(f"Takeover report hash mismatch: {path}")
        if is_integrity_schema and integrity_enabled:
            if g17_parent_preserved:
                if (
                    record.get("g17_parent_report_sha256")
                    != record.get("takeover_report_sha256")
                    or record.get("g17_parent_prompt_sha256")
                    != record.get("takeover_prompt_sha256")
                ):
                    raise ValueError(f"Silent G17 parent identity mismatch: {path}")
            if parent_preserved and (
                record.get("parent_report_sha256")
                != record.get("g17_parent_report_sha256")
                or record.get("parent_prompt_sha256")
                != record.get("g17_parent_prompt_sha256")
            ):
                raise ValueError(f"G17 projection parent identity mismatch: {path}")
        if _sha256_text(canonical_report) != record.get(
            "canonical_a_star_report_sha256"
        ):
            raise ValueError(f"Canonical A-star report hash mismatch: {path}")
        rendered_provenance = INLINE_API_PROVENANCE.findall(canonical_report)
        if not ledger_applied and rendered_provenance != [
            value for value in provenance if value is not None
        ]:
            raise ValueError(f"Takeover inline API provenance does not match audit: {path}")
        if ledger_applied and (
            rendered_provenance
            or canonical_report.count("Verified public-API transition ledger:") != 1
            or "Verified ordered factual events during the executed code:" in canonical_report
        ):
            raise ValueError(f"API transition ledger carrier is malformed: {path}")
        projected = INLINE_API_PROVENANCE.sub("", canonical_report)
        if (
            not projection_extension
            and _sha256_text(projected) != record.get("v2_1_report_sha256")
        ):
            raise ValueError(f"Takeover is not an exact v2.1 report projection: {path}")
        if release_applied and canonical_report.count(
            "Verified release-bound placement result:"
        ) != 1:
            raise ValueError(f"Release-bound placement carrier is malformed: {path}")
        if k2_applied and canonical_report.count(
            "Verified comparison with the previous bound release:"
        ) != 1:
            raise ValueError(f"K2 release delta carrier is malformed: {path}")
        turn_semantics = {
            "public_transition": record["public_transition"],
            "public_interval_grasp_outcomes": record[
                "public_interval_grasp_outcomes"
            ],
            "public_interval_placement_outcomes": record.get(
                "public_interval_placement_outcomes", {}
            ),
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
        manifest.get("treatment") != "ideal_tk_v4"
        or manifest.get("treatment_label") != TREATMENT_LABEL
        or manifest.get("native_mode") != "M3"
        or manifest.get("initial_action_query_seed_offset") != "1"
    ):
        raise ValueError("Pair manifest did not select audited Ideal V4 canonical delta")
    semantic_reference = (
        json.loads(args.m3_semantic_reference.read_text(encoding="utf-8"))
        if args.m3_semantic_reference is not None
        else {}
    )
    expected_hashes = (
        _expected_semantic_hashes(
            semantic_reference,
            task=args.task,
            global_seed=args.global_seed,
            trials=args.trials,
            a1_run_root=args.a1_run_root,
        )
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
            episode_seed_base=args.episode_seed_base,
            allow_deferred_semantic_replay=(args.m3_semantic_reference is not None),
        )
    )
    treatment_summary_count = _assert_no_trial_chain_failures(
        args.treatment_run_root
    )
    roots = sorted(args.treatment_run_root.glob(f"**/{TRACE_DIR}/trial_*"))
    if len(roots) != args.trials:
        raise ValueError(f"Expected {args.trials} takeover traces, found {len(roots)}")
    audit_path = args.treatment_run_root / "qwen_proxy_audit.jsonl"
    action_hashes = _message_hashes(audit_path)
    audit = _jsonl(audit_path)
    response_frozen = _validate_response_frozen_audit(manifest, audit)
    semantic_hashes: dict[str, dict[str, str]] = {}
    turn_count = 0
    for root in roots:
        trial = str(int(root.name.split("_")[-1]))
        count, hashes = _validate_trace(root, action_message_hashes=action_hashes)
        turn_count += count
        semantic_hashes[trial] = hashes

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
    if not response_frozen:
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
        "schema": "capx_ideal_tk_v4_pair_validation_v4",
        "status": "valid",
        "treatment": TREATMENT_LABEL,
        "native_mode": "M3",
        "takeover_trial_trace_count": len(roots),
        "takeover_turn_count": turn_count,
        "runner_chain_failure_count": 0,
        "treatment_trial_summary_count": treatment_summary_count,
        "m3_deferred_semantic_replay_certificates": certified,
        "m3_prefix_replay_divergences": prefix_divergences,
        **response_frozen,
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
    parser.add_argument("--episode-seed-base", type=int)
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
                    "schema": "capx_ideal_tk_v4_pair_validation_v4",
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
