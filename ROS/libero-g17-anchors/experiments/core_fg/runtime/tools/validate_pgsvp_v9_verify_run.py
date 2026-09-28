#!/usr/bin/env python3
"""Fail-closed validation for A1/AS-materialized-PTGT paired runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from validate_paired_run import validate as validate_pair


PUBLIC_FIELDS = {
    "TASK_RELATION",
    "RUNTIME",
    "ATTACHMENT",
    "RELATIVE_X",
    "RELATIVE_Y",
    "RELATIVE_Z",
    "FOOTPRINT",
    "CONTACT_SUPPORT",
    "STAGE_POSTCONDITIONS",
    "EARLIEST_ABSENT_POSTCONDITION",
    "DOMINANT_ABSENT_FACT",
    "FALSE_BELIEFS",
}
PUBLIC_FIELD_ORDER = (
    "TASK_RELATION",
    "RUNTIME",
    "ATTACHMENT",
    "RELATIVE_X",
    "RELATIVE_Y",
    "RELATIVE_Z",
    "FOOTPRINT",
    "CONTACT_SUPPORT",
    "STAGE_POSTCONDITIONS",
    "EARLIEST_ABSENT_POSTCONDITION",
    "DOMINANT_ABSENT_FACT",
    "FALSE_BELIEFS",
)
PUBLIC_BRIDGE_MARKER = "PGSVP MATERIALIZED STATE CONTRACT"
PUBLIC_HEADER = "GT_BACKED_VERIFIER_FACTS:"
DELTA_HEADER = "PGSVP_PT_GT_CORRECTIONS:"
PT_REQUEST_MARKER = "PGSVP PERCEIVED-TRUTH WORKSHEET"
EXACT_NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])")
FORBIDDEN_PUBLIC = (
    re.compile(
        r"\b(?:reward|task_completed|task\s+(?:completed?|incomplete|failed)|"
        r"task\s+postcondition\s+(?:not\s+)?met|success(?:ful)?(?:\s+label)?|"
        r"trial|seed|run_id)\b",
        re.I,
    ),
    re.compile(r"\b(?:cubeA|cubeB|body_xpos|body_xquat|sim\.data|robosuite_env)\b"),
    re.compile(r"\b(?:goto_pose|open_gripper|close_gripper|sample_grasp_pose)\b"),
    re.compile(
        r"\b(?:should|must|recommend|please|need(?:s)?\s+to|ought\s+to|"
        r"next\s+action|repair\s+plan)\b",
        re.I,
    ),
)
FORBIDDEN_CAUSAL_PUBLIC = re.compile(
    r"\b(?:persistent|systematic|calibration|compensation|causes?|results?\s+in|"
    r"actuator\s+bias|command\s+error)\b",
    re.I,
)
def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _count_text_occurrences(value: Any, needle: str) -> int:
    if isinstance(value, str):
        return value.count(needle)
    if isinstance(value, dict):
        return sum(_count_text_occurrences(item, needle) for item in value.values())
    if isinstance(value, list):
        return sum(_count_text_occurrences(item, needle) for item in value)
    return 0


def _expected_current_semantics(fields: dict[str, str]) -> dict[str, str]:
    attachment = {
        "empty": "target not attached to executor",
        "holding_target": "target attached to executor",
        "holding_other": "executor attached to support goal",
    }[fields["ATTACHMENT"]]
    footprint = {
        "target_center_inside_goal_footprint": (
            "target center inside support footprint"
        ),
        "target_center_outside_goal_footprint": (
            "target center outside support footprint"
        ),
    }[fields["FOOTPRINT"]]
    vertical = {
        "target_below_goal": "target below support height band",
        "same_level": "target within support height band",
        "target_above_goal": "target above support height band",
    }[fields["RELATIVE_Z"]]
    contact = {
        "target_contacting_goal": "target contacting support goal",
        "target_separated_from_goal": "target separated from support goal",
    }[fields["CONTACT_SUPPORT"]]
    stage = (
        f"current relations: {attachment}; {footprint}; {vertical}; {contact}"
    )
    support_geometry = bool(
        fields["ATTACHMENT"] == "empty"
        and fields["FOOTPRINT"] == "target_center_inside_goal_footprint"
        and fields["RELATIVE_Z"] == "target_above_goal"
        and fields["CONTACT_SUPPORT"] == "target_contacting_goal"
    )
    if support_geometry:
        absent = "none in the declared target-support relation"
    elif fields["ATTACHMENT"] == "holding_other":
        absent = (
            "executor is currently attached to the support goal rather than target"
        )
    elif fields["ATTACHMENT"] == "empty":
        absent = "target is not currently attached to the executor"
    elif fields["FOOTPRINT"] == "target_center_outside_goal_footprint":
        absent = "target center is not currently inside the support footprint"
    elif fields["RELATIVE_Z"] != "target_above_goal":
        absent = "target center is not currently above the support height band"
    elif fields["CONTACT_SUPPORT"] == "target_separated_from_goal":
        absent = "target is not currently in contact with the support goal"
    else:
        absent = "target remains attached while contacting the support goal"
    return {
        "STAGE_POSTCONDITIONS": stage,
        "EARLIEST_ABSENT_POSTCONDITION": absent,
        "DOMINANT_ABSENT_FACT": absent,
    }


def _validate_saved_prompt_isolation(
    *,
    run_root: Path,
    treatment: bool,
) -> None:
    prompt_files = sorted(run_root.glob("**/all_responses.json"))
    if not prompt_files:
        raise ValueError(f"No saved prompt artifacts found under {run_root}")
    for path in prompt_files:
        responses = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(responses, list):
            raise ValueError(f"Invalid all_responses artifact: {path}")
        for response in responses:
            if not isinstance(response, dict):
                raise ValueError(f"Invalid response record: {path}")
            initial = response.get("initial_prompt")
            if initial is not None and (
                _count_text_occurrences(initial, PUBLIC_BRIDGE_MARKER)
                or _count_text_occurrences(initial, PUBLIC_HEADER)
                or _count_text_occurrences(initial, DELTA_HEADER)
                or _count_text_occurrences(initial, PT_REQUEST_MARKER)
            ):
                raise ValueError(f"PGSVP leaked into an initial prompt: {path}")
            multiturn = response.get("multi_turn_prompt")
            if multiturn is None:
                continue
            bridge_count = _count_text_occurrences(
                multiturn,
                PUBLIC_BRIDGE_MARKER,
            )
            fact_count = _count_text_occurrences(multiturn, PUBLIC_HEADER)
            delta_count = _count_text_occurrences(multiturn, DELTA_HEADER)
            pt_count = _count_text_occurrences(multiturn, PT_REQUEST_MARKER)
            bridge_expected = 1 if treatment else 0
            if (
                bridge_count != bridge_expected
                or delta_count != 0
                or pt_count != 0
                or fact_count != 0
            ):
                raise ValueError(
                    "Saved regeneration prompt PGSVP occurrence mismatch "
                    f"at {path}: bridge={bridge_count}, delta={delta_count}, "
                    f"pt={pt_count}, facts={fact_count}, "
                    f"bridge_expected={bridge_expected}"
                )


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
            raise ValueError(f"Malformed pair manifest line {line_number}: {path}")
        fields[key.strip()] = value.strip()
    return fields


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
            or _sha256_text(source_validation.read_text(encoding="utf-8"))
            != expected_sha
        ):
            raise ValueError("Frozen-reference-only A1 provenance is invalid")
        if frozen_a0_reference is None or not frozen_a0_reference.is_file():
            raise ValueError("Frozen-reference-only A1 requires its reference")
        reference = json.loads(frozen_a0_reference.read_text(encoding="utf-8"))
        if reference.get("source_pair_validation_sha256") != expected_sha:
            raise ValueError("Frozen A1 source validation is not registered")
        source_validation_record = json.loads(
            source_validation.read_text(encoding="utf-8")
        )
        if (
            source_validation_record.get("status") != "valid"
            or source_validation_record.get("task") != task
            or source_validation_record.get("trials") != trials
        ):
            raise ValueError("Frozen A1 source pair validation is incompatible")
    elif mode not in {
        "replayed_in_current_allocation",
        "live_then_export",
    }:
        raise ValueError(f"Unknown A1 execution mode: {mode}")
    return mode


def validate(args: argparse.Namespace) -> dict[str, Any]:
    bridge_text = args.bridge_file.read_text(encoding="utf-8").strip()
    bridge_sha256 = _sha256_text(bridge_text)
    if (
        bridge_text.count(PUBLIC_BRIDGE_MARKER) != 1
        or PUBLIC_HEADER in bridge_text
        or DELTA_HEADER in bridge_text
    ):
        raise ValueError("Approved PGSVP bridge asset is malformed")
    pair_manifest_path = args.pair_root / "PAIR_MANIFEST.txt"
    if not pair_manifest_path.is_file():
        raise ValueError(f"Missing pair manifest: {pair_manifest_path}")
    pair_manifest = _manifest_fields(pair_manifest_path)
    if (
        pair_manifest.get("verifier_public_bridge_text_sha256")
        != bridge_sha256
    ):
        raise ValueError("Pair manifest bridge text hash mismatch")
    frozen_a0_reference = getattr(args, "frozen_a0_reference", None)
    a0_execution_mode = _validate_a0_execution_provenance(
        pair_manifest=pair_manifest,
        frozen_a0_reference=(
            Path(frozen_a0_reference)
            if frozen_a0_reference is not None
            else None
        ),
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
            frozen_a0_reference=frozen_a0_reference,
            frozen_a0_reference_only=(
                a0_execution_mode == "frozen_reference_only"
            ),
        )
    )
    _validate_saved_prompt_isolation(run_root=args.a1_run_root, treatment=False)
    _validate_saved_prompt_isolation(
        run_root=args.treatment_run_root,
        treatment=True,
    )
    trace_roots = sorted(
        args.treatment_run_root.glob(
            "**/pgsvp_v9_verify_traces/trial_*"
        )
    )
    if len(trace_roots) != args.trials:
        raise ValueError(
            f"Expected {args.trials} PGSVP-v9 trial traces, found {len(trace_roots)}"
        )

    trace_count = 0
    prompt_hashes: set[str] = set()
    schema_hashes: set[str] = set()
    for trial_root in trace_roots:
        initialization_path = trial_root / "initialization.json"
        if not initialization_path.is_file():
            raise ValueError(
                f"PGSVP-v9 trial hook initialization is missing: {trial_root}"
            )
        initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
        initial_state = initialization.get("initial_verified_state")
        if (
            initialization.get("schema_version")
            != "capx-pgsvp-v9-as-materialized-ptgt-v1"
            or initialization.get("task") != args.task
            or initialization.get("stage") != "INITIALIZED_BEFORE_ACTION"
            or initialization.get("public_bridge_injected") is not False
            or not isinstance(initial_state, dict)
            or set(initial_state) != PUBLIC_FIELDS
            or any(initial_state[field] != "unknown" for field in PUBLIC_FIELD_ORDER)
        ):
            raise ValueError(
                f"PGSVP-v9 trial hook initialization is invalid: {trial_root}"
            )
        records = sorted(trial_root.glob("turn_*/verification.json"))
        if not records:
            successful_terminal_artifacts = sorted(
                args.treatment_run_root.glob(
                    f"**/{args.task}/{trial_root.name}_*taskcompleted_1"
                )
            )
            if not successful_terminal_artifacts:
                raise ValueError(
                    "PGSVP-v9 trial ended without a verification turn or a "
                    f"successful initial policy: {trial_root}"
                )
            continue
        expected_prior_state = {
            field: "unknown" for field in PUBLIC_FIELD_ORDER
        }
        for path in records:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                record.get("schema_version")
                != "capx-pgsvp-v9-as-materialized-ptgt-v1"
            ):
                raise ValueError(f"Unexpected PGSVP-v9 schema: {path}")
            if record.get("task") != args.task:
                raise ValueError(f"PGSVP-v9 trace task mismatch: {path}")
            pt = record.get("pt")
            if not isinstance(pt, dict) or set(pt) != PUBLIC_FIELDS:
                raise ValueError(f"PGSVP-v9 PT field set mismatch: {path}")
            if (
                record.get("pt_extraction_context_isolated_from_action_history")
                is not True
                or record.get("pt_owned_by_action_student") is not True
                or record.get("pt_persisted_as_assistant_message") is not False
                or record.get("pt_extraction_request_visible_to_action") is not False
                or record.get("prior_verified_state_visible_to_action") is not False
                or record.get("closed_value_enums_visible_to_action") is not False
                or record.get("raw_pt_visible_to_action") is not False
                or record.get("delta_visible_to_action") is not False
                or record.get("verifier_delta_retained_in_private_audit")
                is not True
                or record.get("action_delivery_is_materialized_state_only")
                is not True
                or record.get("materialized_state_persisted_as_assistant_message")
                is not True
                or record.get("full_gt_report_injected") is not False
                or record.get("materialized_verified_state_injected") is not True
                or record.get("pt_plus_delta_equals_verified_state") is not True
                or record.get("materialized_state_equals_verified_state") is not True
                or record.get("direct_gt_render_equals_materialized_render")
                is not True
                or record.get("pt_request_embedded_action_directive_occurrences") != 0
                or record.get("schema_retry_history_visible_to_action") is not False
            ):
                raise ValueError(f"PGSVP-v9 PT ownership audit failed: {path}")
            if (
                record.get("public_bridge_is_materialized_factual_interface")
                is not True
            ):
                raise ValueError(f"PGSVP-v9 public bridge audit failed: {path}")
            prompt_counts = record.get("student_prompt_occurrences")
            if (
                record.get("public_bridge_text") != bridge_text
                or record.get("public_bridge_sha256") != bridge_sha256
                or record.get("student_prompt_audit_passed") is not True
                or record.get("materialized_state_visible_as_assistant_message")
                is not True
                or record.get(
                    "materialized_state_assistant_message_occurrences"
                )
                != 1
                or record.get("uncorrected_pt_visible_to_action") is not False
                or prompt_counts
                != {
                    "pt_request": 0,
                    "prior_verified_worksheet": 0,
                    "closed_physical_values": 0,
                    "materialized_state": 1,
                    "bridge": 1,
                    "delta": 0,
                    "full_gt_report": 0,
                    "action_instruction": 1,
                }
                or not isinstance(record.get("student_action_prompt_sha256"), str)
                or len(record["student_action_prompt_sha256"]) != 64
            ):
                raise ValueError(f"PGSVP-v9 student prompt injection audit failed: {path}")
            if record.get("deterministic_private_alignment_passed") is not True:
                raise ValueError(f"PGSVP-v9 private GT alignment failed: {path}")
            normalization = record.get("pt_normalization")
            if not isinstance(normalization, dict):
                raise ValueError(f"PGSVP-v9 PT normalization audit missing: {path}")
            interface = record.get("verifier_interface_audit")
            if (
                not isinstance(interface, dict)
                or interface.get("final_renderer") != "fixed_order_local_v1"
            ):
                raise ValueError(f"PGSVP-v9 canonical renderer audit failed: {path}")
            privileged = record.get("privileged_packet")
            if not isinstance(privileged, dict):
                raise ValueError(f"PGSVP-v9 privileged packet missing: {path}")
            parsed = record.get("verifier_parsed")
            public_fields = (
                parsed.get("public_student_guidance", {}).get("fields")
                if isinstance(parsed, dict)
                else None
            )
            if not isinstance(public_fields, dict) or set(public_fields) != PUBLIC_FIELDS:
                raise ValueError(f"PGSVP-v9 canonical public fields missing: {path}")
            if not all(isinstance(public_fields[field], str) for field in PUBLIC_FIELDS):
                raise ValueError(f"PGSVP-v9 canonical public field type failed: {path}")
            if (
                public_fields["FOOTPRINT"]
                == "target_center_inside_goal_footprint"
                and (
                    public_fields["RELATIVE_X"] != "aligned"
                    or public_fields["RELATIVE_Y"] != "aligned"
                )
            ):
                raise ValueError(f"PGSVP-v9 horizontal facts disagree: {path}")
            verified_state = record.get("verified_state")
            prior_state = record.get("prior_verified_state")
            corrections = record.get("fact_corrections")
            if (
                not isinstance(verified_state, dict)
                or set(verified_state) != PUBLIC_FIELDS
                or verified_state != public_fields
                or prior_state != expected_prior_state
                or not isinstance(corrections, dict)
            ):
                raise ValueError(f"PGSVP-v9 persistent state audit failed: {path}")
            reconstructed = dict(pt)
            expected_correction_order = [
                field for field in PUBLIC_FIELD_ORDER if field in corrections
            ]
            if list(corrections) != sorted(corrections):
                # JSON serialization sorts nested keys; order is checked by rendering below.
                raise ValueError(f"PGSVP-v9 correction serialization failed: {path}")
            for field in expected_correction_order:
                correction = corrections[field]
                if (
                    not isinstance(correction, dict)
                    or set(correction) != {"perceived", "verified"}
                    or correction["perceived"] != pt[field]
                    or correction["verified"] != verified_state[field]
                ):
                    raise ValueError(f"PGSVP-v9 correction mismatch: {path}")
                reconstructed[field] = correction["verified"]
            if reconstructed != verified_state:
                raise ValueError(f"PGSVP-v9 PT plus delta does not equal GT: {path}")
            materialized_state = record.get("materialized_state")
            materialized_response = record.get("materialized_state_response")
            expected_materialized_response = json.dumps(
                {
                    field: verified_state[field]
                    for field in PUBLIC_FIELD_ORDER
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
            if (
                materialized_state != verified_state
                or record.get("materialized_state_sha256")
                != hashlib.sha256(
                    json.dumps(
                        verified_state,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                or materialized_response != expected_materialized_response
                or _sha256_text(materialized_response)
                != record.get("materialized_state_response_sha256")
            ):
                raise ValueError(
                    f"PGSVP-v9 materialized state does not equal GT: {path}"
                )
            delta = record.get("delta_message")
            expected_delta = " ".join(
                [DELTA_HEADER]
                + (
                    [
                        f"{field}: {verified_state[field]};"
                        for field in expected_correction_order
                    ]
                    if expected_correction_order
                    else ["NONE;"]
                )
            )
            if (
                delta != expected_delta
                or _sha256_text(delta) != record.get("delta_message_sha256")
                or record.get("fact_correction_count")
                != len(expected_correction_order)
            ):
                raise ValueError(f"PGSVP-v9 canonical delta audit failed: {path}")
            expected_prior_state = verified_state
            for flag in (
                "contains_coordinates_public",
                "contains_reward_or_completion_label_public",
                "contains_action_advice_public",
            ):
                if record.get(flag) is not False:
                    raise ValueError(f"PGSVP-v9 public sanitization failed: {path}")
            public = record.get("public_message")
            if (
                not isinstance(public, str)
                or not public.startswith(PUBLIC_HEADER)
                or _sha256_text(public) != record.get("public_message_sha256")
            ):
                raise ValueError(f"PGSVP-v9 public message hash failed: {path}")
            expected_public = " ".join(
                [PUBLIC_HEADER]
                + [
                    f"{field}: {public_fields[field]};"
                    for field in PUBLIC_FIELD_ORDER
                ]
            )
            if public != expected_public:
                raise ValueError(f"PGSVP-v9 local renderer mismatch: {path}")
            if len(public) > 1700 or EXACT_NUMBER.search(public):
                raise ValueError(f"PGSVP-v9 public numeric/length leak: {path}")
            for pattern in FORBIDDEN_PUBLIC:
                if pattern.search(public):
                    raise ValueError(
                        f"PGSVP-v9 public forbidden pattern {pattern.pattern}: {path}"
                    )
            local_sanitizer = record.get("public_local_sanitizer")
            if (
                not isinstance(local_sanitizer, dict)
                or local_sanitizer.get("version")
                != "pgsvp_v9_public_sanitizer_v2"
                or local_sanitizer.get("passed") is not True
            ):
                raise ValueError(f"PGSVP-v9 local sanitizer audit missing: {path}")
            pt_seed = (record.get("pt_audit") or {}).get("seed")
            verifier_seed = (record.get("verifier_audit") or {}).get("seed")
            if not isinstance(pt_seed, int) or not isinstance(verifier_seed, int):
                raise ValueError(f"PGSVP-v9 request seed missing: {path}")
            if pt_seed == verifier_seed:
                raise ValueError(f"PGSVP-v9 PT and GT seeds collide: {path}")
            prompt_hashes.add(str(record.get("prompt_sha256")))
            schema_hashes.add(str(record.get("schema_sha256")))
            trace_count += 1

    if trace_count:
        if len(prompt_hashes) != 1 or "" in prompt_hashes:
            raise ValueError("PGSVP-v9 verifier prompt hash changed within the run")
        if len(schema_hashes) != 1 or "" in schema_hashes:
            raise ValueError("PGSVP-v9 verifier schema hash changed within the run")

    base.update(
        {
            "schema": "capx_pgsvp_v9_as_materialized_ptgt_pair_validation_v1",
            "treatment": "PGSVP-v9-AS-Materialized-PTGT",
            "pgsvp_v9_trial_trace_count": len(trace_roots),
            "pgsvp_v9_verification_turn_count": trace_count,
            "pgsvp_v9_prompt_sha256": (
                next(iter(prompt_hashes)) if prompt_hashes else None
            ),
            "pgsvp_v9_schema_sha256": (
                next(iter(schema_hashes)) if schema_hashes else None
            ),
            "pt_owned_by_action_student": True,
            "pt_persisted_as_assistant_message": False,
            "raw_pt_visible_to_action": False,
            "delta_visible_to_action": False,
            "verifier_delta_retained_in_private_audit": True,
            "action_delivery_is_materialized_state_only": True,
            "materialized_state_persisted_as_assistant_message": True,
            "pt_plus_delta_equals_verified_state": True,
            "materialized_state_equals_verified_state": True,
            "direct_gt_render_equals_materialized_render": True,
            "public_bridge_is_materialized_factual_interface": True,
            "public_bridge_sha256": bridge_sha256,
            "initial_and_a1_prompts_exclude_public_bridge": True,
            "treated_regeneration_prompts_have_one_materialized_state": True,
            "a0_execution_mode": a0_execution_mode,
            "a1_reexecuted": a0_execution_mode != "frozen_reference_only",
        }
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", required=True, type=Path)
    parser.add_argument("--a1-run-root", required=True, type=Path)
    parser.add_argument("--treatment-run-root", required=True, type=Path)
    parser.add_argument(
        "--task", required=True, choices=("cube_stack", "cube_restack")
    )
    parser.add_argument("--trials", required=True, type=int)
    parser.add_argument("--global-seed", required=True, type=int)
    parser.add_argument("--bridge-file", required=True, type=Path)
    parser.add_argument("--frozen-a0-reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        result = validate(args)
    except Exception as exc:
        result = {
            "schema": "capx_pgsvp_v9_as_materialized_ptgt_pair_validation_v1",
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
