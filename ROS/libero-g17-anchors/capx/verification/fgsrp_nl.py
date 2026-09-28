"""Natural-language seamless factual report replacement (FGSRP-NL).

The private verifier compares a native regeneration report with GT-backed
natural-language observations.  The host never interprets entities or
predicates: it only validates and applies exact, claim-local text edits.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from capx.verification import envelope_replacement as engine
from capx.verification import fgsrp

SCHEMA_VERSION = "capx-fgsrp-nl-v1"
VERIFIER_SCHEMA_VERSION = "capx-fgsrp-nl-verifier-v1"
TRACE_DIR = "fgsrp_nl_traces"
STATE_FILE = "fgsrp_nl_state.json"
VERIFIER_SEED_OFFSET = 810_000

_VERDICTS = {"KEEP", "REPLACE", "UNRESOLVED"}
_FORBIDDEN_REPLACEMENT = (
    r"\b(?:reward|success label|task (?:is )?(?:complete|completed)|oracle)\b",
    r"\b(?:should|must|need to|next step|try again|retry)\b",
    r"```|`[^`]+`|\b(?:execute|call|invoke)\b",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _natural_language_evidence(
    evidence: dict[str, dict[str, Any]],
) -> list[str]:
    """Expose only rendered observations, never ontology or simulator fields."""
    statements: list[str] = []
    for evidence_id in sorted(evidence):
        if not evidence_id.startswith(("before.", "after.")):
            continue
        item = evidence[evidence_id]
        statement = item.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"Missing natural-language evidence: {evidence_id}")
        statements.append(statement.strip())
    if len(statements) != len(set(statements)):
        raise ValueError("Natural-language GT evidence is not uniquely rendered")
    return statements


def _messages(
    *,
    prompt: str,
    schema: dict[str, Any],
    task_description: str,
    prepared: fgsrp.PreparedCarrier,
    statements: list[str],
) -> list[dict[str, Any]]:
    packet = {
        "task_description": task_description.split("\nAPIs:", 1)[0].strip(),
        "native_regeneration_report_chunks": [
            {
                "chunk_id": item.chunk_id,
                "text": item.text,
                "sha256": item.sha256,
            }
            for item in prepared.candidates
        ],
        "authoritative_gt_natural_language_observations": statements,
        "required_response_schema": schema,
    }
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                "Privately verify every factual claim in the native report. "
                "Return strict JSON only.\n\n"
                + json.dumps(packet, ensure_ascii=True, sort_keys=True)
            ),
        },
    ]


def _contains_number(text: str) -> bool:
    return re.search(r"(?<![A-Za-z])[-+]?\d", text) is not None


def _surface_rejection(original: str, replacement: str) -> str | None:
    if "\n" in original or "\r" in original:
        return "multiline_original_span"
    if "\n" in replacement or "\r" in replacement:
        return "multiline_replacement"
    if _contains_number(original) or _contains_number(replacement):
        return "numeric_content_is_immutable"
    if len(replacement) > max(300, len(original) * 2):
        return "replacement_not_local"
    if any(re.search(pattern, replacement, re.IGNORECASE) for pattern in _FORBIDDEN_REPLACEMENT):
        return "replacement_contains_advice_code_or_privileged_label"
    return None


def _normalize_response(
    content: str,
    *,
    prepared: fgsrp.PreparedCarrier,
    statements: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Accept valid decisions independently; semantic truth remains LLM-owned."""
    value = engine._extract_json_object(content)
    if value.get("schema_version") != VERIFIER_SCHEMA_VERSION:
        raise ValueError("FGSRP-NL verifier schema version mismatch")
    decisions = value.get("decisions")
    audit = value.get("audit")
    if not isinstance(decisions, list) or not isinstance(audit, dict):
        raise ValueError("FGSRP-NL response lacks decisions or audit")
    if audit.get("all_candidate_chunks_checked") is not True:
        raise ValueError("Verifier did not certify complete report inspection")
    if audit.get("contains_action_advice") is not False:
        raise ValueError("Verifier reported action advice")

    candidate_map = {item.chunk_id: item for item in prepared.candidates}
    statement_set = set(statements)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    intervals: dict[str, list[tuple[int, int]]] = {}

    for index, raw in enumerate(decisions):
        reason: str | None = None
        if not isinstance(raw, dict):
            rejected.append({"decision_index": index, "reason": "not_an_object"})
            continue
        chunk_id = raw.get("chunk_id")
        original = raw.get("original_span")
        verdict = raw.get("verdict")
        replacement = raw.get("replacement")
        quotes = raw.get("evidence_quotes")
        candidate = candidate_map.get(chunk_id) if isinstance(chunk_id, str) else None
        if candidate is None:
            reason = "unknown_chunk"
        elif not isinstance(original, str) or not original.strip():
            reason = "invalid_original_span"
        elif verdict not in _VERDICTS:
            reason = "invalid_verdict"
        elif not isinstance(quotes, list) or not all(isinstance(q, str) for q in quotes):
            reason = "invalid_evidence_quotes"
        elif candidate.text.count(original) != 1:
            reason = "original_span_not_unique_exact_text"
        elif verdict == "REPLACE" and (chunk_id, original) in seen:
            reason = "duplicate_decision"
        elif verdict != "REPLACE":
            if replacement is not None:
                reason = "nonreplacement_decision_has_replacement"
            elif quotes:
                reason = "nonreplacement_decision_has_evidence_quotes"
        elif not isinstance(replacement, str) or not replacement.strip():
            reason = "missing_replacement"
        elif replacement == original:
            reason = "replacement_is_identical"
        elif not quotes or any(quote not in statement_set for quote in quotes):
            reason = "replacement_lacks_exact_gt_evidence_quote"
        else:
            reason = _surface_rejection(original, replacement)

        if (
            reason is None
            and verdict == "REPLACE"
            and candidate is not None
            and isinstance(original, str)
        ):
            start = candidate.text.index(original)
            end = start + len(original)
            if any(start < prior_end and prior_start < end for prior_start, prior_end in intervals.setdefault(chunk_id, [])):
                reason = "overlapping_decision"
            else:
                intervals[chunk_id].append((start, end))

        if reason is not None:
            rejected.append(
                {
                    "decision_index": index,
                    "reason": reason,
                    "decision": copy.deepcopy(raw),
                }
            )
            continue

        if verdict == "REPLACE":
            seen.add((chunk_id, original))
            accepted.append(
                {
                    "chunk_id": chunk_id,
                    "original_claim": original,
                    "replacement_claim": replacement,
                    "replacement_fact_ids": [],
                    "evidence_quotes": list(quotes),
                }
            )

    return accepted, {
        "status": "accepted_claim_local",
        "decision_count": len(decisions),
        "accepted_replacement_count": len(accepted),
        "rejected_decisions": rejected,
        "verifier_audit": copy.deepcopy(audit),
    }


def _query_repairs(
    args: Any,
    *,
    task_description: str,
    prepared: fgsrp.PreparedCarrier,
    statements: list[str],
    trial: int,
    turn: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_path = Path(
        os.environ.get(
            "CAPX_FGSRP_NL_PROMPT_PATH",
            Path(__file__).with_name("fgsrp_nl_prompt.md"),
        )
    )
    schema_path = Path(
        os.environ.get(
            "CAPX_FGSRP_NL_SCHEMA_PATH",
            Path(__file__).with_name("fgsrp_nl_schema.json"),
        )
    )
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = _load_json(schema_path)
    messages = _messages(
        prompt=prompt,
        schema=schema,
        task_description=task_description,
        prepared=prepared,
        statements=statements,
    )
    attempts = int(os.environ.get("CAPX_FGSRP_NL_SCHEMA_ATTEMPTS", "2"))
    seed_base = int(os.environ.get("CAPX_GLOBAL_SEED", "0")) + trial * 1000 + turn
    errors: list[dict[str, Any]] = []
    content: str | None = None
    request_audit: dict[str, Any] = {}
    for attempt in range(attempts):
        request_messages = copy.deepcopy(messages)
        if errors:
            request_messages.extend(
                [
                    {"role": "assistant", "content": content or ""},
                    {
                        "role": "user",
                        "content": (
                            "The prior envelope was invalid: "
                            f"{errors[-1]['error']}. Return corrected strict JSON only."
                        ),
                    },
                ]
            )
        try:
            content, request_audit = engine._request(
                args,
                request_messages,
                seed=seed_base + VERIFIER_SEED_OFFSET + attempt,
                max_tokens=3000,
            )
            repairs, normalization = _normalize_response(
                content,
                prepared=prepared,
                statements=statements,
            )
        except Exception as exc:
            errors.append(
                {
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "request": request_audit,
                    "response_sha256": (
                        engine._sha256_text(content) if content is not None else None
                    ),
                }
            )
            if attempt + 1 < attempts:
                continue
            return [], {
                "status": "abstained_invalid_envelope",
                "request": request_audit,
                "response": content,
                "errors": errors,
            }
        return repairs, {
            "status": "completed",
            "architecture": "private_llm_natural_language_claim_local_replacement",
            "host_semantic_judgment": False,
            "host_alias_matching": False,
            "host_predicate_matching": False,
            "request": {**request_audit, "schema_attempt": attempt},
            "response": content,
            "prior_errors": errors,
            "normalization": normalization,
        }
    raise AssertionError("unreachable")


def initialize_trial_trace(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    task: str,
) -> None:
    root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    root.mkdir(parents=True, exist_ok=False)
    provider = fgsrp._provider_for(task)
    initial = provider.capture(env, task=task)
    state = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "trial": trial,
        "provider": provider.name,
        "snapshots": [
            {
                "phase": "initial_before_first_action",
                "turn": 0,
                "evidence": initial,
                "sha256": engine._sha256_json(initial),
            }
        ],
        "m2_blocks": [],
        "m3_reports": [],
    }
    _save_json(root / STATE_FILE, state)
    _save_json(
        root / "initialization.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "trial": trial,
            "provider": provider.name,
            "action_visible_gt_packet_injected": False,
            "action_visible_bridge_injected": False,
        },
    )


def run_fgsrp_nl(
    env: Any,
    args: Any,
    *,
    output_dir: str,
    trial: int,
    turn: int,
    task: str,
    task_description: str,
    multi_turn_prompt: str,
    executed_code_blocks: list[str],
    executed_code: str,
    console_stdout: str,
    console_stderr: str,
    visual_differencing_feedback: str | None,
    native_decision_prompt: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace only verifier-contradicted exact spans in the native report."""
    trial_root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    turn_root = trial_root / f"turn_{turn:02d}"
    turn_root.mkdir(parents=True, exist_ok=False)
    state = _load_json(trial_root / STATE_FILE)
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("task") != task
        or state.get("trial") != trial
    ):
        raise ValueError("FGSRP-NL trial-state provenance mismatch")
    snapshots = state.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != turn:
        raise ValueError("FGSRP-NL evidence history is not turn-synchronous")

    provider = fgsrp._provider_for(task)
    before = snapshots[-1]["evidence"]
    after = provider.capture(env, task=task)
    evidence = fgsrp._temporal_evidence(before, after)
    statements = _natural_language_evidence(evidence)
    carrier_name = os.environ.get("CAPX_FGSRP_CARRIER", "m2_comments")
    adapter = fgsrp._carrier(carrier_name)
    context = {
        "turn": turn,
        "task": task,
        "task_description": task_description,
        "multi_turn_prompt": multi_turn_prompt,
        "executed_code_blocks": executed_code_blocks,
        "executed_code": executed_code,
        "console_stdout": console_stdout,
        "console_stderr": console_stderr,
        "visual_differencing_feedback": visual_differencing_feedback,
        "native_decision_prompt": native_decision_prompt,
    }
    prepared = adapter.prepare(context, state)
    repairs, protocol = _query_repairs(
        args,
        task_description=task_description,
        prepared=prepared,
        statements=statements,
        trial=trial,
        turn=turn,
    )
    applied = adapter.apply(
        prepared,
        repairs=repairs,
        evidence=evidence,
        context=context,
        state=state,
    )

    native_hash = engine._sha256_json(native_decision_prompt)
    corrected_hash = engine._sha256_json(applied.corrected_prompt)
    if applied.active_repair_count == 0 and corrected_hash != native_hash:
        raise RuntimeError("FGSRP-NL no-op changed the native action request")
    if applied.active_repair_count > 0 and corrected_hash == native_hash:
        raise RuntimeError("FGSRP-NL accepted repair did not change the request")
    if engine._non_text_structure(native_decision_prompt) != engine._non_text_structure(applied.corrected_prompt):
        raise RuntimeError("FGSRP-NL changed action-request roles or media")

    snapshots.append(
        {
            "phase": "after_executed_block",
            "turn": turn,
            "evidence": after,
            "sha256": engine._sha256_json(after),
        }
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "trial": trial,
        "turn": turn,
        "carrier": prepared.kind,
        "provider": provider.name,
        "native_prompt_sha256": native_hash,
        "corrected_prompt_sha256": corrected_hash,
        "native_prompt_structure": engine._prompt_structure(native_decision_prompt),
        "corrected_prompt_structure": engine._prompt_structure(applied.corrected_prompt),
        "non_text_structure_identical": True,
        "native_surface_sha256": engine._sha256_text(prepared.source_text),
        "corrected_surface_sha256": engine._sha256_text(applied.corrected_source),
        "candidate_chunks": [asdict(item) for item in prepared.candidates],
        "candidate_count": len(prepared.candidates),
        "before_evidence_sha256": engine._sha256_json(before),
        "after_evidence_sha256": engine._sha256_json(after),
        "temporal_evidence_sha256": engine._sha256_json(evidence),
        "private_temporal_evidence": evidence,
        "private_gt_natural_language_observations": statements,
        "repairs": repairs,
        "repair_count": len(repairs),
        "active_repair_count": applied.active_repair_count,
        "applied_patches": applied.patches,
        "carrier_state": applied.state_record,
        "verification_protocol": protocol,
        "no_op_prompt_byte_equivalent": corrected_hash == native_hash if not repairs else None,
        "action_visible_gt_packet_injected": False,
        "action_visible_bridge_injected": False,
        "action_visible_verifier_protocol_injected": False,
        "raw_image_replaced": False,
        "console_stdout_replaced": False,
        "console_stderr_replaced": False,
        "task_or_api_text_replaced": False,
    }
    _save_json(turn_root / "replacement.json", record)
    _save_json(trial_root / STATE_FILE, state)
    return applied.corrected_prompt
