"""Surgical correction of explicit false facts in M2 regeneration envelopes.

The action model receives the ordinary A1 prompt with selected model-authored
text spans replaced in place. The private verifier may identify contradictions,
but only the host can render canonical GT facts and modify the envelope.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import time
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from capx.verification.core_fg import extract_core_facts


SCHEMA_VERSION = "capx-envelope-replacement-v1"
VERIFIER_SCHEMA_VERSION = "capx-envelope-repair-verifier-v1"
ADJUDICATOR_SCHEMA_VERSION = "capx-envelope-contradiction-gate-v1"
TRACE_DIR = "envelope_replacement_traces"
HISTORY_FILE = "verified_code_history.json"
VERIFIER_SEED_OFFSET = 730_000
ADJUDICATOR_SEED_OFFSET = 740_000
FORBIDDEN_LOCAL_REPLACEMENT_TEXT = (
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


@dataclass(frozen=True)
class CandidateChunk:
    chunk_id: str
    source: str
    start: int
    end: int
    text: str
    sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _absolute_offset(offsets: list[int], position: tuple[int, int]) -> int:
    line, column = position
    if line < 1 or line >= len(offsets):
        raise ValueError(f"Token position outside source: {position}")
    return offsets[line - 1] + column


def _comment_chunks(
    executed_code: str,
    *,
    chunk_prefix: str = "executed_code.comment",
) -> list[CandidateChunk]:
    """Extract exact Python comment tokens without changing executable tokens."""
    offsets = _line_offsets(executed_code)
    chunks: list[CandidateChunk] = []
    stream = tokenize.generate_tokens(io.StringIO(executed_code).readline)
    try:
        for token in stream:
            if token.type != tokenize.COMMENT:
                continue
            start = _absolute_offset(offsets, token.start)
            end = _absolute_offset(offsets, token.end)
            text = executed_code[start:end]
            chunks.append(
                CandidateChunk(
                    chunk_id=f"{chunk_prefix}.{len(chunks):04d}",
                    source="executed_code_comment",
                    start=start,
                    end=end,
                    text=text,
                    sha256=_sha256_text(text),
                )
            )
    except (IndentationError, tokenize.TokenError):
        # Tokenization can fail after yielding valid comments from incomplete code.
        # The yielded exact spans remain safe candidates.
        pass
    return chunks


def extract_candidate_chunks(
    executed_code: str,
    console_stdout: str,
) -> list[CandidateChunk]:
    del console_stdout
    chunks = _comment_chunks(executed_code)
    identifiers = [chunk.chunk_id for chunk in chunks]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Envelope candidate chunk identifiers are not unique")
    return chunks


def _latest_block_candidates(
    code_block: str,
    *,
    block_index: int,
) -> list[CandidateChunk]:
    return _comment_chunks(
        code_block,
        chunk_prefix=f"code_block_{block_index:02d}.comment",
    )


def _fact_catalog(env: Any) -> dict[str, dict[str, str]]:
    facts = asdict(extract_core_facts(env))
    renderers = {
        "target_grasp": {
            "target_grasped_by_robot": (
                "the red cube is currently attached to the gripper"
            ),
            "target_not_grasped_by_robot": (
                "the red cube is currently not attached to the gripper"
            ),
        },
        "relative_x": {
            "target_center_negative_x_of_goal_band": (
                "the red cube center is currently on the negative-x side of "
                "the green cube's x band"
            ),
            "target_center_within_goal_x_band": (
                "the red cube center is currently within the green cube's x band"
            ),
            "target_center_positive_x_of_goal_band": (
                "the red cube center is currently on the positive-x side of "
                "the green cube's x band"
            ),
        },
        "relative_y": {
            "target_center_negative_y_of_goal_band": (
                "the red cube center is currently on the negative-y side of "
                "the green cube's y band"
            ),
            "target_center_within_goal_y_band": (
                "the red cube center is currently within the green cube's y band"
            ),
            "target_center_positive_y_of_goal_band": (
                "the red cube center is currently on the positive-y side of "
                "the green cube's y band"
            ),
        },
        "relative_z": {
            "target_below_goal_height_band": (
                "the red cube center is currently below the green cube's height band"
            ),
            "target_within_goal_height_band": (
                "the red cube center is currently within the green cube's height band"
            ),
            "target_above_goal_height_band": (
                "the red cube center is currently above the green cube's height band"
            ),
        },
        "footprint": {
            "target_center_inside_goal_footprint": (
                "the red cube center is currently inside the green cube's footprint"
            ),
            "target_center_outside_goal_footprint": (
                "the red cube center is currently outside the green cube's footprint"
            ),
        },
        "target_goal_contact": {
            "target_contacting_goal": (
                "the red cube is currently in contact with the green cube"
            ),
            "target_separated_from_goal": (
                "the red cube is currently separated from the green cube"
            ),
        },
        "target_table_contact": {
            "target_contacting_table": (
                "the red cube is currently in contact with the table"
            ),
            "target_separated_from_table": (
                "the red cube is currently separated from the table"
            ),
        },
    }
    catalog: dict[str, dict[str, str]] = {}
    for predicate, value in facts.items():
        try:
            statement = renderers[predicate][value]
        except KeyError as exc:
            raise ValueError(
                f"No canonical envelope rendering for {predicate}={value}"
            ) from exc
        catalog[predicate] = {
            "predicate": predicate,
            "value": value,
            "statement": statement,
        }
    return catalog


def _extract_json_object(content: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            value, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Verifier response contains no JSON object")


def _request(
    args: Any,
    messages: list[dict[str, Any]],
    *,
    seed: int,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    headers = {"Content-Type": "application/json"}
    if getattr(args, "api_key", None):
        headers["Authorization"] = f"Bearer {args.api_key}"
    timeout = float(os.environ.get("CAPX_MODEL_QUERY_TIMEOUT_SECONDS", "200"))
    attempts = int(
        os.environ.get("CAPX_ENVELOPE_REPLACEMENT_TRANSPORT_ATTEMPTS", "2")
    )
    if attempts < 1:
        raise ValueError("Envelope verifier transport attempts must be positive")
    transport: list[dict[str, Any]] = []
    for attempt in range(attempts):
        try:
            response = requests.post(
                args.server_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
        except requests.RequestException as exc:
            transport.append(
                {"attempt": attempt, "error": type(exc).__name__}
            )
            if attempt + 1 == attempts:
                raise
            time.sleep(1)
            continue
        retriable = response.status_code in {408, 429} or response.status_code >= 500
        transport.append(
            {
                "attempt": attempt,
                "status_code": response.status_code,
                "retriable": retriable,
            }
        )
        if response.ok:
            body = response.json()
            try:
                content = str(body["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("Unexpected envelope verifier response") from exc
            return content, {
                "seed": seed,
                "request_sha256": _sha256_json(payload),
                "response_sha256": _sha256_text(content),
                "transport": transport,
            }
        if not retriable or attempt + 1 == attempts:
            response.raise_for_status()
        time.sleep(1)
    raise RuntimeError("Envelope verifier request exhausted")


def _verifier_messages(
    *,
    prompt_text: str,
    schema: dict[str, Any],
    task_description: str,
    candidates: list[CandidateChunk],
    fact_catalog: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    packet = {
        "task_description": task_description.split("\nAPIs:", 1)[0].strip(),
        "candidate_chunks": [asdict(candidate) for candidate in candidates],
        "authoritative_post_block_fact_catalog": fact_catalog,
        "required_response_schema": schema,
    }
    return [
        {"role": "system", "content": prompt_text},
        {
            "role": "user",
            "content": (
                "Open this regeneration envelope and identify every explicit "
                "physical-state claim in the latest executed block contradicted "
                "by the authoritative post-block catalog. Return strict JSON "
                "only.\n\n"
                + json.dumps(packet, ensure_ascii=True, sort_keys=True)
            ),
        },
    ]


def _adjudicator_messages(
    *,
    prompt_text: str,
    schema: dict[str, Any],
    candidates: list[CandidateChunk],
    fact_catalog: dict[str, dict[str, str]],
    proposed_repairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_map = {candidate.chunk_id: candidate for candidate in candidates}
    proposals = []
    for repair_index, repair in enumerate(proposed_repairs):
        candidate = candidate_map[repair["chunk_id"]]
        proposals.append(
            {
                "repair_index": repair_index,
                "chunk_id": repair["chunk_id"],
                "candidate_text": candidate.text,
                "original_claim": repair["original_claim"],
                "replacement_claim": repair["replacement_claim"],
                "selected_authoritative_facts": [
                    fact_catalog[fact_id]
                    for fact_id in repair["replacement_fact_ids"]
                ],
            }
        )
    packet = {
        "proposed_repairs": proposals,
        "complete_authoritative_post_block_fact_catalog": fact_catalog,
        "required_response_schema": schema,
    }
    return [
        {"role": "system", "content": prompt_text},
        {
            "role": "user",
            "content": (
                "Adjudicate whether each proposed edit corrects a logically "
                "contradicted claim in the native regeneration envelope. "
                "Return strict JSON only.\n\n"
                + json.dumps(packet, ensure_ascii=True, sort_keys=True)
            ),
        },
    ]


def _normalize_adjudications(
    content: str,
    *,
    proposed_repairs: list[dict[str, Any]],
    expected_schema_version: str = ADJUDICATOR_SCHEMA_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value = _extract_json_object(content)
    if value.get("schema_version") != expected_schema_version:
        raise ValueError("Envelope contradiction-gate schema version mismatch")
    raw_decisions = value.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("Envelope contradiction-gate decisions must be a list")
    if len(raw_decisions) != len(proposed_repairs):
        raise ValueError("Contradiction gate did not adjudicate every proposal")

    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    allowed_relations = {
        "contradicted",
        "supported_or_equivalent",
        "compatible",
        "not_decidable",
    }
    for decision in raw_decisions:
        if not isinstance(decision, dict):
            raise ValueError("Contradiction-gate decision must be an object")
        repair_index = decision.get("repair_index")
        if (
            not isinstance(repair_index, int)
            or isinstance(repair_index, bool)
            or repair_index < 0
            or repair_index >= len(proposed_repairs)
            or repair_index in seen
        ):
            raise ValueError("Invalid contradiction-gate repair index")
        seen.add(repair_index)
        repair = proposed_repairs[repair_index]
        if decision.get("chunk_id") != repair["chunk_id"]:
            raise ValueError("Contradiction-gate chunk provenance mismatch")
        if decision.get("original_claim") != repair["original_claim"]:
            raise ValueError("Contradiction-gate claim provenance mismatch")
        if decision.get("replacement_claim") != repair["replacement_claim"]:
            raise ValueError("Contradiction-gate replacement provenance mismatch")
        relation = decision.get("relation")
        if relation not in allowed_relations:
            raise ValueError("Invalid contradiction-gate semantic relation")
        span_is_minimal = decision.get("span_is_smallest_complete_clause")
        selected_facts_are_sufficient = decision.get(
            "selected_facts_entail_claim_negation"
        )
        replacement_is_entailed = decision.get(
            "replacement_claim_entailed_by_selected_facts"
        )
        other_meaning_is_preserved = decision.get(
            "replacement_preserves_all_other_meaning"
        )
        if not all(
            isinstance(certification, bool)
            for certification in (
                span_is_minimal,
                selected_facts_are_sufficient,
                replacement_is_entailed,
                other_meaning_is_preserved,
            )
        ):
            raise ValueError("Contradiction-gate certifications must be boolean")
        normalized.append(
            {
                "repair_index": repair_index,
                "chunk_id": repair["chunk_id"],
                "original_claim": repair["original_claim"],
                "replacement_claim": repair["replacement_claim"],
                "relation": relation,
                "span_is_smallest_complete_clause": span_is_minimal,
                "selected_facts_entail_claim_negation": (
                    selected_facts_are_sufficient
                ),
                "replacement_claim_entailed_by_selected_facts": (
                    replacement_is_entailed
                ),
                "replacement_preserves_all_other_meaning": (
                    other_meaning_is_preserved
                ),
                "accepted": (
                    relation == "contradicted"
                    and span_is_minimal
                    and selected_facts_are_sufficient
                    and replacement_is_entailed
                    and other_meaning_is_preserved
                ),
            }
        )
    normalized.sort(key=lambda item: item["repair_index"])

    audit = value.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("Contradiction-gate audit must be an object")
    if audit.get("all_proposals_checked") is not True:
        raise ValueError("Contradiction gate did not certify complete coverage")
    if audit.get("used_only_supplied_text_and_catalog") is not True:
        raise ValueError("Contradiction gate used evidence outside the envelope")
    return normalized, copy.deepcopy(audit)


def _normalize_repairs(
    content: str,
    *,
    candidates: list[CandidateChunk],
    fact_catalog: dict[str, dict[str, str]],
    expected_schema_version: str = VERIFIER_SCHEMA_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value = _extract_json_object(content)
    response_schema = value.get("schema_version")
    if response_schema not in {None, expected_schema_version}:
        raise ValueError("Envelope verifier schema version mismatch")
    raw_repairs = value.get("repairs")
    if not isinstance(raw_repairs, list):
        raise ValueError("Envelope verifier repairs must be a list")
    candidate_map = {candidate.chunk_id: candidate for candidate in candidates}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for repair in raw_repairs:
        if not isinstance(repair, dict):
            raise ValueError("Envelope repair entry must be an object")
        chunk_id = repair.get("chunk_id")
        original_claim = repair.get("original_claim")
        replacement_claim = repair.get("replacement_claim")
        fact_ids = repair.get("replacement_fact_ids")
        if not isinstance(chunk_id, str) or chunk_id not in candidate_map:
            raise ValueError(f"Unknown envelope chunk id: {chunk_id!r}")
        candidate = candidate_map[chunk_id]
        if (
            not isinstance(original_claim, str)
            or not original_claim.strip()
            or "\n" in original_claim
            or candidate.text.count(original_claim) != 1
        ):
            raise ValueError(f"Invalid exact claim span: {chunk_id}")
        if re.search(r"\d", original_claim):
            raise ValueError(
                f"Envelope repair cannot alter numeric data: {chunk_id}"
            )
        if (
            not isinstance(replacement_claim, str)
            or not replacement_claim.strip()
            or "\n" in replacement_claim
            or replacement_claim == original_claim
            or "#" in replacement_claim
            or re.search(r"\d", replacement_claim)
            or len(replacement_claim) > max(160, len(original_claim) * 2)
            or any(
                term in replacement_claim.casefold()
                for term in FORBIDDEN_LOCAL_REPLACEMENT_TEXT
            )
        ):
            raise ValueError(f"Invalid local replacement claim: {chunk_id}")
        if (
            candidate.source == "executed_code_comment"
            and "#" in original_claim
        ):
            raise ValueError(
                f"Code-comment claim span must exclude the comment marker: {chunk_id}"
            )
        repair_key = f"{chunk_id}:{candidate.text.index(original_claim)}:{len(original_claim)}"
        if repair_key in seen:
            raise ValueError(f"Duplicate envelope repair: {repair_key}")
        seen.add(repair_key)
        if (
            not isinstance(fact_ids, list)
            or not fact_ids
            or len(fact_ids) > 4
            or not all(isinstance(fact_id, str) for fact_id in fact_ids)
        ):
            raise ValueError(f"Invalid replacement fact ids: {chunk_id}")
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError(f"Duplicate replacement fact ids: {chunk_id}")
        unknown = sorted(set(fact_ids) - set(fact_catalog))
        if unknown:
            raise ValueError(
                f"Repair references unknown GT predicates {unknown}: {chunk_id}"
            )
        normalized.append(
            {
                "chunk_id": chunk_id,
                "original_sha256": candidate.sha256,
                "original_claim": original_claim,
                "original_claim_sha256": _sha256_text(original_claim),
                "replacement_claim": replacement_claim,
                "replacement_claim_sha256": _sha256_text(replacement_claim),
                "replacement_fact_ids": fact_ids,
            }
        )
    audit = value.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("Envelope verifier audit must be an object")
    if audit.get("all_candidate_chunks_checked") is not True:
        raise ValueError("Verifier did not certify complete candidate coverage")
    if audit.get("used_only_authoritative_catalog") is not True:
        raise ValueError("Verifier did not certify authoritative fact use")
    if audit.get("contains_action_advice") is not False:
        raise ValueError("Verifier reported action advice")
    audit = copy.deepcopy(audit)
    audit["response_schema_version"] = response_schema
    audit["schema_version_host_normalized"] = response_schema is None
    return normalized, audit


def _apply_source_repairs(
    source: str,
    *,
    candidates: list[CandidateChunk],
    repairs: list[dict[str, Any]],
    fact_catalog: dict[str, dict[str, str]],
    block_index: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    del fact_catalog, block_index
    candidate_map = {candidate.chunk_id: candidate for candidate in candidates}
    patches: list[
        tuple[int, int, str, CandidateChunk, str, list[str]]
    ] = []
    for repair in repairs:
        candidate = candidate_map[repair["chunk_id"]]
        fact_ids = repair["replacement_fact_ids"]
        original_claim = repair["original_claim"]
        replacement = repair["replacement_claim"]
        local_start = candidate.text.index(original_claim)
        start = candidate.start + local_start
        end = start + len(original_claim)
        patches.append(
            (
                start,
                end,
                replacement,
                candidate,
                original_claim,
                fact_ids,
            )
        )
    patches.sort(key=lambda item: item[0], reverse=True)
    updated = source
    audit: list[dict[str, Any]] = []
    last_start = len(source) + 1
    for (
        start,
        end,
        replacement,
        candidate,
        original_claim,
        fact_ids,
    ) in patches:
        if end > last_start:
            raise ValueError("Envelope replacement spans overlap")
        if updated[start:end] != original_claim:
            raise ValueError(f"Envelope source span drifted: {candidate.chunk_id}")
        updated = updated[:start] + replacement + updated[end:]
        audit.append(
            {
                "chunk_id": candidate.chunk_id,
                "source": candidate.source,
                "candidate_text": candidate.text,
                "candidate_sha256": candidate.sha256,
                "original_claim": original_claim,
                "original_claim_sha256": _sha256_text(original_claim),
                "replacement_claim": replacement,
                "replacement_text": replacement,
                "replacement_sha256": _sha256_text(replacement),
                "replacement_fact_ids": fact_ids,
            }
        )
        last_start = start
    audit.reverse()
    return updated, audit


def apply_repairs(
    *,
    executed_code: str,
    console_stdout: str,
    candidates: list[CandidateChunk],
    repairs: list[dict[str, Any]],
    fact_catalog: dict[str, dict[str, str]],
    block_index: int | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    if any(
        candidate.source != "executed_code_comment"
        for candidate in candidates
    ):
        raise ValueError("Only executed-code comments may be repaired")
    code_candidates = [
        candidate
        for candidate in candidates
        if candidate.source == "executed_code_comment"
    ]
    code_ids = {candidate.chunk_id for candidate in code_candidates}
    repaired_code, code_audit = _apply_source_repairs(
        executed_code,
        candidates=code_candidates,
        repairs=[
            repair
            for repair in repairs
            if repair["chunk_id"] in code_ids
        ],
        fact_catalog=fact_catalog,
        block_index=block_index,
    )
    return repaired_code, console_stdout, code_audit


def _replace_prompt_text_once(
    prompt: list[dict[str, Any]],
    *,
    original: str,
    replacement: str,
) -> list[dict[str, Any]]:
    if original == replacement:
        return prompt
    updated = copy.deepcopy(prompt)
    matches = 0
    for message in updated:
        content = message.get("content")
        if isinstance(content, str):
            count = content.count(original)
            if count:
                message["content"] = content.replace(original, replacement)
                matches += count
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    continue
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                count = text.count(original)
                if count:
                    item["text"] = text.replace(original, replacement)
                    matches += count
    if matches != 1:
        raise ValueError(
            f"Expected one regeneration evidence segment, found {matches}"
        )
    return updated


def _prompt_structure(prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    structure: list[dict[str, Any]] = []
    for message_index, message in enumerate(prompt):
        content = message.get("content")
        record: dict[str, Any] = {
            "message_index": message_index,
            "role": message.get("role"),
        }
        if isinstance(content, str):
            record["content"] = [
                {
                    "type": "text",
                    "length": len(content),
                    "sha256": _sha256_text(content),
                }
            ]
        elif isinstance(content, list):
            items: list[dict[str, Any]] = []
            for item_index, item in enumerate(content):
                if not isinstance(item, dict):
                    items.append(
                        {
                            "item_index": item_index,
                            "type": type(item).__name__,
                            "sha256": _sha256_json(item),
                        }
                    )
                    continue
                item_type = item.get("type")
                if item_type == "text" and isinstance(item.get("text"), str):
                    text = item["text"]
                    items.append(
                        {
                            "item_index": item_index,
                            "type": "text",
                            "length": len(text),
                            "sha256": _sha256_text(text),
                        }
                    )
                elif item_type == "image_url":
                    image = ((item.get("image_url") or {}).get("url"))
                    items.append(
                        {
                            "item_index": item_index,
                            "type": "image_url",
                            "length": len(image) if isinstance(image, str) else None,
                            "sha256": (
                                _sha256_text(image)
                                if isinstance(image, str)
                                else _sha256_json(item)
                            ),
                        }
                    )
                else:
                    items.append(
                        {
                            "item_index": item_index,
                            "type": item_type,
                            "sha256": _sha256_json(item),
                        }
                    )
            record["content"] = items
        else:
            record["content"] = {
                "type": type(content).__name__,
                "sha256": _sha256_json(content),
            }
        structure.append(record)
    return structure


def _non_text_structure(prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    value = _prompt_structure(prompt)
    for message in value:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "text":
                item.pop("length", None)
                item.pop("sha256", None)
    return value


def _load_verified_history(root: Path) -> dict[str, Any]:
    path = root / HISTORY_FILE
    if not path.is_file():
        raise ValueError(f"Missing verified code history: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(value.get("blocks"), list)
    ):
        raise ValueError(f"Malformed verified code history: {path}")
    return value


def _save_verified_history(root: Path, history: dict[str, Any]) -> None:
    (root / HISTORY_FILE).write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def initialize_trial_trace(
    *,
    output_dir: str,
    trial: int,
    task: str,
) -> None:
    if task not in {"cube_stack", "cube_restack"}:
        raise ValueError(f"Unsupported envelope replacement task: {task}")
    root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    root.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": SCHEMA_VERSION,
        "trial": trial,
        "task": task,
        "stage": "INITIALIZED_BEFORE_ACTION",
        "action_visible_gt_packet_injected": False,
        "action_visible_bridge_injected": False,
        "student_pt_call_performed": False,
    }
    (root / "initialization.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _save_verified_history(
        root,
        {
            "schema_version": SCHEMA_VERSION,
            "trial": trial,
            "task": task,
            "blocks": [],
        },
    )


def run_envelope_replacement(
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
    native_decision_prompt: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return A1's prompt with only verified false text spans replaced."""
    if task not in {"cube_stack", "cube_restack"}:
        raise ValueError(f"Unsupported envelope replacement task: {task}")
    trial_root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    root = trial_root / f"turn_{turn:02d}"
    root.mkdir(parents=True, exist_ok=False)

    if len(executed_code_blocks) != turn or turn < 1:
        raise ValueError(
            "Envelope turn must equal the number of executed code blocks"
        )
    if "\n".join(executed_code_blocks) != executed_code:
        raise ValueError("Executed code block boundaries do not match A1 evidence")
    history = _load_verified_history(trial_root)
    history_blocks = history["blocks"]
    if len(history_blocks) != turn - 1:
        raise ValueError("Verified code history is not turn-synchronous")
    for block_index, (entry, original) in enumerate(
        zip(history_blocks, executed_code_blocks[:-1], strict=True)
    ):
        if (
            entry.get("block_index") != block_index
            or entry.get("original_sha256") != _sha256_text(original)
            or not isinstance(entry.get("corrected_code"), str)
        ):
            raise ValueError("Verified code history provenance mismatch")

    latest_block_index = turn - 1
    latest_code = executed_code_blocks[-1]
    candidates = _latest_block_candidates(
        latest_code,
        block_index=latest_block_index,
    )
    fact_catalog = _fact_catalog(env)
    prompt_path = Path(
        os.environ.get(
            "CAPX_ENVELOPE_REPLACEMENT_PROMPT_PATH",
            Path(__file__).with_name("envelope_replacement_verifier_prompt.md"),
        )
    )
    schema_path = Path(
        os.environ.get(
            "CAPX_ENVELOPE_REPLACEMENT_SCHEMA_PATH",
            Path(__file__).with_name("envelope_replacement_verifier_schema.json"),
        )
    )
    adjudicator_prompt_path = Path(
        os.environ.get(
            "CAPX_ENVELOPE_ADJUDICATOR_PROMPT_PATH",
            Path(__file__).with_name(
                "envelope_replacement_adjudicator_prompt.md"
            ),
        )
    )
    adjudicator_schema_path = Path(
        os.environ.get(
            "CAPX_ENVELOPE_ADJUDICATOR_SCHEMA_PATH",
            Path(__file__).with_name(
                "envelope_replacement_adjudicator_schema.json"
            ),
        )
    )
    prompt_text = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    adjudicator_prompt_text = adjudicator_prompt_path.read_text(encoding="utf-8")
    adjudicator_schema = json.loads(
        adjudicator_schema_path.read_text(encoding="utf-8")
    )

    repairs: list[dict[str, Any]] = []
    proposed_repairs: list[dict[str, Any]] = []
    verifier_content: str | None = None
    verifier_audit: dict[str, Any] | None = None
    verifier_protocol_audit: dict[str, Any] | None = None
    adjudicator_content: str | None = None
    adjudicator_audit: dict[str, Any] | None = None
    adjudicator_protocol_audit: dict[str, Any] | None = None
    adjudications: list[dict[str, Any]] = []
    if candidates:
        base_seed = (
            int(os.environ.get("CAPX_GLOBAL_SEED", "0"))
            + trial * 1000
            + VERIFIER_SEED_OFFSET
            + turn
        )
        schema_attempts = int(
            os.environ.get("CAPX_ENVELOPE_REPLACEMENT_SCHEMA_ATTEMPTS", "2")
        )
        if schema_attempts < 1:
            raise ValueError("Envelope verifier schema attempts must be positive")
        errors: list[dict[str, Any]] = []
        messages = _verifier_messages(
            prompt_text=prompt_text,
            schema=schema,
            task_description=task_description,
            candidates=candidates,
            fact_catalog=fact_catalog,
        )
        for attempt in range(schema_attempts):
            attempt_messages = copy.deepcopy(messages)
            if errors:
                attempt_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response violated the repair protocol: "
                            f"{errors[-1]['error']}. Return corrected strict JSON "
                            "only; do not add explanation."
                        ),
                    }
                )
            content, request_audit = _request(
                args,
                attempt_messages,
                seed=base_seed + attempt,
                max_tokens=1800,
            )
            try:
                parsed_repairs, protocol_audit = _normalize_repairs(
                    content,
                    candidates=candidates,
                    fact_catalog=fact_catalog,
                )
            except ValueError as exc:
                errors.append(
                    {
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "request": request_audit,
                        "response_sha256": _sha256_text(content),
                    }
                )
                if attempt + 1 == schema_attempts:
                    raise
                continue
            proposed_repairs = parsed_repairs
            verifier_content = content
            verifier_audit = request_audit
            verifier_protocol_audit = protocol_audit
            verifier_audit["schema_attempt"] = attempt
            verifier_audit["prior_schema_errors"] = errors
            break

        if proposed_repairs:
            adjudicator_seed = (
                int(os.environ.get("CAPX_GLOBAL_SEED", "0"))
                + trial * 1000
                + ADJUDICATOR_SEED_OFFSET
                + turn
            )
            adjudicator_attempts = int(
                os.environ.get("CAPX_ENVELOPE_ADJUDICATOR_SCHEMA_ATTEMPTS", "2")
            )
            if adjudicator_attempts < 1:
                raise ValueError(
                    "Envelope contradiction-gate schema attempts must be positive"
                )
            adjudicator_errors: list[dict[str, Any]] = []
            adjudicator_messages = _adjudicator_messages(
                prompt_text=adjudicator_prompt_text,
                schema=adjudicator_schema,
                candidates=candidates,
                fact_catalog=fact_catalog,
                proposed_repairs=proposed_repairs,
            )
            for attempt in range(adjudicator_attempts):
                attempt_messages = copy.deepcopy(adjudicator_messages)
                if adjudicator_errors:
                    attempt_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous response violated the contradiction-"
                                "gate protocol: "
                                f"{adjudicator_errors[-1]['error']}. Return corrected "
                                "strict JSON only."
                            ),
                        }
                    )
                content, request_audit = _request(
                    args,
                    attempt_messages,
                    seed=adjudicator_seed + attempt,
                    max_tokens=1800,
                )
                try:
                    parsed_adjudications, protocol_audit = (
                        _normalize_adjudications(
                            content,
                            proposed_repairs=proposed_repairs,
                        )
                    )
                except ValueError as exc:
                    adjudicator_errors.append(
                        {
                            "attempt": attempt,
                            "error": f"{type(exc).__name__}: {exc}",
                            "request": request_audit,
                            "response_sha256": _sha256_text(content),
                        }
                    )
                    if attempt + 1 == adjudicator_attempts:
                        raise
                    continue
                adjudications = parsed_adjudications
                adjudicator_content = content
                adjudicator_audit = request_audit
                adjudicator_protocol_audit = protocol_audit
                adjudicator_audit["schema_attempt"] = attempt
                adjudicator_audit["prior_schema_errors"] = adjudicator_errors
                break
            repairs = [
                proposed_repairs[decision["repair_index"]]
                for decision in adjudications
                if decision["accepted"]
            ]

    repaired_latest_code, repaired_stdout, patch_audit = apply_repairs(
        executed_code=latest_code,
        console_stdout="",
        candidates=candidates,
        repairs=repairs,
        fact_catalog=fact_catalog,
        block_index=latest_block_index,
    )
    if repaired_stdout:
        raise RuntimeError("Latest-block verification unexpectedly changed stdout")
    history_blocks.append(
        {
            "block_index": latest_block_index,
            "original_sha256": _sha256_text(latest_code),
            "corrected_sha256": _sha256_text(repaired_latest_code),
            "corrected_code": repaired_latest_code,
            "repair_count": len(repairs),
            "verified_after_turn": turn,
            "fact_catalog_sha256": _sha256_json(fact_catalog),
        }
    )
    repaired_code = "\n".join(
        str(entry["corrected_code"]) for entry in history_blocks
    )
    repaired_stdout = console_stdout
    active_repair_count = sum(
        int(entry["repair_count"]) for entry in history_blocks
    )
    native_evidence = multi_turn_prompt.format(
        executed_code=executed_code,
        console_stdout=console_stdout,
        console_stderr=console_stderr,
    )
    corrected_evidence = multi_turn_prompt.format(
        executed_code=repaired_code,
        console_stdout=repaired_stdout,
        console_stderr=console_stderr,
    )
    corrected_prompt = _replace_prompt_text_once(
        native_decision_prompt,
        original=native_evidence,
        replacement=corrected_evidence,
    )
    native_hash = _sha256_json(native_decision_prompt)
    corrected_hash = _sha256_json(corrected_prompt)
    repair_count = len(repairs)
    if active_repair_count == 0 and corrected_hash != native_hash:
        raise RuntimeError("No-op envelope treatment changed the A1 prompt")
    if active_repair_count > 0 and corrected_hash == native_hash:
        raise RuntimeError("Envelope repairs did not change the A1 prompt")
    if _non_text_structure(native_decision_prompt) != _non_text_structure(
        corrected_prompt
    ):
        raise RuntimeError("Envelope treatment changed prompt roles or media")
    if console_stderr not in corrected_evidence:
        raise RuntimeError("Envelope treatment changed or removed stderr evidence")

    record = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "trial": trial,
        "turn": turn,
        "native_prompt_sha256": native_hash,
        "corrected_prompt_sha256": corrected_hash,
        "native_prompt_structure": _prompt_structure(native_decision_prompt),
        "corrected_prompt_structure": _prompt_structure(corrected_prompt),
        "non_text_structure_identical": True,
        "native_execution_evidence_sha256": _sha256_text(native_evidence),
        "corrected_execution_evidence_sha256": _sha256_text(corrected_evidence),
        "executed_code_sha256": _sha256_text(executed_code),
        "repaired_executed_code_sha256": _sha256_text(repaired_code),
        "latest_code_block_index": latest_block_index,
        "latest_code_block_sha256": _sha256_text(latest_code),
        "repaired_latest_code_block_sha256": _sha256_text(
            repaired_latest_code
        ),
        "console_stdout_sha256": _sha256_text(console_stdout),
        "repaired_console_stdout_sha256": _sha256_text(repaired_stdout),
        "console_stderr_sha256": _sha256_text(console_stderr),
        "candidate_chunks": [asdict(candidate) for candidate in candidates],
        "candidate_count": len(candidates),
        "private_fact_catalog": fact_catalog,
        "private_fact_catalog_sha256": _sha256_json(fact_catalog),
        "verifier_raw_response": verifier_content,
        "verifier_audit": verifier_audit,
        "verifier_protocol_audit": verifier_protocol_audit,
        "proposed_repairs": proposed_repairs,
        "adjudicator_raw_response": adjudicator_content,
        "adjudicator_audit": adjudicator_audit,
        "adjudicator_protocol_audit": adjudicator_protocol_audit,
        "adjudications": adjudications,
        "repairs": repairs,
        "repair_count": repair_count,
        "active_persisted_repair_count": active_repair_count,
        "verified_history_sha256": _sha256_json(history),
        "applied_patches": patch_audit,
        "no_op_prompt_byte_equivalent": (
            corrected_hash == native_hash
            if active_repair_count == 0
            else None
        ),
        "only_candidate_sources_mutated": True,
        "action_message_count_unchanged": (
            len(native_decision_prompt) == len(corrected_prompt)
        ),
        "action_visible_gt_packet_injected": False,
        "action_visible_bridge_injected": False,
        "action_visible_verifier_protocol_injected": False,
        "student_pt_call_performed": False,
        "raw_image_replaced": False,
        "runtime_stderr_replaced": False,
        "task_or_api_text_replaced": False,
        "verifier_prompt_sha256": _sha256_text(prompt_text),
        "verifier_schema_sha256": _sha256_bytes(schema_path.read_bytes()),
        "adjudicator_prompt_sha256": _sha256_text(adjudicator_prompt_text),
        "adjudicator_schema_sha256": _sha256_bytes(
            adjudicator_schema_path.read_bytes()
        ),
    }
    (root / "replacement.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _save_verified_history(trial_root, history)
    return corrected_prompt
