"""Factual-Grounded Seamless Report Replacement (FGSRP).

FGSRP verifies factual clauses already present in a native regeneration report
and replaces only clauses that private evidence directly contradicts.  It does
not append a fact packet, expose the verification protocol, or generate advice.
"""

from __future__ import annotations

import copy
import io
import json
import os
import re
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from capx.verification import envelope_replacement as engine
from capx.verification import fgsrp_ir

SCHEMA_VERSION = "capx-fgsrp-v2.0-ir-alignment"
VERIFIER_SCHEMA_VERSION = "capx-fgsrp-claim-ledger-verifier-v2"
ADJUDICATOR_SCHEMA_VERSION = "capx-fgsrp-contradiction-gate-v1"
TRACE_DIR = "fgsrp_traces"
PASSIVE_TRACE_DIR = "fgsrp_passive_traces"
STATE_FILE = "fgsrp_state.json"
VERIFIER_SEED_OFFSET = 750_000
ADJUDICATOR_SEED_OFFSET = 760_000
ATOMIC_REFINER_SEED_OFFSET = 770_000
ATOMIC_REFINER_GATE_SEED_OFFSET = 780_000
IR_PARSER_SEED_OFFSET = 790_000
IR_REWRITER_SEED_OFFSET = 800_000

_UNDECIDABLE_CLAIM_PATTERNS = (
    r"\b(?:may|might|possibly|probably|likely|appears?|seems?|suggest(?:s|ing)?)\b",
    r"\b(?:visible|visibility|obscured|occluded)\b",
    r"\b(?:near|nearby|closer|farther|next to|side by side|adjacent)\b",
    r"\b(?:task|goal)\b.*\b(?:complete|completed|success|succeeded)\b",
)

_NO_CHANGE_CLAIM_PATTERNS = (
    r"\b(?:state|environment) (?:is|was) identical\b",
    r"\bno observable (?:difference|change)\b",
    r"\bno (?:movement|change|manipulation)"
    r"(?: or (?:movement|change|manipulation))* (?:has )?occurred\b",
    r"\b(?:nothing|neither .*?) (?:moved|changed)\b",
    r"\bremained unchanged\b",
    r"\b(?:remain(?:s|ed)?|stay(?:s|ed)?) in (?:the )?same "
    r"(?:relative )?(?:position(?:s)?|configuration|arrangement)\b",
    r"\b(?:relative )?(?:position(?:s)?|configuration|arrangement) "
    r"(?:remain(?:s|ed)?|stay(?:s|ed)?) (?:the )?same\b",
)

_CLAIM_DISPOSITIONS = {
    "preserve_supported",
    "preserve_compatible_or_undecidable",
    "preserve_non_editable",
    "replace_contradicted",
}

_NO_LONGER_PREDICATE_TRANSITIONS = {
    ("target_grasped_by_robot", "target_not_grasped_by_robot"),
    ("target_contacting_goal", "target_separated_from_goal"),
    ("target_contacting_table", "target_separated_from_table"),
}


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _sentence_units(prepared: PreparedCarrier) -> list[dict[str, Any]]:
    """Return deterministic report units whose ledger coverage can be audited."""
    units: list[dict[str, Any]] = []
    for candidate in prepared.candidates:
        sentence_index = 0
        for match in re.finditer(
            r"\S.*?(?:[.!?]+(?=\s|$)|(?=$))",
            candidate.text,
            flags=re.DOTALL,
        ):
            text = match.group(0)
            units.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "sentence_index": sentence_index,
                    "sentence_text": text,
                    "sentence_sha256": engine._sha256_text(text),
                    "start": candidate.start + match.start(),
                    "end": candidate.start + match.end(),
                }
            )
            sentence_index += 1
    return units


def _downgrade_unsupported_no_change_repairs(
    value: dict[str, Any],
    *,
    evidence: dict[str, dict[str, Any]],
) -> int:
    """Preserve a no-change claim when cited transitions all prove no change."""
    raw_ledger = value.get("claim_ledger")
    raw_repairs = value.get("repairs")
    if not isinstance(raw_ledger, list) or not isinstance(raw_repairs, list):
        return 0

    downgraded: list[tuple[str, str, str, tuple[str, ...]]] = []
    for entry in raw_ledger:
        if not isinstance(entry, dict) or not isinstance(entry.get("claims"), list):
            continue
        chunk_id = entry.get("chunk_id")
        if not isinstance(chunk_id, str):
            continue
        for claim in entry["claims"]:
            if not isinstance(claim, dict):
                continue
            original = claim.get("original_claim")
            replacement = claim.get("replacement_claim")
            fact_ids = claim.get("replacement_fact_ids")
            if (
                claim.get("disposition") != "replace_contradicted"
                or not isinstance(original, str)
                or not isinstance(replacement, str)
                or not isinstance(fact_ids, list)
                or not fact_ids
                or not all(isinstance(fact_id, str) for fact_id in fact_ids)
                or not _matches_any(original, _NO_CHANGE_CLAIM_PATTERNS)
            ):
                continue
            cited = [evidence.get(fact_id) for fact_id in fact_ids]
            if not all(
                fact_id.startswith("transition.") and isinstance(fact, dict)
                for fact_id, fact in zip(fact_ids, cited, strict=True)
            ):
                continue
            if any(fact.get("changed") is True for fact in cited):
                continue

            downgraded.append((chunk_id, original, replacement, tuple(fact_ids)))
            claim["disposition"] = "preserve_compatible_or_undecidable"
            claim["replacement_claim"] = None
            claim["replacement_fact_ids"] = []

    normalized_repairs = copy.deepcopy(raw_repairs)
    for expected in downgraded:
        matches = [
            index
            for index, repair in enumerate(normalized_repairs)
            if isinstance(repair, dict)
            and (
                repair.get("chunk_id"),
                repair.get("original_claim"),
                repair.get("replacement_claim"),
                tuple(repair.get("replacement_fact_ids", [])),
            )
            == expected
        ]
        if len(matches) != 1:
            raise ValueError(
                "FGSRP downgraded no-change claim lacks one matching repair"
            )
        normalized_repairs.pop(matches[0])
    value["repairs"] = normalized_repairs
    return len(downgraded)


def _normalize_claim_ledger(
    content: str,
    *,
    prepared: PreparedCarrier,
    evidence: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Validate exhaustive sentence coverage and its one-to-one repair mapping."""
    value = engine._extract_json_object(content)
    host_downgraded_no_change = _downgrade_unsupported_no_change_repairs(
        value,
        evidence=evidence,
    )
    raw_ledger = value.get("claim_ledger")
    expected_units = _sentence_units(prepared)
    if not isinstance(raw_ledger, list) or len(raw_ledger) != len(expected_units):
        raise ValueError("FGSRP claim ledger did not cover every sentence exactly once")

    normalized_ledger: list[dict[str, Any]] = []
    ledger_repairs: list[dict[str, Any]] = []
    for expected, entry in zip(expected_units, raw_ledger, strict=True):
        if not isinstance(entry, dict):
            raise ValueError("FGSRP claim-ledger entry must be an object")
        for field in ("chunk_id", "sentence_index", "sentence_text"):
            if entry.get(field) != expected[field]:
                raise ValueError(f"FGSRP claim-ledger sentence provenance mismatch: {field}")
        claims = entry.get("claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError("FGSRP claim-ledger sentence has no classified claims")
        normalized_claims: list[dict[str, Any]] = []
        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError("FGSRP claim-ledger claim must be an object")
            original = claim.get("original_claim")
            disposition = claim.get("disposition")
            replacement = claim.get("replacement_claim")
            fact_ids = claim.get("replacement_fact_ids")
            if (
                not isinstance(original, str)
                or not original.strip()
                or expected["sentence_text"].count(original) != 1
            ):
                raise ValueError("FGSRP ledger claim is not an exact sentence substring")
            if disposition not in _CLAIM_DISPOSITIONS:
                raise ValueError("FGSRP ledger claim has an invalid disposition")
            if not isinstance(fact_ids, list) or not all(
                isinstance(fact_id, str) for fact_id in fact_ids
            ):
                raise ValueError("FGSRP ledger evidence IDs must be a string list")
            if len(fact_ids) != len(set(fact_ids)):
                raise ValueError("FGSRP ledger evidence IDs are duplicated")
            if set(fact_ids) - set(evidence):
                raise ValueError("FGSRP ledger references unknown evidence IDs")
            if disposition == "replace_contradicted":
                if not isinstance(replacement, str) or not replacement.strip():
                    raise ValueError("FGSRP contradicted claim has no replacement")
                if not fact_ids:
                    raise ValueError(
                        "FGSRP contradicted claim lacks a minimal evidence set"
                    )
                if _matches_any(original, _NO_CHANGE_CLAIM_PATTERNS):
                    changed_transitions = [
                        fact_id
                        for fact_id in fact_ids
                        if fact_id.startswith("transition.")
                        and evidence[fact_id].get("changed") is True
                    ]
                    if len(fact_ids) != 1 or len(changed_transitions) != 1:
                        raise ValueError(
                            "FGSRP categorical no-change correction must select "
                            "exactly one changed transition predicate and state "
                            "only that qualitative transition"
                        )
                elif len(fact_ids) > 4:
                    raise ValueError(
                        "FGSRP contradicted claim selected "
                        f"{len(fact_ids)} evidence IDs; the maximum is four"
                    )
                ledger_repairs.append(
                    {
                        "chunk_id": expected["chunk_id"],
                        "original_claim": original,
                        "replacement_claim": replacement,
                        "replacement_fact_ids": fact_ids,
                    }
                )
            elif replacement is not None:
                raise ValueError("FGSRP preserved claim cannot carry a replacement")
            elif disposition != "preserve_supported" and fact_ids:
                raise ValueError(
                    "FGSRP unsupported preserved claim cannot cite evidence"
                )
            normalized_claims.append(
                {
                    "original_claim": original,
                    "disposition": disposition,
                    "replacement_claim": replacement,
                    "replacement_fact_ids": fact_ids,
                }
            )
        normalized_ledger.append({**expected, "claims": normalized_claims})

    # Validate the duplicated repair projection only after the ledger so retry
    # feedback exposes the precise semantic contract instead of a downstream
    # generic text-bound failure.
    repairs, audit = engine._normalize_repairs(
        json.dumps(value) if host_downgraded_no_change else content,
        candidates=prepared.candidates,
        fact_catalog=evidence,
        expected_schema_version=VERIFIER_SCHEMA_VERSION,
    )
    repair_projection = [
        {
            "chunk_id": repair["chunk_id"],
            "original_claim": repair["original_claim"],
            "replacement_claim": repair["replacement_claim"],
            "replacement_fact_ids": repair["replacement_fact_ids"],
        }
        for repair in repairs
    ]
    if repair_projection != ledger_repairs:
        raise ValueError(
            "FGSRP contradicted ledger claims and proposed repairs are not one-to-one"
        )
    if audit.get("every_sentence_listed_once") is not True:
        raise ValueError("FGSRP verifier did not certify sentence coverage")
    if audit.get("every_atomic_claim_classified") is not True:
        raise ValueError("FGSRP verifier did not certify atomic-claim coverage")
    if audit.get("contradicted_claims_match_repairs") is not True:
        raise ValueError("FGSRP verifier did not certify repair correspondence")
    audit = {
        **audit,
        "sentence_count": len(expected_units),
        "claim_count": sum(len(entry["claims"]) for entry in normalized_ledger),
        "host_downgraded_unsupported_no_change_repairs": (
            host_downgraded_no_change
        ),
    }
    return repairs, audit, normalized_ledger


def _value_is_expressed(value: str, claim: str) -> bool:
    patterns = {
        "target_grasped_by_robot": (
            r"\b(?:holding|holds|grasping|grasped|attached to)\b",
        ),
        "target_not_grasped_by_robot": (
            r"\bnot (?:holding|held|grasping|grasped|attached)\b",
            r"\bno longer (?:holding|held|grasping|grasped|attached)\b",
        ),
        "target_contacting_goal": (r"\b(?:in contact with|touching)\b",),
        "target_separated_from_goal": (
            r"\b(?:separated from|not in contact with)\b",
            r"\bno longer in contact\b",
        ),
        "target_contacting_table": (
            r"\b(?:in contact with|on|resting on) the table\b",
        ),
        "target_separated_from_table": (
            r"\b(?:separated from|not in contact with|off|above) the table\b",
            r"\bheld in (?:the )?air\b",
            r"\bno longer in contact with the table\b",
        ),
        "target_center_negative_x_of_goal_band": (r"\bnegative-x side\b",),
        "target_center_within_goal_x_band": (r"\bwithin .*?x band\b",),
        "target_center_positive_x_of_goal_band": (r"\bpositive-x side\b",),
        "target_center_negative_y_of_goal_band": (r"\bnegative-y side\b",),
        "target_center_within_goal_y_band": (r"\bwithin .*?y band\b",),
        "target_center_positive_y_of_goal_band": (r"\bpositive-y side\b",),
        "target_below_goal_height_band": (r"\b(?:below|under)\b",),
        "target_within_goal_height_band": (r"\bwithin .*?height band\b",),
        "target_above_goal_height_band": (r"\b(?:above|on top of)\b",),
        "target_center_inside_goal_footprint": (r"\binside .*?footprint\b",),
        "target_center_outside_goal_footprint": (r"\boutside .*?footprint\b",),
    }
    return _matches_any(claim, patterns.get(value, ()))


def _transition_replacement_is_entailed(
    before_value: str,
    after_value: str,
    claim: str,
) -> bool:
    """Return whether a replacement states truth entailed by a transition."""
    if re.search(r"\bno longer\b", claim, re.IGNORECASE):
        return (
            (before_value, after_value) in _NO_LONGER_PREDICATE_TRANSITIONS
            and _value_is_expressed(after_value, claim)
        )
    return _value_is_expressed(after_value, claim)


def _value_directly_contradicts(value: str, claim: str) -> bool:
    patterns = {
        "target_grasped_by_robot": (
            r"\bnot (?:holding|held|grasping|grasped|attached)\b",
            r"\bno longer (?:holding|held|grasping|grasped|attached)\b",
        ),
        "target_not_grasped_by_robot": (
            r"\b(?:holding|holds|grasping|grasped|attached to)\b",
        ),
        "target_contacting_goal": (
            r"\b(?:separated from|not in contact with)\b",
            r"\bno longer in contact\b",
        ),
        "target_separated_from_goal": (
            r"\b(?:in contact with|touching|on top of|"
            r"stacked (?:directly )?on|resting (?:directly )?on|"
            r"placed (?:directly )?on)\b",
        ),
        "target_contacting_table": (
            r"\b(?:lifted|off|above) the table\b",
            r"\bheld in (?:the )?air\b",
        ),
        "target_separated_from_table": (
            r"\b(?:in contact with|on|resting on) the table\b",
        ),
        "target_center_negative_x_of_goal_band": (
            r"\b(?:within .*?x band|positive-x side)\b",
        ),
        "target_center_within_goal_x_band": (
            r"\b(?:negative-x side|positive-x side)\b",
        ),
        "target_center_positive_x_of_goal_band": (
            r"\b(?:within .*?x band|negative-x side)\b",
        ),
        "target_center_negative_y_of_goal_band": (
            r"\b(?:within .*?y band|positive-y side)\b",
        ),
        "target_center_within_goal_y_band": (
            r"\b(?:negative-y side|positive-y side)\b",
        ),
        "target_center_positive_y_of_goal_band": (
            r"\b(?:within .*?y band|negative-y side)\b",
        ),
        "target_below_goal_height_band": (
            r"\b(?:above|on top of|stacked (?:directly )?on|"
            r"resting (?:directly )?on|placed (?:directly )?on)\b",
        ),
        "target_within_goal_height_band": (
            r"\b(?:below|under|above|on top of)\b",
        ),
        "target_above_goal_height_band": (r"\b(?:below|under)\b",),
        "target_center_inside_goal_footprint": (r"\boutside .*?footprint\b",),
        "target_center_outside_goal_footprint": (
            r"\binside .*?footprint\b",
            r"\b(?:directly )?(?:above|on top of|stacked on|resting on)\b",
        ),
    }
    return _matches_any(claim, patterns.get(value, ()))


def _deterministic_contradiction_certificate(
    repair: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Certify only direct predicate contradictions expressible by the GT catalog."""
    original = repair["original_claim"]
    replacement = repair["replacement_claim"]
    fact_ids = repair["replacement_fact_ids"]
    if _matches_any(original, _UNDECIDABLE_CLAIM_PATTERNS):
        return {"accepted": False, "reason": "claim_semantics_not_in_gt_vocabulary"}
    transition_ids = [
        fact_id for fact_id in fact_ids if fact_id.startswith("transition.")
    ]
    if transition_ids:
        if len(fact_ids) != 1:
            return {
                "accepted": False,
                "reason": "mixed_transition_evidence_not_certified",
            }
        transition = evidence[transition_ids[0]]
        before_value = str(transition.get("before_value", ""))
        after_value = str(transition.get("after_value", ""))
        if (
            transition.get("changed") is not True
            or not before_value
            or not after_value
            or before_value == after_value
        ):
            return {"accepted": False, "reason": "no_proven_predicate_transition"}
        if not _matches_any(original, _NO_CHANGE_CLAIM_PATTERNS):
            return {
                "accepted": False,
                "reason": "transition_claim_negation_not_certified",
            }
        if not _transition_replacement_is_entailed(
            before_value,
            after_value,
            replacement,
        ):
            return {
                "accepted": False,
                "reason": "transition_replacement_not_canonically_entailed",
            }
        return {
            "accepted": True,
            "reason": "direct_predicate_transition_contradiction",
            "contradicting_fact_ids": transition_ids,
            "entailed_fact_ids": transition_ids,
        }

    selected = [evidence[fact_id] for fact_id in fact_ids]
    values = [str(item.get("value", "")) for item in selected]
    if any(not value or not _value_is_expressed(value, replacement) for value in values):
        return {"accepted": False, "reason": "replacement_not_canonically_entailed"}
    contradicting = [
        fact_id
        for fact_id, value in zip(fact_ids, values, strict=True)
        if _value_directly_contradicts(value, original)
    ]
    if not contradicting:
        return {"accepted": False, "reason": "no_direct_predicate_contradiction"}
    return {
        "accepted": True,
        "reason": "direct_predicate_contradiction",
        "contradicting_fact_ids": contradicting,
        "entailed_fact_ids": list(fact_ids),
    }


def _host_atomic_gate_override(
    decision: dict[str, Any],
    certificate: dict[str, Any],
) -> bool:
    """Resolve a certified false no-change clause when all truth checks pass.

    A global no-change assertion is itself a complete factual clause.  A single
    proven predicate transition falsifies it even when the semantic gate calls
    the assertion non-atomic because it summarizes the whole scene.
    """
    return (
        certificate.get("accepted") is True
        and certificate.get("reason")
        == "direct_predicate_transition_contradiction"
        and decision.get("relation") == "contradicted"
        and decision.get("selected_facts_entail_claim_negation") is True
        and decision.get("replacement_claim_entailed_by_selected_facts") is True
    )


class EvidenceProvider(Protocol):
    """Private evidence backend; implementations are not visible to students."""

    name: str

    def capture(self, env: Any, *, task: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PreparedCarrier:
    kind: str
    source_text: str
    candidates: list[engine.CandidateChunk]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AppliedCarrier:
    corrected_prompt: list[dict[str, Any]]
    corrected_source: str
    active_repair_count: int
    patches: list[dict[str, Any]]
    state_record: dict[str, Any]


class CarrierAdapter(Protocol):
    """Locates and replaces one native report carrier without adding content."""

    kind: str

    def prepare(self, context: dict[str, Any], state: dict[str, Any]) -> PreparedCarrier: ...

    def apply(
        self,
        prepared: PreparedCarrier,
        *,
        repairs: list[dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
        context: dict[str, Any],
        state: dict[str, Any],
    ) -> AppliedCarrier: ...


class CubeSimulatorEvidenceProvider:
    """Initial evidence backend for the two cube tasks.

    Cube-specific entity and simulator details remain confined to this adapter;
    the replacement core consumes only evidence records with stable IDs.
    """

    name = "cube_simulator_gt_v1"

    def capture(self, env: Any, *, task: str) -> dict[str, Any]:
        if task not in {"cube_stack", "cube_restack"}:
            raise ValueError(f"Cube evidence provider does not support {task!r}")
        facts = engine._fact_catalog(env)
        support_value = (
            "target_supported_by_goal"
            if (
                facts["relative_z"]["value"] == "target_above_goal_height_band"
                and facts["footprint"]["value"]
                == "target_center_inside_goal_footprint"
                and facts["target_goal_contact"]["value"]
                == "target_contacting_goal"
                and facts["target_table_contact"]["value"]
                == "target_separated_from_table"
            )
            else "target_not_supported_by_goal"
        )
        facts["target_goal_support"] = {
            "predicate": "target_goal_support",
            "value": support_value,
            "statement": (
                "the red cube is currently supported by the green cube"
                if support_value == "target_supported_by_goal"
                else "the red cube is currently not supported by the green cube"
            ),
        }
        return {
            "provider": self.name,
            "task": task,
            "predicate_ontology": copy.deepcopy(
                fgsrp_ir.CUBE_PREDICATE_ONTOLOGY
            ),
            "entity_registry": {
                "executor.robot_gripper": {
                    "kind": "executor",
                    "public_name": "the gripper",
                    "aliases": ["the gripper", "gripper", "robot gripper"],
                },
                "executor.robot_arm": {
                    "kind": "executor",
                    "public_name": "the robot arm",
                    "aliases": ["the robot arm", "robot arm", "arm"],
                },
                "object.target": {
                    "kind": "object",
                    "public_name": "the red cube",
                    "aliases": ["the red cube", "red cube", "target", "it"],
                },
                "object.goal": {
                    "kind": "object",
                    "public_name": "the green cube",
                    "aliases": ["the green cube", "green cube", "goal", "it"],
                },
                "support.table": {
                    "kind": "support",
                    "public_name": "the table",
                    "aliases": ["the table", "table", "table surface"],
                },
                "scene.environment": {
                    "kind": "scene",
                    "public_name": "the environment",
                    "aliases": [
                        "the environment",
                        "environment",
                        "the current state of the environment",
                        "current state of the environment",
                    ],
                },
            },
            "facts": facts,
        }


_EVIDENCE_PROVIDERS: dict[str, EvidenceProvider] = {
    "cube_stack": CubeSimulatorEvidenceProvider(),
    "cube_restack": CubeSimulatorEvidenceProvider(),
}


def register_evidence_provider(task: str, provider: EvidenceProvider) -> None:
    """Register an environment adapter without changing the FGSRP core."""
    normalized = task.strip()
    if not normalized:
        raise ValueError("FGSRP evidence task name cannot be empty")
    if normalized in _EVIDENCE_PROVIDERS:
        raise ValueError(f"FGSRP evidence provider already registered: {normalized}")
    _EVIDENCE_PROVIDERS[normalized] = provider


def _provider_for(task: str) -> EvidenceProvider:
    try:
        return _EVIDENCE_PROVIDERS[task]
    except KeyError as exc:
        raise ValueError(
            f"No FGSRP evidence adapter is registered for task {task!r}"
        ) from exc


def _temporal_evidence(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if before.get("provider") != after.get("provider"):
        raise ValueError("FGSRP evidence provider changed within a trial")
    if before.get("entity_registry") != after.get("entity_registry"):
        raise ValueError("FGSRP entity registry changed within a trial")
    if before.get("predicate_ontology") != after.get("predicate_ontology"):
        raise ValueError("FGSRP predicate ontology changed within a trial")
    before_facts = before.get("facts")
    after_facts = after.get("facts")
    if not isinstance(before_facts, dict) or not isinstance(after_facts, dict):
        raise ValueError("FGSRP evidence snapshot has no fact mapping")
    if set(before_facts) != set(after_facts):
        raise ValueError("FGSRP evidence vocabulary changed within a transition")

    catalog: dict[str, dict[str, Any]] = {}
    for fact_id in sorted(before_facts):
        before_fact = copy.deepcopy(before_facts[fact_id])
        after_fact = copy.deepcopy(after_facts[fact_id])
        before_statement = str(before_fact.get("statement", "")).strip()
        after_statement = str(after_fact.get("statement", "")).strip()
        if not before_statement or not after_statement:
            raise ValueError(f"FGSRP evidence fact has no statement: {fact_id}")
        catalog[f"before.{fact_id}"] = {
            **before_fact,
            "evidence_id": f"before.{fact_id}",
            "temporal_scope": "immediately_before_execution",
            "statement": f"Immediately before execution, {before_statement}.",
        }
        catalog[f"after.{fact_id}"] = {
            **after_fact,
            "evidence_id": f"after.{fact_id}",
            "temporal_scope": "immediately_after_execution",
            "statement": f"Immediately after execution, {after_statement}.",
        }
        changed = before_fact.get("value") != after_fact.get("value")
        catalog[f"transition.{fact_id}"] = {
            "evidence_id": f"transition.{fact_id}",
            "predicate": fact_id,
            "temporal_scope": "execution_transition",
            "changed": changed,
            "before_value": before_fact.get("value"),
            "after_value": after_fact.get("value"),
            "statement": (
                f"Across this execution, the {fact_id} relation changed from "
                f"'{before_statement}' to '{after_statement}'."
                if changed
                else f"Across this execution, the {fact_id} relation remained "
                f"'{after_statement}'."
            ),
        }
    return catalog


def _non_comment_tokens(source: str) -> list[tuple[int, str]]:
    return [
        (token.type, token.string)
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type != tokenize.COMMENT
    ]


class M2CommentCarrier:
    kind = "m2_code_comments"

    def prepare(self, context: dict[str, Any], state: dict[str, Any]) -> PreparedCarrier:
        blocks = context["executed_code_blocks"]
        turn = context["turn"]
        if len(blocks) != turn or turn < 1:
            raise ValueError("FGSRP M2 turn is not synchronized with code blocks")
        if "\n".join(blocks) != context["executed_code"]:
            raise ValueError("FGSRP M2 code block boundaries do not match evidence")
        history = state.get("m2_blocks")
        if not isinstance(history, list) or len(history) != turn - 1:
            raise ValueError("FGSRP M2 verified history is not turn-synchronous")
        for index, (entry, original) in enumerate(
            zip(history, blocks[:-1], strict=True)
        ):
            if (
                entry.get("block_index") != index
                or entry.get("original_sha256") != engine._sha256_text(original)
                or not isinstance(entry.get("corrected_code"), str)
            ):
                raise ValueError("FGSRP M2 history provenance mismatch")
        latest_index = turn - 1
        latest = blocks[-1]
        candidates = engine._latest_block_candidates(
            latest,
            block_index=latest_index,
        )
        return PreparedCarrier(
            kind=self.kind,
            source_text=latest,
            candidates=candidates,
            metadata={"latest_block_index": latest_index},
        )

    def apply(
        self,
        prepared: PreparedCarrier,
        *,
        repairs: list[dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
        context: dict[str, Any],
        state: dict[str, Any],
    ) -> AppliedCarrier:
        corrected_latest, patches = engine._apply_source_repairs(
            prepared.source_text,
            candidates=prepared.candidates,
            repairs=repairs,
            fact_catalog=evidence,
        )
        if _non_comment_tokens(corrected_latest) != _non_comment_tokens(
            prepared.source_text
        ):
            raise RuntimeError("FGSRP M2 adapter changed executable Python tokens")
        history = state["m2_blocks"]
        history.append(
            {
                "block_index": prepared.metadata["latest_block_index"],
                "original_sha256": engine._sha256_text(prepared.source_text),
                "corrected_sha256": engine._sha256_text(corrected_latest),
                "corrected_code": corrected_latest,
                "repair_count": len(repairs),
            }
        )
        corrected_code = "\n".join(str(entry["corrected_code"]) for entry in history)
        native_evidence = context["multi_turn_prompt"].format(
            executed_code=context["executed_code"],
            console_stdout=context["console_stdout"],
            console_stderr=context["console_stderr"],
        )
        corrected_evidence = context["multi_turn_prompt"].format(
            executed_code=corrected_code,
            console_stdout=context["console_stdout"],
            console_stderr=context["console_stderr"],
        )
        prompt = engine._replace_prompt_text_once(
            context["native_decision_prompt"],
            original=native_evidence,
            replacement=corrected_evidence,
        )
        return AppliedCarrier(
            corrected_prompt=prompt,
            corrected_source=corrected_latest,
            active_repair_count=sum(int(entry["repair_count"]) for entry in history),
            patches=patches,
            state_record={
                "block_index": prepared.metadata["latest_block_index"],
                "native_evidence_sha256": engine._sha256_text(native_evidence),
                "corrected_evidence_sha256": engine._sha256_text(corrected_evidence),
            },
        )


class M3VisualDifferenceCarrier:
    kind = "m3_visual_difference_report"

    def prepare(self, context: dict[str, Any], state: dict[str, Any]) -> PreparedCarrier:
        del state
        report = context.get("visual_differencing_feedback")
        if not isinstance(report, str) or not report.strip():
            raise ValueError("FGSRP M3 carrier requires a native visual-difference report")
        candidate = engine.CandidateChunk(
            chunk_id=f"visual_difference.turn_{context['turn']:02d}.report",
            source="visual_differencing_report",
            start=0,
            end=len(report),
            text=report,
            sha256=engine._sha256_text(report),
        )
        return PreparedCarrier(
            kind=self.kind,
            source_text=report,
            candidates=[candidate],
            metadata={"report_turn": context["turn"]},
        )

    def apply(
        self,
        prepared: PreparedCarrier,
        *,
        repairs: list[dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
        context: dict[str, Any],
        state: dict[str, Any],
    ) -> AppliedCarrier:
        corrected_report, patches = engine._apply_source_repairs(
            prepared.source_text,
            candidates=prepared.candidates,
            repairs=repairs,
            fact_catalog=evidence,
        )
        prompt = engine._replace_prompt_text_once(
            context["native_decision_prompt"],
            original=prepared.source_text,
            replacement=corrected_report,
        )
        state["m3_reports"].append(
            {
                "turn": context["turn"],
                "original_sha256": engine._sha256_text(prepared.source_text),
                "corrected_sha256": engine._sha256_text(corrected_report),
                "repair_count": len(repairs),
            }
        )
        return AppliedCarrier(
            corrected_prompt=prompt,
            corrected_source=corrected_report,
            active_repair_count=len(repairs),
            patches=patches,
            state_record={"report_turn": prepared.metadata["report_turn"]},
        )


_CARRIERS: dict[str, CarrierAdapter] = {
    "m2_comments": M2CommentCarrier(),
    "m3_visual_difference": M3VisualDifferenceCarrier(),
}


def _carrier(name: str) -> CarrierAdapter:
    try:
        return _CARRIERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown FGSRP carrier: {name!r}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def initialize_trial_trace(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    task: str,
) -> None:
    root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    root.mkdir(parents=True, exist_ok=False)
    provider = _provider_for(task)
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
            "initial_evidence_sha256": engine._sha256_json(initial),
            "action_visible_gt_packet_injected": False,
            "action_visible_bridge_injected": False,
        },
    )


def initialize_passive_trial_trace(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    task: str,
) -> None:
    """Capture private GT for A0 replay certification without changing its prompt."""
    root = Path(output_dir) / PASSIVE_TRACE_DIR / f"trial_{trial:02d}"
    root.mkdir(parents=True, exist_ok=False)
    provider = _provider_for(task)
    initial = provider.capture(env, task=task)
    _save_json(
        root / STATE_FILE,
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "passive_action_invisible",
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
        },
    )


def capture_passive_transition(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    turn: int,
    task: str,
) -> None:
    """Record one A0 GT transition; this function has no action-visible output."""
    trial_root = Path(output_dir) / PASSIVE_TRACE_DIR / f"trial_{trial:02d}"
    turn_root = trial_root / f"turn_{turn:02d}"
    turn_root.mkdir(parents=True, exist_ok=False)
    state = _load_json(trial_root / STATE_FILE)
    snapshots = state.get("snapshots")
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("mode") != "passive_action_invisible"
        or state.get("task") != task
        or state.get("trial") != trial
        or not isinstance(snapshots, list)
        or len(snapshots) != turn
    ):
        raise ValueError("FGSRP passive A0 trace is not turn-synchronous")
    provider = _provider_for(task)
    before = snapshots[-1]["evidence"]
    after = provider.capture(env, task=task)
    evidence = _temporal_evidence(before, after)
    snapshots.append(
        {
            "phase": "after_executed_block_before_native_report",
            "turn": turn,
            "evidence": after,
            "sha256": engine._sha256_json(after),
        }
    )
    _save_json(
        turn_root / "transition.json",
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "passive_action_invisible",
            "task": task,
            "trial": trial,
            "turn": turn,
            "provider": provider.name,
            "before_evidence_sha256": engine._sha256_json(before),
            "after_evidence_sha256": engine._sha256_json(after),
            "temporal_evidence_sha256": engine._sha256_json(evidence),
            "private_temporal_evidence": evidence,
            "action_visible_gt_packet_injected": False,
            "action_visible_bridge_injected": False,
            "action_request_changed": False,
        },
    )
    _save_json(trial_root / STATE_FILE, state)


def _verifier_messages(
    *,
    prompt: str,
    schema: dict[str, Any],
    task_description: str,
    prepared: PreparedCarrier,
    entity_registry: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    packet = {
        "task_description": task_description.split("\nAPIs:", 1)[0].strip(),
        "native_report_carrier": prepared.kind,
        "candidate_chunks": [asdict(candidate) for candidate in prepared.candidates],
        "host_sentence_units": _sentence_units(prepared),
        "private_entity_registry": entity_registry,
        "authoritative_temporal_evidence": evidence,
        "required_response_schema": schema,
    }
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                "Classify every atomic claim in every host sentence unit against "
                "the private temporal evidence. Propose minimal corrections only "
                "for directly contradicted claims. Return strict JSON only.\n\n"
                + json.dumps(packet, ensure_ascii=True, sort_keys=True)
            ),
        },
    ]


def _adjudicator_messages(
    *,
    prompt: str,
    schema: dict[str, Any],
    prepared: PreparedCarrier,
    evidence: dict[str, dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_map = {candidate.chunk_id: candidate for candidate in prepared.candidates}
    proposals = []
    for index, repair in enumerate(repairs):
        proposals.append(
            {
                "repair_index": index,
                "chunk_id": repair["chunk_id"],
                "candidate_text": candidate_map[repair["chunk_id"]].text,
                "original_claim": repair["original_claim"],
                "replacement_claim": repair["replacement_claim"],
                "selected_authoritative_evidence": [
                    evidence[item] for item in repair["replacement_fact_ids"]
                ],
            }
        )
    packet = {
        "native_report_carrier": prepared.kind,
        "proposed_repairs": proposals,
        "complete_authoritative_temporal_evidence": evidence,
        "required_response_schema": schema,
    }
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                "Adjudicate each proposed native-report edit. Return strict "
                "JSON only.\n\n"
                + json.dumps(packet, ensure_ascii=True, sort_keys=True)
            ),
        },
    ]


def _atomic_refiner_messages(
    *,
    verifier_messages: list[dict[str, Any]],
    known_false: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages = copy.deepcopy(verifier_messages)
    packet = {
        "known_false_non_atomic_repairs": known_false,
        "required_output": (
            "Return one atomic repair for every listed known-false claim."
        ),
    }
    messages.append(
        {
            "role": "user",
            "content": (
                "The contradiction gate proved that each listed proposal "
                "contains a false physical claim, but rejected its text edit "
                "because the span or replacement also changes supported meaning. "
                "Refine each edit to the exact smallest contradicted atomic "
                "substring. Preserve every byte outside that substring. Use only "
                "the same selected evidence IDs. Do not omit any listed false "
                "claim. Return strict verifier-schema JSON only.\n\n"
                + json.dumps(packet, ensure_ascii=True, sort_keys=True)
            ),
        }
    )
    return messages


def _refinement_parent(
    repair: dict[str, Any],
    known_false: list[dict[str, Any]],
) -> int | None:
    matches = []
    for index, item in enumerate(known_false):
        if repair["chunk_id"] != item["chunk_id"]:
            continue
        if repair["original_claim"] not in item["original_claim"]:
            continue
        if not set(repair["replacement_fact_ids"]).issubset(
            set(item["replacement_fact_ids"])
        ):
            continue
        decision = item["gate_decision"]
        if (
            decision["span_is_smallest_complete_clause"] is False
            and repair["original_claim"] == item["original_claim"]
        ):
            continue
        matches.append(index)
    return matches[0] if len(matches) == 1 else None


def _schema_retry_context(
    *,
    prior_response: str | None,
    error: str,
    label: str,
    directive: str,
) -> list[dict[str, str]]:
    """Materialize the missing assistant turn for a stateless schema retry."""
    if prior_response is None:
        raise ValueError("FGSRP schema retry lacks its prior assistant response")
    return [
        {"role": "assistant", "content": prior_response},
        {
            "role": "user",
            "content": f"The previous {label} was invalid: {error}. {directive}",
        },
    ]


def _ir_parser_messages(
    *,
    prompt: str,
    schema: dict[str, Any],
    task_description: str,
    prepared: PreparedCarrier,
    entity_registry: dict[str, Any],
    ontology: dict[str, Any],
) -> list[dict[str, Any]]:
    packet = {
        "task_description": task_description.split("\nAPIs:", 1)[0].strip(),
        "native_report_carrier": prepared.kind,
        "candidate_chunks": [asdict(item) for item in prepared.candidates],
        "host_sentence_units": _sentence_units(prepared),
        "private_entity_registry": entity_registry,
        "predicate_ontology": ontology,
        "current_gt_values_supplied": False,
        "required_response_schema": schema,
    }
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                "Parse every atomic claim into semantic IR without judging its "
                "truth. Return strict JSON only.\n\n"
                + json.dumps(packet, ensure_ascii=True, sort_keys=True)
            ),
        },
    ]


def _ir_rewriter_messages(
    *,
    prompt: str,
    schema: dict[str, Any],
    prepared: PreparedCarrier,
    false_claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    packet = {
        "native_report_carrier": prepared.kind,
        "false_claims": false_claims,
        "required_response_schema": schema,
    }
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                "Write one minimal local correction for every host-proven false "
                "claim. Return strict JSON only.\n\n"
                + json.dumps(packet, ensure_ascii=True, sort_keys=True)
            ),
        },
    ]


def _ir_unknown_ledger(prepared: PreparedCarrier) -> list[dict[str, Any]]:
    return [
        {
            **unit,
            "claims": [
                {
                    "original_claim": unit["sentence_text"],
                    "claim_type": "unknown",
                    "assertions": [],
                    "disposition": "preserve_compatible_or_undecidable",
                    "replacement_claim": None,
                    "replacement_fact_ids": [],
                }
            ],
        }
        for unit in _sentence_units(prepared)
    ]


def _downgrade_ir_false_claims(ledger: list[dict[str, Any]]) -> None:
    for sentence in ledger:
        for claim in sentence["claims"]:
            if claim["disposition"] != "replace_contradicted":
                continue
            claim["disposition"] = "preserve_compatible_or_undecidable"
            claim["replacement_claim"] = None
            claim["replacement_fact_ids"] = []


def _query_ir_repairs(
    args: Any,
    *,
    task_description: str,
    prepared: PreparedCarrier,
    entity_registry: dict[str, Any],
    ontology: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    trial: int,
    turn: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parser_prompt_path = Path(
        os.environ.get(
            "CAPX_FGSRP_IR_PARSER_PROMPT_PATH",
            Path(__file__).with_name("fgsrp_ir_parser_prompt.md"),
        )
    )
    parser_schema_path = Path(
        os.environ.get(
            "CAPX_FGSRP_IR_PARSER_SCHEMA_PATH",
            Path(__file__).with_name("fgsrp_ir_parser_schema.json"),
        )
    )
    rewriter_prompt_path = Path(
        os.environ.get(
            "CAPX_FGSRP_IR_REWRITER_PROMPT_PATH",
            Path(__file__).with_name("fgsrp_ir_rewriter_prompt.md"),
        )
    )
    rewriter_schema_path = Path(
        os.environ.get(
            "CAPX_FGSRP_IR_REWRITER_SCHEMA_PATH",
            Path(__file__).with_name("fgsrp_ir_rewriter_schema.json"),
        )
    )
    parser_prompt = parser_prompt_path.read_text(encoding="utf-8")
    parser_schema = _load_json(parser_schema_path)
    rewriter_prompt = rewriter_prompt_path.read_text(encoding="utf-8")
    rewriter_schema = _load_json(rewriter_schema_path)
    protocol: dict[str, Any] = {
        "architecture": "llm_semantic_ir_then_host_gt_evaluation",
        "parser_received_current_gt_values": False,
        "verifier_prompt_sha256": engine._sha256_text(parser_prompt),
        "verifier_schema_sha256": engine._sha256_bytes(
            parser_schema_path.read_bytes()
        ),
        "adjudicator_prompt_sha256": engine._sha256_text(rewriter_prompt),
        "adjudicator_schema_sha256": engine._sha256_bytes(
            rewriter_schema_path.read_bytes()
        ),
        "known_false_claims": [],
        "atomic_refinement": None,
    }
    seed_base = int(os.environ.get("CAPX_GLOBAL_SEED", "0")) + trial * 1000 + turn
    parser_messages = _ir_parser_messages(
        prompt=parser_prompt,
        schema=parser_schema,
        task_description=task_description,
        prepared=prepared,
        entity_registry=entity_registry,
        ontology=ontology,
    )
    attempts = int(os.environ.get("CAPX_FGSRP_SCHEMA_ATTEMPTS", "2"))
    errors: list[dict[str, Any]] = []
    content: str | None = None
    for attempt in range(attempts):
        messages = copy.deepcopy(parser_messages)
        if errors:
            messages.extend(
                _schema_retry_context(
                    prior_response=content,
                    error=errors[-1]["error"],
                    label="FGSRP IR parser schema",
                    directive="Return corrected strict JSON only.",
                )
            )
        try:
            content, request_audit = engine._request(
                args,
                messages,
                seed=seed_base + IR_PARSER_SEED_OFFSET + attempt,
                max_tokens=2600,
            )
        except Exception as exc:
            ledger = _ir_unknown_ledger(prepared)
            protocol["verifier"] = {
                "request": {
                    "seed": seed_base + IR_PARSER_SEED_OFFSET + attempt,
                    "schema_attempt": attempt,
                    "error_type": type(exc).__name__,
                },
                "response": None,
                "protocol": {
                    "status": "abstained_transport_error",
                    "sentence_count": len(ledger),
                    "claim_count": len(ledger),
                },
                "proposals": [],
                "claim_ledger": ledger,
            }
            protocol["deterministic_gate"] = []
            protocol["adjudicator"] = None
            protocol["adjudicator_input_map"] = []
            return [], protocol
        try:
            ledger, false_claims, parser_audit = fgsrp_ir.normalize_parser_response(
                content,
                prepared=prepared,
                ontology=ontology,
                evidence=evidence,
                entity_registry=entity_registry,
                sentence_units=_sentence_units(prepared),
            )
        except ValueError as exc:
            errors.append(
                {
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "request": request_audit,
                    "response_sha256": engine._sha256_text(content),
                }
            )
            if attempt + 1 == attempts:
                ledger = _ir_unknown_ledger(prepared)
                protocol["verifier"] = {
                    "request": {
                        **request_audit,
                        "schema_attempt": attempt,
                        "prior_schema_errors": errors,
                    },
                    "response": content,
                    "protocol": {
                        "status": "abstained_invalid_response",
                        "sentence_count": len(ledger),
                        "claim_count": len(ledger),
                    },
                    "proposals": [],
                    "claim_ledger": ledger,
                }
                protocol["deterministic_gate"] = []
                protocol["adjudicator"] = None
                protocol["adjudicator_input_map"] = []
                return [], protocol
            continue
        break
    request_audit["schema_attempt"] = attempt
    request_audit["prior_schema_errors"] = errors
    protocol["verifier"] = {
        "request": request_audit,
        "response": content,
        "protocol": parser_audit,
        "proposals": [],
        "claim_ledger": ledger,
    }
    protocol["ir_evaluation"] = {
        "physical_claims": sum(
            claim["claim_type"] == "physical"
            for sentence in ledger
            for claim in sentence["claims"]
        ),
        "false_claims": len(false_claims),
        "evaluation_is_deterministic": True,
    }
    if not false_claims:
        protocol["deterministic_gate"] = []
        protocol["adjudicator"] = None
        protocol["adjudicator_input_map"] = []
        return [], protocol

    writer_messages = _ir_rewriter_messages(
        prompt=rewriter_prompt,
        schema=rewriter_schema,
        prepared=prepared,
        false_claims=false_claims,
    )
    writer_errors: list[dict[str, Any]] = []
    writer_content: str | None = None
    for writer_attempt in range(attempts):
        messages = copy.deepcopy(writer_messages)
        if writer_errors:
            messages.extend(
                _schema_retry_context(
                    prior_response=writer_content,
                    error=writer_errors[-1]["error"],
                    label="FGSRP IR rewriter schema",
                    directive="Return corrected strict JSON only.",
                )
            )
        try:
            writer_content, writer_request_audit = engine._request(
                args,
                messages,
                seed=seed_base + IR_REWRITER_SEED_OFFSET + writer_attempt,
                max_tokens=1800,
            )
        except Exception as exc:
            _downgrade_ir_false_claims(ledger)
            protocol["verifier"]["claim_ledger"] = ledger
            protocol["ir_evaluation"]["writer_status"] = (
                "abstained_transport_error"
            )
            protocol["ir_evaluation"]["writer_error_type"] = type(exc).__name__
            protocol["deterministic_gate"] = []
            protocol["adjudicator"] = None
            protocol["adjudicator_input_map"] = []
            return [], protocol
        try:
            repairs, writer_audit = fgsrp_ir.normalize_rewriter_response(
                writer_content,
                false_claims=false_claims,
                candidates=prepared.candidates,
                evidence=evidence,
            )
        except ValueError as exc:
            writer_errors.append(
                {
                    "attempt": writer_attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "request": writer_request_audit,
                    "response_sha256": engine._sha256_text(writer_content),
                }
            )
            if writer_attempt + 1 == attempts:
                _downgrade_ir_false_claims(ledger)
                protocol["verifier"]["claim_ledger"] = ledger
                protocol["ir_evaluation"]["writer_status"] = (
                    "abstained_invalid_response"
                )
                protocol["ir_evaluation"]["writer_errors"] = writer_errors
                protocol["deterministic_gate"] = []
                protocol["adjudicator"] = None
                protocol["adjudicator_input_map"] = []
                return [], protocol
            continue
        break

    repair_map = {
        (item["chunk_id"], item["original_claim"]): item for item in repairs
    }
    for sentence in ledger:
        for claim in sentence["claims"]:
            if claim["disposition"] != "replace_contradicted":
                continue
            repair = repair_map[(sentence["chunk_id"], claim["original_claim"])]
            claim["replacement_claim"] = repair["replacement_claim"]
    protocol["verifier"]["proposals"] = repairs
    decisions = []
    host_decisions = []
    for index, repair in enumerate(repairs):
        decision = {
            "repair_index": index,
            "chunk_id": repair["chunk_id"],
            "original_claim": repair["original_claim"],
            "replacement_claim": repair["replacement_claim"],
            "relation": "contradicted",
            "span_is_smallest_complete_clause": True,
            "selected_facts_entail_claim_negation": True,
            "replacement_claim_entailed_by_selected_facts": True,
            "replacement_preserves_all_other_meaning": True,
            "accepted": True,
        }
        decisions.append(decision)
        host_decisions.append(
            {
                "repair_index": index,
                "accepted": True,
                "reason": "canonical_ir_gt_contradiction",
                "llm_gate_accepted": None,
                "effective_accepted": True,
                "adjudicator_skipped": True,
                "contradicting_fact_ids": repair["replacement_fact_ids"],
                "entailed_fact_ids": repair["replacement_fact_ids"],
            }
        )
    writer_request_audit["schema_attempt"] = writer_attempt
    writer_request_audit["prior_schema_errors"] = writer_errors
    protocol["deterministic_gate"] = host_decisions
    protocol["adjudicator_input_map"] = []
    protocol["adjudicator"] = {
        "kind": "ir_rewriter_not_semantic_judge",
        "request": writer_request_audit,
        "response": writer_content,
        "protocol": writer_audit,
        "decisions": decisions,
    }
    return repairs, protocol


def _query_repairs(
    args: Any,
    *,
    task_description: str,
    prepared: PreparedCarrier,
    entity_registry: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    trial: int,
    turn: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_path = Path(
        os.environ.get(
            "CAPX_FGSRP_VERIFIER_PROMPT_PATH",
            Path(__file__).with_name("fgsrp_verifier_prompt.md"),
        )
    )
    schema_path = Path(
        os.environ.get(
            "CAPX_FGSRP_VERIFIER_SCHEMA_PATH",
            Path(__file__).with_name("fgsrp_verifier_schema.json"),
        )
    )
    gate_prompt_path = Path(
        os.environ.get(
            "CAPX_FGSRP_ADJUDICATOR_PROMPT_PATH",
            Path(__file__).with_name("fgsrp_adjudicator_prompt.md"),
        )
    )
    gate_schema_path = Path(
        os.environ.get(
            "CAPX_FGSRP_ADJUDICATOR_SCHEMA_PATH",
            Path(__file__).with_name("fgsrp_adjudicator_schema.json"),
        )
    )
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = _load_json(schema_path)
    gate_prompt = gate_prompt_path.read_text(encoding="utf-8")
    gate_schema = _load_json(gate_schema_path)
    protocol = {
        "verifier_prompt_sha256": engine._sha256_text(prompt),
        "verifier_schema_sha256": engine._sha256_bytes(schema_path.read_bytes()),
        "adjudicator_prompt_sha256": engine._sha256_text(gate_prompt),
        "adjudicator_schema_sha256": engine._sha256_bytes(
            gate_schema_path.read_bytes()
        ),
        "known_false_claims": [],
        "atomic_refinement": None,
    }

    if not prepared.candidates:
        return [], {**protocol, "verifier": None, "adjudicator": None}

    seed_base = int(os.environ.get("CAPX_GLOBAL_SEED", "0")) + trial * 1000 + turn
    verifier_messages = _verifier_messages(
        prompt=prompt,
        schema=schema,
        task_description=task_description,
        prepared=prepared,
        entity_registry=entity_registry,
        evidence=evidence,
    )
    verifier_attempts = int(os.environ.get("CAPX_FGSRP_SCHEMA_ATTEMPTS", "2"))
    if verifier_attempts < 1:
        raise ValueError("FGSRP verifier attempts must be positive")
    verifier_errors: list[dict[str, Any]] = []
    content: str | None = None
    for attempt in range(verifier_attempts):
        messages = copy.deepcopy(verifier_messages)
        if verifier_errors:
            messages.extend(
                _schema_retry_context(
                    prior_response=content,
                    error=verifier_errors[-1]["error"],
                    label="FGSRP schema",
                    directive="Return corrected strict JSON only.",
                )
            )
        try:
            content, verifier_audit = engine._request(
                args,
                messages,
                seed=seed_base + VERIFIER_SEED_OFFSET + attempt,
                max_tokens=2200,
            )
        except Exception as exc:
            protocol["verifier"] = {
                "request": {
                    "seed": seed_base + VERIFIER_SEED_OFFSET + attempt,
                    "schema_attempt": attempt,
                    "prior_schema_errors": verifier_errors,
                    "error_type": type(exc).__name__,
                },
                "response": None,
                "protocol": {
                    "status": "abstained_transport_error",
                    "reason": type(exc).__name__,
                },
                "proposals": [],
            }
            protocol["adjudicator"] = None
            protocol["deterministic_gate"] = []
            return [], protocol
        try:
            proposals, verifier_protocol, claim_ledger = _normalize_claim_ledger(
                content,
                prepared=prepared,
                evidence=evidence,
            )
        except ValueError as exc:
            verifier_errors.append(
                {
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "request": verifier_audit,
                    "response_sha256": engine._sha256_text(content),
                }
            )
            if attempt + 1 == verifier_attempts:
                protocol["verifier"] = {
                    "request": {
                        **verifier_audit,
                        "schema_attempt": attempt,
                        "prior_schema_errors": verifier_errors,
                    },
                    "response": content,
                    "protocol": {
                        "status": "abstained_invalid_response",
                        "reason": verifier_errors[-1]["error"],
                    },
                    "proposals": [],
                }
                protocol["adjudicator"] = None
                protocol["deterministic_gate"] = []
                return [], protocol
            continue
        break
    verifier_audit["schema_attempt"] = attempt
    verifier_audit["prior_schema_errors"] = verifier_errors
    protocol["verifier"] = {
        "request": verifier_audit,
        "response": content,
        "protocol": verifier_protocol,
        "proposals": proposals,
        "claim_ledger": claim_ledger,
    }
    if not proposals:
        protocol["adjudicator"] = None
        return [], protocol

    # The host certificate is a necessary condition for acceptance. Apply it
    # before the private LLM gate so an already-rejected proposal cannot cause
    # a needless provider call or abort the action episode.
    host_decisions = []
    host_certified: list[tuple[int, dict[str, Any]]] = []
    for proposal_index, repair in enumerate(proposals):
        certificate = _deterministic_contradiction_certificate(repair, evidence)
        host_decisions.append(
            {
                "repair_index": proposal_index,
                "llm_gate_accepted": None,
                "adjudicator_skipped": not certificate["accepted"],
                **certificate,
            }
        )
        if certificate["accepted"]:
            host_certified.append((proposal_index, repair))
    protocol["deterministic_gate"] = host_decisions
    if not host_certified:
        protocol["adjudicator"] = None
        protocol["adjudicator_input_map"] = []
        return [], protocol

    gate_repairs = [repair for _, repair in host_certified]
    protocol["adjudicator_input_map"] = [
        {
            "adjudicator_repair_index": gate_index,
            "verifier_repair_index": proposal_index,
        }
        for gate_index, (proposal_index, _) in enumerate(host_certified)
    ]

    gate_messages = _adjudicator_messages(
        prompt=gate_prompt,
        schema=gate_schema,
        prepared=prepared,
        evidence=evidence,
        repairs=gate_repairs,
    )
    gate_attempts = int(os.environ.get("CAPX_FGSRP_GATE_ATTEMPTS", "2"))
    if gate_attempts < 1:
        raise ValueError("FGSRP gate attempts must be positive")
    gate_errors: list[dict[str, Any]] = []
    gate_content: str | None = None
    for attempt in range(gate_attempts):
        messages = copy.deepcopy(gate_messages)
        if gate_errors:
            messages.extend(
                _schema_retry_context(
                    prior_response=gate_content,
                    error=gate_errors[-1]["error"],
                    label="FGSRP gate schema",
                    directive="Return corrected strict JSON only.",
                )
            )
        try:
            gate_content, gate_audit = engine._request(
                args,
                messages,
                seed=seed_base + ADJUDICATOR_SEED_OFFSET + attempt,
                max_tokens=2200,
            )
        except Exception as exc:
            protocol["adjudicator"] = {
                "request": {
                    "seed": seed_base + ADJUDICATOR_SEED_OFFSET + attempt,
                    "schema_attempt": attempt,
                    "prior_schema_errors": gate_errors,
                    "error_type": type(exc).__name__,
                },
                "response": None,
                "protocol": {
                    "status": "abstained_transport_error",
                    "reason": type(exc).__name__,
                },
                "decisions": [],
            }
            return [], protocol
        try:
            decisions, gate_protocol = engine._normalize_adjudications(
                gate_content,
                proposed_repairs=gate_repairs,
                expected_schema_version=ADJUDICATOR_SCHEMA_VERSION,
            )
        except ValueError as exc:
            gate_errors.append(
                {
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "request": gate_audit,
                    "response_sha256": engine._sha256_text(gate_content),
                }
            )
            if attempt + 1 == gate_attempts:
                protocol["adjudicator"] = {
                    "request": {
                        **gate_audit,
                        "schema_attempt": attempt,
                        "prior_schema_errors": gate_errors,
                    },
                    "response": gate_content,
                    "protocol": {
                        "status": "abstained_invalid_response",
                        "reason": gate_errors[-1]["error"],
                    },
                    "decisions": [],
                }
                protocol["deterministic_gate"] = []
                return [], protocol
            continue
        break
    gate_audit["schema_attempt"] = attempt
    gate_audit["prior_schema_errors"] = gate_errors
    accepted = []
    known_false: list[dict[str, Any]] = []
    for decision in decisions:
        proposal_index, repair = host_certified[decision["repair_index"]]
        llm_gate_accepted = decision["accepted"]
        atomic_override = _host_atomic_gate_override(
            decision,
            host_decisions[proposal_index],
        )
        effective_accepted = llm_gate_accepted or atomic_override
        decision["llm_gate_accepted"] = llm_gate_accepted
        decision["host_atomic_override"] = atomic_override
        decision["accepted"] = effective_accepted
        host_decisions[proposal_index]["llm_gate_accepted"] = llm_gate_accepted
        host_decisions[proposal_index]["host_atomic_override"] = atomic_override
        host_decisions[proposal_index]["effective_accepted"] = effective_accepted
        host_decisions[proposal_index]["adjudicator_skipped"] = False
        if effective_accepted:
            accepted.append(repair)
        elif decision["relation"] == "contradicted":
            known_false.append(
                {
                    "verifier_repair_index": proposal_index,
                    "chunk_id": repair["chunk_id"],
                    "original_claim": repair["original_claim"],
                    "replacement_claim": repair["replacement_claim"],
                    "replacement_fact_ids": repair["replacement_fact_ids"],
                    "gate_decision": decision,
                    "resolved": False,
                    "resolution": None,
                }
            )
    protocol["adjudicator"] = {
        "request": gate_audit,
        "response": gate_content,
        "protocol": gate_protocol,
        "decisions": decisions,
    }
    protocol["known_false_claims"] = known_false
    if not known_false:
        return accepted, protocol

    refiner_attempts = int(
        os.environ.get("CAPX_FGSRP_ATOMIC_REFINER_ATTEMPTS", "2")
    )
    if refiner_attempts < 1:
        raise ValueError("FGSRP atomic-refiner attempts must be positive")
    refinement_messages = _atomic_refiner_messages(
        verifier_messages=verifier_messages,
        known_false=known_false,
    )
    refinement_errors: list[dict[str, Any]] = []
    refined_repairs: list[dict[str, Any]] = []
    refinement_content: str | None = None
    refinement_audit: dict[str, Any] | None = None
    refinement_protocol: dict[str, Any] | None = None
    refinement_parents: list[int] = []
    refinement_certificates: list[dict[str, Any]] = []
    for refinement_attempt in range(refiner_attempts):
        messages = copy.deepcopy(refinement_messages)
        if refinement_errors:
            messages.extend(
                _schema_retry_context(
                    prior_response=refinement_content,
                    error=refinement_errors[-1]["error"],
                    label="atomic refinement",
                    directive=(
                        "Return one exact atomic repair for every known-false "
                        "claim and strict JSON only."
                    ),
                )
            )
        try:
            refinement_content, refinement_audit = engine._request(
                args,
                messages,
                seed=(
                    seed_base
                    + ATOMIC_REFINER_SEED_OFFSET
                    + refinement_attempt
                ),
                max_tokens=2200,
            )
        except Exception as exc:
            protocol["atomic_refinement"] = {
                "status": "abstained_transport_error",
                "request": {
                    "seed": (
                        seed_base
                        + ATOMIC_REFINER_SEED_OFFSET
                        + refinement_attempt
                    ),
                    "schema_attempt": refinement_attempt,
                    "prior_errors": refinement_errors,
                    "error_type": type(exc).__name__,
                },
                "response": None,
                "repairs": [],
            }
            return accepted, protocol
        try:
            parsed, refinement_protocol = engine._normalize_repairs(
                refinement_content,
                candidates=prepared.candidates,
                fact_catalog=evidence,
                expected_schema_version=VERIFIER_SCHEMA_VERSION,
            )
            parents = [_refinement_parent(item, known_false) for item in parsed]
            if any(parent is None for parent in parents):
                raise ValueError(
                    "Atomic refinement is not uniquely bound to a known-false claim"
                )
            if (
                len(parents) != len(known_false)
                or set(parents) != set(range(len(known_false)))
            ):
                raise ValueError("Atomic refinement did not cover every false claim")
            certificates = [
                _deterministic_contradiction_certificate(item, evidence)
                for item in parsed
            ]
            if any(not item["accepted"] for item in certificates):
                reasons = [
                    item["reason"] for item in certificates if not item["accepted"]
                ]
                raise ValueError(
                    "Atomic refinement lacks a host contradiction certificate: "
                    + ", ".join(reasons)
                )
            refined_repairs = parsed
            refinement_parents = [int(parent) for parent in parents]
            refinement_certificates = certificates
        except ValueError as exc:
            refinement_errors.append(
                {
                    "attempt": refinement_attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "request": refinement_audit,
                    "response_sha256": engine._sha256_text(refinement_content),
                }
            )
            if refinement_attempt + 1 == refiner_attempts:
                protocol["atomic_refinement"] = {
                    "status": "unresolved_invalid_response",
                    "request": {
                        **refinement_audit,
                        "schema_attempt": refinement_attempt,
                        "prior_errors": refinement_errors,
                    },
                    "response": refinement_content,
                    "repairs": [],
                }
                return accepted, protocol
            continue
        break

    refinement_audit["schema_attempt"] = refinement_attempt
    refinement_audit["prior_errors"] = refinement_errors
    refined_gate_messages = _adjudicator_messages(
        prompt=gate_prompt,
        schema=gate_schema,
        prepared=prepared,
        evidence=evidence,
        repairs=refined_repairs,
    )
    refined_gate_errors: list[dict[str, Any]] = []
    refined_decisions: list[dict[str, Any]] = []
    refined_gate_content: str | None = None
    refined_gate_audit: dict[str, Any] | None = None
    refined_gate_protocol: dict[str, Any] | None = None
    for refined_gate_attempt in range(gate_attempts):
        messages = copy.deepcopy(refined_gate_messages)
        if refined_gate_errors:
            messages.extend(
                _schema_retry_context(
                    prior_response=refined_gate_content,
                    error=refined_gate_errors[-1]["error"],
                    label="atomic gate response",
                    directive="Return corrected strict JSON only.",
                )
            )
        try:
            refined_gate_content, refined_gate_audit = engine._request(
                args,
                messages,
                seed=(
                    seed_base
                    + ATOMIC_REFINER_GATE_SEED_OFFSET
                    + refined_gate_attempt
                ),
                max_tokens=2200,
            )
        except Exception as exc:
            protocol["atomic_refinement"] = {
                "status": "abstained_gate_transport_error",
                "request": refinement_audit,
                "response": refinement_content,
                "repairs": refined_repairs,
                "gate": {
                    "seed": (
                        seed_base
                        + ATOMIC_REFINER_GATE_SEED_OFFSET
                        + refined_gate_attempt
                    ),
                    "error_type": type(exc).__name__,
                },
            }
            return accepted, protocol
        try:
            refined_decisions, refined_gate_protocol = (
                engine._normalize_adjudications(
                    refined_gate_content,
                    proposed_repairs=refined_repairs,
                    expected_schema_version=ADJUDICATOR_SCHEMA_VERSION,
                )
            )
        except ValueError as exc:
            refined_gate_errors.append(
                {
                    "attempt": refined_gate_attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "request": refined_gate_audit,
                    "response_sha256": engine._sha256_text(refined_gate_content),
                }
            )
            if refined_gate_attempt + 1 == gate_attempts:
                protocol["atomic_refinement"] = {
                    "status": "unresolved_invalid_gate_response",
                    "request": refinement_audit,
                    "response": refinement_content,
                    "repairs": refined_repairs,
                    "gate": {
                        "request": {
                            **refined_gate_audit,
                            "schema_attempt": refined_gate_attempt,
                            "prior_errors": refined_gate_errors,
                        },
                        "response": refined_gate_content,
                    },
                }
                return accepted, protocol
            continue
        break

    refined_gate_audit["schema_attempt"] = refined_gate_attempt
    refined_gate_audit["prior_errors"] = refined_gate_errors
    refined_accepted: list[dict[str, Any]] = []
    for refined_index, decision in enumerate(refined_decisions):
        parent = refinement_parents[refined_index]
        llm_gate_accepted = decision["accepted"]
        atomic_override = _host_atomic_gate_override(
            decision,
            refinement_certificates[refined_index],
        )
        effective_accepted = llm_gate_accepted or atomic_override
        decision["llm_gate_accepted"] = llm_gate_accepted
        decision["host_atomic_override"] = atomic_override
        decision["accepted"] = effective_accepted
        if effective_accepted:
            refined_accepted.append(refined_repairs[refined_index])
            known_false[parent]["resolved"] = True
            known_false[parent]["resolution"] = {
                "atomic_repair_index": refined_index,
                "original_claim": refined_repairs[refined_index]["original_claim"],
                "replacement_claim": refined_repairs[refined_index][
                    "replacement_claim"
                ],
            }
    protocol["known_false_claims"] = known_false
    unresolved = [item for item in known_false if item["resolved"] is not True]
    protocol["atomic_refinement"] = {
        "status": "resolved" if not unresolved else "unresolved_gate_rejection",
        "request": refinement_audit,
        "response": refinement_content,
        "protocol": refinement_protocol,
        "repairs": refined_repairs,
        "parent_indices": refinement_parents,
        "host_certificates": refinement_certificates,
        "gate": {
            "request": refined_gate_audit,
            "response": refined_gate_content,
            "protocol": refined_gate_protocol,
            "decisions": refined_decisions,
        },
    }
    return [*accepted, *refined_accepted], protocol


def run_fgsrp(
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
    """Return the native request with only GT-contradicted report spans changed."""
    trial_root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    turn_root = trial_root / f"turn_{turn:02d}"
    turn_root.mkdir(parents=True, exist_ok=False)
    state = _load_json(trial_root / STATE_FILE)
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("task") != task
        or state.get("trial") != trial
    ):
        raise ValueError("FGSRP trial-state provenance mismatch")
    snapshots = state.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != turn:
        raise ValueError("FGSRP evidence history is not turn-synchronous")

    provider = _provider_for(task)
    before = snapshots[-1]["evidence"]
    after = provider.capture(env, task=task)
    evidence = _temporal_evidence(before, after)
    entity_registry = after["entity_registry"]
    carrier_name = os.environ.get("CAPX_FGSRP_CARRIER", "m2_comments")
    adapter = _carrier(carrier_name)
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
    ontology = after.get("predicate_ontology")
    if not isinstance(ontology, dict) or not ontology:
        raise ValueError("FGSRP evidence provider supplied no predicate ontology")
    repairs, protocol = _query_ir_repairs(
        args,
        task_description=task_description,
        prepared=prepared,
        entity_registry=entity_registry,
        ontology=ontology,
        evidence=evidence,
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
        raise RuntimeError("FGSRP no-op changed the native action request")
    if applied.active_repair_count > 0 and corrected_hash == native_hash:
        raise RuntimeError("FGSRP accepted repair did not change the action request")
    if engine._non_text_structure(native_decision_prompt) != engine._non_text_structure(
        applied.corrected_prompt
    ):
        raise RuntimeError("FGSRP changed action-request roles or media")

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
        "corrected_prompt_structure": engine._prompt_structure(
            applied.corrected_prompt
        ),
        "non_text_structure_identical": True,
        "native_surface_sha256": engine._sha256_text(prepared.source_text),
        "corrected_surface_sha256": engine._sha256_text(applied.corrected_source),
        "candidate_chunks": [asdict(item) for item in prepared.candidates],
        "candidate_count": len(prepared.candidates),
        "before_evidence_sha256": engine._sha256_json(before),
        "after_evidence_sha256": engine._sha256_json(after),
        "temporal_evidence_sha256": engine._sha256_json(evidence),
        "private_entity_registry": entity_registry,
        "private_temporal_evidence": evidence,
        "repairs": repairs,
        "repair_count": len(repairs),
        "active_repair_count": applied.active_repair_count,
        "applied_patches": applied.patches,
        "carrier_state": applied.state_record,
        "verification_protocol": protocol,
        "no_op_prompt_byte_equivalent": (
            corrected_hash == native_hash
            if applied.active_repair_count == 0
            else None
        ),
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
