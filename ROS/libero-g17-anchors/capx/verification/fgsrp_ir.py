"""Canonical semantic-IR alignment for FGSRP native report claims."""

from __future__ import annotations

import copy
import json
from typing import Any

from capx.verification import envelope_replacement as engine

PARSER_SCHEMA_VERSION = "capx-fgsrp-ir-parser-v2"
REWRITER_SCHEMA_VERSION = "capx-fgsrp-ir-rewriter-v1"

CUBE_PREDICATE_ONTOLOGY = {
    "target_grasp": {
        "description": (
            "attachment relation between target object and executor; this is "
            "not gripper aperture, so open/closed gripper claims are unmapped"
        ),
        "values": [
            "target_grasped_by_robot",
            "target_not_grasped_by_robot",
        ],
        "subject_entity_id": "object.target",
        "required_entity_ids": ["object.target", "executor.robot_gripper"],
    },
    "relative_x": {
        "description": "target center relative to goal x extent",
        "values": [
            "target_center_negative_x_of_goal_band",
            "target_center_within_goal_x_band",
            "target_center_positive_x_of_goal_band",
        ],
        "subject_entity_id": "object.target",
        "required_entity_ids": ["object.target", "object.goal"],
    },
    "relative_y": {
        "description": "target center relative to goal y extent",
        "values": [
            "target_center_negative_y_of_goal_band",
            "target_center_within_goal_y_band",
            "target_center_positive_y_of_goal_band",
        ],
        "subject_entity_id": "object.target",
        "required_entity_ids": ["object.target", "object.goal"],
    },
    "relative_z": {
        "description": "target center relative to goal height extent",
        "values": [
            "target_below_goal_height_band",
            "target_within_goal_height_band",
            "target_above_goal_height_band",
        ],
        "subject_entity_id": "object.target",
        "required_entity_ids": ["object.target", "object.goal"],
    },
    "footprint": {
        "description": "target center inclusion in goal horizontal footprint",
        "values": [
            "target_center_inside_goal_footprint",
            "target_center_outside_goal_footprint",
        ],
        "subject_entity_id": "object.target",
        "required_entity_ids": ["object.target", "object.goal"],
    },
    "target_goal_contact": {
        "description": "physical contact relation between target and goal objects",
        "values": [
            "target_contacting_goal",
            "target_separated_from_goal",
        ],
        "subject_entity_id": "object.target",
        "required_entity_ids": ["object.target", "object.goal"],
    },
    "target_goal_support": {
        "description": (
            "support topology between target and goal objects; literal claims "
            "that the target is on top of, resting on, or supported by the goal "
            "map here and are not reducible to relative_z alone"
        ),
        "values": [
            "target_supported_by_goal",
            "target_not_supported_by_goal",
        ],
        "subject_entity_id": "object.target",
        "required_entity_ids": ["object.target", "object.goal"],
    },
    "target_table_contact": {
        "description": "physical contact relation between target and table",
        "values": [
            "target_contacting_table",
            "target_separated_from_table",
        ],
        "subject_entity_id": "object.target",
        "required_entity_ids": ["object.target", "support.table"],
    },
}

for _predicate_definition in CUBE_PREDICATE_ONTOLOGY.values():
    _predicate_definition["transition_subject_entity_ids"] = [
        _predicate_definition["subject_entity_id"],
        "scene.environment",
    ]


def _assertion_evidence_id(assertion: dict[str, Any]) -> str:
    return f"{assertion['temporal_scope']}.{assertion['predicate']}"


def _normalize_assertion(
    raw: Any,
    *,
    ontology: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    entity_registry: dict[str, Any] | None = None,
    original_claim: str | None = None,
) -> tuple[dict[str, Any], bool]:
    if not isinstance(raw, dict):
        raise ValueError("FGSRP IR assertion must be an object")
    scope = raw.get("temporal_scope")
    predicate = raw.get("predicate")
    if scope not in {"before", "after", "transition"}:
        raise ValueError("FGSRP IR assertion has invalid temporal scope")
    if not isinstance(predicate, str) or predicate not in ontology:
        raise ValueError("FGSRP IR assertion references unknown predicate")
    subject_entity_id = raw.get("subject_entity_id")
    subject_text = raw.get("subject_text")
    expected_subject = ontology[predicate].get("subject_entity_id")
    if expected_subject is not None:
        allowed_subjects = {expected_subject}
        if scope == "transition":
            allowed_subjects.update(
                ontology[predicate].get("transition_subject_entity_ids", [])
            )
        if subject_entity_id not in allowed_subjects:
            raise ValueError("FGSRP IR assertion predicate/subject mismatch")
        if not isinstance(subject_text, str) or not subject_text.strip():
            raise ValueError("FGSRP IR assertion lacks an exact subject span")
        if not isinstance(original_claim, str) or subject_text not in original_claim:
            raise ValueError("FGSRP IR assertion subject is not a claim substring")
        if not isinstance(entity_registry, dict) or subject_entity_id not in entity_registry:
            raise ValueError("FGSRP IR assertion references unknown subject entity")
        entity = entity_registry[subject_entity_id]
        aliases = entity.get("aliases", [])
        if not isinstance(aliases, list):
            raise ValueError("FGSRP entity aliases must be a list")
        allowed_subjects = {
            str(entity.get("public_name", "")).strip().lower(),
            *(str(alias).strip().lower() for alias in aliases),
        }
        if subject_text.strip().lower() not in allowed_subjects:
            raise ValueError("FGSRP IR subject span is not an alias of its entity")
    referenced_entity_ids = raw.get("referenced_entity_ids")
    required_entity_ids = ontology[predicate].get("required_entity_ids")
    if (
        not isinstance(referenced_entity_ids, list)
        or not all(isinstance(item, str) for item in referenced_entity_ids)
        or referenced_entity_ids != required_entity_ids
    ):
        raise ValueError("FGSRP IR assertion entity arguments do not match predicate")
    values = ontology[predicate].get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("FGSRP predicate ontology has no finite value domain")
    evidence_id = f"{scope}.{predicate}"
    if evidence_id not in evidence:
        raise ValueError("FGSRP IR assertion lacks synchronized GT evidence")

    normalized = {
        "temporal_scope": scope,
        "predicate": predicate,
        "subject_entity_id": subject_entity_id,
        "subject_text": subject_text,
        "referenced_entity_ids": referenced_entity_ids,
        "asserted_value": raw.get("asserted_value"),
        "asserted_changed": raw.get("asserted_changed"),
        "asserted_before_value": raw.get("asserted_before_value"),
        "asserted_after_value": raw.get("asserted_after_value"),
        "evidence_id": evidence_id,
    }
    if scope in {"before", "after"}:
        temporal_field = (
            "asserted_before_value" if scope == "before" else "asserted_after_value"
        )
        other_temporal_field = (
            "asserted_after_value" if scope == "before" else "asserted_before_value"
        )
        value = normalized["asserted_value"]
        temporal_value = normalized[temporal_field]
        if value is None:
            value = temporal_value
        elif temporal_value is not None and temporal_value != value:
            raise ValueError("FGSRP state assertion has conflicting value fields")
        if value not in values:
            raise ValueError("FGSRP state assertion has non-canonical value")
        if (
            normalized["asserted_changed"] is not None
            or normalized[other_temporal_field] is not None
        ):
            raise ValueError("FGSRP state assertion mixes transition fields")
        normalized["asserted_value"] = value
        normalized["asserted_before_value"] = None
        normalized["asserted_after_value"] = None
        supported = value == evidence[evidence_id].get("value")
    else:
        if normalized["asserted_value"] is not None:
            raise ValueError("FGSRP transition assertion carries a state value")
        changed = normalized["asserted_changed"]
        before_value = normalized["asserted_before_value"]
        after_value = normalized["asserted_after_value"]
        if changed is not None and not isinstance(changed, bool):
            raise ValueError("FGSRP transition changed flag is not boolean")
        if before_value is not None and before_value not in values:
            raise ValueError("FGSRP transition has non-canonical before value")
        if after_value is not None and after_value not in values:
            raise ValueError("FGSRP transition has non-canonical after value")
        if changed is None and before_value is None and after_value is None:
            raise ValueError("FGSRP transition assertion has no semantic content")
        fact = evidence[evidence_id]
        supported = all(
            (
                changed is None or changed == fact.get("changed"),
                before_value is None or before_value == fact.get("before_value"),
                after_value is None or after_value == fact.get("after_value"),
            )
        )
    normalized["gt_supported"] = supported
    return normalized, supported


def normalize_parser_response(
    content: str,
    *,
    prepared: Any,
    ontology: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    entity_registry: dict[str, Any],
    sentence_units: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    value = engine._extract_json_object(content)
    if value.get("schema_version") != PARSER_SCHEMA_VERSION:
        raise ValueError("FGSRP IR parser schema version mismatch")
    raw_ledger = value.get("claim_ledger")
    if not isinstance(raw_ledger, list) or len(raw_ledger) != len(sentence_units):
        raise ValueError("FGSRP IR parser did not cover every sentence exactly once")

    ledger: list[dict[str, Any]] = []
    false_claims: list[dict[str, Any]] = []
    for expected, entry in zip(sentence_units, raw_ledger, strict=True):
        if not isinstance(entry, dict):
            raise ValueError("FGSRP IR ledger entry must be an object")
        for field in ("chunk_id", "sentence_index", "sentence_text"):
            if entry.get(field) != expected[field]:
                raise ValueError(f"FGSRP IR sentence provenance mismatch: {field}")
        raw_claims = entry.get("claims")
        if not isinstance(raw_claims, list) or not raw_claims:
            raise ValueError("FGSRP IR sentence has no parsed claims")
        claims = []
        for raw_claim in raw_claims:
            if not isinstance(raw_claim, dict):
                raise ValueError("FGSRP IR claim must be an object")
            original = raw_claim.get("original_claim")
            claim_type = raw_claim.get("claim_type")
            raw_assertions = raw_claim.get("assertions")
            if (
                not isinstance(original, str)
                or not original.strip()
                or expected["sentence_text"].count(original) != 1
            ):
                raise ValueError("FGSRP IR claim is not an exact sentence substring")
            if claim_type not in {"physical", "non_editable", "unknown"}:
                raise ValueError("FGSRP IR claim type is invalid")
            if not isinstance(raw_assertions, list):
                raise ValueError("FGSRP IR assertions must be a list")
            if claim_type != "physical" and raw_assertions:
                raise ValueError("FGSRP non-physical claim carries assertions")
            if claim_type == "physical" and not raw_assertions:
                raise ValueError("FGSRP physical claim has no semantic assertions")
            assertions = []
            supported = []
            for raw_assertion in raw_assertions:
                assertion, is_supported = _normalize_assertion(
                    raw_assertion,
                    ontology=ontology,
                    evidence=evidence,
                    entity_registry=entity_registry,
                    original_claim=original,
                )
                assertions.append(assertion)
                supported.append(is_supported)
            false_ids = [
                item["evidence_id"]
                for item in assertions
                if item["gt_supported"] is False
            ]
            if len(false_ids) > 4:
                raise ValueError("FGSRP IR claim requires more than four corrections")
            disposition = (
                "replace_contradicted"
                if false_ids
                else (
                    "preserve_supported"
                    if claim_type == "physical"
                    else (
                        "preserve_non_editable"
                        if claim_type == "non_editable"
                        else "preserve_compatible_or_undecidable"
                    )
                )
            )
            normalized_claim = {
                "original_claim": original,
                "claim_type": claim_type,
                "assertions": assertions,
                "disposition": disposition,
                "replacement_claim": None,
                "replacement_fact_ids": false_ids,
            }
            if false_ids:
                false_claims.append(
                    {
                        "false_claim_index": len(false_claims),
                        "chunk_id": expected["chunk_id"],
                        "sentence_text": expected["sentence_text"],
                        "original_claim": original,
                        "assertions": assertions,
                        "replacement_fact_ids": false_ids,
                        "corrected_facts": [evidence[item] for item in false_ids],
                    }
                )
            claims.append(normalized_claim)
        ledger.append({**expected, "claims": claims})

    audit = value.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("FGSRP IR parser audit is missing")
    for field in (
        "all_candidate_chunks_checked",
        "every_sentence_listed_once",
        "every_atomic_claim_parsed",
        "parsed_without_current_gt_values",
        "all_physical_subjects_bound",
    ):
        if audit.get(field) is not True:
            raise ValueError(f"FGSRP IR parser did not certify {field}")
    if audit.get("contains_action_advice") is not False:
        raise ValueError("FGSRP IR parser reported action advice")
    return ledger, false_claims, {
        **copy.deepcopy(audit),
        "sentence_count": len(ledger),
        "claim_count": sum(len(item["claims"]) for item in ledger),
        "false_claim_count": len(false_claims),
    }


def normalize_rewriter_response(
    content: str,
    *,
    false_claims: list[dict[str, Any]],
    candidates: list[Any],
    evidence: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value = engine._extract_json_object(content)
    if value.get("schema_version") != REWRITER_SCHEMA_VERSION:
        raise ValueError("FGSRP IR rewriter schema version mismatch")
    raw_repairs = value.get("repairs")
    if not isinstance(raw_repairs, list) or len(raw_repairs) != len(false_claims):
        raise ValueError("FGSRP IR rewriter did not cover every false claim once")
    projected = []
    for index, (raw, expected) in enumerate(
        zip(raw_repairs, false_claims, strict=True)
    ):
        if not isinstance(raw, dict) or raw.get("false_claim_index") != index:
            raise ValueError("FGSRP IR rewriter false-claim provenance mismatch")
        for field in ("chunk_id", "original_claim", "replacement_fact_ids"):
            if raw.get(field) != expected[field]:
                raise ValueError(f"FGSRP IR rewriter provenance mismatch: {field}")
        projected.append(
            {
                "chunk_id": raw["chunk_id"],
                "original_claim": raw["original_claim"],
                "replacement_claim": raw.get("replacement_claim"),
                "replacement_fact_ids": raw["replacement_fact_ids"],
            }
        )
    synthetic = {
        "schema_version": REWRITER_SCHEMA_VERSION,
        "repairs": projected,
        "audit": {
            "all_candidate_chunks_checked": True,
            "used_only_authoritative_catalog": value.get("audit", {}).get(
                "used_only_supplied_corrected_facts"
            ),
            "contains_action_advice": value.get("audit", {}).get(
                "contains_action_advice"
            ),
        },
    }
    repairs, audit = engine._normalize_repairs(
        json.dumps(synthetic),
        candidates=candidates,
        fact_catalog=evidence,
        expected_schema_version=REWRITER_SCHEMA_VERSION,
    )
    if value.get("audit", {}).get("all_false_claims_rewritten_once") is not True:
        raise ValueError("FGSRP IR rewriter did not certify complete coverage")
    return repairs, audit
