from __future__ import annotations

import json
from types import SimpleNamespace

from capx.verification import envelope_replacement as engine
from capx.verification import fgsrp
from capx.verification import fgsrp_nl


def _prepared(text: str = "The red cube moved. It remains on the table.") -> fgsrp.PreparedCarrier:
    candidate = engine.CandidateChunk(
        chunk_id="report.turn_01",
        source="visual_differencing_report",
        start=0,
        end=len(text),
        text=text,
        sha256=engine._sha256_text(text),
    )
    return fgsrp.PreparedCarrier(
        kind="m3_visual_difference_report",
        source_text=text,
        candidates=[candidate],
        metadata={},
    )


def _response(decisions: list[dict]) -> str:
    return json.dumps(
        {
            "schema_version": fgsrp_nl.VERIFIER_SCHEMA_VERSION,
            "decisions": decisions,
            "audit": {
                "all_candidate_chunks_checked": True,
                "true_text_preserved": True,
                "only_directly_contradicted_physical_claims_replaced": True,
                "contains_action_advice": False,
            },
        }
    )


def test_keep_only_is_a_byte_identical_noop() -> None:
    prepared = _prepared()
    repairs, audit = fgsrp_nl._normalize_response(
        _response(
            [
                {
                    "chunk_id": "report.turn_01",
                    "original_span": "The red cube moved.",
                    "verdict": "KEEP",
                    "replacement": None,
                    "evidence_quotes": [],
                }
            ]
        ),
        prepared=prepared,
        statements=["Immediately after execution, the red cube is on the table."],
    )
    updated, patches = engine._apply_source_repairs(
        prepared.source_text,
        candidates=prepared.candidates,
        repairs=repairs,
        fact_catalog={},
    )
    assert repairs == []
    assert patches == []
    assert updated.encode() == prepared.source_text.encode()
    assert audit["accepted_replacement_count"] == 0


def test_invalid_claim_does_not_suppress_valid_replacement() -> None:
    prepared = _prepared()
    statement = "Immediately after execution, the red cube is separated from the table."
    repairs, audit = fgsrp_nl._normalize_response(
        _response(
            [
                {
                    "chunk_id": "missing",
                    "original_span": "bad claim",
                    "verdict": "REPLACE",
                    "replacement": "bad replacement",
                    "evidence_quotes": [statement],
                },
                {
                    "chunk_id": "report.turn_01",
                    "original_span": "It remains on the table.",
                    "verdict": "REPLACE",
                    "replacement": "The red cube is separated from the table.",
                    "evidence_quotes": [statement],
                },
            ]
        ),
        prepared=prepared,
        statements=[statement],
    )
    assert len(repairs) == 1
    assert repairs[0]["original_claim"] == "It remains on the table."
    assert audit["rejected_decisions"][0]["reason"] == "unknown_chunk"


def test_broad_keep_does_not_suppress_a_local_false_claim() -> None:
    prepared = _prepared()
    statement = "Immediately after execution, the red cube is separated from the table."
    repairs, _ = fgsrp_nl._normalize_response(
        _response(
            [
                {
                    "chunk_id": "report.turn_01",
                    "original_span": prepared.source_text,
                    "verdict": "KEEP",
                    "replacement": None,
                    "evidence_quotes": [],
                },
                {
                    "chunk_id": "report.turn_01",
                    "original_span": "It remains on the table.",
                    "verdict": "REPLACE",
                    "replacement": "The red cube is separated from the table.",
                    "evidence_quotes": [statement],
                },
            ]
        ),
        prepared=prepared,
        statements=[statement],
    )
    assert len(repairs) == 1


def test_host_has_no_alias_or_predicate_semantic_gate() -> None:
    prepared = _prepared("The crimson object remains supported by the other shape.")
    statement = "Immediately after execution, the crimson object is separated from the other shape."
    repairs, _ = fgsrp_nl._normalize_response(
        _response(
            [
                {
                    "chunk_id": "report.turn_01",
                    "original_span": "remains supported by the other shape",
                    "verdict": "REPLACE",
                    "replacement": "is separated from the other shape",
                    "evidence_quotes": [statement],
                }
            ]
        ),
        prepared=prepared,
        statements=[statement],
    )
    assert len(repairs) == 1


def test_numeric_and_advice_replacements_are_rejected_independently() -> None:
    prepared = _prepared("The object is at x 0.2. It is on the table.")
    statement = "Immediately after execution, the object is separated from the table."
    repairs, audit = fgsrp_nl._normalize_response(
        _response(
            [
                {
                    "chunk_id": "report.turn_01",
                    "original_span": "The object is at x 0.2.",
                    "verdict": "REPLACE",
                    "replacement": "The object is at x 0.3.",
                    "evidence_quotes": [statement],
                },
                {
                    "chunk_id": "report.turn_01",
                    "original_span": "It is on the table.",
                    "verdict": "REPLACE",
                    "replacement": "The robot should grasp it.",
                    "evidence_quotes": [statement],
                },
            ]
        ),
        prepared=prepared,
        statements=[statement],
    )
    assert repairs == []
    assert [item["reason"] for item in audit["rejected_decisions"]] == [
        "numeric_content_is_immutable",
        "replacement_contains_advice_code_or_privileged_label",
    ]


def test_overlap_rejects_only_the_later_decision() -> None:
    prepared = _prepared("The cube remains on the table.")
    statement = "Immediately after execution, the cube is separated from the table."
    repairs, audit = fgsrp_nl._normalize_response(
        _response(
            [
                {
                    "chunk_id": "report.turn_01",
                    "original_span": "remains on the table",
                    "verdict": "REPLACE",
                    "replacement": "is separated from the table",
                    "evidence_quotes": [statement],
                },
                {
                    "chunk_id": "report.turn_01",
                    "original_span": "on the table",
                    "verdict": "REPLACE",
                    "replacement": "separated from the table",
                    "evidence_quotes": [statement],
                },
            ]
        ),
        prepared=prepared,
        statements=[statement],
    )
    assert len(repairs) == 1
    assert audit["rejected_decisions"][0]["reason"] == "overlapping_decision"


def test_private_packet_omits_aliases_ontology_and_simulator_values() -> None:
    prepared = _prepared()
    statements = ["Immediately after execution, the red cube is on the table."]
    messages = fgsrp_nl._messages(
        prompt="private verifier",
        schema={"schema_version": "test"},
        task_description="Stack the cube.\nAPIs: secret_api(x=0.1)",
        prepared=prepared,
        statements=statements,
    )
    packet_text = messages[1]["content"]
    assert statements[0] in packet_text
    assert "entity_registry" not in packet_text
    assert "predicate_ontology" not in packet_text
    assert "aliases" not in packet_text
    assert "secret_api" not in packet_text


def test_m3_carrier_changes_only_the_native_report_text() -> None:
    report = "The red cube remains on the table."
    prepared = fgsrp.M3VisualDifferenceCarrier().prepare(
        {"turn": 1, "visual_differencing_feedback": report},
        {},
    )
    prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"before\n{report}\nafter"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
    ]
    state = {"m3_reports": []}
    applied = fgsrp.M3VisualDifferenceCarrier().apply(
        prepared,
        repairs=[
            {
                "chunk_id": prepared.candidates[0].chunk_id,
                "original_claim": "remains on the table",
                "replacement_claim": "is separated from the table",
                "replacement_fact_ids": [],
            }
        ],
        evidence={},
        context={"turn": 1, "native_decision_prompt": prompt},
        state=state,
    )
    assert "is separated from the table" in applied.corrected_prompt[1]["content"][0]["text"]
    assert applied.corrected_prompt[1]["content"][1] == prompt[1]["content"][1]
    assert [item["role"] for item in applied.corrected_prompt] == ["system", "user"]


def test_query_uses_private_llm_as_semantic_authority(monkeypatch) -> None:
    prepared = _prepared()
    statement = "Immediately after execution, the red cube is separated from the table."

    def fake_request(args, messages, *, seed, max_tokens):
        del args, messages, max_tokens
        return _response(
            [
                {
                    "chunk_id": "report.turn_01",
                    "original_span": "It remains on the table.",
                    "verdict": "REPLACE",
                    "replacement": "The red cube is separated from the table.",
                    "evidence_quotes": [statement],
                }
            ]
        ), {"seed": seed}

    monkeypatch.setattr(engine, "_request", fake_request)
    repairs, protocol = fgsrp_nl._query_repairs(
        SimpleNamespace(),
        task_description="Stack the cube.",
        prepared=prepared,
        statements=[statement],
        trial=0,
        turn=1,
    )
    assert len(repairs) == 1
    assert protocol["host_semantic_judgment"] is False
    assert protocol["host_alias_matching"] is False
    assert protocol["host_predicate_matching"] is False
