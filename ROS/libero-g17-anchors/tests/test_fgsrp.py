from __future__ import annotations

import json
import tokenize
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest

from capx.verification import envelope_replacement as engine
from capx.verification import fgsrp
from capx.verification import fgsrp_ir


def _fake_env(*, grasped: bool = False) -> SimpleNamespace:
    class FakeSimulator:
        cubeA = SimpleNamespace(size=np.array([0.02, 0.02, 0.02]))
        cubeB = SimpleNamespace(size=np.array([0.025, 0.025, 0.025]))
        cubeA_body_id = 0
        cubeB_body_id = 1
        robots = [SimpleNamespace(gripper=object())]
        sim = SimpleNamespace(
            data=SimpleNamespace(
                body_xpos=np.array([[0.04, 0.0, 0.10], [0.0, 0.0, 0.05]]),
                body_xquat=np.array(
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
                ),
            )
        )

        def _check_grasp(self, *, gripper, object_geoms) -> bool:
            return grasped and object_geoms is self.cubeA

        def check_contact(self, first, second) -> bool:
            return False

    class FakeLowLevel:
        robosuite_env = FakeSimulator()

        def _cube_pose_dict(self, observation):
            return {
                "primary": [*observation["cubeA_pos"], 1.0, 0.0, 0.0, 0.0],
                "secondary": [*observation["cubeB_pos"], 1.0, 0.0, 0.0, 0.0],
            }

    return SimpleNamespace(low_level_env=FakeLowLevel())


def _non_comment_tokens(source: str) -> list[tuple[int, str]]:
    return [
        (token.type, token.string)
        for token in tokenize.generate_tokens(StringIO(source).readline)
        if token.type != tokenize.COMMENT
    ]


def _native_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"prefix\n{text}\nsuffix"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        },
    ]


def _verifier_payload(
    report: str,
    *,
    repairs: list[dict],
    claims: list[dict] | None = None,
    chunk_id: str = "visual_difference.turn_01.report",
) -> str:
    if claims is None:
        if repairs:
            claims = [
                {
                    "original_claim": repair["original_claim"],
                    "disposition": "replace_contradicted",
                    "replacement_claim": repair["replacement_claim"],
                    "replacement_fact_ids": repair["replacement_fact_ids"],
                }
                for repair in repairs
            ]
        else:
            claims = [
                {
                    "original_claim": report,
                    "disposition": "preserve_compatible_or_undecidable",
                    "replacement_claim": None,
                    "replacement_fact_ids": [],
                }
            ]
    return json.dumps(
        {
            "schema_version": fgsrp.VERIFIER_SCHEMA_VERSION,
            "claim_ledger": [
                {
                    "chunk_id": chunk_id,
                    "sentence_index": 0,
                    "sentence_text": report,
                    "claims": claims,
                }
            ],
            "repairs": repairs,
            "audit": {
                "all_candidate_chunks_checked": True,
                "every_sentence_listed_once": True,
                "every_atomic_claim_classified": True,
                "contradicted_claims_match_repairs": True,
                "used_only_authoritative_catalog": True,
                "contains_action_advice": False,
            },
        }
    )


def _ir_parser_payload(report: str, claims: list[dict]) -> str:
    aliases = {
        "object.target": ("the red cube", "red cube", "target", "it"),
        "scene.environment": (
            "the current state of the environment",
            "current state of the environment",
            "the environment",
            "environment",
        ),
    }
    for claim in claims:
        for assertion in claim.get("assertions", []):
            subject = fgsrp_ir.CUBE_PREDICATE_ONTOLOGY[
                assertion["predicate"]
            ]["subject_entity_id"]
            if (
                assertion.get("temporal_scope") == "transition"
                and not any(
                    alias in claim["original_claim"]
                    for alias in aliases[subject]
                )
            ):
                subject = "scene.environment"
            assertion.setdefault("subject_entity_id", subject)
            assertion.setdefault(
                "referenced_entity_ids",
                fgsrp_ir.CUBE_PREDICATE_ONTOLOGY[assertion["predicate"]][
                    "required_entity_ids"
                ],
            )
            assertion.setdefault(
                "subject_text",
                next(
                    alias
                    for alias in aliases[subject]
                    if alias in claim["original_claim"]
                ),
            )
    return json.dumps(
        {
            "schema_version": fgsrp_ir.PARSER_SCHEMA_VERSION,
            "claim_ledger": [
                {
                    "chunk_id": "visual_difference.turn_01.report",
                    "sentence_index": 0,
                    "sentence_text": report,
                    "claims": claims,
                }
            ],
            "audit": {
                "all_candidate_chunks_checked": True,
                "every_sentence_listed_once": True,
                "every_atomic_claim_parsed": True,
                "parsed_without_current_gt_values": True,
                "all_physical_subjects_bound": True,
                "contains_action_advice": False,
            },
        }
    )


def _ir_rewriter_payload(
    *,
    original_claim: str,
    replacement_claim: str,
    fact_ids: list[str],
) -> str:
    return json.dumps(
        {
            "schema_version": fgsrp_ir.REWRITER_SCHEMA_VERSION,
            "repairs": [
                {
                    "false_claim_index": 0,
                    "chunk_id": "visual_difference.turn_01.report",
                    "original_claim": original_claim,
                    "replacement_claim": replacement_claim,
                    "replacement_fact_ids": fact_ids,
                }
            ],
            "audit": {
                "all_false_claims_rewritten_once": True,
                "used_only_supplied_corrected_facts": True,
                "contains_action_advice": False,
            },
        }
    )


def test_temporal_evidence_binds_before_after_and_transition() -> None:
    provider = fgsrp.CubeSimulatorEvidenceProvider()
    before = provider.capture(_fake_env(grasped=False), task="cube_stack")
    after = provider.capture(_fake_env(grasped=True), task="cube_stack")

    evidence = fgsrp._temporal_evidence(before, after)

    assert evidence["before.target_grasp"]["temporal_scope"] == (
        "immediately_before_execution"
    )
    assert evidence["after.target_grasp"]["temporal_scope"] == (
        "immediately_after_execution"
    )
    assert evidence["transition.target_grasp"]["changed"] is True
    assert evidence["transition.relative_x"]["changed"] is False


def test_ir_state_accepts_time_named_value_without_changing_meaning() -> None:
    assertion, supported = fgsrp_ir._normalize_assertion(
        {
            "temporal_scope": "after",
            "predicate": "target_grasp",
            "subject_entity_id": "object.target",
            "subject_text": "red cube",
            "referenced_entity_ids": [
                "object.target",
                "executor.robot_gripper",
            ],
            "asserted_value": None,
            "asserted_changed": None,
            "asserted_before_value": None,
            "asserted_after_value": "target_not_grasped_by_robot",
        },
        ontology=fgsrp_ir.CUBE_PREDICATE_ONTOLOGY,
        evidence={
            "after.target_grasp": {"value": "target_not_grasped_by_robot"}
        },
        entity_registry={
            "object.target": {
                "public_name": "the red cube",
                "aliases": ["red cube"],
            }
        },
        original_claim="red cube is not grasped",
    )

    assert supported is True
    assert assertion["asserted_value"] == "target_not_grasped_by_robot"
    assert assertion["asserted_after_value"] is None


def test_ir_rejects_executor_motion_projected_onto_target_axis() -> None:
    with pytest.raises(ValueError, match="predicate/subject mismatch"):
        fgsrp_ir._normalize_assertion(
            {
                "temporal_scope": "transition",
                "predicate": "relative_z",
                "subject_entity_id": "executor.robot_arm",
                "subject_text": "robot arm",
                "referenced_entity_ids": ["object.target", "object.goal"],
                "asserted_value": None,
                "asserted_changed": True,
                "asserted_before_value": None,
                "asserted_after_value": "target_above_goal_height_band",
            },
            ontology=fgsrp_ir.CUBE_PREDICATE_ONTOLOGY,
            evidence={
                "transition.relative_z": {
                    "changed": False,
                    "before_value": "target_within_goal_height_band",
                    "after_value": "target_within_goal_height_band",
                }
            },
            entity_registry={
                "executor.robot_arm": {
                    "public_name": "the robot arm",
                    "aliases": ["robot arm"],
                }
            },
            original_claim="the robot arm moved upwards",
        )


def test_cube_support_is_a_derived_topology_not_relative_z_alias() -> None:
    provider = fgsrp.CubeSimulatorEvidenceProvider()
    captured = provider.capture(_fake_env(grasped=False), task="cube_restack")

    assert captured["facts"]["relative_z"]["value"] == (
        "target_above_goal_height_band"
    )
    assert captured["facts"]["target_goal_support"]["value"] == (
        "target_not_supported_by_goal"
    )
    assert fgsrp_ir.CUBE_PREDICATE_ONTOLOGY["target_goal_support"][
        "required_entity_ids"
    ] == ["object.target", "object.goal"]


def test_claim_ledger_requires_every_host_sentence() -> None:
    report = "The state changed. No movement occurred."
    context = {
        "turn": 1,
        "visual_differencing_feedback": report,
        "native_decision_prompt": _native_prompt(report),
    }
    prepared = fgsrp.M3VisualDifferenceCarrier().prepare(
        context, {"m3_reports": []}
    )
    evidence = {
        "transition.relative_x": {
            "changed": True,
            "before_value": "target_center_within_goal_x_band",
            "after_value": "target_center_positive_x_of_goal_band",
        }
    }
    payload = json.loads(
        _verifier_payload(
            "The state changed.",
            repairs=[],
        )
    )

    with pytest.raises(ValueError, match="every sentence exactly once"):
        fgsrp._normalize_claim_ledger(
            json.dumps(payload),
            prepared=prepared,
            evidence=evidence,
        )


def test_claim_ledger_replaces_only_false_conjunct() -> None:
    report = (
        "The red cube is attached to the gripper and remains on the table."
    )
    chunk_id = "visual_difference.turn_01.report"
    false_claim = "red cube is attached to the gripper"
    replacement = "red cube is not attached to the gripper"
    repairs = [
        {
            "chunk_id": chunk_id,
            "original_claim": false_claim,
            "replacement_claim": replacement,
            "replacement_fact_ids": ["after.target_grasp"],
        }
    ]
    claims = [
        {
            "original_claim": false_claim,
            "disposition": "replace_contradicted",
            "replacement_claim": replacement,
            "replacement_fact_ids": ["after.target_grasp"],
        },
        {
            "original_claim": "remains on the table",
            "disposition": "preserve_supported",
            "replacement_claim": None,
            "replacement_fact_ids": ["after.target_table_contact"],
        },
    ]
    content = _verifier_payload(report, repairs=repairs, claims=claims)
    context = {
        "turn": 1,
        "visual_differencing_feedback": report,
        "native_decision_prompt": _native_prompt(report),
    }
    prepared = fgsrp.M3VisualDifferenceCarrier().prepare(
        context, {"m3_reports": []}
    )
    evidence = {
        "after.target_grasp": {
            "value": "target_not_grasped_by_robot",
            "statement": "the red cube is not attached to the gripper",
        },
        "after.target_table_contact": {
            "value": "target_contacting_table",
            "statement": "the red cube remains on the table",
        },
    }

    normalized, _, ledger = fgsrp._normalize_claim_ledger(
        content,
        prepared=prepared,
        evidence=evidence,
    )
    corrected, _ = engine._apply_source_repairs(
        report,
        candidates=prepared.candidates,
        repairs=normalized,
        fact_catalog=evidence,
    )

    assert len(ledger[0]["claims"]) == 2
    assert corrected == (
        "The red cube is not attached to the gripper and remains on the table."
    )


def test_no_change_claim_requires_one_changed_transition() -> None:
    report = "There is no observable difference between the two states."
    context = {
        "turn": 1,
        "visual_differencing_feedback": report,
        "native_decision_prompt": _native_prompt(report),
    }
    prepared = fgsrp.M3VisualDifferenceCarrier().prepare(
        context, {"m3_reports": []}
    )
    fact_ids = ["transition.relative_x", "transition.target_grasp"]
    payload = _verifier_payload(
        report,
        repairs=[
            {
                "chunk_id": "visual_difference.turn_01.report",
                "original_claim": report,
                "replacement_claim": "The relative position changed.",
                "replacement_fact_ids": fact_ids,
            }
        ],
    )
    evidence = {
        fact_id: {
            "changed": True,
            "before_value": f"before_{index}",
            "after_value": f"after_{index}",
        }
        for index, fact_id in enumerate(fact_ids)
    }

    with pytest.raises(ValueError, match="exactly one changed transition"):
        fgsrp._normalize_claim_ledger(
            payload,
            prepared=prepared,
            evidence=evidence,
        )


def test_supported_claim_may_cite_complete_private_evidence_set() -> None:
    report = "There is no observable difference between the two states."
    fact_ids = [f"transition.fact_{index}" for index in range(7)]
    claims = [
        {
            "original_claim": report,
            "disposition": "preserve_supported",
            "replacement_claim": None,
            "replacement_fact_ids": fact_ids,
        }
    ]
    content = _verifier_payload(report, repairs=[], claims=claims)
    context = {
        "turn": 1,
        "visual_differencing_feedback": report,
        "native_decision_prompt": _native_prompt(report),
    }
    prepared = fgsrp.M3VisualDifferenceCarrier().prepare(
        context, {"m3_reports": []}
    )
    evidence = {fact_id: {"changed": False} for fact_id in fact_ids}

    repairs, audit, ledger = fgsrp._normalize_claim_ledger(
        content,
        prepared=prepared,
        evidence=evidence,
    )

    assert repairs == []
    assert ledger[0]["claims"][0]["replacement_fact_ids"] == fact_ids
    assert audit["claim_count"] == 1


def test_unsupported_no_change_repair_is_downgraded_without_losing_valid_repair(
) -> None:
    report = (
        "There is no observable difference between the two states; "
        "the red cube remains above the green cube."
    )
    no_change_fact_ids = [
        "transition.relative_x",
        "transition.relative_y",
    ]
    claims = [
        {
            "original_claim": "There is no observable difference between the two states",
            "disposition": "replace_contradicted",
            "replacement_claim": "The registered relations remained unchanged",
            "replacement_fact_ids": no_change_fact_ids,
        },
        {
            "original_claim": "the red cube remains above the green cube",
            "disposition": "replace_contradicted",
            "replacement_claim": "the red cube is below the green cube's height band",
            "replacement_fact_ids": ["after.relative_z"],
        },
    ]
    repairs = [
        {
            "chunk_id": "visual_difference.turn_01.report",
            "original_claim": claim["original_claim"],
            "replacement_claim": claim["replacement_claim"],
            "replacement_fact_ids": claim["replacement_fact_ids"],
        }
        for claim in claims
    ]
    content = _verifier_payload(report, repairs=repairs, claims=claims)
    context = {
        "turn": 1,
        "visual_differencing_feedback": report,
        "native_decision_prompt": _native_prompt(report),
    }
    prepared = fgsrp.M3VisualDifferenceCarrier().prepare(
        context, {"m3_reports": []}
    )
    evidence = {
        fact_id: {"changed": False}
        for fact_id in no_change_fact_ids
    }
    evidence["after.relative_z"] = {
        "value": "target_below_goal_height_band",
        "predicate": "relative_z",
    }

    normalized, audit, ledger = fgsrp._normalize_claim_ledger(
        content,
        prepared=prepared,
        evidence=evidence,
    )

    assert len(normalized) == 1
    assert normalized[0]["original_claim"] == claims[1]["original_claim"]
    assert ledger[0]["claims"][0] == {
        "original_claim": claims[0]["original_claim"],
        "disposition": "preserve_compatible_or_undecidable",
        "replacement_claim": None,
        "replacement_fact_ids": [],
    }
    assert ledger[0]["claims"][1]["disposition"] == "replace_contradicted"
    assert audit["host_downgraded_unsupported_no_change_repairs"] == 1


def test_claim_ledger_keeps_redundant_false_spans_as_separate_repairs() -> None:
    report = (
        "The current state of the environment is identical to the previous state. "
        "No movement or manipulation has occurred yet."
    )
    chunk_id = "visual_difference.turn_01.report"
    replacement = (
        "The red cube center moved from within the green cube's x band to the "
        "positive-x side of the green cube's x band."
    )
    false_claims = [
        "The current state of the environment is identical to the previous state.",
        "No movement or manipulation has occurred yet.",
    ]
    repairs = [
        {
            "chunk_id": chunk_id,
            "original_claim": claim,
            "replacement_claim": replacement,
            "replacement_fact_ids": ["transition.relative_x"],
        }
        for claim in false_claims
    ]
    response = {
        "schema_version": fgsrp.VERIFIER_SCHEMA_VERSION,
        "claim_ledger": [
            {
                "chunk_id": chunk_id,
                "sentence_index": index,
                "sentence_text": claim,
                "claims": [
                    {
                        "original_claim": claim,
                        "disposition": "replace_contradicted",
                        "replacement_claim": replacement,
                        "replacement_fact_ids": ["transition.relative_x"],
                    }
                ],
            }
            for index, claim in enumerate(false_claims)
        ],
        "repairs": repairs,
        "audit": {
            "all_candidate_chunks_checked": True,
            "every_sentence_listed_once": True,
            "every_atomic_claim_classified": True,
            "contradicted_claims_match_repairs": True,
            "used_only_authoritative_catalog": True,
            "contains_action_advice": False,
        },
    }
    context = {
        "turn": 1,
        "visual_differencing_feedback": report,
        "native_decision_prompt": _native_prompt(report),
    }
    prepared = fgsrp.M3VisualDifferenceCarrier().prepare(
        context, {"m3_reports": []}
    )
    evidence = {
        "transition.relative_x": {
            "changed": True,
            "before_value": "target_center_within_goal_x_band",
            "after_value": "target_center_positive_x_of_goal_band",
        }
    }

    normalized, audit, ledger = fgsrp._normalize_claim_ledger(
        json.dumps(response),
        prepared=prepared,
        evidence=evidence,
    )

    assert len(normalized) == 2
    assert len(ledger) == 2
    assert audit["sentence_count"] == 2


def test_m2_adapter_changes_only_comment_tokens() -> None:
    code = "# red cube is currently grasped\npass\n"
    template = "Executed:\n{executed_code}\nstdout:{console_stdout}\nstderr:{console_stderr}"
    native_evidence = template.format(
        executed_code=code,
        console_stdout="",
        console_stderr="",
    )
    context = {
        "turn": 1,
        "executed_code_blocks": [code],
        "executed_code": code,
        "multi_turn_prompt": template,
        "console_stdout": "",
        "console_stderr": "",
        "native_decision_prompt": _native_prompt(native_evidence),
    }
    state = {"m2_blocks": []}
    adapter = fgsrp.M2CommentCarrier()
    prepared = adapter.prepare(context, state)
    repairs = [
        {
            "chunk_id": prepared.candidates[0].chunk_id,
            "original_claim": "red cube is currently grasped",
            "replacement_claim": "red cube is currently not grasped",
            "replacement_fact_ids": ["after.target_grasp"],
        }
    ]
    applied = adapter.apply(
        prepared,
        repairs=repairs,
        evidence={"after.target_grasp": {"statement": "not grasped"}},
        context=context,
        state=state,
    )

    assert "currently not grasped" in json.dumps(applied.corrected_prompt)
    assert _non_comment_tokens(applied.corrected_source) == _non_comment_tokens(code)
    assert applied.active_repair_count == 1


def test_m3_adapter_replaces_only_native_report() -> None:
    report = "The red cube is attached. The robot stopped."
    context = {
        "turn": 1,
        "visual_differencing_feedback": report,
        "native_decision_prompt": _native_prompt(report),
    }
    state = {"m3_reports": []}
    adapter = fgsrp.M3VisualDifferenceCarrier()
    prepared = adapter.prepare(context, state)
    repairs = [
        {
            "chunk_id": prepared.candidates[0].chunk_id,
            "original_claim": "The red cube is attached",
            "replacement_claim": "The red cube is not attached",
            "replacement_fact_ids": ["after.target_grasp"],
        }
    ]
    applied = adapter.apply(
        prepared,
        repairs=repairs,
        evidence={"after.target_grasp": {"statement": "not attached"}},
        context=context,
        state=state,
    )

    serialized = json.dumps(applied.corrected_prompt)
    assert "The red cube is not attached. The robot stopped." in serialized
    assert applied.corrected_prompt[1]["content"][1] == (
        context["native_decision_prompt"][1]["content"][1]
    )
    assert state["m3_reports"][0]["repair_count"] == 1


def test_fgsrp_m3_no_op_returns_byte_identical_native_request(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CAPX_FGSRP_CARRIER", "m3_visual_difference")
    report = "The red cube remained on the table."
    empty = _ir_parser_payload(
        report,
        claims=[
            {
                "original_claim": report,
                "claim_type": "unknown",
                "assertions": [],
            }
        ],
    )

    def fake_request(*args, **kwargs):
        return empty, {
            "seed": kwargs["seed"],
            "request_sha256": "request",
            "response_sha256": engine._sha256_text(empty),
            "transport": [],
        }

    monkeypatch.setattr(engine, "_request", fake_request)
    env = _fake_env(grasped=False)
    fgsrp.initialize_trial_trace(
        env,
        output_dir=str(tmp_path),
        trial=1,
        task="cube_stack",
    )
    native = _native_prompt(report)
    result = fgsrp.run_fgsrp(
        env,
        SimpleNamespace(model="qwen3.7-plus"),
        output_dir=str(tmp_path),
        trial=1,
        turn=1,
        task="cube_stack",
        task_description="Goal: stack cubes.",
        multi_turn_prompt=(
            "{executed_code}{console_stdout}{console_stderr}"
        ),
        executed_code_blocks=["pass\n"],
        executed_code="pass\n",
        console_stdout="",
        console_stderr="",
        visual_differencing_feedback=report,
        native_decision_prompt=native,
    )

    assert result is native
    record = json.loads(
        (
            tmp_path
            / fgsrp.TRACE_DIR
            / "trial_01"
            / "turn_01"
            / "replacement.json"
        ).read_text()
    )
    assert record["no_op_prompt_byte_equivalent"] is True
    assert record["action_visible_gt_packet_injected"] is False


def test_fgsrp_m3_accepted_repair_is_the_only_visible_change(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CAPX_FGSRP_CARRIER", "m3_visual_difference")
    report = "The red cube is currently attached to the gripper."
    chunk_id = "visual_difference.turn_01.report"
    repairs = [
        {
            "chunk_id": chunk_id,
            "original_claim": "red cube is currently attached to the gripper",
            "replacement_claim": "red cube is currently not attached to the gripper",
            "replacement_fact_ids": ["after.target_grasp"],
        }
    ]
    proposal = _ir_parser_payload(
        report,
        claims=[
            {
                "original_claim": repairs[0]["original_claim"],
                "claim_type": "physical",
                "assertions": [
                    {
                        "temporal_scope": "after",
                        "predicate": "target_grasp",
                        "asserted_value": "target_grasped_by_robot",
                        "asserted_changed": None,
                        "asserted_before_value": None,
                        "asserted_after_value": None,
                    }
                ],
            }
        ],
    )
    gate = _ir_rewriter_payload(
        original_claim=repairs[0]["original_claim"],
        replacement_claim=repairs[0]["replacement_claim"],
        fact_ids=repairs[0]["replacement_fact_ids"],
    )
    responses = [proposal, gate]

    def fake_request(*args, **kwargs):
        content = responses.pop(0)
        return content, {
            "seed": kwargs["seed"],
            "request_sha256": "request",
            "response_sha256": engine._sha256_text(content),
            "transport": [],
        }

    monkeypatch.setattr(engine, "_request", fake_request)
    env = _fake_env(grasped=False)
    fgsrp.initialize_trial_trace(
        env,
        output_dir=str(tmp_path),
        trial=1,
        task="cube_restack",
    )
    native = _native_prompt(report)
    result = fgsrp.run_fgsrp(
        env,
        SimpleNamespace(model="qwen3.7-plus"),
        output_dir=str(tmp_path),
        trial=1,
        turn=1,
        task="cube_restack",
        task_description="Goal: restack cubes.",
        multi_turn_prompt="{executed_code}{console_stdout}{console_stderr}",
        executed_code_blocks=["pass\n"],
        executed_code="pass\n",
        console_stdout="",
        console_stderr="",
        visual_differencing_feedback=report,
        native_decision_prompt=native,
    )

    assert "currently not attached" in json.dumps(result)
    assert result[1]["content"][1] == native[1]["content"][1]
    assert len(result) == len(native)


def test_fgsrp_prompts_are_open_world_fact_repair_only() -> None:
    root = fgsrp.Path(fgsrp.__file__).parent
    verifier = " ".join(
        (root / "fgsrp_verifier_prompt.md").read_text().lower().split()
    )
    gate = " ".join(
        (root / "fgsrp_adjudicator_prompt.md").read_text().lower().split()
    )

    assert "do not use a closed-world assumption" in verifier
    assert "do not add omitted facts" in verifier
    assert "verification, not reporting, coaching" in verifier
    assert "missing evidence is not contradiction" in gate
    assert "before/after/transition" in gate


def test_deterministic_gate_rejects_contact_as_support_negation() -> None:
    repair = {
        "original_claim": "the green cube resting on top of the red cube",
        "replacement_claim": "the red cube in contact with the green cube",
        "replacement_fact_ids": ["before.target_goal_contact"],
    }
    evidence = {
        "before.target_goal_contact": {
            "value": "target_contacting_goal",
            "predicate": "target_goal_contact",
        }
    }

    result = fgsrp._deterministic_contradiction_certificate(repair, evidence)

    assert result == {
        "accepted": False,
        "reason": "no_direct_predicate_contradiction",
    }


def test_deterministic_gate_accepts_direct_table_contact_correction() -> None:
    repair = {
        "original_claim": "lifting it slightly above the table",
        "replacement_claim": "keeping it in contact with the table",
        "replacement_fact_ids": ["after.target_table_contact"],
    }
    evidence = {
        "after.target_table_contact": {
            "value": "target_contacting_table",
            "predicate": "target_table_contact",
        }
    }

    result = fgsrp._deterministic_contradiction_certificate(repair, evidence)

    assert result["accepted"] is True
    assert result["contradicting_fact_ids"] == ["after.target_table_contact"]


def test_deterministic_gate_accepts_proven_transition_against_no_change() -> None:
    repair = {
        "original_claim": "The current state is identical to the previous state.",
        "replacement_claim": (
            "The red cube moved from within the green cube's x band to the "
            "positive-x side."
        ),
        "replacement_fact_ids": ["transition.relative_x"],
    }
    evidence = {
        "transition.relative_x": {
            "changed": True,
            "before_value": "target_center_within_goal_x_band",
            "after_value": "target_center_positive_x_of_goal_band",
            "predicate": "relative_x",
        }
    }

    result = fgsrp._deterministic_contradiction_certificate(repair, evidence)

    assert result["accepted"] is True
    assert result["reason"] == "direct_predicate_transition_contradiction"

    repair = {
        "original_claim": (
            "The red and green cubes remain in the same relative positions "
            "on the table"
        ),
        "replacement_claim": "The red and green cubes are no longer in contact",
        "replacement_fact_ids": ["transition.target_goal_contact"],
    }
    evidence = {
        "transition.target_goal_contact": {
            "changed": True,
            "before_value": "target_contacting_goal",
            "after_value": "target_separated_from_goal",
            "predicate": "target_goal_contact",
        }
    }

    result = fgsrp._deterministic_contradiction_certificate(repair, evidence)

    assert result["accepted"] is True
    assert result["reason"] == "direct_predicate_transition_contradiction"

    repair["replacement_claim"] = (
        "The red cube is now separated from the green cube"
    )
    result = fgsrp._deterministic_contradiction_certificate(repair, evidence)

    assert result["accepted"] is True
    assert result["reason"] == "direct_predicate_transition_contradiction"

    repair["replacement_claim"] = (
        "The red and green cubes are no longer in contact"
    )

    evidence["transition.target_goal_contact"].update(
        before_value="target_separated_from_goal",
        after_value="target_contacting_goal",
    )
    result = fgsrp._deterministic_contradiction_certificate(repair, evidence)

    assert result == {
        "accepted": False,
        "reason": "transition_replacement_not_canonically_entailed",
    }

    evidence["transition.target_goal_contact"].update(
        before_value="target_contacting_goal",
        after_value="target_separated_from_goal",
    )

    repair["original_claim"] = "No movement or manipulation has occurred yet."
    result = fgsrp._deterministic_contradiction_certificate(repair, evidence)

    assert result["accepted"] is True
    assert result["reason"] == "direct_predicate_transition_contradiction"


def test_static_relative_state_is_not_misclassified_as_no_change() -> None:
    assert not fgsrp._matches_any(
        "The red cube remains above the green cube.",
        fgsrp._NO_CHANGE_CLAIM_PATTERNS,
    )


def test_host_override_accepts_only_certified_atomic_transition() -> None:
    decision = {
        "relation": "contradicted",
        "span_is_smallest_complete_clause": False,
        "selected_facts_entail_claim_negation": True,
        "replacement_claim_entailed_by_selected_facts": True,
        "replacement_preserves_all_other_meaning": False,
    }

    assert fgsrp._host_atomic_gate_override(
        decision,
        {
            "accepted": True,
            "reason": "direct_predicate_transition_contradiction",
        },
    )
    assert not fgsrp._host_atomic_gate_override(
        decision,
        {"accepted": True, "reason": "direct_predicate_contradiction"},
    )

    for failed_check in (
        "selected_facts_entail_claim_negation",
        "replacement_claim_entailed_by_selected_facts",
    ):
        rejected = {**decision, failed_check: False}
        assert not fgsrp._host_atomic_gate_override(
            rejected,
            {
                "accepted": True,
                "reason": "direct_predicate_transition_contradiction",
            },
        )


def test_certified_global_no_change_claim_is_replaced_end_to_end(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CAPX_FGSRP_CARRIER", "m3_visual_difference")
    report = "The current state of the environment is identical to the previous state."
    replacement = (
        "The red cube center moved from within the green cube's x band to the "
        "positive-x side of the green cube's x band."
    )
    repairs = [
        {
            "chunk_id": "visual_difference.turn_01.report",
            "original_claim": report,
            "replacement_claim": replacement,
            "replacement_fact_ids": ["transition.relative_x"],
        }
    ]
    proposal = _ir_parser_payload(
        report,
        claims=[
            {
                "original_claim": report,
                "claim_type": "physical",
                "assertions": [
                    {
                        "temporal_scope": "transition",
                        "predicate": "relative_x",
                        "asserted_value": None,
                        "asserted_changed": False,
                        "asserted_before_value": None,
                        "asserted_after_value": None,
                    }
                ],
            }
        ],
    )
    rejected_by_llm_gate = _ir_rewriter_payload(
        original_claim=report,
        replacement_claim=replacement,
        fact_ids=["transition.relative_x"],
    )
    responses = [proposal, rejected_by_llm_gate]

    def fake_request(*args, **kwargs):
        content = responses.pop(0)
        return content, {
            "seed": kwargs["seed"],
            "request_sha256": "request",
            "response_sha256": engine._sha256_text(content),
            "transport": [],
        }

    monkeypatch.setattr(engine, "_request", fake_request)
    env = _fake_env(grasped=False)
    env.low_level_env.robosuite_env.sim.data.body_xpos[0, 0] = 0.0
    fgsrp.initialize_trial_trace(
        env, output_dir=str(tmp_path), trial=1, task="cube_restack"
    )
    env.low_level_env.robosuite_env.sim.data.body_xpos[0, 0] = 0.04
    native = _native_prompt(report)

    result = fgsrp.run_fgsrp(
        env,
        SimpleNamespace(model="qwen3.7-plus"),
        output_dir=str(tmp_path),
        trial=1,
        turn=1,
        task="cube_restack",
        task_description="Goal: restack cubes.",
        multi_turn_prompt="{executed_code}{console_stdout}{console_stderr}",
        executed_code_blocks=["pass\n"],
        executed_code="pass\n",
        console_stdout="",
        console_stderr="",
        visual_differencing_feedback=report,
        native_decision_prompt=native,
    )

    assert replacement in json.dumps(result)
    assert not responses
    record = json.loads(
        (
            tmp_path
            / fgsrp.TRACE_DIR
            / "trial_01"
            / "turn_01"
            / "replacement.json"
        ).read_text()
    )
    protocol = record["verification_protocol"]
    assert record["repair_count"] == 1
    assert protocol["atomic_refinement"] is None
    assert protocol["deterministic_gate"][0]["reason"] == (
        "canonical_ir_gt_contradiction"
    )
    assert protocol["deterministic_gate"][0]["effective_accepted"] is True
    assert protocol["adjudicator"]["decisions"][0]["accepted"] is True


def test_invalid_verifier_span_abstains_without_failing_trial(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CAPX_FGSRP_CARRIER", "m3_visual_difference")
    monkeypatch.setenv("CAPX_FGSRP_SCHEMA_ATTEMPTS", "1")
    invalid = json.dumps(
        {
            "schema_version": fgsrp.VERIFIER_SCHEMA_VERSION,
            "repairs": [
                {
                    "chunk_id": "visual_difference.turn_01.report",
                    "original_claim": "a claim absent from the native report",
                    "replacement_claim": "the red cube is currently not attached",
                    "replacement_fact_ids": ["after.target_grasp"],
                }
            ],
            "audit": {
                "all_candidate_chunks_checked": True,
                "used_only_authoritative_catalog": True,
                "contains_action_advice": False,
            },
        }
    )

    def fake_request(*args, **kwargs):
        return invalid, {
            "seed": kwargs["seed"],
            "request_sha256": "request",
            "response_sha256": engine._sha256_text(invalid),
            "transport": [],
        }

    monkeypatch.setattr(engine, "_request", fake_request)
    env = _fake_env(grasped=False)
    fgsrp.initialize_trial_trace(
        env,
        output_dir=str(tmp_path),
        trial=1,
        task="cube_restack",
    )
    report = "The red cube remained on the table."
    native = _native_prompt(report)

    result = fgsrp.run_fgsrp(
        env,
        SimpleNamespace(model="qwen3.7-plus"),
        output_dir=str(tmp_path),
        trial=1,
        turn=1,
        task="cube_restack",
        task_description="Goal: restack cubes.",
        multi_turn_prompt="{executed_code}{console_stdout}{console_stderr}",
        executed_code_blocks=["pass\n"],
        executed_code="pass\n",
        console_stdout="",
        console_stderr="",
        visual_differencing_feedback=report,
        native_decision_prompt=native,
    )

    assert result is native
    record = json.loads(
        (
            tmp_path
            / fgsrp.TRACE_DIR
            / "trial_01"
            / "turn_01"
            / "replacement.json"
        ).read_text()
    )
    assert record["repair_count"] == 0
    assert record["verification_protocol"]["verifier"]["protocol"]["status"] == (
        "abstained_invalid_response"
    )


def test_schema_retry_materializes_prior_assistant_response(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CAPX_FGSRP_CARRIER", "m3_visual_difference")
    monkeypatch.setenv("CAPX_FGSRP_SCHEMA_ATTEMPTS", "2")
    report = "The current state is identical to the previous state."
    chunk_id = "visual_difference.turn_01.report"
    invalid = _ir_parser_payload(
        report,
        claims=[
            {
                "original_claim": report,
                "claim_type": "physical",
                "assertions": [],
            }
        ],
    )
    valid = _ir_parser_payload(
        report,
        claims=[
            {
                "original_claim": report,
                "claim_type": "unknown",
                "assertions": [],
            }
        ],
    )
    responses = [invalid, valid]
    requests = []

    def fake_request(args, messages, **kwargs):
        requests.append(messages)
        content = responses.pop(0)
        return content, {
            "seed": kwargs["seed"],
            "request_sha256": "request",
            "response_sha256": engine._sha256_text(content),
            "transport": [],
        }

    monkeypatch.setattr(engine, "_request", fake_request)
    env = _fake_env(grasped=False)
    fgsrp.initialize_trial_trace(
        env, output_dir=str(tmp_path), trial=1, task="cube_stack"
    )
    native = _native_prompt(report)

    result = fgsrp.run_fgsrp(
        env,
        SimpleNamespace(model="qwen3.7-plus"),
        output_dir=str(tmp_path),
        trial=1,
        turn=1,
        task="cube_stack",
        task_description="Goal: stack cubes.",
        multi_turn_prompt="{executed_code}{console_stdout}{console_stderr}",
        executed_code_blocks=["pass\n"],
        executed_code="pass\n",
        console_stdout="",
        console_stderr="",
        visual_differencing_feedback=report,
        native_decision_prompt=native,
    )

    assert result is native
    assert len(requests) == 2
    assert requests[1][-2] == {"role": "assistant", "content": invalid}
    assert requests[1][-1]["role"] == "user"
    assert "physical claim has no semantic assertions" in requests[1][-1]["content"]
    assert not responses


def test_host_rejected_proposal_skips_private_adjudicator(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CAPX_FGSRP_CARRIER", "m3_visual_difference")
    report = "The green cube is resting on top of the red cube."
    repairs = [
        {
            "chunk_id": "visual_difference.turn_01.report",
            "original_claim": "green cube is resting on top of the red cube",
            "replacement_claim": (
                "red cube is currently in contact with the green cube"
            ),
            "replacement_fact_ids": ["after.target_goal_contact"],
        }
    ]
    proposal = _ir_parser_payload(
        report,
        claims=[
            {
                "original_claim": repairs[0]["original_claim"],
                "claim_type": "physical",
                "assertions": [
                    {
                        "temporal_scope": "after",
                        "predicate": "target_goal_contact",
                        "asserted_value": "target_separated_from_goal",
                        "asserted_changed": None,
                        "asserted_before_value": None,
                        "asserted_after_value": None,
                    }
                ],
            }
        ],
    )
    calls = 0

    def fake_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert calls == 1, "host-rejected proposal reached the private gate"
        return proposal, {
            "seed": kwargs["seed"],
            "request_sha256": "request",
            "response_sha256": engine._sha256_text(proposal),
            "transport": [],
        }

    monkeypatch.setattr(engine, "_request", fake_request)
    env = _fake_env(grasped=False)
    fgsrp.initialize_trial_trace(
        env, output_dir=str(tmp_path), trial=1, task="cube_restack"
    )
    native = _native_prompt(report)

    result = fgsrp.run_fgsrp(
        env,
        SimpleNamespace(model="qwen3.7-plus"),
        output_dir=str(tmp_path),
        trial=1,
        turn=1,
        task="cube_restack",
        task_description="Goal: restack cubes.",
        multi_turn_prompt="{executed_code}{console_stdout}{console_stderr}",
        executed_code_blocks=["pass\n"],
        executed_code="pass\n",
        console_stdout="",
        console_stderr="",
        visual_differencing_feedback=report,
        native_decision_prompt=native,
    )

    assert result is native
    record = json.loads(
        (
            tmp_path
            / fgsrp.TRACE_DIR
            / "trial_01"
            / "turn_01"
            / "replacement.json"
        ).read_text()
    )
    protocol = record["verification_protocol"]
    assert protocol["adjudicator"] is None
    assert protocol["adjudicator_input_map"] == []
    assert protocol["deterministic_gate"] == []
    assert protocol["ir_evaluation"]["false_claims"] == 0


def test_verifier_transport_error_abstains_without_failing_trial(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CAPX_FGSRP_CARRIER", "m3_visual_difference")

    def fake_request(*args, **kwargs):
        raise RuntimeError("private verifier unavailable")

    monkeypatch.setattr(engine, "_request", fake_request)
    env = _fake_env(grasped=False)
    fgsrp.initialize_trial_trace(
        env, output_dir=str(tmp_path), trial=1, task="cube_stack"
    )
    report = "The red cube remained on the table."
    native = _native_prompt(report)

    result = fgsrp.run_fgsrp(
        env,
        SimpleNamespace(model="qwen3.7-plus"),
        output_dir=str(tmp_path),
        trial=1,
        turn=1,
        task="cube_stack",
        task_description="Goal: stack cubes.",
        multi_turn_prompt="{executed_code}{console_stdout}{console_stderr}",
        executed_code_blocks=["pass\n"],
        executed_code="pass\n",
        console_stdout="",
        console_stderr="",
        visual_differencing_feedback=report,
        native_decision_prompt=native,
    )

    assert result is native
    record = json.loads(
        (
            tmp_path
            / fgsrp.TRACE_DIR
            / "trial_01"
            / "turn_01"
            / "replacement.json"
        ).read_text()
    )
    assert record["no_op_prompt_byte_equivalent"] is True
    assert record["verification_protocol"]["verifier"]["protocol"]["status"] == (
        "abstained_transport_error"
    )


def test_adjudicator_transport_error_abstains_without_failing_trial(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CAPX_FGSRP_CARRIER", "m3_visual_difference")
    report = "The red cube is currently attached to the gripper."
    repairs = [
        {
            "chunk_id": "visual_difference.turn_01.report",
            "original_claim": "red cube is currently attached to the gripper",
            "replacement_claim": "red cube is currently not attached to the gripper",
            "replacement_fact_ids": ["after.target_grasp"],
        }
    ]
    proposal = _ir_parser_payload(
        report,
        claims=[
            {
                "original_claim": repairs[0]["original_claim"],
                "claim_type": "physical",
                "assertions": [
                    {
                        "temporal_scope": "after",
                        "predicate": "target_grasp",
                        "asserted_value": "target_grasped_by_robot",
                        "asserted_changed": None,
                        "asserted_before_value": None,
                        "asserted_after_value": None,
                    }
                ],
            }
        ],
    )
    calls = 0

    def fake_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return proposal, {
                "seed": kwargs["seed"],
                "request_sha256": "request",
                "response_sha256": engine._sha256_text(proposal),
                "transport": [],
            }
        raise RuntimeError("private adjudicator unavailable")

    monkeypatch.setattr(engine, "_request", fake_request)
    env = _fake_env(grasped=False)
    fgsrp.initialize_trial_trace(
        env, output_dir=str(tmp_path), trial=1, task="cube_restack"
    )
    native = _native_prompt(report)

    result = fgsrp.run_fgsrp(
        env,
        SimpleNamespace(model="qwen3.7-plus"),
        output_dir=str(tmp_path),
        trial=1,
        turn=1,
        task="cube_restack",
        task_description="Goal: restack cubes.",
        multi_turn_prompt="{executed_code}{console_stdout}{console_stderr}",
        executed_code_blocks=["pass\n"],
        executed_code="pass\n",
        console_stdout="",
        console_stderr="",
        visual_differencing_feedback=report,
        native_decision_prompt=native,
    )

    assert result is native
    record = json.loads(
        (
            tmp_path
            / fgsrp.TRACE_DIR
            / "trial_01"
            / "turn_01"
            / "replacement.json"
        ).read_text()
    )
    protocol = record["verification_protocol"]
    assert record["no_op_prompt_byte_equivalent"] is True
    assert protocol["adjudicator"] is None
    assert protocol["ir_evaluation"]["writer_status"] == (
        "abstained_transport_error"
    )
    assert protocol["deterministic_gate"] == []
    assert protocol["ir_evaluation"]["false_claims"] == 1
