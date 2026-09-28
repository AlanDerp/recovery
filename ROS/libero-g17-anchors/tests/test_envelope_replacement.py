from __future__ import annotations

import json
import tokenize
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest

from capx.verification import envelope_replacement


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


def test_candidate_extraction_is_exact_and_provenance_typed() -> None:
    code = (
        "red = detect('red cube')  # red cube is grasped\n"
        "# placement succeeded\n"
        "open_gripper()\n"
    )
    stdout = "planning\nred cube attached\n"
    chunks = envelope_replacement.extract_candidate_chunks(code, stdout)

    assert [chunk.chunk_id for chunk in chunks] == [
        "executed_code.comment.0000",
        "executed_code.comment.0001",
    ]
    for chunk in chunks:
        assert chunk.source == "executed_code_comment"
        assert code[chunk.start : chunk.end] == chunk.text
        assert envelope_replacement._sha256_text(chunk.text) == chunk.sha256


def test_host_rejects_any_non_comment_repair_surface() -> None:
    catalog = envelope_replacement._fact_catalog(_fake_env(grasped=False))
    candidate = envelope_replacement.CandidateChunk(
        chunk_id="console_stdout.line.0000",
        source="console_stdout_line",
        start=0,
        end=10,
        text="bad output",
        sha256=envelope_replacement._sha256_text("bad output"),
    )

    with pytest.raises(ValueError, match="Only executed-code comments"):
        envelope_replacement.apply_repairs(
            executed_code="pass\n",
            console_stdout="bad output",
            candidates=[candidate],
            repairs=[],
            fact_catalog=catalog,
        )


def test_host_renders_gt_and_changes_only_comment_tokens() -> None:
    code = (
        "red = detect('red cube')\n"
        "# red cube is currently grasped; next pose remains cached\n"
        "pose = red.copy()\n"
    )
    chunks = envelope_replacement.extract_candidate_chunks(code, "")
    catalog = envelope_replacement._fact_catalog(_fake_env(grasped=False))
    repairs = [
        {
            "chunk_id": chunks[0].chunk_id,
            "original_claim": "red cube is currently grasped",
            "replacement_claim": "red cube is currently not grasped",
            "replacement_fact_ids": ["target_grasp"],
        }
    ]
    repaired, stdout, audit = envelope_replacement.apply_repairs(
        executed_code=code,
        console_stdout="",
        candidates=chunks,
        repairs=repairs,
        fact_catalog=catalog,
    )

    assert stdout == ""
    assert "currently not grasped" in repaired
    assert "currently grasped" not in repaired
    assert "next pose remains cached" in repaired
    assert _non_comment_tokens(repaired) == _non_comment_tokens(code)
    assert len(audit) == 1
    compile(repaired, "<repaired>", "exec")


def test_no_repair_preserves_exact_sources_and_prompt_object() -> None:
    code = "# this code attempts a grasp\npass\n"
    stdout = ""
    chunks = envelope_replacement.extract_candidate_chunks(code, stdout)
    catalog = envelope_replacement._fact_catalog(_fake_env())
    repaired_code, repaired_stdout, audit = envelope_replacement.apply_repairs(
        executed_code=code,
        console_stdout=stdout,
        candidates=chunks,
        repairs=[],
        fact_catalog=catalog,
    )
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "ordinary A1 envelope"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        }
    ]
    same = envelope_replacement._replace_prompt_text_once(
        prompt,
        original="ordinary A1 envelope",
        replacement="ordinary A1 envelope",
    )

    assert repaired_code == code
    assert repaired_stdout == stdout
    assert audit == []
    assert same is prompt
    assert envelope_replacement._sha256_json(same) == (
        envelope_replacement._sha256_json(prompt)
    )


def test_prompt_replacement_preserves_roles_and_image_bytes() -> None:
    prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "prefix\nOLD\n"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
                {"type": "text", "text": "decision"},
            ],
        },
    ]
    repaired = envelope_replacement._replace_prompt_text_once(
        prompt,
        original="OLD",
        replacement="NEW",
    )

    assert prompt[1]["content"][0]["text"] == "prefix\nOLD\n"
    assert repaired[1]["content"][0]["text"] == "prefix\nNEW\n"
    assert envelope_replacement._non_text_structure(repaired) == (
        envelope_replacement._non_text_structure(prompt)
    )
    assert repaired[1]["content"][1] == prompt[1]["content"][1]


def test_verifier_output_can_only_select_exact_chunks_and_gt_ids() -> None:
    chunks = envelope_replacement.extract_candidate_chunks(
        "# red cube is grasped\npass\n",
        "",
    )
    catalog = envelope_replacement._fact_catalog(_fake_env(grasped=False))
    response = json.dumps(
        {
            "schema_version": envelope_replacement.VERIFIER_SCHEMA_VERSION,
            "repairs": [
                {
                    "chunk_id": chunks[0].chunk_id,
                    "original_claim": "red cube is grasped",
                    "replacement_claim": "red cube is not grasped",
                    "replacement_fact_ids": ["target_grasp"],
                }
            ],
            "audit": {
                "all_candidate_chunks_checked": True,
                "used_only_authoritative_catalog": True,
                "contains_action_advice": False,
            },
        }
    )
    repairs, audit = envelope_replacement._normalize_repairs(
        response,
        candidates=chunks,
        fact_catalog=catalog,
    )
    assert repairs[0]["replacement_fact_ids"] == ["target_grasp"]
    assert repairs[0]["original_sha256"] == chunks[0].sha256
    assert repairs[0]["original_claim_sha256"] == (
        envelope_replacement._sha256_text("red cube is grasped")
    )
    assert repairs[0]["replacement_claim"] == "red cube is not grasped"
    assert audit["all_candidate_chunks_checked"] is True

    missing_schema = json.loads(response)
    del missing_schema["schema_version"]
    _, normalized_audit = envelope_replacement._normalize_repairs(
        json.dumps(missing_schema),
        candidates=chunks,
        fact_catalog=catalog,
    )
    assert normalized_audit["schema_version_host_normalized"] is True

    invalid = json.loads(response)
    invalid["repairs"][0]["replacement_fact_ids"] = ["next_move"]
    with pytest.raises(ValueError, match="unknown GT predicates"):
        envelope_replacement._normalize_repairs(
            json.dumps(invalid),
            candidates=chunks,
            fact_catalog=catalog,
        )

    numeric = json.loads(response)
    numeric["repairs"][0]["original_claim"] = "cube is grasped at 0.5"
    numeric_chunks = envelope_replacement.extract_candidate_chunks(
        "# cube is grasped at 0.5\npass\n",
        "",
    )
    numeric["repairs"][0]["chunk_id"] = numeric_chunks[0].chunk_id
    with pytest.raises(ValueError, match="cannot alter numeric data"):
        envelope_replacement._normalize_repairs(
            json.dumps(numeric),
            candidates=numeric_chunks,
            fact_catalog=catalog,
        )


def test_contradiction_gate_accepts_only_direct_minimal_contradictions() -> None:
    proposals = [
        {
            "chunk_id": "comment.0",
            "original_claim": "the red cube is grasped",
            "replacement_claim": "the red cube is not grasped",
            "replacement_fact_ids": ["target_grasp"],
        },
        {
            "chunk_id": "comment.1",
            "original_claim": "the red cube is on the table next to the green cube",
            "replacement_claim": "the red cube is separated from the green cube",
            "replacement_fact_ids": ["target_goal_contact"],
        },
    ]
    response = json.dumps(
        {
            "schema_version": envelope_replacement.ADJUDICATOR_SCHEMA_VERSION,
            "decisions": [
                {
                    "repair_index": 0,
                    "chunk_id": "comment.0",
                    "original_claim": "the red cube is grasped",
                    "replacement_claim": "the red cube is not grasped",
                    "relation": "contradicted",
                    "span_is_smallest_complete_clause": True,
                    "selected_facts_entail_claim_negation": True,
                    "replacement_claim_entailed_by_selected_facts": True,
                    "replacement_preserves_all_other_meaning": True,
                },
                {
                    "repair_index": 1,
                    "chunk_id": "comment.1",
                    "original_claim": (
                        "the red cube is on the table next to the green cube"
                    ),
                    "replacement_claim": (
                        "the red cube is separated from the green cube"
                    ),
                    "relation": "compatible",
                    "span_is_smallest_complete_clause": False,
                    "selected_facts_entail_claim_negation": False,
                    "replacement_claim_entailed_by_selected_facts": True,
                    "replacement_preserves_all_other_meaning": False,
                },
            ],
            "audit": {
                "all_proposals_checked": True,
                "used_only_supplied_text_and_catalog": True,
            },
        }
    )

    decisions, _ = envelope_replacement._normalize_adjudications(
        response,
        proposed_repairs=proposals,
    )

    assert decisions[0]["accepted"] is True
    assert decisions[1]["accepted"] is False


def test_verifier_meta_prompt_is_private_fact_repair_only() -> None:
    path = (
        envelope_replacement.Path(envelope_replacement.__file__).with_name(
            "envelope_replacement_verifier_prompt.md"
        )
    )
    prompt = " ".join(path.read_text(encoding="utf-8").lower().split())
    assert "verification, not coaching" in prompt
    assert "smallest local wording correction" in prompt
    assert "do not turn a local edit into a state summary" in prompt
    assert "action, strategy, next" in prompt
    assert "return strict json" in prompt

    gate_path = path.with_name("envelope_replacement_adjudicator_prompt.md")
    gate = " ".join(gate_path.read_text(encoding="utf-8").lower().split())
    assert "more specific gt is not evidence" in gate
    assert "missing information is not contradiction" in gate
    assert "fail-closed" in gate


def test_full_no_candidate_round_is_a_byte_identical_a1_no_op(
    tmp_path,
) -> None:
    trial = 1
    turn = 1
    task = "cube_stack"
    envelope_replacement.initialize_trial_trace(
        output_dir=str(tmp_path),
        trial=trial,
        task=task,
    )
    template = (
        "Executed:\n{executed_code}\n"
        "stdout:\n{console_stdout}\n"
        "stderr:\n{console_stderr}\n"
    )
    evidence = template.format(
        executed_code="pass\n",
        console_stdout="",
        console_stderr="",
    )
    native = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"task\n{evidence}"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        },
    ]
    result = envelope_replacement.run_envelope_replacement(
        _fake_env(),
        SimpleNamespace(),
        output_dir=str(tmp_path),
        trial=trial,
        turn=turn,
        task=task,
        task_description="Goal: stack cubes.\nAPIs:\nprivate docs",
        multi_turn_prompt=template,
        executed_code_blocks=["pass\n"],
        executed_code="pass\n",
        console_stdout="",
        console_stderr="",
        native_decision_prompt=native,
    )

    assert result is native
    record = json.loads(
        (
            tmp_path
            / envelope_replacement.TRACE_DIR
            / "trial_01"
            / "turn_01"
            / "replacement.json"
        ).read_text()
    )
    assert record["candidate_count"] == 0
    assert record["repair_count"] == 0
    assert record["native_prompt_sha256"] == record["corrected_prompt_sha256"]
    assert record["no_op_prompt_byte_equivalent"] is True


def test_full_repair_round_sends_only_rewritten_native_envelope(
    tmp_path,
    monkeypatch,
) -> None:
    trial = 2
    turn = 1
    task = "cube_restack"
    code = "# red cube is currently grasped\npass\n"
    chunks = envelope_replacement._latest_block_candidates(
        code,
        block_index=0,
    )
    response = json.dumps(
        {
            "schema_version": envelope_replacement.VERIFIER_SCHEMA_VERSION,
            "repairs": [
                {
                    "chunk_id": chunks[0].chunk_id,
                    "original_claim": "red cube is currently grasped",
                    "replacement_claim": "red cube is currently not grasped",
                    "replacement_fact_ids": ["target_grasp"],
                }
            ],
            "audit": {
                "all_candidate_chunks_checked": True,
                "used_only_authoritative_catalog": True,
                "contains_action_advice": False,
            },
        }
    )

    gate_response = json.dumps(
        {
            "schema_version": envelope_replacement.ADJUDICATOR_SCHEMA_VERSION,
            "decisions": [
                {
                    "repair_index": 0,
                    "chunk_id": chunks[0].chunk_id,
                    "original_claim": "red cube is currently grasped",
                    "replacement_claim": "red cube is currently not grasped",
                    "relation": "contradicted",
                    "span_is_smallest_complete_clause": True,
                    "selected_facts_entail_claim_negation": True,
                    "replacement_claim_entailed_by_selected_facts": True,
                    "replacement_preserves_all_other_meaning": True,
                }
            ],
            "audit": {
                "all_proposals_checked": True,
                "used_only_supplied_text_and_catalog": True,
            },
        }
    )
    responses = [response, gate_response]

    def fake_request(*args, **kwargs):
        current = responses.pop(0)
        return current, {
            "seed": kwargs["seed"],
            "request_sha256": "request",
            "response_sha256": envelope_replacement._sha256_text(current),
            "transport": [],
        }

    monkeypatch.setattr(envelope_replacement, "_request", fake_request)
    envelope_replacement.initialize_trial_trace(
        output_dir=str(tmp_path),
        trial=trial,
        task=task,
    )
    template = (
        "Executed:\n{executed_code}\n"
        "stdout:\n{console_stdout}\n"
        "stderr:\n{console_stderr}\n"
    )
    evidence = template.format(
        executed_code=code,
        console_stdout="",
        console_stderr="",
    )
    native = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"task\n{evidence}"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        },
    ]
    result = envelope_replacement.run_envelope_replacement(
        _fake_env(grasped=False),
        SimpleNamespace(model="qwen3.7-plus"),
        output_dir=str(tmp_path),
        trial=trial,
        turn=turn,
        task=task,
        task_description="Goal: restack cubes.",
        multi_turn_prompt=template,
        executed_code_blocks=[code],
        executed_code=code,
        console_stdout="",
        console_stderr="",
        native_decision_prompt=native,
    )
    serialized = json.dumps(result)

    assert "currently grasped" not in serialized
    assert "currently not grasped" in serialized
    assert len(result) == len(native)
    assert result[1]["content"][1] == native[1]["content"][1]
    assert "GT_BACKED" not in serialized
    assert "PGSVP" not in serialized


def test_verified_block_repair_persists_without_rechecking_history(
    tmp_path,
    monkeypatch,
) -> None:
    task = "cube_stack"
    code_0 = "# red cube is currently grasped\npass\n"
    code_1 = "# attempt another placement\npass\n"
    first_chunk = envelope_replacement._latest_block_candidates(
        code_0,
        block_index=0,
    )[0]
    responses = [
        json.dumps(
            {
                "repairs": [
                    {
                        "chunk_id": first_chunk.chunk_id,
                        "original_claim": "red cube is currently grasped",
                        "replacement_claim": "red cube is currently not grasped",
                        "replacement_fact_ids": ["target_grasp"],
                    }
                ],
                "audit": {
                    "all_candidate_chunks_checked": True,
                    "used_only_authoritative_catalog": True,
                    "contains_action_advice": False,
                },
            }
        ),
        json.dumps(
            {
                "schema_version": (
                    envelope_replacement.ADJUDICATOR_SCHEMA_VERSION
                ),
                "decisions": [
                    {
                        "repair_index": 0,
                        "chunk_id": first_chunk.chunk_id,
                        "original_claim": "red cube is currently grasped",
                        "replacement_claim": "red cube is currently not grasped",
                        "relation": "contradicted",
                        "span_is_smallest_complete_clause": True,
                        "selected_facts_entail_claim_negation": True,
                        "replacement_claim_entailed_by_selected_facts": True,
                        "replacement_preserves_all_other_meaning": True,
                    }
                ],
                "audit": {
                    "all_proposals_checked": True,
                    "used_only_supplied_text_and_catalog": True,
                },
            }
        ),
        json.dumps(
            {
                "repairs": [],
                "audit": {
                    "all_candidate_chunks_checked": True,
                    "used_only_authoritative_catalog": True,
                    "contains_action_advice": False,
                },
            }
        ),
    ]

    def fake_request(*args, **kwargs):
        response = responses.pop(0)
        return response, {
            "seed": kwargs["seed"],
            "request_sha256": "request",
            "response_sha256": envelope_replacement._sha256_text(response),
            "transport": [],
        }

    monkeypatch.setattr(envelope_replacement, "_request", fake_request)
    envelope_replacement.initialize_trial_trace(
        output_dir=str(tmp_path),
        trial=1,
        task=task,
    )
    template = (
        "Executed:\n{executed_code}\n"
        "stdout:\n{console_stdout}\n"
        "stderr:\n{console_stderr}\n"
    )

    def native_prompt(code: str):
        evidence = template.format(
            executed_code=code,
            console_stdout="",
            console_stderr="",
        )
        return [{"role": "user", "content": [{"type": "text", "text": evidence}]}]

    first = envelope_replacement.run_envelope_replacement(
        _fake_env(grasped=False),
        SimpleNamespace(model="qwen3.7-plus"),
        output_dir=str(tmp_path),
        trial=1,
        turn=1,
        task=task,
        task_description="Goal: stack cubes.",
        multi_turn_prompt=template,
        executed_code_blocks=[code_0],
        executed_code=code_0,
        console_stdout="",
        console_stderr="",
        native_decision_prompt=native_prompt(code_0),
    )
    joined = "\n".join((code_0, code_1))
    second = envelope_replacement.run_envelope_replacement(
        _fake_env(grasped=False),
        SimpleNamespace(model="qwen3.7-plus"),
        output_dir=str(tmp_path),
        trial=1,
        turn=2,
        task=task,
        task_description="Goal: stack cubes.",
        multi_turn_prompt=template,
        executed_code_blocks=[code_0, code_1],
        executed_code=joined,
        console_stdout="",
        console_stderr="",
        native_decision_prompt=native_prompt(joined),
    )

    assert "currently not grasped" in json.dumps(first)
    assert "currently not grasped" in json.dumps(second)
    assert "attempt another placement" in json.dumps(second)
    record = json.loads(
        (
            tmp_path
            / envelope_replacement.TRACE_DIR
            / "trial_01"
            / "turn_02"
            / "replacement.json"
        ).read_text()
    )
    assert record["repair_count"] == 0
    assert record["active_persisted_repair_count"] == 1
    assert record["no_op_prompt_byte_equivalent"] is None
