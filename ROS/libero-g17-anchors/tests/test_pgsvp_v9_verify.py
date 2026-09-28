from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from capx.verification import pgsvp_v9_verify
from capx.envs.trial import _episode_stop_reason
from capx.utils.launch_utils import MULTI_TURN_DECISION_INSTRUCTION


def _fake_env() -> SimpleNamespace:
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
            return object_geoms is self.cubeA

        def check_contact(self, first, second) -> bool:
            return second == "table_collision" and first is self.cubeB

    class FakeLowLevel:
        robosuite_env = FakeSimulator()

        def _cube_pose_dict(self, observation):
            return {
                "primary": [*observation["cubeA_pos"], 1.0, 0.0, 0.0, 0.0],
                "secondary": [*observation["cubeB_pos"], 1.0, 0.0, 0.0, 0.0],
            }

    return SimpleNamespace(low_level_env=FakeLowLevel())


def _pt() -> dict[str, str]:
    return {field: "unknown" for field in pgsvp_v9_verify.PUBLIC_FIELDS}


def _decision_prompt() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Task and latest execution evidence.\n"
                        "Please respond with EXACTLY ONE of the following:\n"
                        "- REGENERATE with code\n- FINISH"
                    ),
                },
                {"type": "text", "text": MULTI_TURN_DECISION_INSTRUCTION},
            ],
        }
    ]


def _verifier_response(message: str) -> str:
    fields = pgsvp_v9_verify._legacy_message_fields(message)
    return json.dumps(
        {
            "schema_version": "pgsvp_v9_pt_gt_verifier_v2",
            "private_verifier_audit": {
                "attachment_facts": {
                    "robot_target_attachment": "holding_target",
                    "target_goal_contact": "separated",
                },
                "axis_grounded_spatial_facts": {
                    "relative_x": "target_offset_positive_x",
                    "relative_y": "aligned",
                    "footprint": "target_center_outside_goal_footprint",
                    "contact_support": "target_separated_from_goal",
                },
            },
            "public_student_guidance": {"fields": fields},
            "leakage_audit": {
                "safe_public_guidance": True,
                "contains_exact_coordinates": False,
                "contains_numeric_constants": False,
                "contains_reward_value": False,
                "contains_trial_id_or_seed": False,
                "contains_hidden_object_id": False,
                "contains_simulator_internal_api": False,
                "contains_oracle_code": False,
                "contains_action_advice": False,
                "contains_repair_plan": False,
                "contains_imperative_language": False,
            },
        }
    )


def test_pt_worksheet_is_canonically_projected_to_closed_fields() -> None:
    assert pgsvp_v9_verify._validate_pt(json.dumps(_pt())) == _pt()
    invalid = _pt()
    invalid["EXTRA"] = "unknown"
    invalid["RUNTIME"] = ["NameError", "no stdout"]
    del invalid["FALSE_BELIEFS"]
    normalized, audit = pgsvp_v9_verify._normalize_pt(json.dumps(invalid))
    assert set(normalized) == set(pgsvp_v9_verify.PUBLIC_FIELDS)
    assert normalized["RUNTIME"] == '["NameError","no stdout"]'
    assert normalized["FALSE_BELIEFS"] == "unknown"
    assert audit["extra_fields_ignored"] == ["EXTRA"]
    assert audit["missing_fields_filled_unknown"] == ["FALSE_BELIEFS"]
    assert audit["non_string_fields_canonicalized"] == ["RUNTIME"]


def test_pt_uses_the_action_context_and_replaces_only_decision_request() -> None:
    prior = _pt()
    messages = pgsvp_v9_verify._build_student_pt_prompt(
        _decision_prompt(),
        prior,
    )
    serialized = json.dumps(messages)
    assert MULTI_TURN_DECISION_INSTRUCTION not in serialized
    assert "Please respond with EXACTLY ONE of the following:" not in serialized
    assert "REGENERATE with code" not in serialized
    assert "Task and latest execution evidence." in serialized
    assert serialized.count("PGSVP PERCEIVED-TRUTH WORKSHEET") == 1
    assert (
        pgsvp_v9_verify._count_text_occurrences(
            messages, '"TASK_RELATION": "unknown"'
        )
        == 1
    )


def test_pt_gt_delta_is_complete_ordered_and_exactly_reconstructs_gt() -> None:
    pt = _pt()
    verified = dict(pt)
    verified["ATTACHMENT"] = "holding_target"
    verified["RELATIVE_X"] = "target_offset_positive_x"
    corrections = pgsvp_v9_verify._pt_gt_corrections(pt, verified)
    assert tuple(corrections) == ("ATTACHMENT", "RELATIVE_X")
    assert pgsvp_v9_verify._apply_pt_gt_delta(pt, corrections) == verified
    message = pgsvp_v9_verify._render_delta_message(corrections)
    assert message == (
        "PGSVP_PT_GT_CORRECTIONS: ATTACHMENT: holding_target; "
        "RELATIVE_X: target_offset_positive_x;"
    )
    pgsvp_v9_verify._validate_delta_message(message, corrections)

    no_change = pgsvp_v9_verify._pt_gt_corrections(verified, verified)
    assert pgsvp_v9_verify._render_delta_message(no_change) == (
        "PGSVP_PT_GT_CORRECTIONS: NONE;"
    )


def test_verified_state_persists_as_next_turn_worksheet(tmp_path) -> None:
    trial_root = (
        tmp_path
        / "pgsvp_v9_verify_traces"
        / "trial_04"
        / "turn_01"
    )
    trial_root.mkdir(parents=True)
    state = _pt()
    state["ATTACHMENT"] = "holding_target"
    (trial_root / "verification.json").write_text(
        json.dumps({"verified_state": state}, indent=2, sort_keys=True) + "\n"
    )
    loaded = pgsvp_v9_verify._load_prior_verified_state(
        output_dir=str(tmp_path),
        trial=4,
        turn=2,
    )
    assert tuple(loaded) == pgsvp_v9_verify.PUBLIC_FIELDS
    assert loaded == state


def test_episode_boundaries_stop_before_another_verifier_or_action_call() -> None:
    assert (
        _episode_stop_reason(
            terminated=True, truncated=False, task_completed=False
        )
        == "terminated"
    )
    assert (
        _episode_stop_reason(
            terminated=False, truncated=False, task_completed=True
        )
        == "task_completed"
    )
    assert (
        _episode_stop_reason(
            terminated=False, truncated=True, task_completed=False
        )
        == "truncated"
    )
    assert (
        _episode_stop_reason(
            terminated=False, truncated=False, task_completed=False
        )
        is None
    )


def test_public_sanitizer_accepts_v9_facts_and_rejects_leaks() -> None:
    message = (
        "GT_BACKED_VERIFIER_FACTS: TASK_RELATION: red cube on green cube. "
        "RUNTIME: no runtime error. ATTACHMENT: robot holding red cube. "
        "RELATIVE_X: target offset positive x. RELATIVE_Y: aligned. "
        "RELATIVE_Z: target above goal. FOOTPRINT: target center outside goal "
        "footprint. CONTACT_SUPPORT: target separated from goal. "
        "STAGE_POSTCONDITIONS: acquire satisfied; align absent. "
        "EARLIEST_ABSENT_POSTCONDITION: align absent. "
        "DOMINANT_ABSENT_FACT: target center outside goal footprint. "
        "FALSE_BELIEFS: target was aligned."
    )
    privileged = pgsvp_v9_verify._privileged_packet(
        _fake_env(), task="cube_restack"
    )
    public, parsed, audit = pgsvp_v9_verify._canonicalize_verifier_output(
        _verifier_response(message), privileged
    )
    assert public.startswith(pgsvp_v9_verify.PUBLIC_HEADER)
    assert parsed["public_student_guidance"]["fields"]["ATTACHMENT"] == "holding_target"
    assert audit["candidate_format"] == "structured_fields"

    unsafe = json.loads(_verifier_response(message))
    unsafe["EXTRA_TOP_LEVEL"] = True
    unsafe["public_student_guidance"]["fields"]["FALSE_BELIEFS"] = (
        "Call open_gripper at coordinate 0.25."
    )
    public, parsed, audit = pgsvp_v9_verify._canonicalize_verifier_output(
        json.dumps(unsafe), privileged
    )
    assert parsed["public_student_guidance"]["fields"]["FALSE_BELIEFS"] == "unknown"
    assert "FALSE_BELIEFS" in audit["candidate_rejected_public_fields"]
    assert audit["candidate_top_level_extra_keys"] == ["EXTRA_TOP_LEVEL"]
    assert "open_gripper" not in public

    outcome = json.loads(_verifier_response(message))
    outcome["public_student_guidance"]["fields"]["RUNTIME"] = (
        "Execution completed; task incomplete."
    )
    outcome["public_student_guidance"]["fields"]["FALSE_BELIEFS"] = (
        "Runtime status is not success."
    )
    public, parsed, audit = pgsvp_v9_verify._canonicalize_verifier_output(
        json.dumps(outcome), privileged
    )
    assert parsed["public_student_guidance"]["fields"]["RUNTIME"] == "unknown"
    assert parsed["public_student_guidance"]["fields"]["FALSE_BELIEFS"] == "unknown"
    assert set(audit["candidate_rejected_public_fields"]) == {
        "FALSE_BELIEFS",
        "RUNTIME",
    }
    assert "task incomplete" not in public
    assert "not success" not in public


def test_private_packet_has_gt_cross_check_without_outcome_labels() -> None:
    packet = pgsvp_v9_verify._privileged_packet(
        _fake_env(), task="cube_restack"
    )
    assert packet["deterministic_cross_check"] == {
        "target_grasp": "target_grasped_by_robot",
        "relative_x": "target_center_positive_x_of_goal_band",
        "relative_y": "target_center_within_goal_y_band",
        "relative_z": "target_above_goal_height_band",
        "footprint": "target_center_outside_goal_footprint",
        "target_goal_contact": "target_separated_from_goal",
        "target_table_contact": "target_separated_from_table",
    }
    assert packet["reward_or_completion_oracle_provided"] is False
    serialized = json.dumps(packet)
    assert "reward_exact" not in serialized
    assert "task_completed" not in serialized


def test_private_verifier_facts_must_match_deterministic_gt() -> None:
    privileged = pgsvp_v9_verify._privileged_packet(
        _fake_env(), task="cube_restack"
    )
    parsed = json.loads(
        _verifier_response("GT_BACKED_VERIFIER_FACTS: factual state.")
    )
    aligned = pgsvp_v9_verify._validate_private_gt_alignment(parsed, privileged)
    assert aligned["relative_x"] == "target_offset_positive_x"

    parsed["private_verifier_audit"]["axis_grounded_spatial_facts"][
        "relative_x"
    ] = "aligned"
    with pytest.raises(ValueError, match="disagree"):
        pgsvp_v9_verify._validate_private_gt_alignment(parsed, privileged)


def test_canonicalizer_overrides_private_and_public_gt_mismatches() -> None:
    privileged = pgsvp_v9_verify._privileged_packet(
        _fake_env(), task="cube_restack"
    )
    candidate = json.loads(
        _verifier_response("GT_BACKED_VERIFIER_FACTS: factual state.")
    )
    candidate["private_verifier_audit"]["axis_grounded_spatial_facts"][
        "relative_x"
    ] = "aligned"
    candidate["public_student_guidance"]["fields"]["RELATIVE_X"] = "aligned"
    _, parsed, audit = pgsvp_v9_verify._canonicalize_verifier_output(
        json.dumps(candidate), privileged
    )
    assert (
        parsed["private_verifier_audit"]["axis_grounded_spatial_facts"][
            "relative_x"
        ]
        == "target_offset_positive_x"
    )
    assert (
        parsed["public_student_guidance"]["fields"]["RELATIVE_X"]
        == "target_offset_positive_x"
    )
    assert "relative_x" in audit["deterministic_gt_overrides"]
    pgsvp_v9_verify._validate_private_gt_alignment(parsed, privileged)


def test_protocol_retry_is_bounded_and_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(("not json", json.dumps(_pt())))

    def fake_request(args, messages, *, seed, max_tokens):
        return next(responses), {
            "seed": seed,
            "request_sha256": f"request-{seed}",
            "response_sha256": f"response-{seed}",
            "transport": [{"status_code": 200}],
        }

    monkeypatch.setattr(pgsvp_v9_verify, "_request", fake_request)
    content, parsed, audit, accepted_request = pgsvp_v9_verify._request_until_valid(
        SimpleNamespace(),
        [{"role": "user", "content": "worksheet"}],
        seed=1000,
        max_tokens=100,
        protocol_name="PT worksheet",
        parser=pgsvp_v9_verify._normalize_pt,
    )
    assert content == json.dumps(_pt())
    assert parsed[0] == _pt()
    assert audit["schema_attempt_count"] == 2
    assert audit["seed"] == 1100
    assert audit["schema_attempts"][0]["parse_error"].startswith("ValueError:")
    assert audit["schema_attempts"][1]["parse_error"] is None
    assert accepted_request[-2]["role"] == "assistant"
    assert accepted_request[-1]["role"] == "user"
    bridge, _ = pgsvp_v9_verify._load_public_bridge()
    clean_action_prompt = pgsvp_v9_verify._compose_student_action_prompt(
        student_decision_prompt=_decision_prompt(),
        materialized_state_response=pgsvp_v9_verify._canonical_pt_response(
            parsed[0]
        ),
        bridge=bridge,
    )
    assert "not json" not in json.dumps(clean_action_prompt)
    assert "PGSVP PERCEIVED-TRUTH WORKSHEET" not in json.dumps(
        clean_action_prompt
    )
    assert "PRIOR VERIFIED WORKSHEET" not in json.dumps(clean_action_prompt)
    assert "CLOSED PHYSICAL VALUES" not in json.dumps(clean_action_prompt)


def test_verification_round_exposes_only_materialized_pt_gt_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    message = (
        "GT_BACKED_VERIFIER_FACTS: TASK_RELATION: red cube on green cube. "
        "RUNTIME: no runtime error. ATTACHMENT: robot holding red cube. "
        "RELATIVE_X: target offset positive x. RELATIVE_Y: aligned. "
        "RELATIVE_Z: target above goal. FOOTPRINT: target center outside goal "
        "footprint. CONTACT_SUPPORT: target separated from goal. "
        "STAGE_POSTCONDITIONS: acquire satisfied; align absent. "
        "EARLIEST_ABSENT_POSTCONDITION: align absent. "
        "DOMINANT_ABSENT_FACT: target center outside goal footprint. "
        "FALSE_BELIEFS: target was aligned."
    )
    responses = iter((json.dumps(_pt()), _verifier_response(message)))

    def fake_request(args, messages, *, seed, max_tokens):
        content = next(responses)
        return content, {
            "seed": seed,
            "request_sha256": "request",
            "response_sha256": "response",
            "transport": [{"status_code": 200}],
        }

    monkeypatch.setattr(pgsvp_v9_verify, "_request", fake_request)
    monkeypatch.setenv("CAPX_GLOBAL_SEED", "20260718")
    pgsvp_v9_verify.initialize_trial_trace(
        _fake_env(),
        output_dir=str(tmp_path),
        trial=2,
        task="cube_restack",
    )
    action_prompt = pgsvp_v9_verify.run_verification_round(
        _fake_env(),
        SimpleNamespace(),
        output_dir=str(tmp_path),
        trial=2,
        turn=1,
        task="cube_restack",
        task_description="Place the red cube on the green cube.",
        execution_evidence="Visible execution output.",
        runtime_stderr="",
        visible_images=[],
        student_decision_prompt=_decision_prompt(),
    )
    pgsvp_v9_verify.record_student_prompt_audit(
        output_dir=str(tmp_path),
        trial=2,
        turn=1,
        decision_prompt=action_prompt,
    )
    serialized_prompt = json.dumps(action_prompt)
    assert pgsvp_v9_verify.PUBLIC_HEADER not in serialized_prompt
    assert serialized_prompt.count(pgsvp_v9_verify.PUBLIC_BRIDGE_MARKER) == 1
    assert serialized_prompt.count(pgsvp_v9_verify.DELTA_HEADER) == 0
    assert (
        pgsvp_v9_verify._count_text_occurrences(
            action_prompt, MULTI_TURN_DECISION_INSTRUCTION
        )
        == 1
    )
    trace = json.loads(
        (
            tmp_path
            / "pgsvp_v9_verify_traces"
            / "trial_02"
            / "turn_01"
            / "verification.json"
        ).read_text()
    )
    assert trace["pt_extraction_context_isolated_from_action_history"] is True
    assert trace["pt_extraction_request_visible_to_action"] is False
    assert trace["prior_verified_state_visible_to_action"] is False
    assert trace["closed_value_enums_visible_to_action"] is False
    assert trace["pt_owned_by_action_student"] is True
    assert trace["pt_persisted_as_assistant_message"] is False
    assert trace["raw_pt_visible_to_action"] is False
    assert trace["delta_visible_to_action"] is False
    assert trace["verifier_delta_retained_in_private_audit"] is True
    assert trace["action_delivery_is_materialized_state_only"] is True
    assert trace["materialized_state_persisted_as_assistant_message"] is True
    assert trace["full_gt_report_injected"] is False
    assert trace["materialized_verified_state_injected"] is True
    assert trace["pt_plus_delta_equals_verified_state"] is True
    assert trace["materialized_state_equals_verified_state"] is True
    assert trace["direct_gt_render_equals_materialized_render"] is True
    assert trace["student_prompt_audit_passed"] is True
    assert trace["public_bridge_is_materialized_factual_interface"] is True
    assert (
        trace["schema_version"]
        == "capx-pgsvp-v9-as-materialized-ptgt-v1"
    )
    assert trace["deterministic_private_alignment_passed"] is True
    assert trace["pt_normalization"]["canonical_projection_applied"] is False
    assert trace["verifier_interface_audit"]["final_renderer"] == "fixed_order_local_v1"
    assert trace["public_local_sanitizer"]["passed"] is True
    assert trace["pt_audit"]["seed"] != trace["verifier_audit"]["seed"]
    assert trace["fact_correction_count"] > 0
    assert any(
        message["role"] == "assistant"
        and message["content"] == trace["materialized_state_response"]
        for message in action_prompt
    )
    assert not any(
        message["role"] == "assistant"
        and message["content"] == trace["pt_canonical_response"]
        for message in action_prompt
    )
    assert trace["student_prompt_occurrences"] == {
        "action_instruction": 1,
        "bridge": 1,
        "closed_physical_values": 0,
        "delta": 0,
        "full_gt_report": 0,
        "materialized_state": 1,
        "prior_verified_worksheet": 0,
        "pt_request": 0,
    }


def test_materialized_action_prompt_equals_direct_gt_rendering() -> None:
    pt = _pt()
    verified = dict(pt)
    verified["ATTACHMENT"] = "holding_target"
    verified["RELATIVE_X"] = "target_offset_positive_x"
    corrections = pgsvp_v9_verify._pt_gt_corrections(pt, verified)
    materialized = pgsvp_v9_verify._apply_pt_gt_delta(pt, corrections)
    assert materialized == verified

    bridge, _ = pgsvp_v9_verify._load_public_bridge()
    from_delta = pgsvp_v9_verify._compose_student_action_prompt(
        student_decision_prompt=_decision_prompt(),
        materialized_state_response=pgsvp_v9_verify._canonical_pt_response(
            materialized
        ),
        bridge=bridge,
    )
    from_direct_gt = pgsvp_v9_verify._compose_student_action_prompt(
        student_decision_prompt=_decision_prompt(),
        materialized_state_response=pgsvp_v9_verify._canonical_pt_response(
            verified
        ),
        bridge=bridge,
    )
    assert from_delta == from_direct_gt
