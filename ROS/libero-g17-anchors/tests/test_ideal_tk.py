from __future__ import annotations

import json

from capx.verification import closed_gt, ideal_tk
from tests.test_closed_gt_takeover import _failed_after, _initial


class _LowLevel:
    def _notify_post_sim_step_observer(self) -> None:
        pass


class _Env:
    def __init__(self) -> None:
        self._apis = {}
        self._exec_globals = {}
        self.low_level_env = _LowLevel()


class _Api:
    def do_work(self) -> str:
        return "unchanged return"

    def functions(self) -> dict:
        return {"do_work": self.do_work}


TASK_CONTRACT = "Task and API contract"


def _native_prompt(context: str, report: str) -> list[dict]:
    return [
        {"role": "system", "content": "system text"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{TASK_CONTRACT}\n\nUnverified initial visual description",
                },
                {"type": "text", "text": f"Execution:\n{context}"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
                {"type": "text", "text": f"Native report:\n{report}"},
                {"type": "text", "text": "Decision instruction"},
            ],
        },
    ]


def test_scope_correct_context_omits_prior_pt_and_preserves_latest_code() -> None:
    context, action_code, removed_comments = ideal_tk.render_scope_correct_execution_context(
        ["first_call()  # stale belief", "# wrong current belief\nsecond_call()"],
        "latest output",
        "latest error",
    )

    assert "first_call()" not in context
    assert "stale belief" not in context
    assert "wrong current belief" in context
    assert "second_call()" in context
    assert action_code == "# wrong current belief\nsecond_call()"
    assert removed_comments == 0
    assert "belongs only to that action code" in context
    assert "latest output" in context
    assert "latest error" in context


def test_public_report_is_complete_reversible_and_without_advice() -> None:
    scene = closed_gt.cube_scene_spec()
    report, fields, _summary = ideal_tk.render_takeover_report(
        scene, _initial(), _failed_after(), (), (), ()
    )

    assert len(report.split()) > ideal_tk.PUBLIC_WORD_AUDIT_TARGET
    assert "Canonical derived state" in report
    assert "Canonical primitive factual basis" in report
    assert "red cube x position relative to green cube" in report
    assert "contact between red cube and green cube" in report
    assert "top access to red cube" in report
    assert "red cube support stability" in report
    assert "mutually exclusive grasp-contact state" in report
    assert "next move" not in report.lower()
    assert "reward" not in report.lower()
    assert "obj01" not in report
    assert len(fields) == len(closed_gt.predicate_specs(scene)) == 51
    assert ideal_tk.parse_canonical_current_state(scene, report) == (
        _failed_after().values
    )
    decoded_before, decoded_transition = ideal_tk.parse_canonical_transition(
        scene, report, _failed_after().values
    )
    assert decoded_before == _initial().values
    assert decoded_transition == closed_gt.compile_transition(
        scene, _initial(), _failed_after()
    ).values


def test_public_report_rejects_missing_or_conflicting_state_slot() -> None:
    scene = closed_gt.cube_scene_spec()
    report, _fields, _summary = ideal_tk.render_takeover_report(
        scene, _initial(), _failed_after(), (), (), ()
    )
    line = next(
        value
        for value in report.splitlines()
        if value.startswith("- robot gripper aperture = ")
    )

    with_missing = report.replace(line + "\n", "", 1)
    try:
        ideal_tk.parse_canonical_current_state(scene, with_missing)
    except ValueError as exc:
        assert "coverage mismatch" in str(exc)
    else:
        raise AssertionError("Missing canonical state slot did not fail closed")

    encoded_value = line.split(" = ", 1)[1].split(" (", 1)[0]
    replacement = next(
        value
        for value in closed_gt.APERTURE_VALUES
        if value != encoded_value
    )
    with_conflict = report.replace(line, line.replace(encoded_value, replacement), 1)
    try:
        ideal_tk.parse_canonical_current_state(scene, with_conflict)
    except ValueError as exc:
        assert "rendering does not match" in str(exc)
    else:
        raise AssertionError("Conflicting canonical state rendering did not fail closed")


def test_every_public_enum_domain_is_bijective_and_sanitized() -> None:
    scene = closed_gt.cube_scene_spec()
    private_ids = {
        entity.entity_id.upper()
        for entity in (*scene.executors, *scene.objects, scene.table)
    }
    for spec in closed_gt.predicate_specs(scene):
        decoder = ideal_tk._enum_decoder(scene, spec)
        assert len(decoder) == len(spec.values)
        assert set(decoder.values()) == set(spec.values)
        assert all(
            private_id not in public_value
            for public_value in decoder
            for private_id in private_ids
        )


def test_public_attention_target_is_audited_not_an_execution_gate() -> None:
    scene = closed_gt.cube_scene_spec()
    long_safe_report = "verified factual state " * (
        ideal_tk.PUBLIC_WORD_AUDIT_TARGET + 1
    )

    ideal_tk._validate_public_report(scene, long_safe_report)


def test_takeover_replaces_exactly_two_text_carriers(tmp_path, monkeypatch) -> None:
    snapshots = iter((_initial(), _failed_after()))
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda env: next(snapshots))
    output_dir = str(tmp_path / "run")
    (tmp_path / "run").mkdir()
    env = _Env()
    native_context = "Cumulative code and latest console result"
    native_report = "The scene did not visibly change."
    native = _native_prompt(native_context, native_report)

    ideal_tk.initialize_trial_trace(
        env, output_dir=output_dir, trial=1, task="cube_stack"
    )
    corrected = ideal_tk.run_takeover(
        env,
        output_dir=output_dir,
        trial=1,
        turn=1,
        task="cube_stack",
        task_description=TASK_CONTRACT,
        executed_code_blocks=["do_work()"],
        console_stdout="",
        console_stderr="",
        native_execution_context=native_context,
        native_report=native_report,
        native_decision_prompt=native,
        private_task_completed=False,
    )

    serialized = json.dumps(corrected)
    assert native_context not in serialized
    assert native_report not in serialized
    assert "Latest executed code block" in serialized
    assert "Authoritative factual reconstruction" in serialized
    assert "Unverified initial visual description" not in serialized
    assert corrected[1]["content"][0]["text"] == TASK_CONTRACT
    assert corrected[1]["content"][2] == native[1]["content"][2]
    record = json.loads(
        (
            tmp_path
            / "run"
            / ideal_tk.TRACE_DIR
            / "trial_01"
            / "turn_01"
            / "takeover.json"
        ).read_text()
    )
    assert record["latest_block_only_binding"] is True
    assert record["initial_visual_pt_suffix_removed"] is True
    assert record["initial_visual_pt_suffix_absent"] is True
    assert record["executed_code_execution_changed"] is False
    assert record["latest_executed_code_sha256"] == record["latest_action_code_sha256"]
    assert record["non_text_structure_identical"] is True
    assert record["private_goal_iff_audit"]["iff_holds"] is True
    assert record["public_goal_iff_audit"]["iff_holds"] is True
    assert record["public_goal_iff_audit"]["goal_definition_visible"] is False
    assert record["public_goal_iff_audit"]["private_poses_used"] is False
    assert record["public_current_state_field_count"] == 51
    assert record["public_current_state_roundtrip_equal"] is True
    assert record["public_prior_state_roundtrip_equal"] is True
    assert record["public_transition_field_count"] == 57
    assert record["public_transition_roundtrip_equal"] is True
    assert record["public_full_closure_visible"] is True
    assert record["public_current_state"] == record["full_current_state"]
    assert record["public_prior_state"] == record["full_prior_state"]
    assert record["public_transition"] == record["full_transition"]
    assert record["completion_or_reward_visible"] is False
    assert record["report_attention_target"] == ideal_tk.PUBLIC_WORD_AUDIT_TARGET
    assert record["report_attention_target_exceeded"] is True


def test_takeover_fails_closed_on_ambiguous_context(tmp_path, monkeypatch) -> None:
    snapshots = iter((_initial(), _failed_after()))
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda env: next(snapshots))
    output_dir = str(tmp_path / "run")
    (tmp_path / "run").mkdir()
    env = _Env()
    context = "duplicated context"
    report = "native report"
    native = _native_prompt(context, report)
    native[0]["content"] = context
    ideal_tk.initialize_trial_trace(
        env, output_dir=output_dir, trial=2, task="cube_stack"
    )

    try:
        ideal_tk.run_takeover(
            env,
            output_dir=output_dir,
            trial=2,
            turn=1,
            task="cube_stack",
            task_description=TASK_CONTRACT,
            executed_code_blocks=["do_work()"],
            console_stdout="",
            console_stderr="",
            native_execution_context=context,
            native_report=report,
            native_decision_prompt=native,
            private_task_completed=False,
        )
    except ValueError as exc:
        assert "expected one execution-context carrier, found 2" in str(exc)
    else:
        raise AssertionError("Ambiguous execution context did not fail closed")


def test_api_observer_preserves_return_value_and_records_call(tmp_path, monkeypatch) -> None:
    snapshots = iter((_initial(), _initial(), _failed_after(), _failed_after()))
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda env: next(snapshots))
    output_dir = str(tmp_path / "run")
    (tmp_path / "run").mkdir()
    env = _Env()
    env._apis = {"control": _Api()}
    context = "native execution context"
    report = "native report"
    ideal_tk.initialize_trial_trace(
        env, output_dir=output_dir, trial=3, task="cube_stack"
    )

    assert env._exec_globals["do_work"]() == "unchanged return"
    ideal_tk.run_takeover(
        env,
        output_dir=output_dir,
        trial=3,
        turn=1,
        task="cube_stack",
        task_description=TASK_CONTRACT,
        executed_code_blocks=["do_work()"],
        console_stdout="",
        console_stderr="",
        native_execution_context=context,
        native_report=report,
        native_decision_prompt=_native_prompt(context, report),
        private_task_completed=False,
    )
    record = json.loads(
        (
            tmp_path
            / "run"
            / ideal_tk.TRACE_DIR
            / "trial_03"
            / "turn_01"
            / "takeover.json"
        ).read_text()
    )
    assert [call["name"] for call in record["api_calls"]] == ["do_work"]
    assert record["api_calls"][0]["status"] == "returned"
    assert record["executed_code_execution_changed"] is False
    assert record["execution_scope_labels_changed"] is True


def test_contract_guard_preserves_collapsed_regeneration_carriers() -> None:
    context = "Latest action context"
    report = "Verified report"
    instruction = "Decision instruction"
    prompt = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{TASK_CONTRACT}\n\nUnverified visual PT\n\n"
                        f"{context}\n\n{report}\n\n{instruction}"
                    ),
                }
            ],
        },
    ]

    corrected, _native, removed = ideal_tk._restore_task_contract(
        prompt,
        task_description=TASK_CONTRACT,
        execution_context=context,
    )
    text = corrected[1]["content"][0]["text"]
    assert removed is True
    assert text == f"{TASK_CONTRACT}\n\n{context}\n\n{report}\n\n{instruction}"
