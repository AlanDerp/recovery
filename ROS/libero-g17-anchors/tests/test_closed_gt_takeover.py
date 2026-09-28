from __future__ import annotations

import json

import numpy as np
import pytest

from capx.verification import closed_gt, closed_gt_takeover


def _body(
    position: tuple[float, float, float],
    *,
    half_extents: tuple[float, float, float] | None,
) -> closed_gt.BodyState:
    return closed_gt.BodyState(
        position=np.asarray(position, dtype=np.float64),
        orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        half_extents=(
            None
            if half_extents is None
            else np.asarray(half_extents, dtype=np.float64)
        ),
        linear_velocity=np.zeros(3, dtype=np.float64),
        angular_velocity=np.zeros(3, dtype=np.float64),
    )


def _snapshot(
    *,
    red: tuple[float, float, float],
    green: tuple[float, float, float],
    contact: bool,
    table_contacts: frozenset[str],
    holding: frozenset[tuple[str, str]] = frozenset(),
    executor_contacts: frozenset[tuple[str, str]] = frozenset(),
    aperture: float = 1.0,
) -> closed_gt.CompiledSnapshot:
    raw = closed_gt.RawSceneState(
        executors={"exe01": _body((0.0, 0.0, 0.20), half_extents=None)},
        objects={
            "obj01": _body(red, half_extents=(0.02, 0.02, 0.02)),
            "obj02": _body(green, half_extents=(0.02, 0.02, 0.02)),
        },
        executor_aperture={"exe01": aperture},
        holding=holding,
        executor_contacts=executor_contacts | holding,
        object_contacts=(
            frozenset({frozenset(("obj01", "obj02"))})
            if contact
            else frozenset()
        ),
        table_contacts=table_contacts,
        table_height=0.0,
    )
    return closed_gt.compile_snapshot(closed_gt.cube_scene_spec(), raw)


def _initial() -> closed_gt.CompiledSnapshot:
    return _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.0, 0.0, 0.061),
        contact=True,
        table_contacts=frozenset({"obj01"}),
    )


def _failed_after() -> closed_gt.CompiledSnapshot:
    return _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )


def _native_prompt(report: str) -> list[dict]:
    return [
        {"role": "system", "content": "system text"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Native report:\n{report}"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
                {"type": "text", "text": "Decision instruction"},
            ],
        },
    ]


def test_takeover_report_is_complete_reversible_and_sanitized() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _initial()
    after = _failed_after()
    transition = closed_gt.compile_transition(scene, before, after)

    rendered = closed_gt_takeover.render_takeover_report(
        scene, before, after, transition
    )

    assert len(rendered.current_fields) == 44
    assert set(rendered.changed_fields) | set(rendered.unchanged_fields) == set(
        rendered.current_fields
    )
    assert set(rendered.changed_fields).isdisjoint(rendered.unchanged_fields)
    assert len(rendered.pose_fields) == 3
    assert rendered.temporal_event_fields == ()
    assert len(rendered.interval_grasp_outcomes) == 2
    assert rendered.text.startswith(
        "Verified grasp result over the entire executed code:"
    )
    assert "red cube" in rendered.text
    assert "green cube" in rendered.text
    assert "Every listed current-state fact not listed as changed remained unchanged" in (
        rendered.text
    )
    assert "mutually exclusive grasp-contact state" in rendered.text
    assert " payload:" not in rendered.text
    assert " holding red cube:" not in rendered.text
    assert "grasp evidence" not in rendered.text
    for forbidden in (
        "exe01",
        "obj01",
        "obj02",
        "reward",
        "task success",
        "next action",
        "0.12",
    ):
        assert forbidden not in rendered.text.lower()


def test_ordered_events_expose_grasp_establishment_and_later_loss() -> None:
    scene = closed_gt.cube_scene_spec()
    initial = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    table_grasp = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    lifted = _snapshot(
        red=(0.0, 0.0, 0.10),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    dropped = _snapshot(
        red=(0.06, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        aperture=0.45,
    )
    recorder = closed_gt_takeover._SemanticEventRecorder(scene, initial, [])
    recorder.observe(table_grasp)
    recorder.observe(lifted)
    recorder.observe(dropped)
    rendered = closed_gt_takeover.render_takeover_report(
        scene,
        initial,
        dropped,
        closed_gt.compile_transition(scene, initial, dropped),
        tuple(recorder.pending),
    )

    table_index = rendered.text.index(
        "bilateral grasp contact, still supported by the table"
    )
    lift_index = rendered.text.index("bilateral grasp contact with lift confirmed")
    lost_index = rendered.text.rindex("changed from bilateral grasp contact")
    assert table_index < lift_index < lost_index
    assert "changed from bilateral grasp contact with lift confirmed to no executor contact" in (
        rendered.text
    )
    assert rendered.temporal_event_fields.count(
        "executor_object.exe01.obj01.grasp_contact_state"
    ) == 3
    assert dict(rendered.interval_grasp_outcomes)[
        "executor_object.exe01.obj01.grasp_contact_state"
    ] == "LIFT_CONFIRMED_THEN_LOST"


@pytest.mark.parametrize(
    ("observations", "expected"),
    (
        ((), "NO_EXECUTOR_CONTACT_OBSERVED"),
        (("contact_only", "initial"), "CONTACT_WITHOUT_BILATERAL_GRASP_OBSERVED"),
        (("table_grasp", "initial"), "BILATERAL_GRASP_WITHOUT_LIFT_THEN_LOST"),
        (("table_grasp",), "BILATERAL_GRASP_WITHOUT_LIFT_RETAINED"),
        (("lifted", "initial"), "LIFT_CONFIRMED_THEN_LOST"),
        (("lifted",), "LIFT_CONFIRMED_AND_RETAINED"),
    ),
)
def test_interval_grasp_outcome_is_mutually_exclusive_and_exhaustive(
    observations: tuple[str, ...], expected: str
) -> None:
    scene = closed_gt.cube_scene_spec()
    states = {
        "initial": _snapshot(
            red=(0.0, 0.0, 0.02),
            green=(0.12, 0.0, 0.02),
            contact=False,
            table_contacts=frozenset({"obj01", "obj02"}),
        ),
        "contact_only": _snapshot(
            red=(0.0, 0.0, 0.02),
            green=(0.12, 0.0, 0.02),
            contact=False,
            table_contacts=frozenset({"obj01", "obj02"}),
            executor_contacts=frozenset({("exe01", "obj01")}),
        ),
        "table_grasp": _snapshot(
            red=(0.0, 0.0, 0.02),
            green=(0.12, 0.0, 0.02),
            contact=False,
            table_contacts=frozenset({"obj01", "obj02"}),
            holding=frozenset({("exe01", "obj01")}),
            aperture=0.45,
        ),
        "lifted": _snapshot(
            red=(0.0, 0.0, 0.10),
            green=(0.12, 0.0, 0.02),
            contact=False,
            table_contacts=frozenset({"obj02"}),
            holding=frozenset({("exe01", "obj01")}),
            aperture=0.45,
        ),
    }
    recorder = closed_gt_takeover._SemanticEventRecorder(
        scene, states["initial"], []
    )
    for name in observations:
        recorder.observe(states[name])
    endpoint = states[observations[-1]] if observations else states["initial"]
    outcomes = dict(
        closed_gt_takeover._interval_grasp_outcomes(
            scene, states["initial"], endpoint, tuple(recorder.pending)
        )
    )

    assert outcomes["executor_object.exe01.obj01.grasp_contact_state"] == expected
    assert expected in closed_gt_takeover.INTERVAL_GRASP_OUTCOME_VALUES


def test_takeover_changes_only_one_native_report_carrier(tmp_path, monkeypatch) -> None:
    snapshots = iter((_initial(), _failed_after()))
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda env: next(snapshots))
    output_dir = str(tmp_path / "run")
    (tmp_path / "run").mkdir()
    env = object()
    report = "The scene did not visibly change."
    native = _native_prompt(report)

    closed_gt_takeover.initialize_trial_trace(
        env, output_dir=output_dir, trial=1, task="cube_stack"
    )
    corrected = closed_gt_takeover.run_takeover(
        env,
        output_dir=output_dir,
        trial=1,
        turn=1,
        task="cube_stack",
        native_report=report,
        native_decision_prompt=native,
        private_task_completed=False,
    )

    assert report in json.dumps(native)
    assert report not in json.dumps(corrected)
    assert corrected[0] == native[0]
    assert corrected[1]["content"][1] == native[1]["content"][1]
    assert corrected[1]["content"][2] == native[1]["content"][2]
    record = json.loads(
        (
            tmp_path
            / "run"
            / closed_gt_takeover.TRACE_DIR
            / "trial_01"
            / "turn_01"
            / "takeover.json"
        ).read_text()
    )
    assert record["current_state_field_count"] == 44
    assert record["changed_state_field_count"] + record[
        "unchanged_state_field_count"
    ] == 44
    assert len(record["public_transition"]) == 57
    assert record["interval_grasp_outcome_count"] == 2
    assert len(record["public_interval_grasp_outcomes"]) == 2
    assert record["non_text_structure_identical"] is True
    assert record["private_goal_iff_audit"]["iff_holds"] is True
    assert record["continuous_state_persisted"] is False
    assert record["completion_or_reward_visible"] is False


def test_takeover_fails_closed_on_ambiguous_carrier(tmp_path, monkeypatch) -> None:
    snapshots = iter((_initial(), _failed_after()))
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda env: next(snapshots))
    output_dir = str(tmp_path / "run")
    (tmp_path / "run").mkdir()
    report = "Repeated report."
    native = _native_prompt(report)
    native[0]["content"] = report
    closed_gt_takeover.initialize_trial_trace(
        object(), output_dir=output_dir, trial=2, task="cube_stack"
    )

    with pytest.raises(ValueError, match="expected one native report carrier, found 2"):
        closed_gt_takeover.run_takeover(
            object(),
            output_dir=output_dir,
            trial=2,
            turn=1,
            task="cube_stack",
            native_report=report,
            native_decision_prompt=native,
            private_task_completed=False,
        )


def test_takeover_fails_closed_on_goal_iff_mismatch(tmp_path, monkeypatch) -> None:
    snapshots = iter((_initial(), _failed_after()))
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda env: next(snapshots))
    output_dir = str(tmp_path / "run")
    (tmp_path / "run").mkdir()
    closed_gt_takeover.initialize_trial_trace(
        object(), output_dir=output_dir, trial=3, task="cube_stack"
    )

    with pytest.raises(RuntimeError, match="benchmark-goal IFF audit failed"):
        closed_gt_takeover.run_takeover(
            object(),
            output_dir=output_dir,
            trial=3,
            turn=1,
            task="cube_stack",
            native_report="The cubes moved.",
            native_decision_prompt=_native_prompt("The cubes moved."),
            private_task_completed=True,
        )


def test_takeover_rejects_tasks_without_an_audited_adapter(tmp_path) -> None:
    with pytest.raises(ValueError, match="no audited adapter"):
        closed_gt_takeover.initialize_trial_trace(
            object(), output_dir=str(tmp_path), trial=1, task="peg_insertion"
        )


def test_passive_capture_is_turn_synchronous_and_action_invisible(
    tmp_path, monkeypatch
) -> None:
    snapshots = iter((_initial(), _failed_after()))
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda env: next(snapshots))
    output_dir = str(tmp_path / "run")
    (tmp_path / "run").mkdir()

    closed_gt_takeover.initialize_passive_trial_trace(
        object(), output_dir=output_dir, trial=4, task="cube_restack"
    )
    assert (
        closed_gt_takeover.capture_passive_transition(
            object(),
            output_dir=output_dir,
            trial=4,
            turn=1,
            task="cube_restack",
        )
        is None
    )
    record = json.loads(
        (
            tmp_path
            / "run"
            / closed_gt_takeover.PASSIVE_TRACE_DIR
            / "trial_04"
            / "turn_01"
            / "transition.json"
        ).read_text()
    )
    assert record["mode"] == "passive_action_invisible"
    assert len(record["public_transition"]) == 57
    assert record["student_prompt_changed"] is False
    assert record["student_action_request_changed"] is False
