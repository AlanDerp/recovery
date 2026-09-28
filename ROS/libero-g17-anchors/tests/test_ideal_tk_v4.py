from __future__ import annotations

import inspect
import json
from dataclasses import asdict

import numpy as np
import pytest

from capx.verification import closed_gt, closed_gt_takeover, ideal_tk_v4


def _body(
    position: tuple[float, float, float],
    *,
    half_extents: tuple[float, float, float] | None,
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> closed_gt.BodyState:
    return closed_gt.BodyState(
        position=np.asarray(position, dtype=np.float64),
        orientation_wxyz=np.asarray(orientation_wxyz, dtype=np.float64),
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
    green_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> closed_gt.CompiledSnapshot:
    raw = closed_gt.RawSceneState(
        executors={"exe01": _body((0.0, 0.0, 0.20), half_extents=None)},
        objects={
            "obj01": _body(red, half_extents=(0.02, 0.02, 0.02)),
            "obj02": _body(
                green,
                half_extents=(0.02, 0.02, 0.02),
                orientation_wxyz=green_orientation,
            ),
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


def test_public_api_roles_cover_single_and_multi_executor_backends() -> None:
    scene = closed_gt.cube_scene_spec()
    bimanual = closed_gt.SceneSpec(
        executors=(
            closed_gt.EntitySpec("exe01", "left gripper", "executor"),
            closed_gt.EntitySpec("exe02", "right gripper", "executor"),
        ),
        objects=scene.objects,
        table=scene.table,
    )

    assert ideal_tk_v4._public_api_role("close_gripper") == "close_gripper"
    assert ideal_tk_v4._public_api_role("open_gripper_arm0") == "open_gripper"
    assert ideal_tk_v4._public_api_role("goto_pose_arm1") == "goto_pose"
    assert ideal_tk_v4._public_api_role("goto_pose_both") == "goto_pose"
    assert ideal_tk_v4._public_api_role("sample_grasp_pose") is None
    assert ideal_tk_v4._public_api_executor_id(scene, "close_gripper") == "exe01"
    assert (
        ideal_tk_v4._public_api_executor_id(bimanual, "close_gripper_arm1")
        == "exe02"
    )
    assert ideal_tk_v4._public_api_executor_id(bimanual, "close_gripper") is None
    assert ideal_tk_v4._api_addresses_executor(
        bimanual, "goto_pose_both", "exe01"
    )
    assert ideal_tk_v4._api_addresses_executor(
        bimanual, "goto_pose_both", "exe02"
    )


def test_inline_api_provenance_projects_exactly_to_frozen_v2_1() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _initial()
    after = _failed_after()
    transition = closed_gt.compile_transition(scene, before, after)
    key = "object_object.obj01.obj02.contact"
    changes = ((key, before.values[key], after.values[key]),)
    event = ideal_tk_v4.SemanticEventBatch(changes, "goto_pose")

    rendered = ideal_tk_v4.render_takeover_report(
        scene, before, after, transition, (event,)
    )
    frozen = closed_gt_takeover.render_takeover_report(
        scene,
        before,
        after,
        transition,
        (closed_gt_takeover.SemanticEventBatch(changes),),
    )

    assert "during public API `goto_pose`" in rendered.text
    assert "API-bound effects" not in rendered.text
    assert ideal_tk_v4.project_to_v2_1_report(rendered.text) == frozen.text


def test_api_wrapper_binds_observed_delta_to_active_public_call(monkeypatch) -> None:
    state = [_initial()]

    class LowLevel:
        def _notify_post_sim_step_observer(self) -> None:
            observer = getattr(self, "_capx_post_sim_step_observer", None)
            if observer is not None:
                observer()

    class Api:
        def act(self) -> None:
            state[0] = _failed_after()

        def functions(self) -> dict:
            return {"act": self.act}

    class Env:
        def __init__(self) -> None:
            self.low_level_env = LowLevel()
            self._apis = {"control": Api()}
            self._exec_globals = {}

    env = Env()
    recorder = ideal_tk_v4._SemanticEventRecorder(
        closed_gt.cube_scene_spec(), state[0], []
    )
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda _env: state[0])

    wrapped = ideal_tk_v4._restore_and_wrap_apis(env, recorder)
    env._exec_globals["act"]()

    assert wrapped == ("act",)
    assert recorder.active_api_calls == []
    assert len(recorder.pending) == 1
    assert recorder.pending[0].api_name == "act"
    assert recorder.pending[0].api_call_index == 0
    assert recorder.pending_api_calls == ["act"]
    assert len(recorder.pending_api_transitions) == 1
    assert recorder.pending_api_transitions[0].api_name == "act"
    assert recorder.pending_api_transitions[0].api_call_index == 0
    assert dict(recorder.pending_api_transitions[0].state_before) == _initial().values
    assert dict(recorder.pending_api_transitions[0].state_after) == _failed_after().values


def _raw_scene_for_extent_test() -> closed_gt.RawSceneState:
    return closed_gt.RawSceneState(
        executors={"exe01": _body((0.0, 0.0, 0.20), half_extents=None)},
        objects={
            "obj01": _body((0.0, 0.0, 0.02), half_extents=(0.02, 0.02, 0.02)),
            "obj02": _body((0.12, 0.0, 0.02), half_extents=(0.02, 0.02, 0.02)),
        },
        executor_aperture={"exe01": 1.0},
        holding=frozenset(),
        executor_contacts=frozenset(),
        object_contacts=frozenset(),
        table_contacts=frozenset({"obj01", "obj02"}),
        table_height=0.0,
    )


@pytest.mark.parametrize(
    ("extent", "expected_status"),
    [
        (np.array([0.05, 0.035, 0.045]), "CONSISTENT"),
        (np.array([1.20, 0.05, 0.04]), "INCONSISTENT"),
        (np.array([0.04, 0.04, 0.000001]), "INCONSISTENT"),
    ],
)
def test_extent_integrity_uses_instance_bound_gt_without_exposing_dimensions(
    monkeypatch, extent, expected_status
) -> None:
    monkeypatch.setattr(
        closed_gt,
        "extract_cube_raw_scene",
        lambda _env: _raw_scene_for_extent_test(),
    )

    audit = ideal_tk_v4._extent_integrity_audit(
        object(),
        closed_gt.cube_scene_spec(),
        api_call_index=4,
        args=("red cube",),
        kwargs={"return_bbox_extent": True},
        result=(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), extent),
    )

    assert audit is not None
    assert audit.object_id == "obj01"
    assert audit.status == expected_status
    assert audit.expected_signature_sha256 != audit.observed_signature_sha256


def test_extent_integrity_renderer_is_task_agnostic_and_sanitized() -> None:
    scene = closed_gt.cube_scene_spec()
    rendered = ideal_tk_v4._render_extent_integrity(scene, ("obj01",))
    source = inspect.getsource(ideal_tk_v4._extent_integrity_audit).lower()

    assert "red cube" in rendered
    assert "simulator-backed rigid geometry" in rendered
    assert not ideal_tk_v4._NUMERIC_TOKEN.search(rendered)
    assert "red cube" not in source
    assert "green cube" not in source
    assert "cube_stack" not in source
    assert "cube_restack" not in source


def test_api_wrapper_records_returned_extent_integrity(monkeypatch) -> None:
    state = _failed_after()

    class LowLevel:
        def _notify_post_sim_step_observer(self) -> None:
            pass

    class Api:
        def get_object_pose(self, object_name, return_bbox_extent=False):
            assert object_name == "red cube"
            extent = np.array([1.20, 0.05, 0.04]) if return_bbox_extent else None
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), extent

        def functions(self) -> dict:
            return {"get_object_pose": self.get_object_pose}

    class Env:
        def __init__(self) -> None:
            self.low_level_env = LowLevel()
            self._apis = {"control": Api()}
            self._exec_globals = {}

    env = Env()
    recorder = ideal_tk_v4._SemanticEventRecorder(
        closed_gt.cube_scene_spec(), state, []
    )
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda _env: state)
    monkeypatch.setattr(
        closed_gt,
        "extract_cube_raw_scene",
        lambda _env: _raw_scene_for_extent_test(),
    )

    ideal_tk_v4._restore_and_wrap_apis(env, recorder)
    env._exec_globals["get_object_pose"]("red cube", return_bbox_extent=True)

    assert len(recorder.pending_extent_integrity) == 1
    assert recorder.pending_extent_integrity[0].status == "INCONSISTENT"
    assert recorder.pending_extent_integrity[0].api_call_index == 0


@pytest.mark.parametrize(
    ("extent", "applied"),
    [
        (np.array([0.05, 0.035, 0.045]), False),
        (np.array([1.20, 0.05, 0.04]), True),
    ],
)
def test_observation_integrity_changes_only_inconsistent_g17_report(
    tmp_path, monkeypatch, extent, applied
) -> None:
    state = _failed_after()

    class LowLevel:
        def _notify_post_sim_step_observer(self) -> None:
            pass

    class Api:
        def get_object_pose(self, object_name, return_bbox_extent=False):
            value = extent if return_bbox_extent else None
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), value

        def functions(self) -> dict:
            return {"get_object_pose": self.get_object_pose}

    class Env:
        def __init__(self) -> None:
            self.low_level_env = LowLevel()
            self._apis = {"control": Api()}
            self._exec_globals = {}

    env = Env()
    output_dir = str(tmp_path / "run")
    (tmp_path / "run").mkdir()
    monkeypatch.setenv("CAPX_IDEAL_TK_V4_OBSERVATION_INTEGRITY", "1")
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda _env: state)
    monkeypatch.setattr(
        closed_gt,
        "extract_cube_raw_scene",
        lambda _env: _raw_scene_for_extent_test(),
    )
    ideal_tk_v4.initialize_trial_trace(
        env, output_dir=output_dir, trial=1, task="cube_restack"
    )
    env._exec_globals["get_object_pose"]("red cube", return_bbox_extent=True)
    native_report = "The native report is uncertain."
    corrected = ideal_tk_v4.run_takeover(
        env,
        output_dir=output_dir,
        trial=1,
        turn=1,
        task="cube_restack",
        native_report=native_report,
        native_decision_prompt=_native_prompt(native_report),
        private_task_completed=False,
        effect_alignment_enabled=False,
    )
    turn_root = (
        tmp_path
        / "run"
        / ideal_tk_v4.TRACE_DIR
        / "trial_01"
        / "turn_01"
    )
    record = json.loads((turn_root / "takeover.json").read_text())
    report = turn_root.joinpath("takeover_report.txt").read_text()

    assert record["observation_integrity_applied"] is applied
    assert record["g17_parent_prompt_preserved"] is (not applied)
    assert ("Verified native-observation integrity:" in report) is applied
    assert ("Verified native-observation integrity:" in json.dumps(corrected)) is applied
    assert record["numeric_state_visible"] is False


def test_takeover_report_is_complete_reversible_and_sanitized() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _initial()
    after = _failed_after()
    transition = closed_gt.compile_transition(scene, before, after)

    rendered = ideal_tk_v4.render_takeover_report(
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
    recorder = ideal_tk_v4._SemanticEventRecorder(scene, initial, [])
    recorder.observe(table_grasp)
    recorder.observe(lifted)
    recorder.observe(dropped)
    rendered = ideal_tk_v4.render_takeover_report(
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
    recorder = ideal_tk_v4._SemanticEventRecorder(
        scene, states["initial"], []
    )
    for name in observations:
        recorder.observe(states[name])
    endpoint = states[observations[-1]] if observations else states["initial"]
    outcomes = dict(
        ideal_tk_v4._interval_grasp_outcomes(
            scene, states["initial"], endpoint, tuple(recorder.pending)
        )
    )

    assert outcomes["executor_object.exe01.obj01.grasp_contact_state"] == expected
    assert expected in ideal_tk_v4.INTERVAL_GRASP_OUTCOME_VALUES


@pytest.mark.parametrize(
    ("observations", "expected"),
    (
        (("center_support",), "CENTER_SUPPORT_AT_ENDPOINT"),
        (("edge_support",), "EDGE_SUPPORT_AT_ENDPOINT"),
        (
            ("center_support", "outside"),
            "CENTER_SUPPORT_ESTABLISHED_THEN_LOST",
        ),
        (
            ("edge_support", "outside"),
            "EDGE_SUPPORT_ESTABLISHED_THEN_LOST",
        ),
        (
            ("center_without_support", "outside"),
            "CENTER_REGION_ENTERED_WITHOUT_SUPPORT",
        ),
        (
            ("edge_without_support", "outside"),
            "EDGE_REGION_ENTERED_WITHOUT_SUPPORT",
        ),
        (("outside",), "NEVER_ENTERED_SUPPORT_REGION"),
    ),
)
def test_placement_outcome_is_mutually_exclusive_and_exhaustive(
    observations: tuple[str, ...], expected: str
) -> None:
    states = {
        "outside": _snapshot(
            red=(0.08, 0.0, 0.10),
            green=(0.0, 0.0, 0.02),
            contact=False,
            table_contacts=frozenset({"obj02"}),
        ),
        "edge_without_support": _snapshot(
            red=(0.018, 0.0, 0.10),
            green=(0.0, 0.0, 0.02),
            contact=False,
            table_contacts=frozenset({"obj02"}),
        ),
        "center_without_support": _snapshot(
            red=(0.0, 0.0, 0.10),
            green=(0.0, 0.0, 0.02),
            contact=False,
            table_contacts=frozenset({"obj02"}),
        ),
        "edge_support": _snapshot(
            red=(0.018, 0.0, 0.061),
            green=(0.0, 0.0, 0.02),
            contact=True,
            table_contacts=frozenset({"obj02"}),
        ),
        "center_support": _snapshot(
            red=(0.0, 0.0, 0.061),
            green=(0.0, 0.0, 0.02),
            contact=True,
            table_contacts=frozenset({"obj02"}),
        ),
    }
    before = states["outside"]
    recorder = ideal_tk_v4._SemanticEventRecorder(
        closed_gt.cube_scene_spec(), before, []
    )
    for name in observations:
        recorder.observe(states[name])
    after = states[observations[-1]]

    outcome = closed_gt_takeover._placement_outcome_for_pair(
        before,
        after,
        tuple(recorder.pending),
        subject_id="obj01",
        reference_id="obj02",
    )

    assert outcome == expected
    assert outcome in closed_gt_takeover.INTERVAL_PLACEMENT_OUTCOME_VALUES


def test_compact_placement_outcome_is_instance_bound_and_hides_raw_region_events(
) -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.08, 0.0, 0.10),
        green=(0.0, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    supported = _snapshot(
        red=(0.0, 0.0, 0.061),
        green=(0.0, 0.0, 0.02),
        contact=True,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    after = _snapshot(
        red=(0.08, 0.0, 0.02),
        green=(0.0, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    recorder = ideal_tk_v4._SemanticEventRecorder(scene, before, [])
    recorder.observe(supported)
    recorder.observe(after)
    events = tuple(recorder.pending)
    transition = closed_gt.compile_transition(scene, before, after)

    rendered = ideal_tk_v4.render_takeover_report(
        scene, before, after, transition, events
    )
    frozen = closed_gt_takeover.render_takeover_report(
        scene,
        before,
        after,
        transition,
        tuple(closed_gt_takeover.SemanticEventBatch(batch.changes) for batch in events),
    )

    assert dict(rendered.interval_placement_outcomes) == {
        "placement_outcome.obj01.obj02": "CENTER_SUPPORT_ESTABLISHED_THEN_LOST"
    }
    assert (
        "red cube relative to green cube: center support was established and was "
        "later absent at the endpoint"
    ) in rendered.text
    assert "center in the support region of green cube changed" not in rendered.text
    assert not any(
        key.endswith("support_region") for key in rendered.temporal_event_fields
    )
    assert ideal_tk_v4.project_to_v2_1_report(rendered.text) == frozen.text
    assert "obj01" not in rendered.text
    assert "0.061" not in rendered.text


def test_release_bound_snapshot_preserves_action_conditioning_at_contact_loss() -> None:
    scene = closed_gt.cube_scene_spec()
    oblique = (0.9238795, 0.3826834, 0.0, 0.0)
    before = _snapshot(
        red=(0.08, 0.0, 0.10),
        green=(0.0, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
        green_orientation=oblique,
    )
    released = _snapshot(
        red=(0.018, 0.0, 0.061),
        green=(0.0, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        aperture=1.0,
        green_orientation=oblique,
    )
    supported = _snapshot(
        red=(0.018, 0.0, 0.061),
        green=(0.0, 0.0, 0.02),
        contact=True,
        table_contacts=frozenset({"obj02"}),
        aperture=1.0,
        green_orientation=oblique,
    )
    after = _snapshot(
        red=(0.08, 0.0, 0.02),
        green=(0.0, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        aperture=1.0,
        green_orientation=oblique,
    )
    recorder = ideal_tk_v4._SemanticEventRecorder(scene, before, [])
    recorder.active_api_calls.append(("open_gripper", 3))
    recorder.observe(released)
    recorder.observe(supported)
    recorder.active_api_calls.pop()
    recorder.observe(after)
    events = tuple(recorder.pending)
    calls = ("close_gripper", "get_object_pose", "goto_pose", "open_gripper")
    interval = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, after, events
    )
    placement = (
        (
            "placement_outcome.obj01.obj02",
            "EDGE_SUPPORT_ESTABLISHED_THEN_LOST",
        ),
    )

    outcome = ideal_tk_v4._release_bound_placement_outcome(
        scene,
        before,
        after,
        events,
        interval,
        placement,
        calls,
        'close_gripper()\nget_object_pose("green cube")\n'
        "goto_pose(place_pos, place_quat)\nopen_gripper()",
    )

    assert outcome is not None
    assert outcome.support_region == "EDGE_REGION"
    assert outcome.reference_orientation == "LOCAL_Z_OBLIQUE_TO_TABLE_NORMAL"
    assert outcome.support_at_contact_loss is False
    assert outcome.support_first_established == "DURING_RELEASE"
    assert outcome.support_at_endpoint is False
    gain = ideal_tk_v4._release_bound_semantic_gain(outcome, events, calls)
    assert gain.status == "novel_release_facts"
    assert gain.novel_fact_ids
    assert all(item.startswith("release_endpoint:") for item in gain.novel_fact_ids)
    rendered = ideal_tk_v4.render_takeover_report(
        scene,
        before,
        after,
        closed_gt.compile_transition(scene, before, after),
        events,
        placement_outcomes_override=placement,
        release_bound_placement=outcome,
    )
    assert "Verified release-bound placement result" in rendered.text
    assert "local z oblique to the table normal" in rendered.text
    assert "established later during release" in rendered.text
    assert "0.018" not in rendered.text
    assert "obj01" not in rendered.text


def test_semantic_gain_gate_is_silent_without_bound_release() -> None:
    gain = ideal_tk_v4._release_bound_semantic_gain(None, (), ())

    assert gain == ideal_tk_v4.ProjectionSemanticGain("no_bound_release", ())


def test_semantic_gain_gate_has_no_task_seed_or_object_name_branch() -> None:
    source = inspect.getsource(ideal_tk_v4._release_bound_semantic_gain).lower()

    for forbidden in (
        "cube_stack",
        "cube_restack",
        "red cube",
        "green cube",
        "trial",
        "seed",
    ):
        assert forbidden not in source


def test_release_bound_snapshot_is_silent_without_recorded_full_state() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.08, 0.0, 0.10),
        green=(0.0, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    after = _snapshot(
        red=(0.08, 0.0, 0.02),
        green=(0.0, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    grasp_key = "executor_object.exe01.obj01.grasp_contact_state"
    events = (
        ideal_tk_v4.SemanticEventBatch(
            ((grasp_key, before.values[grasp_key], after.values[grasp_key]),),
            "open_gripper",
            3,
        ),
    )
    interval = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, after, events
    )

    outcome = ideal_tk_v4._release_bound_placement_outcome(
        scene,
        before,
        after,
        events,
        interval,
        (("placement_outcome.obj01.obj02", "NEVER_ENTERED_SUPPORT_REGION"),),
        ("close_gripper", "get_object_pose", "goto_pose", "open_gripper"),
        'close_gripper()\nget_object_pose("green cube")\n'
        "goto_pose(place_pos, place_quat)\nopen_gripper()",
    )

    assert outcome is None


def test_k2_release_delta_reports_progress_and_preserved_failure() -> None:
    previous = ideal_tk_v4.ReleaseBoundPlacementOutcome(
        "obj01",
        "obj02",
        4,
        "WITHIN_REFERENCE_BAND",
        "WITHIN_REFERENCE_BAND",
        "POSITIVE_OF_REFERENCE_BAND",
        "EDGE_REGION",
        "INSIDE",
        "LOCAL_Z_OBLIQUE_TO_TABLE_NORMAL",
        "TABLE_SUPPORT",
        True,
        "AT_CONTACT_LOSS",
        False,
    )
    current = ideal_tk_v4.ReleaseBoundPlacementOutcome(
        "obj01",
        "obj02",
        7,
        "WITHIN_REFERENCE_BAND",
        "WITHIN_REFERENCE_BAND",
        "POSITIVE_OF_REFERENCE_BAND",
        "CENTER_REGION",
        "INSIDE",
        "LOCAL_Z_OBLIQUE_TO_TABLE_NORMAL",
        "TABLE_SUPPORT",
        True,
        "AT_CONTACT_LOSS",
        False,
    )

    delta = ideal_tk_v4._release_bound_placement_delta(
        asdict(previous), current
    )

    assert delta is not None
    rendered = "\n".join(
        ideal_tk_v4._render_release_bound_delta(
            closed_gt.cube_scene_spec(), delta
        )
    )
    assert "support region changed from in the edge region to in the center region" in rendered
    assert "endpoint support remained absent" in rendered
    assert "green cube orientation remained local z oblique" in rendered
    assert "next" not in rendered.lower()
    assert "offset" not in rendered.lower()
    ideal_tk_v4._validate_public_report(
        closed_gt.cube_scene_spec(), rendered
    )


def test_api_transition_ledger_exposes_bound_no_change_call() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = _snapshot(
        red=(0.08, 0.0, 0.10),
        green=(0.0, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    state = tuple(sorted(snapshot.values.items()))
    transitions = (
        ideal_tk_v4.ApiTransitionBatch("goto_pose", 0, state, state),
    )
    interval = ideal_tk_v4._interval_grasp_outcomes(
        scene, snapshot, snapshot, ()
    )

    ledger = ideal_tk_v4._api_transition_ledger(
        scene,
        closed_gt.cube_goal_definition("cube_stack"),
        interval,
        'sample_grasp_pose("red cube")\ngoto_pose(target, quat)',
        transitions,
        (),
        ("goto_pose",),
    )

    assert len(ledger) == 1
    assert ledger[0].endpoint_changes == ()
    rendered = "\n".join(
        ideal_tk_v4._render_api_transition_ledger(scene, ledger)
    )
    assert "no bound predicate changed" in rendered
    assert "red cube x position relative to green cube remained" in rendered
    assert "endpoint of executed public API `goto_pose`" in rendered
    ideal_tk_v4._validate_public_report(scene, rendered)


def test_api_transition_ledger_preserves_transient_loss_and_recovery() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = _snapshot(
        red=(0.08, 0.0, 0.10),
        green=(0.0, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    grasp_key = "executor_object.exe01.obj01.grasp_contact_state"
    held = snapshot.values[grasp_key]
    absent = "NO_EXECUTOR_CONTACT"
    events = (
        ideal_tk_v4.SemanticEventBatch(
            ((grasp_key, held, absent),), "goto_pose", 0
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((grasp_key, absent, held),), "goto_pose", 0
        ),
    )
    state = tuple(sorted(snapshot.values.items()))
    interval = ideal_tk_v4._interval_grasp_outcomes(
        scene, snapshot, snapshot, events
    )
    ledger = ideal_tk_v4._api_transition_ledger(
        scene,
        closed_gt.cube_goal_definition("cube_stack"),
        interval,
        'sample_grasp_pose("red cube")\ngoto_pose(target, quat)',
        (ideal_tk_v4.ApiTransitionBatch("goto_pose", 0, state, state),),
        events,
        ("goto_pose",),
    )

    assert ledger[0].transient_sequences == (
        (grasp_key, (held, absent, held)),
    )
    rendered = "\n".join(
        ideal_tk_v4._render_api_transition_ledger(scene, ledger)
    )
    assert "transient sequence" in rendered
    assert "no executor contact" in rendered


def test_phase_bound_sparse_ledger_uses_executor_object_facts_for_acquisition() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = _snapshot(
        red=(0.08, 0.0, 0.02),
        green=(0.0, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        aperture=0.0,
    )
    state = tuple(sorted(snapshot.values.items()))
    interval = ideal_tk_v4._interval_grasp_outcomes(
        scene, snapshot, snapshot, ()
    )

    ledger = ideal_tk_v4._api_transition_ledger(
        scene,
        closed_gt.cube_goal_definition("cube_stack"),
        interval,
        'sample_grasp_pose("red cube")\nclose_gripper()',
        (ideal_tk_v4.ApiTransitionBatch("close_gripper", 0, state, state),),
        (),
        ("close_gripper",),
        mode="phase_bound_sparse",
    )

    unchanged = {key for key, _value in ledger[0].endpoint_unchanged}
    assert "executor_object.exe01.obj01.relative_x" in unchanged
    assert "executor_object.exe01.obj01.grasp_contact_state" in unchanged
    assert "object_object.obj01.obj02.relative_x" not in unchanged
    rendered = "\n".join(
        ideal_tk_v4._render_api_transition_ledger(scene, ledger)
    )
    assert "robot gripper x position relative to red cube remained" in rendered
    assert "red cube x position relative to green cube" not in rendered


def test_phase_bound_sparse_ledger_uses_payload_goal_facts_during_transport() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = _snapshot(
        red=(0.08, 0.0, 0.10),
        green=(0.0, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    state = tuple(sorted(snapshot.values.items()))
    interval = ideal_tk_v4._interval_grasp_outcomes(
        scene, snapshot, snapshot, ()
    )

    ledger = ideal_tk_v4._api_transition_ledger(
        scene,
        closed_gt.cube_goal_definition("cube_stack"),
        interval,
        'sample_grasp_pose("red cube")\ngoto_pose(target, quat)',
        (ideal_tk_v4.ApiTransitionBatch("goto_pose", 0, state, state),),
        (),
        ("goto_pose",),
        mode="phase_bound_sparse",
    )

    unchanged = {key for key, _value in ledger[0].endpoint_unchanged}
    assert "object_object.obj01.obj02.relative_x" in unchanged
    assert "object_object.obj01.obj02.supported_by_reference" in unchanged
    assert "executor_object.exe01.obj01.relative_x" not in unchanged
    assert "executor_object.exe01.obj01.grasp_contact_state" in unchanged


def test_phase_bound_sparse_ledger_hides_support_stability_jitter() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = _snapshot(
        red=(0.08, 0.0, 0.10),
        green=(0.0, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    stability_key = "object.obj01.support_stability"
    current = snapshot.values[stability_key]
    events = (
        ideal_tk_v4.SemanticEventBatch(
            ((stability_key, current, "NO_STABLE_SUPPORT"),), "goto_pose", 0
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((stability_key, "NO_STABLE_SUPPORT", current),), "goto_pose", 0
        ),
    )
    state = tuple(sorted(snapshot.values.items()))
    interval = ideal_tk_v4._interval_grasp_outcomes(
        scene, snapshot, snapshot, events
    )

    ledger = ideal_tk_v4._api_transition_ledger(
        scene,
        closed_gt.cube_goal_definition("cube_stack"),
        interval,
        'sample_grasp_pose("red cube")\ngoto_pose(target, quat)',
        (ideal_tk_v4.ApiTransitionBatch("goto_pose", 0, state, state),),
        events,
        ("goto_pose",),
        mode="phase_bound_sparse",
    )

    assert ledger[0].transient_sequences == ()


@pytest.mark.parametrize("task", ("cube_stack", "cube_restack"))
def test_goal_ordered_object_pairs_come_from_relational_goal_atoms(task: str) -> None:
    scene = closed_gt.cube_scene_spec()

    pairs = ideal_tk_v4._goal_ordered_object_pairs(
        scene, closed_gt.cube_goal_definition(task)
    )

    assert pairs == frozenset({("obj01", "obj02")})


def test_salient_placement_outcomes_are_goal_bound_and_sequential() -> None:
    scene = closed_gt.cube_scene_spec()
    goal = closed_gt.cube_goal_definition("cube_restack")
    outcomes = (
        (
            "placement_outcome.obj01.obj02",
            "EDGE_SUPPORT_ESTABLISHED_THEN_LOST",
        ),
        (
            "placement_outcome.obj02.obj01",
            "CENTER_SUPPORT_ESTABLISHED_THEN_LOST",
        ),
        ("placement_outcome.obj01.obj02", "NEVER_ENTERED_SUPPORT_REGION"),
        ("placement_outcome.obj01.obj02", "CENTER_SUPPORT_AT_ENDPOINT"),
    )

    first_recovery = ideal_tk_v4._select_salient_placement_outcomes(
        scene, goal, outcomes, turn=1
    )
    later_recovery = ideal_tk_v4._select_salient_placement_outcomes(
        scene, goal, outcomes, turn=2
    )

    assert first_recovery == (outcomes[0],)
    assert later_recovery == (outcomes[0], outcomes[2])


def test_salient_placement_outcomes_reject_non_recovery_turn() -> None:
    with pytest.raises(ValueError, match="Recovery turn must be positive"):
        ideal_tk_v4._select_salient_placement_outcomes(
            closed_gt.cube_scene_spec(),
            closed_gt.cube_goal_definition("cube_stack"),
            (),
            turn=0,
        )


def test_effect_alignment_reports_grasp_retained_without_required_lift() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    after = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    event = ideal_tk_v4.SemanticEventBatch(
        ((key, before.values[key], after.values[key]),), "close_gripper"
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(scene, before, after, (event,))

    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (event,),
        outcomes,
        ("sample_grasp_pose", "goto_pose", "close_gripper", "goto_pose"),
        ('red_pos, red_quat = sample_grasp_pose("red cube")\nclose_gripper()\ngoto_pose(red_pos, red_quat)',),
    )

    assert aligned.status == "mismatch"
    assert aligned.contradicted_effects == ("LIFT_ESTABLISHED_BEFORE_RELEASE",)
    assert "bilateral grasp was retained, but lift was not established" in aligned.text
    assert "next" not in aligned.text.lower()

    grasp_loss_only = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (event,),
        outcomes,
        ("sample_grasp_pose", "goto_pose", "close_gripper", "goto_pose"),
        (
            'red_pos, red_quat = sample_grasp_pose("red cube")\n'
            "close_gripper()\n"
            "goto_pose(red_pos, red_quat)",
        ),
        mode="grasp_loss_only",
    )
    assert grasp_loss_only.status == "silent"
    assert grasp_loss_only.text == ""
    assert grasp_loss_only.contradicted_effects == (
        "LIFT_ESTABLISHED_BEFORE_RELEASE",
    )


def test_effect_alignment_is_byte_silent_when_operator_effects_hold() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    after = _snapshot(
        red=(0.0, 0.0, 0.10),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    event = ideal_tk_v4.SemanticEventBatch(
        ((key, before.values[key], after.values[key]),), "goto_pose"
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(scene, before, after, (event,))
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (event,),
        outcomes,
        ("sample_grasp_pose", "close_gripper", "goto_pose"),
        (
            'red_pos, red_quat = sample_grasp_pose("red cube")\n'
            "close_gripper()\n"
            "goto_pose(red_pos, red_quat)",
        ),
    )

    canonical = ideal_tk_v4.render_takeover_report(
        scene,
        before,
        after,
        closed_gt.compile_transition(scene, before, after),
        (event,),
    ).text
    treatment = canonical + (f"\n\n{aligned.text}" if aligned.text else "")
    assert aligned.status == "silent"
    assert treatment.encode() == canonical.encode()


def test_effect_alignment_distinguishes_planned_from_executed_motion() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    after = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    event = ideal_tk_v4.SemanticEventBatch(
        ((key, before.values[key], after.values[key]),), "close_gripper"
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(scene, before, after, (event,))
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (event,),
        outcomes,
        ("sample_grasp_pose", "goto_pose", "close_gripper"),
        (
            'red_pos, red_quat = sample_grasp_pose("red cube")\n'
            "goto_pose(red_pos, red_quat)\n"
            "close_gripper()\n"
            "goto_pose(lift_pos, red_quat)",
        ),
    )

    assert aligned.status == "mismatch"
    assert "planned post-grasp motion" in aligned.text
    assert "was not executed" in aligned.text
    assert "bilateral grasp remained established" in aligned.text


def test_effect_alignment_reports_loss_during_motion_but_not_release() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.10),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    after = _snapshot(
        red=(0.06, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        aperture=0.45,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    change = ((key, before.values[key], after.values[key]),)
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene,
        before,
        after,
        (ideal_tk_v4.SemanticEventBatch(change, "goto_pose"),),
    )
    lost = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (ideal_tk_v4.SemanticEventBatch(change, "goto_pose"),),
        outcomes,
        ("goto_pose",),
        ("goto_pose(target_pos, target_quat)",),
        mode="grasp_loss_only",
    )
    released = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (ideal_tk_v4.SemanticEventBatch(change, "open_gripper"),),
        outcomes,
        ("goto_pose", "open_gripper"),
        ("goto_pose(target_pos, target_quat)\nopen_gripper()",),
    )

    assert lost.status == "mismatch"
    assert lost.contradicted_effects == ("GRASP_RETAINED_DURING_POST_GRASP_MOTION",)
    assert "lost during public API `goto_pose` before release" in lost.text
    assert released.status == "silent"
    assert released.text == ""


def test_instance_bound_loss_ignores_old_episode_before_last_close() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.10),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    after = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        aperture=1.0,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    held = before.values[key]
    absent = after.values[key]
    events = (
        ideal_tk_v4.SemanticEventBatch(((key, held, absent),), "home_pose", 0),
        ideal_tk_v4.SemanticEventBatch(((key, absent, held),), "home_pose", 0),
        ideal_tk_v4.SemanticEventBatch(((key, held, absent),), "home_pose", 0),
    )
    calls = (
        "home_pose",
        "sample_grasp_pose",
        "goto_pose",
        "close_gripper",
        "goto_pose",
        "open_gripper",
    )
    code = (
        'sample_grasp_pose("red cube")\n'
        "close_gripper()\n"
        "goto_pose(lift_pos, lift_quat)\n"
        "open_gripper()"
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, after, events
    )

    legacy = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        events,
        outcomes,
        calls,
        (code,),
        mode="grasp_loss_only",
    )
    bound = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        events,
        outcomes,
        calls,
        (code,),
        mode="instance_bound_grasp_loss_only",
    )
    additive = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        events,
        outcomes,
        calls,
        (
            'sample_grasp_pose("red cube")\n'
            "close_gripper()\n"
            "goto_pose(lift_pos, lift_quat)\n"
            "open_gripper()",
        ),
        mode="instance_bound_grasp_loss_plus_off_table_lift_regression",
    )

    assert legacy.status == "mismatch"
    assert "public API `home_pose`" in legacy.text
    assert bound.status == "silent"
    assert bound.text == ""
    assert bound.grasp_episode_binding_status == "bound_no_grasp"
    assert bound.grasp_episode_start_call_index == 3
    assert bound.grasp_episode_release_call_index == 5
    assert additive.status == "silent"
    assert additive.text == ""


def test_instance_bound_loss_reports_same_acquisition_episode() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
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
    after = _snapshot(
        red=(0.06, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        aperture=0.45,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    events = (
        ideal_tk_v4.SemanticEventBatch(
            ((key, before.values[key], table_grasp.values[key]),),
            "close_gripper",
            2,
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((key, table_grasp.values[key], lifted.values[key]),),
            "goto_pose",
            3,
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((key, lifted.values[key], after.values[key]),),
            "goto_pose",
            3,
        ),
    )
    calls = (
        "sample_grasp_pose",
        "goto_pose",
        "close_gripper",
        "goto_pose",
        "open_gripper",
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, after, events
    )
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        events,
        outcomes,
        calls,
        (
            'sample_grasp_pose("red cube")\n'
            "close_gripper()\n"
            "goto_pose(lift_pos, lift_quat)\n"
            "open_gripper()",
        ),
        mode="instance_bound_grasp_loss_only",
    )

    additive = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        events,
        outcomes,
        calls,
        (
            'sample_grasp_pose("red cube")\n'
            "close_gripper()\n"
            "goto_pose(lift_pos, lift_quat)\n"
            "open_gripper()",
        ),
        mode="instance_bound_grasp_loss_plus_off_table_lift_regression",
    )

    assert aligned.status == "mismatch"
    assert "public API `goto_pose` before release" in aligned.text
    assert aligned.grasp_episode_binding_status == "bound"
    assert aligned.grasp_episode_start_call_index == 2
    assert aligned.grasp_episode_release_call_index == 4
    assert aligned.grasp_episode_acquisition_call_index == 2
    assert aligned.grasp_episode_loss_call_index == 3
    assert additive.status == "mismatch"
    assert additive.text == aligned.text


def test_instance_bound_outcome_reports_failed_grasp_acquisition() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    calls = (
        "sample_grasp_pose",
        "goto_pose",
        "close_gripper",
        "goto_pose",
        "open_gripper",
    )
    code = (
        'sample_grasp_pose("red cube")\n'
        "goto_pose(grasp_pos, grasp_quat)\n"
        "close_gripper()\n"
        "goto_pose(lift_pos, grasp_quat)\n"
        "open_gripper()"
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, before, ()
    )

    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        before,
        (),
        outcomes,
        calls,
        (code,),
        mode="instance_bound_grasp_outcome_only",
    )
    loss_only = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        before,
        (),
        outcomes,
        calls,
        (code,),
        mode="instance_bound_grasp_loss_only",
    )
    transport_only = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        before,
        (),
        outcomes,
        calls,
        (code,),
        mode="instance_bound_transport_failure_only",
    )

    assert aligned.status == "mismatch"
    assert aligned.grasp_episode_binding_status == "bound_no_grasp"
    assert aligned.grasp_episode_start_call_index == 2
    assert aligned.grasp_episode_lift_call_index is None
    assert "executed close-gripper effect" in aligned.text
    assert "x, y, and z positions" in aligned.text
    assert dict(aligned.grasp_episode_acquisition_endpoint) == {
        "executor_object.exe01.obj01.contact": "FALSE",
        "executor_object.exe01.obj01.relative_x": "WITHIN_REFERENCE_BAND",
        "executor_object.exe01.obj01.relative_y": "WITHIN_REFERENCE_BAND",
        "executor_object.exe01.obj01.relative_z": "POSITIVE_OF_REFERENCE_BAND",
    }
    assert "lift" not in aligned.text.lower()
    assert loss_only.status == "silent"
    assert loss_only.text == ""
    assert transport_only.status == "silent"
    assert transport_only.text == ""


def test_instance_bound_outcome_reports_grasp_without_lift() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    grasped = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    event = ideal_tk_v4.SemanticEventBatch(
        ((key, before.values[key], grasped.values[key]),),
        "close_gripper",
        2,
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, grasped, (event,)
    )
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        grasped,
        (event,),
        outcomes,
        ("sample_grasp_pose", "goto_pose", "close_gripper", "goto_pose"),
        (
            'sample_grasp_pose("red cube")\n'
            "goto_pose(grasp_pos, grasp_quat)\n"
            "close_gripper()\n"
            "goto_pose(lift_pos, grasp_quat)",
        ),
        mode="instance_bound_grasp_outcome_only",
    )
    transport_only = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        grasped,
        (event,),
        outcomes,
        ("sample_grasp_pose", "goto_pose", "close_gripper", "goto_pose"),
        (
            'sample_grasp_pose("red cube")\n'
            "goto_pose(grasp_pos, grasp_quat)\n"
            "close_gripper()\n"
            "goto_pose(lift_pos, grasp_quat)",
        ),
        mode="instance_bound_transport_failure_only",
    )

    assert aligned.status == "mismatch"
    assert aligned.grasp_episode_binding_status == "bound"
    assert aligned.grasp_episode_acquisition_call_index == 2
    assert aligned.grasp_episode_lift_call_index is None
    assert "lift was not established" in aligned.text
    assert "did not establish bilateral grasp" not in aligned.text
    assert transport_only.status == "mismatch"
    assert "lift was not established" in transport_only.text
    assert "did not establish bilateral grasp" not in transport_only.text


def test_instance_bound_outcome_does_not_fail_an_unattempted_lift() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    grasped = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    event = ideal_tk_v4.SemanticEventBatch(
        ((key, before.values[key], grasped.values[key]),),
        "close_gripper",
        2,
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, grasped, (event,)
    )
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        grasped,
        (event,),
        outcomes,
        ("sample_grasp_pose", "goto_pose", "close_gripper", "get_object_pose"),
        (
            'sample_grasp_pose("red cube")\n'
            "goto_pose(grasp_pos, grasp_quat)\n"
            "close_gripper()\n"
            "goto_pose(lift_pos, grasp_quat)",
        ),
        mode="instance_bound_grasp_outcome_only",
    )

    assert aligned.status == "silent"
    assert aligned.grasp_episode_binding_status == "bound"
    assert aligned.grasp_episode_acquisition_call_index == 2
    assert aligned.grasp_episode_lift_call_index is None
    assert aligned.text == ""


def test_instance_bound_outcome_is_silent_when_grasp_and_lift_hold() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    grasped = _snapshot(
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
    key = "executor_object.exe01.obj01.grasp_contact_state"
    events = (
        ideal_tk_v4.SemanticEventBatch(
            ((key, before.values[key], grasped.values[key]),),
            "close_gripper",
            2,
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((key, grasped.values[key], lifted.values[key]),),
            "goto_pose",
            3,
        ),
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, lifted, events
    )
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        lifted,
        events,
        outcomes,
        ("sample_grasp_pose", "goto_pose", "close_gripper", "goto_pose"),
        (
            'sample_grasp_pose("red cube")\n'
            "goto_pose(grasp_pos, grasp_quat)\n"
            "close_gripper()\n"
            "goto_pose(lift_pos, grasp_quat)",
        ),
        mode="instance_bound_grasp_outcome_only",
    )

    assert aligned.status == "silent"
    assert aligned.text == ""
    assert aligned.grasp_episode_binding_status == "bound"
    assert aligned.grasp_episode_acquisition_call_index == 2
    assert aligned.grasp_episode_lift_call_index == 3


def test_off_table_lift_regression_reports_only_a_true_temporal_regression() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    grasped = _snapshot(
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
    regressed = _snapshot(
        red=(0.0, 0.0, 0.04),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    events = (
        ideal_tk_v4.SemanticEventBatch(
            ((key, before.values[key], grasped.values[key]),),
            "close_gripper",
            2,
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((key, grasped.values[key], lifted.values[key]),),
            "goto_pose",
            3,
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((key, lifted.values[key], regressed.values[key]),),
            "goto_pose",
            3,
        ),
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, regressed, events
    )
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        regressed,
        events,
        outcomes,
        ("sample_grasp_pose", "goto_pose", "close_gripper", "goto_pose"),
        (
            'sample_grasp_pose("red cube")\n'
            "goto_pose(grasp_pos, grasp_quat)\n"
            "close_gripper()\n"
            "goto_pose(lift_pos, grasp_quat)",
        ),
        mode="instance_bound_off_table_lift_regression_only",
    )
    additive = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        regressed,
        events,
        outcomes,
        ("sample_grasp_pose", "goto_pose", "close_gripper", "goto_pose"),
        (
            'sample_grasp_pose("red cube")\n'
            "goto_pose(grasp_pos, grasp_quat)\n"
            "close_gripper()\n"
            "goto_pose(lift_pos, grasp_quat)",
        ),
        mode="instance_bound_grasp_loss_plus_off_table_lift_regression",
    )

    assert aligned.status == "mismatch"
    assert aligned.grasp_episode_binding_status == "bound"
    assert aligned.grasp_episode_lift_call_index is None
    assert aligned.contradicted_effects == (
        "LIFT_CONFIRMED_STATE_RETAINED_AT_ENDPOINT",
    )
    assert "occurred" in aligned.text
    assert "was not retained at the public-API endpoint" in aligned.text
    assert "without table contact" in aligned.text
    assert "lift was not established" not in aligned.text
    assert additive.status == "mismatch"
    assert additive.text == aligned.text


def test_off_table_lift_regression_is_silent_after_return_to_table() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    grasped = _snapshot(
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
    key = "executor_object.exe01.obj01.grasp_contact_state"
    events = (
        ideal_tk_v4.SemanticEventBatch(
            ((key, before.values[key], grasped.values[key]),),
            "close_gripper",
            2,
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((key, grasped.values[key], lifted.values[key]),),
            "goto_pose",
            3,
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((key, lifted.values[key], grasped.values[key]),),
            "goto_pose",
            3,
        ),
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, grasped, events
    )
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        grasped,
        events,
        outcomes,
        ("sample_grasp_pose", "goto_pose", "close_gripper", "goto_pose"),
        (
            'sample_grasp_pose("red cube")\n'
            "goto_pose(grasp_pos, grasp_quat)\n"
            "close_gripper()\n"
            "goto_pose(lift_pos, grasp_quat)",
        ),
        mode="instance_bound_off_table_lift_regression_only",
    )

    assert aligned.status == "silent"
    assert aligned.text == ""
    assert "LIFT_ESTABLISHED_BEFORE_RELEASE" not in aligned.contradicted_effects


def test_instance_bound_loss_excludes_release_detach() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.10),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    after = _snapshot(
        red=(0.12, 0.0, 0.06),
        green=(0.12, 0.0, 0.02),
        contact=True,
        table_contacts=frozenset({"obj02"}),
        aperture=1.0,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    event = ideal_tk_v4.SemanticEventBatch(
        ((key, before.values[key], after.values[key]),),
        "open_gripper",
        1,
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, after, (event,)
    )
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (event,),
        outcomes,
        ("goto_pose", "open_gripper"),
        ("goto_pose(place_pos, place_quat)\nopen_gripper()",),
        mode="instance_bound_grasp_loss_only",
    )
    assert aligned.status == "silent"
    assert aligned.grasp_episode_binding_status == "bound"
    assert aligned.grasp_episode_release_call_index == 1
    assert aligned.grasp_episode_loss_call_index is None


def test_instance_bound_loss_collapses_transient_detach_within_public_api() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    absent = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
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
    key = "executor_object.exe01.obj01.grasp_contact_state"
    events = (
        ideal_tk_v4.SemanticEventBatch(
            ((key, before.values[key], absent.values[key]),),
            "goto_pose",
            0,
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((key, absent.values[key], lifted.values[key]),),
            "goto_pose",
            0,
        ),
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, lifted, events
    )
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        lifted,
        events,
        outcomes,
        ("goto_pose", "open_gripper"),
        ("goto_pose(lift_pos, grasp_quat)\nopen_gripper()",),
        mode="instance_bound_grasp_loss_only",
    )
    additive = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        lifted,
        events,
        outcomes,
        ("goto_pose", "open_gripper"),
        ("goto_pose(lift_pos, grasp_quat)\nopen_gripper()",),
        mode="instance_bound_grasp_loss_plus_off_table_lift_regression",
    )

    assert aligned.status == "silent"
    assert aligned.grasp_episode_binding_status == "bound"
    assert aligned.object_id == "obj01"
    assert aligned.grasp_episode_lift_call_index == 0
    assert aligned.grasp_episode_loss_call_index is None
    assert additive.status == "mismatch"
    assert (
        "was interrupted during public API `goto_pose` before release"
        in additive.text
    )
    assert "was re-established by that API's endpoint" in additive.text
    assert additive.grasp_episode_loss_call_index is None
    assert additive.grasp_episode_recovered_loss_call_index == 0


def test_instance_bound_loss_preserves_complete_failure_within_public_api() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
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
    after = _snapshot(
        red=(0.06, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        aperture=0.45,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    events = (
        ideal_tk_v4.SemanticEventBatch(
            ((key, before.values[key], table_grasp.values[key]),),
            "goto_pose",
            3,
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((key, table_grasp.values[key], lifted.values[key]),),
            "goto_pose",
            3,
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((key, lifted.values[key], after.values[key]),),
            "goto_pose",
            3,
        ),
    )
    calls = (
        "sample_grasp_pose",
        "goto_pose",
        "close_gripper",
        "goto_pose",
        "open_gripper",
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, after, events
    )
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        events,
        outcomes,
        calls,
        (
            'sample_grasp_pose("red cube")\n'
            "goto_pose(grasp_pos, grasp_quat)\n"
            "close_gripper()\n"
            "goto_pose(lift_pos, grasp_quat)\n"
            "open_gripper()",
        ),
        mode="instance_bound_grasp_loss_plus_off_table_lift_regression",
    )

    assert aligned.status == "mismatch"
    assert "public API `goto_pose` before release" in aligned.text
    assert aligned.grasp_episode_binding_status == "bound"
    assert aligned.grasp_episode_acquisition_call_index == 3
    assert aligned.grasp_episode_loss_call_index == 3

    no_lift_events = (
        ideal_tk_v4.SemanticEventBatch(
            ((key, before.values[key], table_grasp.values[key]),),
            "goto_pose",
            3,
        ),
        ideal_tk_v4.SemanticEventBatch(
            ((key, table_grasp.values[key], after.values[key]),),
            "goto_pose",
            3,
        ),
    )
    no_lift = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        no_lift_events,
        ideal_tk_v4._interval_grasp_outcomes(
            scene, before, after, no_lift_events
        ),
        calls,
        (
            'sample_grasp_pose("red cube")\n'
            "goto_pose(grasp_pos, grasp_quat)\n"
            "close_gripper()\n"
            "goto_pose(lift_pos, grasp_quat)\n"
            "open_gripper()",
        ),
        mode="instance_bound_grasp_loss_plus_off_table_lift_regression",
    )

    assert no_lift.status == "mismatch"
    assert "public API `goto_pose` before release" in no_lift.text
    assert no_lift.grasp_episode_acquisition_call_index == 3
    assert no_lift.grasp_episode_loss_call_index == 3


def test_instance_bound_loss_excludes_detach_during_close_call() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.10),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    after = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        aperture=0.45,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    event = ideal_tk_v4.SemanticEventBatch(
        ((key, before.values[key], after.values[key]),),
        "close_gripper",
        0,
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, after, (event,)
    )
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (event,),
        outcomes,
        ("close_gripper", "goto_pose", "open_gripper"),
        ("close_gripper()\ngoto_pose(lift_pos, lift_quat)\nopen_gripper()",),
        mode="instance_bound_grasp_loss_only",
    )

    assert aligned.status == "silent"
    assert aligned.grasp_episode_binding_status == "bound"
    assert aligned.grasp_episode_start_call_index == 0
    assert aligned.grasp_episode_release_call_index == 2
    assert aligned.grasp_episode_loss_call_index is None


def test_instance_bound_loss_is_silent_for_ambiguous_legacy_provenance() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.10),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    after = _snapshot(
        red=(0.06, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        aperture=0.45,
    )
    key = "executor_object.exe01.obj01.grasp_contact_state"
    event = ideal_tk_v4.SemanticEventBatch(
        ((key, before.values[key], after.values[key]),), "goto_pose"
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(
        scene, before, after, (event,)
    )
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (event,),
        outcomes,
        ("goto_pose", "goto_pose"),
        ("goto_pose(mid, quat)\ngoto_pose(target, quat)",),
        mode="instance_bound_grasp_loss_only",
    )

    assert aligned.status == "silent"
    assert aligned.grasp_episode_binding_status == "ambiguous_unindexed_event"
    assert aligned.grasp_episode_loss_call_index is None


def test_effect_alignment_rejects_unknown_gt_outcome() -> None:
    with pytest.raises(ValueError, match="Unknown interval grasp outcome"):
        ideal_tk_v4.align_public_operator_effects(
            closed_gt.cube_scene_spec(),
            _initial(),
            _initial(),
            (),
            (("executor_object.exe01.obj01.grasp_contact_state", "UNKNOWN"),),
            (),
            (),
        )


@pytest.mark.parametrize(
    ("mode", "header"),
    (
        ("first_failure", "Verified operator-effect contradictions:"),
        ("verified_prefix", "Verified operator-effect prefix:"),
        ("release_support", "Verified operator-effect contradictions:"),
        ("post_release_support_only", "Verified operator-effect contradictions:"),
        ("compact_outcome", "Verified compact operator outcome:"),
        ("effect_ledger", "Verified operator-effect ledger:"),
    ),
)
def test_effect_alignment_modes_render_one_gt_backed_appendix(
    mode: str, header: str
) -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    after = _snapshot(
        red=(0.06, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        aperture=1.0,
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(scene, before, after, ())
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (),
        outcomes,
        ("sample_grasp_pose", "close_gripper", "get_object_pose", "open_gripper"),
        (
            'sample_grasp_pose("red cube")\n'
            "close_gripper()\n"
            'get_object_pose("green cube")\n'
            "goto_pose(place_pos, place_quat)\n"
            "open_gripper()",
        ),
        mode=mode,
    )

    assert aligned.status == "mismatch"
    assert aligned.mode == mode
    assert aligned.text.startswith(header)
    assert "CONTRADICTED:" in aligned.text
    assert "obj01" not in aligned.text
    assert "obj02" not in aligned.text
    assert not ideal_tk_v4._NUMERIC_TOKEN.search(aligned.text)
    if mode in {
        "release_support",
        "post_release_support_only",
        "compact_outcome",
        "effect_ledger",
    }:
        assert aligned.release_reference_id == "obj02"
        assert "SUPPORT_BY_PLANNED_REFERENCE_ESTABLISHED_AFTER_RELEASE" in (
            aligned.contradicted_effects
        )


def test_release_alignment_is_silent_when_unique_reference_support_holds() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.10),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
        aperture=0.45,
    )
    after = _snapshot(
        red=(0.12, 0.0, 0.06),
        green=(0.12, 0.0, 0.02),
        contact=True,
        table_contacts=frozenset({"obj02"}),
        aperture=1.0,
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(scene, before, after, ())
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (),
        outcomes,
        ("get_object_pose", "goto_pose", "open_gripper"),
        (
            'get_object_pose("green cube")\n'
            "goto_pose(place_pos, place_quat)\n"
            "open_gripper()",
        ),
        mode="post_release_support_only",
    )

    assert after.values["object_object.obj01.obj02.supported_by_reference"] == "TRUE"
    assert aligned.status == "silent"
    assert aligned.text == ""
    assert aligned.release_reference_id == "obj02"


def test_post_release_support_only_does_not_fall_back_to_grasp_failure() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    after = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        aperture=1.0,
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(scene, before, after, ())
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (),
        outcomes,
        ("close_gripper", "goto_pose", "open_gripper"),
        (
            'sample_grasp_pose("red cube")\n'
            "close_gripper()\n"
            "goto_pose(place_pos, place_quat)\n"
            "open_gripper()",
        ),
        mode="post_release_support_only",
    )

    assert "BILATERAL_GRASP_ESTABLISHED" in aligned.contradicted_effects
    assert aligned.release_reference_id is None
    assert aligned.status == "silent"
    assert aligned.text == ""


def test_effect_ledger_does_not_confirm_unevaluated_retention() -> None:
    scene = closed_gt.cube_scene_spec()
    before = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
    )
    after = _snapshot(
        red=(0.0, 0.0, 0.02),
        green=(0.12, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj01", "obj02"}),
        aperture=0.2,
    )
    outcomes = ideal_tk_v4._interval_grasp_outcomes(scene, before, after, ())
    aligned = ideal_tk_v4.align_public_operator_effects(
        scene,
        before,
        after,
        (),
        outcomes,
        ("close_gripper", "goto_pose"),
        (
            'sample_grasp_pose("red cube")\n'
            "close_gripper()\n"
            "goto_pose(lift_pos, lift_quat)",
        ),
        mode="effect_ledger",
    )

    assert "BILATERAL_GRASP_ESTABLISHED" in aligned.contradicted_effects
    assert "GRASP_RETAINED_DURING_POST_GRASP_MOTION" not in aligned.confirmed_effects
    assert "grasp of red cube was retained" not in aligned.text


def test_takeover_changes_only_one_native_report_carrier(tmp_path, monkeypatch) -> None:
    snapshots = iter((_initial(), _failed_after()))
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda env: next(snapshots))
    output_dir = str(tmp_path / "run")
    (tmp_path / "run").mkdir()
    env = object()
    report = "The scene did not visibly change."
    native = _native_prompt(report)

    ideal_tk_v4.initialize_trial_trace(
        env, output_dir=output_dir, trial=1, task="cube_stack"
    )
    corrected = ideal_tk_v4.run_takeover(
        env,
        output_dir=output_dir,
        trial=1,
        turn=1,
        task="cube_stack",
        native_report=report,
        native_decision_prompt=native,
        private_task_completed=False,
        effect_alignment_enabled=True,
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
            / ideal_tk_v4.TRACE_DIR
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
    assert record["non_report_prompt_roundtrip_equal"] is True
    assert record["v2_1_projection_equal"] is True
    assert record["canonical_a_star_prompt_unchanged"] is True
    assert record["effect_alignment_status"] == "silent"
    assert record["inline_api_provenance_only"] is True
    assert record["standalone_api_effect_section_visible"] is False
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
    ideal_tk_v4.initialize_trial_trace(
        object(), output_dir=output_dir, trial=2, task="cube_stack"
    )

    with pytest.raises(ValueError, match="expected one native report carrier, found 2"):
        ideal_tk_v4.run_takeover(
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
    ideal_tk_v4.initialize_trial_trace(
        object(), output_dir=output_dir, trial=3, task="cube_stack"
    )

    with pytest.raises(RuntimeError, match="benchmark-goal IFF audit failed"):
        ideal_tk_v4.run_takeover(
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
        ideal_tk_v4.initialize_trial_trace(
            object(), output_dir=str(tmp_path), trial=1, task="peg_insertion"
        )


def test_passive_capture_is_turn_synchronous_and_action_invisible(
    tmp_path, monkeypatch
) -> None:
    snapshots = iter((_initial(), _failed_after()))
    monkeypatch.setattr(closed_gt, "extract_cube_snapshot", lambda env: next(snapshots))
    output_dir = str(tmp_path / "run")
    (tmp_path / "run").mkdir()

    ideal_tk_v4.initialize_passive_trial_trace(
        object(), output_dir=output_dir, trial=4, task="cube_restack"
    )
    assert (
        ideal_tk_v4.capture_passive_transition(
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
            / ideal_tk_v4.PASSIVE_TRACE_DIR
            / "trial_04"
            / "turn_01"
            / "transition.json"
        ).read_text()
    )
    assert record["mode"] == "passive_action_invisible"
    assert len(record["public_transition"]) == 57
    assert record["student_prompt_changed"] is False
    assert record["student_action_request_changed"] is False
