from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from capx.verification import closed_gt


def _body(
    position: tuple[float, float, float],
    *,
    half_extents: tuple[float, float, float] | None,
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> closed_gt.BodyState:
    return closed_gt.BodyState(
        position=np.asarray(position, dtype=np.float64),
        orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        half_extents=(
            None if half_extents is None else np.asarray(half_extents, dtype=np.float64)
        ),
        linear_velocity=np.asarray(linear_velocity, dtype=np.float64),
        angular_velocity=np.zeros(3, dtype=np.float64),
    )


def _raw(
    *,
    red: tuple[float, float, float],
    green: tuple[float, float, float],
    contact: bool,
    table_contacts: frozenset[str],
    holding: frozenset[tuple[str, str]] = frozenset(),
    aperture: float = 1.0,
) -> closed_gt.RawSceneState:
    return closed_gt.RawSceneState(
        executors={"exe01": _body((0.0, 0.0, 0.20), half_extents=None)},
        objects={
            "obj01": _body(red, half_extents=(0.02, 0.02, 0.02)),
            "obj02": _body(green, half_extents=(0.02, 0.02, 0.02)),
        },
        executor_aperture={"exe01": aperture},
        holding=holding,
        executor_contacts=holding,
        object_contacts=(
            frozenset({frozenset(("obj01", "obj02"))})
            if contact
            else frozenset()
        ),
        table_contacts=table_contacts,
        table_height=0.0,
    )


def _stack_success_raw() -> closed_gt.RawSceneState:
    return _raw(
        red=(0.0, 0.0, 0.061),
        green=(0.0, 0.0, 0.020),
        contact=True,
        table_contacts=frozenset({"obj02"}),
    )


def _restack_initial_raw() -> closed_gt.RawSceneState:
    return _raw(
        red=(0.0, 0.0, 0.020),
        green=(0.0, 0.0, 0.061),
        contact=True,
        table_contacts=frozenset({"obj01"}),
    )


def test_cube_schema_is_closed_unique_and_separates_derived_facts() -> None:
    scene = closed_gt.cube_scene_spec()
    specs = closed_gt.predicate_specs(scene)

    assert len(specs) == 51
    assert len({spec.key for spec in specs}) == len(specs)
    assert all("current_phase" not in spec.key for spec in specs)
    assert all("next" not in spec.key for spec in specs)
    assert all(spec.values for spec in specs)
    assert all("UNKNOWN" not in spec.values for spec in specs)
    assert {
        spec.layer for spec in specs
    } == {"primitive", "derived"}
    assert next(
        spec for spec in specs if spec.key == "executor.exe01.payload"
    ).dependencies == (
        "executor_object.exe01.obj01.holding",
        "executor_object.exe01.obj02.holding",
    )
    assert all(spec.dependencies for spec in specs if spec.layer == "derived")
    assert all(not spec.dependencies for spec in specs if spec.layer == "primitive")
    key_index = {spec.key: index for index, spec in enumerate(specs)}
    assert all(
        key_index[dependency] < key_index[spec.key]
        for spec in specs
        for dependency in spec.dependencies
    )
    obj01_semantic = next(
        spec for spec in specs if spec.key == "object.obj01.semantic_state"
    )
    assert "SUPPORTED_BY_OBJ01" not in obj01_semantic.values
    assert "SUPPORTED_BY_OBJ02" in obj01_semantic.values
    stability = next(
        spec for spec in specs if spec.key == "object.obj01.support_stability"
    )
    assert "object_object.obj01.obj02.support_region" in stability.dependencies
    grasp_state = next(
        spec
        for spec in specs
        if spec.key == "executor_object.exe01.obj01.grasp_contact_state"
    )
    assert grasp_state.values == closed_gt.GRASP_CONTACT_STATE_VALUES
    assert "object.obj01.table_contact" in grasp_state.dependencies
    assert "object_object.obj01.obj02.supported_by_reference" in (
        grasp_state.dependencies
    )


def test_schema_expands_executor_object_relations_without_pair_ambiguity() -> None:
    scene = closed_gt.SceneSpec(
        executors=(
            closed_gt.EntitySpec("exe01", "left gripper", "executor"),
            closed_gt.EntitySpec("exe02", "right gripper", "executor"),
        ),
        objects=(
            closed_gt.EntitySpec("obj01", "first object", "object"),
            closed_gt.EntitySpec("obj02", "second object", "object"),
        ),
        table=closed_gt.EntitySpec("surface.table", "table", "surface"),
    )
    keys = [spec.key for spec in closed_gt.predicate_specs(scene)]

    assert "executor_object.exe01.obj02.relative_z" in keys
    assert "executor_object.exe02.obj01.holding" in keys
    assert keys.count("object_object.obj01.obj02.contact") == 1
    assert "object_object.obj02.obj01.contact" not in keys


def test_restack_topology_is_directional_and_unambiguous() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = closed_gt.compile_snapshot(scene, _restack_initial_raw())
    values = snapshot.values

    assert values["object_object.obj01.obj02.supported_by_reference"] == "FALSE"
    assert values["object_object.obj02.obj01.supported_by_reference"] == "TRUE"
    assert values["object_object.obj01.obj02.contact"] == "TRUE"
    assert values["object_object.obj02.obj01.support_region"] == "CENTER_REGION"
    assert values["object_object.obj02.obj01.center_in_reference_footprint"] == "INSIDE"
    assert values["object.obj01.table_contact"] == "TRUE"
    assert values["object.obj02.table_contact"] == "FALSE"
    assert values["object.obj01.top_access"] == "BLOCKED_FROM_ABOVE"
    assert values["object.obj01.load_bearing"] == "SUPPORTING_ONE_OBJECT"
    assert values["object.obj02.load_bearing"] == "NOT_SUPPORTING_AN_OBJECT"
    assert values["object.obj02.semantic_state"] == "SUPPORTED_BY_OBJ01"


def test_stack_goal_is_iff_over_declared_factual_atoms() -> None:
    scene = closed_gt.cube_scene_spec()
    success = closed_gt.compile_snapshot(scene, _stack_success_raw())
    initial = closed_gt.compile_snapshot(scene, _restack_initial_raw())

    assert closed_gt.goal_satisfied(
        success, closed_gt.cube_goal_definition("cube_stack")
    )
    assert closed_gt.goal_satisfied(
        success, closed_gt.cube_goal_definition("cube_restack")
    )
    assert not closed_gt.goal_satisfied(
        initial, closed_gt.cube_goal_definition("cube_stack")
    )
    audit = closed_gt.goal_iff_audit(
        success,
        closed_gt.cube_goal_definition("cube_stack"),
        private_task_completed=True,
    )
    assert audit["iff_holds"] is True
    assert audit["action_visible"] is False
    mismatch = closed_gt.goal_iff_audit(
        success,
        closed_gt.cube_goal_definition("cube_stack"),
        private_task_completed=False,
    )
    assert mismatch["iff_holds"] is False


def test_goal_iff_matches_benchmark_atoms_not_stricter_support_semantics() -> None:
    scene = closed_gt.cube_scene_spec()
    side_contact = closed_gt.compile_snapshot(
        scene,
        _raw(
            red=(0.041, 0.0, 0.061),
            green=(0.0, 0.0, 0.020),
            contact=True,
            table_contacts=frozenset({"obj02"}),
        ),
    )

    assert (
        side_contact.values["object_object.obj01.obj02.supported_by_reference"]
        == "FALSE"
    )
    assert closed_gt.goal_satisfied(
        side_contact, closed_gt.cube_goal_definition("cube_stack")
    )

    both_above_workspace_guard = closed_gt.compile_snapshot(
        scene,
        _raw(
            red=(0.0, 0.0, 0.102),
            green=(0.0, 0.0, 0.061),
            contact=True,
            table_contacts=frozenset(),
        ),
    )
    assert closed_gt.goal_satisfied(
        both_above_workspace_guard,
        closed_gt.cube_goal_definition("cube_stack"),
    )
    assert not closed_gt.goal_satisfied(
        both_above_workspace_guard,
        closed_gt.cube_goal_definition("cube_restack"),
    )


def test_goal_validator_rejects_duplicate_unknown_and_unsatisfiable_atoms() -> None:
    scene = closed_gt.cube_scene_spec()
    atom = closed_gt.GoalAtom("executor_object.exe01.obj01.holding", "FALSE")

    with pytest.raises(ValueError, match="duplicates a predicate"):
        closed_gt.validate_goal(scene, closed_gt.GoalDefinition("duplicate", (atom, atom)))
    with pytest.raises(ValueError, match="unknown predicate"):
        closed_gt.validate_goal(
            scene,
            closed_gt.GoalDefinition("unknown", (closed_gt.GoalAtom("missing", "TRUE"),)),
        )
    with pytest.raises(ValueError, match="requires a forbidden"):
        closed_gt.validate_goal(
            scene,
            closed_gt.GoalDefinition("impossible", (atom,), ((atom,),)),
        )


def test_holding_and_aperture_are_distinct_executor_facts() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = closed_gt.compile_snapshot(
        scene,
        _raw(
            red=(0.0, 0.0, 0.10),
            green=(0.10, 0.0, 0.02),
            contact=False,
            table_contacts=frozenset({"obj02"}),
            holding=frozenset({("exe01", "obj01")}),
            aperture=0.45,
        ),
    )

    assert snapshot.values["executor.exe01.aperture"] == "PARTIALLY_OPEN"
    assert snapshot.values["executor_object.exe01.obj01.holding"] == "TRUE"
    assert snapshot.values["executor.exe01.payload"] == "OBJ01"
    assert snapshot.values["executor_object.exe01.obj01.contact"] == "TRUE"
    assert (
        snapshot.values["executor_object.exe01.obj01.grasp_evidence"]
        == "LIFT_CONFIRMED"
    )
    assert snapshot.values["object.obj01.support_stability"] == "EXECUTOR_SUPPORT"
    assert (
        snapshot.values["executor_object.exe01.obj01.grasp_contact_state"]
        == "BILATERAL_GRASP_LIFT_CONFIRMED"
    )


@pytest.mark.parametrize(
    (
        "executor_contact",
        "holding",
        "red",
        "green",
        "object_contact",
        "table_contacts",
        "expected",
    ),
    (
        (
            False,
            False,
            (0.0, 0.0, 0.02),
            (0.12, 0.0, 0.02),
            False,
            frozenset({"obj01", "obj02"}),
            "NO_EXECUTOR_CONTACT",
        ),
        (
            True,
            False,
            (0.0, 0.0, 0.02),
            (0.12, 0.0, 0.02),
            False,
            frozenset({"obj01", "obj02"}),
            "CONTACT_WITHOUT_BILATERAL_GRASP",
        ),
        (
            True,
            True,
            (0.0, 0.0, 0.02),
            (0.12, 0.0, 0.02),
            False,
            frozenset({"obj01", "obj02"}),
            "BILATERAL_GRASP_TABLE_SUPPORTED",
        ),
        (
            True,
            True,
            (0.0, 0.0, 0.061),
            (0.0, 0.0, 0.02),
            True,
            frozenset({"obj02"}),
            "BILATERAL_GRASP_OBJECT_SUPPORTED",
        ),
        (
            True,
            True,
            (0.0, 0.0, 0.03),
            (0.12, 0.0, 0.02),
            False,
            frozenset({"obj02"}),
            "BILATERAL_GRASP_BELOW_LIFT_BAND",
        ),
        (
            True,
            True,
            (0.0, 0.0, 0.10),
            (0.12, 0.0, 0.02),
            False,
            frozenset({"obj02"}),
            "BILATERAL_GRASP_LIFT_CONFIRMED",
        ),
    ),
)
def test_grasp_contact_state_is_a_mutually_exclusive_partition(
    executor_contact: bool,
    holding: bool,
    red: tuple[float, float, float],
    green: tuple[float, float, float],
    object_contact: bool,
    table_contacts: frozenset[str],
    expected: str,
) -> None:
    scene = closed_gt.cube_scene_spec()
    raw = _raw(
        red=red,
        green=green,
        contact=object_contact,
        table_contacts=table_contacts,
        holding=(frozenset({("exe01", "obj01")}) if holding else frozenset()),
    )
    if executor_contact and not holding:
        raw = replace(raw, executor_contacts=frozenset({("exe01", "obj01")}))
    snapshot = closed_gt.compile_snapshot(scene, raw)

    assert (
        snapshot.values["executor_object.exe01.obj01.grasp_contact_state"]
        == expected
    )
    assert expected in closed_gt.GRASP_CONTACT_STATE_VALUES


def test_transition_covers_every_state_field_and_pose() -> None:
    scene = closed_gt.cube_scene_spec()
    before = closed_gt.compile_snapshot(scene, _restack_initial_raw())
    after = closed_gt.compile_snapshot(
        scene,
        _raw(
            red=(0.0, 0.0, 0.02),
            green=(0.12, 0.0, 0.02),
            contact=False,
            table_contacts=frozenset({"obj01", "obj02"}),
        ),
    )
    transition = closed_gt.compile_transition(scene, before, after)

    state_transition_keys = [
        key
        for key in transition.values
        if key.removeprefix("transition.") in before.values
    ]
    assert len(state_transition_keys) == len(before.values) == 51
    assert (
        transition.values[
            "transition.object_object.obj02.obj01.supported_by_reference"
        ]
        == "CHANGED"
    )
    assert transition.values["transition.object.obj02.pose"] == "MOVED"
    assert transition.values["transition.any_relevant_state"] == "CHANGED"
    assert len(transition.values) == 57
    assert dict(
        closed_gt.iter_public_transition_values(scene, before, after, transition)
    ) == transition.values
    canonical_transition = dict(
        closed_gt.iter_public_transition_values(
            scene, before, after, transition, canonical_only=True
        )
    )
    assert len(canonical_transition) == 37
    assert "transition.object.obj02.pose" in canonical_transition
    assert "transition.object.obj02.semantic_state" not in canonical_transition
    assert "transition.any_relevant_state" not in canonical_transition


def test_no_change_transition_is_exactly_unchanged() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = closed_gt.compile_snapshot(scene, _restack_initial_raw())
    transition = closed_gt.compile_transition(scene, snapshot, snapshot)

    assert transition.values["transition.any_relevant_state"] == "UNCHANGED"
    assert not any(value == "CHANGED" for value in transition.values.values())
    assert not any(value == "MOVED" for value in transition.values.values())


def test_validator_rejects_support_and_payload_conflicts() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = closed_gt.compile_snapshot(scene, _restack_initial_raw())

    support_values = dict(snapshot.values)
    support_values[
        "object_object.obj01.obj02.supported_by_reference"
    ] = "TRUE"
    with pytest.raises(ValueError, match="support each other"):
        closed_gt.validate_snapshot(
            scene,
            replace(snapshot, values=support_values),
        )

    payload_values = dict(snapshot.values)
    payload_values["executor.exe01.payload"] = "OBJ01"
    with pytest.raises(ValueError, match="payload"):
        closed_gt.validate_snapshot(
            scene,
            replace(snapshot, values=payload_values),
        )


def test_raw_and_snapshot_reject_holding_without_contact() -> None:
    scene = closed_gt.cube_scene_spec()
    raw = _raw(
        red=(0.0, 0.0, 0.10),
        green=(0.1, 0.0, 0.02),
        contact=False,
        table_contacts=frozenset({"obj02"}),
        holding=frozenset({("exe01", "obj01")}),
    )
    with pytest.raises(ValueError, match="holding requires"):
        closed_gt.compile_snapshot(scene, replace(raw, executor_contacts=frozenset()))

    snapshot = closed_gt.compile_snapshot(scene, raw)
    values = dict(snapshot.values)
    values["executor_object.exe01.obj01.contact"] = "FALSE"
    with pytest.raises(ValueError, match="holding conflicts"):
        closed_gt.validate_snapshot(scene, replace(snapshot, values=values))


def test_validator_rejects_false_derived_footprint_and_support() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = closed_gt.compile_snapshot(scene, _stack_success_raw())

    footprint = dict(snapshot.values)
    footprint[
        "object_object.obj01.obj02.center_in_reference_footprint"
    ] = "OUTSIDE"
    with pytest.raises(ValueError, match="footprint conflicts with support region"):
        closed_gt.validate_snapshot(scene, replace(snapshot, values=footprint))

    support = dict(snapshot.values)
    support["object_object.obj01.obj02.supported_by_reference"] = "FALSE"
    with pytest.raises(ValueError, match="support predicate conflicts"):
        closed_gt.validate_snapshot(scene, replace(snapshot, values=support))


def test_every_derived_field_is_recomputed_by_validation() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = closed_gt.compile_snapshot(scene, _stack_success_raw())

    for spec in closed_gt.predicate_specs(scene):
        if spec.layer != "derived":
            continue
        corrupted = dict(snapshot.values)
        corrupted[spec.key] = next(
            candidate for candidate in spec.values if candidate != corrupted[spec.key]
        )
        with pytest.raises(ValueError):
            closed_gt.validate_snapshot(scene, replace(snapshot, values=corrupted))


def test_validator_rejects_missing_extra_and_out_of_domain_values() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = closed_gt.compile_snapshot(scene, _restack_initial_raw())

    missing = dict(snapshot.values)
    missing.pop("executor.exe01.aperture")
    with pytest.raises(ValueError, match="coverage mismatch"):
        closed_gt.validate_snapshot(scene, replace(snapshot, values=missing))

    invalid = dict(snapshot.values)
    invalid["executor.exe01.aperture"] = "UNKNOWN"
    with pytest.raises(ValueError, match="outside closed domain"):
        closed_gt.validate_snapshot(scene, replace(snapshot, values=invalid))


def test_public_boundary_contains_only_predicate_enums() -> None:
    scene = closed_gt.cube_scene_spec()
    snapshot = closed_gt.compile_snapshot(scene, _restack_initial_raw())
    public = dict(closed_gt.iter_public_values(scene, snapshot))
    canonical = dict(
        closed_gt.iter_public_values(scene, snapshot, canonical_only=True)
    )

    primitive_keys = {
        spec.key for spec in closed_gt.predicate_specs(scene) if spec.layer == "primitive"
    }
    assert public == snapshot.values
    assert set(canonical) == primitive_keys
    assert len(canonical) == 34
    assert all(isinstance(key, str) and isinstance(value, str) for key, value in public.items())
    assert all("[" not in value and "]" not in value for value in public.values())
    assert set(snapshot.private_poses) == {"exe01", "obj01", "obj02"}


def test_transition_validator_recomputes_pose_and_holding_outcome() -> None:
    scene = closed_gt.cube_scene_spec()
    before = closed_gt.compile_snapshot(
        scene,
        _raw(
            red=(0.0, 0.0, 0.10),
            green=(0.1, 0.0, 0.02),
            contact=False,
            table_contacts=frozenset({"obj02"}),
            holding=frozenset({("exe01", "obj01")}),
        ),
    )
    after = closed_gt.compile_snapshot(scene, _stack_success_raw())
    transition = closed_gt.compile_transition(scene, before, after)

    bad_pose = dict(transition.values)
    bad_pose["transition.object.obj01.pose"] = "NO_MEANINGFUL_CHANGE"
    with pytest.raises(ValueError, match="pose transition conflicts"):
        closed_gt.validate_transition(
            scene, before, after, replace(transition, values=bad_pose)
        )

    bad_outcome = dict(transition.values)
    bad_outcome[
        "transition.executor_object.exe01.obj01.holding_outcome"
    ] = "MAINTAINED"
    with pytest.raises(ValueError, match="holding outcome conflicts"):
        closed_gt.validate_transition(
            scene, before, after, replace(transition, values=bad_outcome)
        )


def test_regen_semantic_coverage_is_explicit_about_exclusions() -> None:
    manifest = closed_gt.semantic_coverage_manifest(closed_gt.cube_scene_spec())

    assert manifest["covers_declared_regen_factual_semantics"] is True
    assert manifest["missing_regen_factual_families"] == {}
    assert "executor_aperture" in manifest["covered_physical_families"]
    assert "executor_object_contact" in manifest["covered_physical_families"]
    assert "executor_object_holding" in manifest["covered_physical_families"]
    assert "executor_object_relative_z" in manifest["covered_physical_families"]
    assert "object_object_support" in manifest["covered_physical_families"]
    assert "every_closed_state_predicate_change" in manifest[
        "covered_transition_families"
    ]
    assert "next_move" in manifest[
        "explicitly_excluded_non_factual_or_non_state_families"
    ]
    assert "camera_visibility_or_corruption" in manifest[
        "explicitly_excluded_non_factual_or_non_state_families"
    ]

    audit = closed_gt.schema_audit(closed_gt.cube_scene_spec())
    assert audit == {
        "schema_version": closed_gt.SCHEMA_VERSION,
            "state_field_count": 51,
        "canonical_public_basis_field_count": 34,
        "primitive_field_count": 34,
            "derived_field_count": 17,
        "all_keys_unique": True,
        "all_domains_finite_and_nonempty": True,
        "unknown_is_not_a_gt_value": True,
        "all_derived_fields_have_declared_dependencies": True,
        "all_primitive_fields_are_dependency_free": True,
        "dependency_graph_acyclic": True,
        "dependencies_precede_derived_fields": True,
        "covers_declared_regen_factual_semantics": True,
        "missing_regen_factual_families": {},
    }
