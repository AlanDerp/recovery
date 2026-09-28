"""Ideal V4: v2.1-compatible canonical state and ordered temporal delta."""

from __future__ import annotations

import ast
import copy
import functools
import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from capx.integrations.franka.observation_validity import (
    rigid_extent_is_consistent,
    rigid_extent_signature,
)
from capx.verification import closed_gt, closed_gt_takeover

SCHEMA_VERSION = "capx-ideal-tk-v4-observation-integrity"
TRACE_DIR = "ideal_tk_v4_traces"
PASSIVE_TRACE_DIR = "ideal_tk_v4_passive_traces"

_FORBIDDEN_PUBLIC_TERMS = (
    "reward",
    "task completed",
    "task success",
    "next move",
    "next action",
    "repair plan",
    "oracle",
    "move_to_pose",
    "execute_trajectory",
)
_NUMERIC_TOKEN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class RenderedTakeover:
    text: str
    current_fields: tuple[str, ...]
    changed_fields: tuple[str, ...]
    unchanged_fields: tuple[str, ...]
    pose_fields: tuple[str, ...]
    temporal_event_fields: tuple[str, ...]
    interval_grasp_outcomes: tuple[tuple[str, str], ...]
    interval_placement_outcomes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SemanticEventBatch:
    changes: tuple[tuple[str, str, str], ...]
    api_name: str | None = None
    api_call_index: int | None = None
    state_before: tuple[tuple[str, str], ...] = ()
    state_after: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ApiTransitionBatch:
    api_name: str
    api_call_index: int
    state_before: tuple[tuple[str, str], ...]
    state_after: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ApiTransitionLedgerEntry:
    api_name: str
    api_call_index: int
    endpoint_changes: tuple[tuple[str, str, str], ...]
    endpoint_unchanged: tuple[tuple[str, str], ...]
    transient_sequences: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class ExtentIntegrityAudit:
    api_call_index: int
    object_id: str
    status: str
    observed_signature_sha256: str | None
    expected_signature_sha256: str


API_TRANSITION_LEDGER_MODES = ("full", "phase_bound_sparse")


@dataclass(frozen=True)
class ReleaseBoundPlacementDelta:
    subject_id: str
    reference_id: str
    comparisons: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class ReleaseBoundPlacementOutcome:
    subject_id: str
    reference_id: str
    release_call_index: int
    relative_x: str
    relative_y: str
    relative_z: str
    support_region: str
    footprint: str
    reference_orientation: str
    reference_stability: str
    support_at_contact_loss: bool
    support_first_established: str
    support_at_endpoint: bool


@dataclass(frozen=True)
class ProjectionSemanticGain:
    status: str
    novel_fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class _BoundGraspEpisode:
    status: str
    start_call_index: int | None = None
    release_call_index: int | None = None
    acquisition_call_index: int | None = None
    lift_call_index: int | None = None
    loss_call_index: int | None = None
    acquisition_api: str | None = None
    lift_api: str | None = None
    loss_api: str | None = None
    recovered_loss_call_index: int | None = None
    recovered_loss_api: str | None = None
    executor_id: str | None = None


@dataclass(frozen=True)
class EffectAlignment:
    """Post-hoc comparison of public operator effects with observed GT."""

    status: str
    object_id: str | None
    expected_effects: tuple[str, ...]
    contradicted_effects: tuple[str, ...]
    text: str
    mode: str = "contradictions"
    confirmed_effects: tuple[str, ...] = ()
    release_reference_id: str | None = None
    grasp_episode_binding_status: str = "not_evaluated"
    grasp_episode_start_call_index: int | None = None
    grasp_episode_release_call_index: int | None = None
    grasp_episode_acquisition_call_index: int | None = None
    grasp_episode_lift_call_index: int | None = None
    grasp_episode_loss_call_index: int | None = None
    grasp_episode_acquisition_endpoint: tuple[tuple[str, str], ...] = ()
    grasp_episode_recovered_loss_call_index: int | None = None


EFFECT_ALIGNMENT_MODES = (
    "contradictions",
    "first_failure",
    "verified_prefix",
    "release_support",
    "grasp_loss_only",
    "instance_bound_grasp_loss_only",
    "instance_bound_grasp_outcome_only",
    "instance_bound_transport_failure_only",
    "instance_bound_off_table_lift_regression_only",
    "instance_bound_grasp_loss_plus_off_table_lift_regression",
    "post_release_support_only",
    "compact_outcome",
    "effect_ledger",
)


INTERVAL_GRASP_OUTCOME_VALUES = (
    "NO_EXECUTOR_CONTACT_OBSERVED",
    "CONTACT_WITHOUT_BILATERAL_GRASP_OBSERVED",
    "BILATERAL_GRASP_WITHOUT_LIFT_THEN_LOST",
    "BILATERAL_GRASP_WITHOUT_LIFT_RETAINED",
    "LIFT_CONFIRMED_THEN_LOST",
    "LIFT_CONFIRMED_AND_RETAINED",
)

_GRASP_OBSERVED_OUTCOMES = frozenset(
    {
        "BILATERAL_GRASP_WITHOUT_LIFT_THEN_LOST",
        "BILATERAL_GRASP_WITHOUT_LIFT_RETAINED",
        "LIFT_CONFIRMED_THEN_LOST",
        "LIFT_CONFIRMED_AND_RETAINED",
    }
)
_LIFT_OBSERVED_OUTCOMES = frozenset(
    {"LIFT_CONFIRMED_THEN_LOST", "LIFT_CONFIRMED_AND_RETAINED"}
)
_KNOWN_EFFECT_ALIGNMENT_OUTCOMES = frozenset(INTERVAL_GRASP_OUTCOME_VALUES)


_PUBLIC_API_ROLE = re.compile(
    r"^(?P<role>close_gripper|open_gripper|goto_pose)"
    r"(?:_arm(?P<arm_index>\d+)|_(?P<all_arms>both))?$"
)


def _public_api_role(name: str) -> str | None:
    """Map backend API spellings to task-independent manipulation roles."""
    match = _PUBLIC_API_ROLE.fullmatch(name)
    return match.group("role") if match is not None else None


def _public_api_executor_id(
    scene: closed_gt.SceneSpec, name: str
) -> str | None:
    """Return the uniquely addressed executor, or None for shared/ambiguous calls."""
    match = _PUBLIC_API_ROLE.fullmatch(name)
    if match is None or match.group("all_arms") is not None:
        return None
    arm_index = match.group("arm_index")
    if arm_index is None:
        return scene.executors[0].entity_id if len(scene.executors) == 1 else None
    index = int(arm_index)
    return scene.executors[index].entity_id if index < len(scene.executors) else None


def _api_addresses_executor(
    scene: closed_gt.SceneSpec, name: str, executor_id: str
) -> bool:
    addressed = _public_api_executor_id(scene, name)
    if addressed is not None:
        return addressed == executor_id
    match = _PUBLIC_API_ROLE.fullmatch(name)
    return match is not None and match.group("all_arms") is not None


@dataclass
class _SemanticEventRecorder:
    scene: closed_gt.SceneSpec
    latest: closed_gt.CompiledSnapshot
    pending: list[SemanticEventBatch]
    error: str | None = None
    active_api_calls: list[tuple[str, int]] = field(default_factory=list)
    pending_api_calls: list[str] = field(default_factory=list)
    pending_api_transitions: list[ApiTransitionBatch] = field(default_factory=list)
    pending_extent_integrity: list[ExtentIntegrityAudit] = field(default_factory=list)

    def observe(self, snapshot: closed_gt.CompiledSnapshot) -> None:
        closed_gt.validate_snapshot(self.scene, snapshot)
        changes = tuple(
            (spec.key, self.latest.values[spec.key], snapshot.values[spec.key])
            for spec in _temporal_specs(self.scene)
            if self.latest.values[spec.key] != snapshot.values[spec.key]
        )
        if changes:
            api_name, api_call_index = (
                self.active_api_calls[-1]
                if self.active_api_calls
                else (None, None)
            )
            self.pending.append(
                SemanticEventBatch(
                    changes,
                    api_name,
                    api_call_index,
                    tuple(sorted(self.latest.values.items())),
                    tuple(sorted(snapshot.values.items())),
                )
            )
        self.latest = snapshot


# Continuous poses remain process-private. Trial subprocess isolation makes this
# cache single-trial in production; the key prevents accidental test/run mixing.
_PRIVATE_SNAPSHOTS: dict[tuple[str, int], list[closed_gt.CompiledSnapshot]] = {}
_PRIVATE_EVENT_RECORDERS: dict[tuple[str, int], _SemanticEventRecorder] = {}

_EXTENT_INTEGRITY_STATUSES = frozenset(
    {"CONSISTENT", "INCONSISTENT", "UNAVAILABLE"}
)

_REDUNDANT_ACTION_VISIBLE_FAMILIES = frozenset(
    {
        "executor_object_holding",
        "executor_payload",
        "executor_object_grasp_evidence",
        "object_semantic_state",
    }
)
_TEMPORAL_FAMILIES = frozenset(
    {
        "executor_aperture",
        "executor_object_contact",
        "executor_object_grasp_contact_state",
        "object_object_contact",
        "object_table_contact",
        "object_object_support",
        "object_top_access",
        "object_load_bearing",
        "object_support_stability",
    }
)
_RECORDED_TEMPORAL_FAMILIES = _TEMPORAL_FAMILIES | frozenset(
    {"object_object_support_region"}
)


def _action_visible_specs(
    scene: closed_gt.SceneSpec,
) -> tuple[closed_gt.PredicateSpec, ...]:
    return tuple(
        spec
        for spec in closed_gt.predicate_specs(scene)
        if spec.family not in _REDUNDANT_ACTION_VISIBLE_FAMILIES
    )


def _temporal_specs(scene: closed_gt.SceneSpec) -> tuple[closed_gt.PredicateSpec, ...]:
    return tuple(
        spec
        for spec in _action_visible_specs(scene)
        if spec.family in _RECORDED_TEMPORAL_FAMILIES
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cache_key(output_dir: str, trial: int) -> tuple[str, int]:
    return str(Path(output_dir).resolve()), trial


def _entity_names(scene: closed_gt.SceneSpec) -> dict[str, str]:
    return {
        entity.entity_id: entity.public_name
        for entity in (*scene.executors, *scene.objects, scene.table)
    }


def _public_label(
    scene: closed_gt.SceneSpec,
    spec: closed_gt.PredicateSpec,
) -> str:
    names = _entity_names(scene)
    arguments = tuple(names[item] for item in spec.arguments)
    family = spec.family
    templates: dict[str, str] = {
        "executor_aperture": "{0} aperture",
        "executor_dynamic_state": "{0} motion state",
        "executor_orientation_to_table": "{0} orientation relative to {1}",
        "executor_object_holding": "{0} holding {1}",
        "executor_object_contact": "contact between {0} and {1}",
        "executor_object_relative_x": "{0} x position relative to {1}",
        "executor_object_relative_y": "{0} y position relative to {1}",
        "executor_object_relative_z": "{0} z position relative to {1}",
        "object_object_relative_x": "{0} x position relative to {1}",
        "object_object_relative_y": "{0} y position relative to {1}",
        "object_object_relative_z": "{0} z position relative to {1}",
        "object_object_support_region": "{0} center in the support region of {1}",
        "object_object_volume": "{0} center in the volume of {1}",
        "object_object_contact": "contact between {0} and {1}",
        "object_table_contact": "contact between {0} and {1}",
        "object_table_height": "{0} qualitative height above {1}",
        "object_workspace_height": "{0} qualitative workspace-frame height",
        "object_dynamic_state": "{0} motion state",
        "object_orientation_to_table": "{0} orientation relative to {1}",
        "object_object_footprint": "{0} center in the footprint of {1}",
        "object_object_support": "{0} supported by {1}",
        "executor_payload": "{0} payload",
        "executor_object_grasp_evidence": "grasp evidence for {0} and {1}",
        "executor_object_grasp_contact_state": (
            "mutually exclusive grasp-contact state for {0} and {1}"
        ),
        "object_semantic_state": "{0} support and holding state",
        "object_top_access": "top access to {0}",
        "object_load_bearing": "whether {0} supports other objects",
        "object_support_stability": "{0} support stability",
    }
    try:
        return templates[family].format(*arguments)
    except KeyError as exc:
        raise ValueError(f"No public label for closed-GT family: {family}") from exc


def _public_value(scene: closed_gt.SceneSpec, value: str) -> str:
    fixed = {
        "FALSE": "false",
        "TRUE": "true",
        "NEGATIVE_OF_REFERENCE_BAND": "on the negative side of the reference band",
        "WITHIN_REFERENCE_BAND": "within the reference band",
        "POSITIVE_OF_REFERENCE_BAND": "on the positive side of the reference band",
        "OUTSIDE": "outside",
        "INSIDE": "inside",
        "EDGE_REGION": "in the edge region",
        "CENTER_REGION": "in the center region",
        "AT_OR_BELOW_LIFT_BAND": "at or below the lift band",
        "ABOVE_LIFT_BAND": "above the lift band",
        "AT_OR_BELOW_WORKSPACE_GUARD_BAND": "at or below the workspace guard band",
        "ABOVE_WORKSPACE_GUARD_BAND": "above the workspace guard band",
        "CLOSED": "closed",
        "PARTIALLY_OPEN": "partially open",
        "OPEN": "open",
        "SETTLED": "settled",
        "MOVING": "moving",
        "LOCAL_Z_ALIGNED_WITH_TABLE_NORMAL": "local z aligned with the table normal",
        "LOCAL_Z_OBLIQUE_TO_TABLE_NORMAL": "local z oblique to the table normal",
        "LOCAL_Z_OPPOSED_TO_TABLE_NORMAL": "local z opposed to the table normal",
        "BLOCKED_FROM_ABOVE": "blocked from above",
        "CLEAR_FROM_ABOVE": "clear from above",
        "CENTERED_OBJECT_SUPPORT": "centered object support",
        "EDGE_OBJECT_SUPPORT": "edge object support",
        "TABLE_SUPPORT": "table support",
        "EXECUTOR_SUPPORT": "executor support",
        "MULTIPLE_SUPPORTS": "multiple supports",
        "NO_STABLE_SUPPORT": "no stable support",
        "NOT_HELD": "not held",
        "CONTACT_ONLY_NOT_LIFT_CONFIRMED": "contact only; lift not confirmed",
        "LIFT_CONFIRMED": "lift confirmed",
        "NO_EXECUTOR_CONTACT": "no executor contact",
        "CONTACT_WITHOUT_BILATERAL_GRASP": (
            "executor contact without a bilateral grasp contact"
        ),
        "BILATERAL_GRASP_TABLE_SUPPORTED": (
            "bilateral grasp contact, still supported by the table, transport not established"
        ),
        "BILATERAL_GRASP_OBJECT_SUPPORTED": (
            "bilateral grasp contact, still supported by another object, transport not established"
        ),
        "BILATERAL_GRASP_BELOW_LIFT_BAND": (
            "bilateral grasp contact without other support, but lift not confirmed"
        ),
        "BILATERAL_GRASP_LIFT_CONFIRMED": (
            "bilateral grasp contact with lift confirmed"
        ),
        "NO_EXECUTOR_CONTACT_OBSERVED": (
            "no executor contact was observed during the executed code"
        ),
        "CONTACT_WITHOUT_BILATERAL_GRASP_OBSERVED": (
            "executor contact occurred, but bilateral grasp contact was never established"
        ),
        "BILATERAL_GRASP_WITHOUT_LIFT_THEN_LOST": (
            "bilateral grasp contact occurred, lift was never confirmed, and contact "
            "was absent at the endpoint"
        ),
        "BILATERAL_GRASP_WITHOUT_LIFT_RETAINED": (
            "bilateral grasp contact occurred without confirmed lift and remained at "
            "the endpoint"
        ),
        "LIFT_CONFIRMED_THEN_LOST": (
            "bilateral grasp contact with lift was confirmed and was later absent at "
            "the endpoint"
        ),
        "LIFT_CONFIRMED_AND_RETAINED": (
            "bilateral grasp contact with lift was confirmed and remained at the endpoint"
        ),
        "CENTER_SUPPORT_AT_ENDPOINT": "center support was present at the endpoint",
        "EDGE_SUPPORT_AT_ENDPOINT": "edge support was present at the endpoint",
        "CENTER_SUPPORT_ESTABLISHED_THEN_LOST": (
            "center support was established and was later absent at the endpoint"
        ),
        "EDGE_SUPPORT_ESTABLISHED_THEN_LOST": (
            "edge support was established and was later absent at the endpoint"
        ),
        "CENTER_REGION_ENTERED_WITHOUT_SUPPORT": (
            "the center support region was entered, but support was never established"
        ),
        "EDGE_REGION_ENTERED_WITHOUT_SUPPORT": (
            "the edge support region was entered, but support was never established"
        ),
        "NEVER_ENTERED_SUPPORT_REGION": "the support region was never entered",
        "NOT_SUPPORTING_AN_OBJECT": "not supporting another object",
        "SUPPORTING_ONE_OBJECT": "supporting one other object",
        "SUPPORTING_MULTIPLE_OBJECTS": "supporting multiple other objects",
        "EMPTY": "empty",
        "MULTIPLE": "multiple objects",
        "HELD_NOT_LIFTED": "held but not lifted",
        "HELD_LIFTED": "held and lifted",
        "ON_TABLE": "on the table",
        "SUPPORTED_BY_MULTIPLE_OBJECTS": "supported by multiple objects",
        "AIRBORNE_UNHELD": "airborne and unheld",
        "UNSUPPORTED_OR_UNSETTLED": "unsupported or unsettled",
        "NO_MEANINGFUL_CHANGE": "no meaningful pose change",
        "MOVED": "moved",
        "NOT_HELD_UNCHANGED": "not held before or after",
        "ACQUIRED": "acquired",
        "MAINTAINED": "maintained",
        "DETACHED_TO_STABLE_SUPPORT": "detached onto stable support",
        "DETACHED_WITHOUT_STABLE_SUPPORT": "detached without stable support",
        "UNCHANGED": "unchanged",
        "CHANGED": "changed",
    }
    if value in fixed:
        return fixed[value]
    names = _entity_names(scene)
    for entity_id, public_name in names.items():
        token = entity_id.upper()
        if value == token:
            return public_name
        if value == f"SUPPORTED_BY_{token}":
            return f"supported by {public_name}"
    raise ValueError(f"No public rendering for closed-GT value: {value}")


def _interval_grasp_outcomes(
    scene: closed_gt.SceneSpec,
    before: closed_gt.CompiledSnapshot,
    after: closed_gt.CompiledSnapshot,
    event_batches: tuple[SemanticEventBatch, ...],
) -> tuple[tuple[str, str], ...]:
    """Collapse each executor-object trace into one exhaustive factual outcome."""
    outcomes: list[tuple[str, str]] = []
    for executor in scene.executors:
        for obj in scene.objects:
            key = (
                f"executor_object.{executor.entity_id}.{obj.entity_id}."
                "grasp_contact_state"
            )
            states = [before.values[key]]
            states.extend(
                new
                for batch in event_batches
                for changed_key, _old, new in batch.changes
                if changed_key == key
            )
            if states[-1] != after.values[key]:
                states.append(after.values[key])

            saw_contact = any(state != "NO_EXECUTOR_CONTACT" for state in states)
            saw_bilateral = any(state.startswith("BILATERAL_GRASP_") for state in states)
            saw_lift = "BILATERAL_GRASP_LIFT_CONFIRMED" in states
            retained = after.values[key].startswith("BILATERAL_GRASP_")
            if not saw_contact:
                outcome = "NO_EXECUTOR_CONTACT_OBSERVED"
            elif not saw_bilateral:
                outcome = "CONTACT_WITHOUT_BILATERAL_GRASP_OBSERVED"
            elif saw_lift and retained:
                outcome = "LIFT_CONFIRMED_AND_RETAINED"
            elif saw_lift:
                outcome = "LIFT_CONFIRMED_THEN_LOST"
            elif retained:
                outcome = "BILATERAL_GRASP_WITHOUT_LIFT_RETAINED"
            else:
                outcome = "BILATERAL_GRASP_WITHOUT_LIFT_THEN_LOST"
            if outcome not in INTERVAL_GRASP_OUTCOME_VALUES:
                raise RuntimeError(f"Invalid interval grasp outcome: {outcome}")
            outcomes.append((key, outcome))
    return tuple(outcomes)


def _interval_placement_outcomes(
    scene: closed_gt.SceneSpec,
    before: closed_gt.CompiledSnapshot,
    after: closed_gt.CompiledSnapshot,
    event_batches: tuple[SemanticEventBatch, ...],
    interval_grasp_outcomes: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    return closed_gt_takeover._interval_placement_outcomes(
        scene,
        before,
        after,
        event_batches,
        interval_grasp_outcomes,
    )


_IMMEDIATE_PLACEMENT_EVIDENCE = frozenset(
    {
        "CENTER_SUPPORT_ESTABLISHED_THEN_LOST",
        "EDGE_SUPPORT_ESTABLISHED_THEN_LOST",
        "CENTER_REGION_ENTERED_WITHOUT_SUPPORT",
        "EDGE_REGION_ENTERED_WITHOUT_SUPPORT",
    }
)


def _goal_ordered_object_pairs(
    scene: closed_gt.SceneSpec,
    goal: closed_gt.GoalDefinition,
) -> frozenset[tuple[str, str]]:
    """Derive ordered object pairs from relational goal atoms."""
    object_ids = {obj.entity_id for obj in scene.objects}
    specs = {spec.key: spec for spec in closed_gt.predicate_specs(scene)}
    pairs: set[tuple[str, str]] = set()
    for atom in goal.atoms:
        spec = specs[atom.predicate]
        if len(spec.arguments) != 2:
            continue
        subject_id, reference_id = spec.arguments
        if subject_id in object_ids and reference_id in object_ids:
            pairs.add((subject_id, reference_id))
    return frozenset(pairs)


def _select_salient_placement_outcomes(
    scene: closed_gt.SceneSpec,
    goal: closed_gt.GoalDefinition,
    outcomes: tuple[tuple[str, str], ...],
    *,
    turn: int,
) -> tuple[tuple[str, str], ...]:
    """Select novel goal-bound summaries without removing atomic GT."""
    if turn < 1:
        raise ValueError("Recovery turn must be positive")
    goal_pairs = _goal_ordered_object_pairs(scene, goal)
    selected: list[tuple[str, str]] = []
    for key, outcome in outcomes:
        family, subject_id, reference_id = key.split(".")
        if family != "placement_outcome":
            raise ValueError(f"Invalid placement outcome key: {key}")
        if (subject_id, reference_id) not in goal_pairs:
            continue
        if outcome in _IMMEDIATE_PLACEMENT_EVIDENCE:
            selected.append((key, outcome))
        elif outcome == "NEVER_ENTERED_SUPPORT_REGION" and turn > 1:
            selected.append((key, outcome))
    return tuple(selected)


def _release_bound_placement_outcome(
    scene: closed_gt.SceneSpec,
    before: closed_gt.CompiledSnapshot,
    after: closed_gt.CompiledSnapshot,
    event_batches: tuple[SemanticEventBatch, ...],
    interval_grasp_outcomes: tuple[tuple[str, str], ...],
    placement_outcomes: tuple[tuple[str, str], ...],
    executed_api_calls: tuple[str, ...],
    latest_code: str,
) -> ReleaseBoundPlacementOutcome | None:
    """Return a unique, action-bound qualitative snapshot at release."""
    if len(placement_outcomes) != 1:
        return None
    placement_key, _coarse_outcome = placement_outcomes[0]
    family, subject_id, reference_id = placement_key.split(".")
    if family != "placement_outcome":
        raise ValueError(f"Invalid placement outcome key: {placement_key}")

    effect_object = _bind_effect_object(
        scene, before, interval_grasp_outcomes, latest_code
    )
    if effect_object != subject_id:
        return None
    episode = _bind_grasp_episode(
        scene,
        before,
        subject_id,
        event_batches,
        executed_api_calls,
    )
    if (
        episode.status != "bound"
        or episode.executor_id is None
        or episode.release_call_index is None
    ):
        return None

    grasp_key = (
        f"executor_object.{episode.executor_id}.{subject_id}.grasp_contact_state"
    )
    release_loss_batch_index = None
    for batch_index, batch in enumerate(event_batches):
        if _event_call_index(batch, executed_api_calls) != episode.release_call_index:
            continue
        if any(
            key == grasp_key
            and old.startswith("BILATERAL_GRASP_")
            and not new.startswith("BILATERAL_GRASP_")
            for key, old, new in batch.changes
        ):
            release_loss_batch_index = batch_index
            break
    if release_loss_batch_index is None:
        return None

    release_batch = event_batches[release_loss_batch_index]
    if not release_batch.state_after:
        return None
    state_at_loss = dict(release_batch.state_after)
    prefix = f"object_object.{subject_id}.{reference_id}"
    relation_keys = {
        "relative_x": f"{prefix}.relative_x",
        "relative_y": f"{prefix}.relative_y",
        "relative_z": f"{prefix}.relative_z",
        "support_region": f"{prefix}.support_region",
        "footprint": f"{prefix}.center_in_reference_footprint",
        "support": f"{prefix}.supported_by_reference",
        "reference_orientation": f"object.{reference_id}.orientation_to_table",
        "reference_stability": f"object.{reference_id}.support_stability",
    }
    if not all(key in state_at_loss for key in relation_keys.values()):
        raise ValueError("Release-bound snapshot is missing a closed-GT field")

    support_key = relation_keys["support"]
    support_at_loss = state_at_loss[support_key] == "TRUE"
    first_established = "AT_CONTACT_LOSS" if support_at_loss else "NEVER"
    if not support_at_loss:
        for batch in event_batches[release_loss_batch_index + 1 :]:
            if not batch.state_after:
                continue
            state = dict(batch.state_after)
            if state[support_key] != "TRUE":
                continue
            call_index = _event_call_index(batch, executed_api_calls)
            first_established = (
                "DURING_RELEASE"
                if call_index == episode.release_call_index
                else "AFTER_RELEASE"
            )
            break

    return ReleaseBoundPlacementOutcome(
        subject_id=subject_id,
        reference_id=reference_id,
        release_call_index=episode.release_call_index,
        relative_x=state_at_loss[relation_keys["relative_x"]],
        relative_y=state_at_loss[relation_keys["relative_y"]],
        relative_z=state_at_loss[relation_keys["relative_z"]],
        support_region=state_at_loss[relation_keys["support_region"]],
        footprint=state_at_loss[relation_keys["footprint"]],
        reference_orientation=state_at_loss[
            relation_keys["reference_orientation"]
        ],
        reference_stability=state_at_loss[relation_keys["reference_stability"]],
        support_at_contact_loss=support_at_loss,
        support_first_established=first_established,
        support_at_endpoint=after.values[support_key] == "TRUE",
    )


def _render_release_bound_placement(
    scene: closed_gt.SceneSpec,
    outcome: ReleaseBoundPlacementOutcome,
) -> str:
    names = _entity_names(scene)
    subject = names[outcome.subject_id]
    reference = names[outcome.reference_id]
    if outcome.support_first_established == "AT_CONTACT_LOSS":
        support = "present when grasp contact ended"
    elif outcome.support_first_established == "DURING_RELEASE":
        support = "absent when grasp contact ended and established later during release"
    elif outcome.support_first_established == "AFTER_RELEASE":
        support = "absent during release and established during a later public API"
    elif outcome.support_first_established == "NEVER":
        support = "never established"
    else:
        raise ValueError("Unknown release-bound support phase")
    endpoint = "present" if outcome.support_at_endpoint else "absent"
    return (
        f"- {subject} relative to {reference} when bilateral grasp contact ended "
        "during the executed release: x, y, and z were respectively "
        f"{_public_value(scene, outcome.relative_x)}, "
        f"{_public_value(scene, outcome.relative_y)}, and "
        f"{_public_value(scene, outcome.relative_z)}; the support region was "
        f"{_public_value(scene, outcome.support_region)} and the footprint was "
        f"{_public_value(scene, outcome.footprint)}; {reference} had "
        f"{_public_value(scene, outcome.reference_orientation)} and "
        f"{_public_value(scene, outcome.reference_stability)}; support was "
        f"{support} and was {endpoint} at the endpoint."
    )


def _release_bound_semantic_gain(
    outcome: ReleaseBoundPlacementOutcome | None,
    event_batches: tuple[SemanticEventBatch, ...],
    executed_api_calls: tuple[str, ...],
) -> ProjectionSemanticGain:
    """Certify event-time facts absent from the parent ordered-delta report."""
    if outcome is None:
        return ProjectionSemanticGain("no_bound_release", ())

    release_batch = next(
        (
            batch
            for batch in event_batches
            if _event_call_index(batch, executed_api_calls)
            == outcome.release_call_index
            and any(
                old.startswith("BILATERAL_GRASP_")
                and not new.startswith("BILATERAL_GRASP_")
                for _key, old, new in batch.changes
            )
        ),
        None,
    )
    if release_batch is None or not release_batch.state_after:
        raise RuntimeError("Bound release lacks its recorded event-time state")

    prefix = f"object_object.{outcome.subject_id}.{outcome.reference_id}"
    candidate_facts = {
        f"{prefix}.relative_x": outcome.relative_x,
        f"{prefix}.relative_y": outcome.relative_y,
        f"{prefix}.relative_z": outcome.relative_z,
        f"{prefix}.support_region": outcome.support_region,
        f"{prefix}.center_in_reference_footprint": outcome.footprint,
        f"{prefix}.supported_by_reference": (
            "TRUE" if outcome.support_at_contact_loss else "FALSE"
        ),
        f"object.{outcome.reference_id}.orientation_to_table": (
            outcome.reference_orientation
        ),
        f"object.{outcome.reference_id}.support_stability": (
            outcome.reference_stability
        ),
    }
    recorded_state = dict(release_batch.state_after)
    if any(
        recorded_state.get(key) != value
        for key, value in candidate_facts.items()
    ):
        raise RuntimeError("Release-bound fact does not match recorded GT state")

    # The parent report exposes changed values at each ordered event, but not
    # predicates that remained unchanged at that event. Those time-qualified
    # values are the extension's actual information gain.
    explicitly_visible = {(key, new) for key, _old, new in release_batch.changes}
    novel = tuple(
        sorted(
            f"release_endpoint:{key}"
            for key, value in candidate_facts.items()
            if (key, value) not in explicitly_visible
        )
    )
    return ProjectionSemanticGain(
        "novel_release_facts" if novel else "parent_complete",
        novel,
    )


def _release_bound_placement_delta(
    previous: dict[str, Any] | None,
    current: ReleaseBoundPlacementOutcome | None,
) -> ReleaseBoundPlacementDelta | None:
    if previous is None or current is None:
        return None
    try:
        prior = ReleaseBoundPlacementOutcome(**previous)
    except (TypeError, ValueError):
        return None
    if (
        prior.subject_id != current.subject_id
        or prior.reference_id != current.reference_id
    ):
        return None
    fields = (
        "relative_x",
        "relative_y",
        "relative_z",
        "support_region",
        "footprint",
        "reference_orientation",
        "reference_stability",
        "support_at_contact_loss",
        "support_first_established",
        "support_at_endpoint",
    )
    return ReleaseBoundPlacementDelta(
        current.subject_id,
        current.reference_id,
        tuple((field, str(getattr(prior, field)), str(getattr(current, field))) for field in fields),
    )


def _release_delta_value(scene: closed_gt.SceneSpec, value: str) -> str:
    fixed = {
        "True": "present",
        "False": "absent",
        "AT_CONTACT_LOSS": "present when grasp contact ended",
        "DURING_RELEASE": "first established later during release",
        "AFTER_RELEASE": "first established during a later public API",
        "NEVER": "never established",
    }
    return fixed[value] if value in fixed else _public_value(scene, value)


def _render_release_bound_delta(
    scene: closed_gt.SceneSpec,
    delta: ReleaseBoundPlacementDelta,
) -> tuple[str, ...]:
    names = _entity_names(scene)
    subject = names[delta.subject_id]
    reference = names[delta.reference_id]
    labels = {
        "relative_x": "release-time relative x",
        "relative_y": "release-time relative y",
        "relative_z": "release-time relative z",
        "support_region": "release-time support region",
        "footprint": "release-time footprint",
        "reference_orientation": f"{reference} orientation",
        "reference_stability": f"{reference} support stability",
        "support_at_contact_loss": "support when grasp contact ended",
        "support_first_established": "support establishment phase",
        "support_at_endpoint": "endpoint support",
    }
    lines = [
        f"- Comparable bound attempts: {subject} was released relative to {reference}."
    ]
    for field, old, new in delta.comparisons:
        old_text = _release_delta_value(scene, old)
        new_text = _release_delta_value(scene, new)
        if old == new:
            lines.append(f"- {labels[field]} remained {new_text}.")
        else:
            lines.append(
                f"- {labels[field]} changed from {old_text} to {new_text}."
            )
    return tuple(lines)


def _api_transition_ledger(
    scene: closed_gt.SceneSpec,
    goal: closed_gt.GoalDefinition,
    interval_grasp_outcomes: tuple[tuple[str, str], ...],
    latest_code: str,
    api_transitions: tuple[ApiTransitionBatch, ...],
    event_batches: tuple[SemanticEventBatch, ...],
    executed_api_calls: tuple[str, ...],
    *,
    mode: str = "full",
) -> tuple[ApiTransitionLedgerEntry, ...]:
    """Project each physical API call into a bound closed-state transition."""
    if mode not in API_TRANSITION_LEDGER_MODES:
        raise ValueError(f"Unknown API transition ledger mode: {mode!r}")
    goal_pairs = _goal_ordered_object_pairs(scene, goal)
    if len(goal_pairs) != 1:
        return ()
    subject_id, reference_id = next(iter(goal_pairs))
    # Binding from code is sufficient when exactly one object is sampled. If code
    # samples more than one object, observed grasp contact below supplies binding.
    sampled_names = {
        literal.casefold()
        for _line, name, literal in _called_public_names(latest_code)
        if name == "sample_grasp_pose" and literal is not None
    }
    by_name = {obj.public_name.casefold(): obj.entity_id for obj in scene.objects}
    sampled = {by_name[name] for name in sampled_names if name in by_name}
    observed = {
        key.split(".")[2]
        for key, outcome in interval_grasp_outcomes
        if outcome != "NO_EXECUTOR_CONTACT_OBSERVED"
    }
    candidates = sampled if len(sampled) == 1 else observed
    if len(candidates) != 1:
        return ()
    object_id = next(iter(candidates))

    goal_keys = [
        f"object_object.{subject_id}.{reference_id}.relative_x",
        f"object_object.{subject_id}.{reference_id}.relative_y",
        f"object_object.{subject_id}.{reference_id}.relative_z",
        f"object_object.{subject_id}.{reference_id}.support_region",
        f"object_object.{subject_id}.{reference_id}.center_in_reference_footprint",
        f"object_object.{subject_id}.{reference_id}.supported_by_reference",
        f"object.{object_id}.support_stability",
        f"object.{object_id}.top_access",
        f"object.{reference_id}.orientation_to_table",
        f"object.{reference_id}.support_stability",
    ]
    executor_keys: list[str] = []
    for executor in scene.executors:
        prefix = f"executor_object.{executor.entity_id}.{object_id}"
        executor_keys.extend(
            (
                f"{prefix}.contact",
                f"{prefix}.relative_x",
                f"{prefix}.relative_y",
                f"{prefix}.relative_z",
                f"{prefix}.grasp_contact_state",
            )
        )
    keys = goal_keys + executor_keys
    keys = list(dict.fromkeys(keys))
    spec_by_key = {spec.key: spec for spec in closed_gt.predicate_specs(scene)}
    relevant = tuple(key for key in keys if key in spec_by_key)
    entries: list[ApiTransitionLedgerEntry] = []
    for transition in api_transitions:
        role = _public_api_role(transition.api_name)
        if role is None:
            continue
        before = dict(transition.state_before)
        after = dict(transition.state_after)
        selected = relevant
        if mode == "phase_bound_sparse":
            addressed = _public_api_executor_id(scene, transition.api_name)
            if addressed is None and len(scene.executors) == 1:
                addressed = scene.executors[0].entity_id
            grasp_key = (
                f"executor_object.{addressed}.{object_id}.grasp_contact_state"
                if addressed is not None
                else None
            )
            held_before = bool(
                grasp_key is not None
                and before[grasp_key].startswith("BILATERAL_GRASP_")
            )
            held_after = bool(
                grasp_key is not None
                and after[grasp_key].startswith("BILATERAL_GRASP_")
            )
            if role == "close_gripper" or (
                role == "goto_pose" and not (held_before or held_after)
            ):
                selected = tuple(key for key in executor_keys if key in spec_by_key)
                top_access = f"object.{object_id}.top_access"
                if top_access in spec_by_key:
                    selected += (top_access,)
            else:
                # Once transport has begun, only payload retention and the
                # payload/reference placement relation are operation-relevant.
                selected = tuple(
                    key
                    for key in (
                        *(key for key in executor_keys if key.endswith(".grasp_contact_state")),
                        f"object_object.{subject_id}.{reference_id}.relative_x",
                        f"object_object.{subject_id}.{reference_id}.relative_y",
                        f"object_object.{subject_id}.{reference_id}.relative_z",
                        f"object_object.{subject_id}.{reference_id}.support_region",
                        f"object_object.{subject_id}.{reference_id}.center_in_reference_footprint",
                        f"object_object.{subject_id}.{reference_id}.supported_by_reference",
                        f"object.{object_id}.support_stability",
                    )
                    if key in spec_by_key
                )
        changes = tuple(
            (key, before[key], after[key])
            for key in selected
            if before[key] != after[key]
        )
        unchanged = tuple(
            (key, after[key]) for key in selected if before[key] == after[key]
        )
        sequences: list[tuple[str, tuple[str, ...]]] = []
        transient_keys = selected
        if mode == "phase_bound_sparse":
            transient_keys = tuple(
                key
                for key in selected
                if spec_by_key[key].family
                in {
                    "executor_object_grasp_contact_state",
                    "object_object_support_region",
                    "object_object_support",
                }
            )
        for key in transient_keys:
            values = [before[key]]
            for batch in event_batches:
                if _event_call_index(batch, executed_api_calls) != transition.api_call_index:
                    continue
                values.extend(
                    new for changed_key, _old, new in batch.changes if changed_key == key
                )
            if values[-1] != after[key]:
                values.append(after[key])
            compact = tuple(
                value
                for index, value in enumerate(values)
                if index == 0 or value != values[index - 1]
            )
            if len(compact) > 2:
                sequences.append((key, compact))
        entries.append(
            ApiTransitionLedgerEntry(
                transition.api_name,
                transition.api_call_index,
                changes,
                unchanged,
                tuple(sequences),
            )
        )
    return tuple(entries)


def _render_api_transition_ledger(
    scene: closed_gt.SceneSpec,
    entries: tuple[ApiTransitionLedgerEntry, ...],
) -> tuple[str, ...]:
    specs = {spec.key: spec for spec in closed_gt.predicate_specs(scene)}
    lines: list[str] = []
    for index, entry in enumerate(entries):
        prefix = "First" if index == 0 else "Then"
        clauses = [
            f"{_public_label(scene, specs[key])} changed from "
            f"{_public_value(scene, old)} to {_public_value(scene, new)}"
            for key, old, new in entry.endpoint_changes
        ]
        if clauses:
            endpoint = "; ".join(clauses)
        else:
            stable_keys = tuple(
                item
                for item in entry.endpoint_unchanged
                if specs[item[0]].family
                in {
                    "executor_object_contact",
                    "executor_object_relative_x",
                    "executor_object_relative_y",
                    "executor_object_relative_z",
                    "object_object_relative_x",
                    "object_object_relative_y",
                    "object_object_relative_z",
                    "object_object_support_region",
                    "object_object_support",
                    "executor_object_grasp_contact_state",
                    "object_object_footprint",
                    "object_support_stability",
                    "object_top_access",
                }
            )
            endpoint = "no bound predicate changed"
            if stable_keys:
                endpoint += "; " + "; ".join(
                    f"{_public_label(scene, specs[key])} remained "
                    f"{_public_value(scene, value)}"
                    for key, value in stable_keys
                )
        transient = ""
        if entry.transient_sequences:
            transient = "; transient sequence: " + "; ".join(
                f"{_public_label(scene, specs[key])}: "
                + " then ".join(_public_value(scene, value) for value in values)
                for key, values in entry.transient_sequences
            )
        lines.append(
            f"- {prefix}, at the endpoint of executed public API "
            f"`{entry.api_name}`: {endpoint}{transient}."
        )
    return tuple(lines)


def _public_event_batches(
    scene: closed_gt.SceneSpec,
    event_batches: tuple[SemanticEventBatch, ...],
) -> tuple[SemanticEventBatch, ...]:
    spec_by_key = {spec.key: spec for spec in closed_gt.predicate_specs(scene)}
    public: list[SemanticEventBatch] = []
    for batch in event_batches:
        for key, _old, _new in batch.changes:
            if (
                key not in spec_by_key
                or spec_by_key[key].family not in _RECORDED_TEMPORAL_FAMILIES
            ):
                raise ValueError(f"Takeover event contains an untracked field: {key}")
        changes = tuple(
            change
            for change in batch.changes
            if spec_by_key[change[0]].family in _TEMPORAL_FAMILIES
        )
        if changes:
            public.append(
                SemanticEventBatch(changes, batch.api_name, batch.api_call_index)
            )
    return tuple(public)


def _called_public_names(code: str) -> tuple[tuple[int, str, str | None], ...]:
    """Return source-ordered public calls and literal object arguments."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()
    calls: list[tuple[int, str, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        literal = None
        if node.args and isinstance(node.args[0], ast.Constant):
            value = node.args[0].value
            if isinstance(value, str):
                literal = value
        calls.append((node.lineno, node.func.id, literal))
    return tuple(sorted(calls, key=lambda item: item[0]))


def _bind_effect_object(
    scene: closed_gt.SceneSpec,
    before: closed_gt.CompiledSnapshot,
    interval_outcomes: tuple[tuple[str, str], ...],
    latest_code: str,
) -> str | None:
    held = [
        obj.entity_id
        for executor in scene.executors
        for obj in scene.objects
        if before.values[
            f"executor_object.{executor.entity_id}.{obj.entity_id}.grasp_contact_state"
        ].startswith("BILATERAL_GRASP_")
    ]
    if len(held) == 1:
        return held[0]

    by_public_name = {obj.public_name.casefold(): obj.entity_id for obj in scene.objects}
    sampled = {
        by_public_name[literal.casefold()]
        for _line, name, literal in _called_public_names(latest_code)
        if name == "sample_grasp_pose"
        and literal is not None
        and literal.casefold() in by_public_name
    }
    if len(sampled) == 1:
        return next(iter(sampled))

    observed = {
        key.split(".")[2]
        for key, outcome in interval_outcomes
        if outcome != "NO_EXECUTOR_CONTACT_OBSERVED"
    }
    return next(iter(observed)) if len(observed) == 1 else None


def _bind_release_reference(
    scene: closed_gt.SceneSpec,
    object_id: str,
    latest_code: str,
    planned_window_start: int,
    planned_release_index: int,
) -> str | None:
    """Bind a unique non-payload object referenced before planned release."""
    calls = _called_public_names(latest_code)
    by_public_name = {obj.public_name.casefold(): obj.entity_id for obj in scene.objects}
    candidates = {
        by_public_name[literal.casefold()]
        for index, (_line, name, literal) in enumerate(calls)
        if planned_window_start < index < planned_release_index
        and name == "get_object_pose"
        and literal is not None
        and literal.casefold() in by_public_name
        and by_public_name[literal.casefold()] != object_id
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _loss_api_for_object(
    scene: closed_gt.SceneSpec,
    object_id: str,
    event_batches: tuple[SemanticEventBatch, ...],
) -> str | None:
    for executor in scene.executors:
        key = (
            f"executor_object.{executor.entity_id}.{object_id}.grasp_contact_state"
        )
        for batch in event_batches:
            for changed_key, old, new in batch.changes:
                if (
                    changed_key == key
                    and old.startswith("BILATERAL_GRASP_")
                    and not new.startswith("BILATERAL_GRASP_")
                ):
                    return batch.api_name
    return None


def _acquisition_api_for_object(
    scene: closed_gt.SceneSpec,
    object_id: str,
    event_batches: tuple[SemanticEventBatch, ...],
) -> str | None:
    for executor in scene.executors:
        key = (
            f"executor_object.{executor.entity_id}.{object_id}.grasp_contact_state"
        )
        for batch in event_batches:
            for changed_key, old, new in batch.changes:
                if (
                    changed_key == key
                    and not old.startswith("BILATERAL_GRASP_")
                    and new.startswith("BILATERAL_GRASP_")
                ):
                    return batch.api_name
    return None


def _event_call_index(
    batch: SemanticEventBatch,
    executed_api_calls: tuple[str, ...],
) -> int | None:
    if batch.api_call_index is not None:
        index = batch.api_call_index
        if (
            index < 0
            or index >= len(executed_api_calls)
            or executed_api_calls[index] != batch.api_name
        ):
            raise ValueError("Event API call-index provenance is inconsistent")
        return index
    if batch.api_name is None:
        return None
    candidates = [
        index
        for index, name in enumerate(executed_api_calls)
        if name == batch.api_name
    ]
    return candidates[0] if len(candidates) == 1 else None


def _bind_grasp_episode(
    scene: closed_gt.SceneSpec,
    before: closed_gt.CompiledSnapshot,
    object_id: str,
    event_batches: tuple[SemanticEventBatch, ...],
    executed_api_calls: tuple[str, ...],
) -> _BoundGraspEpisode:
    """Bind grasp effects to the last executed close-to-release call window."""
    held_at_turn_start = [
        executor.entity_id
        for executor in scene.executors
        if before.values[
            f"executor_object.{executor.entity_id}.{object_id}.grasp_contact_state"
        ].startswith("BILATERAL_GRASP_")
    ]
    candidates: list[_BoundGraspEpisode] = []
    any_unresolved = False
    for executor in scene.executors:
        close_indices = [
            index
            for index, name in enumerate(executed_api_calls)
            if _public_api_role(name) == "close_gripper"
            and _api_addresses_executor(scene, name, executor.entity_id)
        ]
        if close_indices:
            start = close_indices[-1]
        elif executor.entity_id in held_at_turn_start:
            start = -1
        else:
            continue
        release = next(
            (
                index
                for index in range(start + 1, len(executed_api_calls))
                if _public_api_role(executed_api_calls[index]) == "open_gripper"
                and _api_addresses_executor(
                    scene, executed_api_calls[index], executor.entity_id
                )
            ),
            len(executed_api_calls),
        )
        key = (
            f"executor_object.{executor.entity_id}.{object_id}.grasp_contact_state"
        )
        state = before.values[key]
        indexed_changes: list[tuple[int, str, str, str | None]] = []
        unresolved = False
        for batch in event_batches:
            for changed_key, old, new in batch.changes:
                if changed_key != key:
                    continue
                index = _event_call_index(batch, executed_api_calls)
                if index is None:
                    unresolved = True
                    any_unresolved = True
                    continue
                indexed_changes.append((index, old, new, batch.api_name))
        if unresolved:
            continue
        for index, _old, new, _api_name in indexed_changes:
            if index < start:
                state = new
        held_before_window = state.startswith("BILATERAL_GRASP_")
        active = state.startswith("BILATERAL_GRASP_")
        acquisition_index = None
        acquisition_api = None
        lift_index = None
        lift_api = None
        loss_index = None
        loss_api = None
        recovered_loss_index = None
        recovered_loss_api = None
        changes_by_call: dict[int, list[str]] = {}
        for index, _old, new, _api_name in indexed_changes:
            if start <= index < release:
                changes_by_call.setdefault(index, []).append(new)
        # Endpoint recovery suppresses contact jitter. A within-call episode that
        # reaches bilateral grasp and ends detached is a complete failed episode,
        # regardless of whether it reached the lift-confirmed band.
        for index in range(max(start, 0), release):
            call_states = changes_by_call.get(index, ())
            entered_active = active
            saw_held = False
            observed_active = entered_active
            saw_detach_after_hold = False
            for new in call_states:
                state = new
                new_active = new.startswith("BILATERAL_GRASP_")
                saw_held = saw_held or new_active
                saw_detach_after_hold = (
                    saw_detach_after_hold or (observed_active and not new_active)
                )
                observed_active = new_active
            endpoint_held = state.startswith("BILATERAL_GRASP_")
            if (
                endpoint_held
                and saw_detach_after_hold
                and index > start
                and recovered_loss_index is None
            ):
                recovered_loss_index = index
                recovered_loss_api = executed_api_calls[index]
            if not entered_active and endpoint_held:
                acquisition_index = index
                acquisition_api = executed_api_calls[index]
            elif (
                not entered_active
                and not endpoint_held
                and saw_held
                and index > start
            ):
                acquisition_index = index
                acquisition_api = executed_api_calls[index]
                loss_index = index
                loss_api = executed_api_calls[index]
                active = False
                break
            elif entered_active and not endpoint_held and index > start:
                loss_index = index
                loss_api = executed_api_calls[index]
                active = False
                break
            active = endpoint_held
            if active and state == "BILATERAL_GRASP_LIFT_CONFIRMED":
                if lift_index is None:
                    lift_index = index
                    lift_api = executed_api_calls[index]
        if held_before_window or acquisition_index is not None:
            candidates.append(
                _BoundGraspEpisode(
                    "bound",
                    None if start < 0 else start,
                    None if release == len(executed_api_calls) else release,
                    acquisition_index,
                    lift_index,
                    loss_index,
                    acquisition_api,
                    lift_api,
                    loss_api,
                    recovered_loss_index,
                    recovered_loss_api,
                    executor.entity_id,
                )
            )
    if any_unresolved:
        return _BoundGraspEpisode("ambiguous_unindexed_event")
    if len(candidates) > 1:
        return _BoundGraspEpisode("ambiguous_executor")
    if candidates:
        return candidates[0]
    addressed_closes = [
        (index, _public_api_executor_id(scene, name))
        for index, name in enumerate(executed_api_calls)
        if _public_api_role(name) == "close_gripper"
        and _public_api_executor_id(scene, name) is not None
    ]
    if len(addressed_closes) == 1:
        start, executor_id = addressed_closes[0]
        release = next(
            (
                index
                for index in range(start + 1, len(executed_api_calls))
                if _public_api_role(executed_api_calls[index]) == "open_gripper"
                and _api_addresses_executor(
                    scene, executed_api_calls[index], executor_id
                )
            ),
            len(executed_api_calls),
        )
        return _BoundGraspEpisode(
            "bound_no_grasp",
            start,
            None if release == len(executed_api_calls) else release,
            executor_id=executor_id,
        )
    has_unbound_close = any(
        _public_api_role(name) == "close_gripper" for name in executed_api_calls
    )
    if len(scene.executors) > 1 and has_unbound_close:
        return _BoundGraspEpisode("ambiguous_executor")
    return _BoundGraspEpisode("no_executed_acquisition_window")


def _state_after_api_call(
    before: closed_gt.CompiledSnapshot,
    event_batches: tuple[SemanticEventBatch, ...],
    executed_api_calls: tuple[str, ...],
    call_index: int,
    required_keys: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Reconstruct selected qualitative state at one indexed public-API boundary."""
    state = dict(before.values)
    for batch in event_batches:
        index = _event_call_index(batch, executed_api_calls)
        if index is None:
            if any(key in required_keys for key, _old, _new in batch.changes):
                return ()
            continue
        if index <= call_index:
            for key, _old, new in batch.changes:
                state[key] = new
    return tuple((key, state[key]) for key in required_keys)


def align_public_operator_effects(
    scene: closed_gt.SceneSpec,
    before: closed_gt.CompiledSnapshot,
    after: closed_gt.CompiledSnapshot,
    event_batches: tuple[SemanticEventBatch, ...],
    interval_outcomes: tuple[tuple[str, str], ...],
    executed_api_calls: tuple[str, ...],
    executed_code_blocks: tuple[str, ...],
    *,
    mode: str = "contradictions",
) -> EffectAlignment:
    """Compare planned public-operator effects with the calls and GT that occurred."""
    if mode not in EFFECT_ALIGNMENT_MODES:
        raise ValueError(f"Unknown effect alignment mode: {mode!r}")
    unknown = {
        outcome
        for _key, outcome in interval_outcomes
        if outcome not in _KNOWN_EFFECT_ALIGNMENT_OUTCOMES
    }
    if unknown:
        raise ValueError(f"Unknown interval grasp outcome: {sorted(unknown)!r}")
    latest_code = executed_code_blocks[-1] if executed_code_blocks else ""
    planned_api_calls = tuple(
        name for _line, name, _literal in _called_public_names(latest_code)
    )
    object_id = _bind_effect_object(scene, before, interval_outcomes, latest_code)
    if object_id is None:
        return EffectAlignment("silent", None, (), (), "", mode)

    planned_close_indices = [
        index
        for index, name in enumerate(planned_api_calls)
        if _public_api_role(name) == "close_gripper"
    ]
    executed_close_indices = [
        index
        for index, name in enumerate(executed_api_calls)
        if _public_api_role(name) == "close_gripper"
    ]
    held_before = any(
        before.values[
            f"executor_object.{executor.entity_id}.{object_id}.grasp_contact_state"
        ].startswith("BILATERAL_GRASP_")
        for executor in scene.executors
    )
    if planned_close_indices:
        planned_window_start = planned_close_indices[-1]
    elif held_before:
        planned_window_start = -1
    else:
        return EffectAlignment("silent", object_id, (), (), "", mode)
    planned_release_index = next(
        (
            index
            for index in range(planned_window_start + 1, len(planned_api_calls))
            if _public_api_role(planned_api_calls[index]) == "open_gripper"
        ),
        len(planned_api_calls),
    )
    motion_planned = any(
        _public_api_role(name) == "goto_pose"
        for name in planned_api_calls[
            planned_window_start + 1 : planned_release_index
        ]
    )
    if executed_close_indices:
        executed_window_start = executed_close_indices[-1]
    elif held_before:
        executed_window_start = -1
    else:
        executed_window_start = len(executed_api_calls)
    executed_release_index = next(
        (
            index
            for index in range(executed_window_start + 1, len(executed_api_calls))
            if _public_api_role(executed_api_calls[index]) == "open_gripper"
        ),
        len(executed_api_calls),
    )
    motion_executed = any(
        _public_api_role(name) == "goto_pose"
        for name in executed_api_calls[
            executed_window_start + 1 : executed_release_index
        ]
    )

    names = _entity_names(scene)
    object_name = names[object_id]
    object_outcomes = [
        outcome
        for key, outcome in interval_outcomes
        if key.split(".")[2] == object_id
    ]
    if len(object_outcomes) != len(scene.executors):
        raise ValueError("Effect alignment outcome registry is incomplete")
    outcome = object_outcomes[0]
    acquisition_api = _acquisition_api_for_object(
        scene, object_id, event_batches
    )
    grasp_episode = (
        _bind_grasp_episode(
            scene,
            before,
            object_id,
            event_batches,
            executed_api_calls,
        )
        if mode
        in {
            "instance_bound_grasp_loss_only",
            "instance_bound_grasp_outcome_only",
            "instance_bound_transport_failure_only",
            "instance_bound_off_table_lift_regression_only",
            "instance_bound_grasp_loss_plus_off_table_lift_regression",
        }
        else _BoundGraspEpisode("not_evaluated")
    )
    expected: list[str] = []
    contradicted: list[str] = []
    effect_lines: dict[str, str] = {}
    confirmed_lines: dict[str, str] = {}
    acquisition_endpoint: tuple[tuple[str, str], ...] = ()

    if planned_close_indices:
        expected.append("BILATERAL_GRASP_ESTABLISHED")
        if mode in {
            "instance_bound_grasp_outcome_only",
            "instance_bound_transport_failure_only",
            "instance_bound_off_table_lift_regression_only",
            "instance_bound_grasp_loss_plus_off_table_lift_regression",
        }:
            grasp_contradicted = grasp_episode.status == "bound_no_grasp"
            grasp_confirmed = grasp_episode.status == "bound"
        else:
            grasp_contradicted = outcome not in _GRASP_OBSERVED_OUTCOMES
            grasp_confirmed = (
                executed_close_indices
                and acquisition_api is not None
                and _public_api_role(acquisition_api) == "close_gripper"
            )
        if grasp_contradicted:
            contradicted.append("BILATERAL_GRASP_ESTABLISHED")
            if not executed_close_indices:
                effect_lines["BILATERAL_GRASP_ESTABLISHED"] = (
                    f"- CONTRADICTED: the planned close-gripper operation for "
                    f"{object_name} was not executed, and bilateral grasp was not "
                    "established."
                )
            else:
                if (
                    mode == "instance_bound_grasp_outcome_only"
                    and grasp_episode.start_call_index is not None
                    and len(scene.executors) == 1
                ):
                    executor_id = scene.executors[0].entity_id
                    prefix = f"executor_object.{executor_id}.{object_id}"
                    endpoint_keys = (
                        f"{prefix}.contact",
                        f"{prefix}.relative_x",
                        f"{prefix}.relative_y",
                        f"{prefix}.relative_z",
                    )
                    acquisition_endpoint = _state_after_api_call(
                        before,
                        event_batches,
                        executed_api_calls,
                        grasp_episode.start_call_index,
                        endpoint_keys,
                    )
                effect_lines["BILATERAL_GRASP_ESTABLISHED"] = (
                    f"- CONTRADICTED: the executed close-gripper effect for "
                    f"{object_name} did not establish bilateral grasp."
                )
                if acquisition_endpoint:
                    endpoint = dict(acquisition_endpoint)
                    effect_lines["BILATERAL_GRASP_ESTABLISHED"] += (
                        " At that operation's verified endpoint, contact between "
                        f"the robot gripper and {object_name} was "
                        f"{_public_value(scene, endpoint[endpoint_keys[0]])}; the "
                        "robot gripper's x, y, and z positions relative to "
                        f"{object_name} were respectively "
                        f"{_public_value(scene, endpoint[endpoint_keys[1]])}, "
                        f"{_public_value(scene, endpoint[endpoint_keys[2]])}, and "
                        f"{_public_value(scene, endpoint[endpoint_keys[3]])}."
                    )
        elif grasp_confirmed:
            confirmed_lines["BILATERAL_GRASP_ESTABLISHED"] = (
                f"- CONFIRMED: bilateral grasp of {object_name} was established."
            )

    if motion_planned:
        expected.append("GRASP_RETAINED_DURING_POST_GRASP_MOTION")
        loss_api = (
            grasp_episode.loss_api
            if mode
            in {
                "instance_bound_grasp_loss_only",
                "instance_bound_grasp_outcome_only",
                "instance_bound_transport_failure_only",
                "instance_bound_off_table_lift_regression_only",
                "instance_bound_grasp_loss_plus_off_table_lift_regression",
            }
            else _loss_api_for_object(scene, object_id, event_batches)
        )
        if (
            motion_executed
            and loss_api is not None
            and _public_api_role(loss_api) != "open_gripper"
        ):
            contradicted.append("GRASP_RETAINED_DURING_POST_GRASP_MOTION")
            effect_lines["GRASP_RETAINED_DURING_POST_GRASP_MOTION"] = (
                f"- CONTRADICTED: bilateral grasp of {object_name} was lost during "
                f"public API `{loss_api}` before release."
            )
        elif (
            mode == "instance_bound_grasp_loss_plus_off_table_lift_regression"
            and motion_executed
            and grasp_episode.recovered_loss_api is not None
        ):
            contradicted.append("GRASP_RETAINED_DURING_POST_GRASP_MOTION")
            effect_lines["GRASP_RETAINED_DURING_POST_GRASP_MOTION"] = (
                f"- CONTRADICTED: bilateral grasp of {object_name} was interrupted "
                f"during public API `{grasp_episode.recovered_loss_api}` before "
                "release and was re-established by that API's endpoint."
            )
        elif mode in {
            "instance_bound_grasp_outcome_only",
            "instance_bound_transport_failure_only",
            "instance_bound_off_table_lift_regression_only",
            "instance_bound_grasp_loss_plus_off_table_lift_regression",
        }:
            retention_confirmed = (
                motion_executed
                and grasp_episode.status == "bound"
                and grasp_episode.loss_call_index is None
                and grasp_episode.recovered_loss_call_index is None
            )
            if retention_confirmed:
                confirmed_lines["GRASP_RETAINED_DURING_POST_GRASP_MOTION"] = (
                    f"- CONFIRMED: bilateral grasp of {object_name} was retained "
                    "during the executed post-grasp motion before release."
                )
        elif (
            motion_executed
            and (held_before or acquisition_api is not None)
            and outcome in _GRASP_OBSERVED_OUTCOMES
        ):
            confirmed_lines["GRASP_RETAINED_DURING_POST_GRASP_MOTION"] = (
                f"- CONFIRMED: bilateral grasp of {object_name} was retained during "
                "the executed post-grasp motion before release."
            )

        table_key = f"object.{object_id}.table_contact"
        if before.values[table_key] == "TRUE":
            expected.append("LIFT_ESTABLISHED_BEFORE_RELEASE")
            if mode in {
                "instance_bound_grasp_outcome_only",
                "instance_bound_transport_failure_only",
                "instance_bound_off_table_lift_regression_only",
                "instance_bound_grasp_loss_plus_off_table_lift_regression",
            }:
                lift_contradicted = (
                    motion_executed
                    and (
                        grasp_episode.status == "bound"
                        or (
                            mode == "instance_bound_grasp_outcome_only"
                            and grasp_episode.status == "bound_no_grasp"
                        )
                    )
                    and outcome not in _LIFT_OBSERVED_OUTCOMES
                )
                lift_confirmed = (
                    grasp_episode.status == "bound"
                    and outcome in _LIFT_OBSERVED_OUTCOMES
                )
            else:
                lift_contradicted = outcome not in _LIFT_OBSERVED_OUTCOMES
                lift_confirmed = not lift_contradicted
            if lift_contradicted:
                contradicted.append("LIFT_ESTABLISHED_BEFORE_RELEASE")
                retained = outcome == "BILATERAL_GRASP_WITHOUT_LIFT_RETAINED"
                if not motion_executed:
                    suffix = (
                        "; bilateral grasp remained established"
                        if retained
                        else ""
                    )
                    effect_lines["LIFT_ESTABLISHED_BEFORE_RELEASE"] = (
                        f"- CONTRADICTED: the planned post-grasp motion of "
                        f"{object_name} was not executed, and lift was not established"
                        f"{suffix}."
                    )
                else:
                    suffix = (
                        "bilateral grasp was retained, but lift was not established."
                        if retained
                        else "lift was not established."
                    )
                    effect_lines["LIFT_ESTABLISHED_BEFORE_RELEASE"] = (
                        f"- CONTRADICTED: post-grasp motion of {object_name} required "
                        f"lift before release; {suffix}"
                    )
            elif lift_confirmed:
                confirmed_lines["LIFT_ESTABLISHED_BEFORE_RELEASE"] = (
                    f"- CONFIRMED: lift of {object_name} was established before "
                    "release."
                )

        if mode in {
            "instance_bound_off_table_lift_regression_only",
            "instance_bound_grasp_loss_plus_off_table_lift_regression",
        }:
            regression_effect = "LIFT_CONFIRMED_STATE_RETAINED_AT_ENDPOINT"
            expected.append(regression_effect)
            grasp_state_key = (
                f"executor_object.{grasp_episode.executor_id}.{object_id}."
                "grasp_contact_state"
                if grasp_episode.executor_id is not None
                else None
            )
            off_table_regression = (
                grasp_state_key is not None
                and motion_executed
                and grasp_episode.status == "bound"
                and grasp_episode.loss_call_index is None
                and outcome in _LIFT_OBSERVED_OUTCOMES
                and after.values[table_key] == "FALSE"
                and after.values[grasp_state_key]
                == "BILATERAL_GRASP_BELOW_LIFT_BAND"
            )
            if off_table_regression:
                contradicted.append(regression_effect)
                effect_lines[regression_effect] = (
                    f"- CONTRADICTED: lift-confirmed bilateral grasp of "
                    f"{object_name} occurred during the executed post-grasp "
                    "motion but was not retained at the public-API endpoint; "
                    f"at that endpoint bilateral grasp of {object_name} remained, "
                    f"{object_name} had executor support without table contact, "
                    "and its grasp state was below the lift-confirmed band."
                )

    release_reference_id = None
    if mode in {
        "release_support",
        "post_release_support_only",
        "compact_outcome",
        "effect_ledger",
    }:
        planned_open = planned_release_index < len(planned_api_calls)
        executed_open = executed_release_index < len(executed_api_calls)
        if planned_open and executed_open:
            release_reference_id = _bind_release_reference(
                scene,
                object_id,
                latest_code,
                planned_window_start,
                planned_release_index,
            )
        if release_reference_id is not None:
            effect = "SUPPORT_BY_PLANNED_REFERENCE_ESTABLISHED_AFTER_RELEASE"
            expected.append(effect)
            reference_name = names[release_reference_id]
            support_key = (
                f"object_object.{object_id}.{release_reference_id}."
                "supported_by_reference"
            )
            if after.values[support_key] != "TRUE":
                contradicted.append(effect)
                stability = _public_value(
                    scene, after.values[f"object.{object_id}.support_stability"]
                )
                effect_lines[effect] = (
                    f"- CONTRADICTED: after the executed release, {object_name} was "
                    f"not supported by the uniquely referenced {reference_name}; its "
                    f"verified endpoint had {stability}."
                )
            else:
                confirmed_lines[effect] = (
                    f"- CONFIRMED: after the executed release, {object_name} was "
                    f"supported by the uniquely referenced {reference_name}."
                )

    text = ""
    confirmed = [effect for effect in expected if effect in confirmed_lines]
    selected: list[str] = []
    header = "Verified operator-effect contradictions:"
    if contradicted:
        if mode == "contradictions":
            selected = [effect_lines[item] for item in contradicted]
        elif mode == "first_failure":
            selected = [effect_lines[contradicted[0]]]
        elif mode == "verified_prefix":
            header = "Verified operator-effect prefix:"
            first_failure = expected.index(contradicted[0])
            selected = [
                confirmed_lines[item]
                for item in expected[:first_failure]
                if item in confirmed_lines
            ]
            selected.append(effect_lines[contradicted[0]])
        elif mode == "release_support":
            selected = [
                effect_lines[item]
                for item in contradicted
                if item == "SUPPORT_BY_PLANNED_REFERENCE_ESTABLISHED_AFTER_RELEASE"
            ]
            if not selected:
                selected = [effect_lines[contradicted[0]]]
        elif mode == "grasp_loss_only":
            selected = [
                effect_lines[item]
                for item in contradicted
                if item == "GRASP_RETAINED_DURING_POST_GRASP_MOTION"
            ]
        elif mode == "instance_bound_grasp_loss_only":
            selected = [
                effect_lines[item]
                for item in contradicted
                if item == "GRASP_RETAINED_DURING_POST_GRASP_MOTION"
            ]
        elif mode == "instance_bound_grasp_outcome_only":
            causal_order = (
                "BILATERAL_GRASP_ESTABLISHED",
                "LIFT_ESTABLISHED_BEFORE_RELEASE",
                "GRASP_RETAINED_DURING_POST_GRASP_MOTION",
            )
            selected = [
                effect_lines[item]
                for item in causal_order
                if item in effect_lines
            ][:1]
        elif mode == "instance_bound_transport_failure_only":
            causal_order = (
                "GRASP_RETAINED_DURING_POST_GRASP_MOTION",
                "LIFT_ESTABLISHED_BEFORE_RELEASE",
            )
            selected = [
                effect_lines[item]
                for item in causal_order
                if item in effect_lines
            ][:1]
        elif mode == "instance_bound_off_table_lift_regression_only":
            selected = [
                effect_lines[item]
                for item in ("LIFT_CONFIRMED_STATE_RETAINED_AT_ENDPOINT",)
                if item in effect_lines
            ]
        elif mode == "instance_bound_grasp_loss_plus_off_table_lift_regression":
            selected = [
                effect_lines[item]
                for item in (
                    "LIFT_CONFIRMED_STATE_RETAINED_AT_ENDPOINT",
                    "GRASP_RETAINED_DURING_POST_GRASP_MOTION",
                )
                if item in effect_lines
            ][:1]
        elif mode == "post_release_support_only":
            selected = [
                effect_lines[item]
                for item in contradicted
                if item == "SUPPORT_BY_PLANNED_REFERENCE_ESTABLISHED_AFTER_RELEASE"
            ]
        elif mode == "compact_outcome":
            header = "Verified compact operator outcome:"
            selected = [
                "- CONFIRMED: "
                + "; ".join(
                    line.removeprefix("- CONFIRMED: ").removesuffix(".")
                    for item in confirmed
                    if (line := confirmed_lines.get(item)) is not None
                )
                + "."
            ] if confirmed else []
            selected.append(effect_lines[contradicted[0]])
        elif mode == "effect_ledger":
            header = "Verified operator-effect ledger:"
            selected = [
                effect_lines[item] if item in effect_lines else confirmed_lines[item]
                for item in expected
                if item in effect_lines or item in confirmed_lines
            ]
        if selected:
            text = header + "\n" + "\n".join(selected)
        _validate_public_report(scene, text)
    return EffectAlignment(
        "mismatch" if text else "silent",
        object_id,
        tuple(expected),
        tuple(contradicted),
        text,
        mode,
        tuple(confirmed),
        release_reference_id,
        grasp_episode.status,
        grasp_episode.start_call_index,
        grasp_episode.release_call_index,
        grasp_episode.acquisition_call_index,
        grasp_episode.lift_call_index,
        grasp_episode.loss_call_index,
        acquisition_endpoint,
        grasp_episode.recovered_loss_call_index,
    )


def _validate_public_report(scene: closed_gt.SceneSpec, report: str) -> None:
    lowered = report.lower()
    if any(term in lowered for term in _FORBIDDEN_PUBLIC_TERMS):
        raise ValueError("Ideal V4 report contains prohibited content")
    private_ids = [
        entity.entity_id.lower()
        for entity in (*scene.executors, *scene.objects, scene.table)
    ]
    if any(entity_id in lowered for entity_id in private_ids):
        raise ValueError("Ideal V4 report exposes a private entity ID")
    if _NUMERIC_TOKEN.search(report):
        raise ValueError("Ideal V4 report exposes a numeric value")


def render_takeover_report(
    scene: closed_gt.SceneSpec,
    before: closed_gt.CompiledSnapshot,
    after: closed_gt.CompiledSnapshot,
    transition: closed_gt.CompiledTransition,
    event_batches: tuple[SemanticEventBatch, ...] = (),
    placement_outcomes_override: tuple[tuple[str, str], ...] | None = None,
    release_bound_placement: ReleaseBoundPlacementOutcome | None = None,
    release_bound_delta: ReleaseBoundPlacementDelta | None = None,
    api_transition_ledger: tuple[ApiTransitionLedgerEntry, ...] | None = None,
) -> RenderedTakeover:
    """Render a reversible controlled-NL encoding of state and transition GT."""
    closed_gt.validate_snapshot(scene, before)
    closed_gt.validate_snapshot(scene, after)
    closed_gt.validate_transition(scene, before, after, transition)
    specs = _action_visible_specs(scene)
    spec_by_key = {spec.key: spec for spec in closed_gt.predicate_specs(scene)}

    interval_outcomes = _interval_grasp_outcomes(
        scene, before, after, event_batches
    )
    placement_outcomes = (
        _interval_placement_outcomes(
            scene, before, after, event_batches, interval_outcomes
        )
        if placement_outcomes_override is None
        else placement_outcomes_override
    )
    public_event_batches = _public_event_batches(scene, event_batches)
    names = _entity_names(scene)
    lines = ["Verified grasp result over the entire executed code:"]
    for key, outcome in interval_outcomes:
        spec = spec_by_key[key]
        executor_name, object_name = (names[item] for item in spec.arguments)
        lines.append(
            f"- {executor_name} and {object_name}: "
            f"{_public_value(scene, outcome)}."
        )

    if release_bound_placement is not None:
        release_pair = (
            release_bound_placement.subject_id,
            release_bound_placement.reference_id,
        )
        placement_pairs = {
            tuple(key.split(".")[1:]) for key, _outcome in placement_outcomes
        }
        if placement_pairs != {release_pair}:
            raise ValueError("Release-bound snapshot conflicts with placement binding")
        lines.extend(("", "Verified release-bound placement result:"))
        lines.append(_render_release_bound_placement(scene, release_bound_placement))
    elif placement_outcomes:
        lines.extend(("", "Verified placement result over the entire executed code:"))
        for key, outcome in placement_outcomes:
            _family, subject_id, reference_id = key.split(".")
            lines.append(
                f"- {names[subject_id]} relative to {names[reference_id]}: "
                f"{_public_value(scene, outcome)}."
            )

    if release_bound_delta is not None:
        lines.extend(("", "Verified comparison with the previous bound release:"))
        lines.extend(_render_release_bound_delta(scene, release_bound_delta))

    lines.extend(("", "Verified physical endpoint after the executed code:"))
    current_fields: list[str] = []
    changed_fields: list[str] = []
    unchanged_fields: list[str] = []
    for spec in specs:
        label = _public_label(scene, spec)
        lines.append(f"- {label}: {_public_value(scene, after.values[spec.key])}.")
        current_fields.append(spec.key)
        if before.values[spec.key] == after.values[spec.key]:
            unchanged_fields.append(spec.key)
        else:
            changed_fields.append(spec.key)

    lines.extend(("", "Verified endpoint difference from the previous endpoint:"))
    if changed_fields:
        for key in changed_fields:
            spec = spec_by_key[key]
            lines.append(
                f"- {_public_label(scene, spec)} changed from "
                f"{_public_value(scene, before.values[key])} to "
                f"{_public_value(scene, after.values[key])}."
            )
        lines.append("- Every listed current-state fact not listed as changed remained unchanged.")
    else:
        lines.append("- Every listed current-state fact remained unchanged.")

    pose_fields: list[str] = []
    for entity in (*scene.executors, *scene.objects):
        key = f"transition.{entity.kind}.{entity.entity_id}.pose"
        lines.append(
            f"- {names[entity.entity_id]} pose: "
            f"{_public_value(scene, transition.values[key])}."
        )
        pose_fields.append(key)

    if api_transition_ledger is not None:
        lines.extend(("", "Verified public-API transition ledger:"))
        if api_transition_ledger:
            lines.extend(_render_api_transition_ledger(scene, api_transition_ledger))
        else:
            lines.append("- No physical public API had a uniquely bound state transition.")
    else:
        lines.extend(("", "Verified ordered factual events during the executed code:"))
    temporal_fields: list[str] = []
    if api_transition_ledger is None and not public_event_batches:
        lines.append(
            "- No tracked grasp-contact, aperture, contact, support, or access fact changed."
        )
    for batch_index, batch in enumerate(
        public_event_batches if api_transition_ledger is None else ()
    ):
        prefix = "First" if batch_index == 0 else "Then"
        provenance = (
            f", during public API `{batch.api_name}`"
            if batch.api_name is not None
            else ""
        )
        rendered_changes: list[str] = []
        for key, old, new in batch.changes:
            if key not in spec_by_key or spec_by_key[key].family not in _TEMPORAL_FAMILIES:
                raise ValueError(f"Takeover event contains an untracked field: {key}")
            rendered_changes.append(
                f"{_public_label(scene, spec_by_key[key])} changed from "
                f"{_public_value(scene, old)} to {_public_value(scene, new)}"
            )
            temporal_fields.append(key)
        lines.append(
            f"- {prefix}{provenance}, in one simulator step: "
            f"{'; '.join(rendered_changes)}."
        )
    report = "\n".join(lines)
    _validate_public_report(scene, report)
    return RenderedTakeover(
        text=report,
        current_fields=tuple(current_fields),
        changed_fields=tuple(changed_fields),
        unchanged_fields=tuple(unchanged_fields),
        pose_fields=tuple(pose_fields),
        temporal_event_fields=tuple(temporal_fields),
        interval_grasp_outcomes=interval_outcomes,
        interval_placement_outcomes=placement_outcomes,
    )


_INLINE_API_PROVENANCE = re.compile(
    r", during public API `[A-Za-z_][A-Za-z0-9_]*`"
)


def project_to_v2_1_report(report: str) -> str:
    """Remove V4 provenance annotations without changing factual content."""
    return _INLINE_API_PROVENANCE.sub("", report)


def _replace_report_once(
    prompt: list[dict[str, Any]], *, original: str, replacement: str
) -> list[dict[str, Any]]:
    if not original.strip() or original == replacement:
        raise ValueError("Takeover requires a distinct non-empty native report")
    updated = copy.deepcopy(prompt)
    matches = 0
    for message in updated:
        content = message.get("content")
        if isinstance(content, str):
            count = content.count(original)
            if count:
                message["content"] = content.replace(original, replacement)
                matches += count
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    continue
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                count = text.count(original)
                if count:
                    item["text"] = text.replace(original, replacement)
                    matches += count
    if matches != 1:
        raise ValueError(f"Takeover expected one native report carrier, found {matches}")
    return updated


def _non_text_structure(prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    structure: list[dict[str, Any]] = []
    for message in prompt:
        content = message.get("content")
        item = {"role": message.get("role"), "extra": sorted(set(message) - {"role", "content"})}
        if isinstance(content, str):
            item["content_types"] = ["text"]
        elif isinstance(content, list):
            item["content_types"] = [
                value.get("type") if isinstance(value, dict) else type(value).__name__
                for value in content
            ]
            item["non_text"] = [
                value
                for value in content
                if not isinstance(value, dict) or value.get("type") != "text"
            ]
        else:
            item["content_types"] = [type(content).__name__]
        structure.append(item)
    return structure


def _install_event_recorder(
    env: Any,
    *,
    key: tuple[str, int],
    initial: closed_gt.CompiledSnapshot,
) -> bool:
    recorder = _SemanticEventRecorder(closed_gt.cube_scene_spec(), initial, [])
    _PRIVATE_EVENT_RECORDERS[key] = recorder
    low_level = getattr(env, "low_level_env", None)
    if low_level is None or not hasattr(low_level, "_notify_post_sim_step_observer"):
        return False

    def observe_after_step() -> None:
        if recorder.error is not None:
            return
        try:
            recorder.observe(closed_gt.extract_cube_snapshot(env))
        except Exception as exc:  # Stored and raised outside student code execution.
            recorder.error = f"{type(exc).__name__}: {exc}"

    low_level._capx_post_sim_step_observer = observe_after_step
    return True


def _extent_integrity_audit(
    env: Any,
    scene: closed_gt.SceneSpec,
    *,
    api_call_index: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> ExtentIntegrityAudit | None:
    """Compare an exposed rigid extent with the bound object's private GT size."""
    query = args[0] if args else kwargs.get("object_name")
    if not isinstance(query, str):
        return None
    object_ids = {
        obj.public_name.casefold(): obj.entity_id for obj in scene.objects
    }
    object_id = object_ids.get(query.strip().casefold())
    if object_id is None:
        return None
    if not isinstance(result, tuple) or len(result) < 3 or result[2] is None:
        return None

    raw = closed_gt.extract_cube_raw_scene(env)
    expected = raw.objects[object_id].half_extents
    if expected is None:
        raise ValueError("Bound rigid object has no GT extent")
    expected_signature = rigid_extent_signature(expected * 2.0)
    expected_hash = _sha256_json(expected_signature.tolist())
    try:
        observed_signature = rigid_extent_signature(result[2])
    except ValueError:
        return ExtentIntegrityAudit(
            api_call_index,
            object_id,
            "INCONSISTENT",
            None,
            expected_hash,
        )
    status = (
        "CONSISTENT"
        if rigid_extent_is_consistent(observed_signature, expected_signature)
        else "INCONSISTENT"
    )
    return ExtentIntegrityAudit(
        api_call_index,
        object_id,
        status,
        _sha256_json(observed_signature.tolist()),
        expected_hash,
    )


def _render_extent_integrity(
    scene: closed_gt.SceneSpec,
    object_ids: tuple[str, ...],
) -> str:
    names = _entity_names(scene)
    facts = "\n".join(
        "- At least one extent returned by executed `get_object_pose` for "
        f"{names[object_id]} was inconsistent with the simulator-backed "
        "rigid geometry of that same object."
        for object_id in object_ids
    )
    return f"Verified native-observation integrity:\n{facts}"


def _restore_and_wrap_apis(env: Any, recorder: _SemanticEventRecorder) -> tuple[str, ...]:
    """Bind observed simulator-step deltas to the active public API call."""
    wrapped_names: list[str] = []
    for api in getattr(env, "_apis", {}).values():
        originals = getattr(api, "_capx_ideal_tk_v4_originals", None)
        if originals is not None:
            for name, original in originals.items():
                setattr(api, name, original)
        originals = dict(api.functions())
        api._capx_ideal_tk_v4_originals = originals
        for name, original in originals.items():
            if not hasattr(api, name):
                continue

            @functools.wraps(original)
            def wrapped(
                *args: Any,
                __name: str = name,
                __original: Callable[..., Any] = original,
                **kwargs: Any,
            ) -> Any:
                call_index = len(recorder.pending_api_calls)
                recorder.pending_api_calls.append(__name)
                state_before = tuple(sorted(recorder.latest.values.items()))
                recorder.active_api_calls.append((__name, call_index))
                result: Any = None
                completed = False
                try:
                    result = __original(*args, **kwargs)
                    completed = True
                    return result
                finally:
                    if completed and __name == "get_object_pose":
                        try:
                            audit = _extent_integrity_audit(
                                env,
                                recorder.scene,
                                api_call_index=call_index,
                                args=args,
                                kwargs=kwargs,
                                result=result,
                            )
                            if audit is not None:
                                recorder.pending_extent_integrity.append(audit)
                        except Exception as exc:
                            recorder.error = f"{type(exc).__name__}: {exc}"
                    try:
                        recorder.observe(closed_gt.extract_cube_snapshot(env))
                    except Exception as exc:
                        recorder.error = f"{type(exc).__name__}: {exc}"
                    popped_name, popped_index = recorder.active_api_calls.pop()
                    if popped_name != __name or popped_index != call_index:
                        recorder.error = "Public API provenance stack is inconsistent"
                    recorder.pending_api_transitions.append(
                        ApiTransitionBatch(
                            __name,
                            call_index,
                            state_before,
                            tuple(sorted(recorder.latest.values.items())),
                        )
                    )

            setattr(api, name, wrapped)
            wrapped_names.append(name)
    for api in getattr(env, "_apis", {}).values():
        for name, function in api.functions().items():
            env._exec_globals[name] = function
    return tuple(sorted(set(wrapped_names)))


def _consume_event_batches(
    env: Any,
    *,
    key: tuple[str, int],
    endpoint: closed_gt.CompiledSnapshot,
) -> tuple[SemanticEventBatch, ...]:
    recorder = _PRIVATE_EVENT_RECORDERS.get(key)
    if recorder is None:
        raise RuntimeError("Closed-GT temporal recorder is missing")
    if recorder.error is not None:
        raise RuntimeError(f"Closed-GT temporal recorder failed: {recorder.error}")
    recorder.observe(endpoint)
    batches = tuple(recorder.pending)
    recorder.pending.clear()
    return batches


def _consume_api_calls(
    *,
    key: tuple[str, int],
) -> tuple[str, ...]:
    recorder = _PRIVATE_EVENT_RECORDERS.get(key)
    if recorder is None:
        raise RuntimeError("Closed-GT temporal recorder is missing")
    calls = tuple(recorder.pending_api_calls)
    recorder.pending_api_calls.clear()
    return calls


def _consume_api_transitions(
    *,
    key: tuple[str, int],
) -> tuple[ApiTransitionBatch, ...]:
    recorder = _PRIVATE_EVENT_RECORDERS.get(key)
    if recorder is None:
        raise RuntimeError("Closed-GT temporal recorder is missing")
    transitions = tuple(
        sorted(recorder.pending_api_transitions, key=lambda item: item.api_call_index)
    )
    recorder.pending_api_transitions.clear()
    return transitions


def _consume_extent_integrity(
    *,
    key: tuple[str, int],
) -> tuple[ExtentIntegrityAudit, ...]:
    recorder = _PRIVATE_EVENT_RECORDERS.get(key)
    if recorder is None:
        raise RuntimeError("Closed-GT temporal recorder is missing")
    audits = tuple(recorder.pending_extent_integrity)
    recorder.pending_extent_integrity.clear()
    return audits


def initialize_trial_trace(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    task: str,
) -> None:
    if task not in {"cube_stack", "cube_restack"}:
        raise ValueError(f"Ideal V4 has no audited adapter for {task!r}")
    initial = closed_gt.extract_cube_snapshot(env)
    root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    root.mkdir(parents=True, exist_ok=False)
    key = _cache_key(output_dir, trial)
    if key in _PRIVATE_SNAPSHOTS:
        raise RuntimeError("Ideal V4 private trial cache already exists")
    _PRIVATE_SNAPSHOTS[key] = [initial]
    observer_installed = _install_event_recorder(env, key=key, initial=initial)
    recorder = _PRIVATE_EVENT_RECORDERS[key]
    wrapped_api_names = _restore_and_wrap_apis(env, recorder)
    _save_json(
        root / "state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "closed_gt_schema_version": closed_gt.SCHEMA_VERSION,
            "task": task,
            "trial": trial,
            "turns_completed": 0,
            "initial_public_state_sha256": _sha256_json(initial.values),
            "continuous_state_persisted": False,
            "action_visible_initial_gt": False,
            "sim_step_observer_installed": observer_installed,
            "wrapped_api_names": list(wrapped_api_names),
        },
    )


def initialize_passive_trial_trace(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    task: str,
) -> None:
    """Initialize action-invisible closed-GT capture for a paired M3 control."""
    if task not in {"cube_stack", "cube_restack"}:
        raise ValueError(f"Ideal V4 passive capture has no adapter for {task!r}")
    initial = closed_gt.extract_cube_snapshot(env)
    root = Path(output_dir) / PASSIVE_TRACE_DIR / f"trial_{trial:02d}"
    root.mkdir(parents=True, exist_ok=False)
    key = _cache_key(output_dir, trial)
    if key in _PRIVATE_SNAPSHOTS:
        raise RuntimeError("Ideal V4 passive private trial cache already exists")
    _PRIVATE_SNAPSHOTS[key] = [initial]
    observer_installed = _install_event_recorder(env, key=key, initial=initial)
    _save_json(
        root / "state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "closed_gt_schema_version": closed_gt.SCHEMA_VERSION,
            "mode": "passive_action_invisible",
            "task": task,
            "trial": trial,
            "turns_completed": 0,
            "initial_public_state_sha256": _sha256_json(initial.values),
            "continuous_state_persisted": False,
            "student_prompt_changed": False,
            "sim_step_observer_installed": observer_installed,
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
    """Record closed semantics without returning or changing an action prompt."""
    key = _cache_key(output_dir, trial)
    snapshots = _PRIVATE_SNAPSHOTS.get(key)
    if snapshots is None or len(snapshots) != turn:
        raise ValueError("Ideal V4 passive state is not turn-synchronous")
    root = Path(output_dir) / PASSIVE_TRACE_DIR / f"trial_{trial:02d}"
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("mode") != "passive_action_invisible"
        or state.get("task") != task
        or state.get("trial") != trial
        or state.get("turns_completed") != turn - 1
    ):
        raise ValueError("Ideal V4 passive trace provenance mismatch")
    scene = closed_gt.cube_scene_spec()
    before = snapshots[-1]
    after = closed_gt.extract_cube_snapshot(env)
    event_batches = _consume_event_batches(env, key=key, endpoint=after)
    api_transitions = _consume_api_transitions(key=key)
    executed_api_calls = _consume_api_calls(key=key)
    if tuple((item.api_call_index, item.api_name) for item in api_transitions) != tuple(
        enumerate(executed_api_calls)
    ):
        raise RuntimeError("Passive API transition ledger is not call-synchronous")
    transition = closed_gt.compile_transition(scene, before, after)
    interval_outcomes = _interval_grasp_outcomes(
        scene, before, after, event_batches
    )
    placement_outcomes = _interval_placement_outcomes(
        scene, before, after, event_batches, interval_outcomes
    )
    public_interval_outcomes = dict(interval_outcomes)
    public_placement_outcomes = dict(placement_outcomes)
    public_event_batches = _public_event_batches(scene, event_batches)
    public_events = [
        {"changes": [list(change) for change in batch.changes]}
        for batch in public_event_batches
    ]
    audited_events = [
        {
            "changes": [list(change) for change in batch.changes],
            "api_name": batch.api_name,
            "api_call_index": batch.api_call_index,
        }
        for batch in event_batches
    ]
    turn_semantics = {
        "public_transition": dict(transition.values),
        "public_interval_grasp_outcomes": public_interval_outcomes,
        "public_interval_placement_outcomes": public_placement_outcomes,
        "public_temporal_event_batches": public_events,
    }
    turn_root = root / f"turn_{turn:02d}"
    turn_root.mkdir(parents=True, exist_ok=False)
    _save_json(
        turn_root / "transition.json",
        {
            "schema_version": SCHEMA_VERSION,
            "closed_gt_schema_version": closed_gt.SCHEMA_VERSION,
            "mode": "passive_action_invisible",
            "task": task,
            "trial": trial,
            "turn": turn,
            "public_before_state_sha256": _sha256_json(before.values),
            "public_after_state_sha256": _sha256_json(after.values),
            "public_transition": dict(transition.values),
            "public_transition_sha256": _sha256_json(transition.values),
            "public_interval_grasp_outcomes": public_interval_outcomes,
            "public_interval_placement_outcomes": public_placement_outcomes,
            "public_temporal_event_batches": public_events,
            "audited_recorded_temporal_event_batches": audited_events,
            "audited_recorded_temporal_event_batches_sha256": _sha256_json(
                audited_events
            ),
            "public_executed_api_calls": list(executed_api_calls),
            "public_turn_semantics_sha256": _sha256_json(turn_semantics),
            "continuous_state_persisted": False,
            "student_prompt_changed": False,
            "student_action_request_changed": False,
        },
    )
    snapshots.append(after)
    state["turns_completed"] = turn
    state["latest_public_state_sha256"] = _sha256_json(after.values)
    state["latest_public_transition_sha256"] = _sha256_json(transition.values)
    _save_json(state_path, state)


def run_takeover(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    turn: int,
    task: str,
    native_report: str | None,
    native_decision_prompt: list[dict[str, Any]],
    private_task_completed: bool,
    executed_code_blocks: tuple[str, ...] = (),
    effect_alignment_enabled: bool | None = None,
    effect_alignment_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Replace exactly the M3 report carrier with deterministic closed GT."""
    if not isinstance(native_report, str) or not native_report.strip():
        raise ValueError("Ideal V4 requires a native M3 report")
    key = _cache_key(output_dir, trial)
    snapshots = _PRIVATE_SNAPSHOTS.get(key)
    if snapshots is None or len(snapshots) != turn:
        raise ValueError("Ideal V4 private state is not turn-synchronous")
    root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("closed_gt_schema_version") != closed_gt.SCHEMA_VERSION
        or state.get("task") != task
        or state.get("trial") != trial
        or state.get("turns_completed") != turn - 1
    ):
        raise ValueError("Ideal V4 trace provenance mismatch")

    scene = closed_gt.cube_scene_spec()
    before = snapshots[-1]
    after = closed_gt.extract_cube_snapshot(env)
    event_batches = _consume_event_batches(env, key=key, endpoint=after)
    api_transitions = _consume_api_transitions(key=key)
    executed_api_calls = _consume_api_calls(key=key)
    extent_integrity_audits = _consume_extent_integrity(key=key)
    if tuple((item.api_call_index, item.api_name) for item in api_transitions) != tuple(
        enumerate(executed_api_calls)
    ):
        raise RuntimeError("API transition ledger is not call-synchronous")
    if any(
        audit.api_call_index < 0
        or audit.api_call_index >= len(executed_api_calls)
        or executed_api_calls[audit.api_call_index] != "get_object_pose"
        or audit.status not in _EXTENT_INTEGRITY_STATUSES
        for audit in extent_integrity_audits
    ):
        raise RuntimeError("Extent-integrity audit is not call-synchronous")
    transition = closed_gt.compile_transition(scene, before, after)
    interval_outcomes = _interval_grasp_outcomes(
        scene, before, after, event_batches
    )
    raw_placement_outcomes = _interval_placement_outcomes(
        scene, before, after, event_batches, interval_outcomes
    )
    goal = closed_gt.cube_goal_definition(task)
    placement_outcomes = _select_salient_placement_outcomes(
        scene,
        goal,
        raw_placement_outcomes,
        turn=turn,
    )
    release_bound_enabled = (
        os.environ.get("CAPX_IDEAL_TK_V4_RELEASE_BOUND_PLACEMENT", "0") == "1"
    )
    latest_code = executed_code_blocks[-1] if executed_code_blocks else ""
    release_bound_candidate = (
        _release_bound_placement_outcome(
            scene,
            before,
            after,
            event_batches,
            interval_outcomes,
            placement_outcomes,
            executed_api_calls,
            latest_code,
        )
        if release_bound_enabled
        else None
    )
    k2_release_enabled = (
        os.environ.get("CAPX_IDEAL_TK_V4_K2_RELEASE_DELTA", "0") == "1"
    )
    if k2_release_enabled and not release_bound_enabled:
        raise ValueError("K2 release delta requires release-bound placement")
    release_bound_delta = (
        _release_bound_placement_delta(
            state.get("last_release_bound_placement"),
            release_bound_candidate,
        )
        if k2_release_enabled
        else None
    )
    api_ledger_enabled = (
        os.environ.get("CAPX_IDEAL_TK_V4_API_TRANSITION_LEDGER", "0") == "1"
    )
    api_ledger_mode = os.environ.get(
        "CAPX_IDEAL_TK_V4_API_TRANSITION_LEDGER_MODE", "full"
    )
    if api_ledger_mode not in API_TRANSITION_LEDGER_MODES:
        raise ValueError(f"Unknown API transition ledger mode: {api_ledger_mode!r}")
    ledger_entries = (
        _api_transition_ledger(
            scene,
            goal,
            interval_outcomes,
            latest_code,
            api_transitions,
            event_batches,
            executed_api_calls,
            mode=api_ledger_mode,
        )
        if api_ledger_enabled
        else ()
    )
    api_ledger_candidate = ledger_entries or None
    semantic_gain_gate_enabled = (
        os.environ.get("CAPX_IDEAL_TK_V4_SEMANTIC_GAIN_GATE", "0") == "1"
    )
    if semantic_gain_gate_enabled and (
        not release_bound_enabled
        or not api_ledger_enabled
        or api_ledger_mode != "phase_bound_sparse"
        or k2_release_enabled
    ):
        raise ValueError(
            "Semantic-gain gate requires release-bound placement and the "
            "phase-bound sparse ledger, with K2 disabled"
        )
    semantic_gain = (
        _release_bound_semantic_gain(
            release_bound_candidate,
            event_batches,
            executed_api_calls,
        )
        if semantic_gain_gate_enabled
        else ProjectionSemanticGain("disabled", ())
    )
    extension_candidate_visible = any(
        (
            release_bound_candidate is not None,
            release_bound_delta is not None,
            api_ledger_candidate is not None,
        )
    )
    extension_applied = extension_candidate_visible and (
        not semantic_gain_gate_enabled
        or semantic_gain.status == "novel_release_facts"
    )
    release_bound_placement = release_bound_candidate if extension_applied else None
    api_transition_ledger = api_ledger_candidate if extension_applied else None
    if not extension_applied:
        release_bound_delta = None
    public_interval_outcomes = dict(interval_outcomes)
    public_placement_outcomes = dict(placement_outcomes)
    public_event_batches = _public_event_batches(scene, event_batches)
    public_events = [
        {"changes": [list(change) for change in batch.changes]}
        for batch in public_event_batches
    ]
    audited_events = [
        {
            "changes": [list(change) for change in batch.changes],
            "api_name": batch.api_name,
            "api_call_index": batch.api_call_index,
        }
        for batch in event_batches
    ]
    event_api_provenance = [batch.api_name for batch in public_event_batches]
    event_api_call_indices = [
        batch.api_call_index for batch in public_event_batches
    ]
    turn_semantics = {
        "public_transition": dict(transition.values),
        "public_interval_grasp_outcomes": public_interval_outcomes,
        "public_interval_placement_outcomes": public_placement_outcomes,
        "public_temporal_event_batches": public_events,
    }
    release_bound_candidate_public = (
        asdict(release_bound_candidate)
        if release_bound_candidate is not None
        else None
    )
    release_bound_public = (
        asdict(release_bound_placement)
        if release_bound_placement is not None
        else None
    )
    release_delta_public = (
        asdict(release_bound_delta) if release_bound_delta is not None else None
    )
    api_ledger_candidate_public = (
        [asdict(entry) for entry in ledger_entries] if ledger_entries else None
    )
    api_ledger_public = (
        api_ledger_candidate_public if api_transition_ledger is not None else None
    )
    parent_rendered = render_takeover_report(
        scene,
        before,
        after,
        transition,
        event_batches,
        placement_outcomes_override=placement_outcomes,
    )
    rendered = render_takeover_report(
        scene,
        before,
        after,
        transition,
        event_batches,
        placement_outcomes_override=placement_outcomes,
        release_bound_placement=release_bound_placement,
        release_bound_delta=release_bound_delta,
        api_transition_ledger=api_transition_ledger,
    )
    if effect_alignment_enabled is None:
        effect_alignment_enabled = (
            os.environ.get("CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT", "0") == "1"
        )
    if effect_alignment_mode is None:
        effect_alignment_mode = os.environ.get(
            "CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT_MODE", "contradictions"
        )
    if effect_alignment_mode not in EFFECT_ALIGNMENT_MODES:
        raise ValueError(
            f"Unknown effect alignment mode: {effect_alignment_mode!r}"
        )
    effect_alignment = (
        align_public_operator_effects(
            scene,
            before,
            after,
            event_batches,
            interval_outcomes,
            executed_api_calls,
            executed_code_blocks,
            mode=effect_alignment_mode,
        )
        if effect_alignment_enabled
        else EffectAlignment(
            "disabled", None, (), (), "", effect_alignment_mode
        )
    )
    observation_integrity_enabled = (
        os.environ.get("CAPX_IDEAL_TK_V4_OBSERVATION_INTEGRITY", "0") == "1"
    )
    inconsistent_extent_objects = tuple(
        sorted(
            {
                audit.object_id
                for audit in extent_integrity_audits
                if audit.status == "INCONSISTENT"
            }
        )
    )
    observation_integrity_text = (
        _render_extent_integrity(scene, inconsistent_extent_objects)
        if observation_integrity_enabled and inconsistent_extent_objects
        else ""
    )
    takeover_text = rendered.text
    if effect_alignment.text:
        takeover_text = f"{takeover_text}\n\n{effect_alignment.text}"
        _validate_public_report(scene, takeover_text)
    g17_takeover_text = takeover_text
    if observation_integrity_text:
        takeover_text = f"{takeover_text}\n\n{observation_integrity_text}"
        _validate_public_report(scene, takeover_text)
    parent_takeover_text = parent_rendered.text
    if effect_alignment.text:
        parent_takeover_text = f"{parent_takeover_text}\n\n{effect_alignment.text}"
        _validate_public_report(scene, parent_takeover_text)
    v2_1_events = tuple(
        closed_gt_takeover.SemanticEventBatch(batch.changes)
        for batch in event_batches
    )
    v2_1_report = closed_gt_takeover.render_takeover_report(
        scene,
        before,
        after,
        transition,
        v2_1_events,
        placement_outcomes_override=placement_outcomes,
    ).text
    canonical_projection_equal = (
        project_to_v2_1_report(rendered.text) == v2_1_report
    )
    projection_extension_visible = any(
        (
            release_bound_placement is not None,
            release_bound_delta is not None,
            api_transition_ledger is not None,
        )
    )
    if not projection_extension_visible and not canonical_projection_equal:
        raise RuntimeError("Ideal V4 report is not an exact projection of v2.1")
    corrected = _replace_report_once(
        native_decision_prompt,
        original=native_report,
        replacement=takeover_text,
    )
    g17_corrected = _replace_report_once(
        native_decision_prompt,
        original=native_report,
        replacement=g17_takeover_text,
    )
    parent_corrected = _replace_report_once(
        native_decision_prompt,
        original=native_report,
        replacement=parent_takeover_text,
    )
    parent_prompt_preserved = g17_corrected == parent_corrected
    g17_parent_prompt_preserved = corrected == g17_corrected
    if g17_parent_prompt_preserved is bool(observation_integrity_text):
        raise RuntimeError("Observation integrity violated G17 prompt identity")
    if semantic_gain_gate_enabled and (
        parent_prompt_preserved is extension_applied
    ):
        raise RuntimeError("Semantic-gain gate violated parent prompt identity")
    if _non_text_structure(native_decision_prompt) != _non_text_structure(corrected):
        raise RuntimeError("Ideal V4 changed request roles or media")
    if _sha256_json(native_decision_prompt) == _sha256_json(corrected):
        raise RuntimeError("Ideal V4 did not change the action request")
    restored = _replace_report_once(
        corrected,
        original=takeover_text,
        replacement=native_report,
    )
    if restored != native_decision_prompt:
        raise RuntimeError("Ideal V4 changed a non-report prompt carrier")

    goal_audit = closed_gt.goal_iff_audit(
        after,
        goal,
        private_task_completed=private_task_completed,
    )
    if goal_audit["iff_holds"] is not True:
        raise RuntimeError("Closed-GT benchmark-goal IFF audit failed")

    turn_root = root / f"turn_{turn:02d}"
    turn_root.mkdir(parents=True, exist_ok=False)
    (turn_root / "native_report.txt").write_text(native_report, encoding="utf-8")
    (turn_root / "takeover_report.txt").write_text(takeover_text, encoding="utf-8")
    record = {
        "schema_version": SCHEMA_VERSION,
        "closed_gt_schema_version": closed_gt.SCHEMA_VERSION,
        "task": task,
        "trial": trial,
        "turn": turn,
        "carrier": "m3_visual_difference_report_exactly_once",
        "native_report_sha256": _sha256_text(native_report),
        "takeover_report_sha256": _sha256_text(takeover_text),
        "parent_report_sha256": _sha256_text(parent_takeover_text),
        "g17_parent_report_sha256": _sha256_text(g17_takeover_text),
        "canonical_a_star_report_sha256": _sha256_text(rendered.text),
        "native_prompt_sha256": _sha256_json(native_decision_prompt),
        "takeover_prompt_sha256": _sha256_json(corrected),
        "parent_prompt_sha256": _sha256_json(parent_corrected),
        "g17_parent_prompt_sha256": _sha256_json(g17_corrected),
        "parent_prompt_preserved": parent_prompt_preserved,
        "g17_parent_prompt_preserved": g17_parent_prompt_preserved,
        "non_text_structure_identical": True,
        "non_report_prompt_roundtrip_equal": True,
        "v2_1_projection_equal": (
            effect_alignment.text == "" and not projection_extension_visible
        ),
        "canonical_v2_1_projection_equal": canonical_projection_equal,
        "release_bound_placement_enabled": release_bound_enabled,
        "release_bound_placement_applied": release_bound_placement is not None,
        "release_bound_placement": release_bound_public,
        "release_bound_placement_sha256": _sha256_json(release_bound_public),
        "release_bound_placement_candidate": release_bound_candidate_public,
        "release_bound_placement_candidate_sha256": _sha256_json(
            release_bound_candidate_public
        ),
        "k2_release_delta_enabled": k2_release_enabled,
        "k2_release_delta_applied": release_bound_delta is not None,
        "k2_release_delta": release_delta_public,
        "k2_release_delta_sha256": _sha256_json(release_delta_public),
        "api_transition_ledger_enabled": api_ledger_enabled,
        "api_transition_ledger_mode": api_ledger_mode,
        "api_transition_ledger_applied": api_transition_ledger is not None,
        "api_transition_ledger": api_ledger_public,
        "api_transition_ledger_sha256": _sha256_json(api_ledger_public),
        "api_transition_ledger_candidate": api_ledger_candidate_public,
        "api_transition_ledger_candidate_sha256": _sha256_json(
            api_ledger_candidate_public
        ),
        "semantic_gain_gate_enabled": semantic_gain_gate_enabled,
        "semantic_gain_status": semantic_gain.status,
        "semantic_gain_novel_fact_ids": list(semantic_gain.novel_fact_ids),
        "semantic_gain_extension_applied": extension_applied,
        "observation_integrity_enabled": observation_integrity_enabled,
        "observation_integrity_applied": bool(observation_integrity_text),
        "observation_integrity_inconsistent_objects": list(
            inconsistent_extent_objects
        ),
        "observation_integrity_text_sha256": _sha256_json(
            observation_integrity_text or None
        ),
        "private_extent_integrity_audit": [
            asdict(audit) for audit in extent_integrity_audits
        ],
        "private_extent_integrity_audit_sha256": _sha256_json(
            [asdict(audit) for audit in extent_integrity_audits]
        ),
        "v2_1_report_sha256": _sha256_text(v2_1_report),
        "inline_api_provenance_only": True,
        "standalone_api_effect_section_visible": effect_alignment.text != "",
        "current_state_field_count": len(rendered.current_fields),
        "current_state_fields": list(rendered.current_fields),
        "changed_state_field_count": len(rendered.changed_fields),
        "changed_state_fields": list(rendered.changed_fields),
        "unchanged_state_field_count": len(rendered.unchanged_fields),
        "unchanged_state_fields": list(rendered.unchanged_fields),
        "pose_transition_fields": list(rendered.pose_fields),
        "interval_grasp_outcome_count": len(rendered.interval_grasp_outcomes),
        "public_interval_grasp_outcomes": dict(rendered.interval_grasp_outcomes),
        "interval_placement_outcome_count": len(
            rendered.interval_placement_outcomes
        ),
        "public_interval_placement_outcomes": dict(
            rendered.interval_placement_outcomes
        ),
        "audited_interval_placement_outcomes": dict(raw_placement_outcomes),
        "temporal_event_field_count": len(rendered.temporal_event_fields),
        "temporal_event_fields": list(rendered.temporal_event_fields),
        "public_temporal_event_batches": public_events,
        "audited_recorded_temporal_event_batches": audited_events,
        "audited_recorded_temporal_event_batches_sha256": _sha256_json(
            audited_events
        ),
        "public_event_api_provenance": event_api_provenance,
        "public_event_api_call_indices": event_api_call_indices,
        "public_event_api_call_indices_sha256": _sha256_json(
            event_api_call_indices
        ),
        "public_executed_api_calls": list(executed_api_calls),
        "public_event_api_provenance_sha256": _sha256_json(
            event_api_provenance
        ),
        "public_turn_semantics_sha256": _sha256_json(turn_semantics),
        "public_current_state": dict(after.values),
        "public_transition": dict(transition.values),
        "private_goal_iff_audit": goal_audit,
        "continuous_state_persisted": False,
        "raw_image_replaced": False,
        "executed_code_replaced": False,
        "console_stdout_replaced": False,
        "console_stderr_replaced": False,
        "task_or_api_text_replaced": False,
        "completion_or_reward_visible": False,
        "next_move_visible": False,
        "numeric_state_visible": False,
        "effect_alignment_enabled": effect_alignment_enabled,
        "effect_alignment_mode": effect_alignment.mode,
        "effect_alignment_status": effect_alignment.status,
        "effect_alignment_object": (
            None
            if effect_alignment.object_id is None
            else _entity_names(scene)[effect_alignment.object_id]
        ),
        "effect_alignment_expected_effects": list(effect_alignment.expected_effects),
        "effect_alignment_contradicted_effects": list(
            effect_alignment.contradicted_effects
        ),
        "effect_alignment_confirmed_effects": list(
            effect_alignment.confirmed_effects
        ),
        "effect_alignment_release_reference": (
            None
            if effect_alignment.release_reference_id is None
            else _entity_names(scene)[effect_alignment.release_reference_id]
        ),
        "effect_alignment_grasp_episode_binding_status": (
            effect_alignment.grasp_episode_binding_status
        ),
        "effect_alignment_grasp_episode_start_call_index": (
            effect_alignment.grasp_episode_start_call_index
        ),
        "effect_alignment_grasp_episode_release_call_index": (
            effect_alignment.grasp_episode_release_call_index
        ),
        "effect_alignment_grasp_episode_acquisition_call_index": (
            effect_alignment.grasp_episode_acquisition_call_index
        ),
        "effect_alignment_grasp_episode_lift_call_index": (
            effect_alignment.grasp_episode_lift_call_index
        ),
        "effect_alignment_grasp_episode_loss_call_index": (
            effect_alignment.grasp_episode_loss_call_index
        ),
        "effect_alignment_grasp_episode_recovered_loss_call_index": (
            effect_alignment.grasp_episode_recovered_loss_call_index
        ),
        "effect_alignment_grasp_episode_acquisition_endpoint": dict(
            effect_alignment.grasp_episode_acquisition_endpoint
        ),
        "canonical_a_star_prompt_unchanged": effect_alignment.text == "",
    }
    _save_json(turn_root / "takeover.json", record)
    snapshots.append(after)
    state["turns_completed"] = turn
    state["latest_public_state_sha256"] = _sha256_json(after.values)
    state["latest_public_transition_sha256"] = _sha256_json(transition.values)
    if release_bound_candidate is not None:
        state["last_release_bound_placement"] = release_bound_candidate_public
    _save_json(state_path, state)
    return corrected
