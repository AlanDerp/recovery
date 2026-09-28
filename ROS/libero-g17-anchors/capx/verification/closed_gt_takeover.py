"""Deterministic whole-report takeover backed by the closed semantic GT basis."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capx.verification import closed_gt

SCHEMA_VERSION = "capx-closed-gt-takeover-v2.2-compact-placement"
TRACE_DIR = "closed_gt_takeover_traces"
PASSIVE_TRACE_DIR = "closed_gt_takeover_passive_traces"

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


INTERVAL_GRASP_OUTCOME_VALUES = (
    "NO_EXECUTOR_CONTACT_OBSERVED",
    "CONTACT_WITHOUT_BILATERAL_GRASP_OBSERVED",
    "BILATERAL_GRASP_WITHOUT_LIFT_THEN_LOST",
    "BILATERAL_GRASP_WITHOUT_LIFT_RETAINED",
    "LIFT_CONFIRMED_THEN_LOST",
    "LIFT_CONFIRMED_AND_RETAINED",
)

INTERVAL_PLACEMENT_OUTCOME_VALUES = (
    "CENTER_SUPPORT_AT_ENDPOINT",
    "EDGE_SUPPORT_AT_ENDPOINT",
    "CENTER_SUPPORT_ESTABLISHED_THEN_LOST",
    "EDGE_SUPPORT_ESTABLISHED_THEN_LOST",
    "CENTER_REGION_ENTERED_WITHOUT_SUPPORT",
    "EDGE_REGION_ENTERED_WITHOUT_SUPPORT",
    "NEVER_ENTERED_SUPPORT_REGION",
)

_GRASP_OBSERVED_OUTCOMES = frozenset(
    {
        "BILATERAL_GRASP_WITHOUT_LIFT_THEN_LOST",
        "BILATERAL_GRASP_WITHOUT_LIFT_RETAINED",
        "LIFT_CONFIRMED_THEN_LOST",
        "LIFT_CONFIRMED_AND_RETAINED",
    }
)


@dataclass
class _SemanticEventRecorder:
    scene: closed_gt.SceneSpec
    latest: closed_gt.CompiledSnapshot
    pending: list[SemanticEventBatch]
    error: str | None = None

    def observe(self, snapshot: closed_gt.CompiledSnapshot) -> None:
        closed_gt.validate_snapshot(self.scene, snapshot)
        changes = tuple(
            (spec.key, self.latest.values[spec.key], snapshot.values[spec.key])
            for spec in _temporal_specs(self.scene)
            if self.latest.values[spec.key] != snapshot.values[spec.key]
        )
        if changes:
            self.pending.append(SemanticEventBatch(changes))
        self.latest = snapshot


# Continuous poses remain process-private. Trial subprocess isolation makes this
# cache single-trial in production; the key prevents accidental test/run mixing.
_PRIVATE_SNAPSHOTS: dict[tuple[str, int], list[closed_gt.CompiledSnapshot]] = {}
_PRIVATE_EVENT_RECORDERS: dict[tuple[str, int], _SemanticEventRecorder] = {}

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


def _pair_state_path(
    before: closed_gt.CompiledSnapshot,
    after: closed_gt.CompiledSnapshot,
    event_batches: tuple[Any, ...],
    *,
    region_key: str,
    support_key: str,
) -> tuple[tuple[str, str], ...]:
    """Reconstruct atomic region/support states without joining unrelated times."""
    current = {
        region_key: before.values[region_key],
        support_key: before.values[support_key],
    }
    path = [(current[region_key], current[support_key])]
    for batch in event_batches:
        changed = False
        for key, _old, new in batch.changes:
            if key in current:
                current[key] = new
                changed = True
        if changed:
            state = (current[region_key], current[support_key])
            if state != path[-1]:
                path.append(state)
    endpoint = (after.values[region_key], after.values[support_key])
    if endpoint != path[-1]:
        path.append(endpoint)
    for region, support in path:
        if (
            region not in closed_gt.SUPPORT_REGION_VALUES
            or support not in {"TRUE", "FALSE"}
        ):
            raise ValueError("Invalid placement state path")
        if support == "TRUE" and region == "OUTSIDE":
            raise ValueError("Support cannot hold outside the support region")
    return tuple(path)


def _placement_outcome_for_pair(
    before: closed_gt.CompiledSnapshot,
    after: closed_gt.CompiledSnapshot,
    event_batches: tuple[Any, ...],
    *,
    subject_id: str,
    reference_id: str,
) -> str:
    prefix = f"object_object.{subject_id}.{reference_id}"
    path = _pair_state_path(
        before,
        after,
        event_batches,
        region_key=f"{prefix}.support_region",
        support_key=f"{prefix}.supported_by_reference",
    )
    endpoint_region, endpoint_support = path[-1]
    if endpoint_support == "TRUE":
        outcome = (
            "CENTER_SUPPORT_AT_ENDPOINT"
            if endpoint_region == "CENTER_REGION"
            else "EDGE_SUPPORT_AT_ENDPOINT"
        )
    else:
        supported_regions = {region for region, support in path if support == "TRUE"}
        visited_regions = {region for region, _support in path}
        if "CENTER_REGION" in supported_regions:
            outcome = "CENTER_SUPPORT_ESTABLISHED_THEN_LOST"
        elif "EDGE_REGION" in supported_regions:
            outcome = "EDGE_SUPPORT_ESTABLISHED_THEN_LOST"
        elif "CENTER_REGION" in visited_regions:
            outcome = "CENTER_REGION_ENTERED_WITHOUT_SUPPORT"
        elif "EDGE_REGION" in visited_regions:
            outcome = "EDGE_REGION_ENTERED_WITHOUT_SUPPORT"
        else:
            outcome = "NEVER_ENTERED_SUPPORT_REGION"
    if outcome not in INTERVAL_PLACEMENT_OUTCOME_VALUES:
        raise RuntimeError(f"Invalid interval placement outcome: {outcome}")
    return outcome


def _interval_placement_outcomes(
    scene: closed_gt.SceneSpec,
    before: closed_gt.CompiledSnapshot,
    after: closed_gt.CompiledSnapshot,
    event_batches: tuple[Any, ...],
    interval_grasp_outcomes: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Bind a total placement outcome only to objects manipulated this interval."""
    specs = {spec.key: spec for spec in closed_gt.predicate_specs(scene)}
    active_ids = {
        specs[key].arguments[1]
        for key, outcome in interval_grasp_outcomes
        if outcome in _GRASP_OBSERVED_OUTCOMES
    }
    outcomes: list[tuple[str, str]] = []
    for subject in scene.objects:
        if subject.entity_id not in active_ids:
            continue
        for reference in scene.objects:
            if reference.entity_id == subject.entity_id:
                continue
            key = f"placement_outcome.{subject.entity_id}.{reference.entity_id}"
            outcomes.append(
                (
                    key,
                    _placement_outcome_for_pair(
                        before,
                        after,
                        event_batches,
                        subject_id=subject.entity_id,
                        reference_id=reference.entity_id,
                    ),
                )
            )
    return tuple(outcomes)


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
            public.append(SemanticEventBatch(changes))
    return tuple(public)


def _validate_public_report(scene: closed_gt.SceneSpec, report: str) -> None:
    lowered = report.lower()
    if any(term in lowered for term in _FORBIDDEN_PUBLIC_TERMS):
        raise ValueError("Closed-GT takeover report contains prohibited content")
    private_ids = [
        entity.entity_id.lower()
        for entity in (*scene.executors, *scene.objects, scene.table)
    ]
    if any(entity_id in lowered for entity_id in private_ids):
        raise ValueError("Closed-GT takeover report exposes a private entity ID")
    if _NUMERIC_TOKEN.search(report):
        raise ValueError("Closed-GT takeover report exposes a numeric value")


def render_takeover_report(
    scene: closed_gt.SceneSpec,
    before: closed_gt.CompiledSnapshot,
    after: closed_gt.CompiledSnapshot,
    transition: closed_gt.CompiledTransition,
    event_batches: tuple[SemanticEventBatch, ...] = (),
    placement_outcomes_override: tuple[tuple[str, str], ...] | None = None,
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

    if placement_outcomes:
        lines.extend(("", "Verified placement result over the entire executed code:"))
        for key, outcome in placement_outcomes:
            _family, subject_id, reference_id = key.split(".")
            lines.append(
                f"- {names[subject_id]} relative to {names[reference_id]}: "
                f"{_public_value(scene, outcome)}."
            )

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

    lines.extend(("", "Verified ordered factual events during the executed code:"))
    temporal_fields: list[str] = []
    if not public_event_batches:
        lines.append(
            "- No tracked grasp-contact, aperture, contact, support, or access fact changed."
        )
    for batch_index, batch in enumerate(public_event_batches):
        prefix = "First" if batch_index == 0 else "Then"
        rendered_changes: list[str] = []
        for key, old, new in batch.changes:
            if key not in spec_by_key or spec_by_key[key].family not in _TEMPORAL_FAMILIES:
                raise ValueError(f"Takeover event contains an untracked field: {key}")
            rendered_changes.append(
                f"{_public_label(scene, spec_by_key[key])} changed from "
                f"{_public_value(scene, old)} to {_public_value(scene, new)}"
            )
            temporal_fields.append(key)
        lines.append(f"- {prefix}, in one simulator step: {'; '.join(rendered_changes)}.")
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


def initialize_trial_trace(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    task: str,
) -> None:
    if task not in {"cube_stack", "cube_restack"}:
        raise ValueError(f"Closed-GT takeover has no audited adapter for {task!r}")
    initial = closed_gt.extract_cube_snapshot(env)
    root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    root.mkdir(parents=True, exist_ok=False)
    key = _cache_key(output_dir, trial)
    if key in _PRIVATE_SNAPSHOTS:
        raise RuntimeError("Closed-GT takeover private trial cache already exists")
    _PRIVATE_SNAPSHOTS[key] = [initial]
    observer_installed = _install_event_recorder(env, key=key, initial=initial)
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
        raise ValueError(f"Closed-GT passive capture has no adapter for {task!r}")
    initial = closed_gt.extract_cube_snapshot(env)
    root = Path(output_dir) / PASSIVE_TRACE_DIR / f"trial_{trial:02d}"
    root.mkdir(parents=True, exist_ok=False)
    key = _cache_key(output_dir, trial)
    if key in _PRIVATE_SNAPSHOTS:
        raise RuntimeError("Closed-GT passive private trial cache already exists")
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
        raise ValueError("Closed-GT passive state is not turn-synchronous")
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
        raise ValueError("Closed-GT passive trace provenance mismatch")
    scene = closed_gt.cube_scene_spec()
    before = snapshots[-1]
    after = closed_gt.extract_cube_snapshot(env)
    event_batches = _consume_event_batches(env, key=key, endpoint=after)
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
        {"changes": [list(change) for change in batch.changes]}
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
) -> list[dict[str, Any]]:
    """Replace exactly the M3 report carrier with deterministic closed GT."""
    if not isinstance(native_report, str) or not native_report.strip():
        raise ValueError("Closed-GT takeover requires a native M3 report")
    key = _cache_key(output_dir, trial)
    snapshots = _PRIVATE_SNAPSHOTS.get(key)
    if snapshots is None or len(snapshots) != turn:
        raise ValueError("Closed-GT takeover private state is not turn-synchronous")
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
        raise ValueError("Closed-GT takeover trace provenance mismatch")

    scene = closed_gt.cube_scene_spec()
    before = snapshots[-1]
    after = closed_gt.extract_cube_snapshot(env)
    event_batches = _consume_event_batches(env, key=key, endpoint=after)
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
        {"changes": [list(change) for change in batch.changes]}
        for batch in event_batches
    ]
    turn_semantics = {
        "public_transition": dict(transition.values),
        "public_interval_grasp_outcomes": public_interval_outcomes,
        "public_interval_placement_outcomes": public_placement_outcomes,
        "public_temporal_event_batches": public_events,
    }
    rendered = render_takeover_report(
        scene, before, after, transition, event_batches
    )
    corrected = _replace_report_once(
        native_decision_prompt,
        original=native_report,
        replacement=rendered.text,
    )
    if _non_text_structure(native_decision_prompt) != _non_text_structure(corrected):
        raise RuntimeError("Closed-GT takeover changed request roles or media")
    if _sha256_json(native_decision_prompt) == _sha256_json(corrected):
        raise RuntimeError("Closed-GT takeover did not change the action request")

    goal_audit = closed_gt.goal_iff_audit(
        after,
        closed_gt.cube_goal_definition(task),
        private_task_completed=private_task_completed,
    )
    if goal_audit["iff_holds"] is not True:
        raise RuntimeError("Closed-GT benchmark-goal IFF audit failed")

    turn_root = root / f"turn_{turn:02d}"
    turn_root.mkdir(parents=True, exist_ok=False)
    (turn_root / "native_report.txt").write_text(native_report, encoding="utf-8")
    (turn_root / "takeover_report.txt").write_text(rendered.text, encoding="utf-8")
    record = {
        "schema_version": SCHEMA_VERSION,
        "closed_gt_schema_version": closed_gt.SCHEMA_VERSION,
        "task": task,
        "trial": trial,
        "turn": turn,
        "carrier": "m3_visual_difference_report_whole_takeover",
        "native_report_sha256": _sha256_text(native_report),
        "takeover_report_sha256": _sha256_text(rendered.text),
        "native_prompt_sha256": _sha256_json(native_decision_prompt),
        "takeover_prompt_sha256": _sha256_json(corrected),
        "non_text_structure_identical": True,
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
        "temporal_event_field_count": len(rendered.temporal_event_fields),
        "temporal_event_fields": list(rendered.temporal_event_fields),
        "public_temporal_event_batches": public_events,
        "audited_recorded_temporal_event_batches": audited_events,
        "audited_recorded_temporal_event_batches_sha256": _sha256_json(
            audited_events
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
    }
    _save_json(turn_root / "takeover.json", record)
    snapshots.append(after)
    state["turns_completed"] = turn
    state["latest_public_state_sha256"] = _sha256_json(after.values)
    state["latest_public_transition_sha256"] = _sha256_json(transition.values)
    _save_json(state_path, state)
    return corrected
