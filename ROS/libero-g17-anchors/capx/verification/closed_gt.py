"""Closed, deterministic semantic GT for manipulation recovery reports.

The simulator adapter may inspect continuous privileged state.  This module
projects that state into a finite qualitative vocabulary and validates all
derived facts against their primitive dependencies.  Numeric state remains
private and is never part of the public predicate mapping.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np

from capx.verification.core_fg import (
    _horizontal_goal_projection,
    _rotation_matrix_wxyz,
)

SCHEMA_VERSION = "capx-closed-semantic-gt-v2"

TRUE_FALSE = ("FALSE", "TRUE")
AXIS_VALUES = (
    "NEGATIVE_OF_REFERENCE_BAND",
    "WITHIN_REFERENCE_BAND",
    "POSITIVE_OF_REFERENCE_BAND",
)
INSIDE_VALUES = ("OUTSIDE", "INSIDE")
SUPPORT_REGION_VALUES = ("OUTSIDE", "EDGE_REGION", "CENTER_REGION")
HEIGHT_VALUES = ("AT_OR_BELOW_LIFT_BAND", "ABOVE_LIFT_BAND")
WORKSPACE_HEIGHT_VALUES = (
    "AT_OR_BELOW_WORKSPACE_GUARD_BAND",
    "ABOVE_WORKSPACE_GUARD_BAND",
)
APERTURE_VALUES = ("CLOSED", "PARTIALLY_OPEN", "OPEN")
DYNAMIC_VALUES = ("SETTLED", "MOVING")
ORIENTATION_VALUES = (
    "LOCAL_Z_ALIGNED_WITH_TABLE_NORMAL",
    "LOCAL_Z_OBLIQUE_TO_TABLE_NORMAL",
    "LOCAL_Z_OPPOSED_TO_TABLE_NORMAL",
)
ACCESS_VALUES = ("BLOCKED_FROM_ABOVE", "CLEAR_FROM_ABOVE")
SUPPORT_STABILITY_VALUES = (
    "CENTERED_OBJECT_SUPPORT",
    "EDGE_OBJECT_SUPPORT",
    "TABLE_SUPPORT",
    "EXECUTOR_SUPPORT",
    "MULTIPLE_SUPPORTS",
    "NO_STABLE_SUPPORT",
)
GRASP_EVIDENCE_VALUES = (
    "NOT_HELD",
    "CONTACT_ONLY_NOT_LIFT_CONFIRMED",
    "LIFT_CONFIRMED",
)
GRASP_CONTACT_STATE_VALUES = (
    "NO_EXECUTOR_CONTACT",
    "CONTACT_WITHOUT_BILATERAL_GRASP",
    "BILATERAL_GRASP_TABLE_SUPPORTED",
    "BILATERAL_GRASP_OBJECT_SUPPORTED",
    "BILATERAL_GRASP_BELOW_LIFT_BAND",
    "BILATERAL_GRASP_LIFT_CONFIRMED",
)
LOAD_BEARING_VALUES = (
    "NOT_SUPPORTING_AN_OBJECT",
    "SUPPORTING_ONE_OBJECT",
    "SUPPORTING_MULTIPLE_OBJECTS",
)
CHANGE_VALUES = ("UNCHANGED", "CHANGED")
MOTION_VALUES = ("NO_MEANINGFUL_CHANGE", "MOVED")
HOLDING_OUTCOME_VALUES = (
    "NOT_HELD_UNCHANGED",
    "ACQUIRED",
    "MAINTAINED",
    "DETACHED_TO_STABLE_SUPPORT",
    "DETACHED_WITHOUT_STABLE_SUPPORT",
)


@dataclass(frozen=True)
class EntitySpec:
    """One sanitized semantic entity, never a simulator object ID."""

    entity_id: str
    public_name: str
    kind: Literal["executor", "object", "surface"]


@dataclass(frozen=True)
class SceneSpec:
    """Entity registry and private discretization constants for one domain."""

    executors: tuple[EntitySpec, ...]
    objects: tuple[EntitySpec, ...]
    table: EntitySpec
    lift_band: float = 0.04
    workspace_guard_height: float = 0.04
    settled_linear_speed: float = 0.01
    settled_angular_speed: float = math.radians(10.0)
    meaningful_translation: float = 0.005
    meaningful_rotation: float = math.radians(5.0)
    closed_aperture_max: float = 0.1
    open_aperture_min: float = 0.9
    centered_support_fraction: float = 0.5
    orientation_alignment_cosine: float = math.cos(math.radians(20.0))

    def __post_init__(self) -> None:
        entities = (*self.executors, *self.objects, self.table)
        ids = [entity.entity_id for entity in entities]
        if len(ids) != len(set(ids)):
            raise ValueError("Closed-GT entity IDs must be unique")
        if not self.executors or not self.objects:
            raise ValueError("Closed-GT requires at least one executor and object")
        if any(not entity.entity_id or not entity.public_name for entity in entities):
            raise ValueError("Closed-GT entities require non-empty IDs and names")
        if any(entity.kind != "executor" for entity in self.executors):
            raise ValueError("Closed-GT executor registry contains a non-executor")
        if any(entity.kind != "object" for entity in self.objects):
            raise ValueError("Closed-GT object registry contains a non-object")
        if self.table.kind != "surface":
            raise ValueError("Closed-GT table entity must be a surface")
        positive = (
            self.lift_band,
            self.workspace_guard_height,
            self.settled_linear_speed,
            self.settled_angular_speed,
            self.meaningful_translation,
            self.meaningful_rotation,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Closed-GT discretization constants must be positive")
        if not (
            0.0 <= self.closed_aperture_max < self.open_aperture_min <= 1.0
        ):
            raise ValueError("Closed-GT aperture bands are invalid")
        if not 0.0 < self.centered_support_fraction < 1.0:
            raise ValueError("Closed-GT centered support fraction must be in (0, 1)")
        if not 0.0 < self.orientation_alignment_cosine < 1.0:
            raise ValueError("Closed-GT orientation cosine must be in (0, 1)")


@dataclass(frozen=True)
class BodyState:
    """Private continuous pose and motion in one declared semantic frame."""

    position: np.ndarray
    orientation_wxyz: np.ndarray
    half_extents: np.ndarray | None
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray


@dataclass(frozen=True)
class RawSceneState:
    """Private simulator snapshot consumed by the deterministic compiler."""

    executors: dict[str, BodyState]
    objects: dict[str, BodyState]
    executor_aperture: dict[str, float]
    holding: frozenset[tuple[str, str]]
    executor_contacts: frozenset[tuple[str, str]]
    object_contacts: frozenset[frozenset[str]]
    table_contacts: frozenset[str]
    table_height: float


@dataclass(frozen=True)
class PredicateSpec:
    key: str
    family: str
    values: tuple[str, ...]
    layer: Literal["primitive", "derived"]
    arguments: tuple[str, ...]
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledSnapshot:
    schema_version: str
    values: dict[str, str]
    private_poses: dict[str, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class CompiledTransition:
    schema_version: str
    values: dict[str, str]


@dataclass(frozen=True)
class GoalAtom:
    predicate: str
    value: str


@dataclass(frozen=True)
class GoalDefinition:
    name: str
    atoms: tuple[GoalAtom, ...]
    forbidden_conjunctions: tuple[tuple[GoalAtom, ...], ...] = ()


REGEN_FACTUAL_REQUIREMENTS: dict[str, frozenset[str]] = {
    "executor_configuration": frozenset(
        {"executor_aperture", "executor_orientation_to_table"}
    ),
    "executor_motion": frozenset({"executor_dynamic_state"}),
    "executor_object_topology": frozenset(
        {
            "executor_object_relative_x",
            "executor_object_relative_y",
            "executor_object_relative_z",
            "executor_object_contact",
        }
    ),
    "grasp_and_payload": frozenset(
        {
            "executor_object_holding",
            "executor_object_grasp_evidence",
            "executor_object_grasp_contact_state",
            "executor_payload",
        }
    ),
    "object_configuration": frozenset(
        {
            "object_dynamic_state",
            "object_orientation_to_table",
            "object_table_height",
            "object_table_contact",
            "object_workspace_height",
        }
    ),
    "object_object_topology": frozenset(
        {
            "object_object_relative_x",
            "object_object_relative_y",
            "object_object_relative_z",
            "object_object_footprint",
            "object_object_support_region",
            "object_object_volume",
            "object_object_contact",
            "object_object_support",
        }
    ),
    "access_and_stability": frozenset(
        {
            "object_top_access",
            "object_load_bearing",
            "object_support_stability",
            "object_semantic_state",
        }
    ),
}


def cube_scene_spec() -> SceneSpec:
    """Return the sanitized entity registry shared by stack and restack."""
    return SceneSpec(
        executors=(EntitySpec("exe01", "robot gripper", "executor"),),
        objects=(
            EntitySpec("obj01", "red cube", "object"),
            EntitySpec("obj02", "green cube", "object"),
        ),
        table=EntitySpec("surface.table", "table", "surface"),
    )


def _payload_values(scene: SceneSpec) -> tuple[str, ...]:
    return ("EMPTY", *(obj.entity_id.upper() for obj in scene.objects), "MULTIPLE")


def _semantic_values(scene: SceneSpec, subject_id: str) -> tuple[str, ...]:
    return (
        "HELD_NOT_LIFTED",
        "HELD_LIFTED",
        "ON_TABLE",
        *(
            f"SUPPORTED_BY_{obj.entity_id.upper()}"
            for obj in scene.objects
            if obj.entity_id != subject_id
        ),
        "SUPPORTED_BY_MULTIPLE_OBJECTS",
        "AIRBORNE_UNHELD",
        "UNSUPPORTED_OR_UNSETTLED",
    )


def predicate_specs(scene: SceneSpec) -> tuple[PredicateSpec, ...]:
    """Build a total fixed-order vocabulary from the entity registry."""
    specs: list[PredicateSpec] = []
    executor_ids = [entity.entity_id for entity in scene.executors]
    object_ids = [entity.entity_id for entity in scene.objects]

    for executor_id in executor_ids:
        specs.extend(
            (
                PredicateSpec(
                    f"executor.{executor_id}.aperture",
                    "executor_aperture",
                    APERTURE_VALUES,
                    "primitive",
                    (executor_id,),
                ),
                PredicateSpec(
                    f"executor.{executor_id}.dynamic_state",
                    "executor_dynamic_state",
                    DYNAMIC_VALUES,
                    "primitive",
                    (executor_id,),
                ),
                PredicateSpec(
                    f"executor.{executor_id}.orientation_to_table",
                    "executor_orientation_to_table",
                    ORIENTATION_VALUES,
                    "primitive",
                    (executor_id, scene.table.entity_id),
                ),
            )
        )
    for executor_id in executor_ids:
        for object_id in object_ids:
            prefix = f"executor_object.{executor_id}.{object_id}"
            specs.append(
                PredicateSpec(
                    f"{prefix}.holding",
                    "executor_object_holding",
                    TRUE_FALSE,
                    "primitive",
                    (executor_id, object_id),
                )
            )
            specs.append(
                PredicateSpec(
                    f"{prefix}.contact",
                    "executor_object_contact",
                    TRUE_FALSE,
                    "primitive",
                    (executor_id, object_id),
                )
            )
            for axis in ("x", "y", "z"):
                specs.append(
                    PredicateSpec(
                        f"{prefix}.relative_{axis}",
                        f"executor_object_relative_{axis}",
                        AXIS_VALUES,
                        "primitive",
                        (executor_id, object_id),
                    )
                )

    for subject_id in object_ids:
        for reference_id in object_ids:
            if subject_id == reference_id:
                continue
            prefix = f"object_object.{subject_id}.{reference_id}"
            for axis in ("x", "y", "z"):
                specs.append(
                    PredicateSpec(
                        f"{prefix}.relative_{axis}",
                        f"object_object_relative_{axis}",
                        AXIS_VALUES,
                        "primitive",
                        (subject_id, reference_id),
                    )
                )
            specs.extend(
                (
                    PredicateSpec(
                        f"{prefix}.support_region",
                        "object_object_support_region",
                        SUPPORT_REGION_VALUES,
                        "primitive",
                        (subject_id, reference_id),
                    ),
                    PredicateSpec(
                        f"{prefix}.center_in_reference_volume",
                        "object_object_volume",
                        INSIDE_VALUES,
                        "primitive",
                        (subject_id, reference_id),
                    ),
                )
            )

    for first_id, second_id in combinations(object_ids, 2):
        specs.append(
            PredicateSpec(
                f"object_object.{first_id}.{second_id}.contact",
                "object_object_contact",
                TRUE_FALSE,
                "primitive",
                (first_id, second_id),
            )
        )

    for object_id in object_ids:
        specs.extend(
            (
                PredicateSpec(
                    f"object.{object_id}.table_contact",
                    "object_table_contact",
                    TRUE_FALSE,
                    "primitive",
                    (object_id, scene.table.entity_id),
                ),
                PredicateSpec(
                    f"object.{object_id}.height_above_table",
                    "object_table_height",
                    HEIGHT_VALUES,
                    "primitive",
                    (object_id, scene.table.entity_id),
                ),
                PredicateSpec(
                    f"object.{object_id}.height_in_workspace_frame",
                    "object_workspace_height",
                    WORKSPACE_HEIGHT_VALUES,
                    "primitive",
                    (object_id,),
                ),
                PredicateSpec(
                        f"object.{object_id}.dynamic_state",
                        "object_dynamic_state",
                    DYNAMIC_VALUES,
                    "primitive",
                        (object_id,),
                    ),
                    PredicateSpec(
                        f"object.{object_id}.orientation_to_table",
                        "object_orientation_to_table",
                        ORIENTATION_VALUES,
                        "primitive",
                        (object_id, scene.table.entity_id),
                    ),
            )
        )

    for subject_id in object_ids:
        for reference_id in object_ids:
            if subject_id == reference_id:
                continue
            prefix = f"object_object.{subject_id}.{reference_id}"
            specs.extend(
                (
                    PredicateSpec(
                        f"{prefix}.center_in_reference_footprint",
                        "object_object_footprint",
                        INSIDE_VALUES,
                        "derived",
                        (subject_id, reference_id),
                        (f"{prefix}.support_region",),
                    ),
                    PredicateSpec(
                        f"{prefix}.supported_by_reference",
                        "object_object_support",
                        TRUE_FALSE,
                        "derived",
                        (subject_id, reference_id),
                        (
                            f"{prefix}.relative_z",
                            f"{prefix}.center_in_reference_footprint",
                            f"object_object.{min(subject_id, reference_id)}.{max(subject_id, reference_id)}.contact",
                        ),
                    ),
                )
            )

    for executor_id in executor_ids:
        holding_dependencies = tuple(
            f"executor_object.{executor_id}.{object_id}.holding"
            for object_id in object_ids
        )
        specs.append(
            PredicateSpec(
                f"executor.{executor_id}.payload",
                "executor_payload",
                _payload_values(scene),
                "derived",
                (executor_id,),
                holding_dependencies,
            )
        )
        for object_id in object_ids:
            specs.append(
                PredicateSpec(
                    f"executor_object.{executor_id}.{object_id}.grasp_evidence",
                    "executor_object_grasp_evidence",
                    GRASP_EVIDENCE_VALUES,
                    "derived",
                    (executor_id, object_id),
                    (
                        f"executor_object.{executor_id}.{object_id}.holding",
                        f"executor_object.{executor_id}.{object_id}.contact",
                        f"object.{object_id}.table_contact",
                        f"object.{object_id}.height_above_table",
                    ),
                )
            )
            specs.append(
                PredicateSpec(
                    f"executor_object.{executor_id}.{object_id}.grasp_contact_state",
                    "executor_object_grasp_contact_state",
                    GRASP_CONTACT_STATE_VALUES,
                    "derived",
                    (executor_id, object_id),
                    (
                        f"executor_object.{executor_id}.{object_id}.holding",
                        f"executor_object.{executor_id}.{object_id}.contact",
                        f"object.{object_id}.table_contact",
                        f"object.{object_id}.height_above_table",
                        *(
                            f"object_object.{object_id}.{other_id}.supported_by_reference"
                            for other_id in object_ids
                            if other_id != object_id
                        ),
                    ),
                )
            )

    for object_id in object_ids:
        relation_dependencies = tuple(
            key
            for other_id in object_ids
            if other_id != object_id
            for key in (
                f"object_object.{object_id}.{other_id}.supported_by_reference",
                f"object_object.{object_id}.{other_id}.support_region",
                f"object_object.{other_id}.{object_id}.relative_z",
                f"object_object.{other_id}.{object_id}.center_in_reference_footprint",
            )
        )
        holding_dependencies = tuple(
            f"executor_object.{executor_id}.{object_id}.holding"
            for executor_id in executor_ids
        )
        base_dependencies = (
            *holding_dependencies,
            f"object.{object_id}.table_contact",
            f"object.{object_id}.height_above_table",
            *relation_dependencies,
        )
        specs.extend(
            (
                PredicateSpec(
                    f"object.{object_id}.semantic_state",
                    "object_semantic_state",
                    _semantic_values(scene, object_id),
                    "derived",
                    (object_id,),
                    base_dependencies,
                ),
                PredicateSpec(
                    f"object.{object_id}.top_access",
                    "object_top_access",
                    ACCESS_VALUES,
                    "derived",
                    (object_id,),
                    relation_dependencies,
                ),
                PredicateSpec(
                    f"object.{object_id}.load_bearing",
                    "object_load_bearing",
                    LOAD_BEARING_VALUES,
                    "derived",
                    (object_id,),
                    tuple(
                        f"object_object.{other_id}.{object_id}.supported_by_reference"
                        for other_id in object_ids
                        if other_id != object_id
                    ),
                ),
                PredicateSpec(
                    f"object.{object_id}.support_stability",
                    "object_support_stability",
                    SUPPORT_STABILITY_VALUES,
                    "derived",
                    (object_id,),
                    base_dependencies,
                ),
            )
        )

    keys = [spec.key for spec in specs]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise RuntimeError(f"Closed-GT predicate schema has duplicates: {duplicates}")
    key_set = set(keys)
    key_index = {key: index for index, key in enumerate(keys)}
    for spec in specs:
        missing = set(spec.dependencies) - key_set
        if missing:
            raise RuntimeError(f"Closed-GT predicate has missing dependencies: {spec.key}")
        if spec.layer == "primitive" and spec.dependencies:
            raise RuntimeError(f"Closed-GT primitive declares dependencies: {spec.key}")
        if spec.layer == "derived" and not spec.dependencies:
            raise RuntimeError(f"Closed-GT derived predicate lacks dependencies: {spec.key}")
        if any(key_index[dependency] >= key_index[spec.key] for dependency in spec.dependencies):
            raise RuntimeError(
                f"Closed-GT dependency does not precede derived field: {spec.key}"
            )

    dependencies = {spec.key: spec.dependencies for spec in specs}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> None:
        if key in visiting:
            raise RuntimeError(f"Closed-GT dependency cycle contains: {key}")
        if key in visited:
            return
        visiting.add(key)
        for dependency in dependencies[key]:
            visit(dependency)
        visiting.remove(key)
        visited.add(key)

    for key in keys:
        visit(key)
    return tuple(specs)


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values with shape {shape}")
    return result


def _validate_body(body: BodyState, *, require_extents: bool, name: str) -> None:
    _array(body.position, (3,), f"{name}.position")
    quaternion = _array(body.orientation_wxyz, (4,), f"{name}.orientation")
    if np.linalg.norm(quaternion) <= np.finfo(np.float64).eps:
        raise ValueError(f"{name}.orientation has zero norm")
    _array(body.linear_velocity, (3,), f"{name}.linear_velocity")
    _array(body.angular_velocity, (3,), f"{name}.angular_velocity")
    if require_extents or body.half_extents is not None:
        extents = _array(body.half_extents, (3,), f"{name}.half_extents")
        if np.any(extents <= 0):
            raise ValueError(f"{name}.half_extents must be positive")


def validate_raw_scene(scene: SceneSpec, raw: RawSceneState) -> None:
    executor_ids = {entity.entity_id for entity in scene.executors}
    object_ids = {entity.entity_id for entity in scene.objects}
    if set(raw.executors) != executor_ids or set(raw.executor_aperture) != executor_ids:
        raise ValueError("Closed-GT raw executor registry mismatch")
    if set(raw.objects) != object_ids:
        raise ValueError("Closed-GT raw object registry mismatch")
    for entity_id, body in raw.executors.items():
        _validate_body(body, require_extents=False, name=entity_id)
    for entity_id, body in raw.objects.items():
        _validate_body(body, require_extents=True, name=entity_id)
    for executor_id, aperture in raw.executor_aperture.items():
        if not np.isfinite(aperture) or not 0.0 <= aperture <= 1.0:
            raise ValueError(f"Invalid aperture fraction for {executor_id}")
    valid_executor_object_pairs = {
        (executor_id, object_id)
        for executor_id in executor_ids
        for object_id in object_ids
    }
    if not raw.holding.issubset(valid_executor_object_pairs):
        raise ValueError("Closed-GT holding relation references an unknown entity")
    if not raw.executor_contacts.issubset(valid_executor_object_pairs):
        raise ValueError("Closed-GT executor contact references an unknown entity")
    if not raw.holding.issubset(raw.executor_contacts):
        raise ValueError("Closed-GT holding requires executor-object contact")
    valid_pairs = {frozenset(pair) for pair in combinations(object_ids, 2)}
    if any(pair not in valid_pairs for pair in raw.object_contacts):
        raise ValueError("Closed-GT contact relation is not a registered object pair")
    if not raw.table_contacts.issubset(object_ids):
        raise ValueError("Closed-GT table contact references an unknown object")
    if not np.isfinite(raw.table_height):
        raise ValueError("Closed-GT table height must be finite")


def _axis_value(delta: float, half_band: float) -> str:
    if delta < -half_band:
        return "NEGATIVE_OF_REFERENCE_BAND"
    if delta > half_band:
        return "POSITIVE_OF_REFERENCE_BAND"
    return "WITHIN_REFERENCE_BAND"


def _projected_half_extents(body: BodyState) -> np.ndarray:
    rotation = _rotation_matrix_wxyz(body.orientation_wxyz)
    return np.abs(rotation) @ _array(body.half_extents, (3,), "half_extents")


def _inside_volume(subject: BodyState, reference: BodyState) -> bool:
    rotation = _rotation_matrix_wxyz(reference.orientation_wxyz)
    local = rotation.T @ (subject.position - reference.position)
    half_extents = _array(reference.half_extents, (3,), "reference half_extents")
    return bool(np.all(np.abs(local) <= half_extents))


def _support_region(
    subject: BodyState,
    reference: BodyState,
    centered_fraction: float,
) -> str:
    rotation = _rotation_matrix_wxyz(reference.orientation_wxyz)
    delta_xy = subject.position[:2] - reference.position[:2]
    half_extents = _array(reference.half_extents, (3,), "reference half_extents")
    _, _, inside = _horizontal_goal_projection(
        rotation,
        half_extents,
        delta_xy,
    )
    if not inside:
        return "OUTSIDE"
    _, _, centered = _horizontal_goal_projection(
        rotation,
        half_extents * centered_fraction,
        delta_xy,
    )
    return "CENTER_REGION" if centered else "EDGE_REGION"


def _orientation_to_table(body: BodyState, alignment_cosine: float) -> str:
    local_z_in_frame = _rotation_matrix_wxyz(body.orientation_wxyz)[:, 2]
    vertical_cosine = float(local_z_in_frame[2])
    if vertical_cosine >= alignment_cosine:
        return "LOCAL_Z_ALIGNED_WITH_TABLE_NORMAL"
    if vertical_cosine <= -alignment_cosine:
        return "LOCAL_Z_OPPOSED_TO_TABLE_NORMAL"
    return "LOCAL_Z_OBLIQUE_TO_TABLE_NORMAL"


def _dynamic_state(body: BodyState, scene: SceneSpec) -> str:
    moving = (
        np.linalg.norm(body.linear_velocity) > scene.settled_linear_speed
        or np.linalg.norm(body.angular_velocity) > scene.settled_angular_speed
    )
    return "MOVING" if moving else "SETTLED"


def _contact_key(first: str, second: str) -> str:
    low, high = sorted((first, second))
    return f"object_object.{low}.{high}.contact"


def _held_by(values: dict[str, str], scene: SceneSpec, object_id: str) -> list[str]:
    return [
        executor.entity_id
        for executor in scene.executors
        if values[f"executor_object.{executor.entity_id}.{object_id}.holding"] == "TRUE"
    ]


def _supporters(values: dict[str, str], scene: SceneSpec, object_id: str) -> list[str]:
    return [
        other.entity_id
        for other in scene.objects
        if other.entity_id != object_id
        and values[
            f"object_object.{object_id}.{other.entity_id}.supported_by_reference"
        ]
        == "TRUE"
    ]


def _derive_payload(values: dict[str, str], scene: SceneSpec, executor_id: str) -> str:
    held = [
        obj.entity_id
        for obj in scene.objects
        if values[f"executor_object.{executor_id}.{obj.entity_id}.holding"] == "TRUE"
    ]
    if not held:
        return "EMPTY"
    if len(held) > 1:
        return "MULTIPLE"
    return held[0].upper()


def _derive_grasp_evidence(
    values: dict[str, str], executor_id: str, object_id: str
) -> str:
    if values[f"executor_object.{executor_id}.{object_id}.holding"] == "FALSE":
        return "NOT_HELD"
    if values[f"executor_object.{executor_id}.{object_id}.contact"] != "TRUE":
        raise ValueError("Closed-GT holding lacks executor-object contact")
    lifted = values[f"object.{object_id}.height_above_table"] == "ABOVE_LIFT_BAND"
    on_table = values[f"object.{object_id}.table_contact"] == "TRUE"
    return "LIFT_CONFIRMED" if lifted and not on_table else "CONTACT_ONLY_NOT_LIFT_CONFIRMED"


def _derive_grasp_contact_state(
    values: dict[str, str],
    scene: SceneSpec,
    executor_id: str,
    object_id: str,
) -> str:
    prefix = f"executor_object.{executor_id}.{object_id}"
    contact = values[f"{prefix}.contact"] == "TRUE"
    bilateral = values[f"{prefix}.holding"] == "TRUE"
    if not contact:
        if bilateral:
            raise ValueError("Closed-GT bilateral grasp lacks executor contact")
        return "NO_EXECUTOR_CONTACT"
    if not bilateral:
        return "CONTACT_WITHOUT_BILATERAL_GRASP"
    if values[f"object.{object_id}.table_contact"] == "TRUE":
        return "BILATERAL_GRASP_TABLE_SUPPORTED"
    if _supporters(values, scene, object_id):
        return "BILATERAL_GRASP_OBJECT_SUPPORTED"
    if values[f"object.{object_id}.height_above_table"] == "ABOVE_LIFT_BAND":
        return "BILATERAL_GRASP_LIFT_CONFIRMED"
    return "BILATERAL_GRASP_BELOW_LIFT_BAND"


def _derive_semantic_state(
    values: dict[str, str], scene: SceneSpec, object_id: str
) -> str:
    held = _held_by(values, scene, object_id)
    on_table = values[f"object.{object_id}.table_contact"] == "TRUE"
    lifted = values[f"object.{object_id}.height_above_table"] == "ABOVE_LIFT_BAND"
    supporters = _supporters(values, scene, object_id)
    if held:
        return "HELD_LIFTED" if lifted and not on_table else "HELD_NOT_LIFTED"
    if on_table:
        return "ON_TABLE"
    if len(supporters) == 1:
        return f"SUPPORTED_BY_{supporters[0].upper()}"
    if len(supporters) > 1:
        return "SUPPORTED_BY_MULTIPLE_OBJECTS"
    if lifted:
        return "AIRBORNE_UNHELD"
    return "UNSUPPORTED_OR_UNSETTLED"


def _derive_top_access(values: dict[str, str], scene: SceneSpec, object_id: str) -> str:
    blocked = any(
        values[
            f"object_object.{other.entity_id}.{object_id}.center_in_reference_footprint"
        ]
        == "INSIDE"
        and values[f"object_object.{other.entity_id}.{object_id}.relative_z"]
        == "POSITIVE_OF_REFERENCE_BAND"
        for other in scene.objects
        if other.entity_id != object_id
    )
    return "BLOCKED_FROM_ABOVE" if blocked else "CLEAR_FROM_ABOVE"


def _derive_load_bearing(
    values: dict[str, str], scene: SceneSpec, object_id: str
) -> str:
    dependent_count = sum(
        values[
            f"object_object.{other.entity_id}.{object_id}.supported_by_reference"
        ]
        == "TRUE"
        for other in scene.objects
        if other.entity_id != object_id
    )
    if dependent_count == 0:
        return "NOT_SUPPORTING_AN_OBJECT"
    if dependent_count == 1:
        return "SUPPORTING_ONE_OBJECT"
    return "SUPPORTING_MULTIPLE_OBJECTS"


def _derive_support_stability(
    values: dict[str, str], scene: SceneSpec, object_id: str
) -> str:
    held = _held_by(values, scene, object_id)
    on_table = values[f"object.{object_id}.table_contact"] == "TRUE"
    supporters = _supporters(values, scene, object_id)
    support_count = len(held) + int(on_table) + len(supporters)
    if support_count > 1:
        return "MULTIPLE_SUPPORTS"
    if held:
        return "EXECUTOR_SUPPORT"
    if on_table:
        return "TABLE_SUPPORT"
    if not supporters:
        return "NO_STABLE_SUPPORT"
    supporter = supporters[0]
    region = values[f"object_object.{object_id}.{supporter}.support_region"]
    if region == "CENTER_REGION":
        return "CENTERED_OBJECT_SUPPORT"
    if region == "EDGE_REGION":
        return "EDGE_OBJECT_SUPPORT"
    raise ValueError("Closed-GT supporter conflicts with support region")


def compile_snapshot(scene: SceneSpec, raw: RawSceneState) -> CompiledSnapshot:
    """Compile one private continuous state into the complete semantic quotient."""
    validate_raw_scene(scene, raw)
    values: dict[str, str] = {}

    for executor in scene.executors:
        aperture = raw.executor_aperture[executor.entity_id]
        values[f"executor.{executor.entity_id}.aperture"] = (
            "CLOSED"
            if aperture <= scene.closed_aperture_max
            else "OPEN"
            if aperture >= scene.open_aperture_min
            else "PARTIALLY_OPEN"
        )
        executor_body = raw.executors[executor.entity_id]
        values[f"executor.{executor.entity_id}.dynamic_state"] = _dynamic_state(
            executor_body, scene
        )
        values[f"executor.{executor.entity_id}.orientation_to_table"] = (
            _orientation_to_table(executor_body, scene.orientation_alignment_cosine)
        )
        for obj in scene.objects:
            prefix = f"executor_object.{executor.entity_id}.{obj.entity_id}"
            values[f"{prefix}.holding"] = (
                "TRUE" if (executor.entity_id, obj.entity_id) in raw.holding else "FALSE"
            )
            values[f"{prefix}.contact"] = (
                "TRUE"
                if (executor.entity_id, obj.entity_id) in raw.executor_contacts
                else "FALSE"
            )
            object_body = raw.objects[obj.entity_id]
            delta = executor_body.position - object_body.position
            bands = _projected_half_extents(object_body)
            for index, axis in enumerate(("x", "y", "z")):
                values[f"{prefix}.relative_{axis}"] = _axis_value(
                    float(delta[index]), float(bands[index])
                )

    for subject in scene.objects:
        for reference in scene.objects:
            if subject.entity_id == reference.entity_id:
                continue
            prefix = f"object_object.{subject.entity_id}.{reference.entity_id}"
            subject_body = raw.objects[subject.entity_id]
            reference_body = raw.objects[reference.entity_id]
            delta = subject_body.position - reference_body.position
            bands = _projected_half_extents(reference_body)
            for index, axis in enumerate(("x", "y", "z")):
                values[f"{prefix}.relative_{axis}"] = _axis_value(
                    float(delta[index]), float(bands[index])
                )
            region = _support_region(
                subject_body,
                reference_body,
                scene.centered_support_fraction,
            )
            values[f"{prefix}.support_region"] = region
            values[f"{prefix}.center_in_reference_footprint"] = (
                "OUTSIDE" if region == "OUTSIDE" else "INSIDE"
            )
            values[f"{prefix}.center_in_reference_volume"] = (
                "INSIDE" if _inside_volume(subject_body, reference_body) else "OUTSIDE"
            )

    for first, second in combinations(scene.objects, 2):
        pair = frozenset((first.entity_id, second.entity_id))
        values[_contact_key(first.entity_id, second.entity_id)] = (
            "TRUE" if pair in raw.object_contacts else "FALSE"
        )

    for subject in scene.objects:
        for reference in scene.objects:
            if subject.entity_id == reference.entity_id:
                continue
            prefix = f"object_object.{subject.entity_id}.{reference.entity_id}"
            supported = (
                values[_contact_key(subject.entity_id, reference.entity_id)] == "TRUE"
                and values[f"{prefix}.center_in_reference_footprint"] == "INSIDE"
                and values[f"{prefix}.relative_z"] == "POSITIVE_OF_REFERENCE_BAND"
            )
            values[f"{prefix}.supported_by_reference"] = "TRUE" if supported else "FALSE"

    for obj in scene.objects:
        body = raw.objects[obj.entity_id]
        values[f"object.{obj.entity_id}.table_contact"] = (
            "TRUE" if obj.entity_id in raw.table_contacts else "FALSE"
        )
        values[f"object.{obj.entity_id}.height_above_table"] = (
            "ABOVE_LIFT_BAND"
            if body.position[2] > raw.table_height + scene.lift_band
            else "AT_OR_BELOW_LIFT_BAND"
        )
        values[f"object.{obj.entity_id}.height_in_workspace_frame"] = (
            "ABOVE_WORKSPACE_GUARD_BAND"
            if body.position[2] > scene.workspace_guard_height
            else "AT_OR_BELOW_WORKSPACE_GUARD_BAND"
        )
        values[f"object.{obj.entity_id}.dynamic_state"] = _dynamic_state(body, scene)
        values[f"object.{obj.entity_id}.orientation_to_table"] = _orientation_to_table(
            body, scene.orientation_alignment_cosine
        )

    for executor in scene.executors:
        values[f"executor.{executor.entity_id}.payload"] = _derive_payload(
            values, scene, executor.entity_id
        )
        for obj in scene.objects:
            values[
                f"executor_object.{executor.entity_id}.{obj.entity_id}.grasp_evidence"
            ] = _derive_grasp_evidence(values, executor.entity_id, obj.entity_id)
            values[
                f"executor_object.{executor.entity_id}.{obj.entity_id}.grasp_contact_state"
            ] = _derive_grasp_contact_state(
                values, scene, executor.entity_id, obj.entity_id
            )

    for obj in scene.objects:
        values[f"object.{obj.entity_id}.semantic_state"] = _derive_semantic_state(
            values, scene, obj.entity_id
        )
        values[f"object.{obj.entity_id}.top_access"] = _derive_top_access(
            values, scene, obj.entity_id
        )
        values[f"object.{obj.entity_id}.load_bearing"] = _derive_load_bearing(
            values, scene, obj.entity_id
        )
        values[f"object.{obj.entity_id}.support_stability"] = _derive_support_stability(
            values, scene, obj.entity_id
        )

    private_poses = {
        **{
            entity_id: (body.position.copy(), body.orientation_wxyz.copy())
            for entity_id, body in raw.executors.items()
        },
        **{
            entity_id: (body.position.copy(), body.orientation_wxyz.copy())
            for entity_id, body in raw.objects.items()
        },
    }
    ordered_values = {spec.key: values[spec.key] for spec in predicate_specs(scene)}
    snapshot = CompiledSnapshot(SCHEMA_VERSION, ordered_values, private_poses)
    validate_snapshot(scene, snapshot)
    return snapshot


def validate_snapshot(
    scene: SceneSpec,
    snapshot: CompiledSnapshot,
) -> None:
    """Fail on missing, duplicate, invalid, or internally conflicting facts."""
    if snapshot.schema_version != SCHEMA_VERSION:
        raise ValueError("Closed-GT snapshot schema mismatch")
    specs = predicate_specs(scene)
    expected = [spec.key for spec in specs]
    if list(snapshot.values) != expected:
        missing = sorted(set(expected) - set(snapshot.values))
        extra = sorted(set(snapshot.values) - set(expected))
        raise ValueError(f"Closed-GT snapshot coverage mismatch missing={missing} extra={extra}")
    spec_by_key = {spec.key: spec for spec in specs}
    for key, value in snapshot.values.items():
        if value not in spec_by_key[key].values:
            raise ValueError(f"Closed-GT value outside closed domain: {key}={value}")
    expected_pose_ids = {
        entity.entity_id for entity in (*scene.executors, *scene.objects)
    }
    if set(snapshot.private_poses) != expected_pose_ids:
        raise ValueError("Closed-GT private pose registry mismatch")
    for entity_id, pose in snapshot.private_poses.items():
        if not isinstance(pose, tuple) or len(pose) != 2:
            raise ValueError(f"Closed-GT private pose is malformed: {entity_id}")
        _array(pose[0], (3,), f"{entity_id}.private_position")
        quaternion = _array(pose[1], (4,), f"{entity_id}.private_orientation")
        if np.linalg.norm(quaternion) <= np.finfo(np.float64).eps:
            raise ValueError(f"Closed-GT private orientation has zero norm: {entity_id}")

    values = snapshot.values
    for first, second in combinations(scene.objects, 2):
        forward = values[
            f"object_object.{first.entity_id}.{second.entity_id}.supported_by_reference"
        ]
        reverse = values[
            f"object_object.{second.entity_id}.{first.entity_id}.supported_by_reference"
        ]
        if forward == reverse == "TRUE":
            raise ValueError("Closed-GT objects cannot support each other simultaneously")
    for subject in scene.objects:
        for reference in scene.objects:
            if subject.entity_id == reference.entity_id:
                continue
            prefix = f"object_object.{subject.entity_id}.{reference.entity_id}"
            region = values[f"{prefix}.support_region"]
            expected_footprint = "OUTSIDE" if region == "OUTSIDE" else "INSIDE"
            if values[f"{prefix}.center_in_reference_footprint"] != expected_footprint:
                raise ValueError("Closed-GT footprint conflicts with support region")
            if expected_footprint == "INSIDE" and (
                values[f"{prefix}.relative_x"] != "WITHIN_REFERENCE_BAND"
                or values[f"{prefix}.relative_y"] != "WITHIN_REFERENCE_BAND"
            ):
                raise ValueError("Closed-GT footprint and axis predicates conflict")
            expected_support = (
                values[_contact_key(subject.entity_id, reference.entity_id)] == "TRUE"
                and values[f"{prefix}.center_in_reference_footprint"] == "INSIDE"
                and values[f"{prefix}.relative_z"] == "POSITIVE_OF_REFERENCE_BAND"
            )
            if values[f"{prefix}.supported_by_reference"] != (
                "TRUE" if expected_support else "FALSE"
            ):
                raise ValueError("Closed-GT support predicate conflicts with its definition")

    for executor in scene.executors:
        if values[f"executor.{executor.entity_id}.payload"] != _derive_payload(
            values, scene, executor.entity_id
        ):
            raise ValueError("Closed-GT executor payload conflicts with holding facts")
        for obj in scene.objects:
            holding_key = f"executor_object.{executor.entity_id}.{obj.entity_id}.holding"
            contact_key = f"executor_object.{executor.entity_id}.{obj.entity_id}.contact"
            if values[holding_key] == "TRUE" and values[contact_key] != "TRUE":
                raise ValueError("Closed-GT holding conflicts with executor-object contact")
            key = f"executor_object.{executor.entity_id}.{obj.entity_id}.grasp_evidence"
            if values[key] != _derive_grasp_evidence(
                values, executor.entity_id, obj.entity_id
            ):
                raise ValueError("Closed-GT grasp evidence conflicts with primitive facts")
            key = (
                f"executor_object.{executor.entity_id}.{obj.entity_id}.grasp_contact_state"
            )
            if values[key] != _derive_grasp_contact_state(
                values, scene, executor.entity_id, obj.entity_id
            ):
                raise ValueError(
                    "Closed-GT grasp-contact state conflicts with primitive facts"
                )
    for obj in scene.objects:
        if values[f"object.{obj.entity_id}.semantic_state"] != _derive_semantic_state(
            values, scene, obj.entity_id
        ):
            raise ValueError("Closed-GT semantic state conflicts with primitive facts")
        if values[f"object.{obj.entity_id}.top_access"] != _derive_top_access(
            values, scene, obj.entity_id
        ):
            raise ValueError("Closed-GT top access conflicts with primitive facts")
        if values[f"object.{obj.entity_id}.load_bearing"] != _derive_load_bearing(
            values, scene, obj.entity_id
        ):
            raise ValueError("Closed-GT load-bearing state conflicts with support facts")
        if values[f"object.{obj.entity_id}.support_stability"] != _derive_support_stability(
            values, scene, obj.entity_id
        ):
            raise ValueError("Closed-GT support stability conflicts with primitive facts")


def _quaternion_angle(first: np.ndarray, second: np.ndarray) -> float:
    first_q = _array(first, (4,), "first quaternion")
    second_q = _array(second, (4,), "second quaternion")
    first_q = first_q / np.linalg.norm(first_q)
    second_q = second_q / np.linalg.norm(second_q)
    dot = float(np.clip(abs(np.dot(first_q, second_q)), -1.0, 1.0))
    return 2.0 * math.acos(dot)


def _pose_motion(
    scene: SceneSpec,
    before: CompiledSnapshot,
    after: CompiledSnapshot,
    entity_id: str,
) -> str:
    before_position, before_orientation = before.private_poses[entity_id]
    after_position, after_orientation = after.private_poses[entity_id]
    moved = (
        np.linalg.norm(after_position - before_position) > scene.meaningful_translation
        or _quaternion_angle(before_orientation, after_orientation)
        > scene.meaningful_rotation
    )
    return "MOVED" if moved else "NO_MEANINGFUL_CHANGE"


def _holding_outcome(
    before: CompiledSnapshot,
    after: CompiledSnapshot,
    executor_id: str,
    object_id: str,
) -> str:
    holding_key = f"executor_object.{executor_id}.{object_id}.holding"
    was_held = before.values[holding_key] == "TRUE"
    is_held = after.values[holding_key] == "TRUE"
    if not was_held and not is_held:
        return "NOT_HELD_UNCHANGED"
    if not was_held and is_held:
        return "ACQUIRED"
    if was_held and is_held:
        return "MAINTAINED"
    stable = after.values[f"object.{object_id}.support_stability"] in {
        "CENTERED_OBJECT_SUPPORT",
        "EDGE_OBJECT_SUPPORT",
        "TABLE_SUPPORT",
        "MULTIPLE_SUPPORTS",
    }
    return "DETACHED_TO_STABLE_SUPPORT" if stable else "DETACHED_WITHOUT_STABLE_SUPPORT"


def compile_transition(
    scene: SceneSpec,
    before: CompiledSnapshot,
    after: CompiledSnapshot,
) -> CompiledTransition:
    """Describe a complete transition over the closed semantic state."""
    validate_snapshot(scene, before)
    validate_snapshot(scene, after)
    values: dict[str, str] = {}
    for key in before.values:
        values[f"transition.{key}"] = (
            "CHANGED" if before.values[key] != after.values[key] else "UNCHANGED"
        )

    for entity in (*scene.executors, *scene.objects):
        values[f"transition.{entity.kind}.{entity.entity_id}.pose"] = _pose_motion(
            scene, before, after, entity.entity_id
        )

    for executor in scene.executors:
        for obj in scene.objects:
            values[
                f"transition.executor_object.{executor.entity_id}.{obj.entity_id}.holding_outcome"
            ] = _holding_outcome(before, after, executor.entity_id, obj.entity_id)

    any_changed = any(
        value == "CHANGED"
        for key, value in values.items()
        if key.startswith("transition.") and key.count(".") > 1
    ) or any(value == "MOVED" for value in values.values())
    values["transition.any_relevant_state"] = "CHANGED" if any_changed else "UNCHANGED"
    transition = CompiledTransition(SCHEMA_VERSION, values)
    validate_transition(scene, before, after, transition)
    return transition


def validate_transition(
    scene: SceneSpec,
    before: CompiledSnapshot,
    after: CompiledSnapshot,
    transition: CompiledTransition,
) -> None:
    if transition.schema_version != SCHEMA_VERSION:
        raise ValueError("Closed-GT transition schema mismatch")
    expected_prefix = [f"transition.{key}" for key in before.values]
    pose_keys = [
        f"transition.{entity.kind}.{entity.entity_id}.pose"
        for entity in (*scene.executors, *scene.objects)
    ]
    outcome_keys = [
        f"transition.executor_object.{executor.entity_id}.{obj.entity_id}.holding_outcome"
        for executor in scene.executors
        for obj in scene.objects
    ]
    expected = [*expected_prefix, *pose_keys, *outcome_keys, "transition.any_relevant_state"]
    if list(transition.values) != expected:
        raise ValueError("Closed-GT transition coverage or order mismatch")
    for key in expected_prefix:
        state_key = key.removeprefix("transition.")
        expected_value = (
            "CHANGED" if before.values[state_key] != after.values[state_key] else "UNCHANGED"
        )
        if transition.values[key] != expected_value:
            raise ValueError(f"Closed-GT transition conflicts with snapshots: {key}")
    for entity, key in zip((*scene.executors, *scene.objects), pose_keys, strict=True):
        expected_value = _pose_motion(scene, before, after, entity.entity_id)
        if transition.values[key] != expected_value:
            raise ValueError(f"Closed-GT pose transition conflicts with poses: {key}")
    for executor in scene.executors:
        for obj in scene.objects:
            key = (
                f"transition.executor_object.{executor.entity_id}.{obj.entity_id}.holding_outcome"
            )
            expected_value = _holding_outcome(
                before, after, executor.entity_id, obj.entity_id
            )
            if transition.values[key] != expected_value:
                raise ValueError(f"Closed-GT holding outcome conflicts with snapshots: {key}")
    derived_any = any(transition.values[key] == "CHANGED" for key in expected_prefix) or any(
        transition.values[key] == "MOVED" for key in pose_keys
    )
    expected_any = "CHANGED" if derived_any else "UNCHANGED"
    if transition.values["transition.any_relevant_state"] != expected_any:
        raise ValueError("Closed-GT aggregate transition conflicts with atomic transitions")


def cube_goal_definition(task: str) -> GoalDefinition:
    """Re-express the unmodified benchmark success check as factual atoms."""
    if task not in {"cube_stack", "cube_restack"}:
        raise ValueError(f"Unsupported cube goal: {task}")
    atoms = [
        GoalAtom("executor_object.exe01.obj01.holding", "FALSE"),
        GoalAtom("object.obj01.height_above_table", "ABOVE_LIFT_BAND"),
        GoalAtom("object_object.obj01.obj02.contact", "TRUE"),
    ]
    forbidden: tuple[tuple[GoalAtom, ...], ...] = ()
    if task == "cube_restack":
        forbidden = (
            (
                GoalAtom(
                    "object.obj01.height_in_workspace_frame",
                    "ABOVE_WORKSPACE_GUARD_BAND",
                ),
                GoalAtom(
                    "object.obj02.height_in_workspace_frame",
                    "ABOVE_WORKSPACE_GUARD_BAND",
                ),
            ),
        )
    goal = GoalDefinition(task, tuple(atoms), forbidden)
    validate_goal(cube_scene_spec(), goal)
    return goal


def validate_goal(scene: SceneSpec, goal: GoalDefinition) -> None:
    """Reject unknown, duplicated, invalid, or internally impossible goal logic."""
    if not goal.name:
        raise ValueError("Closed-GT goal requires a name")
    spec_by_key = {spec.key: spec for spec in predicate_specs(scene)}

    def validate_conjunction(atoms: tuple[GoalAtom, ...], label: str) -> None:
        if not atoms:
            raise ValueError(f"Closed-GT {label} conjunction is empty")
        keys = [atom.predicate for atom in atoms]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Closed-GT {label} conjunction duplicates a predicate")
        for atom in atoms:
            spec = spec_by_key.get(atom.predicate)
            if spec is None:
                raise ValueError(f"Goal references an unknown predicate: {atom.predicate}")
            if atom.value not in spec.values:
                raise ValueError(
                    f"Goal value is outside the predicate domain: {atom.predicate}={atom.value}"
                )

    validate_conjunction(goal.atoms, "required")
    required = {(atom.predicate, atom.value) for atom in goal.atoms}
    normalized_forbidden: list[frozenset[tuple[str, str]]] = []
    for conjunction in goal.forbidden_conjunctions:
        validate_conjunction(conjunction, "forbidden")
        normalized = frozenset((atom.predicate, atom.value) for atom in conjunction)
        if normalized.issubset(required):
            raise ValueError("Closed-GT goal requires a forbidden conjunction")
        if normalized in normalized_forbidden:
            raise ValueError("Closed-GT goal duplicates a forbidden conjunction")
        normalized_forbidden.append(normalized)


def goal_satisfied(snapshot: CompiledSnapshot, goal: GoalDefinition) -> bool:
    """Evaluate required atoms and forbidden conjunctions over the closed state."""
    for atom in goal.atoms:
        if atom.predicate not in snapshot.values:
            raise ValueError(f"Goal references an unknown predicate: {atom.predicate}")
        if snapshot.values[atom.predicate] != atom.value:
            return False
    for conjunction in goal.forbidden_conjunctions:
        for atom in conjunction:
            if atom.predicate not in snapshot.values:
                raise ValueError(f"Goal references an unknown predicate: {atom.predicate}")
        if all(snapshot.values[atom.predicate] == atom.value for atom in conjunction):
            return False
    return True


def goal_iff_audit(
    snapshot: CompiledSnapshot,
    goal: GoalDefinition,
    *,
    private_task_completed: bool,
) -> dict[str, Any]:
    """Compare semantic goal truth to the private environment label for audit only."""
    semantic = goal_satisfied(snapshot, goal)
    return {
        "schema_version": SCHEMA_VERSION,
        "goal": goal.name,
        "semantic_goal_satisfied": semantic,
        "private_task_completed": bool(private_task_completed),
        "iff_holds": semantic == bool(private_task_completed),
        "action_visible": False,
    }


def semantic_coverage_manifest(scene: SceneSpec) -> dict[str, Any]:
    """Describe which native-report semantics the factual basis can replace."""
    specs = predicate_specs(scene)
    families = sorted({spec.family for spec in specs})
    requirement_audit = {
        requirement: sorted(required - set(families))
        for requirement, required in REGEN_FACTUAL_REQUIREMENTS.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "covered_physical_families": families,
        "regen_factual_requirements": {
            requirement: sorted(required)
            for requirement, required in REGEN_FACTUAL_REQUIREMENTS.items()
        },
        "missing_regen_factual_families": {
            requirement: missing
            for requirement, missing in requirement_audit.items()
            if missing
        },
        "covers_declared_regen_factual_semantics": not any(
            requirement_audit.values()
        ),
        "covered_transition_families": [
            "every_closed_state_predicate_change",
            "executor_pose_change",
            "object_pose_change",
            "holding_outcome",
            "aggregate_relevant_change",
        ],
        "explicitly_excluded_non_factual_or_non_state_families": [
            "action_advice",
            "next_move",
            "task_completion_label",
            "reward",
            "oracle_code",
            "continuous_action_parameters",
            "camera_visibility_or_corruption",
            "causal_failure_attribution",
            "unobserved_within_block_events",
        ],
    }


def schema_audit(scene: SceneSpec) -> dict[str, Any]:
    """Return a deterministic structural audit after all fail-closed checks pass."""
    specs = predicate_specs(scene)
    primitive = [spec for spec in specs if spec.layer == "primitive"]
    derived = [spec for spec in specs if spec.layer == "derived"]
    coverage = semantic_coverage_manifest(scene)
    return {
        "schema_version": SCHEMA_VERSION,
        "state_field_count": len(specs),
        "canonical_public_basis_field_count": len(primitive),
        "primitive_field_count": len(primitive),
        "derived_field_count": len(derived),
        "all_keys_unique": len({spec.key for spec in specs}) == len(specs),
        "all_domains_finite_and_nonempty": all(spec.values for spec in specs),
        "unknown_is_not_a_gt_value": all("UNKNOWN" not in spec.values for spec in specs),
        "all_derived_fields_have_declared_dependencies": all(
            spec.dependencies for spec in derived
        ),
        "all_primitive_fields_are_dependency_free": all(
            not spec.dependencies for spec in primitive
        ),
        "dependency_graph_acyclic": True,
        "dependencies_precede_derived_fields": True,
        "covers_declared_regen_factual_semantics": coverage[
            "covers_declared_regen_factual_semantics"
        ],
        "missing_regen_factual_families": coverage[
            "missing_regen_factual_families"
        ],
    }


def public_entity_registry(scene: SceneSpec) -> dict[str, dict[str, str]]:
    return {
        entity.entity_id: {"public_name": entity.public_name, "kind": entity.kind}
        for entity in (*scene.executors, *scene.objects, scene.table)
    }


def _normalized_quaternion(value: Any, name: str) -> np.ndarray:
    quaternion = _array(value, (4,), name)
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} has zero norm")
    return quaternion / norm


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.array(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=np.float64,
    )


def _world_pose_to_frame(
    position_world: np.ndarray,
    orientation_world: np.ndarray,
    frame_position_world: np.ndarray,
    frame_orientation_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_q = _normalized_quaternion(frame_orientation_world, "frame orientation")
    frame_rotation = _rotation_matrix_wxyz(frame_q)
    position = frame_rotation.T @ (position_world - frame_position_world)
    inverse_frame_q = frame_q * np.array((1.0, -1.0, -1.0, -1.0))
    orientation = _quaternion_multiply(inverse_frame_q, orientation_world)
    return position, _normalized_quaternion(orientation, "relative orientation")


def _body_velocity(sim_data: Any, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    linear_source = getattr(sim_data, "body_xvelp", None)
    angular_source = getattr(sim_data, "body_xvelr", None)
    if linear_source is not None and angular_source is not None:
        return (
            _array(linear_source[body_id], (3,), "body linear velocity"),
            _array(angular_source[body_id], (3,), "body angular velocity"),
        )
    cvel = getattr(sim_data, "cvel", None)
    if cvel is None:
        raise ValueError("Simulator exposes no body velocity for absolute GT")
    spatial = _array(cvel[body_id], (6,), "body spatial velocity")
    return spatial[3:].copy(), spatial[:3].copy()


def extract_cube_raw_scene(env: Any) -> RawSceneState:
    """Read cube-task simulator state into the generic private interface."""
    low_level = getattr(env, "low_level_env", None)
    simulator = getattr(low_level, "robosuite_env", None)
    required = (
        "cubeA",
        "cubeB",
        "cubeA_body_id",
        "cubeB_body_id",
        "robots",
        "sim",
        "table_offset",
    )
    if low_level is None or simulator is None or not all(
        hasattr(simulator, name) for name in required
    ):
        raise TypeError("Closed-GT cube adapter requires the CaP-X Robosuite cube environment")

    sim_data = simulator.sim.data
    base_pose = _array(low_level.base_link_wxyz_xyz, (7,), "base pose")
    frame_orientation = _normalized_quaternion(base_pose[:4], "base orientation")
    frame_position = base_pose[4:]
    object_bindings = {
        "obj01": (simulator.cubeA, simulator.cubeA_body_id),
        "obj02": (simulator.cubeB, simulator.cubeB_body_id),
    }
    objects: dict[str, BodyState] = {}
    for object_id, (model, body_id) in object_bindings.items():
        world_position = _array(sim_data.body_xpos[body_id], (3,), f"{object_id} position")
        world_orientation = _normalized_quaternion(
            sim_data.body_xquat[body_id], f"{object_id} orientation"
        )
        position, orientation = _world_pose_to_frame(
            world_position,
            world_orientation,
            frame_position,
            frame_orientation,
        )
        linear, angular = _body_velocity(sim_data, body_id)
        objects[object_id] = BodyState(
            position,
            orientation,
            _array(model.size, (3,), f"{object_id} half extents").copy(),
            linear,
            angular,
        )

    gripper_pose = np.concatenate(
        [
            _array(sim_data.body_xquat[low_level.gripper_link_idx], (4,), "gripper orientation"),
            _array(sim_data.body_xpos[low_level.gripper_link_idx], (3,), "gripper position"),
        ]
    )
    executor_position, executor_orientation = _world_pose_to_frame(
        gripper_pose[4:],
        _normalized_quaternion(gripper_pose[:4], "gripper orientation"),
        frame_position,
        frame_orientation,
    )
    executor_linear, executor_angular = _body_velocity(
        sim_data, low_level.gripper_link_idx
    )
    executor = BodyState(
        executor_position,
        executor_orientation,
        None,
        executor_linear,
        executor_angular,
    )

    robot = simulator.robots[0]
    gripper_indexes = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
    if not isinstance(gripper_indexes, dict) or not gripper_indexes:
        raise ValueError("Simulator exposes no gripper joint position indexes")
    index_groups = [tuple(indexes) for indexes in gripper_indexes.values() if indexes]
    if len(index_groups) != 1 or len(index_groups[0]) != 2:
        raise ValueError("Closed-GT cube adapter requires one two-finger gripper")
    gripper_qpos = _array(
        sim_data.qpos[np.asarray(index_groups[0], dtype=np.int64)],
        (2,),
        "gripper qpos",
    )
    metric_length = float(low_level.gripper_metric_length)
    if not np.isfinite(metric_length) or metric_length <= 0:
        raise ValueError("Invalid gripper metric length")
    aperture = float(np.clip(gripper_qpos[0] / metric_length, 0.0, 1.0))

    holding = frozenset(
        ("exe01", object_id)
        for object_id, (model, _) in object_bindings.items()
        if simulator._check_grasp(gripper=simulator.robots[0].gripper, object_geoms=model)
    )
    executor_contacts = frozenset(
        ("exe01", object_id)
        for object_id, (model, _) in object_bindings.items()
        if ("exe01", object_id) in holding
        or simulator.check_contact(simulator.robots[0].gripper, model)
    )
    contacts = frozenset(
        {frozenset(("obj01", "obj02"))}
        if simulator.check_contact(simulator.cubeA, simulator.cubeB)
        else set()
    )
    table_contacts = frozenset(
        object_id
        for object_id, (model, _) in object_bindings.items()
        if simulator.check_contact(model, "table_collision")
    )
    table_world = _array(simulator.table_offset, (3,), "table offset")
    world_to_frame = _rotation_matrix_wxyz(frame_orientation).T
    table_normal_frame = world_to_frame @ np.array((0.0, 0.0, 1.0))
    if not np.allclose(
        table_normal_frame,
        np.array((0.0, 0.0, 1.0)),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Closed-GT cube frame +Z is not the table normal")
    table_frame = world_to_frame @ (table_world - frame_position)
    return RawSceneState(
        executors={"exe01": executor},
        objects=objects,
        executor_aperture={"exe01": aperture},
        holding=holding,
        executor_contacts=executor_contacts,
        object_contacts=contacts,
        table_contacts=table_contacts,
        table_height=float(table_frame[2]),
    )


def extract_cube_snapshot(env: Any) -> CompiledSnapshot:
    return compile_snapshot(cube_scene_spec(), extract_cube_raw_scene(env))


def iter_public_values(
    scene: SceneSpec,
    snapshot: CompiledSnapshot,
    *,
    canonical_only: bool = False,
) -> Iterable[tuple[str, str]]:
    """Yield the complete closure, or its nonredundant primitive basis."""
    validate_snapshot(scene, snapshot)
    for spec in predicate_specs(scene):
        if not canonical_only or spec.layer == "primitive":
            yield spec.key, snapshot.values[spec.key]


def iter_public_transition_values(
    scene: SceneSpec,
    before: CompiledSnapshot,
    after: CompiledSnapshot,
    transition: CompiledTransition,
    *,
    canonical_only: bool = False,
) -> Iterable[tuple[str, str]]:
    """Yield the complete transition, or its nonredundant primitive projection."""
    validate_transition(scene, before, after, transition)
    if not canonical_only:
        yield from transition.values.items()
        return
    for spec in predicate_specs(scene):
        if spec.layer == "primitive":
            key = f"transition.{spec.key}"
            yield key, transition.values[key]
    for entity in (*scene.executors, *scene.objects):
        key = f"transition.{entity.kind}.{entity.entity_id}.pose"
        yield key, transition.values[key]
