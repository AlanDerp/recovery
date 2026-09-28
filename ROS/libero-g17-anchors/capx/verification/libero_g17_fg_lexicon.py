"""Eye-checkable FG lexicon for the LIBERO G17 closed set.

Machine tokens stay in closed_gt. This module only defines how those tokens
become student-facing English that a human can judge from camera video.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from capx.verification.closed_gt import SceneSpec, predicate_specs

EyeTag = Literal["visual", "visual-coarse", "instrumented", "instrumented+visual"]

_INSTANCE_SUFFIX = re.compile(r"_\d+$")
_SITE_SUFFIX = re.compile(r"_(contain_)?region$")


@dataclass(frozen=True)
class FgEyeEntry:
    family: str
    eye_test: str
    tag: EyeTag
    value_cnl: dict[str, str]
    label_template: str


def humanize_libero_name(raw: str) -> str:
    """Convert a LIBERO instance/site name into an eye-friendly public name."""
    name = str(raw).strip()
    if not name:
        raise ValueError("LIBERO public name cannot be empty")
    name = _SITE_SUFFIX.sub("", name)
    name = _INSTANCE_SUFFIX.sub("", name)
    name = name.replace("_", " ").strip().lower()
    if not name:
        raise ValueError(f"LIBERO name {raw!r} humanizes to empty")
    return name


# Axis CNL uses robot-base frame (+Z up, +Y robot-left), not camera pixels.
_AXIS_VALUE_CNL = {
    "x": {
        "NEGATIVE_OF_REFERENCE_BAND": "closer to the robot than {1}",
        "WITHIN_REFERENCE_BAND": "at the same depth as {1}",
        "POSITIVE_OF_REFERENCE_BAND": "farther from the robot than {1}",
    },
    "y": {
        "NEGATIVE_OF_REFERENCE_BAND": "to the robot's right of {1}",
        "WITHIN_REFERENCE_BAND": "laterally aligned with {1}",
        "POSITIVE_OF_REFERENCE_BAND": "to the robot's left of {1}",
    },
    "z": {
        "NEGATIVE_OF_REFERENCE_BAND": "below {1}",
        "WITHIN_REFERENCE_BAND": "level with {1}",
        "POSITIVE_OF_REFERENCE_BAND": "above {1}",
    },
}

_TRUE_FALSE = {"FALSE": "false", "TRUE": "true"}
_INSIDE = {"OUTSIDE": "outside", "INSIDE": "inside"}
_SUPPORT_REGION = {
    "OUTSIDE": "outside the support area of {1}",
    "EDGE_REGION": "over the edge of the support area of {1}",
    "CENTER_REGION": "over the center of the support area of {1}",
}
_HEIGHT = {
    "AT_OR_BELOW_LIFT_BAND": "at or below a clear lift height above the support surface",
    "ABOVE_LIFT_BAND": "clearly lifted above the support surface",
}
_WORKSPACE = {
    "AT_OR_BELOW_WORKSPACE_GUARD_BAND": (
        "at or below the robot workspace guard height (instrumented)"
    ),
    "ABOVE_WORKSPACE_GUARD_BAND": (
        "above the robot workspace guard height (instrumented)"
    ),
}
_APERTURE = {
    "CLOSED": "closed",
    "PARTIALLY_OPEN": "partially open",
    "OPEN": "open",
}
_DYNAMIC = {"SETTLED": "still", "MOVING": "moving"}
_ORIENT = {
    "LOCAL_Z_ALIGNED_WITH_TABLE_NORMAL": "upright relative to the table",
    "LOCAL_Z_OBLIQUE_TO_TABLE_NORMAL": "tilted relative to the table",
    "LOCAL_Z_OPPOSED_TO_TABLE_NORMAL": "upside-down relative to the table",
}
_ACCESS = {
    "BLOCKED_FROM_ABOVE": "blocked from above",
    "CLEAR_FROM_ABOVE": "clear from above",
}
_STABILITY = {
    "CENTERED_OBJECT_SUPPORT": "resting centered on another object",
    "EDGE_OBJECT_SUPPORT": "resting on the edge of another object",
    "TABLE_SUPPORT": "resting on the table or floor",
    "EXECUTOR_SUPPORT": "held by the gripper",
    "MULTIPLE_SUPPORTS": "supported by more than one contact",
    "NO_STABLE_SUPPORT": "without stable support",
}
_GRASP_EVIDENCE = {
    "NOT_HELD": "not held",
    "CONTACT_ONLY_NOT_LIFT_CONFIRMED": "touching the gripper but not clearly lifted",
    "LIFT_CONFIRMED": "held and clearly lifted",
}
_GRASP_CONTACT = {
    "NO_EXECUTOR_CONTACT": "no gripper contact",
    "CONTACT_WITHOUT_BILATERAL_GRASP": (
        "gripper contact without a firm two-finger grasp"
    ),
    "BILATERAL_GRASP_TABLE_SUPPORTED": (
        "firm grasp while still resting on the table or floor"
    ),
    "BILATERAL_GRASP_OBJECT_SUPPORTED": (
        "firm grasp while still resting on another object"
    ),
    "BILATERAL_GRASP_BELOW_LIFT_BAND": (
        "firm grasp without other support, but not clearly lifted"
    ),
    "BILATERAL_GRASP_LIFT_CONFIRMED": "firm grasp with clear lift",
}
_LOAD = {
    "NOT_SUPPORTING_AN_OBJECT": "not supporting another object",
    "SUPPORTING_ONE_OBJECT": "supporting one other object",
    "SUPPORTING_MULTIPLE_OBJECTS": "supporting multiple other objects",
}


def _axis_entry(axis: str, family: str) -> FgEyeEntry:
    return FgEyeEntry(
        family=family,
        eye_test=(
            f"From the scene video, judge whether the subject is "
            f"{'/'.join(_AXIS_VALUE_CNL[axis].values()).split('{')[0].strip()} "
            f"relative to the named reference (robot-base axes, not camera pixels)."
        ),
        tag="visual",
        value_cnl=dict(_AXIS_VALUE_CNL[axis]),
        label_template="{0} position relative to {1} along robot " + axis,
    )


FG_EYE_CATALOG: dict[str, FgEyeEntry] = {
    "executor_aperture": FgEyeEntry(
        "executor_aperture",
        "Look at the gripper fingers: closed, partly open, or open.",
        "visual",
        dict(_APERTURE),
        "{0} aperture",
    ),
    "executor_dynamic_state": FgEyeEntry(
        "executor_dynamic_state",
        "Look whether the gripper is obviously moving or still.",
        "visual-coarse",
        dict(_DYNAMIC),
        "{0} motion",
    ),
    "executor_orientation_to_table": FgEyeEntry(
        "executor_orientation_to_table",
        "Look whether the gripper looks upright, tilted, or upside-down vs the table.",
        "visual-coarse",
        dict(_ORIENT),
        "{0} orientation relative to the table",
    ),
    "executor_object_holding": FgEyeEntry(
        "executor_object_holding",
        "Look whether the named object rides with the gripper.",
        "visual",
        dict(_TRUE_FALSE),
        "{0} holding {1}",
    ),
    "executor_object_contact": FgEyeEntry(
        "executor_object_contact",
        "Look whether the gripper fingers touch the named object.",
        "visual",
        dict(_TRUE_FALSE),
        "contact between {0} and {1}",
    ),
    "executor_object_relative_x": _axis_entry("x", "executor_object_relative_x"),
    "executor_object_relative_y": _axis_entry("y", "executor_object_relative_y"),
    "executor_object_relative_z": _axis_entry("z", "executor_object_relative_z"),
    "object_object_relative_x": _axis_entry("x", "object_object_relative_x"),
    "object_object_relative_y": _axis_entry("y", "object_object_relative_y"),
    "object_object_relative_z": _axis_entry("z", "object_object_relative_z"),
    "object_object_support_region": FgEyeEntry(
        "object_object_support_region",
        "From above, is the subject center outside, on the rim, or over the center of the reference top?",
        "visual-coarse",
        dict(_SUPPORT_REGION),
        "{0} center over the support area of {1}",
    ),
    "object_object_volume": FgEyeEntry(
        "object_object_volume",
        "Look whether the subject center is inside the receptacle interior.",
        "visual",
        dict(_INSIDE),
        "{0} center inside the volume of {1}",
    ),
    "object_object_contact": FgEyeEntry(
        "object_object_contact",
        "Look whether the two named objects touch.",
        "visual",
        dict(_TRUE_FALSE),
        "contact between {0} and {1}",
    ),
    "object_table_contact": FgEyeEntry(
        "object_table_contact",
        "Look whether the object touches the table or floor.",
        "visual",
        dict(_TRUE_FALSE),
        "contact between {0} and {1}",
    ),
    "object_table_height": FgEyeEntry(
        "object_table_height",
        "Look whether the object is clearly lifted off the support surface.",
        "visual-coarse",
        dict(_HEIGHT),
        "{0} height above the support surface",
    ),
    "object_workspace_height": FgEyeEntry(
        "object_workspace_height",
        "Requires the robot workspace-guard threshold; not a pure eye judgment.",
        "instrumented",
        dict(_WORKSPACE),
        "{0} workspace-guard height",
    ),
    "object_dynamic_state": FgEyeEntry(
        "object_dynamic_state",
        "Look whether the object is obviously moving or still.",
        "visual-coarse",
        dict(_DYNAMIC),
        "{0} motion",
    ),
    "object_orientation_to_table": FgEyeEntry(
        "object_orientation_to_table",
        "Look whether the object looks upright, tilted, or upside-down vs the table.",
        "visual-coarse",
        dict(_ORIENT),
        "{0} orientation relative to the table",
    ),
    "object_object_footprint": FgEyeEntry(
        "object_object_footprint",
        "Look whether the subject center lies inside the reference top footprint.",
        "visual",
        dict(_INSIDE),
        "{0} center inside the footprint of {1}",
    ),
    "object_object_support": FgEyeEntry(
        "object_object_support",
        "Look whether the subject appears to rest on the reference; not equal to BDDL On.",
        "visual-coarse",
        dict(_TRUE_FALSE),
        "{0} supported by {1}",
    ),
    "executor_payload": FgEyeEntry(
        "executor_payload",
        "Look which bound object, if any, is carried by the gripper.",
        "visual",
        {
            "EMPTY": "empty",
            "OBJ01": "{obj01}",
            "OBJ02": "{obj02}",
            "MULTIPLE": "multiple objects",
        },
        "{0} payload",
    ),
    "executor_object_grasp_evidence": FgEyeEntry(
        "executor_object_grasp_evidence",
        "Look whether the object is not held, only touched, or held and lifted.",
        "visual",
        dict(_GRASP_EVIDENCE),
        "grasp evidence for {0} and {1}",
    ),
    "executor_object_grasp_contact_state": FgEyeEntry(
        "executor_object_grasp_contact_state",
        "Combine visible contact/hold/support/lift; firm two-finger grasp is instrumented.",
        "instrumented+visual",
        dict(_GRASP_CONTACT),
        "grasp-contact state for {0} and {1}",
    ),
    "object_semantic_state": FgEyeEntry(
        "object_semantic_state",
        "Look whether the object is held, on the table, on the other object, or airborne.",
        "visual",
        {
            "HELD_NOT_LIFTED": "held but not lifted",
            "HELD_LIFTED": "held and lifted",
            "ON_TABLE": "on the table or floor",
            "SUPPORTED_BY_OBJ01": "supported by {obj01}",
            "SUPPORTED_BY_OBJ02": "supported by {obj02}",
            "SUPPORTED_BY_MULTIPLE_OBJECTS": "supported by multiple objects",
            "AIRBORNE_UNHELD": "airborne and unheld",
            "UNSUPPORTED_OR_UNSETTLED": "unsupported or unsettled",
        },
        "{0} support and holding state",
    ),
    "object_top_access": FgEyeEntry(
        "object_top_access",
        "Look whether another object blocks grasping this one from above.",
        "visual",
        dict(_ACCESS),
        "top access to {0}",
    ),
    "object_load_bearing": FgEyeEntry(
        "object_load_bearing",
        "Look whether this object currently supports another bound object.",
        "visual",
        dict(_LOAD),
        "whether {0} supports other objects",
    ),
    "object_support_stability": FgEyeEntry(
        "object_support_stability",
        "Look what visibly stabilizes the object (table, other object, gripper, none).",
        "visual-coarse",
        dict(_STABILITY),
        "{0} support stability",
    ),
}


def assert_fg_eye_catalog_complete(scene: SceneSpec | None = None) -> None:
    """Fail closed if any closed-GT family lacks an eye-check FG entry."""
    if scene is None:
        from capx.verification.closed_gt import cube_scene_spec

        scene = cube_scene_spec()
    families = {spec.family for spec in predicate_specs(scene)}
    missing = sorted(families - set(FG_EYE_CATALOG))
    extra = sorted(set(FG_EYE_CATALOG) - families)
    if missing or extra:
        raise AssertionError(
            f"FG eye catalog mismatch; missing={missing}; extra={extra}"
        )
    for family, entry in FG_EYE_CATALOG.items():
        if not entry.eye_test.strip():
            raise AssertionError(f"empty eye_test for {family}")
        if not entry.value_cnl:
            raise AssertionError(f"empty value_cnl for {family}")
        if not entry.label_template.strip():
            raise AssertionError(f"empty label_template for {family}")


def _entity_names(scene: SceneSpec) -> dict[str, str]:
    return {
        entity.entity_id: entity.public_name
        for entity in (*scene.executors, *scene.objects, scene.table)
    }


def render_fg_label(scene: SceneSpec, family: str, arguments: tuple[str, ...]) -> str:
    entry = FG_EYE_CATALOG[family]
    names = _entity_names(scene)
    return entry.label_template.format(*(names[arg] for arg in arguments))


def render_fg_value(
    scene: SceneSpec,
    family: str,
    value: str,
    *,
    arguments: tuple[str, ...] = (),
) -> str:
    """Render one closed-GT enum as eye-friendly English."""
    entry = FG_EYE_CATALOG[family]
    names = _entity_names(scene)
    template = entry.value_cnl.get(value)
    if template is None:
        # Entity-valued payload / SUPPORTED_BY_* already listed; allow upper ID.
        token = value.upper()
        for entity_id, public_name in names.items():
            if token == entity_id.upper():
                return public_name
            if value == f"SUPPORTED_BY_{entity_id.upper()}":
                return f"supported by {public_name}"
        raise KeyError(f"No FG CNL for {family}={value}")
    format_args = {
        "obj01": names.get("obj01", "obj01"),
        "obj02": names.get("obj02", "obj02"),
        "0": names[arguments[0]] if arguments else "",
        "1": names[arguments[1]] if len(arguments) > 1 else names.get("obj02", ""),
    }
    # Support both {0}/{1} and {obj01}/{obj02} templates.
    try:
        return template.format(
            *(names[arg] for arg in arguments),
            **{k: v for k, v in format_args.items() if k.startswith("obj")},
        )
    except IndexError:
        return template.format_map(format_args)


def render_fg_clause(
    scene: SceneSpec,
    family: str,
    value: str,
    arguments: tuple[str, ...],
) -> str:
    """One student-facing FG sentence fragment: '<label>: <value>'."""
    label = render_fg_label(scene, family, arguments)
    rendered = render_fg_value(scene, family, value, arguments=arguments)
    return f"{label}: {rendered}"


def fg_allows_solo_projection(family: str) -> bool:
    """Instrumented-only families need a visual corroborator in FG packs."""
    tag = FG_EYE_CATALOG[family].tag
    return tag in {"visual", "visual-coarse", "instrumented+visual"}
