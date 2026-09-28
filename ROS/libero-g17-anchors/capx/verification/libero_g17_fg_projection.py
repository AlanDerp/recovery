"""LIBERO G17 FG projection (offline from the M3 baseline path).

Layer-1 Ideal-aligned world FG for LIBERO pick-place:

- ordered Δ over **goal-directed** eye-allowed fields;
- endpoint split into **acquisition** and **placement** evidence blocks;
- for On/In goals, a public BDDL-aligned IFF block
  (``libero_object_ontop`` / ``libero_object_in``; not G17 volume/support proxies);
- binary ``finish_authorized`` (True iff that public basis holds);
- evidence-gap notes only when finish is not authorized.

Does not patch ``trial.py``, enable Ideal-TK env flags, or write the M3 tree.

Layer-1 sparse mode (``CAPX_LIBERO_G17_L1_SPARSE=1``, schema v2.11):
emit only the active gap's evidence subset plus public IFF. On
``placement_xy``, suppress footprint / coarse RelXY (they conflict with the
on-top alignment band) and emit signed non-numeric offset directions.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from capx.verification.closed_gt import (
    CompiledSnapshot,
    CompiledTransition,
    SceneSpec,
    compile_transition,
    predicate_specs,
    validate_snapshot,
    validate_transition,
)
from capx.verification.libero_g17_adapter import (
    LiberoAttachmentFact,
    LiberoG17Binding,
    attachment_from_bound_snapshot,
)
from capx.verification.libero_g17_fg_lexicon import (
    FG_EYE_CATALOG,
    fg_allows_solo_projection,
    humanize_libero_name,
    render_fg_clause,
)
from capx.verification.libero_g17_public_iff import (
    LIBERO_OBJECT_ONTOP_XY_M,
    derive_libero_object_in_basis,
    derive_libero_object_ontop_basis,
    public_finish_authorized_for_binding,
    render_libero_in_iff_lines,
    render_libero_ontop_iff_lines,
)

GainStatus = Literal[
    "no_bound_release",
    "release_suppressed_no_public_iff",
]

EndpointGap = Literal[
    "acquisition_incomplete",
    "placement_incomplete",
]

TaskGap = Literal[
    "none",
    "acquisition",
    "placement_xy",
    "placement_other",
]

_FORBIDDEN_PUBLIC = (
    "reward",
    "success",
    "failed",
    "task completed",
    "bddl",
    "check_ontop",
    "check_contain",
    "obj01",
    "obj02",
    "exe01",
    "surface.table",
)
_NUMERIC_TOKEN = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])")
_SIM_ID = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)*_\d+\b")
_PAYLOAD_GRASP_KEY = "executor_object.exe01.obj01.grasp_contact_state"
_RECIPROCAL_PREFIX = "object_object.obj02.obj01."


def v24_pin_enabled() -> bool:
    """Pin fair-K v24 report contract (``v2.4-public-in-iff``).

    When set, suppress post-v2.4 report extensions (task/interaction status,
    same-name ordinal, placement surface∥center groups, placement_xy scrub).
    """
    return os.environ.get("CAPX_LIBERO_G17_V24_PIN", "0") == "1"


def acquire_evidence_keys(_binding: LiberoG17Binding) -> tuple[str, ...]:
    """Mutually exclusive-ish Acquire basis (gripper ↔ payload).

    Includes the closed-GT ME partitions ``grasp_evidence`` (3-way) and
    ``grasp_contact_state`` (6-way), plus observable holding/contact/payload
    and gripper–payload relative bands. Excludes place-pair geometry.
    """
    return (
        "executor.exe01.aperture",
        "executor.exe01.payload",
        "executor_object.exe01.obj01.holding",
        "executor_object.exe01.obj01.contact",
        "executor_object.exe01.obj01.grasp_evidence",
        "executor_object.exe01.obj01.grasp_contact_state",
        "executor_object.exe01.obj01.relative_x",
        "executor_object.exe01.obj01.relative_y",
        "executor_object.exe01.obj01.relative_z",
    )


def place_evidence_keys(binding: LiberoG17Binding) -> tuple[str, ...]:
    """Placement basis (payload → receptacle), disjoint from Acquire keys.

    Public On finish is IFF-only. Do **not** expose G17 ``support_region`` /
    ``supported_by_reference`` here: their CNL ("over the center…") conflicts
    with the on-top XY-band fact and is not the finish GT.

    Full key list is for invariants; student-facing endpoint may further
    suppress footprint / coarse RelXY under ``placement_xy`` (see
    ``student_place_evidence_keys``).

    Under ``CAPX_LIBERO_G17_V24_PIN=1``, restore the v2.4 place dump that
    includes ``support_region`` / ``supported_by_reference`` (fair-K winner).
    """
    prefix = "object_object.obj01.obj02"
    if v24_pin_enabled():
        keys = [
            f"{prefix}.relative_x",
            f"{prefix}.relative_y",
            f"{prefix}.relative_z",
            f"{prefix}.support_region",
            f"{prefix}.center_in_reference_footprint",
            f"{prefix}.contact",
            f"{prefix}.supported_by_reference",
            "object.obj01.table_contact",
            "object.obj01.height_above_table",
        ]
        if binding.relation == "in":
            keys.insert(5, f"{prefix}.center_in_reference_volume")
        return tuple(keys)
    keys = [
        f"{prefix}.relative_x",
        f"{prefix}.relative_y",
        f"{prefix}.relative_z",
        f"{prefix}.center_in_reference_footprint",
        f"{prefix}.contact",
        "object.obj01.table_contact",
        "object.obj01.height_above_table",
    ]
    if binding.relation == "in":
        keys.insert(4, f"{prefix}.center_in_reference_volume")
    return tuple(keys)


def placement_xy_suppress_keys(binding: LiberoG17Binding) -> frozenset[str]:
    """Keys that fake On / center-to-center under placement_xy (or always for On Δ).

    Coarse RelXY (plate half-extents), footprint CNL in the mixed place dump,
    and ``support_region`` ("over the center of the support area") must not
    sit beside the high-precision center-to-center group.
    """
    if binding.relation != "on":
        return frozenset()
    return frozenset(
        {
            "object_object.obj01.obj02.center_in_reference_footprint",
            "object_object.obj01.obj02.relative_x",
            "object_object.obj01.obj02.relative_y",
            "object_object.obj01.obj02.support_region",
            "object_object.obj02.obj01.center_in_reference_footprint",
            "object_object.obj02.obj01.relative_x",
            "object_object.obj02.obj01.relative_y",
            "object_object.obj02.obj01.support_region",
            "executor_object.exe01.obj02.relative_x",
            "executor_object.exe01.obj02.relative_y",
        }
    )


def goal_evidence_keys(binding: LiberoG17Binding) -> tuple[str, ...]:
    """Ordered union of Acquire then Place keys (disjoint by construction)."""
    acquire = acquire_evidence_keys(binding)
    place = place_evidence_keys(binding)
    overlap = set(acquire) & set(place)
    if overlap:
        raise RuntimeError(f"Acquire/Place evidence key overlap: {sorted(overlap)}")
    return acquire + place


def layer1_sparse_enabled() -> bool:
    """True when Layer-1 sparse endpoint + near-miss discrepancy mode is on."""
    return os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1"


def sparse_acquire_evidence_keys(
    binding: LiberoG17Binding, gap: EndpointGap
) -> tuple[str, ...]:
    """Gap-conditioned Acquire keys for student-facing endpoint projection."""
    full = acquire_evidence_keys(binding)
    if not layer1_sparse_enabled():
        return full
    if gap == "acquisition_incomplete":
        return full
    # Placement-side work: keep holding / contact / grasp ME only.
    keep = {
        "executor.exe01.payload",
        "executor_object.exe01.obj01.holding",
        "executor_object.exe01.obj01.contact",
        "executor_object.exe01.obj01.grasp_evidence",
    }
    return tuple(k for k in full if k in keep)


def sparse_place_evidence_keys(
    binding: LiberoG17Binding, gap: EndpointGap
) -> tuple[str, ...]:
    """Gap-conditioned Place keys for student-facing endpoint projection."""
    full = place_evidence_keys(binding)
    if not layer1_sparse_enabled():
        return full
    prefix = "object_object.obj01.obj02"
    if gap == "acquisition_incomplete":
        return (f"{prefix}.contact",)
    # Placement / near-miss: RelXYZ + footprint (+ volume) + contact.
    # Drop support_region / supported_by / table_* word dump (IFF is the finish basis).
    keep = {
        f"{prefix}.relative_x",
        f"{prefix}.relative_y",
        f"{prefix}.relative_z",
        f"{prefix}.center_in_reference_footprint",
        f"{prefix}.contact",
    }
    if binding.relation == "in":
        keep.add(f"{prefix}.center_in_reference_volume")
    return tuple(k for k in full if k in keep)


def student_place_evidence_keys(
    binding: LiberoG17Binding,
    after: CompiledSnapshot,
    gap: EndpointGap,
) -> tuple[str, ...]:
    """Student-facing Place keys; footprint lives in the surface group for On."""
    keys = sparse_place_evidence_keys(binding, gap)
    if v24_pin_enabled():
        # v2.4: full place dump including footprint + support_region.
        return keys
    if binding.relation != "on":
        return keys
    # Footprint CNL is replaced by the independent surface group.
    drop = {
        "object_object.obj01.obj02.center_in_reference_footprint",
        "object_object.obj02.obj01.center_in_reference_footprint",
    }
    if classify_task_gap(binding, after) == "placement_xy":
        drop |= set(placement_xy_suppress_keys(binding))
    return tuple(k for k in keys if k not in drop)


def release_candidate_keys(binding: LiberoG17Binding) -> tuple[str, ...]:
    """G17 release-bound fact keys for the bound payload/receptacle pair."""
    subject = "obj01"
    reference = "obj02"
    prefix = f"object_object.{subject}.{reference}"
    keys = [
        f"{prefix}.relative_x",
        f"{prefix}.relative_y",
        f"{prefix}.relative_z",
        f"{prefix}.support_region",
        f"{prefix}.center_in_reference_footprint",
        f"{prefix}.supported_by_reference",
        f"object.{reference}.orientation_to_table",
        f"object.{reference}.support_stability",
    ]
    if binding.relation == "in":
        keys.insert(5, f"{prefix}.center_in_reference_volume")
        keys.insert(6, f"{prefix}.contact")
    return tuple(keys)


@dataclass(frozen=True)
class ProjectionSemanticGain:
    status: GainStatus
    novel_fact_ids: tuple[str, ...]
    candidate_facts: dict[str, str]
    parent_visible_pairs: tuple[tuple[str, str], ...]
    bound_release: bool


@dataclass(frozen=True)
class LiberoFgProjection:
    parent_report: str
    endpoint_report: str
    release_report: str
    combined_report: str
    gain: ProjectionSemanticGain
    parent_sha256: str
    combined_sha256: str
    changed_field_count: int
    public_field_count: int
    goal_evidence_keys: tuple[str, ...]
    world_fg_complete: bool
    endpoint_gap: EndpointGap
    finish_authorized: bool
    executor_attachment: str
    interaction_grasp: str = "none"
    interaction_grasp_peak: str = "none"
    executor_attachment_peak: str = "empty"
    task_gap: TaskGap = "acquisition"
    released_goal_after_transport: bool = False


@dataclass(frozen=True)
class InteractionSlot:
    """Minimal executor-verifiable interaction bar (game-UI style).

    ``grasp`` is ``none`` when nothing is bilaterally held; otherwise a
    public humanized object label (never a sim instance id).
    ``role`` is an explicit referent token for dual-domination vs VDM.
    """

    grasp: str
    role: str  # none | goal_object | receptacle | held_non_goal_movable
    held_raw: tuple[str, ...] = ()
    disambiguator: str = ""


def _axis_band_cnl(axis: str, band: str, reference_label: str) -> str:
    """Render one Rel axis band without numerics."""
    entry = FG_EYE_CATALOG[f"executor_object_relative_{axis}"]
    template = entry.value_cnl[band]
    return template.replace("{1}", reference_label)


def _invert_rel_band(band: str) -> str:
    """Flip subject↔reference polarity (NEGATIVE ↔ POSITIVE)."""
    if band == "NEGATIVE_OF_REFERENCE_BAND":
        return "POSITIVE_OF_REFERENCE_BAND"
    if band == "POSITIVE_OF_REFERENCE_BAND":
        return "NEGATIVE_OF_REFERENCE_BAND"
    return band


def _xy_anchor_axis(bands: tuple[str, str, str]) -> tuple[str, str] | None:
    """Chess-style single-axis anchor: lateral (y) then depth (x).

    Returns ``(axis, held_vs_goal_band)`` when one horizontal axis separates
    the instances; ``None`` when XY-coaligned.
    """
    rx, ry, _rz = bands
    if ry != "WITHIN_REFERENCE_BAND":
        return ("y", ry)
    if rx != "WITHIN_REFERENCE_BAND":
        return ("x", rx)
    return None


def _public_names_collide(raw_a: str, raw_b: str) -> bool:
    return humanize_libero_name(raw_a) == humanize_libero_name(raw_b)


def _held_vs_goal_disambiguator(
    bands: tuple[str, str, str] | None,
    *,
    held_label: str,
    goal_label: str,
) -> str:
    """One eye-checkable spatial contrast between held non-goal and goal."""
    del held_label
    if bands is None:
        return f"not the goal object ({goal_label})"
    anchor = _xy_anchor_axis(bands)
    if anchor is not None:
        axis, band = anchor
        return (
            f"not the goal object; grasped instance is "
            f"{_axis_band_cnl(axis, band, f'goal instance ({goal_label})')}"
        )
    _rx, _ry, rz = bands
    if rz != "WITHIN_REFERENCE_BAND":
        return (
            f"not the goal object; grasped instance is "
            f"{_axis_band_cnl('z', rz, f'goal instance ({goal_label})')}"
        )
    return f"not the goal object ({goal_label}; distinct movable)"


def interaction_slot_from_attachment(
    binding: LiberoG17Binding, attachment: LiberoAttachmentFact
) -> InteractionSlot:
    """Project attachment ME into the public interaction slot."""
    if attachment.kind == "empty" or not attachment.held_movable_names:
        return InteractionSlot(grasp="none", role="none", held_raw=())
    held = tuple(str(x) for x in attachment.held_movable_names)
    primary = held[0]
    payload = binding.payload_name
    receptacle = binding.receptacle_body_name
    payload_pub = humanize_libero_name(payload)
    if attachment.kind == "holding_payload" or (
        len(held) == 1 and primary == payload
    ):
        return InteractionSlot(
            grasp=payload_pub,
            role="goal_object",
            held_raw=held,
        )
    if attachment.kind == "holding_receptacle" or (
        len(held) == 1 and primary == receptacle
    ):
        return InteractionSlot(
            grasp=humanize_libero_name(receptacle),
            role="receptacle",
            held_raw=held,
        )
    # Distractor / mixed: role token + spatial disambiguator when available.
    pub = humanize_libero_name(primary)
    disamb = _held_vs_goal_disambiguator(
        attachment.held_vs_goal_rel_xyz,
        held_label=pub,
        goal_label=payload_pub,
    )
    if pub == payload_pub:
        grasp = f"{pub} [role: held_non_goal_movable; {disamb}]"
    else:
        grasp = f"{pub} [role: held_non_goal_movable; {disamb}]"
    return InteractionSlot(
        grasp=grasp,
        role="held_non_goal_movable",
        held_raw=held,
        disambiguator=disamb,
    )


def released_goal_after_transport(
    *,
    peak: LiberoAttachmentFact,
    after_attachment: LiberoAttachmentFact,
    after: CompiledSnapshot,
) -> bool:
    """GT: interval held goal, endpoint empty, goal–receptacle contact true."""
    return (
        peak.kind == "holding_payload"
        and after_attachment.kind == "empty"
        and after.values.get("object_object.obj01.obj02.contact") == "TRUE"
    )


def classify_task_gap(
    binding: LiberoG17Binding, after: CompiledSnapshot
) -> TaskGap:
    """Task-slot gap from finish IFF GT (not soft narrative).

    Aligned with ``classify_endpoint_gap``: once the bound payload is acquired
    (or any placement-side partial holds), the public task gap stays on the
    placement side — never ``acquisition``.
    """
    if public_finish_authorized(binding, after):
        return "none"
    acquired = _payload_acquired(after)
    if binding.relation == "on":
        basis = derive_libero_object_ontop_basis(after)
        if (
            basis.contact
            and basis.z_order == "REFERENCE_BELOW_OR_LEVEL"
            and basis.xy_band != "WITHIN_ONTOP_XY_BAND"
        ):
            # v2.4 had no placement_xy task-gap token in student reports.
            return "placement_other" if v24_pin_enabled() else "placement_xy"
        if basis.contact or _placement_partial(binding, after) or acquired:
            return "placement_other"
        return "acquisition"
    if binding.relation == "in":
        if _placement_partial(binding, after) or acquired:
            return "placement_other"
        return "acquisition"
    if acquired or _placement_partial(binding, after):
        return "placement_other"
    return "acquisition"


def assert_task_gap_consistency(
    *,
    after: CompiledSnapshot,
    task_gap: TaskGap,
    endpoint_gap: EndpointGap,
    report: str,
    attachment: LiberoAttachmentFact | None = None,
) -> None:
    """Fail closed when public task gap contradicts acquire / endpoint gap."""
    acquired = _payload_acquired(after)
    holding_goal = (
        attachment is not None and attachment.kind == "holding_payload"
    )
    lowered = report.lower()
    holding_outcome = "holding_goal_object" in lowered
    if acquired or holding_goal or holding_outcome:
        if task_gap == "acquisition":
            raise ValueError(
                "task_gap acquisition conflicts with acquired/holding goal payload"
            )
        if "task gap: acquisition" in lowered:
            raise ValueError(
                "public report claims task gap acquisition while holding goal"
            )
    if endpoint_gap == "placement_incomplete" and task_gap == "acquisition":
        raise ValueError(
            "endpoint_gap placement_incomplete conflicts with task_gap acquisition"
        )


def render_task_interaction_status(
    binding: LiberoG17Binding,
    slot: InteractionSlot,
    *,
    peak: InteractionSlot | None = None,
    peak_attachment: LiberoAttachmentFact | None = None,
    after_attachment: LiberoAttachmentFact | None = None,
    after: CompiledSnapshot | None = None,
    task_gap: TaskGap | None = None,
) -> list[str]:
    """Task + interaction status bars as GT-backed facts only."""
    payload = humanize_libero_name(binding.payload_name)
    receptacle = humanize_libero_name(binding.receptacle_eval_name)
    lines = [
        "Task status:",
        f"- goal object: {payload} [role: goal_object].",
        f"- receptacle: {receptacle} [role: receptacle].",
        f"- goal relation: {binding.relation}.",
    ]
    if task_gap is not None:
        lines.append(f"- task gap: {task_gap}.")
    lines.append("Interaction status:")
    if slot.role == "none" or slot.grasp == "none":
        lines.append("- grasp: none.")
    elif slot.role == "goal_object":
        lines.append(f"- grasp: {slot.grasp} [role: goal_object].")
    elif slot.role == "receptacle":
        lines.append(f"- grasp: {slot.grasp} [role: receptacle].")
    else:
        lines.append(f"- grasp: {slot.grasp}.")

    released = False
    if (
        peak_attachment is not None
        and after_attachment is not None
        and after is not None
    ):
        released = released_goal_after_transport(
            peak=peak_attachment,
            after_attachment=after_attachment,
            after=after,
        )
    if released:
        lines.append(
            "- interaction outcome: released_goal_after_transport "
            "(interval grasp was the goal object; endpoint grasp is none; "
            "goal-receptacle contact is true)."
        )
    elif (
        peak is not None
        and peak.role == "held_non_goal_movable"
        and (slot.role == "none" or slot.grasp == "none")
    ):
        lines.append(
            f"- interaction outcome: released_non_goal_before_endpoint "
            f"(interval grasp was {peak.grasp}; endpoint grasp is none)."
        )
    elif slot.role == "held_non_goal_movable":
        lines.append(
            "- interaction outcome: holding_non_goal "
            "(endpoint grasp role is held_non_goal_movable)."
        )
        lines.append(
            f"- fact: bound goal_object ({payload}) is not grasped."
        )
    elif slot.role == "goal_object":
        lines.append("- interaction outcome: holding_goal_object.")
    elif slot.role == "receptacle":
        lines.append("- interaction outcome: holding_receptacle.")
    elif slot.role == "none":
        lines.append("- interaction outcome: no_grasp.")
    # Same-name acquire referent (wrong-grasp ordinal / optional empty XY).
    # Ablation: CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=0 drops the whole block.
    # With XY_LOCK, empty-hand XY is suppressed; wrong-grasp ordinal may remain.
    if after_attachment is not None:
        from capx.verification.libero_g17_xy_lock import xy_lock_enabled

        if os.environ.get("CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL", "1") != "0":
            lines.extend(
                render_same_name_xy_acquire_referent(
                    binding,
                    after_attachment,
                    peak_attachment=peak_attachment,
                    allow_empty_sibling_xy=not xy_lock_enabled(),
                )
            )
    return lines


def render_interaction_geometry_lines(
    binding: LiberoG17Binding,
    attachment: LiberoAttachmentFact,
    after: CompiledSnapshot,
) -> list[str]:
    """Gripper↔held RelXYZ (same style as goal acquire Rel), O(1) held object."""
    if attachment.kind == "empty" or not attachment.held_movable_names:
        return []
    slot = interaction_slot_from_attachment(binding, attachment)
    held_label = slot.grasp.split(" [role:")[0].strip()
    if held_label == "none":
        held_label = humanize_libero_name(attachment.held_movable_names[0])
    bands = attachment.held_rel_xyz
    if bands is None and attachment.kind == "holding_payload":
        bands = (
            after.values["executor_object.exe01.obj01.relative_x"],
            after.values["executor_object.exe01.obj01.relative_y"],
            after.values["executor_object.exe01.obj01.relative_z"],
        )
    if bands is None and attachment.kind == "holding_receptacle":
        bands = (
            after.values.get("executor_object.exe01.obj02.relative_x"),
            after.values.get("executor_object.exe01.obj02.relative_y"),
            after.values.get("executor_object.exe01.obj02.relative_z"),
        )
        if None in bands:
            return []
        bands = (str(bands[0]), str(bands[1]), str(bands[2]))
    if bands is None:
        return []
    lines = [
        "Verified interaction geometry at the endpoint "
        f"(gripper relative to held {slot.role}):",
    ]
    for axis, band in zip(("x", "y", "z"), bands):
        clause = _axis_band_cnl(axis, band, held_label)
        lines.append(
            f"- robot gripper position relative to {held_label} "
            f"along robot {axis}: {clause}."
        )
    if (
        attachment.kind == "holding_other_movable"
        and attachment.held_vs_goal_rel_xyz is not None
    ):
        goal = humanize_libero_name(binding.payload_name)
        gx, gy, gz = attachment.held_vs_goal_rel_xyz
        same_name = _public_names_collide(
            attachment.held_movable_names[0], binding.payload_name
        )
        lines.append(
            "Verified held-object vs goal-object geometry at the endpoint:"
        )
        for axis, band in zip(("x", "y", "z"), (gx, gy, gz)):
            if same_name:
                ref = f"goal instance ({goal})"
                subj = f"grasped instance ({goal})"
                clause = _axis_band_cnl(axis, band, ref)
                lines.append(
                    f"- {subj} position relative to {ref} "
                    f"along robot {axis}: {clause}."
                )
            else:
                entry = FG_EYE_CATALOG[f"object_object_relative_{axis}"]
                clause = entry.value_cnl[band].replace("{1}", goal)
                lines.append(
                    f"- {held_label} position relative to {goal} "
                    f"along robot {axis}: {clause}."
                )
        # R(goal) is emitted once from render_task_interaction_status.
    if attachment.kind == "holding_receptacle":
        lines.extend(
            render_receptacle_grasp_guard(binding, attachment, peak=attachment)
        )
    return lines


def render_interaction_delta_lines(
    binding: LiberoG17Binding,
    before: LiberoAttachmentFact,
    peak: LiberoAttachmentFact,
    after: LiberoAttachmentFact,
    *,
    after_snapshot: CompiledSnapshot | None = None,
) -> list[str]:
    """Ordered interaction-grasp Δ including interval peak (GT-backed)."""
    b = interaction_slot_from_attachment(binding, before)
    p = interaction_slot_from_attachment(binding, peak)
    a = interaction_slot_from_attachment(binding, after)
    lines: list[str] = []
    if b.grasp != p.grasp:
        lines.append(f"- changed: interaction grasp: {b.grasp} -> {p.grasp}.")
    if p.grasp != a.grasp:
        lines.append(f"- changed: interaction grasp: {p.grasp} -> {a.grasp}.")
    if after_snapshot is not None and released_goal_after_transport(
        peak=peak, after_attachment=after, after=after_snapshot
    ):
        lines.append(
            "- fact: interaction outcome released_goal_after_transport: true."
        )
    elif peak.kind == "holding_other_movable" and after.kind == "empty":
        lines.append(
            "- fact: interaction outcome released_non_goal_before_endpoint: true."
        )
    return lines


def render_attachment_lines(attachment: LiberoAttachmentFact) -> list[str]:
    """Legacy occupancy lines (kept for mutex tests / sparse arms)."""
    if attachment.kind == "empty":
        return [
            "- robot gripper attachment: empty.",
            "- robot gripper occupied by a non-goal movable object: false.",
        ]
    if attachment.kind == "holding_payload":
        return [
            "- robot gripper attachment: holding the goal object.",
            "- robot gripper occupied by a non-goal movable object: false.",
        ]
    if attachment.kind == "holding_receptacle":
        return [
            "- robot gripper attachment: holding the receptacle.",
            "- robot gripper occupied by a non-goal movable object: false.",
        ]
    if attachment.kind == "holding_other_movable":
        return [
            "- robot gripper attachment: holding a non-goal movable object.",
            "- robot gripper occupied by a non-goal movable object: true.",
        ]
    raise ValueError(f"unknown attachment kind: {attachment.kind}")


def assert_attachment_report_mutex(
    attachment: LiberoAttachmentFact,
    report: str,
    *,
    peak: LiberoAttachmentFact | None = None,
) -> None:
    """Public text must not claim empty occupancy while holding_other."""
    lowered = report.lower()
    if (
        peak is not None
        and peak.kind == "holding_other_movable"
        and attachment.kind == "empty"
    ):
        if "released_non_goal_before_endpoint" not in lowered and (
            "interaction grasp:" not in lowered
        ):
            raise ValueError(
                "interval holding_other_movable missing released_non_goal fact"
            )
        if "payload not firmly held by the gripper" in lowered:
            raise ValueError(
                "interval holding_other must not use acquisition-empty wording"
            )
        return
    if (
        peak is not None
        and peak.kind == "holding_payload"
        and attachment.kind == "empty"
        and "released_goal_after_transport" in lowered
    ):
        if "payload not firmly held by the gripper" in lowered:
            raise ValueError(
                "released_goal_after_transport must not use acquisition-empty wording"
            )
        if "task gap: acquisition" in lowered:
            raise ValueError(
                "released_goal_after_transport conflicts with task gap acquisition"
            )
        return
    if attachment.kind == "holding_other_movable":
        if "- grasp: none." in lowered and "holding_non_goal" not in lowered:
            raise ValueError("holding_other_movable conflicts with grasp: none")
        if "attachment: empty" in lowered:
            raise ValueError("holding_other_movable conflicts with attachment empty")
        if "holding_non_goal" not in lowered and (
            "occupied by a non-goal movable object: true" not in lowered
        ):
            raise ValueError("holding_other_movable missing holding_non_goal fact")
        if "payload not firmly held by the gripper" in lowered:
            raise ValueError(
                "holding_other_movable must not use bare acquisition-empty wording"
            )
    if attachment.kind == "empty":
        if "occupied by a non-goal movable object: true" in lowered:
            raise ValueError("empty attachment claims non-goal occupancy")
        if "interaction status" in lowered and "grasp: none" not in lowered:
            raise ValueError("empty attachment missing grasp: none")
        if re.search(r"- grasp: (?!none\.)", lowered):
            raise ValueError("empty attachment claims an active grasp slot")


def _spec_by_key(scene: SceneSpec) -> dict[str, Any]:
    return {spec.key: spec for spec in predicate_specs(scene)}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_goal_directed_key(state_key: str) -> bool:
    return not state_key.startswith(_RECIPROCAL_PREFIX)


def _changed_state_keys(transition: CompiledTransition) -> set[str]:
    changed: set[str] = set()
    for key, flag in transition.values.items():
        if flag != "CHANGED" or not key.startswith("transition."):
            continue
        state_key = key.removeprefix("transition.")
        if state_key.startswith(
            ("executor.", "object.", "executor_object.", "object_object.")
        ):
            changed.add(state_key)
    return changed


def detect_bound_release(
    before: CompiledSnapshot, after: CompiledSnapshot
) -> bool:
    """True iff payload loses bilateral grasp over the executed interval."""
    old = before.values.get(_PAYLOAD_GRASP_KEY, "")
    new = after.values.get(_PAYLOAD_GRASP_KEY, "")
    return old.startswith("BILATERAL_GRASP_") and not new.startswith(
        "BILATERAL_GRASP_"
    )


def assert_endpoint_key_invariants(
    binding: LiberoG17Binding, after: CompiledSnapshot
) -> None:
    """Fail closed on FG endpoint contracts before rendering."""
    scene = binding.scene
    specs = _spec_by_key(scene)
    acquire = acquire_evidence_keys(binding)
    place = place_evidence_keys(binding)
    overlap = set(acquire) & set(place)
    if overlap:
        raise ValueError(f"endpoint exclusivity violated: {sorted(overlap)}")
    for key in (*acquire, *place):
        if key not in after.values:
            raise ValueError(f"endpoint completeness: missing snapshot key {key}")
        spec = specs.get(key)
        if spec is None:
            raise ValueError(f"endpoint completeness: no PredicateSpec for {key}")
        if not fg_allows_solo_projection(spec.family):
            raise ValueError(
                f"endpoint observability: family {spec.family} not solo-projectable"
            )
        if FG_EYE_CATALOG[spec.family].tag == "instrumented":
            raise ValueError(
                f"endpoint observability: pure instrumented family {spec.family}"
            )
        value = after.values[key]
        # Bijection: every enum value must have a non-empty CNL clause.
        clause = render_fg_clause(scene, spec.family, value, spec.arguments)
        if not clause.strip():
            raise ValueError(f"endpoint bijection: empty CNL for {key}={value}")
        if value.isupper() and "_" in value and value in clause:
            raise ValueError(
                f"endpoint bijection: raw token leaked for {key}={value} -> {clause}"
            )


def parent_visible_pairs(
    scene: SceneSpec,
    after: CompiledSnapshot,
    transition: CompiledTransition,
) -> tuple[tuple[str, str], ...]:
    """Pairs the ordered-Δ report would expose (goal-directed CHANGED only)."""
    specs = _spec_by_key(scene)
    return tuple(
        (key, after.values[key])
        for key in sorted(_changed_state_keys(transition))
        if key in specs
        and key in after.values
        and _is_goal_directed_key(key)
    )


def public_finish_authorized(
    binding: LiberoG17Binding, after: CompiledSnapshot
) -> bool:
    """Binary public finish warrant: True iff public basis IFF-matches BDDL atom.

    On: ``libero_object_ontop`` ≡ ObjectState.check_ontop replica.
    In: fail-closed False until a proven public In basis exists.
    No soft / maybe-finish state.
    """
    return public_finish_authorized_for_binding(binding, after)

def release_bound_semantic_gain(
    binding: LiberoG17Binding,
    before: CompiledSnapshot,
    after: CompiledSnapshot,
    transition: CompiledTransition,
) -> ProjectionSemanticGain:
    """Track bound release privately; never emit a public success package."""
    validate_snapshot(binding.scene, before)
    validate_snapshot(binding.scene, after)
    bound = detect_bound_release(before, after)
    visible = parent_visible_pairs(binding.scene, after, transition)
    if not bound:
        return ProjectionSemanticGain(
            status="no_bound_release",
            novel_fact_ids=(),
            candidate_facts={},
            parent_visible_pairs=visible,
            bound_release=False,
        )
    # Bound release happened, but public FG cannot certify goal completion.
    return ProjectionSemanticGain(
        status="release_suppressed_no_public_iff",
        novel_fact_ids=(),
        candidate_facts={
            key: after.values[key] for key in release_candidate_keys(binding)
        },
        parent_visible_pairs=visible,
        bound_release=True,
    )


def _emit_clause_line(
    scene: SceneSpec, specs: dict[str, Any], key: str, value: str, *, prefix: str
) -> str | None:
    spec = specs.get(key)
    if spec is None:
        return None
    if not fg_allows_solo_projection(spec.family):
        return None
    if FG_EYE_CATALOG[spec.family].tag == "instrumented":
        return None
    clause = render_fg_clause(scene, spec.family, value, spec.arguments)
    return f"- {prefix}{clause}."


def _payload_acquired(after: CompiledSnapshot) -> bool:
    holding = after.values.get("executor_object.exe01.obj01.holding") == "TRUE"
    payload = after.values.get("executor.exe01.payload")
    grasp_ev = after.values.get("executor_object.exe01.obj01.grasp_evidence")
    contact = after.values.get("executor_object.exe01.obj01.contact") == "TRUE"
    return holding and payload == "OBJ01" and grasp_ev != "NOT_HELD" and contact


def _placement_partial(
    binding: LiberoG17Binding, after: CompiledSnapshot
) -> bool:
    """True when remaining work is placement-side (or legacy: receptacle contact).

    Default (v2.4): receptacle contact only.

    When ``CAPX_LIBERO_G17_WORKGAP_ME=1`` (next_version / v2.5): also treat On
    near-miss (footprint inside / support edge|center) and In volume near-miss
    as placement-side so acquisition notes do not cover near-miss place.
    """
    prefix = "object_object.obj01.obj02"
    if after.values.get(f"{prefix}.contact") == "TRUE":
        return True
    if os.environ.get("CAPX_LIBERO_G17_WORKGAP_ME", "0") != "1":
        return False
    if binding.relation == "on":
        footprint = after.values.get(f"{prefix}.center_in_reference_footprint")
        region = after.values.get(f"{prefix}.support_region")
        return footprint == "INSIDE" or region in {"EDGE_REGION", "CENTER_REGION"}
    if binding.relation == "in":
        return after.values.get(f"{prefix}.center_in_reference_volume") == "TRUE"
    return False


def classify_endpoint_gap(
    binding: LiberoG17Binding, after: CompiledSnapshot
) -> EndpointGap:
    """Binary ME gap: acquisition vs placement. Never a soft 'settled' third state.

    Public finish is gated only by ``public_finish_authorized``. Gap labels
    describe remaining work, not goal success.
    """
    if _payload_acquired(after) or _placement_partial(binding, after):
        return "placement_incomplete"
    return "acquisition_incomplete"


def render_ordered_delta_report(
    scene: SceneSpec,
    before: CompiledSnapshot,
    after: CompiledSnapshot,
    transition: CompiledTransition,
    *,
    interaction_delta_lines: list[str] | None = None,
    suppress_state_keys: frozenset[str] | None = None,
) -> str:
    """Goal-directed ordered Δ of eye-allowed families (+ interaction grasp)."""
    validate_snapshot(scene, before)
    validate_snapshot(scene, after)
    validate_transition(scene, before, after, transition)
    specs = _spec_by_key(scene)
    suppress = suppress_state_keys or frozenset()
    lines = ["Verified ordered state changes over the executed interval:"]
    emitted = 0
    for state_key in sorted(_changed_state_keys(transition)):
        if state_key in suppress:
            continue
        if not _is_goal_directed_key(state_key):
            continue
        spec = specs.get(state_key)
        if spec is None:
            continue
        # Skip semantic alias that doubles holding/support wording in Δ.
        if spec.family == "object_semantic_state":
            continue
        if not fg_allows_solo_projection(spec.family):
            continue
        if FG_EYE_CATALOG[spec.family].tag == "instrumented":
            continue
        old_clause = render_fg_clause(
            scene, spec.family, before.values[state_key], spec.arguments
        )
        new_clause = render_fg_clause(
            scene, spec.family, after.values[state_key], spec.arguments
        )
        lines.append(f"- changed: {old_clause} -> {new_clause}.")
        emitted += 1
    if interaction_delta_lines:
        lines.extend(interaction_delta_lines)
        emitted += len(interaction_delta_lines)
    if emitted == 0:
        lines.append("- no eye-checkable state changes.")
    report = "\n".join(lines)
    validate_public_fg_report(scene, report)
    return report


def render_parent_delta_report(
    scene: SceneSpec,
    before: CompiledSnapshot,
    after: CompiledSnapshot,
    transition: CompiledTransition,
) -> str:
    """Alias retained for callers; ordered Δ only (no endpoint)."""
    return render_ordered_delta_report(scene, before, after, transition)


def _center_xy_residual(
    after: CompiledSnapshot,
) -> tuple[float, float, float]:
    """Private (dx, dy, ||Δxy||) payload−receptacle; never emitted as numerics."""
    p_pos, _ = after.private_poses["obj01"]
    r_pos, _ = after.private_poses["obj02"]
    dx = float(p_pos[0] - r_pos[0])
    dy = float(p_pos[1] - r_pos[1])
    return dx, dy, float((dx * dx + dy * dy) ** 0.5)


def _depth_lateral_signs(dx: float, dy: float) -> tuple[str, str]:
    """Signed qualitative axis labels (no co-centered deadband)."""
    depth = "farther" if dx > 0.0 else "closer"
    lateral = "left" if dy > 0.0 else "right"
    return depth, lateral


def _residual_scale_cnl(radius: float) -> str:
    """Eye-checkable magnitude class vs On XY band (no cm in public text).

    within  — inside the On IFF band
    small   — outside band but within 2× the band radius (near-miss tight)
    large   — farther than 2× the band radius
    """
    band = float(LIBERO_OBJECT_ONTOP_XY_M)
    if radius < band:
        return "within"
    if radius < 2.0 * band:
        return "small"
    return "large"


def _axis_progress_cnl(before_sign: str, after_sign: str) -> str:
    if before_sign == after_sign:
        return "unchanged_sign"
    return "flipped"


def _band_progress_cnl(*, before_within: bool, after_within: bool) -> str:
    if before_within and after_within:
        return "stayed_within"
    if (not before_within) and after_within:
        return "entered_within"
    if before_within and (not after_within):
        return "left_within"
    return "still_not_within"


def _signed_center_offset_facts(
    binding: LiberoG17Binding, after: CompiledSnapshot
) -> list[str]:
    """v2.13-style signed center offsets (no residual scale / progress)."""
    payload = binding.scene.objects[0].public_name
    receptacle = binding.scene.objects[1].public_name
    dx, dy, _radius = _center_xy_residual(after)
    depth, lateral = _depth_lateral_signs(dx, dy)
    depth_clause = (
        f"closer to the robot than {receptacle} center"
        if depth == "closer"
        else f"farther from the robot than {receptacle} center"
    )
    lat_clause = (
        f"to the robot's left of {receptacle} center"
        if lateral == "left"
        else f"to the robot's right of {receptacle} center"
    )
    return [
        (
            f"- fact: center_to_center offset: {payload} center is "
            f"{depth_clause}."
        ),
        (
            f"- fact: center_to_center offset: {payload} center is "
            f"{lat_clause}."
        ),
    ]


def render_on_placement_precision_groups(
    binding: LiberoG17Binding,
    after: CompiledSnapshot,
    basis: Any,
    *,
    before: CompiledSnapshot | None = None,
    task_gap: TaskGap | None = None,
    attachment: LiberoAttachmentFact | None = None,
) -> list[str]:
    """v2.13 surface ∥ center-to-center On placement facts (no G1 salience)."""
    del before
    if binding.relation != "on":
        return []
    payload = binding.scene.objects[0].public_name
    receptacle = binding.scene.objects[1].public_name
    footprint = after.values.get(
        "object_object.obj01.obj02.center_in_reference_footprint"
    )
    surface_on = footprint == "INSIDE"
    center_within = basis.xy_band == "WITHIN_ONTOP_XY_BAND"
    near_miss = (
        basis.contact
        and surface_on
        and not center_within
        and basis.z_order == "REFERENCE_BELOW_OR_LEVEL"
    )
    lines: list[str] = [
        "Verified placement surface group "
        "(center-on-surface; not the On finish basis):",
        f"- fact: center_on_surface: {'on' if surface_on else 'off'}.",
        (
            f"- note: {payload} center "
            f"{'projects onto' if surface_on else 'does not project onto'} "
            f"the support face of {receptacle}."
        ),
        "Verified placement center-to-center group "
        "(high-precision; On XY conjunct; eye-separable):",
    ]
    if center_within:
        lines.append(
            f"- fact: {payload} is center-to-center aligned "
            f"with {receptacle} for on-top."
        )
    else:
        lines.append(
            f"- fact: {payload} is not center-to-center aligned "
            f"with {receptacle} for on-top."
        )
        lines.extend(_signed_center_offset_facts(binding, after))
    if near_miss or task_gap == "placement_xy":
        lines.append(
            "- fact: placement near-miss: center_on_surface is on while "
            "center_to_center_xy is outside "
            "(surface occupancy without on-top center alignment)."
        )
        lines.append("- fact: task gap: placement_xy.")
    elif (
        basis.contact
        and (not surface_on)
        and (not center_within)
    ):
        lines.append(
            "- fact: unfinished On conjuncts: center_on_surface and "
            "center_to_center_xy."
        )
    elif surface_on and center_within and (not basis.contact):
        lines.append(
            "- fact: unfinished On conjunct: goal–receptacle contact."
        )
    if near_miss or task_gap == "placement_xy" or (
        basis.contact and not center_within
    ):
        from capx.verification.libero_g17_band_obligations import (
            render_placement_xy_band_obligations,
        )

        dx, dy, _radius = _center_xy_residual(after)
        depth, lateral = _depth_lateral_signs(dx, dy)
        holding_payload = bool(
            attachment is not None and attachment.kind == "holding_payload"
        )
        lines.extend(
            render_placement_xy_band_obligations(
                binding,
                after,
                basis,
                depth_sign=depth,
                lateral_sign=lateral,
                dx=dx,
                dy=dy,
                holding_payload=holding_payload,
            )
        )
    return lines


def render_instance_distinction_lines(
    binding: LiberoG17Binding,
    attachment: LiberoAttachmentFact,
    *,
    peak: LiberoAttachmentFact | None = None,
) -> list[str]:
    """Retired in v2.16+ (same_public_name lottery harm)."""
    del binding, attachment, peak
    return []


def render_instance_safe_acquire_affordance(
    binding: LiberoG17Binding,
    attachment: LiberoAttachmentFact,
) -> list[str]:
    """Retired in v2.18 — use ``render_same_name_xy_acquire_referent``."""
    return render_same_name_xy_acquire_referent(binding, attachment)


def _wrong_grasp_api_xy_token(goal_vs_held_bands: tuple[str, str, str]) -> str | None:
    """Map goal↔held RelXY to a stable API token: right/left/closer/farther.

    Chess order: lateral (y) then depth (x). Tokens match robot-frame CNL.
    """
    anchor = _xy_anchor_axis(goal_vs_held_bands)
    if anchor is None:
        return None
    axis, band = anchor
    if axis == "y":
        if band == "NEGATIVE_OF_REFERENCE_BAND":
            return "right"
        if band == "POSITIVE_OF_REFERENCE_BAND":
            return "left"
        return None
    if axis == "x":
        if band == "NEGATIVE_OF_REFERENCE_BAND":
            return "closer"
        if band == "POSITIVE_OF_REFERENCE_BAND":
            return "farther"
        return None
    return None


def _wrong_grasp_side_phrase(token: str) -> str:
    """Retired helper; comparative labels use ``_wrong_grasp_comparative``."""
    return _wrong_grasp_comparative(token)


def _wrong_grasp_comparative(token: str) -> str:
    """Retired name; ordinal labels use ``_wrong_grasp_ordinal``."""
    return _wrong_grasp_ordinal(token)


def _wrong_grasp_ordinal(token: str) -> str:
    """Same-name ordinal w.r.t. the robot: first-from-right/left, closest/farthest."""
    return {
        "right": "first {name} from the robot's right",
        "left": "first {name} from the robot's left",
        "closer": "closest {name} to the robot",
        "farther": "farthest {name} from the robot",
    }[token]


def resolve_same_name_goal_ordinal(
    binding: LiberoG17Binding,
    attachment: LiberoAttachmentFact,
) -> str | None:
    """Return ordinal goal referent among same-name siblings, or None."""
    name = humanize_libero_name(binding.payload_name)
    bands = attachment.goal_vs_same_name_other_rel_xyz
    sibling = attachment.same_name_other_raw is not None
    if (
        not sibling
        and attachment.kind == "holding_other_movable"
        and attachment.held_movable_names
        and _public_names_collide(
            attachment.held_movable_names[0], binding.payload_name
        )
    ):
        sibling = True
        if bands is None and attachment.held_vs_goal_rel_xyz is not None:
            hx, hy, hz = attachment.held_vs_goal_rel_xyz
            bands = (
                _invert_rel_band(hx),
                _invert_rel_band(hy),
                _invert_rel_band(hz),
            )
    if not sibling or bands is None:
        return None
    token = _wrong_grasp_api_xy_token(bands)
    if token is None:
        return None
    return _wrong_grasp_ordinal(token).format(name=name)


def native_between_goal_referent(binding: LiberoG17Binding) -> str:
    """Starting-state native NP (Goal between-clause) for this spatial task."""
    name = humanize_libero_name(binding.payload_name)
    return f"{name} between the plate and the ramekin"


def light_target_grasp_object_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_TARGET_GRASP_OBJECT", "0") == "1"


def goal_referent_takeover_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER", "0") == "1"


def _same_name_holding_other(
    binding: LiberoG17Binding, attachment: LiberoAttachmentFact | None
) -> bool:
    if attachment is None or attachment.kind != "holding_other_movable":
        return False
    if not attachment.held_movable_names:
        return False
    return _public_names_collide(
        attachment.held_movable_names[0], binding.payload_name
    )


def render_same_name_xy_acquire_referent(
    binding: LiberoG17Binding,
    attachment: LiberoAttachmentFact,
    *,
    peak_attachment: LiberoAttachmentFact | None = None,
    allow_empty_sibling_xy: bool = True,
) -> list[str]:
    """Logic gate: enter two-bowl XY compare for same-name goal disambiguation.

    Entry (first match wins):
    1. Endpoint holding wrong same-name movable (classic wrong-grasp).
    2. Interval peak held wrong same-name but endpoint already empty/released.
    3. Empty hand with GT sibling bands (acquire-time two-bowl XY compare),
       unless ``allow_empty_sibling_xy`` is false (v223+SGP overload).

    Output is FG judgment facts only (no API coaching). Optional light fact
    under ``CAPX_LIBERO_G17_TARGET_GRASP_OBJECT``.
    """
    name = humanize_libero_name(binding.payload_name)
    source = "endpoint_holding_other"
    judge = attachment
    if not _same_name_holding_other(binding, attachment):
        if _same_name_holding_other(binding, peak_attachment):
            assert peak_attachment is not None
            judge = peak_attachment
            source = "peak_holding_other_released"
        elif (
            allow_empty_sibling_xy
            and attachment.kind == "empty"
            and attachment.same_name_other_raw is not None
            and attachment.goal_vs_same_name_other_rel_xyz is not None
        ):
            judge = attachment
            source = "empty_sibling_xy"
        else:
            return []
    ordinal = resolve_same_name_goal_ordinal(binding, judge)
    if ordinal is None:
        return []
    if source == "empty_sibling_xy":
        bands = judge.goal_vs_same_name_other_rel_xyz
        assert bands is not None
        lines = [
            "Verified same-name goal XY comparison:",
            (
                f"- fact: two graspable bodies share public name {name}; "
                "goal_object must be disambiguated in robot XY."
            ),
        ]
        for axis, band in zip(("x", "y", "z"), bands):
            lines.append(
                f"- fact: goal_object relative to the other {name} "
                f"along robot {axis}: {band}."
            )
        lines.append(f"- fact: the goal_object is the {ordinal}.")
    elif source == "peak_holding_other_released":
        lines = [
            "Verified wrong-grasp same-name acquire:",
            (
                f"- fact: interval held non-goal {name}; "
                "released before endpoint; "
                f"goal_object {name} was not grasped."
            ),
            f"- fact: the goal_object is the {ordinal}.",
        ]
    else:
        lines = [
            "Verified wrong-grasp same-name acquire:",
            (
                f"- fact: holding non-goal {name}; "
                f"goal_object {name} is not grasped."
            ),
            f"- fact: the goal_object is the {ordinal}.",
        ]
    if light_target_grasp_object_enabled():
        lines.append(f"- fact: target grasp object: {ordinal}.")
    return lines


def render_receptacle_grasp_guard(
    binding: LiberoG17Binding,
    attachment: LiberoAttachmentFact,
    *,
    peak: LiberoAttachmentFact | None = None,
) -> list[str]:
    """G3: receptacle is not the On payload during place correction."""
    peak = peak or attachment
    active = attachment.kind == "holding_receptacle"
    interval = peak.kind == "holding_receptacle" and attachment.kind == "empty"
    if not active and not interval:
        return []
    goal = humanize_libero_name(binding.payload_name)
    receptacle = humanize_libero_name(binding.receptacle_body_name)
    lines = [
        "Verified receptacle-grasp guard:",
        f"- fact: grasp role is receptacle ({receptacle}), not goal_object ({goal}).",
        (
            "- fact: On place work concerns goal_object on receptacle; "
            "the receptacle body is not the place payload."
        ),
    ]
    if interval:
        lines.append(
            "- fact: interval peak grasp was receptacle; "
            "endpoint grasp is none."
        )
    return lines


def _placement_precision_active(report: str) -> bool:
    lowered = report.lower()
    return (
        "task gap: placement_xy" in lowered
        or "unfinished on conjunct: center_to_center_xy" in lowered
        or "placement near-miss" in lowered
        or "active placement work" in lowered
        or "on_top_xy_band: not within" in lowered
        or "on_top_xy_band: outside" in lowered
    )


def scrub_placement_xy_conflicts(report: str) -> str:
    """Drop footprint / coarse RelXY / support_region center CNL under placement_xy."""
    if not _placement_precision_active(report):
        return report
    kept: list[str] = []
    for line in report.splitlines():
        ll = line.lower()
        if "center inside the footprint" in ll:
            continue
        if "over the center of the support area" in ll:
            continue
        if "over the edge of the support area" in ll:
            continue
        if "position relative to plate along robot" in ll and (
            "laterally aligned with plate" in ll
            or "at the same depth as plate" in ll
        ):
            continue
        kept.append(line)
    return "\n".join(kept)


def assert_placement_xy_report_mutex(report: str) -> None:
    """Test helper: after scrub, placement_xy must not retain conflicts."""
    cleaned = scrub_placement_xy_conflicts(report)
    lowered = cleaned.lower()
    if not _placement_precision_active(cleaned):
        return
    if "center inside the footprint" in lowered:
        raise ValueError(
            "placement_xy report must not expose footprint (conflicts with On band)"
        )
    if "over the center of the support area" in lowered:
        raise ValueError(
            "placement_xy report must not expose support_region center CNL "
            "(conflicts with center-to-center outside)"
        )
    # Per-line (same scope as scrub): whole-report AND would false-positive on
    # band-obligation goal CNL that legitimately names WITHIN/align.
    for line in cleaned.splitlines():
        ll = line.lower()
        if "position relative to plate along robot" in ll and (
            "laterally aligned with plate" in ll
            or "at the same depth as plate" in ll
        ):
            raise ValueError(
                "placement_xy report must not expose coarse RelXY aligned/same-depth "
                "vs receptacle (conflicts with On band)"
            )


def assert_fg_dominates_native_interaction(
    *,
    native_report: str,
    fg_report: str,
    attachment: LiberoAttachmentFact,
) -> dict[str, bool]:
    """Dual domination vs native VDM on interaction occupancy.

    Correctness: FG must not let native 'holding goal / pick completed' stand
    when ME says holding_other.
    Completeness: FG must name held_non_goal and (when available) interaction
    RelXYZ — not collapse to goal-only empty acquire.
    """
    native_l = native_report.lower()
    fg_l = fg_report.lower()
    verdict = {
        "applicable": attachment.kind == "holding_other_movable",
        "correctness_ok": True,
        "completeness_ok": True,
    }
    if attachment.kind != "holding_other_movable":
        return verdict
    native_claims_goal_hold = (
        (
            "grasping the akita black bowl" in native_l
            or "holding the akita black bowl" in native_l
            or "picked up the akita black bowl" in native_l
        )
        and "another" not in native_l
        and "not the goal" not in native_l
    ) or (
        "picking up the akita black bowl has been completed" in native_l
        or "task of picking up the akita black bowl has been completed" in native_l
    )
    fg_names_non_goal = (
        "held_non_goal_movable" in fg_l or "holding_non_goal" in fg_l
    )
    if native_claims_goal_hold and not fg_names_non_goal:
        verdict["correctness_ok"] = False
    # Completeness: role + geometry; wrong-grasp same-name requires ordinal.
    has_role = "role: held_non_goal_movable" in fg_l
    has_geometry = "verified interaction geometry at the endpoint" in fg_l
    if not (has_role and (has_geometry or "holding_non_goal" in fg_l)):
        # v2.4 reports name non-goal via notes, not status bars.
        if not (v24_pin_enabled() and fg_names_non_goal):
            verdict["completeness_ok"] = False
    if (
        not v24_pin_enabled()
        and attachment.held_movable_names
        and "goal object:" in fg_l
    ):
        held_pub = humanize_libero_name(attachment.held_movable_names[0])
        if f"goal object: {held_pub}" in fg_l:
            has_block = "wrong-grasp same-name acquire" in fg_l
            has_ord = "the goal_object is the " in fg_l and (
                "from the robot's right" in fg_l
                or "from the robot's left" in fg_l
                or "closest " in fg_l
                or "farthest " in fg_l
            )
            if not (has_block and has_ord):
                verdict["completeness_ok"] = False
            if (
                light_target_grasp_object_enabled()
                and "target grasp object:" not in fg_l
            ):
                verdict["completeness_ok"] = False
    elif not v24_pin_enabled() and "wrong-grasp same-name acquire" in fg_l:
        has_ord = "the goal_object is the " in fg_l and (
            "from the robot's right" in fg_l
            or "from the robot's left" in fg_l
            or "closest " in fg_l
            or "farthest " in fg_l
        )
        if not has_ord:
            verdict["completeness_ok"] = False
        if (
            light_target_grasp_object_enabled()
            and "target grasp object:" not in fg_l
        ):
            verdict["completeness_ok"] = False
    # FG must not be dominated by empty-gripper-only reading without non-goal name.
    if "grasp: none" in fg_l and "held_non_goal" not in fg_l:
        if not v24_pin_enabled():
            verdict["completeness_ok"] = False
    return verdict


def assert_fg_dominates_native_placement(
    *,
    native_report: str,
    fg_report: str,
    binding: LiberoG17Binding,
    after: CompiledSnapshot,
) -> dict[str, bool]:
    """Dual domination vs native VDM on On placement XY precision.

    Correctness: when on_top_xy_band is not within, FG must deny finish and
    must not let native 'placed / on plate done' stand without that band fact.
    Completeness: surface + center groups, unfinished conjunct on near-miss.
    """
    verdict = {
        "applicable": False,
        "correctness_ok": True,
        "completeness_ok": True,
    }
    if binding.relation != "on":
        return verdict
    basis = derive_libero_object_ontop_basis(after)
    center_not_within = basis.xy_band != "WITHIN_ONTOP_XY_BAND"
    task_gap = classify_task_gap(binding, after)
    applicable = task_gap == "placement_xy" or (
        center_not_within and basis.contact and task_gap.startswith("placement")
    )
    verdict["applicable"] = applicable
    if not applicable:
        return verdict
    fg_l = fg_report.lower()
    native_l = native_report.lower()
    has_surface = "verified placement surface group" in fg_l
    has_center = "verified placement center-to-center group" in fg_l
    has_band = (
        "not center-to-center aligned" in fg_l
        or "center_to_center_xy is outside" in fg_l
        or "on_top_xy_band: not within" in fg_l
        or "do not lie within the on-top alignment band" in fg_l
    )
    has_offset = "center_to_center offset:" in fg_l
    denies_finish = (
        "public evidence does not authorize finishing the task" in fg_l
    )
    if not (
        has_surface
        and has_center
        and has_band
        and has_offset
        and denies_finish
    ):
        verdict["completeness_ok"] = False
    native_claims_place_done = (
        "has been placed on the plate" in native_l
        or "placing the akita black bowl on the plate has been completed" in native_l
        or "successfully placed" in native_l
        or (
            "on the plate" in native_l
            and "completed" in native_l
            and "pick" not in native_l
        )
    )
    if native_claims_place_done and not (has_band and denies_finish):
        verdict["correctness_ok"] = False
    if center_not_within and not has_band:
        verdict["correctness_ok"] = False
    return verdict


def render_endpoint_evidence_report(
    binding: LiberoG17Binding,
    after: CompiledSnapshot,
    attachment: LiberoAttachmentFact | None = None,
    *,
    attachment_peak: LiberoAttachmentFact | None = None,
    before: CompiledSnapshot | None = None,
) -> tuple[str, EndpointGap]:
    """Always-on Acquire + Place endpoint assignment with ME gap notes.

    Under ``CAPX_LIBERO_G17_L1_SPARSE=1``, only the active gap subset is
    rendered (plus public IFF). Full key invariants are still asserted.
    """
    validate_snapshot(binding.scene, after)
    assert_endpoint_key_invariants(binding, after)
    if attachment is None:
        attachment = attachment_from_bound_snapshot(after)
    attachment.assert_mutex_with_snapshot(after)
    if attachment_peak is None:
        attachment_peak = attachment
    scene = binding.scene
    specs = _spec_by_key(scene)
    gap = classify_endpoint_gap(binding, after)
    task_gap = classify_task_gap(binding, after)
    slot = interaction_slot_from_attachment(binding, attachment)
    peak_slot = interaction_slot_from_attachment(binding, attachment_peak)
    released_goal = released_goal_after_transport(
        peak=attachment_peak,
        after_attachment=attachment,
        after=after,
    )
    acquire_keys = sparse_acquire_evidence_keys(binding, gap)
    place_keys = student_place_evidence_keys(binding, after, gap)

    if v24_pin_enabled():
        # Fair-K v2.4 shape: acquire + place + public IFF + binary notes only.
        lines = [
            "Verified physical endpoint for the bound placement pair:",
            "Verified acquisition evidence at the endpoint:",
        ]
        for key in acquire_keys:
            line = _emit_clause_line(scene, specs, key, after.values[key], prefix="")
            if line is None:
                raise ValueError(f"acquire endpoint failed to project {key}")
            lines.append(line)
        lines.append("Verified placement evidence at the endpoint:")
        for key in place_keys:
            line = _emit_clause_line(scene, specs, key, after.values[key], prefix="")
            if line is None:
                raise ValueError(f"place endpoint failed to project {key}")
            lines.append(line)
        if binding.relation == "on":
            basis = derive_libero_object_ontop_basis(after)
            lines.extend(render_libero_ontop_iff_lines(binding, basis))
        elif binding.relation == "in":
            if binding.containment_geom is None:
                payload = binding.scene.objects[0].public_name
                receptacle = binding.scene.objects[1].public_name
                lines.append(
                    "Verified LIBERO in-receptacle IFF basis at the endpoint:"
                )
                lines.append(
                    f"- libero in-receptacle basis for {payload} in {receptacle}: false."
                )
                lines.append(
                    "- note: containment geometry is unavailable for public In IFF."
                )
            else:
                in_basis = derive_libero_object_in_basis(
                    after, binding.containment_geom
                )
                lines.extend(render_libero_in_iff_lines(binding, in_basis))
        finish_ok = public_finish_authorized(binding, after)
        support_key = "object_object.obj01.obj02.supported_by_reference"
        if (
            binding.relation == "on"
            and after.values.get(support_key) != "TRUE"
        ):
            lines.append(
                "- note: geometric support-by-receptacle is not confirmed; "
                "this is not a benchmark completion claim."
            )
        if finish_ok:
            lines.append(
                "- note: public evidence authorizes finishing the task."
            )
        else:
            if attachment.kind == "holding_other_movable":
                lines.append(
                    "- note: gripper holds a non-goal movable object; "
                    "bound payload is not held."
                )
            elif gap == "acquisition_incomplete":
                lines.append(
                    "- note: bound acquisition evidence is incomplete "
                    "(payload not firmly held by the gripper)."
                )
            elif binding.relation == "on":
                lines.append(
                    "- note: bound placement evidence is incomplete relative to "
                    "a goal-certifying on-receptacle configuration."
                )
            else:
                lines.append(
                    "- note: bound containment evidence is incomplete relative to "
                    "a goal-certifying in-receptacle configuration."
                )
            lines.append(
                "- note: public evidence does not authorize finishing the task."
            )
        report = "\n".join(lines)
        validate_public_fg_report(scene, report)
        assert_attachment_report_mutex(
            attachment, report, peak=attachment_peak
        )
        assert_task_gap_consistency(
            after=after,
            task_gap=task_gap,
            endpoint_gap=gap,
            report=report,
            attachment=attachment,
        )
        return report, gap

    # Status bar first (task + interaction); evidence blocks follow.
    lines = render_task_interaction_status(
        binding,
        slot,
        peak=peak_slot,
        peak_attachment=attachment_peak,
        after_attachment=attachment,
        after=after,
        task_gap=task_gap,
    )
    geom_lines = render_interaction_geometry_lines(binding, attachment, after)
    if geom_lines:
        lines.extend(geom_lines)
    if attachment.kind != "holding_receptacle":
        lines.extend(
            render_receptacle_grasp_guard(
                binding, attachment, peak=attachment_peak
            )
        )
    lines.extend(
        [
            "Verified physical endpoint for the bound placement pair:",
        ]
    )
    if attachment.kind == "holding_other_movable":
        lines.append(
            "Verified goal-object acquire evidence at the endpoint "
            "(bound goal object only; gripper currently holds a different movable):"
        )
    else:
        lines.append("Verified acquisition evidence at the endpoint:")
    for key in acquire_keys:
        line = _emit_clause_line(scene, specs, key, after.values[key], prefix="")
        if line is None:
            raise ValueError(f"acquire endpoint failed to project {key}")
        lines.append(line)
    lines.append("Verified placement evidence at the endpoint:")
    for key in place_keys:
        line = _emit_clause_line(scene, specs, key, after.values[key], prefix="")
        if line is None:
            raise ValueError(f"place endpoint failed to project {key}")
        lines.append(line)

    # Public goal IFF basis (BDDL-aligned). Not G17 supported_by / volume proxies.
    if binding.relation == "on":
        basis = derive_libero_object_ontop_basis(after)
        lines.extend(render_libero_ontop_iff_lines(binding, basis))
        # Always-on surface ∥ center-to-center groups (v2.13).
        lines.extend(
            render_on_placement_precision_groups(
                binding,
                after,
                basis,
                before=before,
                task_gap=task_gap,
                attachment=attachment,
            )
        )
    elif binding.relation == "in":
        if binding.containment_geom is None:
            payload = binding.scene.objects[0].public_name
            receptacle = binding.scene.objects[1].public_name
            lines.append(
                "Verified LIBERO in-receptacle IFF basis at the endpoint:"
            )
            lines.append(
                f"- libero in-receptacle basis for {payload} in {receptacle}: false."
            )
            lines.append(
                "- note: containment geometry is unavailable for public In IFF."
            )
        else:
            in_basis = derive_libero_object_in_basis(after, binding.containment_geom)
            lines.extend(render_libero_in_iff_lines(binding, in_basis))

    finish_ok = public_finish_authorized(binding, after)
    interval_other = (
        attachment_peak.kind == "holding_other_movable"
        and attachment.kind == "empty"
    )
    if finish_ok:
        lines.append("- note: public evidence authorizes finishing the task.")
    else:
        if attachment.kind == "holding_other_movable":
            lines.append(
                "- note: interaction outcome holding_non_goal; "
                "bound goal object is not held."
            )
        elif interval_other:
            lines.append(
                "- note: interaction outcome released_non_goal_before_endpoint."
            )
        elif released_goal and task_gap == "placement_xy":
            lines.append(
                "- note: interaction outcome released_goal_after_transport; "
                "task gap is placement_xy (not acquisition)."
            )
        elif released_goal:
            lines.append(
                "- note: interaction outcome released_goal_after_transport; "
                f"task gap is {task_gap}."
            )
        elif task_gap == "placement_xy":
            lines.append(
                "- note: task gap is placement_xy "
                "(contact holds; on-top alignment band fails; "
                "not centered for on-top)."
            )
        elif task_gap.startswith("placement") or gap == "placement_incomplete":
            if binding.relation == "on":
                lines.append(
                    "- note: bound placement evidence is incomplete relative to "
                    "a goal-certifying on-receptacle configuration."
                )
            else:
                lines.append(
                    "- note: bound containment evidence is incomplete relative to "
                    "a goal-certifying in-receptacle configuration."
                )
        elif gap == "acquisition_incomplete":
            lines.append(
                "- note: bound acquisition evidence is incomplete "
                "(payload not firmly held by the gripper)."
            )
        lines.append(
            "- note: public evidence does not authorize finishing the task."
        )

    report = "\n".join(lines)
    report = scrub_placement_xy_conflicts(report)
    validate_public_fg_report(scene, report)
    assert_attachment_report_mutex(
        attachment, report, peak=attachment_peak
    )
    assert_placement_xy_report_mutex(report)
    assert_task_gap_consistency(
        after=after,
        task_gap=task_gap,
        endpoint_gap=gap,
        report=report,
        attachment=attachment,
    )
    return report, gap


def render_release_package_report(
    binding: LiberoG17Binding,
    after: CompiledSnapshot,
    gain: ProjectionSemanticGain,
) -> str:
    """Release success package: emitted only when finish is publicly authorized.

    With no public IFF basis, this always returns empty.
    """
    del binding, after, gain
    return ""


def render_world_fg_report(
    binding: LiberoG17Binding,
    before: CompiledSnapshot,
    after: CompiledSnapshot,
    transition: CompiledTransition,
    attachment: LiberoAttachmentFact | None = None,
    *,
    attachment_before: LiberoAttachmentFact | None = None,
    attachment_peak: LiberoAttachmentFact | None = None,
) -> tuple[str, str, str, EndpointGap]:
    """Return (ordered_delta, endpoint, world, gap)."""
    if attachment is None:
        attachment = attachment_from_bound_snapshot(after)
    if attachment_before is None:
        attachment_before = attachment_from_bound_snapshot(before)
    if attachment_peak is None:
        from capx.verification.libero_g17_adapter import merge_attachment_peak

        attachment_peak = merge_attachment_peak(attachment_before, attachment)
    interaction_delta = render_interaction_delta_lines(
        binding,
        attachment_before,
        attachment_peak,
        attachment,
        after_snapshot=after,
    )
    # Always drop support_region center CNL for On Δ; under placement_xy also
    # drop footprint + coarse RelXY (surface∥center groups carry those facts).
    if binding.relation == "on" and not v24_pin_enabled():
        suppress = frozenset(
            {
                "object_object.obj01.obj02.support_region",
                "object_object.obj02.obj01.support_region",
            }
        )
        if classify_task_gap(binding, after) == "placement_xy":
            suppress |= placement_xy_suppress_keys(binding)
    else:
        suppress = frozenset()
    delta = render_ordered_delta_report(
        binding.scene,
        before,
        after,
        transition,
        interaction_delta_lines=(
            [] if v24_pin_enabled() else interaction_delta
        ),
        suppress_state_keys=suppress,
    )
    endpoint, gap = render_endpoint_evidence_report(
        binding,
        after,
        attachment=attachment,
        attachment_peak=attachment_peak,
        before=before,
    )
    world = delta + "\n\n" + endpoint
    if not v24_pin_enabled():
        world = scrub_placement_xy_conflicts(world)
    validate_public_fg_report(binding.scene, world)
    if not v24_pin_enabled():
        assert_placement_xy_report_mutex(world)
    return delta, endpoint, world, gap


def world_fg_dominates_goal_evidence(
    binding: LiberoG17Binding, report: str, *, gap: EndpointGap | None = None
) -> bool:
    """Structural dominance: Δ + Acquire + Place endpoint sections.

    For On goals, public IFF (not footprint) is the placement domination basis.
    Sparse Layer-1 mode requires only the active gap subset plus public IFF.
    """
    required_headers = (
        "Verified ordered state changes over the executed interval:",
        "Verified physical endpoint for the bound placement pair:",
        "Verified placement evidence at the endpoint:",
    )
    if any(h not in report for h in required_headers):
        return False
    # Acquire header may be goal-object-scoped when holding_other.
    if (
        "Verified acquisition evidence at the endpoint:" not in report
        and "Verified goal-object acquire evidence at the endpoint" not in report
    ):
        return False
    lowered = report.lower()
    if not v24_pin_enabled():
        if "task status:" not in lowered or "interaction status:" not in lowered:
            return False
        if "grasp:" not in lowered:
            return False
        if "role: goal_object" not in lowered:
            return False
    payload = binding.scene.objects[0].public_name
    receptacle = binding.scene.objects[1].public_name
    placement_xy = (
        not v24_pin_enabled()
        and (
            "task gap: placement_xy" in lowered
            or "unfinished on conjunct: center_to_center_xy" in lowered
            or "placement near-miss" in lowered
        )
    )
    # Endpoint interaction outcome only — never key off Δ substrings
    # (Δ can still mention held_non_goal after the grasp already corrected).
    holding_other_active = "interaction outcome: holding_non_goal" in lowered
    if layer1_sparse_enabled() and gap is not None:
        if gap == "acquisition_incomplete":
            required_snippets = (
                f"robot gripper holding {payload}",
                f"contact between robot gripper and {payload}",
                f"grasp evidence for robot gripper and {payload}",
                f"contact between {payload} and {receptacle}",
            )
        else:
            required_snippets = (
                f"robot gripper holding {payload}",
                f"contact between robot gripper and {payload}",
                f"grasp evidence for robot gripper and {payload}",
                f"contact between {payload} and {receptacle}",
            )
            if binding.relation == "on" and not placement_xy:
                pass
    else:
        required_snippets = (
            f"robot gripper holding {payload}",
            f"contact between robot gripper and {payload}",
            f"grasp evidence for robot gripper and {payload}",
            f"grasp-contact state for robot gripper and {payload}",
            f"robot gripper payload",
            f"contact between {payload} and {receptacle}",
        )
        if not holding_other_active:
            required_snippets = required_snippets + (
                f"robot gripper position relative to {payload} along robot x",
                f"robot gripper position relative to {payload} along robot y",
                f"robot gripper position relative to {payload} along robot z",
            )
        if binding.relation == "in":
            required_snippets = required_snippets + (
                f"{payload} center inside the footprint of {receptacle}",
            )
    if holding_other_active:
        required_snippets = required_snippets + (
            "role: held_non_goal_movable",
            "interaction outcome: holding_non_goal",
        )
    if "wrong-grasp same-name acquire" in lowered:
        required_snippets = required_snippets + (
            "the goal_object is the ",
        )
        if light_target_grasp_object_enabled():
            required_snippets = required_snippets + (
                "target grasp object:",
            )
    if "same-name goal xy comparison" in lowered:
        required_snippets = required_snippets + (
            "the goal_object is the ",
            "goal_object relative to the other",
        )
    if binding.relation == "on":
        required_snippets = required_snippets + (
            "Verified LIBERO on-top IFF basis at the endpoint:",
            f"libero on-top basis for {payload} on {receptacle}",
            "Verified placement surface group",
            "fact: center_on_surface:",
            "Verified placement center-to-center group",
        )
        if placement_xy:
            required_snippets = required_snippets + (
                "center_to_center offset:",
                "not center-to-center aligned",
                "task gap: placement_xy",
            )
    elif binding.relation == "in":
        required_snippets = required_snippets + (
            "Verified LIBERO in-receptacle IFF basis at the endpoint:",
            f"libero in-receptacle basis for {payload} in {receptacle}",
        )
    if not all(snippet.lower() in lowered for snippet in required_snippets):
        return False
    auth = "public evidence authorizes finishing the task" in lowered
    deny = "public evidence does not authorize finishing the task" in lowered
    return auth != deny


def project_libero_fg(
    binding: LiberoG17Binding,
    before: CompiledSnapshot,
    after: CompiledSnapshot,
    attachment: LiberoAttachmentFact | None = None,
    *,
    attachment_before: LiberoAttachmentFact | None = None,
    attachment_peak: LiberoAttachmentFact | None = None,
) -> LiberoFgProjection:
    """Project world FG (Δ + Acquire/Place endpoint); release package only if finish."""
    if attachment is None:
        attachment = attachment_from_bound_snapshot(after)
    if attachment_before is None:
        attachment_before = attachment_from_bound_snapshot(before)
    if attachment_peak is None:
        from capx.verification.libero_g17_adapter import merge_attachment_peak

        attachment_peak = merge_attachment_peak(attachment_before, attachment)
    transition = compile_transition(binding.scene, before, after)
    gain = release_bound_semantic_gain(binding, before, after, transition)
    _delta, endpoint, world, gap = render_world_fg_report(
        binding,
        before,
        after,
        transition,
        attachment=attachment,
        attachment_before=attachment_before,
        attachment_peak=attachment_peak,
    )
    finish_ok = public_finish_authorized(binding, after)
    release = (
        render_release_package_report(binding, after, gain) if finish_ok else ""
    )
    if release:
        combined = world + "\n\n" + release
        validate_public_fg_report(binding.scene, combined)
    else:
        combined = world
    evidence = goal_evidence_keys(binding)
    complete = world_fg_dominates_goal_evidence(binding, world, gap=gap)
    peak_slot = interaction_slot_from_attachment(binding, attachment_peak)
    task_gap = classify_task_gap(binding, after)
    released_goal = released_goal_after_transport(
        peak=attachment_peak,
        after_attachment=attachment,
        after=after,
    )
    return LiberoFgProjection(
        parent_report=world,
        endpoint_report=endpoint,
        release_report=release,
        combined_report=combined,
        gain=gain,
        parent_sha256=_sha256_text(world),
        combined_sha256=_sha256_text(combined),
        changed_field_count=len(
            {
                key
                for key in _changed_state_keys(transition)
                if _is_goal_directed_key(key)
            }
        ),
        public_field_count=sum(
            1
            for spec in predicate_specs(binding.scene)
            if fg_allows_solo_projection(spec.family)
        ),
        goal_evidence_keys=evidence,
        world_fg_complete=complete,
        endpoint_gap=gap,
        finish_authorized=finish_ok,
        executor_attachment=attachment.kind,
        interaction_grasp=interaction_slot_from_attachment(
            binding, attachment
        ).grasp,
        interaction_grasp_peak=peak_slot.grasp,
        executor_attachment_peak=attachment_peak.kind,
        task_gap=task_gap,
        released_goal_after_transport=released_goal,
    )


def validate_public_fg_report(scene: SceneSpec, report: str) -> None:
    """Fail closed on goal leakage, private IDs, sim IDs, or numerics."""
    lowered = report.lower()
    for term in _FORBIDDEN_PUBLIC:
        if term in lowered:
            raise ValueError(f"FG report contains prohibited term: {term}")
    for entity in (*scene.executors, *scene.objects, scene.table):
        if entity.entity_id.lower() in lowered:
            raise ValueError(f"FG report exposes private ID {entity.entity_id}")
    if _NUMERIC_TOKEN.search(report):
        raise ValueError("FG report exposes a numeric value")
    if _SIM_ID.search(report):
        raise ValueError("FG report exposes a simulator-style instance ID")
    # Soft completion framing is banned: finish is binary via finish_authorized.
    if "verified release-bound placement result" in lowered:
        raise ValueError("FG report emits suppressed release-success package")
    if "settled_evidence_present" in lowered or "may finish" in lowered:
        raise ValueError("FG report contains soft/uncertain finish language")


def parent_byte_identity_preserved(projection: LiberoFgProjection) -> bool:
    """True iff no release package was spliced onto the world FG body."""
    if projection.release_report:
        return projection.combined_sha256 != projection.parent_sha256
    return projection.combined_report == projection.parent_report