"""Public goal-IFF basis for LIBERO G17 (On / In), separate from cube closed-GT.

Ideal Layer-1 finish authorization is binary:

    finish_authorized  ⇔  BDDLAtom  ⇔  task_completed

On (ObjectState.check_ontop):
    receptacle.z <= payload.z
    AND contact(receptacle, payload)
    AND ||payload.xy - receptacle.xy|| < 0.03

In (BDDL atom):
    receptacle.check_contain(payload) AND receptacle.check_contact(payload)

Site receptacles (typical basket contain_region):
    contain := SiteObject.in_box(site_pos, site_mat, payload_pos)
    contact := True  (SiteObjectState.check_contact)

Composite/object receptacles:
    contain := CompositeObject.in_box(body_pos, payload_pos) with total_size AABB
    contact := body contact

G17 volume/footprint proxies are **not** these predicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from capx.verification.closed_gt import CompiledSnapshot
from capx.verification.libero_g17_adapter import LiberoG17Binding

# Exact threshold from libero/.../base_object_states.py ObjectState.check_ontop.
LIBERO_OBJECT_ONTOP_XY_M = 0.03
# Exact z fudge from SiteObject.in_box / CompositeObject.in_box.
LIBERO_IN_BOX_Z_FUDGE_M = 0.01

OntopXy = Literal["WITHIN_ONTOP_XY_BAND", "OUTSIDE_ONTOP_XY_BAND"]
OntopZ = Literal["REFERENCE_BELOW_OR_LEVEL", "REFERENCE_ABOVE"]
ContainKind = Literal["site_box", "composite_aabb"]


@dataclass(frozen=True)
class LiberoContainmentGeom:
    """Frame-aligned containment geometry for public In IFF (robot/base frame)."""

    kind: ContainKind
    position: np.ndarray  # (3,)
    size: np.ndarray  # (3,) site size or composite total_size
    rotation_mat: np.ndarray | None  # (3,3) for site_box; None for composite_aabb
    contact_required: bool  # False for sites (BDDL contact always true)


@dataclass(frozen=True)
class LiberoOntopBasis:
    """Public conjuncts for payload-on-receptacle (ObjectState.check_ontop)."""

    contact: bool
    xy_band: OntopXy
    z_order: OntopZ
    libero_object_ontop: bool
    xy_distance_m: float


@dataclass(frozen=True)
class LiberoContainBasis:
    """Public conjuncts for payload-in-receptacle (BDDL In)."""

    kind: ContainKind
    contact: bool
    contact_required: bool
    inside_volume: bool
    libero_object_in: bool


def _pose(snapshot: CompiledSnapshot, entity_id: str) -> np.ndarray:
    pos, _quat = snapshot.private_poses[entity_id]
    return np.asarray(pos, dtype=np.float64).reshape(3)


def _payload_receptacle_contact(
    snapshot: CompiledSnapshot,
    *,
    payload_id: str = "obj01",
    receptacle_id: str = "obj02",
) -> bool:
    contact_key = (
        f"object_object.{min(payload_id, receptacle_id)}."
        f"{max(payload_id, receptacle_id)}.contact"
    )
    return snapshot.values.get(contact_key) == "TRUE"


def derive_libero_object_ontop_basis(
    snapshot: CompiledSnapshot,
    *,
    payload_id: str = "obj01",
    receptacle_id: str = "obj02",
) -> LiberoOntopBasis:
    """Derive ObjectState.check_ontop(receptacle, payload) from private poses + contact."""
    contact = _payload_receptacle_contact(
        snapshot, payload_id=payload_id, receptacle_id=receptacle_id
    )
    payload = _pose(snapshot, payload_id)
    receptacle = _pose(snapshot, receptacle_id)
    xy_distance = float(np.linalg.norm(payload[:2] - receptacle[:2]))
    xy_ok = xy_distance < LIBERO_OBJECT_ONTOP_XY_M
    z_ok = bool(receptacle[2] <= payload[2])
    ontop = bool(contact and xy_ok and z_ok)
    return LiberoOntopBasis(
        contact=contact,
        xy_band="WITHIN_ONTOP_XY_BAND" if xy_ok else "OUTSIDE_ONTOP_XY_BAND",
        z_order="REFERENCE_BELOW_OR_LEVEL" if z_ok else "REFERENCE_ABOVE",
        libero_object_ontop=ontop,
        xy_distance_m=xy_distance,
    )


def replica_object_check_ontop(
    *,
    receptacle_position: np.ndarray,
    payload_position: np.ndarray,
    contact: bool,
    xy_threshold_m: float = LIBERO_OBJECT_ONTOP_XY_M,
) -> bool:
    """Byte-faithful replica of ObjectState.check_ontop geometry."""
    receptacle_position = np.asarray(receptacle_position, dtype=np.float64).reshape(3)
    payload_position = np.asarray(payload_position, dtype=np.float64).reshape(3)
    return bool(
        (receptacle_position[2] <= payload_position[2])
        and contact
        and (
            np.linalg.norm(receptacle_position[:2] - payload_position[:2])
            < xy_threshold_m
        )
    )


def replica_site_in_box(
    *,
    site_position: np.ndarray,
    site_mat: np.ndarray,
    site_size: np.ndarray,
    payload_position: np.ndarray,
    z_fudge_m: float = LIBERO_IN_BOX_Z_FUDGE_M,
) -> bool:
    """Byte-faithful replica of SiteObject.in_box."""
    site_position = np.asarray(site_position, dtype=np.float64).reshape(3)
    site_mat = np.asarray(site_mat, dtype=np.float64).reshape(3, 3)
    site_size = np.asarray(site_size, dtype=np.float64).reshape(3)
    payload_position = np.asarray(payload_position, dtype=np.float64).reshape(3)
    total_size = np.abs(site_mat @ site_size)
    ub = site_position + total_size
    lb = site_position - total_size
    lb = lb.copy()
    lb[2] -= z_fudge_m
    return bool(np.all(payload_position > lb) and np.all(payload_position < ub))


def replica_composite_in_box(
    *,
    body_position: np.ndarray,
    total_size: np.ndarray,
    payload_position: np.ndarray,
    z_fudge_m: float = LIBERO_IN_BOX_Z_FUDGE_M,
) -> bool:
    """Byte-faithful replica of CompositeObject.in_box."""
    body_position = np.asarray(body_position, dtype=np.float64).reshape(3)
    total_size = np.asarray(total_size, dtype=np.float64).reshape(3)
    payload_position = np.asarray(payload_position, dtype=np.float64).reshape(3)
    ub = body_position + total_size
    lb = body_position - total_size
    lb = lb.copy()
    lb[2] -= z_fudge_m
    return bool(np.all(payload_position > lb) and np.all(payload_position < ub))


def derive_libero_object_in_basis(
    snapshot: CompiledSnapshot,
    containment: LiberoContainmentGeom,
    *,
    payload_id: str = "obj01",
) -> LiberoContainBasis:
    """Derive BDDL In conjuncts from containment geom + snapshot contact."""
    payload = _pose(snapshot, payload_id)
    body_contact = _payload_receptacle_contact(snapshot)
    if containment.kind == "site_box":
        if containment.rotation_mat is None:
            raise ValueError("site_box containment requires rotation_mat")
        inside = replica_site_in_box(
            site_position=containment.position,
            site_mat=containment.rotation_mat,
            site_size=containment.size,
            payload_position=payload,
        )
        # SiteObjectState.check_contact always returns True.
        contact = True
    elif containment.kind == "composite_aabb":
        inside = replica_composite_in_box(
            body_position=containment.position,
            total_size=containment.size,
            payload_position=payload,
        )
        contact = body_contact
    else:
        raise ValueError(f"unknown containment kind: {containment.kind}")
    if containment.contact_required:
        in_atom = bool(inside and contact)
    else:
        in_atom = bool(inside)
    return LiberoContainBasis(
        kind=containment.kind,
        contact=contact,
        contact_required=containment.contact_required,
        inside_volume=inside,
        libero_object_in=in_atom,
    )


def public_finish_authorized_for_binding(
    binding: LiberoG17Binding, snapshot: CompiledSnapshot
) -> bool:
    """Binary finish warrant: True iff public basis IFF-matches the BDDL atom family."""
    if binding.relation == "on":
        return derive_libero_object_ontop_basis(snapshot).libero_object_ontop
    if binding.relation == "in":
        geom = binding.containment_geom
        if geom is None:
            # Fail closed: In without extracted containment geometry.
            return False
        return derive_libero_object_in_basis(snapshot, geom).libero_object_in
    return False


def render_libero_ontop_iff_lines(
    binding: LiberoG17Binding, basis: LiberoOntopBasis
) -> list[str]:
    """Eye-facing CNL for the On IFF conjuncts (no success / BDDL leakage)."""
    payload = binding.scene.objects[0].public_name
    receptacle = binding.scene.objects[1].public_name
    contact_cnl = "true" if basis.contact else "false"
    if basis.xy_band == "WITHIN_ONTOP_XY_BAND":
        xy_cnl = (
            f"horizontal centers of {payload} and {receptacle} "
            "lie within the on-top alignment band"
        )
    else:
        xy_cnl = (
            f"horizontal centers of {payload} and {receptacle} "
            "do not lie within the on-top alignment band"
        )
    if basis.z_order == "REFERENCE_BELOW_OR_LEVEL":
        z_cnl = f"{receptacle} is at or below {payload} in height"
    else:
        z_cnl = f"{receptacle} is above {payload} in height"
    ontop_cnl = "true" if basis.libero_object_ontop else "false"
    return [
        "Verified LIBERO on-top IFF basis at the endpoint:",
        f"- contact between {payload} and {receptacle}: {contact_cnl}.",
        f"- {xy_cnl}.",
        f"- {z_cnl}.",
        f"- libero on-top basis for {payload} on {receptacle}: {ontop_cnl}.",
    ]


def render_libero_in_iff_lines(
    binding: LiberoG17Binding, basis: LiberoContainBasis
) -> list[str]:
    """Eye-facing CNL for the In IFF conjuncts."""
    payload = binding.scene.objects[0].public_name
    receptacle = binding.scene.objects[1].public_name
    volume_cnl = "inside" if basis.inside_volume else "outside"
    contact_cnl = "true" if basis.contact else "false"
    in_cnl = "true" if basis.libero_object_in else "false"
    kind_cnl = (
        "site containment box"
        if basis.kind == "site_box"
        else "receptacle axis-aligned containment box"
    )
    lines = [
        "Verified LIBERO in-receptacle IFF basis at the endpoint:",
        f"- {payload} center relative to the {kind_cnl} of {receptacle}: {volume_cnl}.",
    ]
    if basis.contact_required:
        lines.append(
            f"- contact between {payload} and {receptacle}: {contact_cnl}."
        )
    else:
        lines.append(
            f"- receptacle contact requirement for {receptacle}: not required."
        )
    lines.append(
        f"- libero in-receptacle basis for {payload} in {receptacle}: {in_cnl}."
    )
    return lines


def assert_ontop_basis_iff_replica(
    basis: LiberoOntopBasis,
    *,
    contact: bool,
    payload_pos: np.ndarray,
    receptacle_pos: np.ndarray,
) -> None:
    """Fail closed if derived basis disagrees with the check_ontop replica."""
    expected = replica_object_check_ontop(
        receptacle_position=receptacle_pos,
        payload_position=payload_pos,
        contact=contact,
    )
    if basis.libero_object_ontop != expected:
        raise AssertionError(
            "libero_object_ontop disagrees with check_ontop replica: "
            f"basis={basis.libero_object_ontop} replica={expected} "
            f"xy={basis.xy_distance_m} contact={contact}"
        )


def assert_in_basis_iff_replica(
    basis: LiberoContainBasis,
    *,
    containment: LiberoContainmentGeom,
    payload_pos: np.ndarray,
    body_contact: bool,
) -> None:
    """Fail closed if derived In basis disagrees with in_box replicas."""
    if containment.kind == "site_box":
        assert containment.rotation_mat is not None
        expected_inside = replica_site_in_box(
            site_position=containment.position,
            site_mat=containment.rotation_mat,
            site_size=containment.size,
            payload_position=payload_pos,
        )
        expected = expected_inside
    else:
        expected_inside = replica_composite_in_box(
            body_position=containment.position,
            total_size=containment.size,
            payload_position=payload_pos,
        )
        expected = expected_inside and body_contact
    if basis.inside_volume != expected_inside or basis.libero_object_in != expected:
        raise AssertionError(
            "libero_object_in disagrees with in_box replica: "
            f"basis_in={basis.libero_object_in} expected={expected} "
            f"inside={basis.inside_volume} expected_inside={expected_inside}"
        )


def audit_supported_by_vs_ontop(
    snapshot: CompiledSnapshot,
) -> dict[str, Any]:
    """Private audit: quantify the known supported_by ⇏ On hole on one snapshot."""
    basis = derive_libero_object_ontop_basis(snapshot)
    supported = (
        snapshot.values.get("object_object.obj01.obj02.supported_by_reference")
        == "TRUE"
    )
    return {
        "libero_object_ontop": basis.libero_object_ontop,
        "g17_supported_by": supported,
        "agree": basis.libero_object_ontop == supported,
        "supported_true_ontop_false": supported and not basis.libero_object_ontop,
        "ontop_true_supported_false": basis.libero_object_ontop and not supported,
        "xy_band": basis.xy_band,
        "z_order": basis.z_order,
        "contact": basis.contact,
        "xy_distance_m": basis.xy_distance_m,
    }


def audit_volume_vs_in(
    snapshot: CompiledSnapshot,
    containment: LiberoContainmentGeom,
) -> dict[str, Any]:
    """Private audit: G17 center_in_reference_volume vs public In basis."""
    basis = derive_libero_object_in_basis(snapshot, containment)
    volume = snapshot.values.get(
        "object_object.obj01.obj02.center_in_reference_volume"
    )
    g17_in_proxy = volume == "INSIDE" and _payload_receptacle_contact(snapshot)
    return {
        "libero_object_in": basis.libero_object_in,
        "g17_volume": volume,
        "g17_in_proxy": g17_in_proxy,
        "agree_with_g17_proxy": basis.libero_object_in == g17_in_proxy,
        "inside_volume": basis.inside_volume,
        "kind": basis.kind,
    }
