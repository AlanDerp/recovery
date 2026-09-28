"""Unit tests: LIBERO public On IFF ≡ ObjectState.check_ontop replica."""

from __future__ import annotations

import numpy as np

from capx.verification import closed_gt
from capx.verification import libero_g17_adapter as adapter
from capx.verification import libero_g17_fg_projection as projection
from capx.verification import libero_g17_public_iff as public_iff


def _spatial_problem() -> dict:
    return {
        "fixtures": {"table": ["main_table"]},
        "objects": {
            "akita_black_bowl": ["akita_black_bowl_1"],
            "plate": ["plate_1"],
        },
        "goal_state": [["On", "akita_black_bowl_1", "plate_1"]],
    }


def _raw(
    *,
    contact: bool,
    payload_xy: tuple[float, float],
    payload_z: float,
    receptacle_xy: tuple[float, float] = (0.0, 0.0),
    receptacle_z: float = 0.02,
) -> closed_gt.RawSceneState:
    object_contacts = (
        frozenset({frozenset({"obj01", "obj02"})}) if contact else frozenset()
    )
    return closed_gt.RawSceneState(
        executors={
            "exe01": closed_gt.BodyState(
                position=np.array([0.0, 0.0, 0.30]),
                orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                half_extents=np.array([0.02, 0.02, 0.02]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            )
        },
        objects={
            "obj01": closed_gt.BodyState(
                position=np.array([payload_xy[0], payload_xy[1], payload_z]),
                orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                half_extents=np.array([0.03, 0.03, 0.02]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            ),
            "obj02": closed_gt.BodyState(
                position=np.array(
                    [receptacle_xy[0], receptacle_xy[1], receptacle_z]
                ),
                orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                half_extents=np.array([0.08, 0.08, 0.01]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            ),
        },
        executor_aperture={"exe01": 1.0},
        holding=frozenset(),
        executor_contacts=frozenset(),
        object_contacts=object_contacts,
        table_contacts=frozenset({"obj02"}),
        table_height=0.0,
    )


def test_replica_matches_libero_object_check_ontop_definition() -> None:
    # Faithful to base_object_states.ObjectState.check_ontop
    assert public_iff.replica_object_check_ontop(
        receptacle_position=np.array([0.0, 0.0, 0.02]),
        payload_position=np.array([0.0, 0.0, 0.05]),
        contact=True,
    )
    assert not public_iff.replica_object_check_ontop(
        receptacle_position=np.array([0.0, 0.0, 0.02]),
        payload_position=np.array([0.04, 0.0, 0.05]),  # > 0.03 xy
        contact=True,
    )
    assert not public_iff.replica_object_check_ontop(
        receptacle_position=np.array([0.0, 0.0, 0.02]),
        payload_position=np.array([0.0, 0.0, 0.05]),
        contact=False,
    )
    assert not public_iff.replica_object_check_ontop(
        receptacle_position=np.array([0.0, 0.0, 0.06]),
        payload_position=np.array([0.0, 0.0, 0.05]),  # receptacle above
        contact=True,
    )


def test_derived_basis_iff_replica_on_pose_grid() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    cases = []
    for contact in (True, False):
        for dx in (0.0, 0.02, 0.029, 0.03, 0.05):
            for z in (0.01, 0.02, 0.05):
                cases.append((contact, dx, z))
    disagree_supported = 0
    for contact, dx, z in cases:
        snap = closed_gt.compile_snapshot(
            binding.scene,
            _raw(contact=contact, payload_xy=(dx, 0.0), payload_z=z),
        )
        basis = public_iff.derive_libero_object_ontop_basis(snap)
        payload = snap.private_poses["obj01"][0]
        receptacle = snap.private_poses["obj02"][0]
        public_iff.assert_ontop_basis_iff_replica(
            basis,
            contact=contact,
            payload_pos=payload,
            receptacle_pos=receptacle,
        )
        audit = public_iff.audit_supported_by_vs_ontop(snap)
        if not audit["agree"]:
            disagree_supported += 1
    # The whole point of this module: supported_by is not On.
    assert disagree_supported > 0


def test_finish_authorized_iff_ontop_basis() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    success = closed_gt.compile_snapshot(
        binding.scene,
        _raw(contact=True, payload_xy=(0.0, 0.0), payload_z=0.05),
    )
    fail_xy = closed_gt.compile_snapshot(
        binding.scene,
        _raw(contact=True, payload_xy=(0.05, 0.0), payload_z=0.05),
    )
    assert public_iff.public_finish_authorized_for_binding(binding, success) is True
    assert public_iff.public_finish_authorized_for_binding(binding, fail_xy) is False
    proj_ok = projection.project_libero_fg(binding, success, success)
    proj_bad = projection.project_libero_fg(binding, fail_xy, fail_xy)
    assert proj_ok.finish_authorized is True
    assert "public evidence authorizes finishing the task" in proj_ok.endpoint_report
    assert "Verified LIBERO on-top IFF basis" in proj_ok.endpoint_report
    assert proj_bad.finish_authorized is False
    assert "does not authorize finishing" in proj_bad.endpoint_report


def test_footprint_inside_is_not_ontop_iff() -> None:
    """Classic hole: large footprint INSIDE but XY centers > 3cm ⇒ not On."""
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    # Plate half-extent 0.08 ⇒ dx=0.05 still inside footprint, outside 3cm band.
    snap = closed_gt.compile_snapshot(
        binding.scene,
        _raw(contact=True, payload_xy=(0.05, 0.0), payload_z=0.05),
    )
    assert (
        snap.values["object_object.obj01.obj02.center_in_reference_footprint"]
        == "INSIDE"
    )
    basis = public_iff.derive_libero_object_ontop_basis(snap)
    assert basis.libero_object_ontop is False
    assert basis.xy_band == "OUTSIDE_ONTOP_XY_BAND"
