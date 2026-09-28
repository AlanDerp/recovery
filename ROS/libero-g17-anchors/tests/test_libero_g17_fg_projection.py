"""Unit tests for LIBERO G17 FG projection (no MuJoCo, no baseline writes)."""

from __future__ import annotations

import numpy as np
import pytest

from capx.verification import closed_gt
from capx.verification import libero_g17_adapter as adapter
from capx.verification import libero_g17_fg_projection as projection


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
    contact_payload_plate: bool,
    z_payload: float,
    holding_payload: bool = False,
    gripper_contact_payload: bool = False,
    aperture: float = 1.0,
    payload_xy: tuple[float, float] = (0.0, 0.0),
) -> closed_gt.RawSceneState:
    executor_contacts = (
        frozenset({("exe01", "obj01")}) if gripper_contact_payload else frozenset()
    )
    holding = frozenset({("exe01", "obj01")}) if holding_payload else frozenset()
    object_contacts = (
        frozenset({frozenset({"obj01", "obj02"})})
        if contact_payload_plate
        else frozenset()
    )
    table_contacts = frozenset({"obj02"})
    if not holding_payload and z_payload < 0.04:
        table_contacts = frozenset({"obj01", "obj02"})
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
                position=np.array([payload_xy[0], payload_xy[1], z_payload]),
                orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                half_extents=np.array([0.03, 0.03, 0.02]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            ),
            "obj02": closed_gt.BodyState(
                position=np.array([0.0, 0.0, 0.02]),
                orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                half_extents=np.array([0.08, 0.08, 0.01]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            ),
        },
        executor_aperture={"exe01": aperture},
        holding=holding,
        executor_contacts=executor_contacts,
        object_contacts=object_contacts,
        table_contacts=table_contacts,
        table_height=0.0,
    )


def test_acquire_place_keys_are_disjoint_and_complete() -> None:
    binding = adapter.bind_libero_g17_goal(
        _spatial_problem(), suite_name="libero_spatial", task_id=0
    )
    acquire = projection.acquire_evidence_keys(binding)
    place = projection.place_evidence_keys(binding)
    assert set(acquire).isdisjoint(place)
    assert projection.goal_evidence_keys(binding) == acquire + place
    assert all(
        "support_region" not in k and "supported_by" not in k for k in place
    )
    after = closed_gt.compile_snapshot(
        binding.scene, _raw(contact_payload_plate=True, z_payload=0.05)
    )
    projection.assert_endpoint_key_invariants(binding, after)


def test_ontop_success_authorizes_finish_without_release_package() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(
        binding.scene, _raw(contact_payload_plate=True, z_payload=0.05)
    )
    proj = projection.project_libero_fg(binding, snap, snap)
    assert proj.finish_authorized is True
    assert proj.world_fg_complete
    assert "Verified LIBERO on-top IFF basis" in proj.endpoint_report
    assert "public evidence authorizes finishing the task" in proj.endpoint_report
    assert "does not authorize finishing" not in proj.endpoint_report
    assert proj.release_report == ""
    assert "Verified release-bound placement result" not in proj.combined_report


def test_footprint_only_does_not_authorize_finish() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(
        binding.scene,
        _raw(
            contact_payload_plate=True,
            z_payload=0.05,
            payload_xy=(0.05, 0.0),
        ),
    )
    proj = projection.project_libero_fg(binding, snap, snap)
    assert proj.finish_authorized is False
    assert "does not authorize finishing" in proj.endpoint_report
    assert "OUTSIDE" not in proj.endpoint_report  # no raw enum leak required
    assert "do not lie within the on-top alignment band" in proj.endpoint_report
    assert "not center-to-center aligned" in proj.endpoint_report
    assert "placement near-miss" in proj.endpoint_report


def test_acquisition_incomplete_when_not_holding() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(
        binding.scene, _raw(contact_payload_plate=False, z_payload=0.12)
    )
    proj = projection.project_libero_fg(binding, snap, snap)
    assert proj.endpoint_gap == "acquisition_incomplete"
    assert proj.finish_authorized is False
    assert "acquisition evidence is incomplete" in proj.endpoint_report


def test_placement_incomplete_when_holding() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(
        binding.scene,
        _raw(
            contact_payload_plate=False,
            z_payload=0.15,
            holding_payload=True,
            gripper_contact_payload=True,
            aperture=0.2,
        ),
    )
    proj = projection.project_libero_fg(binding, snap, snap)
    assert proj.endpoint_gap == "placement_incomplete"
    assert proj.task_gap == "placement_other"
    assert proj.finish_authorized is False
    assert "placement evidence is incomplete" in proj.endpoint_report
    assert "task gap: placement_other" in proj.endpoint_report.lower()
    assert "task gap: acquisition" not in proj.endpoint_report.lower()
    assert "holding_goal_object" in proj.endpoint_report.lower()


def test_task_gap_acquisition_only_when_not_acquired() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(
        binding.scene, _raw(contact_payload_plate=False, z_payload=0.12)
    )
    assert projection.classify_task_gap(binding, snap) == "acquisition"
    assert projection.classify_endpoint_gap(binding, snap) == "acquisition_incomplete"


def test_task_gap_mutex_rejects_acquisition_while_holding() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(
        binding.scene,
        _raw(
            contact_payload_plate=False,
            z_payload=0.15,
            holding_payload=True,
            gripper_contact_payload=True,
            aperture=0.2,
        ),
    )
    try:
        projection.assert_task_gap_consistency(
            after=snap,
            task_gap="acquisition",
            endpoint_gap="placement_incomplete",
            report=(
                "Task status:\n- task gap: acquisition.\n"
                "Interaction status:\n- interaction outcome: holding_goal_object.\n"
            ),
            attachment=adapter.LiberoAttachmentFact(
                "holding_payload", ("akita_black_bowl_1",)
            ),
        )
    except ValueError as exc:
        assert "acquisition" in str(exc).lower()
    else:
        raise AssertionError("expected task_gap consistency ValueError")


def test_endpoint_gap_is_binary() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    cases = [
        _raw(contact_payload_plate=True, z_payload=0.05),
        _raw(
            contact_payload_plate=False,
            z_payload=0.15,
            holding_payload=True,
            gripper_contact_payload=True,
            aperture=0.2,
        ),
        _raw(contact_payload_plate=False, z_payload=0.12),
    ]
    labels = {
        projection.classify_endpoint_gap(
            binding, closed_gt.compile_snapshot(binding.scene, raw)
        )
        for raw in cases
    }
    assert labels <= {"placement_incomplete", "acquisition_incomplete"}
    assert "settled_evidence_present" not in labels


def test_assert_endpoint_rejects_missing_key() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    after = closed_gt.compile_snapshot(
        binding.scene, _raw(contact_payload_plate=True, z_payload=0.05)
    )
    broken = closed_gt.CompiledSnapshot(
        after.schema_version,
        {k: v for k, v in after.values.items() if k != "executor.exe01.payload"},
        after.private_poses,
    )
    with pytest.raises(ValueError, match="completeness"):
        projection.assert_endpoint_key_invariants(binding, broken)
