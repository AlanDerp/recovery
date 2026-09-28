"""Executor attachment ME + distractor occupancy (no MuJoCo)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from capx.verification import closed_gt
from capx.verification import libero_g17_adapter as adapter
from capx.verification import libero_g17_fg_projection as projection


def _spatial_problem() -> dict:
    return {
        "fixtures": {"table": ["main_table"]},
        "objects": {
            "akita_black_bowl": ["akita_black_bowl_1", "akita_black_bowl_2"],
            "plate": ["plate_1"],
        },
        "goal_state": [["On", "akita_black_bowl_1", "plate_1"]],
    }


def _raw(
    *,
    holding_payload: bool = False,
    contact_payload_plate: bool = False,
    payload_xy: tuple[float, float] = (0.0, 0.0),
    z_payload: float | None = None,
) -> closed_gt.RawSceneState:
    holding = frozenset({("exe01", "obj01")}) if holding_payload else frozenset()
    contacts = frozenset({("exe01", "obj01")}) if holding_payload else frozenset()
    object_contacts = (
        frozenset({frozenset({"obj01", "obj02"})})
        if contact_payload_plate
        else frozenset()
    )
    if z_payload is None:
        z_payload = 0.12 if holding_payload else 0.02
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
        executor_aperture={"exe01": 0.0 if holding_payload else 1.0},
        holding=holding,
        executor_contacts=contacts,
        object_contacts=object_contacts,
        table_contacts=table_contacts,
        table_height=0.0,
    )


def test_attachment_kinds_are_mutually_exclusive() -> None:
    assert len(set(adapter.ATTACHMENT_KINDS)) == 4


def test_classify_distractor_hold() -> None:
    binding = adapter.bind_libero_g17_goal(
        _spatial_problem(), suite_name="libero_spatial", task_id=0
    )
    models = {
        "akita_black_bowl_1": SimpleNamespace(contact_geoms=["a"]),
        "akita_black_bowl_2": SimpleNamespace(contact_geoms=["b"]),
        "plate_1": SimpleNamespace(contact_geoms=["c"]),
        "cookies_1": SimpleNamespace(contact_geoms=["d"]),
    }
    held = {"akita_black_bowl_2"}

    class Domain:
        objects_dict = {k: SimpleNamespace() for k in models}
        object_states_dict = {k: SimpleNamespace() for k in models}

        def get_object(self, name):
            return models[name]

        def _check_grasp(self, gripper, object_geoms):
            for name, model in models.items():
                if model is object_geoms:
                    return name in held
            return False

    fact = adapter.classify_executor_attachment(Domain(), gripper=object(), binding=binding)
    assert fact.kind == "holding_other_movable"
    assert fact.held_movable_names == ("akita_black_bowl_2",)


def test_holding_other_mutex_with_bound_payload_hold() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(binding.scene, _raw(holding_payload=True))
    bad = adapter.LiberoAttachmentFact(
        "holding_other_movable", ("akita_black_bowl_2",)
    )
    with pytest.raises(ValueError, match="conflicts with payload hold"):
        bad.assert_mutex_with_snapshot(snap)


def test_projection_surfaces_distractor_not_empty_gripper() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(binding.scene, _raw(holding_payload=False))
    att = adapter.LiberoAttachmentFact(
        "holding_other_movable",
        ("akita_black_bowl_2",),
        held_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
        ),
        held_vs_goal_rel_xyz=(
            "NEGATIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    att.assert_mutex_with_snapshot(snap)
    proj = projection.project_libero_fg(binding, snap, snap, attachment=att)
    text = proj.combined_report.lower()
    assert proj.executor_attachment == "holding_other_movable"
    assert "role: held_non_goal_movable" in proj.interaction_grasp
    assert "- grasp: none." not in text
    assert "held_non_goal_movable" in text
    assert "holding_non_goal" in text
    assert "verified interaction geometry" in text
    assert "payload not firmly held by the gripper" not in text
    assert "null" not in text
    assert proj.world_fg_complete is True
    native = (
        "In the current state, the robot's gripper is closed and is grasping "
        "the akita black bowl. The task of picking up the akita black bowl "
        "has been completed."
    )
    dom = projection.assert_fg_dominates_native_interaction(
        native_report=native, fg_report=proj.combined_report, attachment=att
    )
    assert dom["applicable"] and dom["correctness_ok"] and dom["completeness_ok"]


def test_empty_and_payload_attachments_render() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    empty_snap = closed_gt.compile_snapshot(
        binding.scene, _raw(holding_payload=False)
    )
    held_snap = closed_gt.compile_snapshot(
        binding.scene, _raw(holding_payload=True)
    )
    p_empty = projection.project_libero_fg(binding, empty_snap, empty_snap)
    assert p_empty.executor_attachment == "empty"
    assert p_empty.interaction_grasp == "none"
    assert "grasp: none" in p_empty.combined_report.lower()
    assert "null" not in p_empty.combined_report.lower()
    p_hold = projection.project_libero_fg(binding, held_snap, held_snap)
    assert p_hold.executor_attachment == "holding_payload"
    assert p_hold.interaction_grasp == "akita black bowl"
    assert "grasp: akita black bowl [role: goal_object]" in p_hold.combined_report.lower()


def test_interaction_slot_facts_for_wrong_target() -> None:
    """Distractor hold must name non-goal occupancy (not soft steer prose)."""
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(binding.scene, _raw(holding_payload=False))
    att = adapter.LiberoAttachmentFact(
        "holding_other_movable",
        ("akita_black_bowl_2",),
        held_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
        ),
        held_vs_goal_rel_xyz=(
            "NEGATIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    slot = projection.interaction_slot_from_attachment(binding, att)
    assert slot.role == "held_non_goal_movable"
    assert "role: held_non_goal_movable" in slot.grasp
    assert "not the goal object" in slot.grasp
    lines = projection.render_task_interaction_status(binding, slot)
    text = "\n".join(lines).lower()
    assert "goal object: akita black bowl" in text
    assert "receptacle: plate" in text
    assert "held_non_goal_movable" in text
    assert "holding_non_goal" in text
    assert "grasp: none" not in text
    assert "null" not in text
    projection.validate_public_fg_report(binding.scene, "\n".join(lines))
    proj = projection.project_libero_fg(binding, snap, snap, attachment=att)
    assert proj.world_fg_complete is True


def test_interaction_delta_captures_wrong_grasp_then_release() -> None:
    """Interval peak must appear in ordered Δ even when endpoint is empty."""
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(binding.scene, _raw(holding_payload=False))
    empty = adapter.LiberoAttachmentFact("empty", ())
    peak = adapter.LiberoAttachmentFact(
        "holding_other_movable",
        ("akita_black_bowl_2",),
        held_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
        ),
    )
    delta_lines = projection.render_interaction_delta_lines(
        binding, empty, peak, empty
    )
    delta_text = "\n".join(delta_lines).lower()
    assert "interaction grasp: none -> akita black bowl" in delta_text
    assert "held_non_goal_movable" in delta_text
    assert "-> none" in delta_text
    assert "released_non_goal_before_endpoint" in delta_text
    proj = projection.project_libero_fg(
        binding,
        snap,
        snap,
        attachment=empty,
        attachment_before=empty,
        attachment_peak=peak,
    )
    text = proj.combined_report.lower()
    assert "changed: interaction grasp: none -> akita black bowl" in text
    assert "held_non_goal_movable" in text
    assert "grasp: none" in text
    assert "released_non_goal_before_endpoint" in text
    assert "payload not firmly held by the gripper" not in text
    assert proj.executor_attachment == "empty"
    assert proj.executor_attachment_peak == "holding_other_movable"
    assert "held_non_goal_movable" in proj.interaction_grasp_peak


def test_released_goal_after_transport_not_acquisition() -> None:
    """Place-then-release with contact+XY miss is placement_xy, not empty grasp."""
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    # Near-miss: contact + z-order ok but XY outside 0.03 band.
    after = closed_gt.compile_snapshot(
        binding.scene,
        _raw(
            holding_payload=False,
            contact_payload_plate=True,
            payload_xy=(0.05, 0.0),
            z_payload=0.04,
        ),
    )
    empty = adapter.LiberoAttachmentFact("empty", ())
    peak = adapter.LiberoAttachmentFact("holding_payload", ("akita_black_bowl_1",))
    assert projection.released_goal_after_transport(
        peak=peak, after_attachment=empty, after=after
    )
    assert projection.classify_task_gap(binding, after) == "placement_xy"
    proj = projection.project_libero_fg(
        binding,
        after,
        after,
        attachment=empty,
        attachment_before=peak,
        attachment_peak=peak,
    )
    text = proj.combined_report.lower()
    assert proj.released_goal_after_transport is True
    assert proj.task_gap == "placement_xy"
    assert "released_goal_after_transport" in text
    assert "task gap: placement_xy" in text
    assert "fact: center_on_surface: on" in text
    assert "not center-to-center aligned" in text
    assert "center_to_center offset:" in text
    assert "placement near-miss" in text
    assert "center_to_center residual scale:" not in text
    assert "active placement work" not in text
    assert "payload not firmly held by the gripper" not in text
    assert "task gap: acquisition" not in text
    assert "center inside the footprint" not in text
    assert "laterally aligned with plate" not in text
    assert "at the same depth as plate" not in text
    assert "over the center of the support area" not in text


def test_acquisition_keeps_v213_placement_groups_without_nearmiss() -> None:
    """Empty-acq: v213 surface∥c2c facts; no residual/G1 noise; no near-miss."""
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    after = closed_gt.compile_snapshot(
        binding.scene,
        _raw(
            holding_payload=False,
            contact_payload_plate=False,
            payload_xy=(0.25, 0.25),
        ),
    )
    report, _gap = projection.render_endpoint_evidence_report(binding, after)
    lowered = report.lower()
    assert "verified placement surface group" in lowered
    assert "verified placement center-to-center group" in lowered
    assert "center_to_center residual scale:" not in lowered
    assert "active placement work" not in lowered
    assert "placement precision active: false" not in lowered
    assert "placement near-miss" not in lowered
    assert "task gap: acquisition" in lowered
    proj = projection.project_libero_fg(binding, after, after)
    assert proj.world_fg_complete is True


def test_released_non_goal_keeps_world_complete() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(binding.scene, _raw(holding_payload=False))
    empty = adapter.LiberoAttachmentFact("empty", ())
    peak = adapter.LiberoAttachmentFact(
        "holding_other_movable",
        ("akita_black_bowl_2",),
        held_vs_goal_rel_xyz=(
            "NEGATIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    proj = projection.project_libero_fg(
        binding,
        snap,
        snap,
        attachment=empty,
        attachment_before=peak,
        attachment_peak=peak,
    )
    text = proj.combined_report.lower()
    assert "released_non_goal_before_endpoint" in text
    assert "verified instance distinction" not in text
    assert "same_public_name_as_goal:" not in text
    assert "next acquisition target role" not in text
    assert proj.world_fg_complete is True


def test_holding_other_emits_same_name_xy_referent() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(binding.scene, _raw(holding_payload=False))
    # Prefer lateral separation (y) so chess priority anchors on left/right.
    # held_vs_goal: held is LEFT of goal (POSITIVE y) → goal is RIGHT of held.
    att = adapter.LiberoAttachmentFact(
        "holding_other_movable",
        ("akita_black_bowl_2",),
        held_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
        ),
        held_vs_goal_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    proj = projection.project_libero_fg(binding, snap, snap, attachment=att)
    text = proj.combined_report.lower()
    assert "verified wrong-grasp same-name acquire" in text
    assert (
        "the goal_object is the first akita black bowl "
        "from the robot's right" in text
    )
    assert "sample_grasp_pose" not in text
    assert "maps to:" not in text
    assert "righter" not in text
    assert "with respect to the robot" not in text
    assert "goal_object acquire language target:" not in text
    assert "role: held_non_goal_movable" in text
    assert "holding_non_goal" in text
    assert "grasped instance (akita black bowl)" in text
    assert "akita black bowl position relative to akita black bowl" not in text
    dom = projection.assert_fg_dominates_native_interaction(
        native_report=(
            "the robot's gripper is grasping the akita black bowl. "
            "picking up the akita black bowl has been completed."
        ),
        fg_report=proj.combined_report,
        attachment=att,
    )
    assert dom["applicable"] and dom["correctness_ok"] and dom["completeness_ok"]
    assert proj.world_fg_complete is True


def test_empty_hand_does_not_emit_wrong_grasp_api_map() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(binding.scene, _raw(holding_payload=False))
    att = adapter.LiberoAttachmentFact(
        "empty",
        (),
        same_name_other_raw="akita_black_bowl_2",
        goal_vs_same_name_other_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "NEGATIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    proj = projection.project_libero_fg(binding, snap, snap, attachment=att)
    text = proj.combined_report.lower()
    # Empty + sibling enters XY *logic* gate (not wrong-grasp hold gate).
    assert "wrong-grasp same-name acquire" not in text
    assert "verified same-name goal xy comparison" in text
    assert "the goal_object is the first akita black bowl from the robot's right" in text
    assert "sample_grasp_pose" not in text
    assert proj.world_fg_complete is True


def test_peak_holding_other_enters_xy_after_release() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(binding.scene, _raw(holding_payload=False))
    empty = adapter.LiberoAttachmentFact("empty", ())
    peak = adapter.LiberoAttachmentFact(
        "holding_other_movable",
        ("akita_black_bowl_2",),
        held_vs_goal_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    proj = projection.project_libero_fg(
        binding,
        snap,
        snap,
        attachment=empty,
        attachment_before=peak,
        attachment_peak=peak,
    )
    text = proj.combined_report.lower()
    assert "verified wrong-grasp same-name acquire" in text
    assert "released before endpoint" in text
    assert "the goal_object is the first akita black bowl from the robot's right" in text
    assert "sample_grasp_pose" not in text
    assert proj.world_fg_complete is True


def test_holding_other_different_name_skips_xy_referent() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    snap = closed_gt.compile_snapshot(binding.scene, _raw(holding_payload=False))
    att = adapter.LiberoAttachmentFact(
        "holding_other_movable",
        ("porcelain_ramekin_1",),
        held_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
        ),
        held_vs_goal_rel_xyz=(
            "NEGATIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    proj = projection.project_libero_fg(binding, snap, snap, attachment=att)
    text = proj.combined_report.lower()
    assert "wrong-grasp same-name acquire" not in text
    assert "sample_grasp_pose" not in text
    assert "holding_non_goal" in text
    assert proj.world_fg_complete is True


def test_ng_to_goal_transition_keeps_world_complete() -> None:
    """Δ may mention held_non_goal; endpoint holding_goal must stay complete."""
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    before = closed_gt.compile_snapshot(binding.scene, _raw(holding_payload=False))
    after = closed_gt.compile_snapshot(binding.scene, _raw(holding_payload=True))
    before_att = adapter.LiberoAttachmentFact(
        "holding_other_movable",
        ("akita_black_bowl_2",),
        held_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
            "POSITIVE_OF_REFERENCE_BAND",
        ),
        held_vs_goal_rel_xyz=(
            "NEGATIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    after_att = adapter.LiberoAttachmentFact(
        "holding_payload", ("akita_black_bowl_1",)
    )
    proj = projection.project_libero_fg(
        binding,
        before,
        after,
        attachment=after_att,
        attachment_before=before_att,
        attachment_peak=before_att,
    )
    text = proj.combined_report.lower()
    assert "held_non_goal_movable" in text  # may appear in Δ
    assert "interaction outcome: holding_goal_object" in text
    assert "interaction outcome: holding_non_goal" not in text
    assert proj.world_fg_complete is True


def test_nearmiss_emits_v213_center_offset_facts() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    after = closed_gt.compile_snapshot(
        binding.scene,
        _raw(
            holding_payload=False,
            contact_payload_plate=True,
            payload_xy=(0.05, 0.0),
            z_payload=0.04,
        ),
    )
    before = closed_gt.compile_snapshot(
        binding.scene,
        _raw(
            holding_payload=False,
            contact_payload_plate=True,
            payload_xy=(0.08, 0.0),
            z_payload=0.04,
        ),
    )
    proj = projection.project_libero_fg(binding, before, after)
    text = proj.combined_report.lower()
    assert "active placement work" not in text
    assert "center_to_center residual scale:" not in text
    assert "on_top_xy_band progress:" not in text
    assert "not center-to-center aligned" in text
    assert "center_to_center offset:" in text
    assert "placement near-miss" in text
    assert "task gap: placement_xy" in text
    assert proj.world_fg_complete is True
    assert proj.task_gap == "placement_xy"


def test_nearmiss_xy_emits_relxy_without_support_center_cnl() -> None:
    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    after = closed_gt.compile_snapshot(
        binding.scene,
        _raw(
            holding_payload=False,
            contact_payload_plate=True,
            payload_xy=(0.05, 0.0),
            z_payload=0.04,
        ),
    )
    before = closed_gt.compile_snapshot(
        binding.scene,
        _raw(
            holding_payload=False,
            contact_payload_plate=False,
            payload_xy=(0.05, 0.0),
            z_payload=0.02,
        ),
    )
    # Endpoint-only check.
    report, _gap = projection.render_endpoint_evidence_report(binding, after)
    lowered = report.lower()
    assert "fact: center_on_surface: on" in lowered
    assert "verified placement surface group" in lowered
    assert "verified placement center-to-center group" in lowered
    assert "task gap: placement_xy" in lowered
    assert "not center-to-center aligned" in lowered
    assert "center_to_center offset:" in lowered
    assert "placement near-miss" in lowered
    assert "center_to_center_xy is outside" in lowered
    assert "center_to_center residual scale:" not in lowered
    assert "active placement work" not in lowered
    assert "center inside the footprint" not in lowered
    assert "laterally aligned with plate" not in lowered
    assert "over the center of the support area" not in lowered
    assert "near-miss placement relative to the receptacle" not in lowered
    # World report (Δ + endpoint) must also suppress footprint→inside and
    # gripper/payload RelXY vs plate (regression: crashed smoke 519805).
    proj = projection.project_libero_fg(binding, before, after)
    world = proj.combined_report.lower()
    assert "center inside the footprint" not in world
    assert "position relative to plate along robot x" not in world
    assert "position relative to plate along robot y" not in world
    assert "laterally aligned with plate" not in world
    assert "at the same depth as plate" not in world or (
        "position relative to plate along robot x" not in world
        and "position relative to plate along robot y" not in world
    )
    assert "over the center of the support area" not in world
    assert proj.world_fg_complete is True
    projection.assert_placement_xy_report_mutex(proj.combined_report)
    native = (
        "The akita black bowl has been placed on the plate. "
        "Placing the akita black bowl on the plate has been completed."
    )
    place_dom = projection.assert_fg_dominates_native_placement(
        native_report=native,
        fg_report=proj.combined_report,
        binding=binding,
        after=after,
    )
    assert place_dom["applicable"] is True
    assert place_dom["correctness_ok"] is True
    assert place_dom["completeness_ok"] is True


def test_classify_skips_site_like_models() -> None:
    binding = adapter.bind_libero_g17_goal(
        _spatial_problem(), suite_name="libero_spatial", task_id=0
    )
    class SiteLike:
        pass  # no contact_geoms

    models = {
        "akita_black_bowl_1": SimpleNamespace(contact_geoms=["g1"]),
        "akita_black_bowl_2": SimpleNamespace(contact_geoms=["g2"]),
        "plate_1": SimpleNamespace(contact_geoms=["g3"]),
        "between_plate_ramekin_region": SiteLike(),
    }
    held = {"akita_black_bowl_2"}

    class Domain:
        objects_dict = {k: SimpleNamespace() for k in models if k != "between_plate_ramekin_region"}
        object_states_dict = {k: SimpleNamespace() for k in models}

        def get_object(self, name):
            return models[name]

        def _check_grasp(self, gripper, object_geoms):
            for name, model in models.items():
                if model is object_geoms:
                    return name in held
            return False

    fact = adapter.classify_executor_attachment(Domain(), gripper=object(), binding=binding)
    assert fact.kind == "holding_other_movable"
