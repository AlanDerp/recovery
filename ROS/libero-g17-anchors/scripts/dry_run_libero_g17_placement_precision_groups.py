#!/usr/bin/env python3
"""Dry-run: v2.13 placement groups + v2.19 R(goal) schema + dual-dom."""

from __future__ import annotations

import sys

import numpy as np

from capx.verification import closed_gt
from capx.verification import libero_g17_adapter as adapter
from capx.verification import libero_g17_fg_projection as projection
from capx.verification import libero_g17_takeover as takeover


def _problem() -> dict:
    return {
        "fixtures": {"table": ["main_table"]},
        "objects": {
            "akita_black_bowl": ["akita_black_bowl_1", "akita_black_bowl_2"],
            "plate": ["plate_1"],
        },
        "goal_state": [["On", "akita_black_bowl_1", "plate_1"]],
    }


def _raw(*, payload_xy: tuple[float, float], contact: bool) -> closed_gt.RawSceneState:
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
                position=np.array([payload_xy[0], payload_xy[1], 0.04]),
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
        executor_aperture={"exe01": 0.0},
        holding=frozenset(),
        executor_contacts=frozenset(),
        object_contacts=(
            frozenset({frozenset({"obj01", "obj02"})}) if contact else frozenset()
        ),
        table_contacts=frozenset({"obj01", "obj02"}),
        table_height=0.0,
    )


def main() -> int:
    print("schema", takeover.schema_version())
    schema = takeover.schema_version()
    if not any(
        k in schema
        for k in (
            "wrong-grasp-ordinal-xy",
            "same-name-xy-logic",
            "placement-xy-instance",
            "placement-silent-xy",
            "ordinal-sgp-overload",
            "sgp-overload-no-ordinal",
            "band-obligations",
            "band-contract",
            "task-gap-aligned",
            "post-miss-clear",
            "ordinal-sgp-overload",
            "placement-precision-groups",
            "target-grasp-object",
            "goal-referent-takeover",
            "goal-referent-bind",
            "goal-identity-lock",
            "goal-identity-verify",
        )
    ):
        print("DRY_RUN_FAIL schema", file=sys.stderr)
        return 1
    binding = adapter.bind_libero_g17_goal(
        _problem(), suite_name="libero_spatial", task_id=0
    )
    after = closed_gt.compile_snapshot(
        binding.scene, _raw(payload_xy=(0.05, 0.0), contact=True)
    )
    before = closed_gt.compile_snapshot(
        binding.scene, _raw(payload_xy=(0.08, 0.0), contact=True)
    )
    peak = adapter.LiberoAttachmentFact("holding_payload", ("akita_black_bowl_1",))
    empty = adapter.LiberoAttachmentFact("empty", ())
    proj = projection.project_libero_fg(
        binding,
        before,
        after,
        attachment=empty,
        attachment_before=peak,
        attachment_peak=peak,
    )
    text = proj.combined_report.lower()
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
    far = closed_gt.compile_snapshot(
        binding.scene, _raw(payload_xy=(0.25, 0.25), contact=False)
    )
    pre = projection.project_libero_fg(binding, far, far)
    pre_t = pre.combined_report.lower()
    checks = {
        "task_gap_placement_xy": proj.task_gap == "placement_xy",
        "surface_group": "verified placement surface group" in text,
        "center_group": "verified placement center-to-center group" in text,
        "not_aligned": "not center-to-center aligned" in text,
        "offset": "center_to_center offset:" in text,
        "near_miss": "placement near-miss" in text,
        "no_active_work": "active placement work" not in text,
        "no_residual": "center_to_center residual scale:" not in text,
        "no_progress": "on_top_xy_band progress:" not in text,
        "world_complete": proj.world_fg_complete,
        "dom_ok": place_dom["applicable"]
        and place_dom["correctness_ok"]
        and place_dom["completeness_ok"],
        "acq_has_groups": "verified placement surface group" in pre_t,
        "acq_no_nearmiss": "placement near-miss" not in pre_t,
        "acq_no_residual": "center_to_center residual scale:" not in pre_t,
        "acq_complete": pre.world_fg_complete,
        "no_instance": "verified instance distinction" not in text,
        "no_same_public": "same_public_name_as_goal:" not in text,
    }
    print("--- checks ---")
    for k, v in checks.items():
        print(f"{k}={int(bool(v))}")
    if not all(checks.values()):
        print("DRY_RUN_FAIL", file=sys.stderr)
        print(proj.combined_report[:2500])
        return 1
    print("DRY_RUN_OK v213 placement groups + v223 wrong-grasp ordinal schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
