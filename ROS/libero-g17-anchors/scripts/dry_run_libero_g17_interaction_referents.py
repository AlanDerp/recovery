#!/usr/bin/env python3
"""Dry-run: same-name ordinal w.r.t. robot (first-from-right; no API naming)."""

from __future__ import annotations

import sys

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


def _raw() -> closed_gt.RawSceneState:
    import numpy as np

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
                position=np.array([0.0, 0.0, 0.02]),
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
        object_contacts=frozenset(),
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
            "ordinal-sgp-task-gap-aligned",
            "ordinal-sgp-post-miss-clear",
            "sgp-overload-no-ordinal",
            "band-contract",
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
    snap = closed_gt.compile_snapshot(binding.scene, _raw())
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
    native = (
        "In the current state, the robot's gripper is closed and is grasping "
        "the akita black bowl. The task of picking up the akita black bowl "
        "has been completed. The next step is to place it on the plate."
    )
    dom = projection.assert_fg_dominates_native_interaction(
        native_report=native, fg_report=proj.combined_report, attachment=att
    )
    empty = adapter.LiberoAttachmentFact("empty", ())
    pre = projection.project_libero_fg(binding, snap, snap, attachment=empty)
    pre_t = pre.combined_report.lower()
    light = projection.light_target_grasp_object_enabled()
    checks = {
        "wrong_grasp_block": "verified wrong-grasp same-name acquire" in text,
        "ordinal_right": (
            "the goal_object is the first akita black bowl "
            "from the robot's right" in text
        ),
        "no_api_name": "sample_grasp_pose" not in text,
        "no_righter": "righter" not in text,
        "role_token": "role: held_non_goal_movable" in text,
        "holding_non_goal": "holding_non_goal" in text,
        "world_complete": proj.world_fg_complete,
        "dom_ok": dom["applicable"]
        and dom["correctness_ok"]
        and dom["completeness_ok"],
        "empty_no_block": "wrong-grasp same-name acquire" not in pre_t,
        "empty_complete": pre.world_fg_complete,
        "light_fact": (not light)
        or ("target grasp object:" in text),
    }
    print("--- checks ---")
    for k, v in checks.items():
        print(f"{k}={int(bool(v))}")
    if not all(checks.values()):
        print("DRY_RUN_FAIL", file=sys.stderr)
        print(proj.combined_report[:2500])
        return 1
    print("DRY_RUN_OK wrong-grasp ordinal / light-target / goal-takeover schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
