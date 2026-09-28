#!/usr/bin/env python3
"""Dry-run: distractor grasp-then-release must appear in ordered Δ (not endpoint-only)."""

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


def _empty_raw() -> closed_gt.RawSceneState:
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
        executor_aperture={"exe01": 1.0},
        holding=frozenset(),
        executor_contacts=frozenset(),
        object_contacts=frozenset(),
        table_contacts=frozenset({"obj01", "obj02"}),
        table_height=0.0,
    )


def main() -> int:
    print("schema", takeover.schema_version())
    binding = adapter.bind_libero_g17_goal(
        _problem(), suite_name="libero_spatial", task_id=0
    )
    snap = closed_gt.compile_snapshot(binding.scene, _empty_raw())
    empty = adapter.LiberoAttachmentFact("empty", ())
    peak = adapter.LiberoAttachmentFact(
        "holding_other_movable", ("akita_black_bowl_2",)
    )
    proj = projection.project_libero_fg(
        binding,
        snap,
        snap,
        attachment=empty,
        attachment_before=empty,
        attachment_peak=peak,
    )
    text = proj.combined_report
    print("--- report (head) ---")
    print("\n".join(text.splitlines()[:28]))
    checks = {
        "delta_none_to_other": (
            "changed: interaction grasp: none -> another akita black bowl"
            in text.lower()
        ),
        "delta_other_to_none": "-> none." in text.lower()
        and "interaction grasp: another akita black bowl" in text.lower(),
        "endpoint_none": "- grasp: none." in text.lower(),
        "released_non_goal": "released_non_goal_before_endpoint" in text.lower(),
        "no_null": "null" not in text.lower(),
        "no_bare_empty_acq": "payload not firmly held by the gripper" not in text.lower(),
        "peak_other": proj.executor_attachment_peak == "holding_other_movable",
        "world_complete": proj.world_fg_complete,
        "fact_slots_schema": "ontop-band-domination" in takeover.schema_version()
        or "fact-slots" in takeover.schema_version(),
    }
    print("--- checks ---")
    for k, v in checks.items():
        print(f"{k}={int(bool(v))}")
    if not all(checks.values()):
        print("DRY_RUN_FAIL", file=sys.stderr)
        return 1
    print("DRY_RUN_OK grasp-then-release in ordered Δ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
