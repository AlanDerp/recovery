#!/usr/bin/env python3
"""Dry-run: On task_gap aligns with acquire / endpoint_gap (v238 gold case)."""

from __future__ import annotations

import os
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


def _raw_holding_no_plate_contact() -> closed_gt.RawSceneState:
    """v238 rep-02 t2 turn_02 shape: holding goal, no plate contact."""
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
                position=np.array([-0.04, 0.02, 0.08]),
                orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                half_extents=np.array([0.04, 0.04, 0.02]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            ),
            "obj02": closed_gt.BodyState(
                position=np.array([0.0, 0.0, 0.01]),
                orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                half_extents=np.array([0.08, 0.08, 0.01]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            ),
        },
        executor_aperture={"exe01": 0.2},
        holding=frozenset({("exe01", "obj01")}),
        executor_contacts=frozenset({("exe01", "obj01")}),
        object_contacts=frozenset(),
        table_contacts=frozenset({"obj01", "obj02"}),
        table_height=0.0,
    )


def main() -> int:
    os.environ.setdefault("CAPX_LIBERO_G17_XY_LOCK", "1")
    os.environ.setdefault("CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL", "1")
    os.environ.pop("CAPX_LIBERO_G17_BAND_OBLIGATIONS", None)
    print("schema", takeover.schema_version())
    binding = adapter.bind_libero_g17_goal(
        _problem(), suite_name="libero_spatial", task_id=0
    )
    after = closed_gt.compile_snapshot(
        binding.scene, _raw_holding_no_plate_contact()
    )
    holding = adapter.LiberoAttachmentFact(
        "holding_payload", ("akita_black_bowl_1",)
    )
    proj = projection.project_libero_fg(
        binding,
        after,
        after,
        attachment=holding,
        attachment_before=holding,
        attachment_peak=holding,
    )
    text = proj.combined_report.lower()
    checks = {
        "schema_v240": "task-gap-aligned" in takeover.schema_version(),
        "endpoint_placement": proj.endpoint_gap == "placement_incomplete",
        "task_placement_other": proj.task_gap == "placement_other",
        "public_task_gap": "task gap: placement_other" in text,
        "no_public_acquisition_gap": "task gap: acquisition" not in text,
        "holding_outcome": "holding_goal_object" in text,
        "holding_true": "holding akita black bowl: true" in text,
        "plate_contact_false": "contact between akita black bowl and plate: false"
        in text,
        "placement_note": "placement evidence is incomplete" in text,
        "no_acq_empty_note": "payload not firmly held by the gripper" not in text,
    }
    print("--- checks ---")
    for k, v in checks.items():
        print(f"{k}={int(bool(v))}")
    if not all(checks.values()):
        print("DRY_RUN_FAIL", file=sys.stderr)
        print(proj.combined_report[:2000])
        return 1
    print("DRY_RUN_OK task_gap aligned with acquire/endpoint_gap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
