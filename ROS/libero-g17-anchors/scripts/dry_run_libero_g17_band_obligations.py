#!/usr/bin/env python3
"""Dry-run: band obligation mini predicate contract on placement_xy near-miss."""

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


def _raw(
    *,
    payload_xy: tuple[float, float],
    contact: bool,
    holding_payload: bool = False,
) -> closed_gt.RawSceneState:
    object_contacts = (
        frozenset({frozenset({"obj01", "obj02"})}) if contact else frozenset()
    )
    holding = frozenset({("exe01", "obj01")}) if holding_payload else frozenset()
    executor_contacts = (
        frozenset({("exe01", "obj01")}) if holding_payload else frozenset()
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
                position=np.array([payload_xy[0], payload_xy[1], 0.04]),
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
        executor_aperture={"exe01": 0.0 if holding_payload else 1.0},
        holding=holding,
        executor_contacts=executor_contacts,
        object_contacts=object_contacts,
        table_contacts=frozenset({"obj01", "obj02"}),
        table_height=0.0,
    )


def main() -> int:
    os.environ.setdefault("CAPX_LIBERO_G17_BAND_OBLIGATIONS", "1")
    os.environ.setdefault("CAPX_LIBERO_G17_XY_LOCK", "1")
    os.environ.setdefault("CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL", "1")
    print("schema", takeover.schema_version())
    print("band_obligations", os.environ.get("CAPX_LIBERO_G17_BAND_OBLIGATIONS"))
    binding = adapter.bind_libero_g17_goal(
        _problem(), suite_name="libero_spatial", task_id=0
    )
    after = closed_gt.compile_snapshot(
        binding.scene,
        _raw(payload_xy=(0.05, 0.04), contact=True, holding_payload=True),
    )
    peak = adapter.LiberoAttachmentFact("holding_payload", ("akita_black_bowl_1",))
    holding = adapter.LiberoAttachmentFact("holding_payload", ("akita_black_bowl_1",))
    proj = projection.project_libero_fg(
        binding,
        after,
        after,
        attachment=holding,
        attachment_before=peak,
        attachment_peak=peak,
    )
    text = proj.combined_report.lower()
    checks = {
        "task_gap_placement_xy": proj.task_gap == "placement_xy",
        "near_miss": "placement near-miss" in text,
        "contract_block": "instantiated predicate contract:" in text,
        "current_required": "current" in text and "required" in text,
        "not_satisfied": "not satisfied" in text,
        "holding_terminal": "terminal factual conditions:" in text,
        "no_arrow_transition": "→" not in proj.combined_report,
        "no_replace_note": "before other placement" not in text,
        "no_api": "sample_grasp_pose" not in text and "which=" not in text,
        "world_complete": proj.world_fg_complete is True,
        "surface_group": "verified placement surface group" in text,
        "c2c_group": "verified placement center-to-center group" in text,
        "schema_v239": "band-contract" in takeover.schema_version(),
    }
    print("--- checks ---")
    for k, v in checks.items():
        print(f"{k}={int(bool(v))}")
    if not all(checks.values()):
        print("DRY_RUN_FAIL", file=sys.stderr)
        print(proj.combined_report[:2500])
        return 1
    print("DRY_RUN_OK band obligation mini predicate contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
