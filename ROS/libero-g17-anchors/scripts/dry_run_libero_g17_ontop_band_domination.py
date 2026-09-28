#!/usr/bin/env python3
"""Dry-run: v2.11 on-top band domination (no footprint / coarse RelXY on placement_xy)."""

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


def _raw(
    *,
    contact_payload_plate: bool = False,
    payload_xy: tuple[float, float] = (0.0, 0.0),
    z_payload: float = 0.02,
) -> closed_gt.RawSceneState:
    import numpy as np

    object_contacts = (
        frozenset({frozenset({"obj01", "obj02"})})
        if contact_payload_plate
        else frozenset()
    )
    table_contacts = frozenset({"obj02"})
    if z_payload < 0.04:
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
        executor_aperture={"exe01": 1.0},
        holding=frozenset(),
        executor_contacts=frozenset(),
        object_contacts=object_contacts,
        table_contacts=table_contacts,
        table_height=0.0,
    )


def main() -> int:
    print("schema", takeover.schema_version())
    if "ontop-band-domination" not in takeover.schema_version():
        print("DRY_RUN_FAIL schema", file=sys.stderr)
        return 1
    binding = adapter.bind_libero_g17_goal(
        _problem(), suite_name="libero_spatial", task_id=0
    )
    empty = adapter.LiberoAttachmentFact("empty", ())
    goal_peak = adapter.LiberoAttachmentFact(
        "holding_payload", ("akita_black_bowl_1",)
    )
    before = closed_gt.compile_snapshot(
        binding.scene, _raw(payload_xy=(0.05, 0.0), z_payload=0.02)
    )
    near = closed_gt.compile_snapshot(
        binding.scene,
        _raw(contact_payload_plate=True, payload_xy=(0.05, 0.0), z_payload=0.04),
    )
    place = projection.project_libero_fg(
        binding,
        before,
        near,
        attachment=empty,
        attachment_before=goal_peak,
        attachment_peak=goal_peak,
    )
    text = place.combined_report.lower()
    checks = {
        "schema": "ontop-band-domination" in takeover.schema_version(),
        "task_gap_xy": place.task_gap == "placement_xy",
        "released_goal": place.released_goal_after_transport,
        "band_outside": "on_top_xy_band: outside" in text,
        "not_centered": "not centered on plate for on-top" in text,
        "offset_dir": "offset direction:" in text,
        "no_footprint": "center inside the footprint" not in text,
        "no_aligned_plate": "laterally aligned with plate" not in text,
        "no_same_depth_plate": "at the same depth as plate" not in text
        or "position relative to plate along robot" not in text,
        "no_acq": "payload not firmly held by the gripper" not in text,
        "world_complete": place.world_fg_complete,
        "finish_denied": "does not authorize finishing" in text,
    }
    # Tighten same-depth check: allow unrelated phrases, forbid Rel to plate.
    checks["no_same_depth_plate"] = (
        "position relative to plate along robot x: at the same depth as plate"
        not in text
        and "position relative to plate along robot y: laterally aligned with plate"
        not in text
    )
    print("--- checks ---")
    for k, v in checks.items():
        print(f"{k}={int(bool(v))}")
    if not all(checks.values()):
        print("DRY_RUN_FAIL", file=sys.stderr)
        print(place.combined_report[:2000])
        return 1
    print("DRY_RUN_OK ontop-band-domination placement_xy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
