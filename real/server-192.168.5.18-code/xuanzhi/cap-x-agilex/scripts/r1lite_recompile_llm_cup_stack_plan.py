#!/usr/bin/env python3
"""Recompile an approved LLM intent with the current deterministic policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.cup_stack_llm import compile_llm_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--loaded-transfer-x-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--second-phase-low-transfer-fallback",
        action="store_true",
    )
    parser.add_argument(
        "--all-phases-low-transfer-fallback",
        action="store_true",
    )
    parser.add_argument(
        "--reuse-saved-support-joints-for-all-phases",
        action="store_true",
    )
    parser.add_argument(
        "--reuse-saved-support-joints-for-orange-only",
        action="store_true",
    )
    parser.add_argument(
        "--green-above-support-minimum-z-m",
        type=float,
    )
    parser.add_argument("--saved-above-support", type=Path)
    parser.add_argument("--saved-release-hover", type=Path)
    args = parser.parse_args()

    source = json.loads(args.source_plan.read_text())
    plan = compile_llm_plan(
        source["llm_intent"],
        source["rgbd_scene"],
        cup_height_m=float(source["cup_height_m"]),
        cup_diameter_m=float(source["cup_diameter_m"]),
        gripper_close_position=float(source["gripper_close_position"]),
    )
    if args.loaded_transfer_x_offset_m:
        for phase in plan["phases"]:
            above_support = next(
                item
                for item in phase["waypoints"]
                if item["id"] == "above_support"
            )
            transfer = list(above_support["tcp_position_base_m"])
            transfer[0] += float(args.loaded_transfer_x_offset_m)
            phase["loaded_transfer_tcp_position_base_m"] = transfer
            phase["loaded_transfer_offset_xy_m"] = [
                float(args.loaded_transfer_x_offset_m),
                0.0,
            ]
            phase["loaded_transfer_policy"] = (
                "ik_scanned_high_staging_then_release_hover"
            )
    if args.second_phase_low_transfer_fallback:
        second = plan["phases"][1]
        second["loaded_transfer_tcp_position_base_m"] = list(
            second["release_hover_tcp_position_base_m"]
        )
        second["loaded_transfer_offset_xy_m"] = [0.0, 0.0]
        second["loaded_transfer_policy"] = (
            "five_centimeter_lift_then_direct_release_fallback"
        )
    if args.all_phases_low_transfer_fallback:
        for phase in plan["phases"]:
            phase["loaded_transfer_tcp_position_base_m"] = list(
                phase["release_hover_tcp_position_base_m"]
            )
            phase["loaded_transfer_offset_xy_m"] = [0.0, 0.0]
            phase["loaded_transfer_policy"] = (
                "five_centimeter_lift_then_direct_release_fallback"
            )
    if (
        args.reuse_saved_support_joints_for_all_phases
        or args.reuse_saved_support_joints_for_orange_only
    ):
        if args.saved_above_support is None or args.saved_release_hover is None:
            raise ValueError(
                "saved above_support and release_hover files are required"
            )
        saved_above = json.loads(args.saved_above_support.read_text())
        saved_release = json.loads(args.saved_release_hover.read_text())
        phases = (
            plan["phases"]
            if args.reuse_saved_support_joints_for_all_phases
            else [plan["phases"][0]]
        )
        for phase in phases:
            phase["saved_stage_joint_positions_rad"] = {
                "above_support": list(
                    saved_above["measured_joint_position_rad"]
                ),
                "release_hover": list(
                    saved_release["measured_joint_position_rad"]
                ),
            }
            phase["saved_stage_joint_position_policy"] = (
                "reuse_orange_measured_support_joints"
            )
            phase["saved_stage_joint_position_sources"] = {
                "above_support": str(args.saved_above_support.resolve()),
                "release_hover": str(args.saved_release_hover.resolve()),
            }
    if args.green_above_support_minimum_z_m is not None:
        minimum_z = float(args.green_above_support_minimum_z_m)
        if minimum_z < 0.95:
            raise ValueError(
                "green above_support minimum Z must be at least 0.95m"
            )
        green_phase = plan["phases"][1]
        above_support = next(
            item
            for item in green_phase["waypoints"]
            if item["id"] == "above_support"
        )
        above_support["tcp_position_base_m"][2] = max(
            float(above_support["tcp_position_base_m"][2]),
            minimum_z,
        )
        green_phase["minimum_loaded_transport_tcp_z_m"] = minimum_z
        green_phase["above_support_joint_policy"] = (
            "regenerate_with_online_ik_each_execution"
        )
        green_phase["loaded_transfer_orientation_policy"] = (
            "free_per_waypoint_topdown_at_above_support"
        )
        green_phase["loaded_transfer_route_policy"] = (
            "mandatory_rear_retract_lift_rotate_then_advance"
        )
        green_phase["loaded_transfer_rear_x_offset_m"] = -0.16
        green_phase["loaded_transfer_transit_tcp_z_m"] = 1.10
        green_phase["release_policy"] = (
            "gravity_drop_directly_from_above_support"
        )
        green_phase["release_trigger_policy"] = (
            "xy_aligned_before_above_support"
        )
        green_phase["release_maximum_xy_error_m"] = 0.01
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    )
    print(args.output_plan.resolve())


if __name__ == "__main__":
    main()
