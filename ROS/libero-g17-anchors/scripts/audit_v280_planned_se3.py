#!/usr/bin/env python3
"""Exact v272 trial-2 replay with a deliberately imperfect joint controller."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from capx.integrations.franka import planned_se3 as planned
from capx.integrations.franka import trusted_se3 as trusted
from scripts.audit_v279_reachable_se3 import (
    GRASP_POSITION,
    GRASP_WXYZ,
    PLACE_WXYZ,
    PLATE_POSITION,
    _fk,
    _hand_target,
    _make_api,
)


class _LaggedEnv:
    """92% joint settling exposes plan/execution representation mismatches."""

    def __init__(self, api: object, gain: float = 0.92) -> None:
        self.api = api
        self.gain = float(gain)
        self.history: list[np.ndarray] = []
        self.command_history: list[np.ndarray] = []

    def move_to_joints_blocking(self, joints: np.ndarray, *, max_steps: int) -> None:
        assert max_steps == 40
        command = np.asarray(joints, dtype=np.float64)
        before = self.api.live.copy()
        self.api.live[:7] += self.gain * (command - self.api.live[:7])
        self.api.cfg = self.api.live.copy()
        self.command_history.append(command.copy())
        self.history.append(self.api.live.copy())
        self.api.executed_max_dq = max(
            self.api.executed_max_dq,
            float(np.max(np.abs(self.api.live[:7] - before[:7]))),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--joint-join", action="store_true")
    args = parser.parse_args()
    if args.joint_join:
        os.environ["CAPX_LIBERO_G17_V281_JOINT_JOIN"] = "1"

    original_trusted_fk = trusted.fk_panda_hand
    original_planned_fk = planned.fk_panda_hand
    trusted.fk_panda_hand = _fk
    planned.fk_panda_hand = _fk
    try:
        api = _make_api()
        api._env = _LaggedEnv(api)
        rot = Rotation.from_quat(
            [GRASP_WXYZ[1], GRASP_WXYZ[2], GRASP_WXYZ[3], GRASP_WXYZ[0]]
        )
        grasp_goal = _hand_target(GRASP_POSITION, GRASP_WXYZ)
        grasp_approach = grasp_goal + rot.apply(np.array([0.0, 0.0, -0.1]))
        lift_goal = _hand_target(
            GRASP_POSITION + np.array([0.0, 0.0, 0.15]), GRASP_WXYZ
        )
        place_goal = _hand_target(
            PLATE_POSITION + np.array([0.0, 0.0, 0.20]), PLACE_WXYZ
        )

        prefix = []
        for name, position in (
            ("grasp_approach", grasp_approach),
            ("grasp_goal", grasp_goal),
            ("grasp_lift", lift_goal),
        ):
            result = planned.move_hand_pose_planned(
                api, position, GRASP_WXYZ, log=False
            )
            prefix.append({"stage": name, "ok": result.ok, "message": result.message})
            if not result.ok:
                raise SystemExit(f"lagged prefix failed at {name}: {result.message}")

        before_place_cfg = api.live.copy()
        commands_before = len(api._env.command_history)
        straight = trusted.plan_hand_pose_trusted(
            api, place_goal, PLACE_WXYZ, start_cfg=before_place_cfg
        )
        if straight.ok:
            raise SystemExit("v272 placement dead end was not reproduced")
        if len(api._env.command_history) != commands_before:
            raise SystemExit("preflight sent a controller command")

        result = planned.move_hand_pose_planned(
            api, place_goal, PLACE_WXYZ, log=False
        )
        final_position, final_wxyz = _fk(api, api.live)
        final_pos_err = float(np.linalg.norm(final_position - place_goal))
        final_ori_err = trusted._quaternion_angle_rad(final_wxyz, PLACE_WXYZ)
        commands = np.asarray(api._env.command_history)
        command_path = np.vstack([before_place_cfg[:7], commands[commands_before:]])
        max_command_delta = float(
            np.max(np.abs(np.diff(command_path, axis=0)))
        )
        ok = bool(
            result.ok
            and result.status == "recovered"
            and str(result.selected_route).startswith("neutral_")
            and final_pos_err <= 0.007
            and final_ori_err <= 0.030
            and max_command_delta <= 0.180001
            and api.executed_max_dq <= 0.2002
        )
        data = {
            "schema": (
                "capx_v281_lagged_joint_join_audit_v1"
                if args.joint_join
                else "capx_v280_lagged_planned_se3_audit_v1"
            ),
            "source": {"job_id": 523469, "task_id": 0, "replicate": 1, "trial": 2},
            "controller": {"kind": "joint_first_order_lag", "settling_gain": 0.92},
            "prefix": prefix,
            "straight_preflight": {
                "ok": straight.ok,
                "status": straight.status,
                "waypoints": len(straight.waypoint_cfgs),
                "message": straight.message,
                "executed_commands": 0,
            },
            "planned": {
                "ok": ok,
                "status": result.status,
                "selected_route": result.selected_route,
                "rollback_performed": result.rollback_performed,
                "stages": [name for name, _ in result.stages],
                "max_commanded_arm_delta_rad": max_command_delta,
                "max_measured_arm_delta_rad": api.executed_max_dq,
                "final_position_error_m": final_pos_err,
                "final_orientation_error_rad": final_ori_err,
                "message": result.message,
            },
            "contract": (
                planned.command_faithful_join_contract()
                if args.joint_join
                else planned.planned_se3_contract()
            ),
        }
    finally:
        trusted.fk_panda_hand = original_trusted_fk
        planned.fk_panda_hand = original_planned_fk

    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if not data["planned"]["ok"]:
        raise SystemExit("v280 lagged mechanical replay failed")


if __name__ == "__main__":
    main()
