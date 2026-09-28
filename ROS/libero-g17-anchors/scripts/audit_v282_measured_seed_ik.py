#!/usr/bin/env python3
"""Counterfactual audit of v282 on the exact v272 trial-2 grasp seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from capx.integrations.franka import measured_seed_ik as measured
from capx.integrations.motion import pyroki_snippets as pks
from capx.integrations.motion.pyroki_context import get_pyroki_context
from scripts.audit_v279_reachable_se3 import (
    GRASP_POSITION,
    GRASP_WXYZ,
    PLACE_WXYZ,
    PLATE_POSITION,
    RESET_CFG,
    _fk,
    _hand_target,
)


class _NativeIdealEnv:
    def __init__(self, api: SimpleNamespace) -> None:
        self.api = api
        self.commands: list[np.ndarray] = []

    def move_to_joints_blocking(self, joints: np.ndarray, *, max_steps: int) -> None:
        if max_steps != 480:
            raise AssertionError(f"unexpected native controller budget: {max_steps}")
        command = np.asarray(joints, dtype=np.float64).reshape(7)
        self.commands.append(command.copy())
        self.api.live[:7] = command


def _make_api() -> SimpleNamespace:
    ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
    api = SimpleNamespace(
        live=RESET_CFG.copy(),
        cfg=None,
        _robot=ctx.robot,
        _target_link_name=ctx.target_link_name,
        _pks=pks,
    )
    api._env = _NativeIdealEnv(api)
    api.get_observation = lambda: {"robot_joint_pos": api.live[:7].copy()}
    return api


def _solution_row(api: SimpleNamespace, cfg: np.ndarray, target: np.ndarray) -> dict:
    position, wxyz = _fk(api, cfg)
    return {
        "arm_cfg": np.asarray(cfg[:7], dtype=float).tolist(),
        "max_delta_from_reset_rad": float(
            np.max(np.abs(cfg[:7] - RESET_CFG[:7]))
        ),
        "position_error_m": float(np.linalg.norm(position - target)),
        "orientation_error_rad": measured._quaternion_angle_rad(wxyz, GRASP_WXYZ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    api = _make_api()
    grasp_rot = Rotation.from_quat(
        [GRASP_WXYZ[1], GRASP_WXYZ[2], GRASP_WXYZ[3], GRASP_WXYZ[0]]
    )
    grasp_goal = _hand_target(GRASP_POSITION, GRASP_WXYZ)
    grasp_approach = grasp_goal + grasp_rot.apply(np.array([0.0, 0.0, -0.1]))
    lift_goal = _hand_target(
        GRASP_POSITION + np.array([0.0, 0.0, 0.15]), GRASP_WXYZ
    )
    place_goal = _hand_target(
        PLATE_POSITION + np.array([0.0, 0.0, 0.20]), PLACE_WXYZ
    )

    unseeded = np.asarray(
        pks.solve_ik(
            robot=api._robot,
            target_link_name=api._target_link_name,
            target_position=grasp_approach,
            target_wxyz=GRASP_WXYZ,
        ),
        dtype=np.float64,
    )
    seeded = np.asarray(
        pks.solve_ik_seeded(
            robot=api._robot,
            target_link_name=api._target_link_name,
            target_position=grasp_approach,
            target_wxyz=GRASP_WXYZ,
            initial_cfg=RESET_CFG,
        ),
        dtype=np.float64,
    )
    old_row = _solution_row(api, unseeded, grasp_approach)
    new_row = _solution_row(api, seeded, grasp_approach)

    original_fk = measured.fk_panda_hand
    measured.fk_panda_hand = _fk
    try:
        stages = []
        for name, position, wxyz in (
            ("grasp_approach", grasp_approach, GRASP_WXYZ),
            ("grasp_goal", grasp_goal, GRASP_WXYZ),
            ("grasp_lift", lift_goal, GRASP_WXYZ),
            ("place_above", place_goal, PLACE_WXYZ),
        ):
            before = len(api._env.commands)
            result = measured.move_hand_pose_measured_seeded(
                api, position, wxyz, log=False
            )
            stages.append(
                {
                    "stage": name,
                    "ok": result.ok,
                    "execution_commands": len(api._env.commands) - before,
                    "candidate_arm_delta_rad": result.candidate_arm_delta_rad,
                    "final_position_error_m": result.final_position_error_m,
                    "final_orientation_error_rad": result.final_orientation_error_rad,
                    "message": result.message,
                }
            )
    finally:
        measured.fk_panda_hand = original_fk

    contract = measured.measured_seed_ik_contract()
    accepted = bool(
        new_row["max_delta_from_reset_rad"] < old_row["max_delta_from_reset_rad"]
        and new_row["position_error_m"] <= 1e-4
        and new_row["orientation_error_rad"] <= 1e-4
        and len(api._env.commands) == 4
        and all(row["ok"] and row["execution_commands"] == 1 for row in stages)
        and all(row["final_position_error_m"] <= 1e-4 for row in stages)
        and all(row["final_orientation_error_rad"] <= 1e-4 for row in stages)
        and contract["cartesian_waypoints"] == 0
        and contract["joint_delta_gate"] == 0
        and contract["terminal_residual_gate"] == 0
    )
    data = {
        "schema": "capx_v282_measured_seed_ik_audit_v1",
        "source": {"job_id": 523469, "task_id": 0, "replicate": 1, "trial": 2},
        "first_grasp_approach_counterfactual": {
            "legacy_unseeded": old_row,
            "measured_seeded": new_row,
        },
        "native_execution": {
            "accepted": accepted,
            "total_stages": 4,
            "total_blocking_commands": len(api._env.commands),
            "stages": stages,
        },
        "contract": contract,
    }
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if not accepted:
        raise SystemExit("v282 measured-seed audit failed")


if __name__ == "__main__":
    main()
