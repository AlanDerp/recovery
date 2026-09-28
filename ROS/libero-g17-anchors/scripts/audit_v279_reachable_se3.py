#!/usr/bin/env python3
"""Exact mechanical replay of the v272/v278 task-0 trial-2 dead-end path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import jaxlie
import numpy as np
from scipy.spatial.transform import Rotation

from capx.integrations.franka import reachable_se3 as reach
from capx.integrations.franka import trusted_se3 as trusted
from capx.integrations.motion.pyroki_context import get_pyroki_context
from capx.integrations.motion.pyroki_snippets._solve_ik_vel_cost import solve_ik_local


RESET_CFG = np.array(
    [0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, np.pi / 4, 0.02]
)
GRASP_POSITION = np.array([0.64733738, 0.19962308, 0.02439878])
GRASP_WXYZ = np.array([-0.34075639, 0.60183579, -0.62495569, -0.36208998])
PLATE_POSITION = np.array([0.72179002, 0.20976031, 0.00102008])
PLACE_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])


class _LocalSolver:
    def solve_ik_local(self, **kwargs: object) -> np.ndarray:
        return solve_ik_local(
            kwargs["robot"],
            str(kwargs["target_link_name"]),
            np.asarray(kwargs["target_wxyz"], dtype=np.float64),
            np.asarray(kwargs["target_position"], dtype=np.float64),
            np.asarray(kwargs["prev_cfg"], dtype=np.float64),
            max_arm_delta=float(kwargs["max_arm_delta"]),
        )


class _IdealEnv:
    """Ideal actuator: isolates IK/path mechanics from simulator control error."""

    def __init__(self, api: SimpleNamespace) -> None:
        self.api = api
        self.home_joint_position = RESET_CFG[:7].copy()
        self.history: list[np.ndarray] = []

    def move_to_joints_blocking(self, joints: np.ndarray, *, max_steps: int) -> None:
        assert max_steps == 40
        before = self.api.live.copy()
        self.api.live[:7] = np.asarray(joints, dtype=np.float64)
        self.api.cfg = self.api.live.copy()
        self.history.append(self.api.live.copy())
        self.api.executed_max_dq = max(
            self.api.executed_max_dq,
            float(np.max(np.abs(self.api.live[:7] - before[:7]))),
        )


def _make_api() -> SimpleNamespace:
    ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
    api = SimpleNamespace(
        live=RESET_CFG.copy(),
        cfg=RESET_CFG.copy(),
        _robot=ctx.robot,
        _target_link_name=ctx.target_link_name,
        _pks=_LocalSolver(),
        executed_max_dq=0.0,
    )
    api._env = _IdealEnv(api)
    api.get_observation = lambda: {"robot_joint_pos": api.live[:7].copy()}
    return api


def _fk(api: SimpleNamespace, cfg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    link_i = api._robot.links.names.index(api._target_link_name)
    pose = jaxlie.SE3(
        api._robot.forward_kinematics(jnp.asarray(cfg, dtype=jnp.float32))[link_i]
    )
    return (
        np.asarray(pose.translation(), dtype=np.float64),
        np.asarray(pose.rotation().wxyz, dtype=np.float64),
    )


def _hand_target(tip_position: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
    rot = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    return tip_position + rot.apply(np.array([0.0, 0.0, -0.1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # Bind the production modules to the exact same FK used by this replay.
    original_trusted_fk = trusted.fk_panda_hand
    original_reach_fk = reach.fk_panda_hand
    trusted.fk_panda_hand = _fk
    reach.fk_panda_hand = _fk
    try:
        api = _make_api()
        grasp_rot = Rotation.from_quat(
            [GRASP_WXYZ[1], GRASP_WXYZ[2], GRASP_WXYZ[3], GRASP_WXYZ[0]]
        )
        grasp_goal = _hand_target(GRASP_POSITION, GRASP_WXYZ)
        grasp_approach = grasp_goal + grasp_rot.apply(np.array([0.0, 0.0, -0.1]))
        lift_tip = GRASP_POSITION + np.array([0.0, 0.0, 0.15])
        lift_goal = _hand_target(lift_tip, GRASP_WXYZ)
        place_tip_above = PLATE_POSITION + np.array([0.0, 0.0, 0.20])
        place_goal = _hand_target(place_tip_above, PLACE_WXYZ)

        prefix = []
        for name, position in (
            ("grasp_approach", grasp_approach),
            ("grasp_goal", grasp_goal),
            ("grasp_lift", lift_goal),
        ):
            result = trusted.move_hand_pose_trusted(
                api, position, GRASP_WXYZ, log=False
            )
            prefix.append({"stage": name, "ok": result.ok, "message": result.message})
            if not result.ok:
                raise SystemExit(f"exact prefix failed at {name}: {result.message}")

        preplacement_cfg = api.live.copy()
        history_before = len(api._env.history)
        straight = trusted.plan_hand_pose_trusted(
            api, place_goal, PLACE_WXYZ, start_cfg=preplacement_cfg
        )
        if len(api._env.history) != history_before:
            raise SystemExit("straight preflight mutated the actuator")
        if straight.ok:
            raise SystemExit("exact dead-end was not reproduced")

        result = reach.move_hand_pose_reachable(
            api, place_goal, PLACE_WXYZ, log=False
        )
        final_position, final_wxyz = _fk(api, api.live)
        final_pos_err = float(np.linalg.norm(final_position - place_goal))
        final_ori_err = trusted._quaternion_angle_rad(final_wxyz, PLACE_WXYZ)
        ok = bool(
            result.ok
            and result.selected_route is not None
            and result.selected_route.startswith("neutral_")
            and result.primary is None
            and result.primary_plan is not None
            and not result.primary_plan.ok
            and api.executed_max_dq <= 0.2002
            and final_pos_err <= 0.007
            and final_ori_err <= 0.030
        )
        data = {
            "schema": "capx_v279_exact_reachable_se3_audit_v1",
            "source": {"job_id": 523469, "task_id": 0, "replicate": 1, "trial": 2},
            "inputs": {
                "reset_cfg": RESET_CFG.tolist(),
                "grasp_position": GRASP_POSITION.tolist(),
                "grasp_wxyz": GRASP_WXYZ.tolist(),
                "plate_position": PLATE_POSITION.tolist(),
                "place_wxyz": PLACE_WXYZ.tolist(),
                "place_hand_target": place_goal.tolist(),
            },
            "prefix": prefix,
            "straight_preflight": {
                "ok": straight.ok,
                "status": straight.status,
                "waypoints": len(straight.waypoint_cfgs),
                "rejections": straight.preflight_rejections,
                "message": straight.message,
                "executed_commands": 0,
            },
            "reachable": {
                "ok": ok,
                "status": result.status,
                "selected_route": result.selected_route,
                "recovery_stages": [name for name, _ in result.recovery_stages],
                "max_executed_arm_joint_delta_rad": api.executed_max_dq,
                "final_position_error_m": final_pos_err,
                "final_orientation_error_rad": final_ori_err,
                "message": result.message,
            },
            "contract": reach.reachable_se3_contract(),
        }
    finally:
        trusted.fk_panda_hand = original_trusted_fk
        reach.fk_panda_hand = original_reach_fk

    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if not data["reachable"]["ok"]:
        raise SystemExit("v279 exact mechanical replay failed")


if __name__ == "__main__":
    main()
