#!/usr/bin/env python3
"""Offline replay of v272 job 523227's first, pathological goto_pose target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import numpy as np
from scipy.spatial.transform import Rotation

from capx.integrations.franka.trusted_se3 import (
    _quaternion_angle_rad,
    _slerp_shortest,
    _waypoint_fraction,
    trusted_se3_contract,
)
from capx.integrations.motion.pyroki_context import get_pyroki_context
from capx.integrations.motion.pyroki_snippets._solve_ik import solve_ik
from capx.integrations.motion.pyroki_snippets._solve_ik_vel_cost import solve_ik_local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
    robot = ctx.robot
    link_i = robot.links.names.index(ctx.target_link_name)

    def fk(cfg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose = jaxlie.SE3(
            robot.forward_kinematics(jnp.asarray(cfg, dtype=jnp.float32))[link_i]
        )
        return (
            np.asarray(pose.translation(), dtype=np.float64),
            np.asarray(pose.rotation().wxyz, dtype=np.float64),
        )

    live = np.array(
        [0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, np.pi / 4, 0.02]
    )
    grasp_position = np.array([0.64733738, 0.19962308, 0.02439878])
    grasp_wxyz = np.array([-0.34075639, 0.60183579, -0.62495569, -0.36208998])
    rot = Rotation.from_quat(
        [grasp_wxyz[1], grasp_wxyz[2], grasp_wxyz[3], grasp_wxyz[0]]
    )
    hand_goal = grasp_position + rot.apply(np.array([0.0, 0.0, -0.1]))
    target = hand_goal + rot.apply(np.array([0.0, 0.0, -0.15]))

    start_position, start_wxyz = fk(live)
    legacy = solve_ik(robot, ctx.target_link_name, grasp_wxyz, target)
    legacy_position, legacy_wxyz = fk(legacy)
    legacy_path = []
    for alpha in np.linspace(0.1, 1.0, 10):
        q = live + alpha * (legacy - live)
        p, _ = fk(q)
        delta = p - start_position
        requested = target - start_position
        direction = requested / np.linalg.norm(requested)
        progress = float(np.dot(delta, direction))
        lateral = float(np.linalg.norm(delta - progress * direction))
        legacy_path.append(
            {
                "alpha": float(alpha),
                "position": p.tolist(),
                "signed_progress_m": progress,
                "lateral_m": lateral,
            }
        )

    cfg = live.copy()
    trusted_rows = []
    for waypoint in range(1, 97):
        position, wxyz = fk(cfg)
        pos_err = float(np.linalg.norm(target - position))
        ori_err = _quaternion_angle_rad(grasp_wxyz, wxyz)
        if pos_err <= 0.003 and ori_err <= 0.03:
            break
        fraction = _waypoint_fraction(pos_err, ori_err)
        wp_position = position + fraction * (target - position)
        wp_wxyz = _slerp_shortest(wxyz, grasp_wxyz, fraction)
        candidate = solve_ik_local(
            robot,
            ctx.target_link_name,
            wp_wxyz,
            wp_position,
            cfg,
            max_arm_delta=0.20,
        )
        candidate_position, candidate_wxyz = fk(candidate)
        step = candidate_position - position
        direction = (target - position) / max(pos_err, 1e-12)
        progress = float(np.dot(step, direction))
        lateral = float(np.linalg.norm(step - progress * direction))
        trusted_rows.append(
            {
                "waypoint": waypoint,
                "fraction": fraction,
                "max_arm_joint_delta_rad": float(
                    np.max(np.abs(candidate[:7] - cfg[:7]))
                ),
                "candidate_fk_position_error_m": float(
                    np.linalg.norm(candidate_position - wp_position)
                ),
                "candidate_fk_orientation_error_rad": _quaternion_angle_rad(
                    candidate_wxyz, wp_wxyz
                ),
                "signed_progress_m": progress,
                "lateral_m": lateral,
                "remaining_position_error_m": float(
                    np.linalg.norm(target - candidate_position)
                ),
                "remaining_orientation_error_rad": _quaternion_angle_rad(
                    grasp_wxyz, candidate_wxyz
                ),
            }
        )
        cfg = candidate

    final_position, final_wxyz = fk(cfg)
    trusted_ok = bool(
        trusted_rows
        and max(r["max_arm_joint_delta_rad"] for r in trusted_rows) <= 0.2002
        and max(r["candidate_fk_position_error_m"] for r in trusted_rows) <= 0.004
        and max(r["candidate_fk_orientation_error_rad"] for r in trusted_rows) <= 0.05
        and min(r["signed_progress_m"] for r in trusted_rows) > 0.0
        and np.linalg.norm(final_position - target) <= 0.003
        and _quaternion_angle_rad(final_wxyz, grasp_wxyz) <= 0.03
    )
    data = {
        "schema": "capx_v272_first_goto_trusted_se3_audit_v1",
        "source": {"job_id": 523227, "task_id": 0, "replicate": 1, "trial": 2},
        "contract": trusted_se3_contract(),
        "live_cfg": live.tolist(),
        "start_hand_position": start_position.tolist(),
        "target_hand_approach_position": target.tolist(),
        "target_hand_wxyz": grasp_wxyz.tolist(),
        "requested_translation_m": (target - start_position).tolist(),
        "legacy": {
            "candidate": legacy.tolist(),
            "max_arm_joint_delta_rad": float(np.max(np.abs(legacy[:7] - live[:7]))),
            "l2_arm_joint_delta_rad": float(np.linalg.norm(legacy[:7] - live[:7])),
            "endpoint_fk_position_error_m": float(np.linalg.norm(legacy_position - target)),
            "endpoint_fk_orientation_error_rad": _quaternion_angle_rad(
                legacy_wxyz, grasp_wxyz
            ),
            "joint_interpolation_path": legacy_path,
        },
        "trusted": {
            "ok": trusted_ok,
            "waypoint_count": len(trusted_rows),
            "max_arm_joint_delta_rad": max(
                r["max_arm_joint_delta_rad"] for r in trusted_rows
            ),
            "min_signed_progress_m": min(
                r["signed_progress_m"] for r in trusted_rows
            ),
            "max_lateral_m": max(r["lateral_m"] for r in trusted_rows),
            "final_position_error_m": float(np.linalg.norm(final_position - target)),
            "final_orientation_error_rad": _quaternion_angle_rad(
                final_wxyz, grasp_wxyz
            ),
            "waypoints": trusted_rows,
        },
    }
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if not trusted_ok:
        raise SystemExit("trusted SE(3) exact replay failed contract")


if __name__ == "__main__":
    main()
