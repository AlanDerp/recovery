#!/usr/bin/env python3
"""Compare legacy, soft-live, and hard-local PyRoKi candidates for tip XY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import numpy as np

from capx.integrations.motion.pyroki_context import get_pyroki_context
from capx.integrations.motion.pyroki_snippets._solve_ik_vel_cost import (
    solve_ik,
    solve_ik_local,
)


CONFIGURATIONS = (
    (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02),
    (0.6, -0.4, 0.8, -1.9, -0.5, 1.5, 0.4, 0.02),
    (-0.8, 0.2, -0.7, -1.4, 0.8, 2.0, -0.6, 0.02),
    (1.2, -1.0, 1.0, -2.6, -1.0, 0.8, 1.2, 0.02),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
    robot = ctx.robot
    link_i = robot.links.names.index(ctx.target_link_name)
    lower = np.asarray(robot.joints.lower_limits, dtype=np.float64)
    upper = np.asarray(robot.joints.upper_limits, dtype=np.float64)

    def fk(cfg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        se3 = jaxlie.SE3(
            robot.forward_kinematics(jnp.asarray(cfg, dtype=jnp.float32))[link_i]
        )
        return (
            np.asarray(se3.translation(), dtype=np.float64),
            np.asarray(se3.rotation().wxyz, dtype=np.float64),
        )

    rows = []
    for case, raw_cfg in enumerate(CONFIGURATIONS):
        live = np.clip(np.asarray(raw_cfg, dtype=np.float64), lower + 1e-3, upper - 1e-3)
        position, wxyz = fk(live)
        for axis, delta in (
            ("x", np.array([0.03, 0.0, 0.0])),
            ("y", np.array([0.0, 0.03, 0.0])),
        ):
            target = position + delta
            legacy = solve_ik(robot, ctx.target_link_name, wxyz, target, live)
            trusted = solve_ik(
                robot,
                ctx.target_link_name,
                wxyz,
                target,
                live,
                initial_cfg=live,
            )
            local = solve_ik_local(
                robot,
                ctx.target_link_name,
                wxyz,
                target,
                live,
                max_arm_delta=0.20,
            )
            legacy_pos, _ = fk(legacy)
            trusted_pos, _ = fk(trusted)
            local_pos, _ = fk(local)
            rows.append(
                {
                    "case": case,
                    "axis": axis,
                    "legacy_max_arm_joint_delta_rad": float(
                        np.max(np.abs(legacy[:7] - live[:7]))
                    ),
                    "legacy_fk_position_error_m": float(
                        np.linalg.norm(legacy_pos - target)
                    ),
                    "trusted_max_arm_joint_delta_rad": float(
                        np.max(np.abs(trusted[:7] - live[:7]))
                    ),
                    "trusted_fk_position_error_m": float(
                        np.linalg.norm(trusted_pos - target)
                    ),
                    "hard_local_max_arm_joint_delta_rad": float(
                        np.max(np.abs(local[:7] - live[:7]))
                    ),
                    "hard_local_fk_position_error_m": float(
                        np.linalg.norm(local_pos - target)
                    ),
                    "hard_local_contract_ok": bool(
                        np.max(np.abs(local[:7] - live[:7])) <= 0.20 + 1e-6
                        and np.linalg.norm(local_pos - target) <= 0.004
                    ),
                }
            )

    data = {
        "schema": "capx_tip_xy_local_solver_audit_v2",
        "target_delta_m": 0.03,
        "legacy_call": "prev_cfg soft cost only; optimizer default initialization",
        "trusted_call": "prev_cfg=live and initial_cfg=live",
        "hard_local_call": "live initialization plus hard |q_arm-live_q_arm|<=0.20",
        "robot_velocity_limits_rad_s": np.asarray(
            robot.joints.velocity_limits, dtype=np.float64
        ).tolist(),
        "rows": rows,
    }
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
