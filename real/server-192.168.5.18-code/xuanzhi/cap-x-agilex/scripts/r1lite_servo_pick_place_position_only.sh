#!/usr/bin/env bash
set -euo pipefail

cd /home/nvidia/xuanzhi/cap-x-agilex

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

STEP_M="${STEP_M:-0.025}"
POS_TOL_M="${POS_TOL_M:-0.025}"
MAX_ITERS_PER_TARGET="${MAX_ITERS_PER_TARGET:-45}"

/home/nvidia/xuanzhi/cap-x-agilex/.capx_r1lite_venv/bin/python - "${STEP_M}" "${POS_TOL_M}" "${MAX_ITERS_PER_TARGET}" <<'PY'
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel

repo = Path("/home/nvidia/xuanzhi/cap-x-agilex")
step_m = float(sys.argv[1])
pos_tol_m = float(sys.argv[2])
max_iters = int(sys.argv[3])

cfg = yaml.safe_load(
    (repo / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml").read_text()
)
low_cfg = dict(cfg["env"]["cfg"]["low_level"])
low_cfg.pop("_target_", None)
sig = inspect.signature(R1LiteRealLowLevel.__init__)
kwargs = {k: v for k, v in low_cfg.items() if k in sig.parameters}
r1 = R1LiteRealLowLevel(**kwargs)

scene = json.loads(
    (repo / "outputs/r1lite_tennis_ball_in_box/current_scene_3d.json").read_text()
)
ball = np.asarray(scene["tennis_ball"]["position_base_m"], dtype=float)

# Use the current reliable box detection from the offline IK plan.
plan = json.loads(
    (repo / "outputs/r1lite_tennis_ball_in_box/offline_ik_pick_place_joint_poses_3d.json").read_text()
)
box = np.asarray(plan["objects"]["box_center_base_m"], dtype=float)
drop = np.asarray(plan["objects"]["drop_position_base_m"], dtype=float)

log_path = (
    repo
    / "outputs/r1lite_tennis_ball_in_box"
    / f"position_servo_pick_place_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
)


def write(event):
    event["t_unix"] = time.time()
    with log_path.open("a") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps(event, ensure_ascii=False), flush=True)


def physical_ee_pose():
    return r1.get_physical_tool_z_pose()


def capped_step(delta):
    norm = float(np.linalg.norm(delta))
    if norm <= step_m or norm <= 1e-12:
        return delta
    return delta * (step_m / norm)


def servo_to(name, target_position, hold_s=0.18):
    target = np.asarray(target_position, dtype=float).reshape(3)
    write({"event": "servo_target_begin", "name": name, "target_position_base_m": target.tolist()})
    reached = False
    for iteration in range(max_iters):
        current_pos, current_quat = physical_ee_pose()
        current_joints = r1.get_arm_joints()
        delta = target - current_pos
        error_m = float(np.linalg.norm(delta))
        write(
            {
                "event": "servo_iteration_before",
                "name": name,
                "iteration": iteration,
                "actual_position_base_m": current_pos.tolist(),
                "target_position_base_m": target.tolist(),
                "position_error_m": error_m,
            }
        )
        if error_m <= pos_tol_m:
            reached = True
            break

        next_pos = current_pos + capped_step(delta)
        try:
            joints = r1.get_relaxed_ik_candidate(
                next_pos,
                current_quat,
                timeout_s=3.0,
                reference_joints=current_joints,
                start_joints=current_joints,
            )
        except Exception as exc:
            write(
                {
                    "event": "servo_ik_failed",
                    "name": name,
                    "iteration": iteration,
                    "next_position_base_m": next_pos.tolist(),
                    "error": repr(exc),
                }
            )
            break

        result = r1.move_joints_feedforward(
            joints,
            duration_s=0.65,
            context=f"position_servo_{name}_{iteration:03d}",
            skip_feedback_settle=True,
        )
        time.sleep(hold_s)
        after_pos, _after_quat = physical_ee_pose()
        write(
            {
                "event": "servo_iteration_after",
                "name": name,
                "iteration": iteration,
                "commanded_next_position_base_m": next_pos.tolist(),
                "actual_after_position_base_m": after_pos.tolist(),
                "remaining_error_m": float(np.linalg.norm(target - after_pos)),
                "command_result": result,
            }
        )

    final_pos, final_quat = physical_ee_pose()
    final_error = float(np.linalg.norm(target - final_pos))
    write(
        {
            "event": "servo_target_done",
            "name": name,
            "reached": bool(reached or final_error <= pos_tol_m),
            "final_position_base_m": final_pos.tolist(),
            "target_position_base_m": target.tolist(),
            "final_error_m": final_error,
        }
    )
    return final_error <= pos_tol_m


write(
    {
        "event": "start",
        "mode": "online_position_only_servo_pick_place",
        "step_m": step_m,
        "pos_tol_m": pos_tol_m,
        "max_iters_per_target": max_iters,
        "ball_center_base_m": ball.tolist(),
        "box_center_base_m": box.tolist(),
        "drop_position_base_m": drop.tolist(),
        "log": str(log_path),
    }
)
write({"event": "initial_pose", "position_base_m": physical_ee_pose()[0].tolist(), "gripper": float(r1.get_gripper_position())})

try:
    g = r1.command_gripper(float(r1.gripper_open_position), duration_s=1.0)
    write({"event": "gripper_open_start", "feedback": float(g)})
except Exception as exc:
    write({"event": "gripper_open_start_error", "error": repr(exc), "continuing": True})

safe_z = max(float(physical_ee_pose()[0][2]), float(ball[2] + 0.25), float(drop[2] + 0.18))
targets = [
    ("above_ball_high", np.array([ball[0], ball[1], safe_z])),
    ("pregrasp", np.array([ball[0], ball[1], ball[2] + 0.14])),
    ("approach", np.array([ball[0], ball[1], ball[2] + 0.08])),
    ("grasp", np.array([ball[0], ball[1], ball[2] + 0.045])),
]
for name, target in targets:
    servo_to(name, target)

try:
    g = r1.command_gripper(float(r1.gripper_close_position), duration_s=1.2)
    write({"event": "gripper_close_grasp", "feedback": float(g)})
except Exception as exc:
    write({"event": "gripper_close_grasp_error", "error": repr(exc), "continuing": True})

place_targets = [
    ("lift_after_grasp", np.array([ball[0], ball[1], safe_z])),
    ("above_box", np.array([drop[0], drop[1], safe_z])),
    ("release", np.array([drop[0], drop[1], drop[2]])),
]
for name, target in place_targets:
    servo_to(name, target)

try:
    g = r1.command_gripper(float(r1.gripper_open_position), duration_s=1.2)
    write({"event": "gripper_open_release", "feedback": float(g)})
except Exception as exc:
    write({"event": "gripper_open_release_error", "error": repr(exc), "continuing": True})

servo_to("retreat", np.array([drop[0], drop[1], drop[2] + 0.15]))

write(
    {
        "event": "finished",
        "final_pose": physical_ee_pose()[0].tolist(),
        "final_gripper": float(r1.get_gripper_position()),
        "log": str(log_path),
    }
)
PY
