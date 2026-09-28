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

PLAN_PATH="${1:-/home/nvidia/xuanzhi/cap-x-agilex/outputs/r1lite_tennis_ball_in_box/offline_ik_pick_place_joint_poses_3d.json}"
POS_TOL_M="${POS_TOL_M:-0.035}"
MAX_CORRECTION_ITERS="${MAX_CORRECTION_ITERS:-3}"
MAX_COMPENSATION_STEP_M="${MAX_COMPENSATION_STEP_M:-0.08}"

echo "[plan] ${PLAN_PATH}"
echo "[params] POS_TOL_M=${POS_TOL_M} MAX_CORRECTION_ITERS=${MAX_CORRECTION_ITERS} MAX_COMPENSATION_STEP_M=${MAX_COMPENSATION_STEP_M}"

/home/nvidia/xuanzhi/cap-x-agilex/.capx_r1lite_venv/bin/python - "${PLAN_PATH}" "${POS_TOL_M}" "${MAX_CORRECTION_ITERS}" "${MAX_COMPENSATION_STEP_M}" <<'PY'
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel

repo = Path("/home/nvidia/xuanzhi/cap-x-agilex")
plan_path = Path(sys.argv[1])
pos_tol_m = float(sys.argv[2])
max_iters = int(sys.argv[3])
max_comp_step_m = float(sys.argv[4])

cfg = yaml.safe_load(
    (repo / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml").read_text()
)
low_cfg = dict(cfg["env"]["cfg"]["low_level"])
low_cfg.pop("_target_", None)
sig = inspect.signature(R1LiteRealLowLevel.__init__)
kwargs = {k: v for k, v in low_cfg.items() if k in sig.parameters}

r1 = R1LiteRealLowLevel(**kwargs)
plan = json.loads(plan_path.read_text())
waypoints = plan["waypoints"]

log_path = (
    repo
    / "outputs/r1lite_tennis_ball_in_box"
    / f"pose_closed_loop_keyframes_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
)


def write(event: dict) -> None:
    event["t_unix"] = time.time()
    with log_path.open("a") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps(event, ensure_ascii=False), flush=True)


def physical_ee_pose() -> tuple[np.ndarray, np.ndarray]:
    """Read physical feedback converted to RelaxedIK's tool-Z TCP."""
    return r1.get_physical_tool_z_pose()


def quat_error_rad(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> float:
    a = np.asarray(a_xyzw, dtype=float).reshape(4)
    b = np.asarray(b_xyzw, dtype=float).reshape(4)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(2.0 * math.acos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


def capped(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-12:
        return vector
    return vector * (max_norm / norm)


write(
    {
        "event": "start",
        "mode": "physical_pose_closed_loop_no_gripper",
        "plan": str(plan_path),
        "log": str(log_path),
        "pos_tol_m": pos_tol_m,
        "max_iters": max_iters,
        "max_compensation_step_m": max_comp_step_m,
        "note": "actual pose is physical /motion_control/pose_ee_arm_right",
    }
)

current_pos, current_quat = physical_ee_pose()
write(
    {
        "event": "initial_physical_pose",
        "position_base_m": current_pos.tolist(),
        "quaternion_xyzw": current_quat.tolist(),
        "arm_joints": r1.get_arm_joints().tolist(),
    }
)

for wp in waypoints[1:]:
    wid = wp["id"]
    desired_position = np.asarray(wp["target_ee_position_base_m"], dtype=float).reshape(3)
    desired_quat = np.asarray(wp["target_quaternion_xyzw"], dtype=float).reshape(4)
    nominal_joints = np.asarray(wp["joint_feedback_solution_rad"], dtype=float).reshape(6)

    write(
        {
            "event": "keyframe_begin",
            "id": wid,
            "label": wp["label"],
            "desired_position_base_m": desired_position.tolist(),
            "desired_quaternion_xyzw": desired_quat.tolist(),
        }
    )

    # Iteration 0 uses the originally selected IK solution. Later iterations
    # compensate in Cartesian target space based on physical pose error.
    command_joints = nominal_joints
    compensated_target = desired_position.copy()
    for iteration in range(max_iters + 1):
        before_pos, before_quat = physical_ee_pose()
        before_joints = r1.get_arm_joints()
        result = r1.move_joints_feedforward(
            command_joints,
            duration_s=2.5,
            context=f"pose_closed_loop_{wid}_iter_{iteration:02d}",
            skip_feedback_settle=True,
        )
        time.sleep(0.35)
        actual_pos, actual_quat = physical_ee_pose()
        actual_joints = r1.get_arm_joints()
        pos_error = desired_position - actual_pos
        pos_error_m = float(np.linalg.norm(pos_error))
        ori_error_rad = quat_error_rad(desired_quat, actual_quat)

        write(
            {
                "event": "keyframe_iteration",
                "id": wid,
                "iteration": iteration,
                "compensated_target_position_base_m": compensated_target.tolist(),
                "before_position_base_m": before_pos.tolist(),
                "actual_position_base_m": actual_pos.tolist(),
                "desired_position_base_m": desired_position.tolist(),
                "position_error_m": pos_error_m,
                "position_error_xyz_m": pos_error.tolist(),
                "orientation_error_rad": ori_error_rad,
                "before_joints": before_joints.tolist(),
                "actual_joints": actual_joints.tolist(),
                "command_result": result,
            }
        )

        if pos_error_m <= pos_tol_m:
            write({"event": "keyframe_reached", "id": wid, "iteration": iteration})
            break

        if iteration >= max_iters:
            write(
                {
                    "event": "keyframe_not_reached",
                    "id": wid,
                    "final_position_error_m": pos_error_m,
                }
            )
            break

        compensation = capped(pos_error, max_comp_step_m)
        compensated_target = desired_position + compensation
        try:
            command_joints = r1.get_relaxed_ik_candidate(
                compensated_target,
                desired_quat,
                timeout_s=3.0,
                reference_joints=actual_joints,
                start_joints=actual_joints,
            )
        except Exception as exc:
            write(
                {
                    "event": "correction_ik_failed",
                    "id": wid,
                    "iteration": iteration,
                    "compensated_target_position_base_m": compensated_target.tolist(),
                    "error": repr(exc),
                }
            )
            break

write(
    {
        "event": "finished",
        "final_physical_pose": physical_ee_pose()[0].tolist(),
        "final_arm_joints": r1.get_arm_joints().tolist(),
        "log": str(log_path),
    }
)
PY
