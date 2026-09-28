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

ARM_TOPIC="/capx/motion_target/target_joint_state_arm_right"
GRIPPER_TOPIC="/capx/motion_target/target_position_gripper_right"

ensure_moca() {
  local info
  info="$(ros2 topic info -v "${ARM_TOPIC}" 2>/dev/null || true)"
  if grep -q "Node name: moca_adapter" <<<"${info}"; then
    echo "[ok] MOCA already subscribes to ${ARM_TOPIC}"
    return
  fi

  echo "[start] isolated MOCA for Cap-X arm/gripper targets"
  pkill -f "/moca_adapter/lib/moca_adapter/moca_main_node" 2>/dev/null || true
  sleep 2
  mkdir -p logs
  ros2 run moca_adapter moca_main_node --ros-args \
    -r __node:=moca_adapter \
    -r /motion_target/target_joint_state_arm_right:="${ARM_TOPIC}" \
    -r /motion_target/target_pose_arm_right:=/capx/disabled_target_pose_arm_right \
    -r /motion_target/target_position_gripper_right:="${GRIPPER_TOPIC}" \
    -p robot_name:=r1lite \
    -p robot_type:=R1-LITE \
    -p config_file:=/home/nvidia/galaxea/install/moca_adapter/share/moca_adapter/config/moca_main_r1lite.toml \
    -p hardware_json:=/opt/galaxea/body/hardware.json \
    -p robot_description_file:=/home/nvidia/galaxea/install/r1lite_urdf/share/r1lite_urdf/urdf/r1lite_2026.urdf \
    -p robot_srdf_file:=/home/nvidia/galaxea/install/moca_adapter/share/moca_adapter/srdf/r1lite_srdf.xml \
    -p rviz_enabled:=false \
    -p end_effector_control.enable:=false \
    -p end_effector_control.enable_right_ee:=false \
    -p end_effector_control.use_interpolation:=false \
    -p end_effector_control.ee_control_mode:=independent \
    > logs/moca_capx_isolated_return_home.log 2>&1 &

  local deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    info="$(ros2 topic info -v "${ARM_TOPIC}" 2>/dev/null || true)"
    if grep -q "Node name: moca_adapter" <<<"${info}"; then
      echo "[ok] MOCA subscribed to ${ARM_TOPIC}"
      return
    fi
    sleep 1
  done

  echo "[error] MOCA did not subscribe to ${ARM_TOPIC}" >&2
  echo "See logs/moca_capx_isolated_return_home.log" >&2
  exit 1
}

reset_mit_mode() {
  echo "[mit] reset right arm into MIT mode"
  for command in 2 5 1; do
    local response
    response="$(timeout 8 ros2 service call /function_frame_right_arm hdas_msg/srv/FunctionFrame "{command: ${command}}")"
    echo "${response}"
    if ! grep -q "success=True" <<<"${response}"; then
      echo "[error] MIT mode command ${command} failed" >&2
      exit 1
    fi
  done
}

ensure_moca
reset_mit_mode

/home/nvidia/xuanzhi/cap-x-agilex/.capx_r1lite_venv/bin/python - <<'PY'
import inspect
from pathlib import Path

import yaml

from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel

repo = Path("/home/nvidia/xuanzhi/cap-x-agilex")
cfg = yaml.safe_load(
    (repo / "tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml").read_text()
)
low_cfg = dict(cfg["env"]["cfg"]["low_level"])
low_cfg.pop("_target_", None)

sig = inspect.signature(R1LiteRealLowLevel.__init__)
kwargs = {k: v for k, v in low_cfg.items() if k in sig.parameters}

r1 = R1LiteRealLowLevel(**kwargs)
target = r1.safe_joint_home_feedback

print("[state] current:", r1.get_arm_joints().tolist())
print("[target] safe_joint_home_feedback:", target.tolist())

result = r1.move_joint_waypoints_adaptive(
    target,
    waypoint_step_rad=0.03,
    context="manual_return_home_segmented",
    skip_feedback_settle=True,
)

print("[result]", result)
print("[state] after:", r1.get_arm_joints().tolist())
print("[done] return home command completed")
PY
