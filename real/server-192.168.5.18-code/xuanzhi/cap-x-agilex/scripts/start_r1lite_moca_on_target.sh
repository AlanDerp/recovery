#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
GALAXEA_SETUP="/home/nvidia/galaxea/install/local_setup.bash"
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

set +u
source /opt/ros/humble/setup.bash
source "${GALAXEA_SETUP}"
set -u

# The homogeneous teleoperation node publishes a zero-centered vendor command
# at 200 Hz. It must not compete with Cap-X for the right-arm target.
tmux kill-session -t r1lite_teleop 2>/dev/null || true
pkill -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch teleoperation_ros2 r1litet_homogeneous_teleoperation_launch.py' 2>/dev/null || true
pkill -f '^/home/nvidia/galaxea/install/teleoperation_ros2/lib/teleoperation_ros2/homogeneous_controller_node' 2>/dev/null || true

arm_topic="/capx/motion_target/target_joint_state_arm_right"
gripper_topic="/capx/motion_target/target_position_gripper_right"
topic_info="$(ros2 topic info -v "${arm_topic}" 2>/dev/null || true)"
if rg -q "Node name: moca_adapter" <<<"${topic_info}"; then
  echo "[error] refusing to reuse an existing isolated MOCA session" >&2
  exit 1
fi
if pgrep -f '[m]oca_adapter/lib/moca_adapter/moca_main_node' >/dev/null; then
  echo "[error] stale isolated MOCA process exists without a discovered subscriber" >&2
  exit 1
fi

echo "[start] isolated MOCA for Cap-X targets"
tmux kill-session -t moca_capx_isolated 2>/dev/null || true
mkdir -p "${REPO_ROOT}/logs"
tmux new-session -d -s moca_capx_isolated "bash -lc '
  export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
  export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}
  export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}
  source /opt/ros/humble/setup.bash
  source ${GALAXEA_SETUP}
  exec ros2 run moca_adapter moca_main_node --ros-args \
    -r __node:=moca_adapter \
    -r /motion_target/target_joint_state_arm_right:=${arm_topic} \
    -r /motion_target/target_pose_arm_right:=/capx/disabled_target_pose_arm_right \
    -r /motion_target/target_position_gripper_right:=${gripper_topic} \
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
    >> ${REPO_ROOT}/logs/moca_capx_isolated.log 2>&1
'"

deadline=$((SECONDS + 20))
while (( SECONDS < deadline )); do
  topic_info="$(ros2 topic info -v "${arm_topic}" 2>/dev/null || true)"
  subscription_count="$(
    rg -m1 '^Subscription count: ' <<<"${topic_info}" \
      | awk '{print $3}' || true
  )"
  subscription_count="${subscription_count:-0}"
  if [[ "${subscription_count}" == "1" ]] \
      && rg -q "Node name: moca_adapter" <<<"${topic_info}"; then
    echo "[ok] isolated MOCA subscribed to ${arm_topic}"
    exit 0
  fi
  if (( subscription_count > 1 )); then
    echo "[error] multiple isolated MOCA subscribers detected" >&2
    "${REPO_ROOT}/scripts/stop_r1lite_moca_isolated.sh" || true
    exit 1
  fi
  sleep 1
done

echo "[error] isolated MOCA did not subscribe to ${arm_topic}" >&2
echo "See ${REPO_ROOT}/logs/moca_capx_isolated.log" >&2
exit 1
