#!/usr/bin/env bash
set -euo pipefail

cd /home/nvidia/xuanzhi/cap-x-agilex
mkdir -p logs

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

sn_l="$(< /opt/galaxea/sensor/realsense/RS_LEFT)"
sn_r="$(< /opt/galaxea/sensor/realsense/RS_RIGHT)"

exec ros2 launch realsense2_camera rs_multi_camera_2.0_launch.py \
  serial_no1:="${sn_l}" \
  serial_no2:="${sn_r}" \
  enable_depth1:=false \
  enable_depth2:=true \
  depth_module.depth_profile1:=424x240x15 \
  depth_module.depth_profile2:=424x240x15 \
  align_depth.enable1:=false \
  align_depth.enable2:=true
