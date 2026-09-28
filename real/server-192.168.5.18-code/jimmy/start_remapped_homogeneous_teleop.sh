#!/usr/bin/env bash

set -eo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-01}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec ros2 launch "${SCRIPT_DIR}/r1litet_homogeneous_keyboard_gripper.launch.py"

