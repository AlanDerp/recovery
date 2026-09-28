#!/bin/bash
set -eo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash

exec python3 /home/nvidia/jimmy/native_head_rectifier.py "$@"
