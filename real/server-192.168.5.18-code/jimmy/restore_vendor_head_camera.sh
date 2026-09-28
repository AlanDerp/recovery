#!/bin/bash
# Stop the Jimmy rectifier and restore the unchanged vendor head-camera node.
# This script does not launch or publish any robot motion command.
set -eo pipefail

mapfile -t rectifier_pids < <(
  pgrep -f '/home/nvidia/jimmy/native_head_rectifier.py' || true
)
if ((${#rectifier_pids[@]})); then
  echo "Stopping Jimmy head rectifier: ${rectifier_pids[*]}"
  kill -INT "${rectifier_pids[@]}"
  for _ in $(seq 1 20); do
    pgrep -f '/home/nvidia/jimmy/native_head_rectifier.py' >/dev/null || break
    sleep 0.5
  done
fi

if pgrep -f '/home/nvidia/jimmy/native_head_rectifier.py' >/dev/null; then
  echo "ERROR: Jimmy rectifier did not stop; refusing device conflict." >&2
  exit 1
fi

source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-01}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

exec /home/nvidia/galaxea/install/startup_config/share/startup_config/script/boot/modules/hdas/start_signal_camera_head.sh
