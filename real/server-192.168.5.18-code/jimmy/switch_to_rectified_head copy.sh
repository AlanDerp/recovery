#!/bin/bash
# Replace only the vendor head-camera process with the Jimmy VPI rectifier.
# This script does not launch or publish any robot motion command.
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-01}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

if ! ss -lun | awk '$5 ~ /:11811$/ {found=1} END {exit !found}'; then
  echo "ERROR: Fast DDS discovery server is not listening on port 11811." >&2
  echo "Start the normal Galaxea session first, then run this script." >&2
  exit 1
fi

mapfile -t vendor_pids < <(pgrep -f '/signal_camera/signal_camera_node' || true)
if ((${#vendor_pids[@]})); then
  echo "Stopping vendor head camera: ${vendor_pids[*]}"
  kill -INT "${vendor_pids[@]}"
  for _ in $(seq 1 20); do
    pgrep -f '/signal_camera/signal_camera_node' >/dev/null || break
    sleep 0.5
  done
fi

if pgrep -f '/signal_camera/signal_camera_node' >/dev/null; then
  echo "ERROR: vendor signal_camera_node did not stop; refusing device conflict." >&2
  exit 1
fi

if [[ ! -x /home/nvidia/jimmy/native_head_publisher/build/jimmy_compressed_publisher ]]; then
  echo "ERROR: C++ publisher is missing. Run:" >&2
  echo "  /home/nvidia/jimmy/build_native_head_publisher.sh" >&2
  exit 1
fi

exec /home/nvidia/jimmy/run_native_head_rectifier.sh \
  --side both \
  --node-name jimmy_native_head_rectifier
