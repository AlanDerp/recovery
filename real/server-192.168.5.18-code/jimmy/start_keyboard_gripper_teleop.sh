#!/usr/bin/env bash

set -eo pipefail

if [[ ! -t 0 || ! -t 1 ]]; then
  echo "ERROR: keyboard gripper control requires an interactive terminal." >&2
  exit 2
fi

if pgrep -af 'keyboard_gripper_[t]eleop.py' >/dev/null 2>&1; then
  echo "ERROR: a keyboard gripper controller is already running." >&2
  echo "Inspect it before starting another copy:" >&2
  echo "  pgrep -af 'keyboard_gripper_[t]eleop.py'" >&2
  exit 3
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-01}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ERROR: /opt/ros/humble/setup.bash is missing." >&2
  exit 4
fi

if [[ ! -f /home/nvidia/galaxea/install/setup.bash ]]; then
  echo "ERROR: /home/nvidia/galaxea/install/setup.bash is missing." >&2
  exit 4
fi

if [[ ! -f "${FASTRTPS_DEFAULT_PROFILES_FILE}" ]]; then
  echo "ERROR: Fast DDS profile is missing: ${FASTRTPS_DEFAULT_PROFILES_FILE}" >&2
  exit 4
fi

# Source only the required environments. Do not call the vendor prestart
# wrapper because it performs unrelated privileged system changes.
set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/keyboard_gripper_teleop.py" "$@"
