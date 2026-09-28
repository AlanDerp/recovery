#!/usr/bin/env bash
# Start Piper driver for the left arm only (can_left).
set -euo pipefail

CAPX_PYTHON="${CAPX_PYTHON:-/home/agilex/miniconda3/envs/capx/bin/python}"
PIPER_SDK_PATH="${PIPER_SDK_PATH:-/home/agilex/piper_sdk-0_3_0_beta}"

pick_python() {
  if python3 -c "import piper_sdk" >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if [[ -x "${CAPX_PYTHON}" ]] && "${CAPX_PYTHON}" -c "import piper_sdk" >/dev/null 2>&1; then
    echo "${CAPX_PYTHON}"
    return 0
  fi
  return 1
}

if ! PYTHON_BIN="$(pick_python)"; then
  echo "[error] piper_sdk not found in the active Python env."
  echo "Install into capx (recommended):"
  echo "  conda activate capx"
  echo "  pip install rospkg catkin_pkg pyyaml"
  echo "  pip install -e ${PIPER_SDK_PATH}"
  echo "Or set CAPX_PYTHON=/path/to/python before running this script."
  exit 1
fi

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

source /opt/ros/noetic/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

# Only one Piper MS node may own can_left. Stale instances block motion.
if command -v rosnode >/dev/null 2>&1; then
  mapfile -t stale_nodes < <(rosnode list 2>/dev/null | grep -E 'piper_left|piper_start_all_node' || true)
  if ((${#stale_nodes[@]} > 0)); then
    echo "[warn] Stopping ${#stale_nodes[@]} stale Piper node(s) before launch:"
    printf '  %s\n' "${stale_nodes[@]}"
    for node in "${stale_nodes[@]}"; do
      rosnode kill "${node}" >/dev/null 2>&1 || true
    done
    sleep 1
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec roslaunch "${SCRIPT_DIR}/launch/piper_left_ms.launch" mode:=1 auto_enable:=true "$@"
