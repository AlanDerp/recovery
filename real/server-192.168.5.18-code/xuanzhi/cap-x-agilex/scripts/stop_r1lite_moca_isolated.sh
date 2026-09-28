#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

process_pattern='[m]oca_adapter/lib/moca_adapter/moca_main_node'
arm_topic="/capx/motion_target/target_joint_state_arm_right"

tmux kill-session -t moca_capx_isolated 2>/dev/null || true
pkill -TERM -f "${process_pattern}" 2>/dev/null || true

deadline=$((SECONDS + 5))
while pgrep -f "${process_pattern}" >/dev/null && ((SECONDS < deadline)); do
  sleep 0.1
done

if pgrep -f "${process_pattern}" >/dev/null; then
  pkill -KILL -f "${process_pattern}" 2>/dev/null || true
  deadline=$((SECONDS + 3))
  while pgrep -f "${process_pattern}" >/dev/null && ((SECONDS < deadline)); do
    sleep 0.1
  done
fi

if pgrep -f "${process_pattern}" >/dev/null; then
  echo "[error] isolated MOCA process survived TERM and KILL" >&2
  exit 1
fi

# Fast DDS can retain a dead endpoint briefly after the process exits.  Do not
# let a new controller start until discovery also reports no MOCA subscriber.
deadline=$((SECONDS + 15))
while ((SECONDS < deadline)); do
  topic_info="$(ROS2CLI_NO_DAEMON=1 ros2 topic info -v "${arm_topic}" 2>/dev/null || true)"
  if ! rg -q "Node name: moca_adapter" <<<"${topic_info}"; then
    break
  fi
  sleep 0.2
done

topic_info="$(ROS2CLI_NO_DAEMON=1 ros2 topic info -v "${arm_topic}" 2>/dev/null || true)"
if rg -q "Node name: moca_adapter" <<<"${topic_info}"; then
  echo "[error] stale MOCA DDS subscriber remained after process exit" >&2
  exit 1
fi

if tmux has-session -t moca_capx_isolated 2>/dev/null; then
  echo "[error] isolated MOCA tmux session remained after stop" >&2
  exit 1
fi

echo "[ok] isolated MOCA process, session, and DDS endpoint stopped"
