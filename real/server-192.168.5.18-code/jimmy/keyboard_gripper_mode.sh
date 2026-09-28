#!/usr/bin/env bash

# Switch only the existing R1 Lite homogeneous teleoperation tmux pane between
# the vendor launch and Jimmy's gripper-remapped launch. This script never
# starts the keyboard publisher and never sends a position command.

set -euo pipefail

ACTION="${1:-status}"
SESSION="r1lite_teleop"
VENDOR_DIR="/home/nvidia/galaxea/install/startup_config/share/startup_config/script/boot/modules/teleop"
VENDOR_COMMAND="./start_r1litet_homogeneous_teleop.sh"
CUSTOM_COMMAND="/home/nvidia/jimmy/start_remapped_homogeneous_teleop.sh"
CUSTOM_LAUNCH="r1litet_homogeneous_keyboard_gripper.launch.py"
LEFT_TOPIC="/motion_target/target_position_gripper_left"
RIGHT_TOPIC="/motion_target/target_position_gripper_right"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

load_ros() {
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-01}"
  export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
  export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
  set +u
  source /opt/ros/humble/setup.bash
  source /home/nvidia/galaxea/install/setup.bash
  set -u
}

session_attached() {
  tmux display-message -p -t "${SESSION}" '#{session_attached}'
}

vendor_running() {
  pgrep -f '[r]os2 launch teleoperation_ros2 r1litet_homogeneous_teleoperation_launch.py' >/dev/null 2>&1
}

custom_running() {
  pgrep -f "[${CUSTOM_LAUNCH:0:1}]${CUSTOM_LAUNCH:1}" >/dev/null 2>&1
}

controller_running() {
  pgrep -f '[h]omogeneous_controller_node' >/dev/null 2>&1
}

wait_until_stopped() {
  local deadline=$((SECONDS + 12))
  while controller_running || vendor_running || custom_running; do
    (( SECONDS >= deadline )) && return 1
    sleep 0.2
  done
}

wait_for_mode() {
  local mode="$1"
  local deadline=$((SECONDS + 15))
  while (( SECONDS < deadline )); do
    if [[ "${mode}" == "custom" ]] && custom_running && controller_running; then
      return 0
    fi
    if [[ "${mode}" == "vendor" ]] && vendor_running && controller_running; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

publisher_count() {
  local topic="$1"
  local topic_info
  # Keep an unknown/disconnected topic inside the explicit restore path below.
  # With `set -e`, a failing `ros2 topic info` in command substitution would
  # otherwise exit this script immediately and leave the remap active.
  topic_info="$(
    ros2 topic info --no-daemon --spin-time 3 "${topic}" 2>/dev/null || true
  )"
  awk '/Publisher count:/ {print $3; exit}' <<<"${topic_info}"
}

show_status() {
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session: missing (${SESSION})"
  else
    echo "tmux session: present (${SESSION}), attached=$(session_attached)"
  fi

  if custom_running; then
    echo "teleoperation mode: keyboard-gripper remap"
  elif vendor_running; then
    echo "teleoperation mode: vendor/Bluetooth gripper"
  elif controller_running; then
    echo "teleoperation mode: unknown homogeneous controller"
  else
    echo "teleoperation mode: stopped"
  fi
}

require_managed_session() {
  tmux has-session -t "${SESSION}" 2>/dev/null \
    || die "tmux session ${SESSION} is missing; use normal robot startup first"
  [[ "$(session_attached)" == "0" ]] \
    || die "tmux session ${SESSION} is attached; detach the other operator first"
}

send_mode_command() {
  local command="$1"
  tmux send-keys -t "${SESSION}" "cd '${VENDOR_DIR}'" Enter
  tmux send-keys -t "${SESSION}" "${command}" Enter
}

start_custom() {
  require_managed_session
  pgrep -af 'keyboard_gripper_[t]eleop.py' >/dev/null 2>&1 \
    && die "stop the keyboard gripper controller before switching authority"

  if custom_running; then
    echo "Keyboard-gripper remap is already active."
    show_status
    return
  fi
  vendor_running || die "expected vendor homogeneous teleoperation process is not running"

  tmux send-keys -t "${SESSION}" C-c
  wait_until_stopped \
    || die "vendor teleoperation did not stop within 12 seconds; no replacement was started"

  send_mode_command "${CUSTOM_COMMAND}"
  if ! wait_for_mode custom; then
    echo "ERROR: remapped teleoperation did not become ready; restoring vendor mode" >&2
    tmux send-keys -t "${SESSION}" C-c
    wait_until_stopped || true
    send_mode_command "${VENDOR_COMMAND}"
    wait_for_mode vendor || true
    exit 2
  fi

  load_ros
  local left_count right_count
  left_count="$(publisher_count "${LEFT_TOPIC}")"
  right_count="$(publisher_count "${RIGHT_TOPIC}")"
  if [[ "${left_count}" != "0" || "${right_count}" != "0" ]]; then
    echo "ERROR: real gripper publishers remain (left=${left_count:-unknown}, right=${right_count:-unknown}); restoring vendor mode" >&2
    tmux send-keys -t "${SESSION}" C-c
    wait_until_stopped || true
    send_mode_command "${VENDOR_COMMAND}"
    wait_for_mode vendor || true
    exit 2
  fi

  echo "Keyboard-gripper remap active; real gripper topics have zero publishers."
}

restore_vendor() {
  require_managed_session
  pgrep -af 'keyboard_gripper_[t]eleop.py' >/dev/null 2>&1 \
    && die "exit the keyboard gripper controller with Esc before restoring vendor mode"

  if vendor_running; then
    echo "Vendor teleoperation is already active."
    show_status
    return
  fi
  custom_running || die "refusing to replace an unknown or stopped controller state"

  tmux send-keys -t "${SESSION}" C-c
  wait_until_stopped \
    || die "remapped teleoperation did not stop within 12 seconds"
  send_mode_command "${VENDOR_COMMAND}"
  wait_for_mode vendor \
    || die "vendor teleoperation did not become ready; inspect tmux attach -t ${SESSION}"
  echo "Vendor/Bluetooth gripper authority restored."
}

case "${ACTION}" in
  start)
    start_custom
    ;;
  restore)
    restore_vendor
    ;;
  status)
    show_status
    ;;
  *)
    echo "Usage: $0 {start|restore|status}" >&2
    exit 64
    ;;
esac
