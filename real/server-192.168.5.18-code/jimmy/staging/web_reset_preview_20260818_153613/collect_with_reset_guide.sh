#!/bin/bash
# One-command corrected-camera data collection with an object-reset overlay.
# This script never launches or publishes robot motion commands.
set -o pipefail

DISPLAY_MODE="opencv"
WEB_PORT=8765
WEB_FPS=5

usage() {
  cat <<'EOF'
Usage: collect_with_reset_guide.sh [--display opencv|web] [--web-port PORT] [--web-fps 1-10]

  opencv  Existing robot-local OpenCV reset window (default).
  web     Read-only reset overlay through an SSH tunnel; controls stay here.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --display)
      [[ $# -ge 2 ]] || { echo "ERROR: --display requires a value." >&2; exit 2; }
      DISPLAY_MODE="$2"
      shift 2
      ;;
    --web-port)
      [[ $# -ge 2 ]] || { echo "ERROR: --web-port requires a value." >&2; exit 2; }
      WEB_PORT="$2"
      shift 2
      ;;
    --web-fps)
      [[ $# -ge 2 ]] || { echo "ERROR: --web-fps requires a value." >&2; exit 2; }
      WEB_FPS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$DISPLAY_MODE" == "opencv" || "$DISPLAY_MODE" == "web" ]] || {
  echo "ERROR: --display must be opencv or web." >&2
  exit 2
}
[[ "$WEB_PORT" =~ ^[0-9]+$ ]] && (( WEB_PORT >= 1 && WEB_PORT <= 65535 )) || {
  echo "ERROR: --web-port must be between 1 and 65535." >&2
  exit 2
}
[[ "$WEB_FPS" =~ ^([1-9]|10)$ ]] || {
  echo "ERROR: --web-fps must be an integer from 1 to 10." >&2
  exit 2
}

source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash

set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/nvidia/.Xauthority}"

if [[ -z "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]]; then
  export FASTRTPS_DEFAULT_PROFILES_FILE="/opt/galaxea/find_server/super_client_configuration_file.xml"
fi

echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"
echo "FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE"
echo "DISPLAY=$DISPLAY"

ros2 topic list --no-daemon --spin-time 3 >/dev/null || {
  echo "ERROR: ROS 2 discovery is unavailable in this shell." >&2
  exit 1
}

HEAD_LEFT_TOPIC="/hdas/camera_head/left_raw/image_raw_color/compressed"
HEAD_RIGHT_TOPIC="/hdas/camera_head/right_raw/image_raw_color/compressed"
WRIST_LEFT_TOPIC="/hdas/camera_wrist_left/color/image_raw/compressed"
WRIST_RIGHT_TOPIC="/hdas/camera_wrist_right/color/image_raw/compressed"
OVERLAY_PID=""
RECORDING=0
DATA_SESSION_STARTED=0
SESSION_DIR="$(mktemp -d /tmp/jimmy_reset_guide.XXXXXX)"
READY_FILE="$SESSION_DIR/reference.json"
WEB_STATUS_FILE="$SESSION_DIR/web_status.json"
COMMAND_FIFO="$SESSION_DIR/commands.fifo"

cleanup() {
  if [[ "$RECORDING" == "1" ]]; then
    echo
    echo "WARNING: the shell exited while an episode may still be recording."
    echo "Use /stop_data_collection to save or /delete_data_collection to reject it."
  fi
  if [[ -n "$OVERLAY_PID" ]] && kill -0 "$OVERLAY_PID" 2>/dev/null; then
    kill -TERM "$OVERLAY_PID" 2>/dev/null || true
    wait "$OVERLAY_PID" 2>/dev/null || true
  fi
  rm -f \
    "$READY_FILE" "$READY_FILE.tmp" \
    "$WEB_STATUS_FILE" "$WEB_STATUS_FILE.tmp" \
    "$COMMAND_FIFO"
  rmdir "$SESSION_DIR" 2>/dev/null || true
  if [[ "$DATA_SESSION_STARTED" == "1" ]] && tmux has-session -t jimmy_data_collection 2>/dev/null; then
    tmux kill-session -t jimmy_data_collection
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

ask_yes_no() {
  local prompt="$1"
  local answer
  while true; do
    read -r -p "$prompt [y/n]: " answer || return 1
    case "${answer,,}" in
      y|yes) return 0 ;;
      n|no) return 1 ;;
      *) echo "Please enter y or n." ;;
    esac
  done
}

camera_streams_have_frames() {
  timeout 25 python3 - "$@" <<'PY'
import os
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

topics = sys.argv[1:]
missing = set(topics)
rclpy.init()
node = rclpy.create_node(f"jimmy_camera_preflight_{os.getpid()}")
subscriptions = []

for topic in topics:
    subscriptions.append(
        node.create_subscription(
            CompressedImage,
            topic,
            lambda _message, current_topic=topic: missing.discard(current_topic),
            qos_profile_sensor_data,
        )
    )

deadline = time.monotonic() + 20.0
while missing and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.2)

node.destroy_node()
rclpy.shutdown()

if missing:
    print("No frame received from:", file=sys.stderr)
    for topic in sorted(missing):
        print(f"  {topic}", file=sys.stderr)
    raise SystemExit(1)
PY
}

collection_services() {
  ros2 service list --no-daemon --spin-time 5 2>/dev/null
}

has_required_collection_services() {
  local services="$1"
  local service
  for service in /start_data_collection /stop_data_collection /delete_data_collection; do
    grep -qx "$service" <<<"$services" || return 1
  done
}

call_trigger() {
  local service="$1"
  local output
  output="$(timeout 30 ros2 service call "$service" std_srvs/srv/Trigger "{}" 2>&1)"
  local status=$?
  echo "$output"
  if [[ $status -ne 0 ]] || ! grep -Eq 'success=(True|true)|success:[[:space:]]*true' <<<"$output"; then
    echo "ERROR: service call failed: $service" >&2
    return 1
  fi
  return 0
}

send_overlay_command() {
  local command="$1"
  if [[ -z "$OVERLAY_PID" ]] || ! kill -0 "$OVERLAY_PID" 2>/dev/null; then
    echo "ERROR: reset-overlay process is not running." >&2
    return 1
  fi
  timeout 2 bash -c 'printf "%s\n" "$1" > "$2"' \
    _ "$command" "$COMMAND_FIFO"
}

if [[ "$DISPLAY_MODE" == "opencv" && ! -S /tmp/.X11-unix/X0 ]]; then
  echo "ERROR: local display :0 is unavailable; cannot open the reset-guide window." >&2
  echo "Use --display web for a read-only browser preview through SSH." >&2
  exit 1
fi

if ! ss -H -lun 'sport = :11811' | grep -q .; then
  echo "ERROR: Fast DDS discovery server is not running on UDP port 11811." >&2
  echo "Start the normal Galaxea robot session first." >&2
  exit 1
fi

if ! camera_streams_have_frames "$HEAD_LEFT_TOPIC"; then
  if pgrep -f '/home/nvidia/jimmy/native_head_rectifier.py' >/dev/null; then
    echo "ERROR: a corrected-head process exists, but no left-camera frame arrived." >&2
    echo "Refusing to start a duplicate rectifier that would contend for /dev/video0." >&2
    echo "Inspect the existing rectifier tmux session and restart it if needed." >&2
    exit 1
  fi

  echo "Corrected head camera is not running; starting it in tmux..."
  if tmux has-session -t jimmy_rectified_head 2>/dev/null; then
    tmux kill-session -t jimmy_rectified_head
  fi
  tmux new-session -d -s jimmy_rectified_head \
    "/home/nvidia/jimmy/switch_to_rectified_head.sh"

  if ! camera_streams_have_frames "$HEAD_LEFT_TOPIC"; then
    echo "ERROR: corrected head camera started, but produced no left-camera frame." >&2
    echo "Inspect: tmux attach -t jimmy_rectified_head" >&2
    exit 1
  fi
fi

if ! camera_streams_have_frames \
  "$HEAD_LEFT_TOPIC" \
  "$HEAD_RIGHT_TOPIC" \
  "$WRIST_LEFT_TOPIC" \
  "$WRIST_RIGHT_TOPIC"; then
  echo "ERROR: one or more required camera streams are not delivering frames." >&2
  echo "The script will not record an episode without all four cameras." >&2
  exit 1
fi

services="$(collection_services)"
if ! has_required_collection_services "$services"; then
  if pgrep -f '/data_collection/.*/data_collection_node_wrapper.py' >/dev/null; then
    echo "ERROR: the vendor data-collection process exists, but its services were not discovered." >&2
    echo "Refusing to launch a duplicate data-collection node." >&2
    exit 1
  fi

  echo "Data-collection service is not running; starting the vendor launcher in tmux..."
  if tmux has-session -t jimmy_data_collection 2>/dev/null; then
    tmux kill-session -t jimmy_data_collection
  fi
  tmux new-session -d -s jimmy_data_collection \
    "source /opt/ros/humble/setup.bash; source /home/nvidia/galaxea/install/setup.bash; export ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE; exec ros2 launch data_collection data_collection_launch.py"
  DATA_SESSION_STARTED=1
  for _ in $(seq 1 6); do
    services="$(collection_services)"
    has_required_collection_services "$services" && break
  done
fi

for service in /start_data_collection /stop_data_collection /delete_data_collection; do
  if ! grep -qx "$service" <<<"$services"; then
    echo "ERROR: required service is unavailable: $service" >&2
    exit 1
  fi
done

echo
if [[ "$DISPLAY_MODE" == "opencv" ]]; then
  echo "Opening the object-reset reference window on DISPLAY=$DISPLAY"
  echo "First capture and accept a reference photo in that window."
  python3 /home/nvidia/jimmy/head_reset_overlay.py \
    --ready-file "$READY_FILE" &
  OVERLAY_PID=$!
else
  mkfifo -m 600 "$COMMAND_FIFO"
  echo "Starting the read-only object-reset web preview."
  python3 /home/nvidia/jimmy/head_reset_web_overlay.py \
    --ready-file "$READY_FILE" \
    --status-file "$WEB_STATUS_FILE" \
    --command-fifo "$COMMAND_FIFO" \
    --port "$WEB_PORT" \
    --fps "$WEB_FPS" &
  OVERLAY_PID=$!

  for _ in $(seq 1 50); do
    [[ -s "$WEB_STATUS_FILE" ]] && break
    if ! kill -0 "$OVERLAY_PID" 2>/dev/null; then
      wait "$OVERLAY_PID" || true
      echo "ERROR: web reset overlay exited during startup." >&2
      exit 1
    fi
    sleep 0.2
  done
  if [[ ! -s "$WEB_STATUS_FILE" ]]; then
    echo "ERROR: web reset overlay did not become ready within 10 seconds." >&2
    exit 1
  fi

  echo
  echo "On the SSH client, keep a tunnel open:"
  echo "  ssh -L ${WEB_PORT}:localhost:${WEB_PORT} nvidia@100.106.44.120"
  echo "Then open this separate browser window:"
  echo "  http://localhost:${WEB_PORT}"
  echo "The browser is display-only; all reset decisions stay in this terminal."

  while [[ ! -s "$READY_FILE" ]]; do
    read -r -p "Press ENTER to capture the current reset reference, or q to cancel: " capture_answer
    if [[ "${capture_answer,,}" == "q" ]]; then
      send_overlay_command quit || true
      echo "Reset-reference capture cancelled."
      exit 1
    fi
    send_overlay_command capture || exit 1

    while true; do
      read -r -p "Inspect the browser overlay: accept, retake, or cancel? [y/r/q]: " reference_answer
      case "${reference_answer,,}" in
        y|yes)
          send_overlay_command accept || exit 1
          break
          ;;
        r|retake)
          send_overlay_command retake || exit 1
          break
          ;;
        q|quit)
          send_overlay_command quit || true
          echo "Reset-reference capture cancelled."
          exit 1
          ;;
        *) echo "Please enter y, r, or q." ;;
      esac
    done

    if [[ "${reference_answer,,}" == "y" || "${reference_answer,,}" == "yes" ]]; then
      for _ in $(seq 1 25); do
        [[ -s "$READY_FILE" ]] && break
        if ! kill -0 "$OVERLAY_PID" 2>/dev/null; then
          wait "$OVERLAY_PID" || true
          echo "ERROR: web reset overlay exited before accepting the reference." >&2
          exit 1
        fi
        sleep 0.2
      done
      if [[ ! -s "$READY_FILE" ]]; then
        echo "ERROR: reset reference was not accepted within 5 seconds." >&2
        exit 1
      fi
    fi
  done
fi

while [[ ! -s "$READY_FILE" ]]; do
  if ! kill -0 "$OVERLAY_PID" 2>/dev/null; then
    wait "$OVERLAY_PID" || true
    echo "ERROR: reset-guide window closed before a reference was accepted." >&2
    exit 1
  fi
  sleep 0.2
done

echo
echo "Reset reference accepted:"
python3 - "$READY_FILE" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
print("  " + data["reference_path"])
PY

episode_number=1
while true; do
  echo
  if ! ask_yes_no "Start episode $episode_number? Enter n to end the overall collection"; then
    break
  fi

  if ! call_trigger /start_data_collection; then
    echo "Episode did not start; returning to the menu."
    continue
  fi
  RECORDING=1

  echo
  read -r -p "Recording episode $episode_number. Press ENTER when the task is complete..." _

  if ask_yes_no "Accept and save episode $episode_number"; then
    if call_trigger /stop_data_collection; then
      RECORDING=0
      echo "Episode $episode_number saved."
      episode_number=$((episode_number + 1))
    else
      echo "The stop call failed; recording state is uncertain." >&2
      exit 1
    fi
  else
    if call_trigger /delete_data_collection; then
      RECORDING=0
      echo "Episode $episode_number rejected and deleted."
    else
      echo "The delete call failed; recording state is uncertain." >&2
      exit 1
    fi
  fi
done

echo
echo "Overall data-collection loop ended."
echo "The corrected camera remains running in its existing tmux session."
echo "To restore the vendor distorted camera later, run:"
echo "  /home/nvidia/jimmy/restore_vendor_head_camera.sh"
