#!/bin/bash
# One-command corrected-camera data collection with an object-reset overlay.
# This script never launches or publishes robot motion commands.
set -o pipefail

source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-01}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/nvidia/.Xauthority}"

HEAD_LEFT_TOPIC="/hdas/camera_head/left_raw/image_raw_color/compressed"
HEAD_RIGHT_TOPIC="/hdas/camera_head/right_raw/image_raw_color/compressed"
WRIST_LEFT_TOPIC="/hdas/camera_wrist_left/color/image_raw/compressed"
WRIST_RIGHT_TOPIC="/hdas/camera_wrist_right/color/image_raw/compressed"
OVERLAY_PID=""
RECORDING=0
DATA_SESSION_STARTED=0
READY_FILE="$(mktemp /tmp/jimmy_reset_guide.XXXXXX.json)"
rm -f "$READY_FILE"

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
  rm -f "$READY_FILE" "$READY_FILE.tmp"
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

topic_has_publisher() {
  local topic="$1"
  ros2 topic info "$topic" 2>/dev/null | grep -Eq 'Publisher count: [1-9][0-9]*'
}

call_trigger() {
  local service="$1"
  local output
  output="$(ros2 service call "$service" std_srvs/srv/Trigger "{}" 2>&1)"
  local status=$?
  echo "$output"
  if [[ $status -ne 0 ]] || ! grep -Eq 'success=(True|true)|success:[[:space:]]*true' <<<"$output"; then
    echo "ERROR: service call failed: $service" >&2
    return 1
  fi
  return 0
}

if [[ ! -S /tmp/.X11-unix/X0 ]]; then
  echo "ERROR: local display :0 is unavailable; cannot open the reset-guide window." >&2
  exit 1
fi

if ! ss -lun | awk '$5 ~ /:11811$/ {found=1} END {exit !found}'; then
  echo "ERROR: Fast DDS discovery server is not running on port 11811." >&2
  echo "Start the normal Galaxea robot session first." >&2
  exit 1
fi

if ! ros2 topic list 2>/dev/null | grep -qx '/jimmy/head_rectifier/status'; then
  echo "Corrected head camera is not running; starting it in tmux..."
  if tmux has-session -t jimmy_rectified_head 2>/dev/null; then
    tmux kill-session -t jimmy_rectified_head
  fi
  tmux new-session -d -s jimmy_rectified_head \
    "/home/nvidia/jimmy/switch_to_rectified_head.sh"

  for _ in $(seq 1 60); do
    ros2 topic list 2>/dev/null | grep -qx '/jimmy/head_rectifier/status' && break
    sleep 0.5
  done
fi

if ! ros2 topic list 2>/dev/null | grep -qx '/jimmy/head_rectifier/status'; then
  echo "ERROR: corrected head camera did not start." >&2
  echo "Inspect: tmux attach -t jimmy_rectified_head" >&2
  exit 1
fi

missing_topics=()
for topic in "$HEAD_LEFT_TOPIC" "$HEAD_RIGHT_TOPIC" "$WRIST_LEFT_TOPIC" "$WRIST_RIGHT_TOPIC"; do
  topic_has_publisher "$topic" || missing_topics+=("$topic")
done
if ((${#missing_topics[@]})); then
  echo "ERROR: required camera publishers are missing:" >&2
  printf '  %s\n' "${missing_topics[@]}" >&2
  echo "The script will not record an episode without all four cameras." >&2
  exit 1
fi

if ! ros2 service list 2>/dev/null | grep -qx '/start_data_collection'; then
  echo "Data-collection service is not running; starting the vendor launcher in tmux..."
  if tmux has-session -t jimmy_data_collection 2>/dev/null; then
    tmux kill-session -t jimmy_data_collection
  fi
  tmux new-session -d -s jimmy_data_collection \
    "source /opt/ros/humble/setup.bash; source /home/nvidia/galaxea/install/setup.bash; export ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE; exec ros2 launch data_collection data_collection_launch.py"
  DATA_SESSION_STARTED=1
  for _ in $(seq 1 60); do
    ros2 service list 2>/dev/null | grep -qx '/start_data_collection' && break
    sleep 0.5
  done
fi

for service in /start_data_collection /stop_data_collection /delete_data_collection; do
  if ! ros2 service list 2>/dev/null | grep -qx "$service"; then
    echo "ERROR: required service is unavailable: $service" >&2
    exit 1
  fi
done

echo
echo "Opening the object-reset reference window on DISPLAY=$DISPLAY"
echo "First capture and accept a reference photo in that window."
python3 /home/nvidia/jimmy/head_reset_overlay.py \
  --ready-file "$READY_FILE" &
OVERLAY_PID=$!

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
echo "The corrected camera remains available in tmux session: jimmy_rectified_head"
echo "To restore the vendor distorted camera later, run:"
echo "  /home/nvidia/jimmy/restore_vendor_head_camera.sh"
