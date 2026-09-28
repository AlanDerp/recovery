#!/usr/bin/env bash
# One-click Agilex/Piper tennis-ball pipeline using only the top camera (camera_h).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

CONFIG_PATH="${CONFIG_PATH:-env_configs/agilex/agilex_real_tennis_ball_in_box_top_camera_single_turn.yaml}"
CAMERA_SERIAL="${CAMERA_SERIAL:-CC1WC5200ZV}"
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-/home/agilex/checkpoints/sam3}"
USE_GPT_DIRECT="${USE_GPT_DIRECT:-0}"
LLM_PROVIDER="${LLM_PROVIDER:-openrouter}"
OPENROUTER_MODEL="${OPENROUTER_MODEL:-qwen/qwen3.7-max}"
OPENROUTER_SERVER_URL="${OPENROUTER_SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
GPT_MODEL="${GPT_MODEL:-gpt-5.5}"
GPT_SERVER_URL="${GPT_SERVER_URL:-https://usezyla.com/v1/chat/completions}"
QWEN_MODEL="${QWEN_MODEL:-${OPENROUTER_MODEL}}"
QWEN_SERVER_URL="${QWEN_SERVER_URL:-${OPENROUTER_SERVER_URL}}"

export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/capx_ros_logs}"
mkdir -p "${ROS_LOG_DIR}"
source /opt/ros/noetic/setup.bash
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash

resolve_python() {
  if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
    printf '%s\n' "${PYTHON}"
    return 0
  fi
  local repo_root
  repo_root="$(git rev-parse --show-toplevel)"
  local candidate
  for candidate in \
    "${repo_root}/.venv/bin/python" \
    "/home/agilex/miniconda3/envs/capx/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  echo "[error] python not found. Set PYTHON=/path/to/python or create .venv" >&2
  return 1
}

PYTHON="$(resolve_python)"
export PYTHON
export PATH="$(dirname "${PYTHON}"):${PATH}"

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local name="$3"
  local deadline=$((SECONDS + ${4:-120}))
  while (( SECONDS < deadline )); do
    if timeout 1 bash -c ":</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
      echo "[ok] ${name} is listening on ${host}:${port}"
      return 0
    fi
    sleep 1
  done
  echo "[error] timed out waiting for ${name} on ${host}:${port}" >&2
  return 1
}

wait_for_topic_once() {
  local topic="$1"
  local timeout_s="${2:-20}"
  "${PYTHON}" - "$topic" "$timeout_s" <<'PY'
import sys

import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image

topic = sys.argv[1]
timeout_s = float(sys.argv[2])
msg_type = Image if ("camera_h" in topic or "camera_ugreen" in topic) else JointState
if not rospy.core.is_initialized():
    rospy.init_node("capx_wait_for_topic_once", anonymous=True, disable_signals=True)
rospy.wait_for_message(topic, msg_type, timeout=timeout_s)
print(f"[ok] received one message from {topic}")
PY
}

start_top_camera_preview() {
  local preview_enabled="${TOP_CAMERA_PREVIEW:-1}"
  if [[ "${preview_enabled}" != "1" ]]; then
    echo "[skip] top camera preview disabled (TOP_CAMERA_PREVIEW=${preview_enabled})"
    return 0
  fi

  local preview_output="${TOP_CAMERA_PREVIEW_OUTPUT:-outputs/agilex_top_camera/latest_rgb.png}"
  local preview_show="${TOP_CAMERA_PREVIEW_SHOW:-auto}"
  local preview_watch="${TOP_CAMERA_PREVIEW_WATCH:-1}"
  local preview_interval="${TOP_CAMERA_PREVIEW_INTERVAL:-1}"
  mkdir -p "$(dirname "${preview_output}")"

  echo "[preview] saving top camera frame to ${preview_output}"
  if ! "${PYTHON}" scripts/preview_agilex_top_camera.py \
    --mode once \
    --output "${preview_output}" \
    --show "${preview_show}" \
    --hold-s "${TOP_CAMERA_PREVIEW_HOLD_S:-8}"; then
    echo "[warn] top camera preview snapshot failed; continuing startup" >&2
  fi

  if [[ "${preview_watch}" == "1" ]]; then
    echo "[preview] background watch updating ${preview_output} every ${preview_interval}s"
    "${PYTHON}" scripts/preview_agilex_top_camera.py \
      --mode watch \
      --output "${preview_output}" \
      --show "${preview_show}" \
      --interval "${preview_interval}" \
      > logs/top_camera_preview.log 2>&1 &
    echo "[preview] watch pid=$! log=logs/top_camera_preview.log"
  fi
}

start_side_usb_camera() {
  local enabled="${SIDE_USB_CAMERA:-1}"
  if [[ "${enabled}" != "1" ]]; then
    echo "[skip] side USB camera disabled (SIDE_USB_CAMERA=${enabled})"
    return 0
  fi
  echo "[start] side USB camera (UGREEN) for pre-grasp depth check"
  UGREEN_DEVICE="${UGREEN_DEVICE:-/dev/video2}" \
  UGREEN_IMAGE_TOPIC="${UGREEN_IMAGE_TOPIC:-/camera_ugreen/color/image_raw}" \
  bash scripts/start_ugreen_camera.sh
}

start_side_camera_preview() {
  local enabled="${SIDE_CAMERA_PREVIEW:-1}"
  if [[ "${enabled}" != "1" ]]; then
    echo "[skip] side camera preview disabled (SIDE_CAMERA_PREVIEW=${enabled})"
    return 0
  fi
  local side_topic="${UGREEN_IMAGE_TOPIC:-/camera_ugreen/color/image_raw}"
  local preview_output="${SIDE_CAMERA_PREVIEW_OUTPUT:-outputs/agilex_side_camera/latest_rgb.png}"
  local preview_interval="${SIDE_CAMERA_PREVIEW_INTERVAL:-1}"
  mkdir -p "$(dirname "${preview_output}")"
  echo "[preview] side camera watch -> ${preview_output} every ${preview_interval}s"
  "${PYTHON}" scripts/preview_agilex_top_camera.py \
    --topic "${side_topic}" \
    --mode watch \
    --output "${preview_output}" \
    --show "${SIDE_CAMERA_PREVIEW_SHOW:-0}" \
    --interval "${preview_interval}" \
    > logs/side_camera_preview.log 2>&1 &
  echo "[preview] side watch pid=$! log=logs/side_camera_preview.log"
}

run_pre_grasp_depth_check() {
  local enabled="${PRE_GRASP_DEPTH_CHECK:-1}"
  if [[ "${enabled}" != "1" ]]; then
    echo "[skip] pre-grasp depth check disabled (PRE_GRASP_DEPTH_CHECK=${enabled})"
    return 0
  fi
  echo "[check] pre-grasp depth alignment (top RGB-D + side USB)"
  if "${PYTHON}" scripts/verify_agilex_pregrasp_depth.py \
    --config "${CONFIG_PATH}"; then
    echo "[ok] pre-grasp depth check passed"
    return 0
  fi
  if [[ "${PRE_GRASP_DEPTH_CHECK_STRICT:-0}" == "1" ]]; then
    echo "[error] pre-grasp depth check failed (strict mode)" >&2
    exit 1
  fi
  echo "[warn] pre-grasp depth check failed; continuing pipeline" >&2
}

if ! rostopic list >/dev/null 2>&1; then
  echo "[start] roscore"
  roscore > logs/roscore_tennis_top_camera.log 2>&1 &
  sleep 3
else
  echo "[skip] roscore already available"
fi

if ! rostopic list 2>/dev/null | grep -qx "/camera_h/color/image_raw"; then
  echo "[start] top camera camera_h only"
  roslaunch astra_camera dabai.launch \
    camera_name:=camera_h \
    serial_number:="${CAMERA_SERIAL}" \
    device_num:=1 \
    > logs/camera_h_top_only.log 2>&1 &
  wait_for_topic_once /camera_h/color/image_raw 30
else
  echo "[skip] /camera_h/color/image_raw already exists"
fi

start_top_camera_preview

if ! rostopic echo /puppet/joint_left -n 1 >/dev/null 2>&1; then
  echo "[start] Piper left arm"
  bash scripts/start_piper_left.sh > logs/piper_left_tennis_top_camera.log 2>&1 &
  wait_for_topic_once /puppet/joint_left 30
else
  echo "[skip] /puppet/joint_left already publishing"
fi

if ! timeout 1 bash -c ":</dev/tcp/127.0.0.1/8114" >/dev/null 2>&1; then
  echo "[start] SAM3 server on 8114"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
  uv run --no-sync --active python -m capx.serving.launch_sam3_server \
    --device cuda \
    --port 8114 \
    --host 127.0.0.1 \
    --checkpoint-path "${SAM3_CHECKPOINT_PATH}" \
    > logs/sam3_tennis_top_camera.log 2>&1 &
fi
wait_for_tcp 127.0.0.1 8114 SAM3 180

if ! timeout 1 bash -c ":</dev/tcp/127.0.0.1/8115" >/dev/null 2>&1; then
  echo "[start] GraspNet server on 8115"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
  uv run --no-sync --active python -m capx.serving.launch_contact_graspnet_server \
    --device cuda \
    --port 8115 \
    --host 127.0.0.1 \
    > logs/graspnet_tennis_top_camera.log 2>&1 &
fi
wait_for_tcp 127.0.0.1 8115 GraspNet 180

start_side_usb_camera
run_pre_grasp_depth_check
start_side_camera_preview

if [[ "${LLM_PROVIDER}" == "openrouter" ]]; then
  if [[ -z "${OPENROUTER_API_KEY:-}" && -f .openrouterkey ]]; then
    OPENROUTER_API_KEY="$(grep -v '^[[:space:]]*#' .openrouterkey | awk 'NF {print; exit}')"
  fi
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "[error] LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set and .openrouterkey is empty." >&2
    exit 1
  fi
  if ! curl -sf --connect-timeout 2 http://127.0.0.1:8110/health >/dev/null 2>&1; then
    echo "[start] OpenRouter proxy on 8110"
    UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    uv run --no-sync --active python -m capx.serving.openrouter_server \
      --key-file .openrouterkey \
      --port 8110 \
      --host 127.0.0.1 \
      > logs/openrouter_8110_tennis_top_camera.log 2>&1 &
  else
    echo "[skip] OpenRouter proxy already available on 8110"
  fi
  wait_for_tcp 127.0.0.1 8110 OpenRouter 120
fi

export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:10808}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:10808}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

if [[ "${USE_GPT_DIRECT}" == "1" ]]; then
  if [[ -z "${GPT_API_KEY:-}" && -f .gpt_api_key ]]; then
    GPT_API_KEY="$(grep -v '^[[:space:]]*#' .gpt_api_key | awk 'NF {print; exit}')"
  fi
  if [[ -z "${GPT_API_KEY:-}" ]]; then
    echo "[error] USE_GPT_DIRECT=1 but GPT_API_KEY is not set and .gpt_api_key is empty." >&2
    echo "Set GPT_API_KEY, write .gpt_api_key, or run with USE_GPT_DIRECT=0 for local 8110." >&2
    exit 1
  fi
  echo "[run] tennis-ball task with GPT direct API"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
  uv run --no-sync --active capx/envs/launch.py \
    --config-path "${CONFIG_PATH}" \
    --model "${GPT_MODEL}" \
    --server-url "${GPT_SERVER_URL}" \
    --api-key "${GPT_API_KEY}"
else
  echo "[run] tennis-ball task with OpenRouter ${QWEN_MODEL} at ${QWEN_SERVER_URL}"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
  uv run --no-sync --active capx/envs/launch.py \
    --config-path "${CONFIG_PATH}" \
    --model "${QWEN_MODEL}" \
    --server-url "${QWEN_SERVER_URL}"
fi
