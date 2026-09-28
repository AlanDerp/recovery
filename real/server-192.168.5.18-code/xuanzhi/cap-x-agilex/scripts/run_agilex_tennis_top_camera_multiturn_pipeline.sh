#!/usr/bin/env bash
# One-click Agilex tennis-ball pipeline with multi-turn LLM error feedback.
# Uses launch_agilex_multiturn.py (copy) so stderr is sent back to the LLM for REGENERATE retries.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

CONFIG_PATH="${CONFIG_PATH:-env_configs/agilex/agilex_real_tennis_ball_in_box_top_camera_multiturn_gpt.yaml}"
CAMERA_SERIAL="${CAMERA_SERIAL:-CC1WC5200ZV}"
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-/home/agilex/checkpoints/sam3}"
USE_GPT_DIRECT="${USE_GPT_DIRECT:-0}"
GPT_MODEL="${GPT_MODEL:-gpt-5.5}"
GPT_SERVER_URL="${GPT_SERVER_URL:-https://usezyla.com/v1/chat/completions}"
QWEN_MODEL="${QWEN_MODEL:-qwen/qwen3.7-max}"
QWEN_SERVER_URL="${QWEN_SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
REFEREE_MODEL="${REFEREE_MODEL:-qwen/qwen3.5-122b-a10b}"
REFEREE_SERVER_URL="${REFEREE_SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
REFEREE_API_KEY="${REFEREE_API_KEY:-${GPT_API_KEY:-}}"
MAX_OUTER_ATTEMPTS="${MAX_OUTER_ATTEMPTS:-3}"
AGILEX_MULTITURN_LIMIT="${AGILEX_MULTITURN_LIMIT:-15}"
AGILEX_ERROR_RETRY_LIMIT="${AGILEX_ERROR_RETRY_LIMIT:-8}"
AGILEX_FAILURE_VISION="${AGILEX_FAILURE_VISION:-1}"
AGILEX_FAILURE_VISION_MAX_FRAMES="${AGILEX_FAILURE_VISION_MAX_FRAMES:-4}"
AGILEX_STRATEGY_SHIFT="${AGILEX_STRATEGY_SHIFT:-1}"
AGILEX_STRATEGY_SHIFT_AFTER_FAILURES="${AGILEX_STRATEGY_SHIFT_AFTER_FAILURES:-2}"
AGILEX_STRATEGY_SHIFT_BASE_FAILURES="${AGILEX_STRATEGY_SHIFT_BASE_FAILURES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qwen_agilex_real_tennis_ball_top_camera_multiturn}"

export AGILEX_MULTITURN_LIMIT
export AGILEX_ERROR_RETRY_LIMIT
export AGILEX_FAILURE_VISION
export AGILEX_FAILURE_VISION_MAX_FRAMES
export AGILEX_STRATEGY_SHIFT
export AGILEX_STRATEGY_SHIFT_AFTER_FAILURES
export AGILEX_STRATEGY_SHIFT_BASE_FAILURES
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
# Allow accidental comma-separated model strings; use the first valid token.
REFEREE_MODEL="${REFEREE_MODEL%%,*}"

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

start_side_usb_camera() {
  local enabled="${SIDE_USB_CAMERA:-1}"
  if [[ "${enabled}" != "1" ]]; then
    echo "[skip] side USB camera disabled (SIDE_USB_CAMERA=${enabled})"
    return 0
  fi
  echo "[start] side USB camera (UGREEN) for multi-turn correction"
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
    > logs/side_camera_preview_multiturn.log 2>&1 &
  echo "[preview] side watch pid=$! log=logs/side_camera_preview_multiturn.log"
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

trial_succeeded() {
  local out_dir="$1"
  "${PYTHON}" - "$out_dir" <<'PY'
import glob
import json
import os
import sys

out_dir = sys.argv[1]
base = os.path.normpath(out_dir)
parent = os.path.dirname(base)
name = os.path.basename(base)
roots = [
    base,
    os.path.join(parent, "*", name),  # runner inserts model folder here
]

trial_dirs: list[str] = []
for root in roots:
    trial_dirs.extend(glob.glob(os.path.join(root, "**", "trial_*"), recursive=True))

for trial_dir in sorted(set(trial_dirs)):
    if not os.path.isdir(trial_dir):
        continue
    base_name = os.path.basename(trial_dir)
    sandbox_ok = "_sandboxrc_0_" in base_name
    if not sandbox_ok:
        continue

    responses_path = os.path.join(trial_dir, "all_responses.json")
    if os.path.exists(responses_path):
        try:
            with open(responses_path, "r", encoding="utf-8") as f:
                responses = json.load(f)
        except Exception:
            responses = []
        decisions = [str(item.get("decision", "")).lower() for item in responses if isinstance(item, dict)]
        if "finish_with_errors" in decisions:
            continue
        if "finish" in decisions and "_taskcompleted_1" in base_name:
            raise SystemExit(0)

    # Fallback for non-multiturn outputs that still use taskcompleted in path.
    if "_taskcompleted_1" in base_name:
        raise SystemExit(0)

raise SystemExit(1)
PY
}

if ! rostopic list >/dev/null 2>&1; then
  echo "[start] roscore"
  roscore > logs/roscore_tennis_top_camera_multiturn.log 2>&1 &
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
    > logs/camera_h_top_only_multiturn.log 2>&1 &
  wait_for_topic_once /camera_h/color/image_raw 30
else
  echo "[skip] /camera_h/color/image_raw already exists"
fi

if ! rostopic echo /puppet/joint_left -n 1 >/dev/null 2>&1; then
  echo "[start] Piper left arm"
  bash scripts/start_piper_left.sh > logs/piper_left_tennis_top_camera_multiturn.log 2>&1 &
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
    > logs/sam3_tennis_top_camera_multiturn.log 2>&1 &
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
    > logs/graspnet_tennis_top_camera_multiturn.log 2>&1 &
fi
wait_for_tcp 127.0.0.1 8115 GraspNet 180

start_side_usb_camera
run_pre_grasp_depth_check
start_side_camera_preview

export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:10808}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:10808}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

if [[ "${USE_GPT_DIRECT}" != "1" ]]; then
  if [[ -z "${OPENROUTER_API_KEY:-}" && -f .openrouterkey ]]; then
    OPENROUTER_API_KEY="$(grep -v '^[[:space:]]*#' .openrouterkey | awk 'NF {print; exit}')"
  fi
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "[error] USE_GPT_DIRECT=0 requires OPENROUTER_API_KEY (or non-empty .openrouterkey)." >&2
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
      > logs/openrouter_8110_tennis_top_camera_multiturn.log 2>&1 &
  else
    echo "[skip] OpenRouter proxy already available on 8110"
  fi
  wait_for_tcp 127.0.0.1 8110 OpenRouter 120
fi

if [[ "${REFEREE_MODEL}" == gpt-* || "${REFEREE_MODEL}" == openai/* ]]; then
  if [[ -z "${REFEREE_API_KEY:-}" && -f .gpt_api_key ]]; then
    REFEREE_API_KEY="$(grep -v '^[[:space:]]*#' .gpt_api_key | awk 'NF {print; exit}')"
  fi
fi

run_launch() {
  if [[ "${USE_GPT_DIRECT}" == "1" ]]; then
    if [[ -z "${GPT_API_KEY:-}" && -f .gpt_api_key ]]; then
      GPT_API_KEY="$(grep -v '^[[:space:]]*#' .gpt_api_key | awk 'NF {print; exit}')"
    fi
    if [[ -z "${GPT_API_KEY:-}" ]]; then
      echo "[error] USE_GPT_DIRECT=1 but GPT_API_KEY is not set and .gpt_api_key is empty." >&2
      exit 1
    fi
    echo "[run] multi-turn tennis-ball task with GPT (${GPT_MODEL})"
    UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    uv run --no-sync --active capx/envs/launch_agilex_multiturn.py \
      --config-path "${CONFIG_PATH}" \
      --model "${GPT_MODEL}" \
      --server-url "${GPT_SERVER_URL}" \
      --visual-differencing-model "${REFEREE_MODEL}" \
      --visual-differencing-model-server-url "${REFEREE_SERVER_URL}" \
      --visual-differencing-model-api-key "${REFEREE_API_KEY:-}" \
      --api-key "${GPT_API_KEY}" \
      --output-dir "${OUTPUT_DIR}"
  else
    echo "[run] multi-turn tennis-ball task with local LLM ${QWEN_SERVER_URL}"
    UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    uv run --no-sync --active capx/envs/launch_agilex_multiturn.py \
      --config-path "${CONFIG_PATH}" \
      --model "${QWEN_MODEL}" \
      --server-url "${QWEN_SERVER_URL}" \
      --visual-differencing-model "${REFEREE_MODEL}" \
      --visual-differencing-model-server-url "${REFEREE_SERVER_URL}" \
      --visual-differencing-model-api-key "${REFEREE_API_KEY:-}" \
      --output-dir "${OUTPUT_DIR}"
  fi
}

attempt=1
while (( attempt <= MAX_OUTER_ATTEMPTS )); do
  export AGILEX_STRATEGY_SHIFT_BASE_FAILURES=$((attempt - 1))
  echo "[outer] trial attempt ${attempt}/${MAX_OUTER_ATTEMPTS}"
  echo "[outer] strategy-shift base failures: ${AGILEX_STRATEGY_SHIFT_BASE_FAILURES}"
  run_launch || true
  if trial_succeeded "${OUTPUT_DIR}"; then
    echo "[outer] task completed successfully on attempt ${attempt}"
    exit 0
  fi
  echo "[outer] attempt ${attempt} did not complete the task; retrying after arm reset sleep"
  sleep 5
  attempt=$((attempt + 1))
done

echo "[outer] gave up after ${MAX_OUTER_ATTEMPTS} outer attempts" >&2
exit 1
