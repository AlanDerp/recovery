#!/usr/bin/env bash
# One-click Agilex tennis-ball pipeline with dual-model ensemble (gpt-5.5 + OpenRouter qwen3.7-max).
# Uses launch_agilex_ensemble.py (copy) with separate API keys per model.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

CONFIG_PATH="${CONFIG_PATH:-env_configs/agilex/agilex_real_tennis_ball_in_box_top_camera_ensemble_gpt_codex53.yaml}"
CAMERA_SERIAL="${CAMERA_SERIAL:-CC1WC5200ZV}"
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-/home/agilex/checkpoints/sam3}"
USE_GPT_DIRECT="${USE_GPT_DIRECT:-1}"
GPT_MODEL="${GPT_MODEL:-gpt-5.5}"
GPT_SERVER_URL="${GPT_SERVER_URL:-https://usezyla.com/v1/chat/completions}"
LLM_PROVIDER="${LLM_PROVIDER:-openrouter}"
OPENROUTER_MODEL="${OPENROUTER_MODEL:-qwen/qwen3.7-max}"
OPENROUTER_SERVER_URL="${OPENROUTER_SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
ENSEMBLE_SECONDARY_MODEL="${ENSEMBLE_SECONDARY_MODEL:-${OPENROUTER_MODEL}}"
ENSEMBLE_SECONDARY_SERVER_URL="${ENSEMBLE_SECONDARY_SERVER_URL:-${OPENROUTER_SERVER_URL}}"
QWEN_MODEL="${QWEN_MODEL:-qwen/qwen3.7-max}"
QWEN_SERVER_URL="${QWEN_SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
AGILEX_MULTITURN_LIMIT="${AGILEX_MULTITURN_LIMIT:-15}"
AGILEX_ERROR_RETRY_LIMIT="${AGILEX_ERROR_RETRY_LIMIT:-8}"
AGILEX_FAILURE_VISION="${AGILEX_FAILURE_VISION:-1}"
AGILEX_FAILURE_VISION_MAX_FRAMES="${AGILEX_FAILURE_VISION_MAX_FRAMES:-4}"
AGILEX_STRATEGY_SHIFT="${AGILEX_STRATEGY_SHIFT:-1}"
AGILEX_STRATEGY_SHIFT_AFTER_FAILURES="${AGILEX_STRATEGY_SHIFT_AFTER_FAILURES:-2}"
AGILEX_STRATEGY_SHIFT_BASE_FAILURES="${AGILEX_STRATEGY_SHIFT_BASE_FAILURES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/gpt_agilex_real_tennis_ball_top_camera_ensemble_gpt55_qwen32b}"
MAX_OUTER_ATTEMPTS="${MAX_OUTER_ATTEMPTS:-3}"
OUTER_ANALYSIS_MODEL="${OUTER_ANALYSIS_MODEL:-google/gemini-2.5-flash-lite}"
OUTER_RETRY_HINTS_FILE="${OUTER_RETRY_HINTS_FILE:-${OUTPUT_DIR}/outer_retry_hints.txt}"
OUTER_FAILURE_LOG="${OUTER_FAILURE_LOG:-logs/outer_failure_analysis.log}"
OUTER_FAILURE_HISTORY="${OUTER_FAILURE_HISTORY:-${OUTPUT_DIR}/outer_failure_history.txt}"
PIPELINE_LOG="${PIPELINE_LOG:-logs/tennis_top_camera_ensemble_pipeline.log}"

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
msg_type = Image if "camera_h" in topic else JointState
if not rospy.core.is_initialized():
    rospy.init_node("capx_wait_for_topic_once", anonymous=True, disable_signals=True)
rospy.wait_for_message(topic, msg_type, timeout=timeout_s)
print(f"[ok] received one message from {topic}")
PY
}

topic_has_message() {
  local topic="$1"
  local timeout_s="${2:-5}"
  "${PYTHON}" - "$topic" "$timeout_s" <<'PY'
import sys

import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image

topic = sys.argv[1]
timeout_s = float(sys.argv[2])
msg_type = Image if "camera_h" in topic else JointState
if not rospy.core.is_initialized():
    rospy.init_node("capx_topic_has_message", anonymous=True, disable_signals=True)
try:
    rospy.wait_for_message(topic, msg_type, timeout=timeout_s)
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

ensure_camera_rgb() {
  if topic_has_message /camera_h/color/image_raw 5; then
    echo "[skip] /camera_h/color/image_raw already publishing"
    return 0
  fi
  echo "[warn] /camera_h/color/image_raw not publishing; restarting camera_h"
  pkill -f 'roslaunch astra_camera dabai.launch camera_name:=camera_h' 2>/dev/null || true
  pkill -f 'astra_camera_node.*camera_h' 2>/dev/null || true
  sleep 2
  echo "[start] top camera camera_h only"
  roslaunch astra_camera dabai.launch \
    camera_name:=camera_h \
    serial_number:="${CAMERA_SERIAL}" \
    device_num:=1 \
    > logs/camera_h_top_only_ensemble.log 2>&1 &
  wait_for_topic_once /camera_h/color/image_raw 30
}

ensure_piper_left() {
  if topic_has_message /puppet/joint_left 5; then
    echo "[skip] /puppet/joint_left already publishing"
    return 0
  fi
  echo "[warn] /puppet/joint_left not publishing; restarting Piper left arm"
  pkill -f 'roslaunch .*piper_left_ms.launch' 2>/dev/null || true
  pkill -f 'piper_start_ms_node.py.*puppet/joint_left' 2>/dev/null || true
  sleep 2
  echo "[start] Piper left arm"
  bash scripts/start_piper_left.sh > logs/piper_left_tennis_top_camera_ensemble.log 2>&1 &
  wait_for_topic_once /puppet/joint_left 30
}

resolve_attempt_trial_root() {
  local attempt_output="$1"
  local model_name="${2:-${GPT_MODEL}}"
  local attempt_parent attempt_basename
  attempt_parent="$(dirname "${attempt_output}")"
  attempt_basename="$(basename "${attempt_output}")"
  printf '%s/%s/%s' "${attempt_parent}" "${model_name//\//_}" "${attempt_basename}"
}

trial_succeeded() {
  local out_dir="$1"
  local trial_root
  trial_root="$(resolve_attempt_trial_root "${out_dir}")"
  "${PYTHON}" - "$out_dir" "$trial_root" <<'PY'
import glob
import json
import sys

out_dir, trial_root = sys.argv[1:3]

visual_reports = glob.glob(f"{trial_root}/**/agilex_visual_success.json", recursive=True)
if not visual_reports:
    visual_reports = glob.glob(f"{out_dir}/**/agilex_visual_success.json", recursive=True)
if visual_reports:
    visual_reports.sort(key=lambda p: p, reverse=True)
    with open(visual_reports[0], encoding="utf-8") as f:
        report = json.load(f)
    raise SystemExit(0 if report.get("success") else 1)

patterns = [
    f"{trial_root}/trial_*_reward_1.000_taskcompleted_1",
    f"{trial_root}/trial_*_sandboxrc_0_reward_1.000_taskcompleted_1",
    f"{out_dir}/**/trial_*_reward_1.000_taskcompleted_1",
]
for pattern in patterns:
    if glob.glob(pattern, recursive=True):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

analyze_outer_failure() {
  local attempt="$1"
  local attempt_output="$2"
  local trial_root
  trial_root="$(resolve_attempt_trial_root "${attempt_output}")"
  echo "[outer] analyzing failed attempt ${attempt} using logs and camera images"
  echo "[outer] trial root: ${trial_root}"
  echo "[outer] failure log: ${OUTER_FAILURE_LOG}"
  echo "[outer] failure history: ${OUTER_FAILURE_HISTORY}"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
  OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
  OUTER_FAILURE_LOG="${OUTER_FAILURE_LOG}" \
  OUTER_FAILURE_HISTORY="${OUTER_FAILURE_HISTORY}" \
  uv run --no-sync --active python -m capx.envs.agilex_outer_failure_analysis \
    --output-dir "${trial_root}" \
    --attempt "${attempt}" \
    --hints-file "${OUTER_RETRY_HINTS_FILE}" \
    --analysis-file "${OUTPUT_DIR}/outer_failure_analysis_attempt_$(printf '%02d' "${attempt}").txt" \
    --log-file "${OUTER_FAILURE_LOG}" \
    --history-file "${OUTER_FAILURE_HISTORY}" \
    --model "${OUTER_ANALYSIS_MODEL}" \
    --server-url "${OPENROUTER_SERVER_URL}"
}

if ! rostopic list >/dev/null 2>&1; then
  echo "[start] roscore"
  roscore > logs/roscore_tennis_top_camera_ensemble.log 2>&1 &
  sleep 3
else
  echo "[skip] roscore already available"
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
    > logs/sam3_tennis_top_camera_ensemble.log 2>&1 &
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
    > logs/graspnet_tennis_top_camera_ensemble.log 2>&1 &
fi
wait_for_tcp 127.0.0.1 8115 GraspNet 180

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
      > logs/openrouter_8110_ensemble.log 2>&1 &
  else
    echo "[skip] OpenRouter proxy already available on 8110"
  fi
  wait_for_tcp 127.0.0.1 8110 OpenRouter 120
fi

export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:10808}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:10808}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

run_launch() {
  local attempt_output="$1"
  if [[ "${USE_GPT_DIRECT}" == "1" ]]; then
    if [[ -z "${GPT_API_KEY:-}" && -f .gpt_api_key ]]; then
      GPT_API_KEY="$(grep -v '^[[:space:]]*#' .gpt_api_key | awk 'NF {print; exit}')"
    fi
    if [[ -z "${GPT_API_KEY:-}" ]]; then
      echo "[error] USE_GPT_DIRECT=1 but GPT_API_KEY is not set and .gpt_api_key is empty." >&2
      exit 1
    fi
    echo "[run] dual-model ensemble tennis-ball task: ${GPT_MODEL} + ${ENSEMBLE_SECONDARY_MODEL} (${LLM_PROVIDER})"
    UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    GPT_API_KEY="${GPT_API_KEY}" \
    OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
    LLM_PROVIDER="${LLM_PROVIDER}" \
    OPENROUTER_MODEL="${OPENROUTER_MODEL}" \
    AGILEX_OUTER_ATTEMPT="${AGILEX_OUTER_ATTEMPT}" \
    AGILEX_OUTER_RETRY_HINTS_FILE="${OUTER_RETRY_HINTS_FILE}" \
    uv run --no-sync --active capx/envs/launch_agilex_ensemble.py \
      --config-path "${CONFIG_PATH}" \
      --model "${GPT_MODEL}" \
      --server-url "${GPT_SERVER_URL}" \
      --api-key "${GPT_API_KEY}" \
      --output-dir "${attempt_output}"
  else
    echo "[run] ensemble pipeline requires USE_GPT_DIRECT=1 (gpt-5.5 + OpenRouter secondary)" >&2
    exit 1
  fi
}

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs
: > "${PIPELINE_LOG}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1
echo "[outer] pipeline log: ${PIPELINE_LOG}"
echo "[outer] failure analysis log: ${OUTER_FAILURE_LOG}"

attempt=1
while (( attempt <= MAX_OUTER_ATTEMPTS )); do
  export AGILEX_OUTER_ATTEMPT="${attempt}"
  export AGILEX_STRATEGY_SHIFT_BASE_FAILURES=$((attempt - 1))
  attempt_output="${OUTPUT_DIR}/outer_attempt_$(printf '%02d' "${attempt}")"
  mkdir -p "${attempt_output}"
  export AGILEX_OUTER_RETRY_HINTS_FILE="${OUTER_RETRY_HINTS_FILE}"

  echo "[outer] trial attempt ${attempt}/${MAX_OUTER_ATTEMPTS}"
  echo "[outer] output dir: ${attempt_output}"
  echo "[outer] strategy-shift base failures: ${AGILEX_STRATEGY_SHIFT_BASE_FAILURES}"
  if [[ -f "${OUTER_RETRY_HINTS_FILE}" ]]; then
    echo "[outer] cumulative retry hints preview:"
    sed -n '1,40p' "${OUTER_RETRY_HINTS_FILE}"
  else
    echo "[outer] no cumulative retry hints yet"
  fi

  ensure_camera_rgb
  ensure_piper_left

  run_launch "${attempt_output}" || true

  if trial_succeeded "${attempt_output}"; then
    echo "[outer] task completed successfully on attempt ${attempt}"
    exit 0
  fi

  if (( attempt >= MAX_OUTER_ATTEMPTS )); then
    break
  fi

  analyze_outer_failure "${attempt}" "${attempt_output}" || true
  echo "[outer] retrying after arm reset sleep"
  sleep 5
  attempt=$((attempt + 1))
done

echo "[outer] gave up after ${MAX_OUTER_ATTEMPTS} outer attempts" >&2
exit 1
