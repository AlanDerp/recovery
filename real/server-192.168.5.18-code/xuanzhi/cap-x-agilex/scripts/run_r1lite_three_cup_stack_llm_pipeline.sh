#!/usr/bin/env bash
# LLM intent -> validated plan -> hash-bound generated code -> optional execution.
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

MODEL="${MODEL:-gpt-5.5}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
LLM_BASE_URL="${LLM_BASE_URL:-https://api.usezyla.com/v1/}"
API_KEY_FILE="${API_KEY_FILE:-${REPO_ROOT}/.usezyla_key}"
SOURCE_CONFIG_PATH="${SOURCE_CONFIG_PATH:-tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r1lite_three_cup_stack_llm}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"
CUP_STACK_LLM_EXECUTE_ROBOT="${CUP_STACK_LLM_EXECUTE_ROBOT:-0}"
CUP_STACK_STOP_AFTER_FIRST_GRASP="${CUP_STACK_STOP_AFTER_FIRST_GRASP:-0}"
FIXED_CUP_SCENE_PATH="${FIXED_CUP_SCENE_PATH:-outputs/r1lite_three_cup_stack/fixed_cup_scene_user_20260827.json}"
CUP_STACK_ALLOW_SAFE_STACK_START="${CUP_STACK_ALLOW_SAFE_STACK_START:-0}"

: "${CUP_HEIGHT_M:?Set measured CUP_HEIGHT_M}"
: "${CUP_DIAMETER_M:?Set measured CUP_DIAMETER_M}"
: "${CUP_GRIPPER_CLOSE_POSITION:?Set a tested CUP_GRIPPER_CLOSE_POSITION}"

for path in "${API_KEY_FILE}" "${SOURCE_CONFIG_PATH}" "${FIXED_CUP_SCENE_PATH}"; do
  [[ -r "${path}" ]] || {
    echo "[error] required file is unreadable: ${path}" >&2
    exit 1
  }
done

if [[ "${ALLOW_ROBOT_STACK_RESTART:-0}" == "1" ]]; then
  echo "[error] full vendor startup is forbidden for the cup task" >&2
  echo "[error] use CUP_STACK_ALLOW_SAFE_STACK_START=1 instead" >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

if ! timeout 3 ros2 topic echo --once \
  /hdas/feedback_arm_right sensor_msgs/msg/JointState \
  >/dev/null 2>&1; then
  if [[ "${CUP_STACK_ALLOW_SAFE_STACK_START}" != "1" ]]; then
    echo "[error] robot stack is offline; refusing full vendor restart" >&2
    echo "[error] set CUP_STACK_ALLOW_SAFE_STACK_START=1 for no-teleop startup" >&2
    exit 1
  fi
  bash scripts/r1lite_start_stack_without_teleop.sh
fi

CONFIG_PATH="${OUTPUT_DIR}/cup_stack_llm_config_${RUN_ID}.yaml"
PLAN_PATH="${OUTPUT_DIR}/cup_stack_llm_plan_${RUN_ID}.json"
PLAN_AUDIT="${OUTPUT_DIR}/cup_stack_llm_plan_${RUN_ID}_audit.json"
CODE_PATH="${OUTPUT_DIR}/cup_stack_llm_code_${RUN_ID}.py"
CODE_AUDIT="${OUTPUT_DIR}/cup_stack_llm_code_${RUN_ID}_audit.json"
EXECUTION_AUDIT="${OUTPUT_DIR}/cup_stack_llm_execution_${RUN_ID}_audit.json"
TORSO_HOLD_STATUS="${OUTPUT_DIR}/torso_hold_${RUN_ID}.json"
mkdir -p "${OUTPUT_DIR}" logs

"${PYTHON}" scripts/r1lite_prepare_three_cup_stack_config.py \
  --source-config "${SOURCE_CONFIG_PATH}" \
  --output "${CONFIG_PATH}" \
  --cup-height-m "${CUP_HEIGHT_M}" \
  --cup-diameter-m "${CUP_DIAMETER_M}" \
  --gripper-close-position "${CUP_GRIPPER_CLOSE_POSITION}"

R1LITE_PIPELINE_SETUP_ONLY=1 \
ALLOW_ROBOT_STACK_RESTART=0 \
USE_ORACLE_CODE=0 \
MODEL="${MODEL}" \
SERVER_URL="${SERVER_URL}" \
LLM_BASE_URL="${LLM_BASE_URL}" \
API_KEY_FILE="${API_KEY_FILE}" \
CONFIG_PATH="${CONFIG_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
SKIP_GRASPNET_DRY_RUN=1 \
SKIP_RIGHT_ARM_MIT_MODE_RESET=1 \
PIPELINE_LOG="logs/r1lite_three_cup_stack_llm_setup_${RUN_ID}.log" \
bash scripts/run_r1lite_tennis_ball_in_box_pipeline.sh

if pgrep -af "scripts/r1lite_hold_torso.py" | rg -v "pgrep -af" >/dev/null; then
  echo "[error] another Cap-X torso hold process is already running" >&2
  exit 1
fi
rm -f "${TORSO_HOLD_STATUS}"
"${PYTHON}" scripts/r1lite_hold_torso.py \
  --ready-file "${TORSO_HOLD_STATUS}" \
  --maximum-drift-rad 0.03 \
  >"logs/r1lite_torso_hold_${RUN_ID}.log" 2>&1 &
TORSO_HOLD_PID=$!
cleanup_torso_hold() {
  kill "${TORSO_HOLD_PID}" 2>/dev/null || true
  wait "${TORSO_HOLD_PID}" 2>/dev/null || true
}
trap cleanup_torso_hold EXIT

for _ in $(seq 1 150); do
  [[ -s "${TORSO_HOLD_STATUS}" ]] && break
  kill -0 "${TORSO_HOLD_PID}" 2>/dev/null || {
    echo "[error] torso hold process exited before becoming ready" >&2
    exit 1
  }
  sleep 0.1
done
[[ -s "${TORSO_HOLD_STATUS}" ]] || {
  echo "[error] timed out waiting for torso hold feedback latch" >&2
  exit 1
}
"${PYTHON}" - "${TORSO_HOLD_STATUS}" <<'PY'
import json
import sys

status = json.load(open(sys.argv[1]))
if status.get("violation"):
    raise RuntimeError(f"torso hold violated during startup: {status}")
if len(status.get("reference_position_rad") or []) != 3:
    raise RuntimeError(f"torso hold reference is invalid: {status}")
PY
echo "[ok] torso hold latched before planning and arm mode changes"

"${PYTHON}" scripts/r1lite_generate_llm_cup_stack_plan.py \
  --config "${CONFIG_PATH}" \
  --output-plan "${PLAN_PATH}" \
  --audit-output "${PLAN_AUDIT}" \
  --api-key-file "${API_KEY_FILE}" \
  --model "${MODEL}" \
  --server-url "${SERVER_URL}" \
  --cup-height-m "${CUP_HEIGHT_M}" \
  --cup-diameter-m "${CUP_DIAMETER_M}" \
  --gripper-close-position "${CUP_GRIPPER_CLOSE_POSITION}" \
  --fixed-scene "${FIXED_CUP_SCENE_PATH}"

"${PYTHON}" scripts/r1lite_generate_llm_cup_stack_code.py \
  --plan "${PLAN_PATH}" \
  --output-code "${CODE_PATH}" \
  --audit-output "${CODE_AUDIT}" \
  --api-key-file "${API_KEY_FILE}" \
  --model "${MODEL}" \
  --server-url "${SERVER_URL}"

echo "plan=${PLAN_PATH}"
echo "plan_audit=${PLAN_AUDIT}"
echo "generated_code=${CODE_PATH}"
echo "code_audit=${CODE_AUDIT}"

if [[ "${CUP_STACK_LLM_EXECUTE_ROBOT}" != "1" ]]; then
  echo "[ready] LLM plan/code generation and restricted dry-run passed."
  echo "[safe] No robot motion requested."
  exit 0
fi

PLAN_SHA256="$(sha256sum "${PLAN_PATH}" | awk '{print $1}')"
CODE_SHA256="$(sha256sum "${CODE_PATH}" | awk '{print $1}')"
EXECUTION_SCOPE_ARGS=()
if [[ "${CUP_STACK_STOP_AFTER_FIRST_GRASP}" == "1" ]]; then
  EXECUTION_SCOPE_ARGS+=(--stop-after-first-grasp)
fi
"${PYTHON}" scripts/r1lite_execute_llm_cup_stack.py \
  --config "${CONFIG_PATH}" \
  --plan "${PLAN_PATH}" \
  --expected-plan-sha256 "${PLAN_SHA256}" \
  --code "${CODE_PATH}" \
  --expected-code-sha256 "${CODE_SHA256}" \
  --audit-output "${EXECUTION_AUDIT}" \
  --torso-hold-status "${TORSO_HOLD_STATUS}" \
  "${EXECUTION_SCOPE_ARGS[@]}"
echo "execution_audit=${EXECUTION_AUDIT}"
