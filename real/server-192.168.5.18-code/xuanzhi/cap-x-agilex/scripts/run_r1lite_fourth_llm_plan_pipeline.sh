#!/usr/bin/env bash
# Fourth pipeline: LLM plans all variable Cartesian semantic anchors.
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

MODEL="${MODEL:-gpt-5.5}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
LLM_BASE_URL="${LLM_BASE_URL:-https://api.usezyla.com/v1/}"
API_KEY_FILE="${API_KEY_FILE:-${REPO_ROOT}/.usezyla_key}"
SOURCE_CONFIG_PATH="${SOURCE_CONFIG_PATH:-tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml}"
TEMPLATE_PLAN="${TEMPLATE_PLAN:-outputs/r1lite_tennis_ball_in_box/offline_ik_pick_place_joint_poses_3d.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r1lite_tennis_ball_in_box_llm_plan}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${OUTPUT_DIR}/fourth_pipeline_config_${RUN_ID}.yaml"
GENERATED_PLAN="${OUTPUT_DIR}/llm_f2_f6_plan_${RUN_ID}.json"
PLANNER_AUDIT="${OUTPUT_DIR}/llm_f2_f6_plan_${RUN_ID}_audit.json"
GENERATED_CODE="${OUTPUT_DIR}/llm_task_code_${RUN_ID}.py"
CODE_AUDIT="${OUTPUT_DIR}/llm_task_code_${RUN_ID}_audit.json"
PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"
LLM_MAX_CARTESIAN_FRAME_STEP_M="${LLM_MAX_CARTESIAN_FRAME_STEP_M:-0.10}"
LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD="${LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD:-1.0}"
FOURTH_PIPELINE_EXECUTE_ROBOT="${FOURTH_PIPELINE_EXECUTE_ROBOT:-0}"

for required in "${API_KEY_FILE}" "${SOURCE_CONFIG_PATH}" "${TEMPLATE_PLAN}"; do
  if [[ ! -r "${required}" ]]; then
    echo "[error] required file is missing or unreadable: ${required}" >&2
    exit 1
  fi
done
if [[ -z "$(tr -d '[:space:]' < "${API_KEY_FILE}")" ]]; then
  echo "[error] API key file is empty: ${API_KEY_FILE}" >&2
  exit 1
fi
if [[ "${FOURTH_PIPELINE_EXECUTE_ROBOT}" != "0" && "${FOURTH_PIPELINE_EXECUTE_ROBOT}" != "1" ]]; then
  echo "[error] FOURTH_PIPELINE_EXECUTE_ROBOT must be 0 or 1" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
"${PYTHON}" - "${SOURCE_CONFIG_PATH}" "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
cfg = yaml.safe_load(source.read_text())
cfg["use_wrist_camera"] = False
low_level = cfg["env"]["cfg"]["low_level"]
low_level["wrist_camera_enabled"] = False
low_level["arm_target_topic"] = "/motion_target/target_joint_state_arm_right"
low_level["gripper_target_topic"] = "/motion_target/target_position_gripper_right"
low_level["relaxed_ik_target_topic"] = "/capx/relaxed_ik/target_pose_arm_right"
destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "[fourth-pipeline] run_id=${RUN_ID}"
echo "[fourth-pipeline] model=${MODEL}"
echo "[fourth-pipeline] template_plan=${TEMPLATE_PLAN}"
echo "[fourth-pipeline] generated_plan=${GENERATED_PLAN}"
echo "[fourth-pipeline] generated_code=${GENERATED_CODE}"
echo "[fourth-pipeline] planner_audit=${PLANNER_AUDIT}"
echo "[fourth-pipeline] code_audit=${CODE_AUDIT}"
echo "[fourth-pipeline] execution=llm_plan_then_constrained_llm_code"
echo "[fourth-pipeline] max_cartesian_frame_step_m=${LLM_MAX_CARTESIAN_FRAME_STEP_M}"
echo "[fourth-pipeline] max_incremental_joint_delta_rad=${LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD}"

R1LITE_PIPELINE_SETUP_ONLY=1 \
USE_ORACLE_CODE=0 \
MODEL="${MODEL}" \
SERVER_URL="${SERVER_URL}" \
LLM_BASE_URL="${LLM_BASE_URL}" \
API_KEY_FILE="${API_KEY_FILE}" \
CONFIG_PATH="${CONFIG_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
PIPELINE_LOG="logs/r1lite_fourth_pipeline_setup_${RUN_ID}.log" \
ALLOW_ROBOT_STACK_RESTART="${ALLOW_ROBOT_STACK_RESTART:-0}" \
scripts/run_r1lite_tennis_ball_in_box_pipeline.sh

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u

"${PYTHON}" scripts/r1lite_generate_llm_f2_f6_plan.py \
  --config "${CONFIG_PATH}" \
  --template-plan "${TEMPLATE_PLAN}" \
  --output-plan "${GENERATED_PLAN}" \
  --audit-output "${PLANNER_AUDIT}" \
  --model "${MODEL}" \
  --server-url "${SERVER_URL}" \
  --api-key-file "${API_KEY_FILE}" \
  --max-cartesian-step-m "${LLM_MAX_CARTESIAN_FRAME_STEP_M}"

"${PYTHON}" scripts/r1lite_generate_llm_plan_execution_code.py \
  --plan "${GENERATED_PLAN}" \
  --output-code "${GENERATED_CODE}" \
  --audit-output "${CODE_AUDIT}" \
  --model "${MODEL}" \
  --server-url "${SERVER_URL}" \
  --api-key-file "${API_KEY_FILE}"

if [[ "${FOURTH_PIPELINE_EXECUTE_ROBOT}" != "1" ]]; then
  echo "[fourth-pipeline] generation and restricted dry-run passed"
  echo "[fourth-pipeline] no robot motion requested; review plan/code/audits above"
  echo "[fourth-pipeline] set FOURTH_PIPELINE_EXECUTE_ROBOT=1 to execute"
  exit 0
fi

R1LITE_CONFIG_PATH="${CONFIG_PATH}" \
R1LITE_ROUND_COUNT=1 \
R1LITE_SINGLE_ROUND_POLICY=llm_absolute_20 \
MAX_CORRECTIONS=20 \
CORRECTION_GAIN=0.5 \
CORRECTION_STEP_CAP_M=0.02 \
LLM_MAX_CARTESIAN_FRAME_STEP_M="${LLM_MAX_CARTESIAN_FRAME_STEP_M}" \
LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD="${LLM_MAX_INCREMENTAL_JOINT_DELTA_RAD}" \
LLM_GENERATED_POLICY_PATH="${GENERATED_CODE}" \
ARM_TOPIC="/motion_target/target_joint_state_arm_right" \
GRIPPER_TOPIC="/motion_target/target_position_gripper_right" \
scripts/r1lite_execute_10_keyframes_xyz_corrected.sh \
  "${GENERATED_PLAN}" \
  cartesian_absolute
