#!/usr/bin/env bash
# Fifth pipeline: LLM-generated invocation of unchanged pipeline-2 F0-F6.
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

MODEL="${MODEL:-gpt-5.5}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
LLM_BASE_URL="${LLM_BASE_URL:-https://api.usezyla.com/v1/}"
API_KEY_FILE="${API_KEY_FILE:-${REPO_ROOT}/.usezyla_key}"
SOURCE_CONFIG_PATH="${SOURCE_CONFIG_PATH:-tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml}"
FIXED_PLAN="${FIXED_PLAN:-outputs/r1lite_tennis_ball_in_box/offline_ik_pick_place_joint_poses_3d.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r1lite_tennis_ball_in_box_llm_fixed_plan}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${OUTPUT_DIR}/fifth_pipeline_config_${RUN_ID}.yaml"
GENERATED_CODE="${OUTPUT_DIR}/llm_invocation_${RUN_ID}.py"
INVOCATION_AUDIT="${OUTPUT_DIR}/llm_invocation_${RUN_ID}_audit.json"
PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"

for required in "${API_KEY_FILE}" "${SOURCE_CONFIG_PATH}" "${FIXED_PLAN}"; do
  if [[ ! -r "${required}" ]]; then
    echo "[error] required file is missing or unreadable: ${required}" >&2
    exit 1
  fi
done
if [[ -z "$(tr -d '[:space:]' < "${API_KEY_FILE}")" ]]; then
  echo "[error] API key file is empty: ${API_KEY_FILE}" >&2
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
cfg["use_oracle_code"] = False
cfg["use_wrist_camera"] = False
low_level = cfg["env"]["cfg"]["low_level"]
low_level["wrist_camera_enabled"] = False
low_level["arm_target_topic"] = "/motion_target/target_joint_state_arm_right"
low_level["gripper_target_topic"] = "/motion_target/target_position_gripper_right"
low_level["relaxed_ik_target_topic"] = "/capx/relaxed_ik/target_pose_arm_right"
destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

EXPECTED_PLAN_SHA256="$(sha256sum "${FIXED_PLAN}" | awk '{print $1}')"
echo "[fifth-pipeline] run_id=${RUN_ID}"
echo "[fifth-pipeline] model=${MODEL}"
echo "[fifth-pipeline] code_source=llm oracle=false"
echo "[fifth-pipeline] fixed_plan=${FIXED_PLAN}"
echo "[fifth-pipeline] fixed_plan_sha256=${EXPECTED_PLAN_SHA256}"
echo "[fifth-pipeline] execution=one_round_pipeline_2_F0_F7"

R1LITE_PIPELINE_SETUP_ONLY=1 \
USE_ORACLE_CODE=0 \
MODEL="${MODEL}" \
SERVER_URL="${SERVER_URL}" \
LLM_BASE_URL="${LLM_BASE_URL}" \
API_KEY_FILE="${API_KEY_FILE}" \
CONFIG_PATH="${CONFIG_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
PIPELINE_LOG="logs/r1lite_fifth_pipeline_setup_${RUN_ID}.log" \
ALLOW_ROBOT_STACK_RESTART="${ALLOW_ROBOT_STACK_RESTART:-0}" \
scripts/run_r1lite_tennis_ball_in_box_pipeline.sh

"${PYTHON}" scripts/r1lite_generate_llm_fixed_plan_invocation.py \
  --plan "${FIXED_PLAN}" \
  --output-code "${GENERATED_CODE}" \
  --audit-output "${INVOCATION_AUDIT}" \
  --model "${MODEL}" \
  --server-url "${SERVER_URL}" \
  --api-key-file "${API_KEY_FILE}"

ACTUAL_PLAN_SHA256="$(sha256sum "${FIXED_PLAN}" | awk '{print $1}')"
if [[ "${ACTUAL_PLAN_SHA256}" != "${EXPECTED_PLAN_SHA256}" ]]; then
  echo "[error] fixed pipeline-2 plan changed after LLM generation" >&2
  exit 1
fi

R1LITE_CONFIG_PATH="${CONFIG_PATH}" \
R1LITE_ROUND_COUNT=1 \
ARM_TOPIC="/motion_target/target_joint_state_arm_right" \
GRIPPER_TOPIC="/motion_target/target_position_gripper_right" \
scripts/r1lite_execute_10_keyframes_xyz_corrected.sh \
  "${FIXED_PLAN}" \
  cartesian_absolute
