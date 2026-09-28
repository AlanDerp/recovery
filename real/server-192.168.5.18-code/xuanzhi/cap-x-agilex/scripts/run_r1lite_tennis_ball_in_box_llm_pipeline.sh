#!/usr/bin/env bash
# Third R1 Lite pipeline: LLM-generated pick/place code on the second
# pipeline's low-level robot configuration. The cartesian_absolute pipeline
# and its plan executor are intentionally not modified or invoked here.
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

MODEL="${MODEL:-gpt-5.5}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
LLM_BASE_URL="${LLM_BASE_URL:-https://api.usezyla.com/v1/}"
API_KEY_FILE="${API_KEY_FILE:-${REPO_ROOT}/.usezyla_key}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/r1lite_tennis_ball_in_box_llm}"
PIPELINE_LOG="${PIPELINE_LOG:-logs/r1lite_tennis_ball_in_box_llm_pipeline.log}"
SOURCE_CONFIG_PATH="${SOURCE_CONFIG_PATH:-tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml}"
CONFIG_PATH="${CONFIG_PATH:-${OUTPUT_DIR}/llm_generated_code_config.yaml}"
PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"

if [[ ! -r "${API_KEY_FILE}" ]]; then
  echo "[error] LLM API key file is missing or unreadable: ${API_KEY_FILE}" >&2
  echo "Set API_KEY_FILE to a readable file containing only the API key." >&2
  exit 1
fi
if [[ -z "$(tr -d '[:space:]' < "${API_KEY_FILE}")" ]]; then
  echo "[error] LLM API key file is empty: ${API_KEY_FILE}" >&2
  exit 1
fi
if [[ ! -r "${SOURCE_CONFIG_PATH}" ]]; then
  echo "[error] source robot config is missing: ${SOURCE_CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "[error] Python interpreter is unavailable: ${PYTHON}" >&2
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
cfg["r1lite_staged_tennis_execution"] = False
cfg["r1lite_staged_use_fixed_plan"] = False
cfg["use_wrist_camera"] = False
low_level = cfg["env"]["cfg"]["low_level"]
low_level["wrist_camera_enabled"] = False
low_level["max_joint_step"] = 0.15
low_level["goto_pose_joint_waypoint_step_rad"] = 0.15
low_level["r1lite_motion_validation_enabled"] = True
low_level["r1lite_require_pose_arrival"] = True
low_level["arm_target_topic"] = "/motion_target/target_joint_state_arm_right"
low_level["gripper_target_topic"] = "/motion_target/target_position_gripper_right"
low_level["relaxed_ik_target_topic"] = "/capx/relaxed_ik/target_pose_arm_right"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "[third-pipeline] mode=llm_generated_code"
echo "[third-pipeline] model=${MODEL}"
echo "[third-pipeline] source_config=${SOURCE_CONFIG_PATH}"
echo "[third-pipeline] generated_config=${CONFIG_PATH}"
echo "[third-pipeline] output=${OUTPUT_DIR}"

export USE_ORACLE_CODE=0
export MODEL
export SERVER_URL
export LLM_BASE_URL
export API_KEY_FILE
export CONFIG_PATH
export OUTPUT_DIR
export PIPELINE_LOG

exec "${REPO_ROOT}/scripts/run_r1lite_tennis_ball_in_box_pipeline.sh"
