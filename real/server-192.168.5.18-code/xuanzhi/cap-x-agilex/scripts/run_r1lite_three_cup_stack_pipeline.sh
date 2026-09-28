#!/usr/bin/env bash
# LLM-planned right-arm task: orange on purple, then green on orange.
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

MODE="${1:-generate}"
PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"
SOURCE_CONFIG_PATH="${SOURCE_CONFIG_PATH:-tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r1lite_three_cup_stack}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${OUTPUT_DIR}/three_cup_stack_config_${RUN_ID}.yaml"

usage() {
  echo "usage: $0 {prepare|generate|execute|oracle-execute}" >&2
}

prepare_config() {
  local height="${CUP_HEIGHT_M:-0.100}"
  local diameter="${CUP_DIAMETER_M:-0.070}"
  local close_position="${CUP_GRIPPER_CLOSE_POSITION:-20.0}"
  mkdir -p "${OUTPUT_DIR}"
  "${PYTHON}" scripts/r1lite_prepare_three_cup_stack_config.py \
    --source-config "${SOURCE_CONFIG_PATH}" \
    --output "${CONFIG_PATH}" \
    --cup-height-m "${height}" \
    --cup-diameter-m "${diameter}" \
    --gripper-close-position "${close_position}"
  echo "config=${CONFIG_PATH}"
  echo "cup_height_m=${height}"
  echo "cup_diameter_m=${diameter}"
  echo "gripper_close_position=${close_position}"
}

case "${MODE}" in
  prepare)
    prepare_config
    echo "[ready] Config generated without starting ROS services or moving the robot."
    echo "[required] Measure the real cup height, grasp diameter, and safe gripper command before execution."
    ;;
  generate)
    exec bash scripts/run_r1lite_three_cup_stack_llm_pipeline.sh
    ;;
  execute)
    export CUP_STACK_LLM_EXECUTE_ROBOT=1
    exec bash scripts/run_r1lite_three_cup_stack_llm_pipeline.sh
    ;;
  oracle-execute)
    : "${CUP_HEIGHT_M:?Set measured CUP_HEIGHT_M before robot execution}"
    : "${CUP_DIAMETER_M:?Set measured CUP_DIAMETER_M before robot execution}"
    : "${CUP_GRIPPER_CLOSE_POSITION:?Set a tested CUP_GRIPPER_CLOSE_POSITION}"
    prepare_config
    export CONFIG_PATH OUTPUT_DIR
    export USE_ORACLE_CODE=1
    export SKIP_GRASPNET_DRY_RUN=1
    export PIPELINE_LOG="logs/r1lite_three_cup_stack_${RUN_ID}.log"
    bash scripts/run_r1lite_tennis_ball_in_box_pipeline.sh
    ;;
  *)
    usage
    exit 2
    ;;
esac
