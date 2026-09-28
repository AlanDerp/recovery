#!/usr/bin/env bash
# Seventh pipeline: LLM-generated, audited dual-arm towel-fold preview.
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

MODEL="${MODEL:-gpt-5.5}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
API_KEY_FILE="${API_KEY_FILE:-${REPO_ROOT}/.usezyla_key}"
SOURCE_CONFIG_PATH="${SOURCE_CONFIG_PATH:-tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml}"
CALIBRATION_PATH="${CALIBRATION_PATH:-outputs/r1lite/head_plane_calibration.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r1lite_dual_arm_towel_fold}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"
SEVENTH_PIPELINE_EXECUTE_ROBOT="${SEVENTH_PIPELINE_EXECUTE_ROBOT:-0}"

CONFIG_PATH="${OUTPUT_DIR}/seventh_pipeline_config_${RUN_ID}.yaml"
PLAN_PATH="${OUTPUT_DIR}/towel_fold_plan_${RUN_ID}.json"
RAW_IMAGE="${OUTPUT_DIR}/towel_raw_${RUN_ID}.png"
MASK_IMAGE="${OUTPUT_DIR}/towel_mask_${RUN_ID}.png"
PREVIEW_IMAGE="${OUTPUT_DIR}/towel_preview_${RUN_ID}.png"
TRAJECTORY_HTML="${OUTPUT_DIR}/towel_trajectory_${RUN_ID}.html"
PERCEPTION_AUDIT="${OUTPUT_DIR}/towel_perception_audit_${RUN_ID}.json"
GENERATED_CODE="${OUTPUT_DIR}/towel_fold_llm_program_${RUN_ID}.py"
CODE_AUDIT="${OUTPUT_DIR}/towel_fold_llm_program_audit_${RUN_ID}.json"

for required in "${API_KEY_FILE}" "${SOURCE_CONFIG_PATH}" "${CALIBRATION_PATH}"; do
  if [[ ! -r "${required}" ]]; then
    echo "[error] required file is missing or unreadable: ${required}" >&2
    exit 1
  fi
done
if [[ ! -x "${PYTHON}" ]]; then
  echo "[error] Python interpreter is unavailable: ${PYTHON}" >&2
  exit 1
fi
if [[ "${SEVENTH_PIPELINE_EXECUTE_ROBOT}" != "0" && "${SEVENTH_PIPELINE_EXECUTE_ROBOT}" != "1" ]]; then
  echo "[error] SEVENTH_PIPELINE_EXECUTE_ROBOT must be 0 or 1" >&2
  exit 1
fi
if [[ "${SEVENTH_PIPELINE_EXECUTE_ROBOT}" == "1" ]]; then
  : "${APPROVED_CONFIG_PATH:?execute requires APPROVED_CONFIG_PATH}"
  : "${APPROVED_PLAN_PATH:?execute requires APPROVED_PLAN_PATH}"
  : "${APPROVED_CODE_PATH:?execute requires APPROVED_CODE_PATH}"
  : "${APPROVED_BUNDLE_PATH:?execute requires APPROVED_BUNDLE_PATH}"
  : "${CAPABILITY_CERTIFICATE_PATH:?execute requires CAPABILITY_CERTIFICATE_PATH}"
  for approved in \
    "${APPROVED_CONFIG_PATH}" "${APPROVED_PLAN_PATH}" \
    "${APPROVED_CODE_PATH}" "${APPROVED_BUNDLE_PATH}" \
    "${CAPABILITY_CERTIFICATE_PATH}" "${CALIBRATION_PATH}"; do
    if [[ ! -r "${approved}" ]]; then
      echo "[error] approved execution artifact is unreadable: ${approved}" >&2
      exit 1
    fi
  done
  set +u
  source /opt/ros/humble/setup.bash
  source /home/nvidia/galaxea/install/local_setup.bash
  set -u
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
  export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
  export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
  for command in 2 5 1; do
    response="$(ros2 service call /function_frame_right_arm \
      hdas_msg/srv/FunctionFrame "{command: ${command}}")"
    if [[ "${response}" != *"success=True"* ]]; then
      echo "[error] right-arm MIT mode command ${command} failed" >&2
      exit 1
    fi
  done
  echo "[ok] right arm entered MIT mode immediately before execution"
  mkdir -p "${OUTPUT_DIR}" logs
  LEFT_IK_LOG="logs/r1lite_seventh_left_ik_${RUN_ID}.log"
  RIGHT_IK_LOG="logs/r1lite_seventh_right_ik_${RUN_ID}.log"
  ros2 run mobiman relaxed_ik.py 1 0 0 --ros-args \
    -p robot_name:=R1_Lite \
    -r __node:=capx_relaxed_ik_left_isolated \
    -r /motion_target/target_pose_arm_left:=/capx/relaxed_ik/target_pose_arm_left \
    -r /motion_target/target_joint_state_arm_left:=/relaxed_ik/isolated_target_joint_state_arm_left \
    >"${LEFT_IK_LOG}" 2>&1 &
  LEFT_IK_PID=$!
  ros2 run mobiman relaxed_ik.py 0 1 0 --ros-args \
    -p robot_name:=R1_Lite \
    -r __node:=capx_relaxed_ik_right_isolated \
    -r /motion_target/target_pose_arm_right:=/capx/relaxed_ik/target_pose_arm_right \
    -r /motion_target/target_joint_state_arm_right:=/relaxed_ik/isolated_target_joint_state_arm_right \
    >"${RIGHT_IK_LOG}" 2>&1 &
  RIGHT_IK_PID=$!
  cleanup_ik() {
    kill "${LEFT_IK_PID}" "${RIGHT_IK_PID}" 2>/dev/null || true
    wait "${LEFT_IK_PID}" "${RIGHT_IK_PID}" 2>/dev/null || true
  }
  trap cleanup_ik EXIT INT TERM
  sleep 2
  PLAN_SHA256="$(sha256sum "${APPROVED_PLAN_PATH}" | awk '{print $1}')"
  CODE_SHA256="$(sha256sum "${APPROVED_CODE_PATH}" | awk '{print $1}')"
  BUNDLE_SHA256="$(sha256sum "${APPROVED_BUNDLE_PATH}" | awk '{print $1}')"
  CERTIFICATE_SHA256="$(sha256sum "${CAPABILITY_CERTIFICATE_PATH}" | awk '{print $1}')"
  EXECUTION_AUDIT="${OUTPUT_DIR}/towel_execution_audit_${RUN_ID}.json"
  "${PYTHON}" scripts/r1lite_seventh_dual_arm_towel_fold.py \
    --mode execute \
    --config "${APPROVED_CONFIG_PATH}" \
    --calibration "${CALIBRATION_PATH}" \
    --audit-output "${EXECUTION_AUDIT}" \
    --approved-plan "${APPROVED_PLAN_PATH}" \
    --expected-plan-sha256 "${PLAN_SHA256}" \
    --generated-code "${APPROVED_CODE_PATH}" \
    --expected-code-sha256 "${CODE_SHA256}" \
    --execution-bundle "${APPROVED_BUNDLE_PATH}" \
    --expected-execution-bundle-sha256 "${BUNDLE_SHA256}" \
    --capability-certificate "${CAPABILITY_CERTIFICATE_PATH}" \
    --expected-capability-certificate-sha256 "${CERTIFICATE_SHA256}"
  echo "execution_audit=${EXECUTION_AUDIT}"
  exit 0
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
env_cfg = cfg["env"]["cfg"]
env_cfg["r1lite_ik_harness_enabled"] = True
env_cfg["r1lite_motion_validation_enabled"] = True
env_cfg["r1lite_adaptive_joint_control_enabled"] = True
low_level = env_cfg["low_level"]
low_level["wrist_camera_enabled"] = False
low_level["relaxed_ik_target_topic"] = "/capx/relaxed_ik/target_pose_arm_right"
destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "[seventh-pipeline] run_id=${RUN_ID}"
echo "[seventh-pipeline] mode=preview_only"
echo "[seventh-pipeline] model=${MODEL}"
echo "[seventh-pipeline] final_fingertip_center_separation_m=0.02"
echo "[seventh-pipeline] fold_orientation=mirrored_outward_tilt_search_45_to_60_deg"
echo "[seventh-pipeline] maximum_keypoint_extrapolation_m=0.15"
echo "[seventh-pipeline] mask_must_be_inside_calibration_polygon=false"
echo "[seventh-pipeline] left_tcp=left_gripper_tool_z,symmetric_to_right,tool_forward=+Z"

R1LITE_PIPELINE_SETUP_ONLY=1 \
SKIP_RIGHT_ARM_MIT_MODE_RESET=1 \
USE_ORACLE_CODE=0 \
MODEL="${MODEL}" \
SERVER_URL="${SERVER_URL}" \
API_KEY_FILE="${API_KEY_FILE}" \
CONFIG_PATH="${CONFIG_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
PIPELINE_LOG="logs/r1lite_seventh_pipeline_setup_${RUN_ID}.log" \
ALLOW_ROBOT_STACK_RESTART="${ALLOW_ROBOT_STACK_RESTART:-0}" \
GRASPNET_FORCE_RESTART=0 \
SKIP_GRASPNET_DRY_RUN=1 \
scripts/run_r1lite_tennis_ball_in_box_pipeline.sh

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

"${PYTHON}" scripts/r1lite_seventh_dual_arm_towel_fold.py \
  --mode preview \
  --config "${CONFIG_PATH}" \
  --calibration "${CALIBRATION_PATH}" \
  --output-plan "${PLAN_PATH}" \
  --audit-output "${PERCEPTION_AUDIT}" \
  --raw-image "${RAW_IMAGE}" \
  --preview-image "${PREVIEW_IMAGE}" \
  --mask-image "${MASK_IMAGE}" \
  --sequential-side right

"${PYTHON}" scripts/r1lite_generate_llm_towel_fold_code.py \
  --plan "${PLAN_PATH}" \
  --output-code "${GENERATED_CODE}" \
  --audit-output "${CODE_AUDIT}" \
  --model "${MODEL}" \
  --server-url "${SERVER_URL}" \
  --api-key-file "${API_KEY_FILE}"

"${PYTHON}" scripts/render_r1lite_seventh_towel_fold_html.py \
  --plan "${PLAN_PATH}" \
  --output "${TRAJECTORY_HTML}"

PLAN_SHA256="$(sha256sum "${PLAN_PATH}" | awk '{print $1}')"
CODE_SHA256="$(sha256sum "${GENERATED_CODE}" | awk '{print $1}')"
echo "[seventh-pipeline] preview and restricted LLM dry-run passed"
echo "[seventh-pipeline] no robot motion was sent"
echo "[seventh-pipeline] plan=${PLAN_PATH}"
echo "[seventh-pipeline] plan_sha256=${PLAN_SHA256}"
echo "[seventh-pipeline] generated_code=${GENERATED_CODE}"
echo "[seventh-pipeline] generated_code_sha256=${CODE_SHA256}"
echo "[seventh-pipeline] preview=${PREVIEW_IMAGE}"
echo "[seventh-pipeline] trajectory_html=${TRAJECTORY_HTML}"
echo "[seventh-pipeline] execution remains fail-closed pending dual-arm certification"
