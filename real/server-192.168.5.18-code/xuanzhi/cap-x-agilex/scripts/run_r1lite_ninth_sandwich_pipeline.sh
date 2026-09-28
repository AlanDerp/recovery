#!/usr/bin/env bash
# Pipeline 9: donut on brown bread, then mixed bread on the fresh stack.
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

STAGE="${1:-}"
PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"
SOURCE_CONFIG_PATH="${SOURCE_CONFIG_PATH:-tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml}"
CALIBRATION_PATH="${CALIBRATION_PATH:-outputs/r1lite/head_plane_calibration.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r1lite_ninth_sandwich_pipeline}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-checkpoints/sam3/sam3.pt}"

usage() {
  echo "usage: $0 {prepare-donut|execute-donut-and-prepare-top|execute-top}" >&2
}

[[ -x "${PYTHON}" ]] || {
  echo "[error] Python interpreter unavailable: ${PYTHON}" >&2
  exit 1
}
mkdir -p "${OUTPUT_DIR}" logs

source_ros() {
  set +u
  source /opt/ros/humble/setup.bash
  source /home/nvidia/galaxea/install/local_setup.bash
  set -u
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
  export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
  export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
}

ros_topic_has_message() {
  local topic="$1"
  local timeout_s="$2"
  timeout "${timeout_s}" ros2 topic echo --once "${topic}" >/dev/null 2>&1
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local label="$3"
  local deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    if timeout 1 bash -c ":</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
      echo "[ok] ${label} is available on ${host}:${port}"
      return 0
    fi
    sleep 2
  done
  echo "[error] timed out waiting for ${label} on ${host}:${port}" >&2
  return 1
}

ensure_perception_services() {
  [[ -r "${CALIBRATION_PATH}" ]] || {
    echo "[error] missing homography calibration: ${CALIBRATION_PATH}" >&2
    return 1
  }
  if ! timeout 1 bash -c ":</dev/tcp/127.0.0.1/8114" >/dev/null 2>&1; then
    [[ -f "${SAM3_CHECKPOINT_PATH}" ]] || {
      echo "[error] missing SAM3 checkpoint: ${SAM3_CHECKPOINT_PATH}" >&2
      return 1
    }
    echo "[start] SAM3 server on 8114"
    "${PYTHON}" -m capx.serving.launch_sam3_server \
      --device cuda \
      --port 8114 \
      --host 127.0.0.1 \
      --checkpoint-path "${SAM3_CHECKPOINT_PATH}" \
      --no-load-from-hf \
      >"logs/sam3_r1lite_ninth_${RUN_ID}.log" 2>&1 &
    wait_for_tcp 127.0.0.1 8114 SAM3
  fi
}

start_isolated_ik() {
  ros2 run mobiman relaxed_ik.py 1 0 0 --ros-args \
    -p robot_name:=R1_Lite \
    -r __node:=capx_ninth_relaxed_ik_left \
    -r /motion_target/target_pose_arm_left:=/capx/relaxed_ik/target_pose_arm_left \
    -r /motion_target/target_joint_state_arm_left:=/relaxed_ik/isolated_target_joint_state_arm_left \
    >"logs/r1lite_ninth_left_ik_${RUN_ID}.log" 2>&1 &
  LEFT_IK_PID=$!
  ros2 run mobiman relaxed_ik.py 0 1 0 --ros-args \
    -p robot_name:=R1_Lite \
    -r __node:=capx_ninth_relaxed_ik_right \
    -r /motion_target/target_pose_arm_right:=/capx/relaxed_ik/target_pose_arm_right \
    -r /motion_target/target_joint_state_arm_right:=/relaxed_ik/isolated_target_joint_state_arm_right \
    >"logs/r1lite_ninth_right_ik_${RUN_ID}.log" 2>&1 &
  RIGHT_IK_PID=$!
  sleep 2
}

cleanup_ik() {
  if [[ -n "${LEFT_IK_PID:-}" || -n "${RIGHT_IK_PID:-}" ]]; then
    kill "${LEFT_IK_PID:-}" "${RIGHT_IK_PID:-}" 2>/dev/null || true
    wait "${LEFT_IK_PID:-}" "${RIGHT_IK_PID:-}" 2>/dev/null || true
  fi
}

prepare_config() {
  local destination="$1"
  [[ -r "${SOURCE_CONFIG_PATH}" ]] || {
    echo "[error] unreadable source config: ${SOURCE_CONFIG_PATH}" >&2
    exit 1
  }
  "${PYTHON}" - "${SOURCE_CONFIG_PATH}" "${destination}" <<'PY'
from pathlib import Path
import sys
import yaml

source, destination = map(Path, sys.argv[1:])
cfg = yaml.safe_load(source.read_text())
cfg["use_wrist_camera"] = False
env = cfg["env"]["cfg"]
env["r1lite_ik_harness_enabled"] = True
env["r1lite_motion_validation_enabled"] = True
env["r1lite_adaptive_joint_control_enabled"] = True
low = env["low_level"]
low["wrist_camera_enabled"] = False
required = list(low.get("startup_required_feedback", []))
for name in ("head_left_rect_rgb", "head_left_depth", "head_left_camera_info"):
    if name in required:
        required.remove(name)
if "head_left_rgb" not in required:
    required.append("head_left_rgb")
low["startup_required_feedback"] = required
destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

run_execute() {
  local phase="$1"
  local config="$2"
  local plan="$3"
  local policy="$4"
  local bundle="$5"
  local certificate="$6"
  local audit="$7"
  shift 7
  local plan_sha policy_sha bundle_sha certificate_sha
  plan_sha="$(sha256sum "${plan}" | awk '{print $1}')"
  policy_sha="$(sha256sum "${policy}" | awk '{print $1}')"
  bundle_sha="$(sha256sum "${bundle}" | awk '{print $1}')"
  certificate_sha="$(sha256sum "${certificate}" | awk '{print $1}')"
  "${PYTHON}" scripts/r1lite_ninth_sandwich.py \
    --mode execute \
    --phase "${phase}" \
    --config "${config}" \
      --calibration "${CALIBRATION_PATH}" \
    --audit-output "${audit}" \
    --approved-plan "${plan}" \
    --expected-plan-sha256 "${plan_sha}" \
    --approved-policy "${policy}" \
    --expected-policy-sha256 "${policy_sha}" \
    --execution-bundle "${bundle}" \
    --expected-execution-bundle-sha256 "${bundle_sha}" \
    --capability-certificate "${certificate}" \
    --expected-capability-certificate-sha256 "${certificate_sha}" \
    "$@"
}

case "${STAGE}" in
  prepare-donut)
    CONFIG_PATH="${OUTPUT_DIR}/ninth_config_${RUN_ID}.yaml"
    INTENT_PATH="${OUTPUT_DIR}/ninth_donut_intent_${RUN_ID}.json"
    prepare_config "${CONFIG_PATH}"
    source_ros
    ensure_perception_services
    "${PYTHON}" scripts/r1lite_ninth_sandwich.py \
      --mode preview \
      --phase donut_on_brown \
      --config "${CONFIG_PATH}" \
      --calibration "${CALIBRATION_PATH}" \
      --output-intent "${INTENT_PATH}" \
      --audit-output "${OUTPUT_DIR}/ninth_donut_preview_audit_${RUN_ID}.json" \
      --raw-image "${OUTPUT_DIR}/ninth_donut_raw_${RUN_ID}.png" \
      --preview-image "${OUTPUT_DIR}/ninth_donut_preview_${RUN_ID}.png" \
      --mask-output-dir "${OUTPUT_DIR}/ninth_donut_masks_${RUN_ID}"
    "${PYTHON}" scripts/render_r1lite_ninth_sandwich_html.py \
      --plan "${INTENT_PATH}" \
      --preview-image "${OUTPUT_DIR}/ninth_donut_preview_${RUN_ID}.png" \
      --output "${OUTPUT_DIR}/ninth_donut_preview_${RUN_ID}.html"
    echo "[approval-boundary] Review phase-one SAM3 masks and four-point-homography intent."
    echo "[approval-boundary] Run r1lite_validate_ninth_sandwich_phase.py, review the auto-selected arm and trajectory, then issue a capability certificate."
    echo "config=${CONFIG_PATH}"
    echo "intent=${INTENT_PATH}"
    ;;

  execute-donut-and-prepare-top)
    : "${APPROVED_CONFIG_PATH:?required}"
    : "${APPROVED_DONUT_PLAN_PATH:?required}"
    : "${APPROVED_DONUT_POLICY_PATH:?required}"
    : "${APPROVED_DONUT_BUNDLE_PATH:?required}"
    : "${DONUT_CAPABILITY_CERTIFICATE_PATH:?required}"
    for path in \
      "${APPROVED_CONFIG_PATH}" "${APPROVED_DONUT_PLAN_PATH}" \
      "${APPROVED_DONUT_POLICY_PATH}" "${APPROVED_DONUT_BUNDLE_PATH}" \
      "${DONUT_CAPABILITY_CERTIFICATE_PATH}"; do
      [[ -r "${path}" ]] || {
        echo "[error] approved artifact unreadable: ${path}" >&2
        exit 1
      }
    done
    source_ros
    ensure_perception_services
    start_isolated_ik
    trap cleanup_ik EXIT INT TERM
    DONUT_AUDIT="${OUTPUT_DIR}/ninth_donut_execution_audit_${RUN_ID}.json"
    run_execute \
      donut_on_brown \
      "${APPROVED_CONFIG_PATH}" \
      "${APPROVED_DONUT_PLAN_PATH}" \
      "${APPROVED_DONUT_POLICY_PATH}" \
      "${APPROVED_DONUT_BUNDLE_PATH}" \
      "${DONUT_CAPABILITY_CERTIFICATE_PATH}" \
      "${DONUT_AUDIT}"
    cleanup_ik
    unset LEFT_IK_PID RIGHT_IK_PID
    ensure_perception_services
    TOP_INTENT="${OUTPUT_DIR}/ninth_top_intent_${RUN_ID}.json"
    "${PYTHON}" scripts/r1lite_ninth_sandwich.py \
      --mode preview \
      --phase mixed_bread_on_stack \
      --config "${APPROVED_CONFIG_PATH}" \
      --calibration "${CALIBRATION_PATH}" \
      --output-intent "${TOP_INTENT}" \
      --audit-output "${OUTPUT_DIR}/ninth_top_preview_audit_${RUN_ID}.json" \
      --raw-image "${OUTPUT_DIR}/ninth_top_raw_${RUN_ID}.png" \
      --preview-image "${OUTPUT_DIR}/ninth_top_preview_${RUN_ID}.png" \
      --mask-output-dir "${OUTPUT_DIR}/ninth_top_masks_${RUN_ID}"
    "${PYTHON}" scripts/render_r1lite_ninth_sandwich_html.py \
      --plan "${APPROVED_DONUT_PLAN_PATH}" \
      --execution-bundle "${APPROVED_DONUT_BUNDLE_PATH}" \
      --execution-audit "${DONUT_AUDIT}" \
      --output "${OUTPUT_DIR}/ninth_donut_execution_${RUN_ID}.html"
    "${PYTHON}" scripts/render_r1lite_ninth_sandwich_html.py \
      --plan "${TOP_INTENT}" \
      --preview-image "${OUTPUT_DIR}/ninth_top_preview_${RUN_ID}.png" \
      --output "${OUTPUT_DIR}/ninth_top_preview_${RUN_ID}.html"
    echo "[approval-boundary] Donut execution/home and fresh stack verification passed."
    echo "[approval-boundary] Independently certify the fresh top-bread intent."
    echo "donut_execution_audit=${DONUT_AUDIT}"
    echo "top_intent=${TOP_INTENT}"
    ;;

  execute-top)
    : "${APPROVED_CONFIG_PATH:?required}"
    : "${APPROVED_DONUT_PLAN_PATH:?required for final session manifest}"
    : "${APPROVED_DONUT_POLICY_PATH:?required for final session manifest}"
    : "${APPROVED_DONUT_BUNDLE_PATH:?required for final session manifest}"
    : "${DONUT_CAPABILITY_CERTIFICATE_PATH:?required for final session manifest}"
    : "${DONUT_EXECUTION_AUDIT_PATH:?required for final session manifest}"
    : "${APPROVED_TOP_PLAN_PATH:?required}"
    : "${APPROVED_TOP_POLICY_PATH:?required}"
    : "${APPROVED_TOP_BUNDLE_PATH:?required}"
    : "${TOP_CAPABILITY_CERTIFICATE_PATH:?required}"
    for path in \
      "${APPROVED_CONFIG_PATH}" "${APPROVED_TOP_PLAN_PATH}" \
      "${APPROVED_TOP_POLICY_PATH}" "${APPROVED_TOP_BUNDLE_PATH}" \
      "${TOP_CAPABILITY_CERTIFICATE_PATH}"; do
      [[ -r "${path}" ]] || {
        echo "[error] approved artifact unreadable: ${path}" >&2
        exit 1
      }
    done
    source_ros
    ensure_perception_services
    start_isolated_ik
    trap cleanup_ik EXIT INT TERM
    TOP_AUDIT="${OUTPUT_DIR}/ninth_top_execution_audit_${RUN_ID}.json"
    run_execute \
      mixed_bread_on_stack \
      "${APPROVED_CONFIG_PATH}" \
      "${APPROVED_TOP_PLAN_PATH}" \
      "${APPROVED_TOP_POLICY_PATH}" \
      "${APPROVED_TOP_BUNDLE_PATH}" \
      "${TOP_CAPABILITY_CERTIFICATE_PATH}" \
      "${TOP_AUDIT}" \
      --verify-after-execution \
      --verification-image "${OUTPUT_DIR}/ninth_final_verification_${RUN_ID}.png"
    SESSION_MANIFEST="${OUTPUT_DIR}/ninth_session_manifest_${RUN_ID}.json"
    "${PYTHON}" scripts/r1lite_build_ninth_session_manifest.py \
      --donut-plan "${APPROVED_DONUT_PLAN_PATH}" \
      --donut-policy "${APPROVED_DONUT_POLICY_PATH}" \
      --donut-bundle "${APPROVED_DONUT_BUNDLE_PATH}" \
      --donut-certificate "${DONUT_CAPABILITY_CERTIFICATE_PATH}" \
      --donut-execution-audit "${DONUT_EXECUTION_AUDIT_PATH}" \
      --top-plan "${APPROVED_TOP_PLAN_PATH}" \
      --top-policy "${APPROVED_TOP_POLICY_PATH}" \
      --top-bundle "${APPROVED_TOP_BUNDLE_PATH}" \
      --top-certificate "${TOP_CAPABILITY_CERTIFICATE_PATH}" \
      --top-execution-audit "${TOP_AUDIT}" \
      --output "${SESSION_MANIFEST}"
    "${PYTHON}" scripts/render_r1lite_ninth_sandwich_html.py \
      --plan "${APPROVED_TOP_PLAN_PATH}" \
      --execution-bundle "${APPROVED_TOP_BUNDLE_PATH}" \
      --execution-audit "${TOP_AUDIT}" \
      --preview-image "${OUTPUT_DIR}/ninth_final_verification_${RUN_ID}.png" \
      --output "${OUTPUT_DIR}/ninth_top_execution_${RUN_ID}.html"
    echo "[complete] Pipeline-nine sandwich stack and final visual verification passed."
    echo "top_execution_audit=${TOP_AUDIT}"
    echo "session_manifest=${SESSION_MANIFEST}"
    ;;

  *)
    usage
    exit 2
    ;;
esac
