#!/usr/bin/env bash
# Pipeline 8 first phase, then fresh Pipeline-7 perception and execution.
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

STAGE="${1:-}"
PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"
SOURCE_CONFIG_PATH="${SOURCE_CONFIG_PATH:-tmp/r1lite_tennis_ball_in_box_oracle_calibrated_table.yaml}"
CALIBRATION_PATH="${CALIBRATION_PATH:-outputs/r1lite/head_plane_calibration.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r1lite_eighth_towel_pipeline}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SAVED_P7_PLAN_PATH="${SAVED_P7_PLAN_PATH:-outputs/r1lite_dual_arm_towel_fold/towel_fold_plan_selected_20260814_2223_right_v34.json}"
SAVED_P7_BUNDLE_PATH="${SAVED_P7_BUNDLE_PATH:-outputs/r1lite_dual_arm_towel_fold/sequential_execution_bundle_20260814_2223_right_v34.json}"
SAVED_P7_GRASP_SEQUENCE_PATH="${SAVED_P7_GRASP_SEQUENCE_PATH:-outputs/r1lite_dual_arm_towel_fold/saved_grasp_joint_sequence_20260815_1920.json}"
SAVED_EIGHTH_GRASP_REFINEMENT_PATH="${SAVED_EIGHTH_GRASP_REFINEMENT_PATH:-outputs/r1lite_eighth_towel_pipeline/saved_eighth_grasp_refinement_20260816_1752.json}"

usage() {
  echo "usage: $0 {prepare-eighth|execute-eighth|execute-eighth-then-saved-right|prepare-right|execute-right}" >&2
}

[[ -x "${PYTHON}" ]] || {
  echo "[error] Python interpreter unavailable: ${PYTHON}" >&2
  exit 1
}
mkdir -p "${OUTPUT_DIR}" logs

case "${STAGE}" in
  prepare-eighth)
    for path in "${SOURCE_CONFIG_PATH}" "${CALIBRATION_PATH}"; do
      [[ -r "${path}" ]] || {
        echo "[error] unreadable: ${path}" >&2
        exit 1
      }
    done
    CONFIG_PATH="${OUTPUT_DIR}/eighth_config_${RUN_ID}.yaml"
    PLAN_PATH="${OUTPUT_DIR}/eighth_plan_${RUN_ID}.json"
    POLICY_PATH="${OUTPUT_DIR}/eighth_policy_${RUN_ID}.json"
    "${PYTHON}" - "${SOURCE_CONFIG_PATH}" "${CONFIG_PATH}" <<'PY'
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
env["low_level"]["wrist_camera_enabled"] = False
destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
    set +u
    source /opt/ros/humble/setup.bash
    source /home/nvidia/galaxea/install/local_setup.bash
    set -u
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
    export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
    export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
    "${PYTHON}" scripts/r1lite_eighth_towel_recenter_then_right.py \
      --mode preview \
      --config "${CONFIG_PATH}" \
      --calibration "${CALIBRATION_PATH}" \
      --output-plan "${PLAN_PATH}" \
      --output-policy "${POLICY_PATH}" \
      --audit-output "${OUTPUT_DIR}/eighth_preview_audit_${RUN_ID}.json" \
      --raw-image "${OUTPUT_DIR}/eighth_raw_${RUN_ID}.png" \
      --preview-image "${OUTPUT_DIR}/eighth_preview_${RUN_ID}.png" \
      --mask-image "${OUTPUT_DIR}/eighth_mask_${RUN_ID}.png"
    echo "[approval-boundary] Review and offline-certify the first-phase plan."
    echo "config=${CONFIG_PATH}"
    echo "plan=${PLAN_PATH}"
    echo "policy=${POLICY_PATH}"
    ;;

  execute-eighth)
    : "${APPROVED_CONFIG_PATH:?required}"
    : "${APPROVED_EIGHTH_PLAN_PATH:?required}"
    : "${APPROVED_EIGHTH_POLICY_PATH:?required}"
    : "${APPROVED_EIGHTH_BUNDLE_PATH:?required}"
    : "${EIGHTH_CAPABILITY_CERTIFICATE_PATH:?required}"
    for path in \
      "${APPROVED_CONFIG_PATH}" "${APPROVED_EIGHTH_PLAN_PATH}" \
      "${APPROVED_EIGHTH_POLICY_PATH}" "${APPROVED_EIGHTH_BUNDLE_PATH}" \
      "${EIGHTH_CAPABILITY_CERTIFICATE_PATH}"; do
      [[ -r "${path}" ]] || {
        echo "[error] approved artifact unreadable: ${path}" >&2
        exit 1
      }
    done
    set +u
    source /opt/ros/humble/setup.bash
    source /home/nvidia/galaxea/install/local_setup.bash
    set -u
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
    export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
    export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
    PLAN_SHA="$(sha256sum "${APPROVED_EIGHTH_PLAN_PATH}" | awk '{print $1}')"
    POLICY_SHA="$(sha256sum "${APPROVED_EIGHTH_POLICY_PATH}" | awk '{print $1}')"
    BUNDLE_SHA="$(sha256sum "${APPROVED_EIGHTH_BUNDLE_PATH}" | awk '{print $1}')"
    CERT_SHA="$(sha256sum "${EIGHTH_CAPABILITY_CERTIFICATE_PATH}" | awk '{print $1}')"
    "${PYTHON}" scripts/r1lite_eighth_towel_recenter_then_right.py \
      --mode execute \
      --config "${APPROVED_CONFIG_PATH}" \
      --audit-output "${OUTPUT_DIR}/eighth_execution_audit_${RUN_ID}.json" \
      --approved-plan "${APPROVED_EIGHTH_PLAN_PATH}" \
      --expected-plan-sha256 "${PLAN_SHA}" \
      --approved-policy "${APPROVED_EIGHTH_POLICY_PATH}" \
      --expected-policy-sha256 "${POLICY_SHA}" \
      --execution-bundle "${APPROVED_EIGHTH_BUNDLE_PATH}" \
      --expected-execution-bundle-sha256 "${BUNDLE_SHA}" \
      --capability-certificate "${EIGHTH_CAPABILITY_CERTIFICATE_PATH}" \
      --expected-capability-certificate-sha256 "${CERT_SHA}" \
      --accept-first-contact-indication \
      --saved-grasp-refinement-sequence "${SAVED_EIGHTH_GRASP_REFINEMENT_PATH}"
    echo "[complete] Eighth first phase finished with both arms home."
    echo "[next] Run prepare-right for fresh perception."
    ;;

  execute-eighth-then-saved-right)
    : "${APPROVED_CONFIG_PATH:?required}"
    : "${APPROVED_EIGHTH_PLAN_PATH:?required}"
    : "${APPROVED_EIGHTH_POLICY_PATH:?required}"
    : "${APPROVED_EIGHTH_BUNDLE_PATH:?required}"
    : "${EIGHTH_CAPABILITY_CERTIFICATE_PATH:?required}"
    for path in \
      "${SAVED_P7_PLAN_PATH}" "${SAVED_P7_BUNDLE_PATH}" \
      "${SAVED_P7_GRASP_SEQUENCE_PATH}" \
      "${SAVED_EIGHTH_GRASP_REFINEMENT_PATH}"; do
      [[ -r "${path}" ]] || {
        echo "[error] saved Pipeline-7 artifact unreadable: ${path}" >&2
        exit 1
      }
    done
    set +u
    source /opt/ros/humble/setup.bash
    source /home/nvidia/galaxea/install/local_setup.bash
    set -u
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
    export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
    export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
    LEFT_IK_LOG="logs/r1lite_eighth_saved_p7_left_ik_${RUN_ID}.log"
    RIGHT_IK_LOG="logs/r1lite_eighth_saved_p7_right_ik_${RUN_ID}.log"
    setsid ros2 run mobiman relaxed_ik.py 1 0 0 --ros-args \
      -p robot_name:=R1_Lite \
      -r __node:=capx_eighth_saved_p7_left \
      -r /motion_target/target_pose_arm_left:=/capx/relaxed_ik/target_pose_arm_left \
      -r /motion_target/target_joint_state_arm_left:=/relaxed_ik/isolated_target_joint_state_arm_left \
      >"${LEFT_IK_LOG}" 2>&1 &
    LEFT_IK_PGID=$!
    setsid ros2 run mobiman relaxed_ik.py 0 1 0 --ros-args \
      -p robot_name:=R1_Lite \
      -r __node:=capx_eighth_saved_p7_right \
      -r /motion_target/target_pose_arm_right:=/capx/relaxed_ik/target_pose_arm_right \
      -r /motion_target/target_joint_state_arm_right:=/relaxed_ik/isolated_target_joint_state_arm_right \
      >"${RIGHT_IK_LOG}" 2>&1 &
    RIGHT_IK_PGID=$!
    cleanup_saved_ik() {
      kill -- "-${LEFT_IK_PGID}" "-${RIGHT_IK_PGID}" 2>/dev/null || true
    }
    trap cleanup_saved_ik EXIT INT TERM
    sleep 5
    FIRST_PHASE_RUN_ID="${RUN_ID}_first_phase"
    RUN_ID="${FIRST_PHASE_RUN_ID}" "$0" execute-eighth
    FIRST_PHASE_AUDIT="${OUTPUT_DIR}/eighth_execution_audit_${FIRST_PHASE_RUN_ID}.json"
    "${PYTHON}" - "${FIRST_PHASE_AUDIT}" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text())
if audit.get("passed") is not True or audit.get("home_completed") is not True:
    raise RuntimeError("eighth first phase did not pass its full-home gate")
PY
    for command in 2 5 1; do
      response="$(ros2 service call /function_frame_right_arm \
        hdas_msg/srv/FunctionFrame "{command: ${command}}")"
      [[ "${response}" == *"success=True"* ]] || {
        echo "[error] right-arm MIT mode command ${command} failed" >&2
        exit 1
      }
    done
    sleep 2
    SAVED_REPLAY_AUDIT="${OUTPUT_DIR}/saved_p7_replay_audit_${RUN_ID}.json"
    "${PYTHON}" scripts/r1lite_replay_saved_ik_sequence.py \
      --bundle "${SAVED_P7_BUNDLE_PATH}" \
      --saved-grasp-joint-sequence "${SAVED_P7_GRASP_SEQUENCE_PATH}" \
      --grasp-refinement-plan "${SAVED_P7_PLAN_PATH}" \
      --feedback-downward-step-m 0.005 \
      --feedback-downward-step-candidates-m \
        0.005 0.006 0.007 0.008 0.009 0.01 \
      --feedback-downward-refinement-iterations 1 \
      --stop-on-effort-contact \
      --ignore-table-height-clamp \
      --ignore-local-fk-interpolation-z-gate \
      --audit-output "${SAVED_REPLAY_AUDIT}"
    COMBINED_AUDIT="${OUTPUT_DIR}/eighth_then_saved_p7_audit_${RUN_ID}.json"
    "${PYTHON}" - "${FIRST_PHASE_AUDIT}" "${SAVED_REPLAY_AUDIT}" "${COMBINED_AUDIT}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

first_path, replay_path, output_path = map(Path, sys.argv[1:])
first_raw = first_path.read_bytes()
replay_raw = replay_path.read_bytes()
first = json.loads(first_raw)
replay = json.loads(replay_raw)
result = {
    "schema": "capx.r1lite.eighth_then_saved_p7.v1",
    "created_at": datetime.now().astimezone().isoformat(),
    "passed": first.get("passed") is True and replay.get("passed") is True,
    "fresh_perception_after_eighth": False,
    "first_phase_home_completed": first.get("home_completed") is True,
    "saved_p7_online_refinement_stage_ids": replay.get(
        "refinement_stage_ids", []
    ),
    "saved_p7_collision_check_enabled": replay.get(
        "grasp_refinement_collision_check_enabled"
    ),
    "saved_p7_grasp_joint_replay_enabled": replay.get(
        "saved_grasp_joint_replay_enabled"
    ),
    "saved_p7_grasp_joint_sequence": replay.get(
        "saved_grasp_joint_sequence"
    ),
    "first_phase_audit_sha256": hashlib.sha256(first_raw).hexdigest(),
    "saved_p7_replay_audit_sha256": hashlib.sha256(replay_raw).hexdigest(),
}
if result["passed"] is not True:
    raise RuntimeError("combined eighth and saved Pipeline-7 execution failed")
output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps(result, ensure_ascii=False))
PY
    echo "[complete] Eighth first phase and saved Pipeline-7 replay passed."
    echo "combined_audit=${COMBINED_AUDIT}"
    ;;

  prepare-right)
    : "${APPROVED_CONFIG_PATH:?required}"
    : "${APPROVED_EIGHTH_PLAN_PATH:?required}"
    : "${APPROVED_EIGHTH_POLICY_PATH:?required}"
    : "${APPROVED_EIGHTH_BUNDLE_PATH:?required}"
    : "${EIGHTH_CAPABILITY_CERTIFICATE_PATH:?required}"
    : "${EIGHTH_EXECUTION_AUDIT_PATH:?required}"
    RIGHT_PLAN="${OUTPUT_DIR}/fresh_right_plan_${RUN_ID}.json"
    RIGHT_PERCEPTION_AUDIT="${OUTPUT_DIR}/fresh_right_perception_audit_${RUN_ID}.json"
    HANDOFF_MANIFEST="${OUTPUT_DIR}/eighth_to_seventh_handoff_${RUN_ID}.json"
    set +u
    source /opt/ros/humble/setup.bash
    source /home/nvidia/galaxea/install/local_setup.bash
    set -u
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
    export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
    export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
    "${PYTHON}" scripts/r1lite_eighth_towel_recenter_then_right.py \
      --mode handoff \
      --config "${APPROVED_CONFIG_PATH}" \
      --calibration "${CALIBRATION_PATH}" \
      --approved-plan "${APPROVED_EIGHTH_PLAN_PATH}" \
      --approved-policy "${APPROVED_EIGHTH_POLICY_PATH}" \
      --execution-bundle "${APPROVED_EIGHTH_BUNDLE_PATH}" \
      --capability-certificate "${EIGHTH_CAPABILITY_CERTIFICATE_PATH}" \
      --eighth-execution-audit "${EIGHTH_EXECUTION_AUDIT_PATH}" \
      --handoff-manifest "${HANDOFF_MANIFEST}" \
      --right-plan "${RIGHT_PLAN}" \
      --right-perception-audit "${RIGHT_PERCEPTION_AUDIT}" \
      --right-raw-image "${OUTPUT_DIR}/fresh_right_raw_${RUN_ID}.png" \
      --right-preview-image "${OUTPUT_DIR}/fresh_right_preview_${RUN_ID}.png" \
      --right-mask-image "${OUTPUT_DIR}/fresh_right_mask_${RUN_ID}.png"
    echo "[approval-boundary] Fresh Pipeline-7 perception completed."
    echo "right_plan=${RIGHT_PLAN}"
    echo "right_perception_audit=${RIGHT_PERCEPTION_AUDIT}"
    echo "handoff_manifest=${HANDOFF_MANIFEST}"
    ;;

  execute-right)
    : "${APPROVED_CONFIG_PATH:?required}"
    : "${APPROVED_RIGHT_PLAN_PATH:?required}"
    : "${APPROVED_RIGHT_CODE_PATH:?required}"
    : "${APPROVED_RIGHT_BUNDLE_PATH:?required}"
    : "${RIGHT_CAPABILITY_CERTIFICATE_PATH:?required}"
    RUN_ID="${RUN_ID}" \
    SEVENTH_PIPELINE_EXECUTE_ROBOT=1 \
    APPROVED_CONFIG_PATH="${APPROVED_CONFIG_PATH}" \
    APPROVED_PLAN_PATH="${APPROVED_RIGHT_PLAN_PATH}" \
    APPROVED_CODE_PATH="${APPROVED_RIGHT_CODE_PATH}" \
    APPROVED_BUNDLE_PATH="${APPROVED_RIGHT_BUNDLE_PATH}" \
    CAPABILITY_CERTIFICATE_PATH="${RIGHT_CAPABILITY_CERTIFICATE_PATH}" \
      scripts/run_r1lite_seventh_dual_arm_towel_fold_pipeline.sh
    echo "[complete] Fresh Pipeline-7 right-arm phase finished."
    ;;

  *)
    usage
    exit 2
    ;;
esac
