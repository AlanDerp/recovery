#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Pipeline 8 V1:
# - replay the saved Pipeline-8 plan and saved grasp refinement
# - right joint-2 control compensation fixed at 0 degrees
# - joint-arrival hard gate disabled for the complete motion
# - soft arrival, stale-feedback, contact, gripper, and Home gates remain active
export APPROVED_CONFIG_PATH="${APPROVED_CONFIG_PATH:-outputs/r1lite_eighth_towel_pipeline/eighth_config_20260815_1956_first_phase_only.yaml}"
export APPROVED_EIGHTH_PLAN_PATH="${APPROVED_EIGHTH_PLAN_PATH:-outputs/r1lite_eighth_towel_pipeline/eighth_plan_20260815_1956_first_phase_only.json}"
export APPROVED_EIGHTH_POLICY_PATH="${APPROVED_EIGHTH_POLICY_PATH:-outputs/r1lite_eighth_towel_pipeline/eighth_policy_20260815_1956_first_phase_only.json}"
export APPROVED_EIGHTH_BUNDLE_PATH="${APPROVED_EIGHTH_BUNDLE_PATH:-outputs/r1lite_eighth_towel_pipeline/eighth_execution_bundle_20260815_1956_first_phase_only.json}"
export EIGHTH_CAPABILITY_CERTIFICATE_PATH="${EIGHTH_CAPABILITY_CERTIFICATE_PATH:-outputs/r1lite_eighth_towel_pipeline/eighth_capability_certificate_20260825_2012_saved_replay.json}"
export SAVED_EIGHTH_GRASP_REFINEMENT_PATH="${SAVED_EIGHTH_GRASP_REFINEMENT_PATH:-outputs/r1lite_eighth_towel_pipeline/saved_eighth_grasp_refinement_20260816_1752.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/r1lite_eighth_towel_pipeline}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_pipeline8_v1}"
export R1LITE_RIGHT_JOINT2_CONTROL_OFFSET_DEG=0
export R1LITE_ARRIVAL_HARD_THRESHOLD_RAD=inf

for path in \
  "${APPROVED_CONFIG_PATH}" \
  "${APPROVED_EIGHTH_PLAN_PATH}" \
  "${APPROVED_EIGHTH_POLICY_PATH}" \
  "${APPROVED_EIGHTH_BUNDLE_PATH}" \
  "${EIGHTH_CAPABILITY_CERTIFICATE_PATH}" \
  "${SAVED_EIGHTH_GRASP_REFINEMENT_PATH}"; do
  [[ -r "${path}" ]] || {
    echo "[error] unreadable Pipeline 8 V1 artifact: ${path}" >&2
    exit 1
  }
done

if [[ "${PIPELINE8_V1_EXECUTE_ROBOT:-0}" != "1" ]]; then
  echo "[dry-run] Pipeline 8 V1 is configured but robot execution is disabled."
  echo "right_joint2_control_offset_deg=0"
  echo "saved_grasp_refinement=${SAVED_EIGHTH_GRASP_REFINEMENT_PATH}"
  echo "joint_arrival_hard_gate=disabled"
  echo "execute=PIPELINE8_V1_EXECUTE_ROBOT=1 $0"
  exit 0
fi

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

mkdir -p "${OUTPUT_DIR}" logs
LEFT_IK_LOG="logs/r1lite_pipeline8_v1_left_ik_${RUN_ID}.log"
RIGHT_IK_LOG="logs/r1lite_pipeline8_v1_right_ik_${RUN_ID}.log"

setsid ros2 run mobiman relaxed_ik.py 1 0 0 --ros-args \
  -p robot_name:=R1_Lite \
  -r __node:=capx_pipeline8_v1_left \
  -r /motion_target/target_pose_arm_left:=/capx/relaxed_ik/target_pose_arm_left \
  -r /motion_target/target_joint_state_arm_left:=/relaxed_ik/isolated_target_joint_state_arm_left \
  >"${LEFT_IK_LOG}" 2>&1 &
LEFT_IK_PGID=$!

setsid ros2 run mobiman relaxed_ik.py 0 1 0 --ros-args \
  -p robot_name:=R1_Lite \
  -r __node:=capx_pipeline8_v1_right \
  -r /motion_target/target_pose_arm_right:=/capx/relaxed_ik/target_pose_arm_right \
  -r /motion_target/target_joint_state_arm_right:=/relaxed_ik/isolated_target_joint_state_arm_right \
  >"${RIGHT_IK_LOG}" 2>&1 &
RIGHT_IK_PGID=$!

cleanup() {
  kill -TERM -- "-${LEFT_IK_PGID}" "-${RIGHT_IK_PGID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
sleep 5

for command in 2 5 1; do
  response="$(
    ros2 service call /function_frame_right_arm \
      hdas_msg/srv/FunctionFrame "{command: ${command}}"
  )"
  [[ "${response}" == *"success=True"* ]] || {
    echo "[error] right-arm mode command ${command} failed" >&2
    exit 1
  }
done

scripts/run_r1lite_eighth_towel_pipeline.sh execute-eighth
