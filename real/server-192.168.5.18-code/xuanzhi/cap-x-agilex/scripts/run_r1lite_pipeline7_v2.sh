#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r1lite_dual_arm_towel_fold}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_pipeline7_v2}"
PIPELINE_LABEL="${PIPELINE_LABEL:-pipeline7-v2}"
EXECUTE_ENV_HINT="${EXECUTE_ENV_HINT:-PIPELINE7_V2_EXECUTE_ROBOT=1}"

CONFIG_PATH="${CONFIG_PATH:-${OUTPUT_DIR}/seventh_pipeline_config_20260825_1750_fresh_head_p7.yaml}"
CALIBRATION_PATH="${CALIBRATION_PATH:-outputs/r1lite/head_plane_calibration.json}"
PLAN_PATH="${PLAN_PATH:-${OUTPUT_DIR}/towel_fold_plan_selected_20260825_1928_right_side_06_z_monotonic.json}"
CODE_PATH="${CODE_PATH:-${OUTPUT_DIR}/towel_fold_llm_program_selected_20260825_1928_right06_zmono.py}"
BUNDLE_PATH="${BUNDLE_PATH:-${OUTPUT_DIR}/sequential_execution_bundle_20260825_1944_right06_zmono_strict.json}"
COLLISION_AUDIT_PATH="${COLLISION_AUDIT_PATH:-${OUTPUT_DIR}/towel_collision_audit_20260825_1944_right06_zmono_strict.json}"
CERTIFICATE_PATH="${CERTIFICATE_PATH:-${OUTPUT_DIR}/towel_capability_certificate_20260825_1944_right06_zmono_strict.json}"
EXECUTION_AUDIT_PATH="${EXECUTION_AUDIT_PATH:-${OUTPUT_DIR}/towel_execution_audit_${RUN_ID}.json}"

for required in \
  "${PYTHON}" \
  "${CONFIG_PATH}" \
  "${CALIBRATION_PATH}" \
  "${PLAN_PATH}" \
  "${CODE_PATH}" \
  "${BUNDLE_PATH}" \
  "${COLLISION_AUDIT_PATH}" \
  "${CERTIFICATE_PATH}"; do
  if [[ ! -r "${required}" ]]; then
    echo "[pipeline7-v2] missing required artifact: ${required}" >&2
    exit 1
  fi
done

"${PYTHON}" - "${PLAN_PATH}" "${COLLISION_AUDIT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text())
audit = json.loads(Path(sys.argv[2]).read_text())
policy = plan.get("safety", {}).get("post_close_tcp_z_policy", {})
z_audit = audit.get("post_close_tcp_z_monotonic", {})
if policy.get("enabled") is not True:
    raise SystemExit("Pipeline 7 V2 requires the post-close TCP Z policy")
if audit.get("passed") is not True or z_audit.get("passed") is not True:
    raise SystemExit("Pipeline 7 V2 collision/Z certification did not pass")
PY

if [[ "${PIPELINE7_V2_EXECUTE_ROBOT:-0}" != "1" ]]; then
  echo "[${PIPELINE_LABEL}] artifacts validated; no robot motion sent"
  echo "[${PIPELINE_LABEL}] set ${EXECUTE_ENV_HINT} to execute"
  echo "[${PIPELINE_LABEL}] plan=${PLAN_PATH}"
  echo "[${PIPELINE_LABEL}] bundle=${BUNDLE_PATH}"
  exit 0
fi

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
export R1LITE_RIGHT_JOINT2_CONTROL_OFFSET_DEG=0
if [[ "${PIPELINE7_V1_MODE:-0}" == "1" ]]; then
  # Preserve the historical replay refinement and permissive close decision.
  export R1LITE_P7_STRICT_CONTACT=0
  export R1LITE_P7_LEGACY_REPLAY_REFINEMENT=1
else
  # Require stable near-table effort contact for three consecutive samples.
  # No-contact and unavailable-descent outcomes retract and abort without closing.
  export R1LITE_P7_STRICT_CONTACT=1
  export R1LITE_P7_LEGACY_REPLAY_REFINEMENT=0
  export R1LITE_ARRIVAL_SOFT_THRESHOLD_RAD=0.01
  export R1LITE_INITIAL_STATE_TOLERANCE_RAD=0.01
fi
# After the initial 10 mm probe, test 5..10 mm downward IK targets from the
# same post-probe pose and execute only the first target that passes all gates.
export R1LITE_P7_CANDIDATE_DESCENTS_MM=5,6,7,8,9,10
# Correct small Home-reference drift inside the executor before its 10 mrad
# initial-state gate; the normal 50 mrad hard limit remains in force.
export R1LITE_PREALIGN_INITIAL_STATE=1

mkdir -p "${OUTPUT_DIR}" logs
setsid ros2 run mobiman relaxed_ik.py 1 0 0 --ros-args \
  -p robot_name:=R1_Lite \
  -r __node:="capx_pipeline7_v2_left_${RUN_ID}" \
  -r /motion_target/target_pose_arm_left:=/capx/relaxed_ik/target_pose_arm_left \
  -r /motion_target/target_joint_state_arm_left:=/relaxed_ik/isolated_target_joint_state_arm_left \
  >"logs/pipeline7_v2_left_${RUN_ID}.log" 2>&1 &
LEFT_IK_PGID=$!
setsid ros2 run mobiman relaxed_ik.py 0 1 0 --ros-args \
  -p robot_name:=R1_Lite \
  -r __node:="capx_pipeline7_v2_right_${RUN_ID}" \
  -r /motion_target/target_pose_arm_right:=/capx/relaxed_ik/target_pose_arm_right \
  -r /motion_target/target_joint_state_arm_right:=/relaxed_ik/isolated_target_joint_state_arm_right \
  >"logs/pipeline7_v2_right_${RUN_ID}.log" 2>&1 &
RIGHT_IK_PGID=$!

cleanup() {
  kill -TERM -- "-${LEFT_IK_PGID}" "-${RIGHT_IK_PGID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
sleep 5

for command in 2 5 1; do
  response="$(ros2 service call \
    /function_frame_right_arm hdas_msg/srv/FunctionFrame \
    "{command: ${command}}")"
  if [[ "${response}" != *"success=True"* ]]; then
    echo "[pipeline7-v2] failed to enter right-arm MIT mode" >&2
    exit 1
  fi
done

PLAN_SHA256="$(sha256sum "${PLAN_PATH}" | awk '{print $1}')"
CODE_SHA256="$(sha256sum "${CODE_PATH}" | awk '{print $1}')"
BUNDLE_SHA256="$(sha256sum "${BUNDLE_PATH}" | awk '{print $1}')"
CERTIFICATE_SHA256="$(sha256sum "${CERTIFICATE_PATH}" | awk '{print $1}')"

"${PYTHON}" scripts/r1lite_seventh_dual_arm_towel_fold.py \
  --mode execute \
  --config "${CONFIG_PATH}" \
  --calibration "${CALIBRATION_PATH}" \
  --audit-output "${EXECUTION_AUDIT_PATH}" \
  --approved-plan "${PLAN_PATH}" \
  --expected-plan-sha256 "${PLAN_SHA256}" \
  --generated-code "${CODE_PATH}" \
  --expected-code-sha256 "${CODE_SHA256}" \
  --execution-bundle "${BUNDLE_PATH}" \
  --expected-execution-bundle-sha256 "${BUNDLE_SHA256}" \
  --capability-certificate "${CERTIFICATE_PATH}" \
  --expected-capability-certificate-sha256 "${CERTIFICATE_SHA256}"

echo "[pipeline7-v2] execution_audit=${EXECUTION_AUDIT_PATH}"
