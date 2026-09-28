#!/usr/bin/env bash
# beizi-general: c1 -> c3, then c2 -> c3 using saved cup positions.
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 <first-color> <second-color> <base-color>" >&2
  echo "colors must be one permutation of: orange purple green" >&2
  exit 2
fi

FIRST_COLOR="${1,,}"
SECOND_COLOR="${2,,}"
BASE_COLOR="${3,,}"
PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-outputs/r1lite_three_cup_stack_llm/cup_stack_llm_config_20260828_1703_fixed_orange_grasp.yaml}"
FIXED_SCENE_PATH="${FIXED_SCENE_PATH:-outputs/r1lite_three_cup_stack/fixed_cup_scene_user_20260827.json}"
TEMPLATE_PLAN_PATH="${TEMPLATE_PLAN_PATH:-outputs/r1lite_three_cup_stack_llm/cup_stack_llm_plan_20260830_beizi2_xy_early_drop.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r1lite_three_cup_stack_llm}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_beizi_general_${FIRST_COLOR}_${SECOND_COLOR}_${BASE_COLOR}}"
PLAN_PATH="${PLAN_PATH:-${OUTPUT_DIR}/cup_stack_plan_${RUN_ID}.json}"
CODE_PATH="${CODE_PATH:-${OUTPUT_DIR}/cup_stack_code_${RUN_ID}.py}"
COMPILE_AUDIT_PATH="${COMPILE_AUDIT_PATH:-${OUTPUT_DIR}/cup_stack_compile_${RUN_ID}_audit.json}"
EXECUTION_AUDIT_PATH="${EXECUTION_AUDIT_PATH:-${OUTPUT_DIR}/cup_stack_execution_${RUN_ID}_audit.json}"
TORSO_STATUS="${TORSO_STATUS:-${OUTPUT_DIR}/torso_hold_${RUN_ID}.json}"

for path in "${CONFIG_PATH}" "${FIXED_SCENE_PATH}" "${TEMPLATE_PLAN_PATH}"; do
  [[ -r "${path}" ]] || {
    echo "[error] required file is unreadable: ${path}" >&2
    exit 1
  }
done

"${PYTHON}" scripts/r1lite_compile_beizi_general.py \
  "${FIRST_COLOR}" "${SECOND_COLOR}" "${BASE_COLOR}" \
  --fixed-scene "${FIXED_SCENE_PATH}" \
  --template-plan "${TEMPLATE_PLAN_PATH}" \
  --output-plan "${PLAN_PATH}" \
  --output-code "${CODE_PATH}" \
  --audit-output "${COMPILE_AUDIT_PATH}"

PLAN_SHA256="$(sha256sum "${PLAN_PATH}" | awk '{print $1}')"
CODE_SHA256="$(sha256sum "${CODE_PATH}" | awk '{print $1}')"
echo "pipeline=beizi-general"
echo "color_order=${FIRST_COLOR},${SECOND_COLOR},${BASE_COLOR}"
echo "phase_1=${FIRST_COLOR}_on_${BASE_COLOR}"
echo "phase_2=${SECOND_COLOR}_on_${BASE_COLOR}"
echo "plan=${PLAN_PATH}"
echo "plan_sha256=${PLAN_SHA256}"
echo "code=${CODE_PATH}"
echo "code_sha256=${CODE_SHA256}"

if [[ "${BEIZI_GENERAL_EXECUTE_ROBOT:-0}" != "1" ]]; then
  echo "[safe] Compiled and validated without robot motion."
  echo "[run] Set BEIZI_GENERAL_EXECUTE_ROBOT=1 to execute."
  exit 0
fi

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

if ! timeout 3 ros2 topic echo --once \
  /hdas/feedback_arm_right sensor_msgs/msg/JointState >/dev/null 2>&1; then
  echo "[error] robot stack is offline; beizi-general will not restart it" >&2
  exit 1
fi
if pgrep -af homogeneous_teleoperation | rg -v "pgrep -af" >/dev/null; then
  echo "[error] teleoperation process is active" >&2
  exit 1
fi
if pgrep -af "scripts/r1lite_hold_torso.py" | rg -v "pgrep -af" >/dev/null; then
  echo "[error] another torso hold process is active" >&2
  exit 1
fi

rm -f "${TORSO_STATUS}"
"${PYTHON}" scripts/r1lite_hold_torso.py \
  --ready-file "${TORSO_STATUS}" \
  --maximum-drift-rad 0.03 \
  >"logs/r1lite_torso_hold_${RUN_ID}.log" 2>&1 &
TORSO_PID=$!
cleanup() {
  kill "${TORSO_PID}" 2>/dev/null || true
  wait "${TORSO_PID}" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 150); do
  [[ -s "${TORSO_STATUS}" ]] && break
  kill -0 "${TORSO_PID}" 2>/dev/null || exit 21
  sleep 0.1
done
[[ -s "${TORSO_STATUS}" ]] || {
  echo "[error] torso hold did not become ready" >&2
  exit 1
}

"${PYTHON}" scripts/r1lite_execute_llm_cup_stack.py \
  --config "${CONFIG_PATH}" \
  --plan "${PLAN_PATH}" \
  --expected-plan-sha256 "${PLAN_SHA256}" \
  --code "${CODE_PATH}" \
  --expected-code-sha256 "${CODE_SHA256}" \
  --audit-output "${EXECUTION_AUDIT_PATH}" \
  --torso-hold-status "${TORSO_STATUS}" \
  --maximum-planned-torso-delta-rad 0.015 \
  --maximum-initial-joint-delta-rad 0.05
echo "execution_audit=${EXECUTION_AUDIT_PATH}"
