#!/usr/bin/env bash
# beizi-2: frozen custom LLM plan + per-waypoint orientation-search executor.
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-outputs/r1lite_three_cup_stack_llm/cup_stack_llm_config_20260828_1703_fixed_orange_grasp.yaml}"
PLAN_PATH="${PLAN_PATH:-outputs/r1lite_three_cup_stack_llm/cup_stack_llm_plan_20260830_beizi2_xy_early_drop.json}"
PLAN_SHA256="4c3f2406055bc1d268d1449f891eb065fe78210422fa8eb2a19ae94c80bb8060"
CODE_PATH="${CODE_PATH:-outputs/r1lite_three_cup_stack_llm/cup_stack_llm_code_20260830_beizi2_xy_early_drop.py}"
CODE_SHA256="1bee191978189b88b3909093421a6f97e62270d7ee415f5572097ecbe8818d83"
PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_beizi_2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r1lite_three_cup_stack_llm}"
AUDIT_PATH="${AUDIT_PATH:-${OUTPUT_DIR}/cup_stack_llm_execution_${RUN_ID}_audit.json}"
TORSO_STATUS="${TORSO_STATUS:-${OUTPUT_DIR}/torso_hold_${RUN_ID}.json}"

if [[ "${BEIZI_2_EXECUTE_ROBOT:-0}" != "1" ]]; then
  echo "pipeline=beizi-2"
  echo "mode=rear-retract+lift+free-orientation+advance"
  echo "plan=${PLAN_PATH}"
  echo "code=${CODE_PATH}"
  echo "[safe] No robot motion requested."
  echo "[run] Set BEIZI_2_EXECUTE_ROBOT=1 to execute."
  exit 0
fi

for path in "${CONFIG_PATH}" "${PLAN_PATH}" "${CODE_PATH}"; do
  [[ -r "${path}" ]] || {
    echo "[error] required file is unreadable: ${path}" >&2
    exit 1
  }
done
[[ "$(sha256sum "${PLAN_PATH}" | awk '{print $1}')" == "${PLAN_SHA256}" ]] || {
  echo "[error] frozen beizi-2 plan hash mismatch" >&2
  exit 1
}
[[ "$(sha256sum "${CODE_PATH}" | awk '{print $1}')" == "${CODE_SHA256}" ]] || {
  echo "[error] frozen beizi-2 code hash mismatch" >&2
  exit 1
}

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

if ! timeout 3 ros2 topic echo --once \
  /hdas/feedback_arm_right sensor_msgs/msg/JointState >/dev/null 2>&1; then
  echo "[error] robot stack is offline; beizi-2 will not restart it" >&2
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

RESUME_ARGS=()
if [[ "${BEIZI_2_RESUME_AFTER_SECOND_LIFT:-0}" == "1" ]]; then
  RESUME_ARGS+=(
    --resume-after-second-lift
    --resume-hold-gripper-command "${BEIZI_2_RESUME_GRIPPER_COMMAND:-42.0}"
  )
elif [[ "${BEIZI_2_RESUME_AFTER_SECOND_TRANSFER:-0}" == "1" ]]; then
  RESUME_ARGS+=(
    --resume-after-second-transfer
    --resume-hold-gripper-command "${BEIZI_2_RESUME_GRIPPER_COMMAND:-42.0}"
  )
fi

"${PYTHON}" scripts/r1lite_execute_llm_cup_stack.py \
  --config "${CONFIG_PATH}" \
  --plan "${PLAN_PATH}" \
  --expected-plan-sha256 "${PLAN_SHA256}" \
  --code "${CODE_PATH}" \
  --expected-code-sha256 "${CODE_SHA256}" \
  --audit-output "${AUDIT_PATH}" \
  --torso-hold-status "${TORSO_STATUS}" \
  --maximum-planned-torso-delta-rad 0.015 \
  --maximum-initial-joint-delta-rad 0.05 \
  "${RESUME_ARGS[@]}"
echo "execution_audit=${AUDIT_PATH}"
