#!/usr/bin/env bash
# put-general: perceive tennis/peach/apple and place them into a box in order.
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 <first-object> <second-object> <third-object>" >&2
  echo "objects must be one permutation of: tennis peach apple" >&2
  exit 2
fi

PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-env_configs/r1lite/r1lite_tennis_ball_in_box_head_only_live.yaml}"
REFERENCE_PLAN="${REFERENCE_PLAN:-outputs/r1lite_three_cup_stack_llm/cup_stack_llm_plan_20260830_beizi2_xy_early_drop.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/r1lite_put_general}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_put_general}"
AUDIT_PATH="${AUDIT_PATH:-${OUTPUT_DIR}/put_general_${RUN_ID}_audit.json}"
TORSO_STATUS="${TORSO_STATUS:-${OUTPUT_DIR}/torso_hold_${RUN_ID}.json}"
LEFT_HOLD_STATUS="${LEFT_HOLD_STATUS:-${OUTPUT_DIR}/left_home_hold_${RUN_ID}.json}"

normalize_object() {
  case "${1,,}" in
    tennis|tennis-ball|tennis_ball|"tennis ball") echo "tennis" ;;
    peach) echo "peach" ;;
    apple) echo "apple" ;;
    *) return 1 ;;
  esac
}
FIRST_OBJECT="$(normalize_object "$1")" || {
  echo "[error] objects must be one unique permutation of tennis, peach, apple" >&2
  exit 2
}
SECOND_OBJECT="$(normalize_object "$2")" || {
  echo "[error] objects must be one unique permutation of tennis, peach, apple" >&2
  exit 2
}
THIRD_OBJECT="$(normalize_object "$3")" || {
  echo "[error] objects must be one unique permutation of tennis, peach, apple" >&2
  exit 2
}
if [[ "${FIRST_OBJECT}" == "${SECOND_OBJECT}" \
      || "${FIRST_OBJECT}" == "${THIRD_OBJECT}" \
      || "${SECOND_OBJECT}" == "${THIRD_OBJECT}" ]]; then
  echo "[error] objects must be one unique permutation of tennis, peach, apple" >&2
  exit 2
fi

echo "pipeline=put-general"
echo "object_order=${FIRST_OBJECT},${SECOND_OBJECT},${THIRD_OBJECT}"
echo "config=${CONFIG_PATH}"
echo "audit=${AUDIT_PATH}"

if [[ "${PUT_GENERAL_EXECUTE_ROBOT:-0}" != "1" \
      && "${PUT_GENERAL_PERCEPTION_ONLY:-0}" != "1" ]]; then
  echo "[safe] No perception or robot motion requested."
  echo "[perceive] Set PUT_GENERAL_PERCEPTION_ONLY=1."
  echo "[run] Set PUT_GENERAL_EXECUTE_ROBOT=1."
  exit 0
fi

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

if ! timeout 5 ros2 topic echo --once \
  /hdas/feedback_arm_right sensor_msgs/msg/JointState >/dev/null 2>&1; then
  echo "[error] robot stack is offline" >&2
  exit 1
fi
if ! timeout 8 ros2 topic echo --once \
  /capx/head_stereo/aligned_depth_to_left/image_raw >/dev/null 2>&1; then
  echo "[error] head stereo depth is offline" >&2
  echo "Run scripts/run_r1lite_tennis_ball_in_box_pipeline.sh in setup-only mode." >&2
  exit 1
fi
if ! timeout 1 bash -c ":</dev/tcp/127.0.0.1/8114" >/dev/null 2>&1; then
  echo "[error] SAM3 service is unavailable on port 8114" >&2
  exit 1
fi

if [[ "${PUT_GENERAL_PERCEPTION_ONLY:-0}" == "1" ]]; then
  "${PYTHON}" scripts/r1lite_execute_put_general.py \
    "${FIRST_OBJECT}" "${SECOND_OBJECT}" "${THIRD_OBJECT}" \
    --config "${CONFIG_PATH}" \
    --reference-plan "${REFERENCE_PLAN}" \
    --audit-output "${AUDIT_PATH}" \
    --perception-only
  echo "perception_audit=${AUDIT_PATH}"
  exit 0
fi

if pgrep -af homogeneous_teleoperation | rg -v "pgrep -af" >/dev/null; then
  echo "[error] teleoperation process is active" >&2
  exit 1
fi
if pgrep -af "scripts/r1lite_hold_torso.py" | rg -v "pgrep -af" >/dev/null; then
  echo "[error] another torso hold process is active" >&2
  exit 1
fi

echo "[prepare] start isolated right-arm RelaxedIK"
tmux kill-session -t r1lite_relaxed_ik_isolated 2>/dev/null || true
pkill -TERM -f '[r]elaxed_ik.py 0 1 0' 2>/dev/null || true
sleep 2
tmux new-session -d -s r1lite_relaxed_ik_isolated "bash -lc '
  export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
  export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}
  export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}
  source /opt/ros/humble/setup.bash
  source /home/nvidia/galaxea/install/local_setup.bash
  exec ros2 run mobiman relaxed_ik.py 0 1 0 --ros-args \
    -p robot_name:=R1_Lite \
    -r __node:=relaxed_ik_right \
    -r /motion_target/target_pose_arm_right:=/capx/relaxed_ik_target_pose_arm_right \
    -r /motion_target/target_joint_state_arm_right:=/relaxed_ik/isolated_target_joint_state_arm_right \
    >> ${REPO_ROOT}/logs/relaxed_ik_right_isolated.log 2>&1
'"
RIGHT_IK_READY=0
for _ in $(seq 1 40); do
  IK_INFO="$(ROS2CLI_NO_DAEMON=1 ros2 topic info -v \
    /capx/relaxed_ik_target_pose_arm_right 2>/dev/null || true)"
  if rg -q "Node name: relaxed_ik_right" <<<"${IK_INFO}"; then
    RIGHT_IK_READY=1
    break
  fi
  sleep 0.5
done
if [[ "${RIGHT_IK_READY}" != "1" ]]; then
  tmux kill-session -t r1lite_relaxed_ik_isolated 2>/dev/null || true
  pkill -TERM -f '[r]elaxed_ik.py 0 1 0' 2>/dev/null || true
  echo "[error] isolated right-arm RelaxedIK did not become ready" >&2
  exit 1
fi
cleanup_right_ik() {
  tmux kill-session -t r1lite_relaxed_ik_isolated 2>/dev/null || true
  pkill -TERM -f '[r]elaxed_ik.py 0 1 0' 2>/dev/null || true
}
trap cleanup_right_ik EXIT

echo "[prepare] stop left-arm RelaxedIK and hold left Home"
pkill -TERM -f '[r]elaxed_ik.py 1 0 0' 2>/dev/null || true
sleep 1
if pgrep -f '[r]elaxed_ik.py 1 0 0' >/dev/null; then
  echo "[error] left-arm RelaxedIK did not stop" >&2
  exit 1
fi

rm -f "${TORSO_STATUS}" "${LEFT_HOLD_STATUS}"
"${PYTHON}" scripts/r1lite_hold_left_arm_home.py \
  --ready-file "${LEFT_HOLD_STATUS}" \
  >"logs/r1lite_left_home_hold_${RUN_ID}.log" 2>&1 &
LEFT_HOLD_PID=$!
"${PYTHON}" scripts/r1lite_hold_torso.py \
  --ready-file "${TORSO_STATUS}" \
  --maximum-drift-rad 0.03 \
  >"logs/r1lite_torso_hold_${RUN_ID}.log" 2>&1 &
TORSO_PID=$!
cleanup() {
  "${REPO_ROOT}/scripts/stop_r1lite_moca_isolated.sh" 2>/dev/null || true
  cleanup_right_ik
  kill "${LEFT_HOLD_PID}" 2>/dev/null || true
  kill "${TORSO_PID}" 2>/dev/null || true
  wait "${LEFT_HOLD_PID}" 2>/dev/null || true
  wait "${TORSO_PID}" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 150); do
  [[ -s "${TORSO_STATUS}" && -s "${LEFT_HOLD_STATUS}" ]] && break
  kill -0 "${LEFT_HOLD_PID}" 2>/dev/null || exit 20
  kill -0 "${TORSO_PID}" 2>/dev/null || exit 21
  sleep 0.1
done
[[ -s "${TORSO_STATUS}" && -s "${LEFT_HOLD_STATUS}" ]] || {
  echo "[error] left-arm or torso hold did not become ready" >&2
  exit 1
}

"${PYTHON}" scripts/r1lite_execute_put_general.py \
  "${FIRST_OBJECT}" "${SECOND_OBJECT}" "${THIRD_OBJECT}" \
  --config "${CONFIG_PATH}" \
  --reference-plan "${REFERENCE_PLAN}" \
  --torso-hold-status "${TORSO_STATUS}" \
  --left-arm-hold-status "${LEFT_HOLD_STATUS}" \
  --audit-output "${AUDIT_PATH}" \
  --maximum-initial-joint-delta-rad 0.05 \
  --maximum-torso-drift-rad 0.03
echo "execution_audit=${AUDIT_PATH}"
