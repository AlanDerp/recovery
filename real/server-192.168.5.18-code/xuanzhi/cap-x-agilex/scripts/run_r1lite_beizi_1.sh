#!/usr/bin/env bash
# beizi-1: native Cap-X Agent0, without the custom cup planner/executor.
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-outputs/r1lite_three_cup_stack_llm/cup_stack_llm_config_20260828_1703_fixed_orange_grasp.yaml}"
MODEL="${MODEL:-gpt-5.5}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
API_KEY_FILE="${API_KEY_FILE:-${REPO_ROOT}/.usezyla_key}"
PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"

if [[ "${BEIZI_1_EXECUTE_ROBOT:-0}" != "1" ]]; then
  echo "pipeline=beizi-1"
  echo "mode=native-capx-agent0"
  echo "config=${CONFIG_PATH}"
  echo "[safe] No robot motion requested."
  echo "[run] Set BEIZI_1_EXECUTE_ROBOT=1 to execute."
  exit 0
fi

[[ -r "${CONFIG_PATH}" ]] || {
  echo "[error] missing Agent0 config: ${CONFIG_PATH}" >&2
  exit 1
}
[[ -r "${API_KEY_FILE}" ]] || {
  echo "[error] missing API key file: ${API_KEY_FILE}" >&2
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
  echo "[error] robot stack is offline; beizi-1 will not restart it" >&2
  exit 1
fi
if pgrep -af homogeneous_teleoperation | rg -v "pgrep -af" >/dev/null; then
  echo "[error] teleoperation process is active" >&2
  exit 1
fi

API_KEY="$(tr -d '\r\n' <"${API_KEY_FILE}")"
exec "${PYTHON}" -m capx.envs.launch \
  --config-path "${CONFIG_PATH}" \
  --model "${MODEL}" \
  --server-url "${SERVER_URL}" \
  --api-key "${API_KEY}" \
  --use-oracle-code False \
  --total-trials 1 \
  --num-workers 1
