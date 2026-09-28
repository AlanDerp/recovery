#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
GALAXEA_SETUP="/home/nvidia/galaxea/install/local_setup.bash"
ROBOT_STARTUP="/home/nvidia/galaxea/install/startup_config/share/startup_config/script/robot_startup.sh"
ROBOT_SESSION="../sessions.d/ATCStandard/R1LITEBody.d"
LOG_DIR="${REPO_ROOT}/logs"

export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

set +u
source /opt/ros/humble/setup.bash
source "${GALAXEA_SETUP}"
set -u

mkdir -p "${LOG_DIR}"

controllers_started=0
cleanup_failed_startup() {
  local exit_code=$?
  if ((controllers_started)); then
    tmux kill-session -t moca_capx_isolated 2>/dev/null || true
    tmux kill-session -t r1lite_relaxed_ik_isolated 2>/dev/null || true
    pkill -TERM -f '[m]oca_adapter/lib/moca_adapter/moca_main_node' 2>/dev/null || true
    pkill -TERM -f '[r]elaxed_ik.py 0 1 0' 2>/dev/null || true
    echo "[stop] isolated right-arm controllers after startup failure" >&2
  fi
  tmux kill-session -t r1lite_head_stereo_depth 2>/dev/null || true
  exit "${exit_code}"
}
trap cleanup_failed_startup ERR INT TERM

# A failed high-resolution stereo process can monopolize CPU and starve HDAS feedback.
tmux kill-session -t r1lite_head_stereo_depth 2>/dev/null || true

wait_for_topic() {
  local topic="$1"
  local timeout_s="$2"
  local label="$3"
  local deadline=$((SECONDS + timeout_s))
  while ((SECONDS < deadline)); do
    if ROS2CLI_NO_DAEMON=1 timeout 2 ros2 topic echo "${topic}" --once \
      >/dev/null 2>&1; then
      echo "[ok] ${label}"
      return 0
    fi
    sleep 1
  done
  echo "[error] timed out waiting for ${label}: ${topic}" >&2
  return 1
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local label="$3"
  local timeout_s="$4"
  local deadline=$((SECONDS + timeout_s))
  while ((SECONDS < deadline)); do
    if timeout 1 bash -c ":</dev/tcp/${host}/${port}" 2>/dev/null; then
      echo "[ok] ${label} on ${host}:${port}"
      return 0
    fi
    sleep 1
  done
  echo "[error] timed out waiting for ${label} on ${host}:${port}" >&2
  return 1
}

wait_for_endpoint() {
  local topic="$1"
  local pattern="$2"
  local label="$3"
  local timeout_s="$4"
  local deadline=$((SECONDS + timeout_s))
  while ((SECONDS < deadline)); do
    if ROS2CLI_NO_DAEMON=1 ros2 topic info -v "${topic}" 2>/dev/null \
      | grep -q "${pattern}"; then
      echo "[ok] ${label}"
      return 0
    fi
    sleep 1
  done
  echo "[error] timed out waiting for ${label}" >&2
  return 1
}

if ! ROS2CLI_NO_DAEMON=1 timeout 3 ros2 topic echo \
  /hdas/feedback_arm_right --once >/dev/null 2>&1; then
  echo "[start] R1 Lite body stack in ROS domain ${ROS_DOMAIN_ID}"
  (
    cd "$(dirname "${ROBOT_STARTUP}")"
    "${ROBOT_STARTUP}" boot "${ROBOT_SESSION}"
  )
fi
wait_for_topic /hdas/feedback_arm_right 45 "R1 Lite right-arm feedback"

echo "[prepare] isolated Cap-X right-arm solver"
pkill -TERM -f '[m]oca_adapter/lib/moca_adapter/moca_main_node' 2>/dev/null || true
pkill -TERM -f '[r]elaxed_ik.py 0 1 0' 2>/dev/null || true
tmux kill-session -t moca_capx_isolated 2>/dev/null || true
tmux kill-session -t r1lite_relaxed_ik_isolated 2>/dev/null || true
sleep 2

tmux new-session -d -s r1lite_relaxed_ik_isolated "bash -lc '
  export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
  export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}
  export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}
  source /opt/ros/humble/setup.bash
  source ${GALAXEA_SETUP}
  exec ros2 run mobiman relaxed_ik.py 0 1 0 --ros-args \
    -p robot_name:=R1_Lite \
    -r __node:=relaxed_ik_right \
    -r /motion_target/target_pose_arm_right:=/capx/relaxed_ik_target_pose_arm_right \
    -r /motion_target/target_joint_state_arm_right:=/relaxed_ik/isolated_target_joint_state_arm_right \
    >> ${LOG_DIR}/relaxed_ik_right_isolated.log 2>&1
'"
controllers_started=1

echo "[restart] head stereo depth"
tmux kill-session -t r1lite_head_stereo_depth 2>/dev/null || true
tmux new-session -d -s r1lite_head_stereo_depth "bash -lc '
  export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
  export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}
  export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}
  export PYTHONPATH=${REPO_ROOT}
  source /opt/ros/humble/setup.bash
  source ${GALAXEA_SETUP}
  cd ${REPO_ROOT}
  exec .capx_r1lite_venv/bin/python scripts/r1lite_head_stereo_depth.py \
    >> ${LOG_DIR}/r1lite_head_stereo_depth.log 2>&1
'"
wait_for_topic \
  /capx/head_stereo/aligned_depth_to_left/image_raw 90 "head aligned depth"

if ! timeout 1 bash -c ':</dev/tcp/127.0.0.1/8114' 2>/dev/null; then
  echo "[start] SAM3 container"
  docker start capx-sam3 >/dev/null
fi
wait_for_tcp 127.0.0.1 8114 SAM3 60

if ! timeout 1 bash -c ':</dev/tcp/127.0.0.1/8117' 2>/dev/null; then
  echo "[start] native Contact-GraspNet"
  tmux kill-session -t capx_graspnet_native 2>/dev/null || true
  tmux new-session -d -s capx_graspnet_native "bash -lc '
    export PYTHONPATH=${REPO_ROOT}
    cd ${REPO_ROOT}
    exec .capx_r1lite_venv/bin/python \
      -m capx.serving.launch_contact_graspnet_server \
      --device cuda --port 8117 --host 127.0.0.1 \
      >> ${LOG_DIR}/graspnet_r1lite_tennis.log 2>&1
  '"
fi
wait_for_tcp 127.0.0.1 8117 Contact-GraspNet 120

echo "[set] right arm MIT mode"
for command in 2 5 1; do
  response="$(timeout 8 ros2 service call \
    /function_frame_right_arm hdas_msg/srv/FunctionFrame \
    "{command: ${command}}")"
  if ! grep -q 'success=True' <<<"${response}"; then
    echo "[error] FunctionFrame command ${command} failed" >&2
    echo "${response}" >&2
    exit 1
  fi
  sleep 1
done

echo "[ready] R1 Lite Cap-X services are running; MOCA is stopped until the first valid joint target"
echo "[ready] Start the task with scripts/run_r1lite_capx_full_place.sh"
trap - ERR INT TERM
