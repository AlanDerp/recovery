#!/usr/bin/env bash
# One-click R1 Lite tennis-ball-in-box pipeline using Cap-X meta-loop (multi-turn harness).
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"
mkdir -p logs outputs/r1lite

CONFIG_PATH="${CONFIG_PATH:-env_configs/r1lite/r1lite_tennis_ball_in_box_head_only_live.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/r1lite_tennis_ball_in_box}"
USE_ORACLE_CODE="${USE_ORACLE_CODE:-0}"
MODEL="${MODEL:-gpt-5.5}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
API_KEY_FILE="${API_KEY_FILE:-.usezyla_key}"
LLM_BASE_URL="${LLM_BASE_URL:-https://api.usezyla.com/v1/}"
PIPELINE_LOG="${PIPELINE_LOG:-logs/r1lite_tennis_ball_in_box_pipeline.log}"
GRASPNET_PYTHON="${GRASPNET_PYTHON:-}"
RUN_GRASPNET_DRY_RUN="${RUN_GRASPNET_DRY_RUN:-1}"
GRASPNET_FORCE_RESTART="${GRASPNET_FORCE_RESTART:-1}"
ALLOW_ROBOT_STACK_RESTART="${ALLOW_ROBOT_STACK_RESTART:-0}"
WRIST_TF_CALIBRATION="${WRIST_TF_CALIBRATION:-outputs/r1lite/right_touch_hand_eye_calibration/T_ee_camera.yaml}"
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-checkpoints/sam3/sam3.pt}"

source_ros_env() {
  set +u
  source /opt/ros/humble/setup.bash
  if [[ -f /home/nvidia/galaxea/install/local_setup.bash ]]; then
    source /home/nvidia/galaxea/install/local_setup.bash
  fi
  set -u
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
  export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
  export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
  # Cap-X first, then keep ROS paths from setup.bash (required for rclpy).
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
}

resolve_python() {
  if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
    printf '%s\n' "${PYTHON}"
    return 0
  fi
  for candidate in \
    "${REPO_ROOT}/.capx_r1lite_venv/bin/python" \
    "${REPO_ROOT}/.venv/bin/python" \
    /usr/bin/python3; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "[error] python not found" >&2
  return 1
}

PYTHON="$(resolve_python)"
export PYTHON
source_ros_env

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local name="$3"
  local deadline=$((SECONDS + ${4:-120}))
  while (( SECONDS < deadline )); do
    if timeout 1 bash -c ":</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
      echo "[ok] ${name} is listening on ${host}:${port}"
      return 0
    fi
    sleep 1
  done
  echo "[error] timed out waiting for ${name} on ${host}:${port}" >&2
  return 1
}

ros_topic_has_message() {
  local topic="$1"
  local timeout_s="${2:-8}"
  source_ros_env
  timeout "${timeout_s}" ros2 topic echo --once "${topic}" >/dev/null 2>&1
}

ensure_graspnet_tool_calibration() {
  if ! "${PYTHON}" - "${CONFIG_PATH}" <<'PY'
import sys
import numpy as np
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
low_level = cfg["env"]["cfg"]["low_level"]
matrix = low_level.get("graspnet_ee_from_grasp")
if matrix is None:
    raise SystemExit("graspnet_ee_from_grasp is null")
t = np.asarray(matrix, dtype=float)
if t.shape != (4, 4) or not np.isfinite(t).all():
    raise SystemExit("graspnet_ee_from_grasp must be a finite 4x4 matrix")
if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6):
    raise SystemExit("graspnet_ee_from_grasp has an invalid homogeneous row")
r = t[:3, :3]
if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(
    np.linalg.det(r), 1.0, atol=1e-5
):
    raise SystemExit("graspnet_ee_from_grasp rotation is not rigid")
observation_joints = low_level.get("graspnet_observation_joints")
if observation_joints is None:
    raise SystemExit("graspnet_observation_joints is null")
q = np.asarray(observation_joints, dtype=float)
if q.shape != (6,) or not np.isfinite(q).all():
    raise SystemExit("graspnet_observation_joints must be a finite 6-vector")
observation_feedback = low_level.get("graspnet_observation_feedback_joints")
if observation_feedback is None:
    raise SystemExit("graspnet_observation_feedback_joints is null")
q_feedback = np.asarray(observation_feedback, dtype=float)
if q_feedback.shape != (6,) or not np.isfinite(q_feedback).all():
    raise SystemExit(
        "graspnet_observation_feedback_joints must be a finite 6-vector"
    )
PY
  then
    echo "[error] missing or invalid GraspNet tool/observation calibration in ${CONFIG_PATH}" >&2
    echo "Refusing to assume identity or invent a wrist-view pose." >&2
    return 1
  fi
  echo "[ok] calibrated GraspNet grasp-to-EE transform is configured"
}

ensure_ros_stack() {
  if ros_topic_has_message /hdas/feedback_arm_right 8; then
    echo "[skip] robot stack already publishing /hdas/feedback_arm_right"
    return 0
  fi
  echo "[warn] robot stack not ready; waiting 10s before continuing"
  sleep 10
  if ros_topic_has_message /hdas/feedback_arm_right 8; then
    echo "[ok] robot stack feedback became available"
    return 0
  fi
  if [[ "${ALLOW_ROBOT_STACK_RESTART}" != "1" ]]; then
    echo "[error] /hdas/feedback_arm_right is unavailable; refusing to restart teleop/mobiman automatically" >&2
    echo "Set ALLOW_ROBOT_STACK_RESTART=1 only when intentionally restarting the full vendor stack." >&2
    return 1
  fi
  echo "[warn] starting ~/start_r1_teleop_ready.sh"
  bash ~/start_r1_teleop_ready.sh
  sleep 12
  ros_topic_has_message /hdas/feedback_arm_right 25
}

ensure_vendor_moca_input() {
  source_ros_env
  local vendor_topic="/motion_target/target_joint_state_arm_right"
  local topic_info
  tmux kill-session -t r1lite_teleop 2>/dev/null || true
  pkill -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch teleoperation_ros2 r1litet_homogeneous_teleoperation_launch.py' 2>/dev/null || true
  pkill -f '^/home/nvidia/galaxea/install/teleoperation_ros2/lib/teleoperation_ros2/homogeneous_controller_node' 2>/dev/null || true
  local deadline=$((SECONDS + 10))
  while ((SECONDS < deadline)); do
    topic_info="$(ros2 topic info -v "${vendor_topic}" 2>/dev/null || true)"
    if ! grep -q "Node name: homogeneous_teleoperation" <<<"${topic_info}"; then
      break
    fi
    sleep 1
  done

  topic_info="$(ros2 topic info -v "${vendor_topic}" 2>/dev/null || true)"
  if ! grep -q "Node name: moca_adapter" <<<"${topic_info}"; then
    echo "[error] vendor MOCA is not subscribed to ${vendor_topic}" >&2
    echo "[error] refusing to restart MOCA without a latched p_des seed" >&2
    return 1
  fi
  if grep -q "Node name: homogeneous_teleoperation" <<<"${topic_info}"; then
    echo "[error] homogeneous teleoperation still publishes ${vendor_topic}" >&2
    return 1
  fi
  echo "[ok] retained vendor MOCA and removed teleoperation command ownership"
}

ensure_single_right_arm_input_source() {
  source_ros_env
  local arm_topic="/motion_target/target_joint_state_arm_right"
  local gripper_topic="/motion_target/target_position_gripper_right"

  local moca_count
  moca_count="$(
    pgrep -af "/moca_adapter/lib/moca_adapter/moca_main_node" \
      | grep -v "pgrep -af" \
      | wc -l \
      | tr -d ' '
  )"
  if [[ "${moca_count}" != "1" ]]; then
    echo "[error] expected exactly one MOCA process, found ${moca_count}" >&2
    pgrep -af "/moca_adapter/lib/moca_adapter/moca_main_node" >&2 || true
    return 1
  fi

  local arm_info gripper_info
  arm_info="$(ros2 topic info -v "${arm_topic}" 2>/dev/null || true)"
  gripper_info="$(ros2 topic info -v "${gripper_topic}" 2>/dev/null || true)"
  if ! grep -q "Node name: moca_adapter" <<<"${arm_info}"; then
    echo "[error] MOCA is not subscribed to ${arm_topic}" >&2
    return 1
  fi
  if ! grep -q "Node name: moca_adapter" <<<"${gripper_info}"; then
    echo "[error] MOCA is not subscribed to ${gripper_topic}" >&2
    return 1
  fi
  if grep -q "Node name: homogeneous_teleoperation" <<<"${arm_info}"; then
    echo "[error] competing homogeneous right-arm publisher is still present" >&2
    return 1
  fi
  if grep -q "Node name: relaxed_ik_right" <<<"${arm_info}"; then
    echo "[error] RelaxedIK output is not isolated from live arm control" >&2
    return 1
  fi
  echo "[ok] right-arm input contract: Cap-X -> vendor topic -> MOCA"
}

ensure_right_arm_mit_mode() {
  source_ros_env
  local service="/function_frame_right_arm"
  local command
  for command in 2 5 1; do
    local response
    response="$(timeout 8 ros2 service call \
      "${service}" hdas_msg/srv/FunctionFrame "{command: ${command}}")"
    if ! grep -q "success=True" <<<"${response}"; then
      echo "[error] right-arm MIT mode command ${command} failed" >&2
      echo "${response}" >&2
      return 1
    fi
  done
  echo "[ok] right arm reset into MIT mode"
}

ensure_isolated_relaxed_ik() {
  source_ros_env
  local live_topic="/motion_target/target_joint_state_arm_right"
  local isolated_topic="/relaxed_ik/isolated_target_joint_state_arm_right"
  local isolated_pose_topic="/capx/relaxed_ik_target_pose_arm_right"
  local topic_info
  topic_info="$(ros2 topic info -v "${live_topic}" 2>/dev/null || true)"
  if grep -q "Node name: relaxed_ik_right" <<<"${topic_info}"; then
    echo "[restart] replacing live RelaxedIK with isolated solver"
    pkill -f "relaxed_ik.py 0 1 0" 2>/dev/null || true
    sleep 2
  fi
  topic_info="$(ros2 topic info -v "${isolated_topic}" 2>/dev/null || true)"
  if grep -q "Node name: relaxed_ik_right" <<<"${topic_info}"; then
    echo "[skip] isolated right-arm RelaxedIK already running"
    return 0
  fi
  echo "[start] isolated right-arm RelaxedIK solver"
  ros2 run mobiman relaxed_ik.py 0 1 0 --ros-args \
    -p robot_name:=R1_Lite \
    -r __node:=relaxed_ik_right \
    -r /motion_target/target_pose_arm_right:="${isolated_pose_topic}" \
    -r "${live_topic}:=${isolated_topic}" \
    > logs/relaxed_ik_right_isolated.log 2>&1 &
  local deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    topic_info="$(ros2 topic info -v "${isolated_topic}" 2>/dev/null || true)"
    if grep -q "Node name: relaxed_ik_right" <<<"${topic_info}"; then
      echo "[ok] RelaxedIK output is isolated from live joint control"
      return 0
    fi
    sleep 1
  done
  echo "[error] isolated RelaxedIK did not become ready" >&2
  return 1
}

ensure_head_plane_calibration() {
  local calib="${REPO_ROOT}/outputs/r1lite/head_plane_calibration.json"
  if [[ -f "${calib}" ]]; then
    echo "[ok] head plane calibration found: ${calib}"
    return 0
  fi
  echo "[error] missing ${calib}" >&2
  echo "Run head calibration first:" >&2
  echo "  cd ${REPO_ROOT} && bash scripts/run_r1lite_head_calibration.sh all" >&2
  return 1
}

ensure_head_stereo_depth() {
  local rect_topic="/capx/head_stereo/left/image_rect"
  local depth_topic="/capx/head_stereo/aligned_depth_to_left/image_raw"
  local sync_ms="${HEAD_STEREO_MAX_SYNC_MS:-100}"
  source_ros_env
  if ros_topic_has_message "${rect_topic}" 8; then
    echo "[skip] head stereo rectified RGB already publishing on ${rect_topic}"
    return 0
  fi
  echo "[start] head stereo depth publisher (max_sync_ms=${sync_ms})"
  pkill -f "scripts/r1lite_head_stereo_depth.py" 2>/dev/null || true
  sleep 1
  "${PYTHON}" scripts/r1lite_head_stereo_depth.py \
    --max-sync-ms "${sync_ms}" \
    > logs/r1lite_head_stereo_depth.log 2>&1 &
  local deadline=$((SECONDS + 120))
  while (( SECONDS < deadline )); do
    if ros_topic_has_message "${rect_topic}" 8; then
      echo "[ok] head stereo rectified RGB is publishing on ${rect_topic}"
      if ros_topic_has_message "${depth_topic}" 8; then
        echo "[ok] head stereo aligned depth is publishing on ${depth_topic}"
      fi
      return 0
    fi
    if ! pgrep -f "scripts/r1lite_head_stereo_depth.py" >/dev/null; then
      echo "[error] head stereo process exited early; see logs/r1lite_head_stereo_depth.log" >&2
      tail -n 20 logs/r1lite_head_stereo_depth.log >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "[error] timed out waiting for ${rect_topic}" >&2
  echo "Check head camera topics and logs/r1lite_head_stereo_depth.log" >&2
  tail -n 20 logs/r1lite_head_stereo_depth.log >&2 || true
  return 1
}

ensure_sam3_server() {
  if timeout 1 bash -c ":</dev/tcp/127.0.0.1/8114" >/dev/null 2>&1; then
    echo "[skip] SAM3 already available on 8114"
    return 0
  fi
  if [[ ! -f "${SAM3_CHECKPOINT_PATH}" ]]; then
    echo "[error] missing mandatory SAM3 checkpoint: ${SAM3_CHECKPOINT_PATH}" >&2
    return 1
  fi
  if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
  then
    echo "[error] SAM3 service is unavailable and host torch is not installed" >&2
    echo "Start the persistent capx-sam3 container before running this pipeline." >&2
    return 1
  fi
  echo "[start] SAM3 server on 8114"
  "${PYTHON}" -m capx.serving.launch_sam3_server \
    --device cuda \
    --port 8114 \
    --host 127.0.0.1 \
    --checkpoint-path "${SAM3_CHECKPOINT_PATH}" \
    --no-load-from-hf \
    > logs/sam3_r1lite_tennis.log 2>&1 &
  wait_for_tcp 127.0.0.1 8114 SAM3 300
}

ensure_graspnet_server() {
  local service_url
  local endpoint
  local host
  local port
  service_url="$("${PYTHON}" - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg["env"]["cfg"]["low_level"]["graspnet_service_url"])
PY
  )"
  endpoint="${service_url#http://}"
  endpoint="${endpoint#https://}"
  endpoint="${endpoint%%/*}"
  host="${endpoint%%:*}"
  port="${endpoint##*:}"
  if [[ "${GRASPNET_FORCE_RESTART}" == "1" && ( "${host}" == "127.0.0.1" || "${host}" == "localhost" ) ]]; then
    if timeout 1 bash -c ":</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
      echo "[restart] recycling local Contact-GraspNet on ${host}:${port}"
      pkill -f "launch_contact_graspnet_server.*--port ${port}" >/dev/null 2>&1 || true
      pkill -f "launch_contact_graspnet_server --port ${port}" >/dev/null 2>&1 || true
      sleep 3
    fi
  elif timeout 1 bash -c ":</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
    echo "[skip] Contact-GraspNet already available on ${host}:${port}"
    return 0
  fi
  if [[ "${host}" != "127.0.0.1" && "${host}" != "localhost" ]]; then
    echo "[error] configured remote Contact-GraspNet is unavailable on ${host}:${port}" >&2
    return 1
  fi
  local graspnet_python="${GRASPNET_PYTHON:-${PYTHON}}"
  if ! "${graspnet_python}" -c "import torch, tyro, viser" >/dev/null 2>&1; then
    echo "[error] ${graspnet_python} lacks Contact-GraspNet runtime dependencies" >&2
    echo "Set GRASPNET_PYTHON to a CUDA Python environment with torch, tyro, and viser." >&2
    return 1
  fi
  echo "[start] Contact-GraspNet server on ${port}"
  "${graspnet_python}" -m capx.serving.launch_contact_graspnet_server \
    --device cuda \
    --port "${port}" \
    --host "${host}" \
    > logs/graspnet_r1lite_tennis.log 2>&1 &
  wait_for_tcp "${host}" "${port}" Contact-GraspNet 300
}

ensure_calibrated_wrist_tf() {
  source_ros_env
  local camera_frame="camera_wrist_right_color_optical_frame"
  local tf_output
  tf_output="$(timeout 4 ros2 run tf2_ros tf2_echo base_link "${camera_frame}" 2>&1 || true)"
  if grep -q "Translation:" <<<"${tf_output}"; then
    echo "[skip] base_link <- ${camera_frame} TF already available"
    return 0
  fi
  if [[ ! -f "${WRIST_TF_CALIBRATION}" ]]; then
    echo "[error] missing wrist hand-eye calibration: ${WRIST_TF_CALIBRATION}" >&2
    return 1
  fi
  echo "[start] calibrated wrist-camera dynamic TF"
  "${PYTHON}" scripts/r1lite_wrist_calibrated_tf.py \
    --calibration "${WRIST_TF_CALIBRATION}" \
    > logs/r1lite_wrist_calibrated_tf.log 2>&1 &
  local deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    tf_output="$(
      timeout 3 ros2 run tf2_ros tf2_echo base_link "${camera_frame}" 2>&1 || true
    )"
    if grep -q "Translation:" <<<"${tf_output}"; then
      echo "[ok] calibrated base_link <- ${camera_frame} TF is available"
      return 0
    fi
    sleep 1
  done
  echo "[error] calibrated wrist-camera TF did not become available" >&2
  return 1
}

ensure_wrist_rgbd_tf() {
  if [[ "${SKIP_WRIST_RGBD_PREFLIGHT:-0}" == "1" ]]; then
    echo "[warn] skipping mandatory wrist RGB-D preflight (SKIP_WRIST_RGBD_PREFLIGHT=1)"
    return 0
  fi
  source_ros_env
  local topics
  topics="$("${PYTHON}" - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as stream:
    config = yaml.safe_load(stream)
low_level = config["env"]["cfg"]["low_level"]
print(low_level["wrist_rgb_topic"])
print(low_level["wrist_depth_topic"])
print(low_level["wrist_camera_info_topic"])
PY
  )"
  local rgb_topic depth_topic info_topic
  rgb_topic="$(sed -n '1p' <<<"${topics}")"
  depth_topic="$(sed -n '2p' <<<"${topics}")"
  info_topic="$(sed -n '3p' <<<"${topics}")"
  local topic
  for topic in "${rgb_topic}" "${depth_topic}" "${info_topic}"; do
    local received=0
    local attempt
    for attempt in 1 2 3; do
      if ros_topic_has_message "${topic}" 12; then
        received=1
        break
      fi
      sleep 1
    done
    if [[ "${received}" != "1" ]]; then
      echo "[error] mandatory wrist RGB-D topic has no message: ${topic}" >&2
      return 1
    fi
  done

  local camera_frame
  for attempt in 1 2 3; do
    camera_frame="$(
      timeout 12 ros2 topic echo --once "${info_topic}" \
        --field header.frame_id 2>/dev/null \
        | awk 'NF && $0 != "---" {print; exit}'
    )"
    [[ -n "${camera_frame}" ]] && break
    sleep 1
  done
  if [[ -z "${camera_frame}" ]]; then
    echo "[error] wrist CameraInfo has no frame_id; GraspNet pose cannot be transformed" >&2
    return 1
  fi

  local tf_output
  tf_output="$(timeout 8 ros2 run tf2_ros tf2_echo base_link "${camera_frame}" 2>&1 || true)"
  if ! grep -q "Translation:" <<<"${tf_output}"; then
    echo "[error] missing ROS TF: base_link <- ${camera_frame}" >&2
    return 1
  fi
  echo "[ok] wrist RGB-D and base_link <- ${camera_frame} TF are available"
}

config_uses_wrist_camera() {
  "${PYTHON}" - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as stream:
    config = yaml.safe_load(stream)
enabled = config["env"]["cfg"]["low_level"].get("wrist_camera_enabled", True)
raise SystemExit(0 if enabled else 1)
PY
}

run_graspnet_dry_run() {
  if [[ "${SKIP_GRASPNET_DRY_RUN:-0}" == "1" ]]; then
    echo "[warn] skipping mandatory GraspNet dry-run (SKIP_GRASPNET_DRY_RUN=1)"
    return 0
  fi
  if [[ "${RUN_GRASPNET_DRY_RUN}" != "1" ]]; then
    echo "[error] mandatory GraspNet dry-run cannot be disabled for this pipeline" >&2
    return 1
  fi
  echo "[check] planning mandatory Contact-GraspNet grasp without arm motion"
  local -a cmd=(
    "${PYTHON}"
    scripts/smoke_test_r1lite_graspnet.py
    --config "${CONFIG_PATH}"
  )
  if ! timeout 1 bash -c ":</dev/tcp/127.0.0.1/8114" >/dev/null 2>&1; then
    echo "[error] mandatory SAM3 service is unavailable on 127.0.0.1:8114" >&2
    return 1
  fi
  "${cmd[@]}"
  echo "[ok] no-motion Contact-GraspNet planning passed"
}

ensure_llm_proxy() {
  if [[ "${USE_ORACLE_CODE}" == "1" ]]; then
    echo "[skip] oracle mode does not need LLM proxy"
    return 0
  fi
  if [[ "${SKIP_LLM_PROXY:-0}" == "1" ]]; then
    echo "[skip] LLM proxy disabled by SKIP_LLM_PROXY=1"
    return 0
  fi
  if timeout 1 bash -c ":</dev/tcp/127.0.0.1/8110" >/dev/null 2>&1; then
    echo "[skip] LLM proxy already available on 8110"
    return 0
  fi
  if [[ ! -f "${API_KEY_FILE}" ]]; then
    echo "[error] missing API key file: ${API_KEY_FILE}" >&2
    exit 1
  fi
  echo "[start] LLM proxy on 8110 (base_url=${LLM_BASE_URL})"
  "${PYTHON}" -m capx.serving.openrouter_server \
    --key-file "${API_KEY_FILE}" \
    --base-url "${LLM_BASE_URL}" \
    --port 8110 \
    --host 127.0.0.1 \
    > logs/openrouter_8110_r1lite_tennis.log 2>&1 &
  wait_for_tcp 127.0.0.1 8110 LLM_proxy 120
}

run_launch() {
  source_ros_env
  mkdir -p "${OUTPUT_DIR}"
  local -a cmd=(
    "${PYTHON}" capx/envs/launch.py
    --config-path "${CONFIG_PATH}"
    --output-dir "${OUTPUT_DIR}"
  )
  if [[ "${USE_ORACLE_CODE}" == "1" ]]; then
    cmd+=(--use-oracle-code True)
  else
    local use_wrist_camera=False
    if config_uses_wrist_camera; then
      use_wrist_camera=True
    fi
    cmd+=(
      --model "${MODEL}"
      --server-url "${SERVER_URL}"
      --use-visual-feedback True
      --use-wrist-camera "${use_wrist_camera}"
    )
    if [[ -f "${API_KEY_FILE}" ]]; then
      cmd+=(--api-key "$(tr -d '\r\n' < "${API_KEY_FILE}")")
    fi
  fi
  echo "[run] ${PYTHON} capx/envs/launch.py --config-path ${CONFIG_PATH} --output-dir ${OUTPUT_DIR}"
  if [[ "${USE_ORACLE_CODE}" == "1" ]]; then
    echo "[run] code_source=oracle"
  else
    echo "[run] code_source=llm model=${MODEL} server_url=${SERVER_URL} api_key=[redacted]"
  fi
  "${cmd[@]}"
}

mkdir -p logs
: > "${PIPELINE_LOG}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1
echo "[pipeline] log: ${PIPELINE_LOG}"
echo "[pipeline] config: ${CONFIG_PATH}"
echo "[pipeline] output: ${OUTPUT_DIR}"

ensure_graspnet_tool_calibration
ensure_ros_stack
ensure_vendor_moca_input
ensure_isolated_relaxed_ik
ensure_single_right_arm_input_source
if [[ "${SKIP_RIGHT_ARM_MIT_MODE_RESET:-0}" == "1" ]]; then
  echo "[skip] right-arm MIT mode reset deferred until real execution"
else
  ensure_right_arm_mit_mode
fi
ensure_head_plane_calibration
ensure_head_stereo_depth
ensure_sam3_server
ensure_graspnet_server
if config_uses_wrist_camera; then
  ensure_calibrated_wrist_tf
  ensure_wrist_rgbd_tf
else
  echo "[skip] wrist camera disabled by config"
fi
run_graspnet_dry_run
ensure_llm_proxy
if [[ "${R1LITE_PIPELINE_SETUP_ONLY:-0}" == "1" ]]; then
  echo "[ready] R1 Lite perception, control, and LLM services passed setup-only checks"
  exit 0
fi
run_launch
