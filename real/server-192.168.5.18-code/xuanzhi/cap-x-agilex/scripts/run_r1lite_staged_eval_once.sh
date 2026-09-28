#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"

OUT="${1:-${REPO_ROOT}/outputs/r1lite_wrist_eval_$(date +%Y%m%d_%H%M)}"
CONFIG="${2:-env_configs/r1lite/r1lite_tennis_ball_in_box.yaml}"
PY="${REPO_ROOT}/.capx_r1lite_venv/bin/python"

mkdir -p "${OUT}" logs

set +u
source /opt/ros/humble/setup.bash
if [[ -f /home/nvidia/galaxea/install/local_setup.bash ]]; then
  source /home/nvidia/galaxea/install/local_setup.bash
fi
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Robot stack helpers from the main pipeline (functions only).
export CONFIG_PATH="${CONFIG}"
HELPERS="$(mktemp)"
sed -n '23,238p' "${REPO_ROOT}/scripts/run_r1lite_tennis_ball_in_box_pipeline.sh" > "${HELPERS}"
# shellcheck source=/dev/null
source "${HELPERS}"
rm -f "${HELPERS}"
source_ros_env
ensure_ros_stack || true
ensure_capx_isolated_moca
ensure_right_arm_mit_mode
ensure_isolated_relaxed_ik

if ! timeout 4 ros2 run tf2_ros tf2_echo base_link camera_wrist_right_color_optical_frame 2>&1 | grep -q Translation:; then
  echo "[setup] starting wrist calibrated TF"
  "${PY}" scripts/r1lite_wrist_calibrated_tf.py \
    --calibration outputs/r1lite/right_touch_hand_eye_calibration/T_ee_camera.yaml \
    >> logs/r1lite_wrist_calibrated_tf.log 2>&1 &
  sleep 8
fi

echo "[setup] output=${OUT}"
echo "[setup] config=${CONFIG}"

REC=""
cleanup() {
  if [[ -n "${REC}" ]]; then
    kill "${REC}" 2>/dev/null || true
    wait "${REC}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"${PY}" scripts/r1lite_record_wrist_trajectory.py \
  --output "${OUT}/measured_right_wrist_trajectory.json" \
  >> "${OUT}/wrist_recorder.log" 2>&1 &
REC=$!
sleep 2

echo "[run] staged real eval"
set +e
"${PY}" scripts/run_r1lite_wrist_staged_real.py \
  --config-path "${CONFIG}" \
  --output-dir "${OUT}" \
  2>&1 | tee "${OUT}/execution.log"
RC=${PIPESTATUS[0]}
set -e

echo "[done] rc=${RC} output=${OUT}"
exit "${RC}"
