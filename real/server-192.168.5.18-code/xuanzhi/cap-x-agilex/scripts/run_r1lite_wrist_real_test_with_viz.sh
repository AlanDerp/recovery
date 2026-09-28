#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
cd "${REPO_ROOT}"
mkdir -p logs outputs/r1lite

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/r1lite_wrist_real_test_$(date +%Y%m%d_%H%M)}"
CONFIG_PATH="${CONFIG_PATH:-env_configs/r1lite/r1lite_tennis_ball_in_box.yaml}"
RUN_GRASPNET_DRY_RUN="${RUN_GRASPNET_DRY_RUN:-0}"
SCENE_PORT="${SCENE_PORT:-8091}"
WEB_PORT="${WEB_PORT:-8766}"
PYTHON="${PYTHON:-${REPO_ROOT}/.capx_r1lite_venv/bin/python}"

export CONFIG_PATH OUTPUT_DIR RUN_GRASPNET_DRY_RUN PYTHON REPO_ROOT

# Source pipeline helpers without executing the launch body.
eval "$(sed -n '1,518p' "${REPO_ROOT}/scripts/run_r1lite_tennis_ball_in_box_pipeline.sh")"
source_ros_env

mkdir -p "${OUTPUT_DIR}"
TRAJECTORY_JSON="${OUTPUT_DIR}/measured_right_wrist_trajectory.json"
SCENE_JSON="${OUTPUT_DIR}/initial_scene.json"
HTML_PATH="${OUTPUT_DIR}/wrist_real_test_scene_and_trajectory.html"

cleanup() {
  if [[ -n "${RECORDER_PID:-}" ]]; then
    kill "${RECORDER_PID}" 2>/dev/null || true
    wait "${RECORDER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SCENE_PID:-}" ]]; then
    kill "${SCENE_PID}" 2>/dev/null || true
    wait "${SCENE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[setup] output=${OUTPUT_DIR}"
ensure_graspnet_tool_calibration
ensure_ros_stack
ensure_capx_isolated_moca
ensure_right_arm_mit_mode
ensure_isolated_relaxed_ik
ensure_head_plane_calibration
ensure_head_stereo_depth
ensure_sam3_server
ensure_graspnet_server
ensure_calibrated_wrist_tf
ensure_wrist_rgbd_tf

echo "[viz] starting scene collector on :${SCENE_PORT}"
"${PYTHON}" scripts/r1lite_scene_web.py \
  --config "${CONFIG_PATH}" \
  --port "${SCENE_PORT}" \
  --trajectory-json "${TRAJECTORY_JSON}" \
  > "${OUTPUT_DIR}/scene_web.log" 2>&1 &
SCENE_PID=$!

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${SCENE_PORT}/api/scene" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "[viz] starting wrist trajectory recorder"
"${PYTHON}" scripts/r1lite_record_wrist_trajectory.py \
  --output "${TRAJECTORY_JSON}" \
  > "${OUTPUT_DIR}/wrist_recorder.log" 2>&1 &
RECORDER_PID=$!
sleep 2

echo "[viz] capturing initial scene snapshot"
curl -fsS "http://127.0.0.1:${SCENE_PORT}/api/scene" -o "${SCENE_JSON}"
"${PYTHON}" - <<PY
import json
from pathlib import Path
scene = json.loads(Path("${SCENE_JSON}").read_text())
print("scene_ok=", scene.get("ok"), "status=", scene.get("status"))
print("ball=", scene.get("objects", {}).get("tennis_ball", {}).get("position"))
print("box=", scene.get("objects", {}).get("cardboard_box", {}).get("position"))
print("table_z=", scene.get("table_plane_z_m"))
PY

echo "[run] wrist staged real execution"
set +e
"${PYTHON}" scripts/run_r1lite_wrist_staged_real.py \
  --config-path "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  | tee "${OUTPUT_DIR}/execution.log"
RUN_RC=${PIPESTATUS[0]}
set -e

sleep 3
kill "${RECORDER_PID}" 2>/dev/null || true
wait "${RECORDER_PID}" 2>/dev/null || true
RECORDER_PID=""

echo "[viz] rendering standalone 3D HTML"
"${PYTHON}" scripts/render_r1lite_scene_trajectory_html.py \
  --scene-json "${SCENE_JSON}" \
  --trajectory-json "${TRAJECTORY_JSON}" \
  --output-html "${HTML_PATH}"

if command -v python3 >/dev/null 2>&1; then
  pkill -f "python3 -m http.server ${WEB_PORT}" 2>/dev/null || true
  (cd "${OUTPUT_DIR}" && python3 -m http.server "${WEB_PORT}" > http.log 2>&1 &)
fi

echo "[done] html=${HTML_PATH}"
echo "[done] local_url=http://127.0.0.1:${WEB_PORT}/wrist_real_test_scene_and_trajectory.html"
exit "${RUN_RC}"
