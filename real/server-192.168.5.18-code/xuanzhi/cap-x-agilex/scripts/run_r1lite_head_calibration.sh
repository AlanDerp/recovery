#!/usr/bin/env bash
# R1 Lite head-camera tabletop calibration workflow.
#
# Usage:
#   bash scripts/run_r1lite_head_calibration.sh check
#   bash scripts/run_r1lite_head_calibration.sh capture
#   bash scripts/run_r1lite_head_calibration.sh collect   # needs arm movement
#   bash scripts/run_r1lite_head_calibration.sh solve
#   bash scripts/run_r1lite_head_calibration.sh verify
#   bash scripts/run_r1lite_head_calibration.sh all         # capture -> collect -> solve -> verify
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

set +u
source /opt/ros/humble/setup.bash
if [[ -f "$HOME/galaxea/install/local_setup.bash" ]]; then
  source "$HOME/galaxea/install/local_setup.bash"
fi
set -u

PY="${ROOT}/.venv/bin/python"
COLLECT="${ROOT}/scripts/r1lite_collect_head_plane_points.py"

step="${1:-help}"

run_check() {
  echo "=== [1/5] ROS topic check ==="
  set +e
  "$PY" "$COLLECT" check
  local rc=$?
  set -e
  return $rc
}

run_capture() {
  echo "=== [2/5] Capture head camera image ==="
  set +e
  "$PY" "$COLLECT" check
  local rc=$?
  set -e
  if [[ "$rc" -eq 1 ]]; then
    echo "No camera topics available. Start robot stack first."
    exit 1
  fi

  if ! "$PY" "$COLLECT" capture \
      --camera left \
      --output outputs/r1lite/latest_head_left_rgb.png; then
    echo "Left head camera unavailable, trying right ..."
    "$PY" "$COLLECT" capture \
      --camera right \
      --output outputs/r1lite/latest_head_left_rgb.png
  fi
  echo "Saved: outputs/r1lite/latest_head_left_rgb.png"
}

run_collect() {
  echo "=== [3/5] Collect calibration points (arm movement required) ==="
  echo ""
  echo "准备工作："
  echo "  - 确认桌面上有平整工作面（如黑色设备箱顶面）"
  echo "  - 右臂已上电，可手动遥操作"
  echo "  - 4 个点全部用右臂，夹爪状态保持一致"
  echo ""
  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" && -S /tmp/.X11-unix/X0 ]]; then
    export DISPLAY=:0
    echo "[collect] DISPLAY 未设置，已自动使用机器人本地屏幕 DISPLAY=:0"
    echo "          弹窗会出现在机器人接的显示器上（不是 SSH 窗口里）"
    echo ""
  fi
  set +e
  "$PY" "$COLLECT" check
  local rc=$?
  set -e
  if [[ "$rc" -eq 3 ]]; then
    echo ""
    echo "缺少右臂末端位姿。请另开一个终端执行："
    echo "  cd ~/galaxea/install/startup_config/share/startup_config/script/boot/modules/mobiman"
    echo "  ./start_r1lite_ee_pose.sh"
    echo "然后验证："
    echo "  ros2 topic echo /motion_control/pose_ee_arm_right --once"
    exit 3
  fi
  if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
  fi
  "$PY" "$COLLECT" collect \
    --image outputs/r1lite/latest_head_left_rgb.png \
    --output outputs/r1lite/head_plane_points.json \
    --camera left
}

run_solve() {
  echo "=== [4/5] Compute homography ==="
  "$PY" "$COLLECT" solve \
    --points-json outputs/r1lite/head_plane_points.json \
    --output outputs/r1lite/head_plane_calibration.json
}

run_verify() {
  echo "=== [5/5] Verify calibration ==="
  "$PY" "$COLLECT" verify \
    --calibration outputs/r1lite/head_plane_calibration.json
}

case "$step" in
  check)   run_check ;;
  capture) run_check; run_capture ;;
  collect) run_collect ;;
  solve)   run_solve ;;
  verify)  run_verify ;;
  all)
    set +e
    "$PY" "$COLLECT" check
    cam_rc=$?
    set -e
    if [[ "$cam_rc" -eq 1 ]]; then
      echo "Robot stack not ready. Run: bash ~/start_r1_teleop_ready.sh"
      exit 1
    fi
    run_capture
    if [[ "$cam_rc" -eq 2 ]]; then
      echo ""
      echo "相机已就绪。请启动完整机器人栈后再继续 collect 步骤："
      echo "  bash ~/start_r1_teleop_ready.sh"
      echo "然后运行:"
      echo "  bash scripts/run_r1lite_head_calibration.sh collect"
      exit 0
    fi
    run_collect
    run_solve
    run_verify
    echo ""
    echo "标定完成: outputs/r1lite/head_plane_calibration.json"
    ;;
  help|*)
    cat <<'EOF'
R1 Lite 头部相机标定流程

步骤:
  1. check    - 检查 ROS 话题（相机、机械臂）
  2. capture  - 拍摄头部左相机图像
  3. collect  - 在图像上点选 4 个角点，并移动机械臂记录位姿  <-- 需要动机械臂
  4. solve    - 计算单应矩阵
  5. verify   - 验证重投影误差
  all         - 依次执行 capture -> collect -> solve -> verify

如果机器人未启动:
  bash ~/start_r1_teleop_ready.sh

示例:
  bash scripts/run_r1lite_head_calibration.sh check
  bash scripts/run_r1lite_head_calibration.sh capture
  bash scripts/run_r1lite_head_calibration.sh collect
  bash scripts/run_r1lite_head_calibration.sh solve
  bash scripts/run_r1lite_head_calibration.sh verify
EOF
    ;;
esac
