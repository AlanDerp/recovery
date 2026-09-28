#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
PYTHON="${REPO_ROOT}/.capx_r1lite_venv/bin/python"
SCRIPT="${REPO_ROOT}/scripts/r1lite_collect_bimanual_head_plane_points.py"
STEREO_SCRIPT="${REPO_ROOT}/scripts/r1lite_calibrate_head_stereo_rectified.py"
STEREO_VALIDATE_SCRIPT="${REPO_ROOT}/scripts/r1lite_validate_head_stereo_table_plane.py"
STEREO_REFINE_SCRIPT="${REPO_ROOT}/scripts/r1lite_refine_head_stereo_table_plane.py"
COMMAND="${1:-}"

cd "${REPO_ROOT}"
set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

case "${COMMAND}" in
  check)
    exec "${PYTHON}" "${SCRIPT}" check
    ;;
  capture)
    exec "${PYTHON}" "${SCRIPT}" capture
    ;;
  collect)
    echo "[bimanual-calibration] 依次点击5个左臂点、5个右臂点、4个双臂重叠点。"
    echo "[bimanual-calibration] 脚本不会发送机械臂运动命令；请手动移动指定机械臂。"
    exec "${PYTHON}" "${SCRIPT}" collect
    ;;
  solve)
    exec "${PYTHON}" "${SCRIPT}" solve
    ;;
  verify)
    exec "${PYTHON}" "${SCRIPT}" verify
    ;;
  board-detect)
    exec "${PYTHON}" "${SCRIPT}" board-detect
    ;;
  board-collect)
    echo "[chessboard] 规格：14x18方格，20mm边长，自动检测13x17内角点。"
    echo "[chessboard] tool_z姿态不强制竖直；脚本会用实时四元数补偿指尖偏移。"
    echo "[chessboard] 左右臂均以完全打开的 finger_link2（局部-Y侧）前端中心轻触内角点。"
    exec "${PYTHON}" "${SCRIPT}" board-collect
    ;;
  board-solve)
    exec "${PYTHON}" "${SCRIPT}" board-solve
    ;;
  board-all)
    "${PYTHON}" "${SCRIPT}" check
    "${PYTHON}" "${SCRIPT}" board-collect
    "${PYTHON}" "${SCRIPT}" board-solve
    "${PYTHON}" "${SCRIPT}" verify
    ;;
  stereo-convert)
    exec "${PYTHON}" "${STEREO_SCRIPT}"
    ;;
  stereo-validate)
    exec "${PYTHON}" "${STEREO_VALIDATE_SCRIPT}"
    ;;
  stereo-refine)
    exec "${PYTHON}" "${STEREO_REFINE_SCRIPT}" \
      --validation-audit \
      "${REPO_ROOT}/outputs/r1lite/head_stereo_table_validation.json"
    ;;
  all)
    "${PYTHON}" "${SCRIPT}" check
    "${PYTHON}" "${SCRIPT}" collect
    "${PYTHON}" "${SCRIPT}" solve
    "${PYTHON}" "${SCRIPT}" verify
    ;;
  *)
    echo "usage: $0 {check|capture|collect|solve|verify|all|board-detect|board-collect|board-solve|board-all|stereo-convert|stereo-validate|stereo-refine}" >&2
    exit 2
    ;;
esac
