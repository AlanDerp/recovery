#!/usr/bin/env bash
# Launch UGREEN Camera 2K as a ROS rgb publisher.
# Typical mount: side view, lens ~20 cm above table (tennis-ball setup).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

DEVICE="${UGREEN_DEVICE:-/dev/video2}"
FPS="${UGREEN_FPS:-30}"
IMAGE_TOPIC="${UGREEN_IMAGE_TOPIC:-/camera_ugreen/color/image_raw}"

source /opt/ros/noetic/setup.bash

if timeout 2 rostopic echo "${IMAGE_TOPIC}" -n 1 >/dev/null 2>&1; then
  echo "[skip] ${IMAGE_TOPIC} already publishing"
  exit 0
fi

if ! rostopic list >/dev/null 2>&1; then
  echo "[error] roscore is not running" >&2
  exit 1
fi

echo "[start] UGREEN camera on ${DEVICE} -> ${IMAGE_TOPIC}"
python3 scripts/ugreen_camera_node.py \
  _device:="${DEVICE}" \
  _fps:="${FPS}" \
  _image_topic:="${IMAGE_TOPIC}" \
  > logs/camera_ugreen.log 2>&1 &
camera_pid=$!

for _ in $(seq 1 30); do
  if timeout 1 rostopic echo "${IMAGE_TOPIC}" -n 1 >/dev/null 2>&1; then
    echo "[ok] ${IMAGE_TOPIC} is publishing"
    exit 0
  fi
  if ! kill -0 "${camera_pid}" >/dev/null 2>&1; then
    echo "[error] UGREEN camera node exited unexpectedly (pid=${camera_pid})" >&2
    tail -n 20 logs/camera_ugreen.log >&2 || true
    exit 1
  fi
  sleep 1
done

echo "[error] timed out waiting for ${IMAGE_TOPIC}" >&2
tail -n 20 logs/camera_ugreen.log >&2 || true
exit 1
