#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"

STARTUP_ROOT="/home/nvidia/galaxea/install/startup_config/share/startup_config"
SOURCE_SESSIONS="${STARTUP_ROOT}/sessions.d/ATCStandard/R1LITEBody.d"
SAFE_SESSION_NAME="capx_no_teleop_sessions.d"
SAFE_SESSIONS="${STARTUP_ROOT}/script/${SAFE_SESSION_NAME}"
SAFE_BIN="/tmp/capx-r1lite-safe-bin"

rm -rf "${SAFE_SESSIONS}"
mkdir -p "${SAFE_SESSIONS}"
for session in hdas mobiman ros2_discovery system tools; do
  cp "${SOURCE_SESSIONS}/${session}.yaml" "${SAFE_SESSIONS}/${session}.yaml"
done
rm -rf "${SAFE_BIN}"
mkdir -p "${SAFE_BIN}"
cat >"${SAFE_BIN}/sudo" <<'EOF'
#!/usr/bin/env bash
has_stdin_password=0
for arg in "$@"; do
  [[ "${arg}" == "-S" ]] && has_stdin_password=1
done
if [[ "${has_stdin_password}" == "1" ]]; then
  exec /usr/bin/sudo "$@"
fi
exec /usr/bin/sudo -A "$@"
EOF
chmod 700 "${SAFE_BIN}/sudo"
export PATH="${SAFE_BIN}:${PATH}"

cd "${STARTUP_ROOT}/script"
./robot_startup.sh kill || true
sleep 2
./robot_startup.sh boot "${SAFE_SESSION_NAME}"

deadline=$((SECONDS + 90))
while ((SECONDS < deadline)); do
  if timeout 2 ros2 topic echo --once \
    /hdas/feedback_arm_right sensor_msgs/msg/JointState \
    >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! timeout 2 ros2 topic echo --once \
  /hdas/feedback_arm_right sensor_msgs/msg/JointState \
  >/dev/null 2>&1; then
  echo "[error] right-arm feedback did not start" >&2
  exit 1
fi

deadline=$((SECONDS + 90))
topic_info=""
while ((SECONDS < deadline)); do
  topic_info="$(ros2 topic info -v \
    /motion_target/target_joint_state_arm_right 2>/dev/null || true)"
  if grep -q "Node name: moca_adapter" <<<"${topic_info}" \
    && timeout 2 ros2 topic echo --once \
      /motion_control/control_arm_right hdas_msg/msg/MotorControl \
      >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! grep -q "Node name: moca_adapter" <<<"${topic_info}"; then
  echo "[error] vendor MOCA did not become ready" >&2
  exit 1
fi
if grep -q "Node name: homogeneous_teleoperation" <<<"${topic_info}"; then
  echo "[error] unsafe teleoperation publisher appeared" >&2
  exit 1
fi
if pgrep -af "homogeneous_teleoperation|homogeneous_controller_node" \
  >/dev/null; then
  echo "[error] unsafe teleoperation process appeared" >&2
  exit 1
fi

echo "[ok] vendor stack started without homogeneous teleoperation"
