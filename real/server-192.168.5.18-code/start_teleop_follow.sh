#!/usr/bin/env bash
set -eo pipefail

R1_IP="${1:-10.42.0.202}"
MAC_R="98:B6:E9:BA:01:70"
MAC_L="98:B6:E9:E3:28:7D"

echo "[Teleop] setup env"
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=~/galaxea/find_server/super_client_configuration_file.xml

set +u
source /opt/ros/humble/setup.bash
source ~/galaxea/install/local_setup.bash
set -u

echo "[Teleop] patch fastdds address -> ${R1_IP}"
sed -i "s#<address>[^<]*</address>#<address>${R1_IP}</address>#" ~/galaxea/find_server/super_client_configuration_file.xml
grep -n address ~/galaxea/find_server/super_client_configuration_file.xml

echo "[Teleop] bluetooth connect joy-cons"
bluetoothctl power on >/dev/null || true
bluetoothctl agent on >/dev/null || true
bluetoothctl default-agent >/dev/null || true
bluetoothctl pairable on >/dev/null || true
bluetoothctl discoverable on >/dev/null || true
bluetoothctl connect "$MAC_R" >/dev/null || true
bluetoothctl connect "$MAC_L" >/dev/null || true

echo "[Teleop] joycon status:"
bluetoothctl info "$MAC_R" | grep -E "Name|Connected|Paired" || true
bluetoothctl info "$MAC_L" | grep -E "Name|Connected|Paired" || true

echo "[Teleop] stop old sessions"
pkill -f joy_evdev_launch.py || true
pkill -f joy_evdev_exe_node || true
pkill -f r1litet_gello_teleop_launch.py || true
pkill -f gello_teleop.py || true

echo "[Teleop] start joy publisher in tmux"
tmux kill-session -t joy_pub 2>/dev/null || true
tmux new -d -s joy_pub 'bash -lc "
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=~/galaxea/find_server/super_client_configuration_file.xml
source /opt/ros/humble/setup.bash
source ~/galaxea/install/local_setup.bash
ros2 launch joycon_evdev_publisher joy_evdev_launch.py
"'

echo "[Teleop] start gello teleop in tmux"
tmux kill-session -t gello_teleop 2>/dev/null || true
tmux new -d -s gello_teleop 'bash -lc "
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=~/galaxea/find_server/super_client_configuration_file.xml
source /opt/ros/humble/setup.bash
source ~/galaxea/install/local_setup.bash
ros2 launch mobiman r1litet_gello_teleop_launch.py
"'

echo "[Teleop] tmux sessions:"
tmux ls
echo "[Teleop] done"
