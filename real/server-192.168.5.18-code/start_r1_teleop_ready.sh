#!/usr/bin/env bash
set -eo pipefail

echo "[R1] setup env"
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/galaxea/find_server/super_client_configuration_file.xml

# 避免 setup.bash 在 nounset 模式报错
set +u
source /opt/ros/humble/setup.bash
source ~/galaxea/install/local_setup.bash
set -u

echo "[R1] current eth1 ip:"
R1_IP=$(ip -4 a show eth1 | awk '/inet /{print $2}' | cut -d/ -f1 | head -n1)
echo "R1_IP=${R1_IP:-<none>}"

echo "[R1] fastdds file:"
echo "$FASTRTPS_DEFAULT_PROFILES_FILE"
grep -n address "$FASTRTPS_DEFAULT_PROFILES_FILE" || true

echo "[R1] restart startup sessions"
cd ~/galaxea/install/startup_config/share/startup_config/script/
./robot_startup.sh kill || true
sleep 2
./robot_startup.sh boot ../sessions.d/ATCStandard/R1LITEBody.d

echo "[R1] done"
