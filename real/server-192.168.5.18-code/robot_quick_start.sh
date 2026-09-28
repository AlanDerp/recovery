#!/bin/bash

# 颜色定义
COLOR_RED="\033[31m"
COLOR_GREEN="\033[32m"
COLOR_YELLOW="\033[33m"
COLOR_BLUE="\033[34m"
COLOR_MAGENTA="\033[35m"
COLOR_CYAN="\033[36m"
COLOR_RESET="\033[0m"

# 检查并读取硬件配置文件
HARDWARE_FILE="/opt/galaxea/body/hardware.json"

if [[ ! -f "$HARDWARE_FILE" ]]; then
    echo -e "${COLOR_RED}错误：硬件配置文件不存在！${COLOR_RESET}"
    echo "请确保文件存在于 $HARDWARE_FILE"
    exit 1
fi

# 获取机器人型号
if ! command -v jq &> /dev/null; then
    echo "正在安装jq工具..."
    if [[ -f /etc/debian_version ]]; then
        sudo apt-get update && sudo apt-get install -y jq
    elif [[ -f /etc/redhat-release ]]; then
        sudo yum install -y jq
    else
        echo "请手动安装jq工具"
        exit 1
    fi
fi

ROBOT_MODEL=$(jq -r 'keys[0]' "$HARDWARE_FILE")

if [[ -z "$ROBOT_MODEL" ]] || [[ "$ROBOT_MODEL" == "null" ]]; then
    echo -e "${COLOR_RED}错误：无法读取机器人型号！${COLOR_RESET}"
    exit 1
fi

echo "=================================================="
echo "检测到的机器人型号: $ROBOT_MODEL"
echo "=================================================="

# 路径定义
HOME_DIR="$HOME"
START_SHELL_PATH="$HOME_DIR/galaxea/install/startup_config/share/startup_config/script/robot_startup.sh"
SETUP_BASH="$HOME_DIR/galaxea/install/setup.bash"

# 通用函数
show_menu() {
    echo ""
    echo "请选择要执行的操作："
    echo "=================================================="
    echo "1. 执行开箱检测"
    
    # 根据机器人型号显示不同的菜单项
    if [[ "$ROBOT_MODEL" == "R1-PRO" ]]; then
        echo "2. 执行开箱脚本 (控制躯干蹲起)"
    elif [[ "$ROBOT_MODEL" == "R1-LITE" ]]; then
        echo "2. R1-LITE不支持开箱脚本功能"
    fi
    
    echo "3. 执行DEMO展示"
    echo "4. 配置发现服务"
    echo "5. 杀死当前SDK进程"
    echo "6. 退出"
    echo "=================================================="
    
    # 根据机器人型号显示不同的选项范围
    if [[ "$ROBOT_MODEL" == "R1-PRO" ]]; then
        echo -n "请输入选项 (1-6): "
    elif [[ "$ROBOT_MODEL" == "R1-LITE" ]]; then
        echo -n "请输入选项 (1,3-6): "
    fi
}

kill_sdk() {
    echo -e "${COLOR_YELLOW}正在杀死当前SDK进程...${COLOR_RESET}"
    bash "$START_SHELL_PATH" kill 2>/dev/null
    if [[ $? -eq 0 ]]; then
        echo -e "${COLOR_GREEN}✓ SDK进程已杀死${COLOR_RESET}"
    else
        echo -e "${COLOR_YELLOW}⚠ 未发现正在运行的SDK进程${COLOR_RESET}"
    fi
}

wait_seconds() {
    local seconds=$1
    echo -e "${COLOR_YELLOW}等待${seconds}秒...${COLOR_RESET}"
    sleep $seconds
}

start_sdk() {
    echo -e "${COLOR_YELLOW}正在启动SDK进程...${COLOR_RESET}"
    
    # 根据机器人型号选择不同的目录
    if [[ "$ROBOT_MODEL" == "R1-PRO" ]]; then
        SESSION_DIR="R1PROBody.d"
    elif [[ "$ROBOT_MODEL" == "R1-LITE" ]]; then
        SESSION_DIR="R1LITEBody.d"
    else
        echo -e "${COLOR_YELLOW}⚠ 未知机器人型号，使用默认目录${COLOR_RESET}"
        SESSION_DIR="R1PROBody.d"
    fi
    
    # 进入脚本目录并启动
    cd "$HOME_DIR/galaxea/install/startup_config/share/startup_config/script" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        echo "执行命令: ./robot_startup.sh boot ../sessions.d/ATCStandard/$SESSION_DIR"
        ./robot_startup.sh boot "../sessions.d/ATCStandard/$SESSION_DIR" &
        SDK_PID=$!
        echo -e "${COLOR_GREEN}✓ SDK进程已启动 (PID: $SDK_PID)${COLOR_RESET}"
        return 0
    else
        echo -e "${COLOR_RED}错误：无法进入脚本目录！${COLOR_RESET}"
        return 1
    fi
}

source_environment() {
    echo -e "${COLOR_YELLOW}加载环境变量...${COLOR_RESET}"
    
    if [[ ! -f "$SETUP_BASH" ]]; then
        echo -e "${COLOR_RED}错误：找不到 setup.bash 文件！${COLOR_RESET}"
        return 1
    fi
    
    source "$SETUP_BASH"
    if [[ $? -ne 0 ]]; then
        echo -e "${COLOR_RED}错误：无法source setup.bash！${COLOR_RESET}"
        return 1
    fi
    
    echo -e "${COLOR_GREEN}✓ 环境变量加载成功${COLOR_RESET}"
    return 0
}

execute_python_script() {
    local script_type="$1"
    local script_dir="$2"
    local python_script="$3"
    local description="$4"
    
    echo -e "${COLOR_YELLOW}进入${script_type}目录...${COLOR_RESET}"
    
    if [[ ! -d "$script_dir" ]]; then
        echo -e "${COLOR_RED}错误：找不到脚本目录！${COLOR_RESET}"
        return 1
    fi
    
    cd "$script_dir"
    if [[ $? -ne 0 ]]; then
        echo -e "${COLOR_RED}错误：无法进入脚本目录！${COLOR_RESET}"
        return 1
    fi
    
    echo -e "${COLOR_GREEN}✓ 进入目录: $script_dir${COLOR_RESET}"
    
    if [[ ! -f "$python_script" ]]; then
        echo -e "${COLOR_RED}错误：找不到Python脚本！${COLOR_RESET}"
        return 1
    fi
    
    echo -e "${COLOR_YELLOW}执行${description}...${COLOR_RESET}"
    echo "执行命令: python3 $python_script"
    echo -e "${COLOR_CYAN}==================================================${COLOR_RESET}"
    echo -e "${COLOR_CYAN}开始${description}${COLOR_RESET}"
    echo -e "${COLOR_CYAN}==================================================${COLOR_RESET}"
    
    python3 "$python_script"
    
    echo ""
    echo -e "${COLOR_YELLOW}${description}执行完成${COLOR_RESET}"
}

ask_kill_sdk() {
    echo -n "是否要杀掉SDK进程？(y/n): "
    read -n 1 kill_after
    echo ""
    if [[ "$kill_after" == "y" ]] || [[ "$kill_after" == "Y" ]]; then
        echo "正在杀死SDK进程..."
        bash "$START_SHELL_PATH" kill 2>/dev/null
        echo -e "${COLOR_GREEN}✓ SDK进程已杀死${COLOR_RESET}"
    else
        echo "SDK进程保持运行状态"
    fi
}

# 配置发现服务 - 客户端
configure_discovery_client() {
    echo ""
    echo "正在配置发现服务（客户端）..."
    
    # 获取用户输入
    read -p "请输入服务端的IP: " robot_ip
    read -sp "请输入您本机的密码: " passwd
    echo ""
    
    # 文件路径
    hardware_file="/opt/galaxea/body/hardware.json"
    xml_file="/opt/galaxea/find_server/super_client_configuration_file.xml"
    xml_dir=$(dirname "${xml_file}")
    bash_file="$HOME/.bashrc"
    
    # 创建XML目录
    if [ ! -d "${xml_dir}" ]; then
        echo "XML文件目录 ${xml_dir} 不存在，正在创建..."
        echo "$passwd" | sudo -S mkdir -p "${xml_dir}" || {
            echo -e "${COLOR_RED}错误：创建目录失败${COLOR_RESET}"
            return 1
        }
    fi
    
    # 创建XML文件（如果不存在）
    if [ ! -f "${xml_file}" ]; then
        echo "正在创建XML文件..."
        echo "$passwd" | sudo -S tee "${xml_file}" > /dev/null <<EOF
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
    <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
        <participant profile_name="super_client_profile" is_default_profile="true">
            <rtps>
                <builtin>
                    <discovery_config>
                        <discoveryProtocol>SUPER_CLIENT</discoveryProtocol>
                        <discoveryServersList>
                            <RemoteServer prefix="44.53.00.5f.45.50.52.4f.53.49.4d.41">
                                <metatrafficUnicastLocatorList>
                                    <locator>
                                        <udpv4>
                                            <address>ros_ip</address>
                                            <port>11811</port>
                                        </udpv4>
                                    </locator>
                                </metatrafficUnicastLocatorList>
                            </RemoteServer>
                        </discoveryServersList>
                    </discovery_config>
                </builtin>
            </rtps>
        </participant>
    </profiles>
</dds>
EOF
        if [ $? -ne 0 ]; then
            echo -e "${COLOR_RED}错误：创建XML文件失败${COLOR_RESET}"
            return 1
        fi
    fi
    
    # 设置权限
    echo "$passwd" | sudo -S chmod 777 "${xml_file}" "${xml_dir}"
    
    # 更新IP地址
    echo "$passwd" | sudo -S sed -i "s|<address>.*</address>|<address>${robot_ip}</address>|g" "${xml_file}"
    
    echo "已将XML文件中<address>的值修改为: ${robot_ip}"
    
    # 更新环境变量
    sed -i "/^export FASTRTPS_DEFAULT_PROFILES_FILE=/d" "${bash_file}"
    echo "export FASTRTPS_DEFAULT_PROFILES_FILE=${xml_file}" >> "${bash_file}"
    
    # 停止ROS2守护进程
    ros2 daemon stop 2>/dev/null
    
    echo -e "${COLOR_GREEN}✓ 发现服务客户端配置完成${COLOR_RESET}"
}

# 配置发现服务 - 服务器端
configure_discovery_server() {
    echo ""
    echo "正在配置发现服务（服务器端）..."
    
    # 根据机器人型号设置参数
    case "${ROBOT_MODEL}" in
        R1-LITE)
            passwd="1"
            wlan_device="wlo1"
            ethernet_device="enp2s0"
            ;;
        R1-PRO|R1)
            passwd="nvidia"
            wlan_device="wlan0"
            ethernet_device="eth1"
            ;;
        *)
            echo -e "${COLOR_RED}错误：不支持的机器人型号${COLOR_RESET}"
            return 1
            ;;
    esac
    
    # 获取IP地址
    get_ip() {
        local iface="$1"
        if ip link show "$iface" &>/dev/null && ip link show "$iface" | grep -q "UP"; then
            ip -4 addr show "$iface" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1
        fi
    }
    
    # 尝试获取IP
    ip_address=$(get_ip "${ethernet_device}")
    [ -z "$ip_address" ] && ip_address=$(get_ip "${wlan_device}")
    
    if [ -z "$ip_address" ]; then
        echo -e "${COLOR_RED}错误：未获取到任何活动网卡IP地址${COLOR_RESET}"
        return 1
    fi
    
    echo "获取到的IP地址: ${ip_address}"
    
    # 文件路径
    xml_file="/opt/galaxea/find_server/super_client_configuration_file.xml"
    xml_dir=$(dirname "${xml_file}")
    bash_file="$HOME/.bashrc"
    
    # 创建XML目录
    if [ ! -d "${xml_dir}" ]; then
        echo "XML文件目录不存在，正在创建：${xml_dir}"
        echo "$passwd" | sudo -S mkdir -p "${xml_dir}" || {
            echo -e "${COLOR_RED}错误：创建目录失败${COLOR_RESET}"
            return 1
        }
    fi
    
    # 创建XML文件（如果不存在）
    if [ ! -f "${xml_file}" ]; then
        echo "正在创建XML文件..."
        echo "$passwd" | sudo -S tee "${xml_file}" > /dev/null <<EOF
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
    <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
        <participant profile_name="super_client_profile" is_default_profile="true">
            <rtps>
                <builtin>
                    <discovery_config>
                        <discoveryProtocol>SUPER_CLIENT</discoveryProtocol>
                        <discoveryServersList>
                            <RemoteServer prefix="44.53.00.5f.45.50.52.4f.53.49.4d.41">
                                <metatrafficUnicastLocatorList>
                                    <locator>
                                        <udpv4>
                                            <address>ros_ip</address>
                                            <port>11811</port>
                                        </udpv4>
                                    </locator>
                                </metatrafficUnicastLocatorList>
                            </RemoteServer>
                        </discoveryServersList>
                    </discovery_config>
                </builtin>
            </rtps>
        </participant>
    </profiles>
</dds>
EOF
        if [ $? -ne 0 ]; then
            echo -e "${COLOR_RED}错误：创建XML文件失败${COLOR_RESET}"
            return 1
        fi
    fi
    
    # 设置权限
    echo "$passwd" | sudo -S chmod 777 "${xml_file}" "${xml_dir}"
    
    # 更新IP地址
    echo "$passwd" | sudo -S sed -i "/<metatrafficUnicastLocatorList>/,/<\/metatrafficUnicastLocatorList>/s|<address>.*</address>|<address>${ip_address}</address>|" "${xml_file}"
    
    echo "XML文件中的<address>已更新为：${ip_address}"
    
    # 清理和更新环境变量
    sed -i '/^export ROS_IP=/d;/^export ROS_LOCALHOST_ONLY=/d;/^export ROS_DISCOVERY_SERVER=/d;/^export ROS_DOMAIN_ID=/d;/^export FASTRTPS_DEFAULT_PROFILES_FILE=/d' "${bash_file}"
    echo "export FASTRTPS_DEFAULT_PROFILES_FILE=${xml_file}" >> "${bash_file}"
    
    echo "已更新 ${bash_file} 中的环境变量配置"
    
    # 重启提示
    read -p "发现服务配置完成，是否重启计算机？(y/n): " reboot_choice
    if [[ "$reboot_choice" == "y" ]] || [[ "$reboot_choice" == "Y" ]]; then
        echo "正在重启计算机..."
        echo "$passwd" | sudo -S reboot
    else
        echo "已跳过重启，请手动重启以应用配置"
    fi
}

# 配置发现服务
configure_discovery_service() {
    CURRENT_USER=$(whoami)
    echo "当前用户: $CURRENT_USER"
    
    if [[ "$CURRENT_USER" == "nvidia" ]] || [[ "$CURRENT_USER" == "r1lite" ]]; then
        configure_discovery_server
    else
        configure_discovery_client
    fi
}

# 核心功能函数
run_unboxing_detection() {
    echo ""
    echo "正在执行开箱检测..."
    echo "机器人型号: $ROBOT_MODEL"
    
    # 检查启动脚本是否存在
    if [[ ! -f "$START_SHELL_PATH" ]]; then
        echo -e "${COLOR_RED}错误：找不到机器人启动脚本！${COLOR_RESET}"
        return 1
    fi
    
    # SDK进程管理
    kill_sdk
    wait_seconds 3
    start_sdk || return 1
    wait_seconds 3  # 开箱检测等待3秒
    
    # 回到安装目录并启动rds_ros_node
    cd ~/galaxea/install 2>/dev/null
    if [[ $? -ne 0 ]]; then
        echo -e "${COLOR_RED}错误：无法回到安装目录！${COLOR_RESET}"
        return 1
    fi
    
    source_environment
    
    echo -e "${COLOR_YELLOW}启动rds_ros_node...${COLOR_RESET}"
    echo -e "${COLOR_CYAN}开始开箱检测${COLOR_RESET}"
    echo -e "${COLOR_CYAN}==================================================${COLOR_RESET}"
    
    ros2 run rds_ros rds_ros_node
    
    # 完成后询问
    ask_kill_sdk
}

run_unboxing_script() {
    echo ""
    
    # R1-LITE不执行开箱脚本
    if [[ "$ROBOT_MODEL" == "R1-LITE" ]]; then
        echo -e "${COLOR_RED}R1-LITE机器人不支持开箱脚本(控制躯干蹲起)功能${COLOR_RESET}"
        echo "请选择其他功能"
        return 1
    fi
    
    echo "正在执行开箱脚本..."
    echo "机器人型号: $ROBOT_MODEL"
    
    # SDK进程管理
    kill_sdk
    wait_seconds 3
    start_sdk || return 1
    wait_seconds 3  # 开箱脚本等待3秒
    
    # 加载环境变量
    source_environment || return 1
    
    # 根据机器人型号选择目录和脚本
    if [[ "$ROBOT_MODEL" == "R1-PRO" ]]; then
        SCRIPT_DIR="$HOME_DIR/galaxea/install/mobiman/share/mobiman/scripts/robotOpenbox/R1Pro"
        PYTHON_SCRIPT="r1pro_first_motion.py"
        SCRIPT_TYPE="R1-PRO开箱脚本"
        DESCRIPTION="开箱脚本 (控制躯干蹲起)"
    else
        echo -e "${COLOR_RED}错误：不支持的机器人型号！${COLOR_RESET}"
        return 1
    fi
    
    # 执行Python脚本
    execute_python_script "$SCRIPT_TYPE" "$SCRIPT_DIR" "$PYTHON_SCRIPT" "$DESCRIPTION"
    
    # 完成后询问
    ask_kill_sdk
}

run_demo_show() {
    echo ""
    echo "正在执行DEMO展示..."
    echo "机器人型号: $ROBOT_MODEL"
    
    # SDK进程管理
    kill_sdk
    wait_seconds 3
    start_sdk || return 1
    wait_seconds 12  # DEMO展示等待12秒
    
    # 加载环境变量
    source_environment || return 1
    
    # 根据机器人型号选择目录和脚本
    if [[ "$ROBOT_MODEL" == "R1-PRO" ]]; then
        SCRIPT_DIR="$HOME_DIR/galaxea/install/mobiman/share/mobiman/scripts/robotOpenbox/R1Pro"
        PYTHON_SCRIPT="r1pro_test_open_box.py"
        SCRIPT_TYPE="执行R1-PRO DEMO展示"
        DESCRIPTION="执行DEMO展示"
    elif [[ "$ROBOT_MODEL" == "R1-LITE" ]]; then
        SCRIPT_DIR="$HOME_DIR/galaxea/install/mobiman/share/mobiman/scripts/robotOpenbox/R1lite"
        PYTHON_SCRIPT="r1_lite_test_open_box.py"
        SCRIPT_TYPE="执行R1-LITE DEMO展示"
        DESCRIPTION="执行DEMO展示"
    else
        echo -e "${COLOR_RED}错误：不支持的机器人型号！${COLOR_RESET}"
        return 1
    fi
    
    # 执行Python脚本
    execute_python_script "$SCRIPT_TYPE" "$SCRIPT_DIR" "$PYTHON_SCRIPT" "$DESCRIPTION"
    
    # 完成后询问
    ask_kill_sdk
}

kill_menu() {
    echo ""
    echo -e "${COLOR_GREEN}[Y/y]${COLOR_RESET} 杀死当前SDK进程"
    echo -e "${COLOR_GREEN}[N/n]${COLOR_RESET} 继续使用SDK"
    echo -e "${COLOR_GREEN}[Q/q]${COLOR_RESET} 返回上级目录"
}

kill_process() {
    while true; do
        clear
        kill_menu
        read -n 1 -s choice
        case $choice in
            [Yy])
                echo ""
                echo "正在杀死当前SDK进程..."
                if [[ -f "$START_SHELL_PATH" ]]; then
                    bash "$START_SHELL_PATH" kill
                else
                    echo -e "${COLOR_RED}错误：找不到启动脚本 $START_SHELL_PATH${COLOR_RESET}"
                fi
                echo "按任意键继续..."
                read -n 1
                break
                ;;
            [Nn])
                echo ""
                echo "继续使用当前SDK进程,按Q返回上级目录。"
                continue
                ;;
            [Qq])
                echo ""
                echo "返回上级目录..."
                break
                ;;
            *)
                echo ""
                echo "无效选项，请重新选择！"
                sleep 1
                ;;
        esac
    done
}

# 主循环
while true; do
    show_menu
    read choice
    
    case $choice in
        1)
            run_unboxing_detection
            ;;
        2)
            if [[ "$ROBOT_MODEL" == "R1-PRO" ]]; then
                run_unboxing_script
            else
                echo -e "${COLOR_RED}R1-LITE机器人不支持开箱脚本功能，请选择其他选项${COLOR_RESET}"
            fi
            ;;
        3)
            run_demo_show
            ;;
        4)
            configure_discovery_service
            ;;
        5)
            kill_process
            ;;
        6)
            echo ""
            echo "再见！"
            echo "机器人型号: $ROBOT_MODEL 已记录"
            exit 0
            ;;
        *)
            # 根据机器人型号显示不同的提示
            if [[ "$ROBOT_MODEL" == "R1-PRO" ]]; then
                echo "无效选项，请重新选择！(1-6)"
            elif [[ "$ROBOT_MODEL" == "R1-LITE" ]]; then
                echo "无效选项，请重新选择！(1,3-6)"
            fi
            ;;
    esac
    
    echo ""
    echo "按Enter键继续..."
    read
done