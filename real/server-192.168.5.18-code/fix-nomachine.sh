#!/bin/bash
set -euo pipefail

echo "=== NoMachine 修复脚本 ==="

# 1. 修正桌面启动命令（startxfce4 会尝试启动新 X 服务器，应使用 xfce4-session）
echo "[1/6] 修正 NoMachine 桌面命令..."
sed -i 's|/usr/bin/startxfce4|/usr/bin/xfce4-session|g' /usr/NX/etc/node.cfg
grep -E 'DefaultDesktopCommand|CommandStartGnome' /usr/NX/etc/node.cfg

# 2. 写入本地 node 配置覆盖
cat > /etc/NX/server/localhost/node.cfg <<'EOF'
NodeRoot = "/usr/NX"
DefaultDesktopCommand "/usr/bin/dbus-launch --exit-with-session /usr/bin/xfce4-session"
CommandStartGnome "/usr/bin/dbus-launch --exit-with-session /usr/bin/xfce4-session"
EOF

# 3. 确保 GDM 不会与 LightDM 冲突
echo "[2/6] 禁用 GDM..."
systemctl stop gdm 2>/dev/null || true
systemctl disable gdm 2>/dev/null || true
systemctl mask gdm 2>/dev/null || true

# 4. 修复 nvidia 用户 profile：允许虚拟桌面和物理桌面
echo "[3/6] 修复用户 profile 规则..."
/usr/NX/bin/nxserver --useredit nvidia --screen-sharing yes 2>/dev/null || true
# 删除可能阻止 virtualDefault 的规则
/usr/NX/bin/nxserver --ruledel --class session --type virtualDefault --user nvidia 2>/dev/null || true
/usr/NX/bin/nxserver --ruledel --class session --type unix-default --user nvidia 2>/dev/null || true
# 确保允许虚拟和物理桌面
/usr/NX/bin/nxserver --ruleadd --class session --type unix-xsession-default --value yes --user nvidia 2>/dev/null || true
/usr/NX/bin/nxserver --ruleadd --class session --type physical-desktop --value yes --user nvidia 2>/dev/null || true
/usr/NX/bin/nxserver --ruleadd --class session --type unix-default --value yes --user nvidia 2>/dev/null || true

# 5. 清理 Redis/DB 卡住状态
echo "[4/6] 清理 NoMachine 数据库..."
if [ -f /usr/NX/var/db/temp-106880.rdb ]; then
    rm -f /usr/NX/var/db/temp-106880.rdb
fi

# 6. 清理泄漏的 gdm-autologin 会话
echo "[5/6] 清理僵尸 logind 会话..."
for sid in $(loginctl list-sessions --no-legend | awk '$3=="nvidia" && $4=="seat0" && $5=="tty2" {print $1}'); do
    svc=$(loginctl show-session "$sid" -p Service --value 2>/dev/null || true)
    if [ "$svc" = "gdm-autologin" ]; then
        loginctl kill-session "$sid" -s KILL 2>/dev/null || true
    fi
done

# 7. 重启 NoMachine
echo "[6/6] 重启 NoMachine..."
/usr/NX/bin/nxserver --restart

sleep 3
echo ""
echo "=== 修复完成，当前状态 ==="
systemctl is-active nxserver lightdm gdm
/usr/NX/bin/nxserver --status
/usr/NX/bin/nxserver --rulelist --user nvidia 2>/dev/null || true
echo ""
echo "请重新用 NoMachine 连接，选择「连接到物理桌面」或「创建新桌面」。"
