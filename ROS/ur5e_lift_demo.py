import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

# BASIC 控制器：手臂是 OSC（末端增量位姿控制），夹爪是开合控制
controller_config = load_composite_controller_config(controller="BASIC", robot="UR5e")

env = suite.make(
    env_name="Lift",              # 桌上有一个方块，任务是举起它
    robots="UR5e",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="frontview",
    use_camera_obs=False,
    control_freq=20,
)

# 动作空间: [dx, dy, dz, droll, dpitch, dyaw, gripper]，共 7 维
# gripper: -1 = 张开, +1 = 闭合
print("Action dim:", env.action_dim)


def scripted_policy(obs, phase):
    """一个简单的分阶段抓取策略。返回 (action, next_phase)。"""
    eef_pos = obs["robot0_eef_pos"]   # 末端执行器位置
    cube_pos = obs["cube_pos"]        # 方块位置

    action = np.zeros(7)
    kp = 5.0  # 简单的比例增益

    if phase == "reach":
        # 移动到方块正上方 10cm
        target = cube_pos + np.array([0, 0, 0.10])
        action[:3] = kp * (target - eef_pos)
        action[6] = -1  # 张开夹爪
        if np.linalg.norm(target - eef_pos) < 0.01:
            phase = "descend"

    elif phase == "descend":
        # 下降到方块高度
        target = cube_pos + np.array([0, 0, -0.005])
        action[:3] = kp * (target - eef_pos)
        action[6] = -1
        if np.linalg.norm(target - eef_pos) < 0.008:
            phase = "grasp"

    elif phase == "grasp":
        action[6] = 1  # 闭合夹爪，位置保持不动
        phase = "grasp_wait"

    elif phase == "grasp_wait":
        action[6] = 1
        phase = "lift" if obs.get("_grasp_timer", 0) is None else phase
        # 简单起见用固定等待，见主循环

    elif phase == "lift":
        action[:3] = kp * np.array([0, 0, 0.02])  # 持续向上
        action[6] = 1

    action[:3] = np.clip(action[:3], -1, 1)
    return action, phase


obs = env.reset()
phase = "reach"
grasp_wait_steps = 0

for step in range(1000):
    action, phase = scripted_policy(obs, phase)

    # 合爪后等 15 步再抬升，让夹爪夹稳
    if phase == "grasp_wait":
        grasp_wait_steps += 1
        if grasp_wait_steps > 15:
            phase = "lift"

    obs, reward, done, info = env.step(action)
    env.render()

    if reward >= 1.0:  # Lift 任务成功时 reward = 1
        print(f"任务成功！用了 {step} 步")
        # 成功后再渲染一会儿
        for _ in range(60):
            obs, _, _, _ = env.step(np.concatenate([np.zeros(6), [1]]))
            env.render()
        break

env.close()