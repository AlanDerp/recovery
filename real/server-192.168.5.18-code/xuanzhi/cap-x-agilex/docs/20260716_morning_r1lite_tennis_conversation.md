# 2026-07-16 早上 R1 Lite 右臂真机测试记录

保存时间：2026-07-16 13:29 CST

## 用户要求

- 将右臂的 home 状态设置成现在的状态，不要伸到前面。
- 进行一次真机测试。
- 将 home 更新为右臂断电时收起来的状态。
- 进行一次真机抓网球放入盒子的测试。
- 不使用腕部 depth。

## Home 状态更新

右臂 home 已更新为断电时收起来的 tucked 状态。

新的 home command：

```text
[0.001534, 0.017453, -0.017453, -0.001534, 0.079767, -0.024544]
```

新的 home feedback：

```text
[0.017234, 0.242766, 0.005319, 0.017021, 0.100638, 0.001915]
```

涉及文件：

- `capx/envs/simulators/r1lite_real.py`
- `env_configs/r1lite/r1lite_tennis_ball_in_box.yaml`
- `env_configs/r1lite/r1lite_pen_grasp_safe.yaml`
- `env_configs/r1lite/r1lite_tennis_ball_in_box_head_only.yaml`

已做验证：

- Python 编译检查通过。
- YAML 解析检查通过。
- 真机 home 验证通过，最终反馈保持在 tucked home 附近。

## 第一次抓取测试

运行 full pipeline：

```text
scripts/run_r1lite_tennis_ball_in_box_pipeline.sh
```

输出目录：

```text
outputs/r1lite_tennis_real_test_20260716_130752
```

日志：

```text
logs/r1lite_tennis_real_test_20260716_130752.log
```

已通过的检查：

- GraspNet calibration OK。
- robot stack publishing `/hdas/feedback_arm_right`。
- MOCA isolated OK。
- right arm MIT mode OK。
- isolated RelaxedIK OK。
- head plane calibration OK。
- SAM3 服务可用：`127.0.0.1:8114`。
- remote Contact-GraspNet 可用：`192.168.5.100:8115`。
- wrist TF calibration 可用。

中止原因：

```text
mandatory wrist RGB-D topic has no message:
/hdas/camera_wrist_right/aligned_depth_to_color/image_raw
```

结论：失败点是腕部 depth 无消息。

## 不使用腕部 depth 的测试

用户明确要求“不用腕部的 depth”后，改为 head-only 测试路径：

- 使用头部相机定位 tennis ball 和 cardboard box。
- 不把腕部 depth 作为必需输入。
- 使用隔离 RelaxedIK 做 IK probe。
- 使用标准 MOCA 执行 live `/motion_target` joint target。

创建临时配置：

```text
env_configs/r1lite/r1lite_tennis_ball_in_box_head_only_live.yaml
```

临时配置变更：

```text
arm_target_topic: /motion_target/target_joint_state_arm_right
gripper_target_topic: /motion_target/target_position_gripper_right
relaxed_ik_target_topic: /capx/relaxed_ik_target_pose_arm_right
relaxed_ik_solution_topic: /relaxed_ik/isolated_target_joint_state_arm_right
```

同时停止 teleop，避免覆盖目标。

## Head-only 第一次尝试

日志：

```text
logs/r1lite_tennis_head_only_live_20260716_131703.log
```

中止原因：

```text
RelaxedIK probe topic is not isolated; subscriptions=['moca_adapter', 'relaxed_ik_right']
```

结论：安全保护逻辑阻止了测试继续，因为 live RelaxedIK pose topic 还可能被 MOCA 看到。

处理方式：

- 停止 live 右臂 RelaxedIK。
- 启动隔离版右臂 RelaxedIK。
- 隔离输入 topic：`/capx/relaxed_ik_target_pose_arm_right`
- 隔离输出 topic：`/relaxed_ik/isolated_target_joint_state_arm_right`

验证结果：

- `/capx/relaxed_ik_target_pose_arm_right` 只有 `relaxed_ik_right` 订阅。
- `/relaxed_ik/isolated_target_joint_state_arm_right` 只有 `relaxed_ik_right` 发布。

## Head-only 第二次尝试

日志：

```text
logs/r1lite_tennis_head_only_live_20260716_132047.log
```

输出目录：

```text
outputs/r1lite_tennis_head_only_live_20260716_132047
```

定位结果：

```text
box_pos: [0.7670191, -0.05910579, 0.96476664]
drop_pos: [0.7670191, -0.05910579, 1.06476664]

grasp_pos: [0.48769443, -0.2934045, 0.94876664]
pre_grasp_pos: [0.48769443, -0.2934045, 1.06876664]
```

测试状态：

- 头部相机定位成功。
- 已越过此前的腕部 depth 问题。
- 已越过 RelaxedIK 隔离安全检查。
- 右臂实际运动到 pre-grasp 附近。
- 未进入下探、闭爪和搬运阶段。

中止原因：

```text
adaptive IK waypoint reached joint feedback but not the requested end-effector pose
```

关键误差：

```text
target_position: [0.4876944270899141, -0.29340450140002844, 1.0687666359375712]
measured_position: [0.4853837067179893, -0.3210344725599184, 1.0631222587555897]
position_error_m: 0.028295118460717777
orientation_error_rad: 0.02318870194472479
```

结论：关节反馈到位，但末端实际位置与目标还有约 2.8 cm 误差，超过当前安全容差，因此程序停止。未抓取物体。

## 测试后恢复

测试失败后，右臂停在 pre-grasp 附近：

```text
[0.053404, 0.682128, -0.618085, 0.190426, 0.085957, 0.007021]
```

随后连续发布新的 tucked home 目标到：

```text
/motion_target/target_joint_state_arm_right
```

最终右臂已收回 tucked home 附近：

```text
[0.018298, 0.242766, -0.000426, 0.018936, 0.101064, 0.002340]
```

## 当前状态

- 右臂已收回新的 tucked home 附近。
- teleop 保持停止，避免覆盖控制目标。
- MOCA 仍在运行。
- 左臂 RelaxedIK 仍在运行。
- 右臂 RelaxedIK 为隔离版，只用于 IK probe。
- 本次未使用腕部 depth 作为必需输入。

## 后续建议

下一步如果继续测试，应优先处理 pre-grasp 处 2.8 cm 末端位置误差：

- 检查 RelaxedIK 解算 frame 与 MOCA/反馈 FK frame 是否一致。
- 检查 `right_gripper_link`、EE feedback、base frame 的 TF 定义。
- 适当放宽 pre-grasp pose 校验容差只适合临时验证，不建议直接用于抓取闭环。
- 更稳的方案是 head-only 定位后，在 pre-grasp 附近加入视觉或接触修正，而不是直接依赖一次 open-loop IK。
