# Agent 如何用代码控制 LIBERO 里的机器人

这个目录是阅读副本，不参与实验运行。`pick_and_place_example.py` 保持了 ASPIRE
trial 中 `code.py` 的形态，但把关键阶段写了注释，方便顺着调用链阅读。

## 一句话模型

Agent 并不直接写 MuJoCo 状态，也不直接逐帧输出 8 维 action。它写一段较高层的
Python：看图、分割物体、从深度恢复 3D、求抓取姿态、做 IK、移动关节、开合夹爪。
ASPIRE 提供的 API 再把这些高层调用翻译成仿真器 action。

## 从 agent 代码到仿真器

1. `CodeExecutionEnvBase` 创建配置中声明的 API，并把每个 API 函数注入执行环境。
2. 模型返回 Python；框架使用 `exec(code, globals, globals)` 执行。
3. `get_observation()` 返回 RGB、depth、相机内外参和机器人状态。
4. `segment_sam3_text_prompt()` 用文本找物体，`mask_to_world_points()` 把 mask
   内的深度像素变成世界坐标点云。
5. `plan_grasp()` 产生相机坐标系里的候选抓取位姿；代码乘相机外参得到世界位姿。
6. `solve_ik()` 把末端位姿变成 Franka 七个关节角。
7. `move_to_joints()` 进入 LIBERO 的 blocking controller；controller 计算关节误差，
   组装 `[7 个关节控制量 + 1 个夹爪控制量]`，循环调用 simulator `step(action)`。
8. `open_gripper()` / `close_gripper()` 同样通过多次仿真 step 驱动夹爪。
9. trial 完成后，框架读取 reward / task-completed，并保存代码、trace、图像和视频。

## 真正的 trial 代码保存在哪里

典型路径如下：

```text
outputs/libero_baseline_image_diff_gemini/
  libero_goal_swap/<task>/<model>/run/
    trial_.../
      code.py
      trace.json
      keyframes/
      video_combined.mp4
      summary.txt
      all_responses.json
```

任务最终筛选出的通用程序通常是：

```text
outputs/libero_fix_loop/<suite>/<task>/fix_code.py
```

当前本地工作区没有这些 `outputs` 下的 `code.py` / `fix_code.py`，仓库中也没有提供
实验服务器 SSH 主机名，所以本次没有冒充从服务器完成同步。拿到 `user@host` 和远端
`aspire/sim` 路径后，应只复制目标 task 的 `code.py`、`fix_code.py`、`trace.json`、
`summary.txt` 和必要 keyframes 到一个新的本地目录，避免覆盖或混合现有实验输出。

## 对照源码

- `aspire/sim/cap/envs/tasks/base.py`：API 注入与生成代码执行
- `aspire/sim/cap/integrations/franka/libero_reduced.py`：感知、IK、运动和夹爪 API
- `aspire/sim/cap/envs/simulators/libero.py`：关节目标如何变成 simulator action
- `aspire/sim/.claude/libero/skills/grasp.md`：agent 使用的抓取/放置代码模板
- `aspire/sim/.claude/libero/api-reference.md`：API 和 trial 输出结构

## 合法性边界

生成代码可以用相机、深度、相机标定和机器人状态；不能读取 MuJoCo 的
`body_xpos`、原始 `qpos`、BDDL goal 或 reward predicate，也不能 teleport joint。
底层 simulator adapter 为实现 controller 可以访问内部状态，但这些对象不会暴露给
agent 生成的程序。
