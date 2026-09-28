# CaP-X 运行时架构分析

交互图： [capx-runtime.html](capx-runtime.html)  
可编辑规格： [capx-runtime.architecture.json](capx-runtime.architecture.json)

分析基于当前仓库源码，版本为 `53e9966d7a8e2fa7494676772bccc35280f5c0ed`。这是静态代码分析，并未启动 GPU 仿真、外部模型或机器人硬件。现有研究草稿不作为已实现架构的依据。

## 主执行链路

`launch.py` 读取 YAML 和 CLI 参数，按配置启动或复用 API 服务，再选择 CLI 评测或 Web 模式。CLI 的 `runner.py` 使用 `spawn` 多进程和任务／结果队列分派试验；每个 worker 初始化一次环境，在多个 trial 之间复用。

单次 trial 重置环境，组装任务指令与 API 文档，向模型发送请求，提取 Python 并调用 `env.step(code)`。`CodeExecutionEnvBase._exec_user_code()` 使用进程内 `exec` 执行代码；函数、观测与低层环境绑定到持久命名空间，多轮之间保留变量，reset 时清空。这里没有独立的容器执行沙箱。

机器人 API 在同一环境进程内调用，负责感知、抓取、移动与夹爪控制。它们通过底层适配器操作 Robosuite、LIBERO、BEHAVIOR 或真机，并通过 HTTP 调用所需的感知／规划服务。`step()` 返回观测、奖励、结束状态、stdout 和 stderr。多轮编排可结合图像／视频差分，选择重新生成代码或结束。

依据：`capx/envs/launch.py:201`、`capx/envs/runner.py:122`、`capx/utils/parallel_eval.py:227`、`capx/envs/trial.py:629`、`capx/envs/tasks/base.py:153`。

## 运行边界与通信

| 边界 | 源码中的实现 |
| --- | --- |
| 浏览器 ↔ Web 服务 | React 经 HTTP 管理会话，`/ws/{session_id}` 传送命令和事件；FastAPI 默认监听 8200，并提供构建后的前端资源。 |
| Web ↔ 环境执行 | `run_trial_async` 与 CLI 的 `_run_single_trial` 是两个编排实现；Web 使用协程和单线程环境执行器。图中将它们归为一个逻辑角色，不表示 Web 调用 CLI worker。 |
| 编排 ↔ LLM / VLM | `capx/llm/client.py` 使用 HTTP 请求可配置端点；可以使用本地 OpenRouter 代理或 vLLM，视觉差分模型也可单独配置。 |
| API ↔ 感知／规划 | 示例配置使用 SAM3 `:8114/segment`、Contact-GraspNet `:8115/plan`、PyRoKi `:8116/ik` 或 `/plan`；这是一组独立服务，图中合并展示。其他配置可选 OWL-ViT、SAM2、cuRobo 等实现。 |
| 真机适配器 ↔ 机器人侧客户端 | `FrankaRealLowLevel` 在 `:9000` 启动 TCP 服务，通过带长度前缀的 MsgPack / NumPy 数据交换观测与动作。 |
| 试验 ↔ 存储 | 保存代码、响应、日志、视频与汇总到本地 `outputs/`；图中的存储节点是文件系统。Web 会话状态存放于内存。 |

依据：`capx/web/server.py:70`、`:353`、`:550`，`capx/web/async_trial_runner.py:139`，`capx/llm/client.py:180`，`capx/integrations/vision/sam3.py:65`，`capx/integrations/vision/graspnet.py:163`，`capx/integrations/motion/pyroki.py:43`，`capx/envs/simulators/franka_real.py:104`，`capx/utils/msgpack_server_client_utils.py`，`capx/utils/launch_utils.py:376`。

## 配置与可选路径

- **服务生命周期**：`runner._start_api_servers()` 检查端口，复用已运行的服务；新服务通过子进程启动，入口退出时清理所启动的进程。为减少连线干扰，图中用说明卡记录生命周期，不额外绘制启动控制线。
- **模型地址**：`LaunchArgs.server_url` 默认使用 `:8110/chat/completions`，而 OpenRouter 代理脚本默认端口是 8111；README 的启动示例显式传入 `--port 8110`。图中保留“地址可配置”的语义。
- **后端选择**：底层环境由配置和注册表选择；这些替代实现不意味着全部同时启动。真机 TCP 通道只适用于真机配置。
- **Agent0 能力**：多轮反馈、可选视觉差分、并行候选推理和技能库由配置与具体 API 变体启用。`control_reduced_skill_library.py` 等模块提供可复用技能函数，不是必经的远程服务。
- **CaP-RL**：独立训练脚本 `scripts/train_franka_grpo.sh:60` 启动 VeRL，`:90` 注册 `verl_agent_reward/capx_franka_reward.py` 的 `compute_score`。奖励函数缓存环境、执行 `env.step(solution_str)`，再将分数返回训练器。它复用同类执行环境，不经过评测入口；图中以说明卡标注这条可选路径。
- **可视化**：Web 服务还支持 Viser 代理；该辅助显示通道未单独列为主节点。

## 交付与验证

图内节点附有可追溯源码引用，支持主题切换、缩放、搜索和导出。

- `capx-runtime.delivery.json`：规格与 HTML 的 SHA-256、字节数、源码验证和 9 项 showcase 校验结果。
- `capx-runtime.visual-check.json`：与最终 HTML 哈希绑定的四种桌面尺寸检查及浅色／深色截图记录。
- `capx-runtime.visual-check.html`：浏览器检查截图总览。
- `capx-runtime.review.json`：图像复核状态与交付摘要；浏览器自动结果与图像复核分开记录。
