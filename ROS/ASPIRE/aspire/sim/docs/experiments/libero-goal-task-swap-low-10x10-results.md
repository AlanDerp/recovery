# LIBERO-Pro Goal-Task / Goal-Swap 实验结果

## 1. 实验概览

| 项目 | Goal-Task | Goal-Swap |
|---|---|---|
| Suite | `libero_goal_task` | `libero_goal_swap` |
| 基础提交 | `0a36d7a02abad08943f2124146d783fb253677b7` | `0a36d7a02abad08943f2124146d783fb253677b7` |
| 模型 | `deepseek-flash` | `deepseek-flash` |
| Stage 1 | seeds 51–60，10 trials/task | seeds 51–60，10 trials/task |
| Stage 2 | seeds 1–10，10 held-out trials/task | seeds 1–10，10 held-out trials/task |
| 任务数 | 10 | 10 |
| GPU 拓扑 | GPU 0–2：感知/IK 服务；GPU 3–7：5 个任务槽 | GPU 0–2：感知/IK 服务；GPU 3–7：5 个任务槽 |
| Campaign 目录 | `/mnt/nimloth/aspire_campaigns/libero_goal_task_main_10x10_20260923` | `/mnt/nimloth/aspire_campaigns/libero_goal_swap_main_10x10_20260923` |

Goal-Task 使用运行时 `env.handle.task_language` 作为唯一权威目标，不以 slot slug 推断任务。两个实验的 Stage 1 与 Stage 2 seed 分区严格隔离；Stage 2 仅在代码冻结后由协调器执行。

## 2. 总体结果

| Suite | Stage 1 成功数 | Stage 1 成功率 | Stage 2 成功数 | Stage 2 成功率 | Stage 2 满分任务 |
|---|---:|---:|---:|---:|---:|
| Goal-Task | 58/100 | 58% | **56/100** | **56%** | 4/10 |
| Goal-Swap | 66/92 | 71.7% | **69/100** | **69%** | 6/10 |
| 合计 | 124/192 | 64.6% | **125/200** | **62.5%** | 10/20 |

Goal-Swap 的 held-out 成功率比 Goal-Task 高 **13 个百分点**（69% 对 56%）。

## 3. 逐任务结果

| # | Slot / Task | Goal-Task 运行时目标 | Task S1 | Task S2 | Swap S1 | Swap S2 | S2 差值（Swap − Task） |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | open middle drawer | open bottom drawer of the cabinet | 0/10 | **0/10** | 10/10 | **10/10** | +10 |
| 2 | put bowl on stove | put plate on the stove | 0/10 | **0/10** | 0/5 | **0/10** | 0 |
| 3 | put wine bottle on cabinet | put wine bottle in the bowl | 10/10 | **10/10** | 10/10 | **10/10** | 0 |
| 4 | open top drawer and put bowl inside | open top drawer and put cream cheese inside | 6/10 | **6/10** | 8/10 | **9/10** | +3 |
| 5 | put bowl on cabinet | put plate on top of the drawer | 0/10 | **0/10** | 10/10 | **10/10** | +10 |
| 6 | push plate to front of stove | push cream cheese to the front of the stove | 10/10 | **10/10** | 0/10 | **0/10** | −10 |
| 7 | put cream cheese in bowl | put wine bottle in the bowl | 10/10 | **10/10** | 8/10 | **10/10** | 0 |
| 8 | turn on stove | turn off the stove | 10/10 | **8/10** | 10/10 | **10/10** | +2 |
| 9 | put bowl on plate | put wine bottle on the plate | 10/10 | **10/10** | 10/10 | **10/10** | 0 |
| 10 | put wine bottle on rack | put cream cheese on the rack | 2/10 | **2/10** | 0/7 | **0/10** | −2 |
|  | **总计** |  | **58/100** | **56/100** | **66/92** | **69/100** | **+13** |

