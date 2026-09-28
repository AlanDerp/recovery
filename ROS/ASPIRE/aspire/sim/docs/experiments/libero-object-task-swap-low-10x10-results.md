# LIBERO-Pro Object-Task / Object-Swap 实验结果

## 1. 实验概览

| 项目 | Object-Task | Object-Swap |
|---|---|---|
| Suite | `libero_object_task` | `libero_object_swap` |
| 基础提交 | `0a36d7a02abad08943f2124146d783fb253677b7` | `0a36d7a02abad08943f2124146d783fb253677b7` |
| 模型 | `deepseek-flash` | `deepseek-flash` |
| Stage 1 | seeds 51–60，10 trials/task | seeds 51–60，10 trials/task |
| Stage 2 | seeds 1–10，10 held-out trials/task | seeds 1–10，10 held-out trials/task |
| 任务数 | 10 | 10 |
| GPU 拓扑 | GPU 0–2：感知/IK 服务；GPU 3–7：5 个任务槽 | GPU 0–2：感知/IK 服务；GPU 3–7：5 个任务槽 |
| Campaign 目录 | `/mnt/nimloth/aspire_campaigns/libero_object_task_main_10x10_20260923` | `/mnt/nimloth/aspire_campaigns/libero_object_swap_main_10x10_20260923` |

Object-Task 使用运行时 `env.handle.task_language` 作为唯一权威目标，不以 slot slug 推断任务。两个实验的 Stage 1 与 Stage 2 seed 分区严格隔离；Stage 2 仅在代码冻结后由协调器执行。

## 2. 总体结果

| Suite | Stage 1 成功数 | Stage 1 成功率 | Stage 2 成功数 | Stage 2 成功率 | Stage 2 满分任务 |
|---|---:|---:|---:|---:|---:|
| Object-Task | 92/100 | 92% | **85/100** | **85%** | 5/10 |
| Object-Swap | 94/100 | 94% | **91/100** | **91%** | 8/10 |
| 合计 | 186/200 | 93% | **176/200** | **88%** | 13/20 |

Object-Swap 的 held-out 成功率比 Object-Task 高 **6 个百分点**（91% 对 85%）。

## 3. 逐任务结果

| # | Slot / Object | Object-Task 运行时目标 | Task S1 | Task S2 | Swap S1 | Swap S2 | S2 差值（Swap − Task） |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | alphabet soup | cream cheese | 10/10 | **10/10** | 7/10 | **5/10** | −5 |
| 2 | cream cheese | alphabet soup | 9/10 | **9/10** | 10/10 | **10/10** | +1 |
| 3 | salad dressing | tomato sauce | 3/10 | **3/10** | 7/10 | **6/10** | +3 |
| 4 | BBQ sauce | ketchup | 10/10 | **10/10** | 10/10 | **10/10** | 0 |
| 5 | ketchup | milk | 10/10 | **7/10** | 10/10 | **10/10** | +3 |
| 6 | tomato sauce | BBQ sauce | 10/10 | **8/10** | 10/10 | **10/10** | +2 |
| 7 | butter | orange juice | 10/10 | **10/10** | 10/10 | **10/10** | 0 |
| 8 | milk | butter | 10/10 | **8/10** | 10/10 | **10/10** | +2 |
| 9 | chocolate pudding | salad dressing | 10/10 | **10/10** | 10/10 | **10/10** | 0 |
| 10 | orange juice | chocolate pudding | 10/10 | **10/10** | 10/10 | **10/10** | 0 |
|  | **总计** |  | **92/100** | **85/100** | **94/100** | **91/100** | **+6** |
