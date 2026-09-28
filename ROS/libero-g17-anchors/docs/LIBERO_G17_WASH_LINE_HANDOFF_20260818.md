# LIBERO G17 Fair-FG 课题交接（洗白线）

> 写给从零接手的新 agent。表述尽量贴近用户原话与已钉死的实验结论。  
> 工作树：`/home/qshou/capx_g17_observation_integrity`  
> 产出根：`/project/peilab/qshou/cap-x-agilex/outputs/`  
> 对话锚点：`agent-transcripts/cbdce25e-4cef-42e5-91af-b12ddfd89f96`  
> 日期：2026-08-18

---

## 1. 课题目的（用户原意）

问题拆成两块：

1. **抓对** — 大体已做成（XY-lock / wrong-grasp ordinal / SGP keep-stock）。
2. **放对** — 主战场仍在迭代。

当前主线不是先冲 PT，而是：

> **把 GT 洗白成 FG**（因为 FG 可以用人机协同来 backup）。

整条洗白 / 消融链路：

```text
GT → FG → PT
```

- **GT**：特权真值（连续 residual 等）— 上限 / 同构证明用。
- **FG**：方法形态 — 用**离散逻辑谓词**替换特权连续量；人机协同可 backup。
- **PT**：落地真值源 — 机器人感知 + VLM；相对 GT 的缺口是 **PT−GT**，属 future work，可训可优化。

用户钉死的理论点：

> 无论身份还是 xy placement / c2c 指向，本质上都是**离散的逻辑谓词**，而非连续的特权真值。  
> 这也是 FG 和 GT 的**唯一差别**，但两者都应绝对正确。  
> 按理性能不应该降；洗白后也应能到 **9/10**。

相对 v256：

> **唯一应有的差别 = 符号/skip 从 GT 连续 residual 换成 FG c2c 三值 RelXY；幅度仍用 15−3t。**

GraspNet 连续抓姿噪声：

> **一直存在，但不在 FG 洗白的解决范围内**（要用 freeze/replay 做实验控制，不是 RelXY 洗白该消的）。

---

## 2. 评测协议（所有 fair-K 共用）

| 项 | 值 |
|---|---|
| Suite / task | `libero_spatial` task 0 |
| Cells | 5 replicates × 2 trials = **10 cells** |
| Pair control | baseline `job-517994`：trial1 fail / trial2 success |
| Treatment | 同 init + **冻结 π0 prefix** |
| Prefix 包 | `…/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json`（SHA 钉死） |
| 指标 | treatment success / cell；`valid_prefix_arm` 才进 rate |
| 工作树 | G17 overlay 必须盖过 editable agilex |

**已冻 vs 未冻（历史坑）：**

- 已冻很久：**code_prefix_replay（π0 / CB0）** — 公平对比成立。
- 直到 v263/v264 才冻：**GraspNet `sample_grasp_pose`**。同 seed 下常「首抓对齐、第 2+ 次分叉」→ 会把 place 对比搞脏。

---

## 3. 版本综述（放对 / 洗白线）

| 版本 | Schema 要点 | 机制（人话） | 真值 / 符号源 | Rate | 角色 |
|---|---|---|---|---|---|
| **v254** | v2.52 tipxy-bandstep | RR + tipxy，3cm band-step（GT residual 滑） | GT 连续 residual | **9/10** | 早期 champion（vector snap 族） |
| **v256** | v2.54 bandwindow | shared-t **15−3t** + tipxy window；**无** residual-vector snap | GT 连续 residual 符号/skip | **9/10** | **洗白对照冠军** |
| **v260** | v2.58 fg-predicate-wash | **只换符号源**：FG RelXY 三值；仍 15−3t；无圆中方/无 no-Z | FG 离散带（底层仍读 `private_poses` c2c） | **7/10** | **最小洗白臂**（命题直接对照） |
| v261 | v2.59 inscribed-no-z | v260 + 圆中方 `/√2` + 取消 Z-abort | 同上 | **4/10** | 脏（方内圆外死锁修复副作用 + horizon） |
| v262 | v2.60 inscribed-no-z-guard | v261 + no-Z tipxy budget / terminated refuse | 同上 | **7/10** | 护栏回稳，仍非最小洗白 |
| v263 | v2.61 + grasp-replay | v262 + 冻 v256 grasp；**overflow=miss** | 同上 + grasp 冻 | valid **5/6**（2 miss） | 公平意图对、实现错 |
| **v264** | v2.62 wash + grasp-replay | **v260 + 冻 grasp；overflow=repeat_last** | 同上 | **7/10**（K=5 收齐） | 公平 grasp 控制下仍未回到 9 |

更早（抓对 / 报告层，非本交接主线但要知道）：

- **v238** 等：ordinal + SGP overload，「抓对」底座；placement near-miss 仍是大头。
- 用户逻辑链（交互对齐）：抓错 → 对的在右 → `sample_grasp_pose` 稳定映射到右边实例（skill 内，不公开 `which=` 教练）。

---

## 4. 进展（到哪一步了）

### 已做成

1. **抓对**：同名消歧 / wrong-grasp ordinal / keep-stock — 主瓶颈已从「抓错」挪到「放对」。
2. **放对有 GT 连续符号上限**：**v254 / v256 = 9/10**。
3. **单位同构已证**：`axis_delta_from_fg_band ≡ −sign(r)·mag`（WITHIN skip）— 「离散谓词更弱」不成立。
4. **最小洗白实现存在 = v260**；但 cell 率 **7/10 ≠ 9/10**。
5. **公平对照资产**：
   - prefix replay（成熟）
   - grasp replay 包：`…/libero_g17_grasp_replay_v256_20260817/`（SHA `ce436389…`）
6. **用户共识**：
   - 主线 = GT→FG 洗白，目标公平复现到 9。
   - Grasp 噪音不在 FG 范围；公平对比必须关死 grasp。
   - v261+ 的圆中方 / no-Z **不是**「唯一差别」命题的一部分。

### 未做成 / 卡住

1. **「只换符号源 ⇒ 仍 9/10」尚未被干净实证撑住**  
   - v260（无 grasp 冻）= 7/10  
   - v264（应有的干净 fair）在跑，用来回答：关死 grasp 后 v260 能否回到 9。
2. Place 路径仍偏 privilege / 硬编码重；端到端可泛化 FG 组件是终局，PG 是消融。
3. PT 锚定尚未接入；`private_poses` 仍是几何 oracle（洗的是**决策接口**，不是 pose 源）。

### v264 状态（2026-08-18 收齐）

- Jobs：`522746–522750`；输出：`…/libero_g17_fair_fg_v264_20260818/`  
- **t1 5/5, t2 2/5, joint 2/5 → 7/10**（gain positive 2/5）  
- 结论：冻 grasp 后仍 ≈v260，**缺口不主要在 GraspNet**；下一步做 v256 vs v264 同 cell place 差分。

---

## 5. 关键突破（带版本锚点）

| 突破 | 锚点 | 一句话 |
|---|---|---|
| Tipxy 把 near-miss 做成可滑的执行门 | **v254** | 相对 v238，放对从 ~6/10 → **9/10** |
| 去掉 vector snap，固定 15−3t 仍 9/10 | **v256** | 洗白对照冠军；符号仍走 GT residue |
| 符号/skip 可换成 FG RelXY 且单位同构 | **v260 代码 + 单测** | 决策同构成立；**rate 未回到 9** |
| 方内圆外死锁机理说清 | **v261 圆中方** | open 用 L2，axis skip 用 L∞ → `/√2`；但是叠加差分 |
| Prefix 跨臂公平 | **prefix_replay_20260815** | π0 一致；fair-K 合法前提 |
| Grasp 必须单独冻 | **v263 失败 / v264 设计** | 证明「一直有噪音」；FG 不管这个 |

---

## 6. 关键问题（带版本锚点）

| 问题 | 锚点 | 状态 |
|---|---|---|
| 最小洗白未到 9/10 | **v260 = 7/10** | 核心开放问题；勿用 v261+ 否定命题 |
| GraspNet 非确定性污染 fair | 全程；**v263 miss** | v264 用 `repeat_last`；仍需看是否够 |
| 圆中方 / no-Z 引入非同构 | **v261/v262** | 不应当成「唯一差别」洗白结论 |
| v263 overflow=miss 搞脏 fair | **v263** | 已弃用该姿势做主结论 |
| 超时 validity 计量洞 | 预存；v262 起 `invalid_timeout` | 评分层，非方法 |
| 放对仍依赖 sim `private_poses` | 全线直至 PT | FG 接口洗了，oracle 未换 |

---

## 7. 新 agent 立刻该做什么

1. **v264 已收齐 = 7/10**：在 **v256 vs v264（或 v260）同 trial** 上对比 tipxy sign/skip 决策，解释相对 9 的 2-cell 缺口。  
2. **不要**再叠圆中方 / no-Z 来「证明洗白」。最小臂 = v260/v264 机制。
3. 改代码优先看：
   - `capx/verification/libero_g17_nearmiss_osc.py`（v256 / wash runner）
   - `capx/verification/libero_g17_grasp_replay.py`（`OVERFLOW=repeat_last`）
   - `capx/verification/libero_g17_takeover.py`（schema v2.54 / v2.58 / v2.62）
   - `scripts/sbatch_smoke_libero_g17_fair_fg_v264.sh` / `submit_…_v264_kseed.sh`
4. 提交习惯：`K=5 bash scripts/submit_libero_g17_fair_fg_<arm>_kseed.sh`；指纹门控勿关。

---

## 8. 一句话给新 agent

> 用户要的是：**GT→FG 洗白到能公平复现 v256 的 9/10**；最小差分是 **符号走 FG c2c 三值、数值仍 15−3t（v260）**；prefix 已公平，grasp 要用 replay 关死（v264）；GraspNet 噪音不是 FG 该解的；圆中方/no-Z 是岔路。当前任务 = **把 v264 公平对比跑完并解释相对 9 的剩余缺口**。
