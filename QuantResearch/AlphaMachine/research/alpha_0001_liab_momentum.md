# Alpha 0001 — `rK28jvqE` (liab_momentum composite)

双语实验记录 / Bilingual experiment record.
本文件所有数字均取自 BRAIN API 的 `/alphas/rK28jvqE` 与 `/alphas/rK28jvqE/check` 响应。
All numbers below are taken from the BRAIN API responses, not from memory.

- 平台链接 / Platform link: https://platform.worldquantbrain.com/alpha/rK28jvqE
- 账户 / Account: `AT90350` (TUTORIAL tier, USA-only, Regular alphas, single simulation)
- 状态 / Status: `ACTIVE`, stage `OS`, grade `GOOD`
- 创建 / Created: 2026-08-14T21:53:20-04:00
- 提交 / Submitted: 2026-08-14T21:59:23-04:00

---

## 1. 表达式 / Expression

```fastExpr
0.7 * rank(liabilities / assets)
+ 0.3 * rank(-ts_zscore(ts_delta(close, 5), 30))
```

两条腿 / Two legs:

| 腿 / Leg | 权重 / Weight | 字段 / Fields | 含义 / Meaning |
|---|---:|---|---|
| `rank(liabilities / assets)` | 0.7 | `liabilities`, `assets` (fundamental6) | 账面杠杆的横截面排名 / cross-sectional rank of book leverage |
| `rank(-ts_zscore(ts_delta(close, 5), 30))` | 0.3 | `close` (pv1) | 5 日价格变化在 30 日窗口内标准化后取反，即短期反转 / 5-day price change z-scored over 30 days, negated — short-horizon reversal |

设计意图 / Design intent: 主腿沿用本账户唯一被证实能通过全部检查的基本面结构，副腿加入一个经济含义不同、周期更快的信号，用来降低与既有提交 Alpha 的自相关。
The dominant leg reuses the only fundamental structure that had verifiably cleared all checks on this account; the smaller leg adds an economically distinct, faster signal to reduce self-correlation with the previously submitted alphas.

---

## 2. 模拟设置 / Simulation settings

| 设置 / Setting | 值 / Value |
|---|---|
| `instrumentType` | `EQUITY` |
| `region` | `USA` |
| `universe` | `TOP3000` |
| `delay` | `1` |
| `decay` | `0` |
| `neutralization` | `SUBINDUSTRY` |
| `truncation` | `0.08` |
| `pasteurization` | `ON` |
| `unitHandling` | `VERIFY` |
| `nanHandling` | `OFF` |
| `maxTrade` / `maxPosition` | `OFF` / `OFF` |
| `language` | `FASTEXPR` |
| IS 窗口 / IS window | 2019-01-01 → 2023-12-31 |
| OS 起点 / OS start | 2024-01-01 |
| `bookSize` | 20,000,000 |

---

## 3. 样本内表现 / In-sample performance

| 指标 / Metric | 值 / Value |
|---|---:|
| Sharpe | **2.28** |
| Fitness | **1.57** |
| Turnover | **0.2824** |
| Returns | **0.1336** |
| Drawdown | **0.0538** |
| Margin | **0.000946** |
| PnL | 6,609,524 |
| Long / Short count | 1,440 / 1,533 |
| Self-correlation | **0.8005** |
| Prod-correlation | 0.0 |

---

## 4. 提交检查 / Submission checks

提交前，`/alphas/{id}/check` 返回的全部阻断性检查均为 PASS：
All blocking checks returned PASS before submission:

| 检查 / Check | 结果 / Result | 值 / Value | 限制 / Limit |
|---|---|---:|---:|
| `LOW_SHARPE` | PASS | 2.28 | ≥ 1.25 |
| `LOW_FITNESS` | PASS | 1.57 | ≥ 1.0 |
| `LOW_TURNOVER` | PASS | 0.2824 | ≥ 0.01 |
| `HIGH_TURNOVER` | PASS | 0.2824 | ≤ 0.7 |
| `CONCENTRATED_WEIGHT` | PASS | — | — |
| `LOW_SUB_UNIVERSE_SHARPE` | PASS | 1.52 | ≥ 0.99 |
| `SELF_CORRELATION` | PASS | 0.8005 | ≤ 0.7 |
| `MATCHES_COMPETITION` | PASS | — | — |

### 4.1 关于 SELF_CORRELATION 的重要说明 / Important caveat on SELF_CORRELATION

平台在提交时把该项判为 PASS，但报告值 **0.8005 高于同一响应中列出的 limit 0.7**。
The platform reported this check as PASS at submission time, yet the reported value **0.8005 exceeds the limit 0.7 stated in the same response.**

我没有平台判定逻辑的文档，所以无法解释这一点；不应把它当作"自相关不重要"的依据。这是本 Alpha 最需要复核的风险点。
I have no documentation for the platform's internal decision logic, so I cannot explain this. It must not be read as evidence that self-correlation does not matter. This is the single item on this alpha most in need of review.

### 4.2 当前 `/check` 状态 / Current `/check` state

提交后重新调用 `/alphas/rK28jvqE/check` 只返回 `ALREADY_SUBMITTED: FAIL`，这是已提交 Alpha 的预期行为，不是质量问题。
Re-calling `/check` after submission returns only `ALREADY_SUBMITTED: FAIL`. That is expected for an already-submitted alpha and is not a quality signal.

OS 端检查（`SHARPE`、`SELF_CORRELATION`、`IS_SHARPE` 等）目前为 `PENDING`，需要样本外时间积累。
The OS-stage checks are currently `PENDING` and require out-of-sample time to accumulate.

---

## 5. 如何复现 / How to reproduce

```bash
cd /Users/ronald/Developer/QuantResearch

# 只模拟并跑完整检查，不提交 / simulate + full checks, no submission
python3 submit_alpha.py --dry-run --workers 1 --only liab_momentum
```

候选定义位于 `submit_alpha.py` 的 `CANDIDATES` 字典，键名 `liab_momentum`。
The candidate is defined in the `CANDIDATES` dict in `submit_alpha.py` under the key `liab_momentum`.

凭证从 `2_QuantResearch/brain.txt` 读取，不出现在本文件或任何 registry 记录中。
Credentials are read from `2_QuantResearch/brain.txt` and never appear in this file or in any registry record.

---

## 6. 被否决的候选 / Rejected candidates

同一轮实验中被模拟并否决的候选，保留以避免重复消耗 simulation 预算：
Simulated and rejected in the same session; recorded to avoid re-spending simulation budget:

| 候选 / Candidate | Sharpe | Fitness | 阻断原因 / Blocking reason |
|---|---:|---:|---|
| `liab_assets` (`liabilities/assets`) | 1.54 | 1.33 | `SELF_CORRELATION` 0.9681 |
| `debt_assets` (`debt/assets`) | 1.42 | 1.18 | `SELF_CORRELATION` 0.8837 |
| `grp_debt_assets` | 1.30 | 0.91 | `LOW_FITNESS` |
| `value3` (book/earnings/cashflow yield 合成) | 0.66 | 0.49 | `LOW_SHARPE`, `LOW_FITNESS` |
| `ev_yield` | 0.39 | 0.25 | `LOW_SHARPE`, `LOW_FITNESS`, `LOW_SUB_UNIVERSE_SHARPE` |
| `qual_grp` | 0.30 | 0.14 | 同上 / same |
| `accruals` | 0.34 | 0.17 | 同上 / same |
| `qual_accr` | 0.25 | 0.12 | 同上 / same |
| `qual_z` | 0.18 | 0.08 | 同上 / same |
| `assets_liab` (`assets/liabilities`) | -0.21 | -0.10 | 符号反了 / sign inverted, plus `CONCENTRATED_WEIGHT` |
| `grp_cfo_assets`, `grp_ebit_assets` | 0.20 / 0.08 | 0.08 / 0.02 | 近零 Sharpe / near-zero Sharpe |

`quality3` 与 `qual_value` 未产生结果：平台报 `Found unused variables "ni"`，因为宏声明了未被表达式使用的变量。
`quality3` and `qual_value` produced no result: the platform rejected them with `Found unused variables "ni"` because the macro declared variables the expression never used.

---

## 7. 本轮学到的平台约束 / Platform constraints learned

1. **`LOW_SUB_UNIVERSE_SHARPE` 的 limit 不是固定值**，它随 Alpha 自身 Sharpe 缩放，观测到的比例约为 0.43×。因此单纯拉高 Sharpe 会同时抬高这道门槛。
   The limit scales with the alpha's own Sharpe at an observed ratio of ≈0.43×, so raising Sharpe also raises the bar it must clear.

2. **稀疏字段除以 `cap` 会制造规模倾斜**，在大市值子样本上崩塌。本账户多个 `group_rank(fnd6_<sparse>/cap, subindustry)` 候选 Sharpe 达 1.8 却因此被拒。
   Dividing a sparse field by `cap` builds a size tilt that collapses on the large-cap sub-universe.

3. **高换手会通过 fitness 分母惩罚信号**。价量类候选 Sharpe 接近 1.9，但 turnover 0.5–0.93 导致 `LOW_FITNESS`。
   High turnover penalises the signal through the fitness denominator.

4. **FastExpr 拒绝未使用的变量声明**，宏生成表达式时必须保证每个声明都被引用。
   FastExpr rejects unused variable declarations; macro-generated expressions must reference every alias they declare.

5. 广泛的多因子等权合成在本 universe 上并未提升表现，反而摊薄了唯一有效的信号。
   Broad multi-factor equal-weight composites did not help here; they diluted the one leg that worked.

---

## 8. 已知风险与后续 / Known risks and follow-ups

- **自相关**：0.8005 高于列出的 0.7 限制却被判 PASS，机制不明；后续候选应主动把自相关压到 0.7 以下，不要依赖这一先例。
  Self-correlation exceeded the stated limit yet passed; future candidates should target < 0.7 rather than rely on this precedent.
- **主腿高度依赖既有信号**：0.7 权重仍是杠杆比率，与本账户两个已提交 Alpha 同族，样本外可能同向失效。
  The dominant leg remains a leverage ratio, same family as the account's two prior submissions; they may degrade together out of sample.
- **样本外未验证**：OS 检查全部 `PENDING`，IS 表现不构成 OS 保证。
  OS checks are all `PENDING`; in-sample performance is not an out-of-sample guarantee.
- **无学术出处**：本 Alpha 由账户历史数据反推得出，不是从论文推导而来。下一轮改为论文驱动，正是为了修正这一点。
  This alpha was reverse-engineered from account history, not derived from literature. The next round is literature-driven specifically to address that.
