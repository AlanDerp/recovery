# 论文库 / Paper library

本目录记录驱动 alpha 构建的公开学术文献。规则很简单：**每条构造细节都必须来自论文本身，不能来自回忆。**
This directory records the public literature driving alpha construction. One rule: **every construction detail must come from the paper itself, never from recollection.**

## 记录要求 / Requirements per paper

1. 完整引用 + DOI/URL（可公开访问）
2. 核心经济机制与预测方向（符号）
3. **原文的变量构造**：公式、频率、回看窗口、形成/持有期
4. BRAIN 字段与算子映射；不可实现的部分必须明说
5. 与已提交 alpha 的预期重叠
6. **实测结果**——包括否定结果

不要把文献结论当作本账户的样本内证据。文献只提供假设，平台模拟才是证据。
Literature supplies hypotheses, not evidence. Only platform simulation is evidence for this account.

---

## P001 — Ang, Hodrick, Xing & Zhang (2006)

- *The Cross-Section of Volatility and Expected Returns*, **Journal of Finance** 61(1), 259–299
- NBER WP 10852, DOI [10.3386/w10852](https://doi.org/10.3386/w10852)
- 已核验来源 / Verified from: NBER abstract page + full working-paper PDF text

**机制 / Mechanism.** 高特质波动率股票未来收益极低（原文用词 "abysmally low"），且无法由总波动率风险暴露、规模、账面市值比、动量或流动性解释。
High-idiosyncratic-volatility stocks earn "abysmally low" average returns, not explained by aggregate volatility exposure, size, B/M, momentum, or liquidity.

**原文构造 / Construction as stated in the paper.**
- IVOL = `sqrt(var(ε))`，ε 为 **FF-3 模型残差**，用**日频**收益估计
- `L/M/N` 策略：估计期 L 月、等待期 M 月、持有期 N 月
- 主分析为 **1/0/1**：按过去 1 个月日频收益算 IVOL，市值加权分 5 组，持有 1 个月，每月再平衡
- 5−1 组价差 **−1.06%/月**
- **1/1/1** 变体（滞后 1 月，排除同期测量误差）FF-3 alpha 仍为 **−0.82%/月，t = −4.63**
- 论文亦声明同时检验 total volatility，"without using any control for systematic risk"

**BRAIN 映射 / Mapping.** FastExpr 无法做 FF-3 回归。因此采用论文自己承认的两个替代：total volatility，以及用 `group_rank(..., subindustry)` 部分剥离系统性成分。22 个交易日 ≈ 1 个月。
FastExpr cannot run an FF-3 regression, so we use the two proxies the paper itself entertains.

**实测 / Measured on this account** — USA/TOP3000/d1/SUBINDUSTRY/decay0/trunc0.08:

| 候选 | 表达式 | Sharpe | Fitness | 结果 |
|---|---|---:|---:|---|
| `ivol_total` | `-rank(ts_std_dev(returns, 22))` | 0.15 | 0.07 | 否决 |
| `ivol_lagged` | `-rank(ts_delay(ts_std_dev(returns, 22), 22))` | 0.09 | 0.03 | 否决 |
| `ivol_grp` | `-group_rank(ts_std_dev(returns, 22), subindustry)` | 0.04 | 0.01 | 否决 |

**结论 / Verdict.** 三个变体全部近零，且 sub-universe Sharpe 为负。低波动效应在此设定下没有独立信号。
All three variants are flat with negative sub-universe Sharpe. No independent low-volatility edge in this setting.

---

## P002 — Cooper, Gulen & Ion (2024)

- *The use of asset growth in empirical asset pricing models*, **Journal of Financial Economics** 151, 103746
- DOI [10.1016/j.jfineco.2023.103746](https://doi.org/10.1016/j.jfineco.2023.103746)
- 已核验来源 / Verified from: full article PDF text

**机制 / Mechanism.** HXZ (2015) 与 FF5 (2015) 的投资因子表现**关键取决于其构造方式**。两者都用 total asset growth，即 Cooper et al. (2008) 的"账面总资产同比百分比变化"。改用传统投资度量（capex、PPE 增长）时，定价能力**显著下降**。
The performance of the HXZ and FF5 investment factors depends crucially on how the factor is built.

**本文核心发现 / The paper's central result.** 定价信息主要来自**存货与应收账款增长**，而非广义总资产增长：
> "factors based on growth in inventory and accounts receivable contain the bulk of the pricing information in the asset growth factor"

论文亦指出 total asset growth 的四个概念缺陷，其中之一：若企业用现金购置 PPE，总资产增长为零，但投资确实发生了。
The paper also lists conceptual defects of total-asset growth, e.g. cash-financed PPE purchases show zero asset growth despite real investment.

**BRAIN 映射 / Mapping.** `assets`、`inventory`、`receivable` 均已确认存在于 fundamental6。240 交易日 ≈ 1 年。基本面为季度披露，故沿用仓库既有的 `winsorize(ts_backfill(f, 120), std=4)` 预处理。

**实测 / Measured on this account** — 同上设定:

| 候选 | 表达式要点 | Sharpe | Fitness | Sub-uni | 结果 |
|---|---|---:|---:|---:|---|
| `asset_growth` | `-rank(assets / ts_delay(assets, 240) - 1)` | 0.36 | 0.15 | 0.18 (PASS) | 否决 |
| `inv_rec_grp` | 存货+应收增长，subindustry 相对 | 0.23 | 0.06 | 0.20 (PASS) | 否决 |
| `inv_rec_growth` | 存货+应收增长，全市场 rank | 0.02 | 0.00 | −0.04 (FAIL) | 否决 |

**结论 / Verdict.** 符号方向与论文一致（低增长→高收益，取负号后为正 Sharpe），但幅度远不足以通过 `LOW_SHARPE` 1.25 门槛。
值得注意：`asset_growth` 与 `inv_rec_grp` 的 `LOW_SUB_UNIVERSE_SHARPE` **通过**了——说明这类信号在大市值子样本上稳定，只是太弱。
The sign agrees with the paper, but magnitude is far below the 1.25 threshold. Notably both PASS the sub-universe check — these signals are stable across the cap spectrum, merely weak.

---

## 本轮总体结论 / Round-level conclusion

六个论文驱动候选全部作为独立 alpha 被否决，Sharpe 区间 0.02–0.36。
All six literature-driven candidates were rejected standalone, Sharpe 0.02–0.36.

这是一个有信息量的否定结果，不是失败：
This is an informative negative result, not a failure:

1. **这些异象已被充分套利。** AHXZ 发表于 2006，Cooper et al. 原始论文 2008；平台 IS 窗口为 2019-01-01 至 2023-12-31。教科书因子在近期美股大盘上不再提供独立超额收益。
   These anomalies are decades old; the platform's IS window is 2019–2023.

2. **`SUBINDUSTRY` 中性化吃掉了大部分横截面信号。** 这些因子的原始检验多为市值加权分组、无行业中性化。
   The original tests were value-weighted quintile sorts without industry neutralisation.

3. **平台 fitness 公式偏好低换手。** IVOL 类候选换手 0.12–0.13，比基本面比率高一个量级，进一步压低 fitness。

4. **有用的副产品**：`asset_growth` 与 `inv_rec_grp` 的 sub-universe 检查通过，而这正是本账户历史上最常失败的一项。它们作为**弱而稳的正交分量**可能有价值，而非独立 alpha。
   Useful by-product: these two pass the sub-universe check, historically this account's most common failure. Their value may be as weak-but-stable orthogonal components, not standalone alphas.

下一步应当验证第 4 点，而不是继续加大教科书因子的权重。
The next step should test point 4 rather than pushing textbook factors harder.

---

## 后续：把 asset growth 当作弱正交腿测试 / Follow-up: asset growth as a weak orthogonal leg

把 `asset_growth` 与本账户已证实有效的杠杆腿（`liabilities/assets`）按不同权重混合，同样设定下测试：
Blended `asset_growth` with the account's proven leverage leg at different weights, same settings:

| 候选 | 表达式要点 | Sharpe | Fitness | Sub-uni | 结果 |
|---|---|---:|---:|---:|---|
| `liab_ag_30` | 0.7 杠杆 + 0.3 资产增长 | 1.13 | 0.87 | 0.70 (PASS) | 否决：LOW_SHARPE, LOW_FITNESS |
| `liab_ag_20` | 0.8 杠杆 + 0.2 资产增长 | 1.25 | 0.98 | 0.81 (PASS) | 否决：LOW_FITNESS（0.98 差 0.02） |
| `liab_rev_ag` | 0.6 杠杆 + 0.2 反转(rK28jvqE 同款) + 0.2 资产增长 | **1.80** | **1.29** | 1.22 (PASS) | 否决：**SELF_CORRELATION 0.9257 > 0.7** |

**结论 / Verdict.**

1. 单独混入资产增长（不含反转腿）**不够强**：两个权重都过不了 fitness 或 sharpe 门槛。资产增长信号确实太弱，无法独立支撑第二条腿。
   Blending asset growth alone (no reversal leg) isn't strong enough at either weight — it's too weak to carry a second leg by itself.

2. 加回已验证的反转腿后指标很好（Sharpe 1.8、Fitness 1.29、Sub-uni 1.22），但这只是复用了 `rK28jvqE` 的结构，`SELF_CORRELATION` 相应地**正确判定为 FAIL**（0.9257 > 0.7）。
   Re-adding the proven reversal leg produces strong metrics, but that's just reusing rK28jvqE's structure — and SELF_CORRELATION correctly FAILs as a result.

3. 这次的 FAIL 判定（0.9257 明确高于 0.7 且报告为 FAIL）与 `rK28jvqE` 提交时的异常 PASS（0.8005 高于 0.7 却报告为 PASS）形成对照，让人更倾向于相信 **0.7 是真实门槛**，此前那次是特例而非软约束。后续候选应继续以 ≤0.70 为目标，不依赖那次先例。
   This clean FAIL (0.9257, correctly flagged) contrasts with rK28jvqE's anomalous PASS (0.8005) and is evidence the 0.70 limit is real, not soft. Future candidates should still target <=0.70.

4. **本轮无候选通过全部门槛，因此没有提交任何 alpha**——这是自动提交策略在正确工作：宁可空手，不提交未清空自相关的候选。
   No candidate cleared every gate this round, so nothing was submitted — the auto-submit policy working as intended: no submission is preferable to one with unresolved self-correlation.

**对下一轮的启示 / Implication for the next round.** 本账户的杠杆信号家族（含其反转变体）已经饱和：任何重用它的候选都会撞上自相关上限。真正需要的是一个**经济上独立**、不与 `liabilities/assets` 共享驱动因子的信号家族，而不是继续在这条已知曲线上调权重。
This account's leverage-signal family (including its reversal variant) is saturated. What's needed next is a genuinely independent signal family, not further reweighting of the same known curve.

