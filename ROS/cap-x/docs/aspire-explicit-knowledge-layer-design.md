# ASPIRE 显式知识层设计草案

> **2026-08-21 方向修订：** 原草案第 16 节建议先做 `skill + atomic-op`，适合验证底层结构化，却不能直接验证高层抽象对知识爆炸的治理效果。首个研究 MVP 已改为 skill-code-first：先积累 task-specific code，经重复性审计形成 canonical skills，再在每个纵向能力族内提炼 principle tree，并以跨树 graph 表达组合关系。详见 [纵向抽象森林工程蓝图 v0.3](./aspire-upward-abstraction-knowledge-graph-blueprint.md)。原分析仍作为问题诊断与基础设施参考。

> 状态：Design Draft v0.1
> 日期：2026-08-20
> 目标：判断 ASPIRE 的隐式知识分层是否放大了技能库扩张后的检索、去重、剪枝与失效管理问题，并提出一个兼容现有 ASPIRE acquisition/promotion 流程的显式知识层。

## 0. 结论

ASPIRE 的隐式分层不是技能库规模化问题的唯一原因，但它是一个重要的**结构性放大器**。

真正的近因不是“用了 Markdown”或“目录不够深”，而是：

1. 知识单元没有稳定、机器可识别的 identity 和 granularity；
2. `principle / method / strategy / skill / atomic-op` 混合存在于规则文件、runbook、topic skill 和 Python API 中；
3. applicability、dependency、alternative、recovery、contradiction 大多只写在自然语言里；
4. version、status、evidence、freshness 主要存在于 Git、promotion ledger 和正文注释中，而不属于节点本身；
5. 当前检索单位往往是整个 topic 文件，而不是与当前状态匹配的最小知识组合。

因此，当技能库变大时，检索、去重、剪枝和失效管理都只能继续依赖 coordinator/LLM 阅读与判断，维护成本会随正文体积、条目数和跨条目关系快速增长。

Pi 式显式模型可以成为有效破局方向，但前提是它不能停留在 TypeScript 类型声明。完整方案必须同时实现：

> **Typed nodes + typed relations + machine-checkable contracts + evidence ledger + hybrid retrieval + portfolio compilation + lifecycle maintenance + embodied re-validation**

本草案暂称该层为 **AKL（ASPIRE Knowledge Layer）**。

---

## 1. 当前 ASPIRE 的隐式知识结构

本地 ASPIRE 当前存在以下概念层次：

| 隐式层 | 当前载体 | 例子 |
|---|---|---|
| 原则和不变量 | `AGENTS.md`、`CLAUDE.md` | 禁止 simulator ground truth；held-out seed 不得参与调试 |
| 方法和工作流 | experiment `SKILL.md`、`INSTRUCTIONS.md`、coordinator prompt | fix loop、evolutionary search、skill promotion |
| 策略和操作技能 | suite/experiment `skills/*.md` | multi-angle approach、pre-probe IK、SAM3 disambiguation |
| 原子能力 | API reference、Python control API | `get_observation`、`solve_ik`、`move_to_joints` |
| 经验和证据 | `findings.md`、trial trace、promotion JSONL、Git tag | seed success、patch、before/after hash、library snapshot |

这些层次是真实存在的，但没有统一 schema：同一个 Markdown 文件可以同时包含原则、经验性阈值、完整代码模板、失败恢复和 API 调用。文件路径表达的是 suite/experiment/topic scope，不是可计算的语义层。

ASPIRE 已经具备不可丢失的治理能力：

- worker 产生 findings，coordinator 串行入库；
- Stage 1 与 held-out evaluation 隔离；
- success-rate、跨任务泛化和禁止过拟合等 admission gate；
- promotion 前后快照、SHA256、精确 patch 和 append-only ledger；
- Git tag 冻结 library snapshot；
- robot execution trace 和任务 verifier 提供具身证据。

AKL 应扩展这些机制，而不是重写 ASPIRE 的 discovery engine。

---

## 2. 隐式分层如何放大四类问题

### 2.1 检索

当前 LIBERO 共享库主要按 `localize / grasp / transport / manipulation` 路由。这个设计在库小时有效，但文件继续增长后会产生：

- 检索粒度过粗：命中 `transport.md` 等于加载其中大量无关模式；
- query 与知识的匹配主要依赖文件名、description 和 agent 自行阅读；
- 无法先找 high-level strategy，再自动补齐 required skill、validator 和 recovery；
- 一个任务需要多个相互依赖条目时，flat top-k 或整文件加载都难保证 dependency coverage；
- 否定知识、contraindication 和失败恢复容易被正向示例淹没。

因果链是：

```text
隐式知识边界
  -> 检索单位只能近似为文件/段落
  -> description 变宽、正文变长、噪声增加
  -> context 成本和误触发上升
  -> 相关知识被遗漏或错误组合
```

大规模 skill retrieval 并不会仅靠更好的 embedding 自动解决。SkillRet 在 17,810 个真实 skills 上发现通用 retriever 仍有明显缺口；Graph of Skills 则表明 dependency-aware structural retrieval 相比整库加载和简单向量检索有更好的 reward/token trade-off。

### 2.2 去重与合并

当前 coordinator 被要求“merge, don't append”，但“是否重复”没有形式定义：

- 相同策略可能使用不同自然语言、代码变量和 task-specific 常数；
- 两个条目可能共享核心 principle，但 applicability 不同；
- 相同 failure signature 可能存在 alternative repair，而不是 duplicate；
- 相似代码不一定语义等价，相似文本也不一定产生相同 state transition。

没有规范化 contract 和 relation 后，系统只能选择：

1. 过度合并，丢失 guard、反例和 embodiment 差异；或
2. 保守追加，造成 skill inflation 和相互重叠。

因此显式化首先要解决的不是 embedding dedup，而是定义**可比较的语义表面**：goal、preconditions、operation、effects、validators、failure modes 和 compatibility。

### 2.3 剪枝

当前 snapshot/hash 能回答“文件是否改变”，但不能可靠回答“哪一条知识应该退出 active library”。缺少节点级状态会导致：

- 无法区分低频但关键的 safety principle 与真正无用的 skill；
- 无法表示 `deprecated`、`blocked`、`superseded`、`archived`；
- 无法知道某个条目被哪个 strategy、task family 或 recovery path 引用；
- 删除一段 Markdown 前无法做 dependency impact analysis；
- library-size scaling 只能冻结整库快照，不能控制活跃节点集合。

因此剪枝不应直接删除，而应先从 active retrieval index 中软退役，并保留 provenance、证据和 rollback 路径。

### 2.4 失效与重验证

自然语言条目通常没有结构化的：

- `last_validated_at`；
- API、tool、embodiment 和 environment version；
- 验证所覆盖的 task families / seeds；
- 最近使用表现及置信区间；
- upstream dependency；
- 失败严重度和 distribution shift 信号。

因此系统不能自动判断：

- API signature 改变后哪些 skills 失效；
- 一个 skill 在 LIBERO 有效但迁移到 YAM 时需要 adapter；
- 某条经验是否只对旧 perception model 有效；
- 新证据是在补充、反驳还是缩小旧结论的适用范围。

ASPIRE 论文明确把 stale、overly specific、redundant、misleading entries，以及 retrieval、pruning、ranking、re-validation 列为尚未解决的问题。这与本地实现中缺少节点级 lifecycle metadata 是一致的。

### 2.5 因果判断边界

不能声称“只要改成显式类型，四个问题就会消失”。即使 schema 完整，系统仍需要：

- 可靠的 contract/edge extraction；
- state predicate 和 validator 的可执行实现；
- usage/outcome telemetry；
- held-out regression suite；
- relation consistency 和 cycle 检查；
- 对错误抽象、错误边和过时证据的治理。

所以合理结论是：

> 隐式分层不是充分原因，却使四类维护任务缺少机器可操作的对象。显式模型是必要的基础设施，但不是单独充分的解决方案。

---

## 3. 外部证据与设计启示

| 工作 | 与本设计相关的发现 | 设计启示 |
|---|---|---|
| [ASPIRE](https://arxiv.org/abs/2607.00272) | validated repair library 能迁移，但作者明确指出长期 memory management、retrieval、ranking、pruning、re-validation 未解决 | 保留 acquisition/validation，补 organization/governance |
| [SkillRet](https://arxiv.org/abs/2605.05726) | 大规模真实 skill retrieval 仍困难；专门训练的 retriever 明显优于通用模型 | 需要独立 retrieval benchmark、taxonomy 与 hybrid index |
| [Graph of Skills](https://arxiv.org/abs/2604.05333) | dependency-aware graph retrieval 优于整库加载与简单向量检索 | 检索结果应是 bounded subgraph，不是 flat top-k |
| [HiSkill](https://arxiv.org/abs/2607.25853) | skill/AtomicOp nodes 与 decomposition、support、recovery 等 typed edges 改善执行并减少 token | 显式连接 high-level guidance 与 executable API |
| [SkillFlow](https://arxiv.org/abs/2604.17308) | 高 skill usage 不等于高 utility；部分模型随 skill 演化发生退化 | 必须测 negative transfer、bad-skill repair 和 library health |
| [SkillOps](https://arxiv.org/abs/2605.13716) | typed contract 和 library-time maintenance 可缓解 skill technical debt | 把维护做成独立闭环，而非 task-time 临时修补 |
| [SkillNet](https://arxiv.org/abs/2603.04448) | ontology、关系和多维质量评价支持大规模技能组织 | 节点不能只有文本和全局成功率 |
| [Agent Skills specification](https://agentskills.io/specification) | progressive disclosure 解决 context loading，但不定义组合、依赖和生命周期 | 保留 SKILL.md 兼容层，增加 registry/graph 层 |

这些结果支持“显式结构值得做”，但多数仍是新近 preprint，环境和指标不同，不能直接视为 ASPIRE 上的因果证明。必须通过 ASPIRE-specific ablation 验证。

---

## 4. 设计目标与非目标

### 4.1 目标

AKL 应做到：

1. 将 ASPIRE 的异构知识表示为稳定、版本化、可验证的节点；
2. 保留 `principle -> method -> strategy -> skill -> atomic-op` 的语义层，但用 typed graph 而不是严格树；
3. 生成 task-specific、版本锁定、预算受限的 knowledge portfolio；
4. 支持 dependency-complete retrieval，而不是只返回语义相近条目；
5. 让 duplicate、alternative、specialization 和 contradiction 可区分；
6. 让失效、重验证、软退役和回滚成为常规操作；
7. 与现有 findings、promotion ledger、Git snapshots 和 robot traces 兼容；
8. 使所有自动维护决策可解释、可审计、可逆。

### 4.2 非目标

第一版不追求：

- 自动把所有自然语言知识完美形式化；
- 用一个全局分数替代环境 verifier；
- 让 LLM 直接修改 stable nodes；
- 删除 Markdown/SKILL.md 人类接口；
- 在没有 held-out evidence 时自动生成高层 principle；
- 用严格树强迫每个 skill 只有一个 parent；
- 一开始就引入复杂图数据库或在线学习系统。

---

## 5. 总体架构

```text
Actor traces / findings / existing skills / human rules
                         |
                         v
                Candidate Ingestor
                         |
                         v
       Contract + Predicate Normalizer
                         |
                         v
       Typed Registry <-> Evidence Ledger
             |                 ^
             v                 |
  Lexical + Dense + Graph Index|
             |                 |
             v                 |
      Portfolio Compiler       |
             |                 |
             v                 |
  ASPIRE Actor / Evo Search -> Runtime Telemetry
             |
             v
       Library Maintainer
   merge / split / deprecate / revalidate
```

### 5.1 核心组件

#### Candidate Ingestor

输入：

- task `findings.md`；
- successful/failed code and traces；
- 当前 suite/experiment skills；
- 人工规则和 API reference；
- real-robot evidence。

输出不是直接修改 stable library，而是一个或多个 `candidate` node proposals，加上 source pointers。

#### Contract Normalizer

负责：

- canonicalize predicate、tool name、argument type 和 failure code；
- 将 task-specific 实体替换为 typed placeholders；
- 分离 general rule、procedure、code artifact 和 evidence；
- 对不能安全形式化的内容保留 `unstructured_notes`，不伪造精确性。

#### Typed Registry

保存 immutable node revisions、relations 和 active manifest。更新节点意味着创建新 revision；运行中的 portfolio 始终 pin 具体版本。

#### Evidence Ledger

只追加 observation，不直接覆盖结论。它记录每次 retrieval、selection、execution、validator outcome、task context、API/model/environment version 和 trace pointer。

#### Portfolio Compiler

给定 task/state/budget，生成一个紧凑、依赖完整的知识子图：anchor、prerequisite、support、validator、recovery 和 avoid。

#### Library Maintainer

在 library-time 运行，提出 merge、split、specialize、deprecate、block 和 revalidate 建议。MVP 阶段所有结构变更都需要 coordinator 确认。

---

## 6. 数据模型

### 6.1 节点种类

```typescript
type KnowledgeNodeKind =
  | "principle"
  | "method"
  | "strategy"
  | "skill"
  | "atomic-op";

type KnowledgeNodeStatus =
  | "candidate"
  | "validated"
  | "stable"
  | "deprecated"
  | "blocked";
```

不要用含义不清的 `L0-L4` 作为持久字段。UI 可以显示层号，但数据使用语义名称。

### 6.2 公共字段

```yaml
id: navigation.multi-angle-approach
version: 1.2.0
kind: skill
status: validated
title: Multi-angle obstacle-aware approach
summary: Search equivalent approach poses after repeated planner failure near obstacles.
tags: [navigation, recovery, collision]
scope:
  domains: [robotics]
  suites: [behavior-1k]
  embodiments: [r1pro-sim]
relations: []
provenance: {}
validation_policy: {}
```

公共字段最少包括：

- stable `id`；
- immutable `version`；
- `kind/status`；
- title、summary、tags；
- domain/suite/embodiment/API scope；
- typed relations；
- provenance；
- validation policy；
- artifact/reference pointers。

### 6.3 各类型职责

#### Principle

表示跨技能不变量、选择准则和安全边界。

```yaml
statement: Preserve task semantics while searching equivalent geometric realizations.
enforcement: soft
applicability: [...]
invariants: [...]
counterexamples: [...]
```

Principle 不直接执行，也不应由一次成功轨迹生成。

#### Method

表示与具体对象弱耦合的求解方法，例如 trace-guided diagnosis、held-out validation、hypothesis diversification。

```yaml
objective: Diagnose recurring motion-planning failures
procedure:
  - cluster failures by signature
  - inspect primitive traces
  - propose competing hypotheses
  - validate on development seeds
applicability: [...]
```

#### Strategy

表示目标导向、可分解的多阶段方案。

```yaml
goal_pattern: locate-grasp-transport-place
stages:
  - id: observe
    objective: acquire fresh state
    node_ids: [skill.localize-object]
    depends_on: []
  - id: grasp
    objective: secure object
    node_ids: [skill.topdown-grasp, skill.graspnet-grasp]
    depends_on: [observe]
```

#### Skill

表示带 contract 的 operational procedure。建议使用：

```text
P/O/A/V/F = Preconditions / Operation / Artifacts / Validators / Failure modes
```

最小字段：

```yaml
goal: reach a graspable pose near the target
inputs: [...]
outputs: [...]
preconditions: [...]
contraindications: [...]
operation:
  kind: composite
  steps: [...]
effects: [...]
validators: [...]
failure_modes: [...]
hard_invariants: [...]
compatibility: {...}
cost: {...}
```

#### AtomicOp

表示可被执行 harness 直接调用的 API/action template。

```yaml
tool_name: navigate_to_pose
execution_mode: sequential
input_type: NavigatePoseInput
result_type: NavigatePoseResult
api_version: capx.behavior.v2
side_effects: [robot-base-motion]
```

AtomicOp 必须由 runtime/tool registry 生成或验证，不能只靠 LLM 猜测。

### 6.4 关系类型

```typescript
type KnowledgeRelationKind =
  | "is-a"
  | "decomposes-to"
  | "requires"
  | "can-follow"
  | "compatible-with"
  | "alternative-to"
  | "supports"
  | "recovers-with"
  | "contradicts"
  | "supersedes";
```

关系可以带条件：

```yaml
- kind: recovers-with
  target_id: navigation.waypoint-hopping
  when:
    - name: all_rotated_approaches_infeasible
      expected: true
```

层级主要由 `is-a` 和 `decomposes-to` 表达，跨层依赖和恢复使用其他边，因此整体是 typed DAG/graph，不是严格树。

### 6.5 Evidence record

```yaml
evidence_id: ev.behavior-radio.seed31.20260820
node:
  id: navigation.multi-angle-approach
  version: 1.2.0
task:
  suite: behavior-1k
  family: radio-pickup
  task_id: radio-31
context_fingerprint: sha256:...
environment:
  embodiment: r1pro-sim
  api_versions: {navigate_to_pose: v2}
  perception_model: sam3-2026-06
selection:
  portfolio_id: portfolio.radio31.3
  role: recovery
outcome:
  executed: true
  validator_passed: true
  task_success: true
  severity: none
artifacts:
  trace: outputs/.../trace.json
  code: outputs/.../fix_code.py
  video: outputs/.../video.mp4
timestamp: 2026-08-20T10:00:00Z
```

Evidence 与 node revision 分离，避免每次执行都修改知识节点本身。

---

## 7. 存储布局

建议先采用 Git-friendly 文件存储和 SQLite 索引，不在 MVP 引入远程图数据库：

```text
knowledge/
  schema/
    knowledge-node.schema.json
    evidence.schema.json
    portfolio.schema.json
  nodes/
    principle/<id>/<version>.yaml
    method/<id>/<version>.yaml
    strategy/<id>/<version>.yaml
    skill/<id>/<version>.yaml
    atomic-op/<id>/<version>.yaml
  artifacts/
    code/
    prompts/
    references/
  evidence/
    YYYY/MM/<evidence-id>.json
  manifests/
    active.yaml
    deprecated.yaml
    blocked.yaml
  indexes/
    registry.sqlite       # metadata + FTS5 + edge table
    embeddings.*         # optional dense index
  views/
    libero/
      grasp.md
      localize.md
      transport.md
      manipulation.md
```

原则：

- `nodes/` 和 `evidence/` 是 source of truth；
- `views/*.md` 与兼容的 `SKILL.md` 自动生成；
- Git commit/tag 冻结 active manifest 和所有 pinned revisions；
- SQLite/embedding index 可以从 source of truth 重建，不作为不可恢复状态。

---

## 8. Task-time 检索与组合

### 8.1 输入规范化

从 task、当前 observation 和 runtime registry 提取：

- goal pattern；
- entities / object types；
- task state predicates；
- constraints / safety invariants；
- embodiment、API 和 model versions；
- token、latency、action、risk budget。

无法从 observation 验证的 predicate 保持 `unknown`，不能默认 `true`。

### 8.2 分层 seed retrieval

分别检索少量：

- principles/methods：解释约束与求解方式；
- strategies/skills：提供候选方案；
- AtomicOps 不做语义主检索，主要沿 dependency 或 runtime tool match 加入。

每层使用 lexical BM25 + dense similarity；不能只用一个 embedding score。

### 8.3 Graph hydration

从 seed nodes 按预算扩展：

- 必须加入 `requires`；
- 优先加入 validator、support、recovery；
- 加入适用条件满足的 alternatives；
- 加入 contradict/avoid 信息用于 gating；
- 控制最大节点数、边数和 context tokens。

### 8.4 Hard gate

以下条件在排序前排除：

- hard invariant 冲突；
- precondition 明确为 false；
- contraindication 为 true；
- tool/API 不存在或版本不兼容；
- embodiment 不支持；
- node 为 blocked/deprecated 且无显式 override；
- 必需 dependency 无法满足。

安全条件不能被高成功率抵消。

### 8.5 Context-conditioned ranking

在通过 hard gate 的候选上计算：

```text
Score(node | context) =
    w_rel   * relevance
  + w_app   * applicability
  + w_util  * contextual_utility
  + w_trans * transfer_evidence
  + w_fresh * freshness
  + w_comp  * portfolio_compatibility
  - w_cost  * expected_cost
  - w_risk  * severity_weighted_risk
  - w_dup   * marginal_redundancy
```

`contextual_utility` 第一版可使用按 suite/task-family/embodiment 分桶的 Beta-Bernoulli posterior；证据不足时应返回更高不确定性，而不是虚假的精确分数。

### 8.6 Portfolio

检索结果不是 top-k list，而是版本锁定的 portfolio：

```yaml
id: portfolio.behavior-radio-31.3
status: active
entries:
  - role: anchor
    node: {id: strategy.search-grasp-transport, version: 2.0.0}
  - role: prerequisite
    node: {id: skill.fresh-localization, version: 1.4.0}
  - role: recovery
    node: {id: navigation.multi-angle-approach, version: 1.2.0}
  - role: validator
    node: {id: skill.verify-base-distance, version: 1.0.1}
  - role: avoid
    node: {id: principle.no-stale-pose-after-navigation, version: 1.0.0}
edges: [...]
budget: {max_context_tokens: 6000, max_actions: 40}
```

Portfolio compiler 输出：

1. machine-readable plan context；
2. 给 actor 的 compact rendered guidance；
3. 选择、拒绝和 gate 的解释；
4. execution 后的 evidence attribution targets。

---

## 9. 去重、合并与原则提炼

### 9.1 四阶段 dedup

1. **Exact dedup**：内容 hash、相同 source artifact；
2. **Lexical/semantic candidate generation**：BM25、embedding、代码 fingerprint；
3. **Contract comparison**：比较 goal、guards、effects、validator、failure signature；
4. **Evidence-aware decision**：基于实际 outcome 判断 merge、specialize、alternative 或 conflict。

### 9.2 决策类型

| 决策 | 条件 |
|---|---|
| Merge | contract 与 state transition 等价，差异主要是措辞/变量名 |
| Specialize | 核心 operation 相同，但 applicability/embodiment 不同 |
| Alternative | goal/effect 相同，但 operation、cost 或 failure mode 不同 |
| Supersede | 新版本在相同 context 上稳定更优，旧版可软退役 |
| Contradict | 两条知识在重叠 applicability 下给出互斥约束或效果 |
| Keep separate | 证据不足，无法安全合并 |

### 9.3 Principle distillation

只有多个 task families 的稳定节点共享可解释 invariant，并通过 held-out transfer 时，才产生 principle candidate。

Principle promotion gate：

- 至少来自两个 task families；
- 原始 specialization 保留；
- context tokens 或重复 contract 实际减少；
- held-out performance 不下降；
- contraindication、counterexample 和 rollback 明确；
- 可沿 provenance 回到原始 traces。

禁止把“几个文本看起来相似”直接当作因果 principle。

---

## 10. 生命周期、剪枝与失效管理

### 10.1 状态机

```text
candidate -> validated -> stable
    |            |           |
    v            v           v
 blocked      blocked     deprecated
                  \          /
                   -> candidate revision -> validated
```

- `candidate`：来自 findings/import/distillation，尚未达到 admission gate；
- `validated`：通过指定 development/held-out policy；
- `stable`：跨时间、task family 或 embodiment 复验；
- `deprecated`：保留历史和 rollback，但默认不进入 active index；
- `blocked`：存在安全、合规或已知严重错误，hard gate 排除。

状态转换必须引用 evidence 和 reviewer/coordinator decision。

### 10.2 Staleness signals

- tool/API version drift；
- perception/planner/model version drift；
- embodiment/configuration change；
- 超过 validation TTL；
- 最近窗口成功率显著低于历史 posterior；
- contradiction edge 新增；
- validator coverage 不足；
- dependency 被 deprecated/blocked；
- task distribution 改变；
- 长期未命中且被稳定替代。

### 10.3 Revalidation priority

```text
Priority =
    dependency_impact
  * usage_frequency
  * staleness_probability
  * failure_severity
  * uncertainty
  / estimated_revalidation_cost
```

高风险 principle、被大量 strategy 引用的 skill、API 漂移后的 AtomicOp 优先重验证。

### 10.4 Pruning policy

剪枝分三步：

1. **Exclude from default retrieval**；
2. **Deprecate and archive**；
3. **Physical deletion** 仅在 retention policy 到期、无 active references、可从 Git/evidence 恢复时执行。

Safety principle、negative constraint 和罕见但严重的 recovery 不能仅因低使用率被剪掉。

---

## 11. 与现有 ASPIRE 流程的集成

### 11.1 Promotion 流程升级

现有流程：

```text
Actor -> findings.md -> Coordinator -> edit topic Markdown
      -> record patch/hash -> verify -> next task
```

建议流程：

```text
Actor -> findings.md + trace pointers
      -> Candidate Ingestor
      -> candidate nodes/relations/evidence
      -> Coordinator review
      -> contract validation + dedup analysis
      -> active manifest transaction
      -> regenerate Markdown views
      -> record node diff + library diff + hashes
      -> verify -> next task
```

`record_skill_promotion.py` 可逐步增加：

- `candidate_node_ids`；
- `activated_revisions`；
- `deprecated_revisions`；
- `relation_changes`；
- `evidence_ids`；
- schema validation result；
- index rebuild hash；
- generated Markdown diff。

现有 before snapshot、patch、SHA256 和 serial promotion gate 全部保留。

### 11.2 Markdown 兼容层

现有 `.claude/libero/skills/*.md` 不应立即删除。迁移后它们成为生成视图：

- `grasp.md` 渲染匹配 `tags/domain` 的 active skills；
- 每个 subsection 标注 node id/version/status；
- 正文继续供 Claude Code 或普通 Agent Skills host 阅读；
- 人工编辑通过 import command 转换成 candidate revision，避免直接绕过 registry。

### 11.3 AtomicOp 同步

从当前 control API 的 introspection、docstring 和显式 allowlist 生成 AtomicOp candidates，并由测试确认：

- tool 是否存在；
- signature 和 return type；
- side effect；
- blocking/cancellable behavior；
- simulator/real compatibility；
- forbidden API policy。

### 11.4 Real-robot transfer

跨 embodiment 默认只迁移：

- principle；
- method；
- 部分 strategy；

operational skill 必须经过 embodiment adapter 和新的 validator。AtomicOp 不跨 embodiment 直接复用。

---

## 12. 分阶段实施

### Phase 0：建立 flat-library 基线

不改变行为，只记录：

- topic 文件 token 数和增长速度；
- 每任务实际读取的条目；
- retrieval precision/recall；
- duplicate/stale/conflict 人工标注集；
- negative transfer；
- promotion/维护耗时。

没有这一步，后续无法证明显式层真正解决了问题。

### Phase 1：Shadow Registry

- 定义 JSON Schema/Pydantic model；
- 将现有四个 LIBERO topic 文件按 subsection 导入 candidate nodes；
- 注册 AtomicOps；
- 生成 Markdown view 并与原文件做语义/快照对比；
- task-time 仍使用原有整文件加载。

验收：100% active 内容有 source pointer；round-trip 不丢正文；schema/index 可重建。

### Phase 2：显式检索，不自动维护

- lexical + dense seed retrieval；
- typed edge hydration；
- hard gate；
- portfolio rendering；
- 与 current topic routing 做 A/B test。

验收：dependency coverage 提高、输入 token 下降、success 不下降、无新增安全违规。

### Phase 3：Lifecycle 和维护建议

- usage/outcome telemetry；
- freshness 和 API drift；
- duplicate/alternative/contradiction proposals；
- revalidation queue；
- coordinator approve/reject；
- soft deprecation。

验收：能发现注入的 duplicate/stale/bad skills；误剪安全节点为零。

### Phase 4：Principle distillation

- frequent contract/subgraph mining；
- LLM 生成候选 principle 和 counterexamples；
- held-out validation；
- principle + specialization context compression。

验收：在减少 tokens/重复节点的同时不降低 held-out transfer。

### Phase 5：受控自动化

仅对低风险操作自动执行：

- exact duplicate alias；
- index rebuild；
- TTL revalidation scheduling；
- deprecated dependency warning。

merge、principle promotion、blocking 和 physical deletion 继续需要 review。

---

## 13. 验证设计是否真的“破局”

### 13.1 对照组

1. 当前 ASPIRE topic files 全量加载；
2. 当前人工 topic routing；
3. flat BM25 top-k；
4. flat dense top-k；
5. hybrid top-k；
6. typed hierarchy，无 graph hydration；
7. typed graph + portfolio；
8. typed graph + portfolio + lifecycle maintenance。

### 13.2 Library stress matrix

在相同原始技能上构造：

- 10 / 100 / 1,000 / 10,000 nodes；
- 0% / 10% / 30% paraphrase duplicates；
- API drift；
- stale validators；
- conflicting skills；
- early bad skill；
- cross-embodiment transfer；
- multi-skill dependency tasks。

### 13.3 指标

检索：

- Recall@k、NDCG@k；
- dependency/validator/recovery coverage；
- contraindication recall；
- context tokens 和 latency。

执行：

- task success、time/tokens/rollouts to first success；
- invalid composition；
- unmet precondition；
- recovery success；
- severity-weighted safety failures。

维护：

- duplicate detection precision/recall；
- stale detection lead time；
- revalidation yield；
- bad-skill repair latency；
- false deprecation；
- active library size 与 archived ratio；
- performance 随 library size 的单调性/方差。

### 13.4 关键因果实验

要证明“显式结构”而不是“多花了计算”带来提升，应固定：

- 相同 skill content；
- 相同模型；
- 相同 retrieval candidate budget；
- 相同 context token budget；
- 相同 simulator seeds；
- 相同 validation policy。

只改变 representation/retrieval/maintenance layer，并分别消融：

- 去掉 `kind`；
- 去掉 typed edges；
- 去掉 hard gate；
- 去掉 node status/version；
- 去掉 evidence-aware ranking；
- 去掉 library-time maintenance。

---

## 14. 主要风险

| 风险 | 后果 | 缓解 |
|---|---|---|
| Schema false precision | 自然语言知识被强行转成错误 predicate | 允许 unknown/unstructured 字段；validator 验证 |
| 错误 relation | graph hydration 放大错误依赖 | edge confidence、source evidence、cycle/consistency check |
| 过早抽象 principle | 丢失 guard，造成 negative transfer | 多 task family + held-out + counterexample gate |
| 过度剪枝 | 删除罕见 recovery/safety 知识 | soft deprecation、severity-aware retention、rollback |
| Evidence leakage | held-out 结果反向污染 skill | 延续 ASPIRE 的 seed partition 和只追加证据策略 |
| Index/registry drift | rendered Markdown 与 source of truth 不一致 | deterministic generation、hash verification、CI check |
| 维护成本大于收益 | 小库场景复杂化 | shadow mode、规模阈值、按阶段启用 |
| 图过密 | retrieval 退化成全库 hydration | typed edge budget、confidence threshold、role quotas |

---

## 15. 示例：Multi-Angle Approach

```yaml
id: navigation.multi-angle-approach
version: 1.0.0
kind: skill
status: validated
title: Multi-angle approach after boundary planning failures
summary: Rotate the object-relative approach vector and test equivalent target poses.
tags: [navigation, collision-recovery]

goal: reach a graspable observation pose near a localized target

inputs:
  - {name: object_pose, type: pose2d, required: true}
  - {name: robot_pose, type: pose2d, required: true}
outputs:
  - {name: approach_pose, type: pose2d, required: false}

preconditions:
  - name: repeated_planning_error
    expected: true
  - name: target_near_obstacle_boundary
    expected: true
contraindications:
  - name: object_pose_stale
    expected: true

operation:
  kind: composite
  steps:
    - id: sample
      node_id: atomic-op.sample-rotated-approaches
      depends_on: []
      on_failure: abort
    - id: navigate
      node_id: atomic-op.navigate-to-pose
      depends_on: [sample]
      on_failure: recover
    - id: verify
      node_id: skill.verify-approach-distance
      depends_on: [navigate]
      on_failure: replan

effects:
  - predicate: {name: graspable_observation_pose_reached}
    confidence: expected

failure_modes:
  - code: ALL_APPROACHES_INFEASIBLE
    severity: recoverable
    retryable: false
    recovery_node_ids: [navigation.waypoint-hopping]

hard_invariants:
  - {name: no_simulator_ground_truth, expected: true, source: policy}

compatibility:
  embodiments: [r1pro-sim]
  required_tools: [navigate_to_pose, get_observation]
  api_versions: {navigate_to_pose: "2"}

relations:
  - {kind: supports, target_id: strategy.search-grasp-transport}
  - {kind: alternative-to, target_id: navigation.direct-approach}
  - {kind: recovers-with, target_id: navigation.waypoint-hopping}
  - {kind: requires, target_id: skill.fresh-object-localization}

provenance:
  source: trajectory
  source_ids: [behavior-radio-seed31, behavior-radio-seed34]
  created_at: 1787191200
  updated_at: 1787191200

validation_policy:
  minimum_task_families: 2
  minimum_development_trials: 10
  held_out_required_for_stable: true
  ttl_days: 90
```

这个例子展示了显式化带来的直接能力：

- 检索可以基于 failure signature 和 applicability；
- compiler 能自动补齐 localization、navigate、validator 和 recovery；
- API 版本变化能触发失效；
- direct approach 可以保留为 alternative，而不是被覆盖；
- evidence 和 node revision 分离；
- 跨 embodiment 时可复用 principle，但该 operational skill 会被 compatibility gate 拦截。

---

## 16. 建议的第一个实现切片

最小但有研究价值的实现不应一次完成全部五层。建议从以下 vertical slice 开始：

1. 只迁移 LIBERO 的四个 shared skill 文件；
2. 只实现 `skill` 和 `atomic-op` 两种节点，保留未来 `principle/method/strategy` 字段；
3. 实现 `requires / alternative-to / recovers-with / contradicts` 四种边；
4. 实现 candidate/validated/deprecated/blocked；
5. 导入现有 promotion evidence；
6. 实现 BM25 + graph hydration + hard gate；
7. 输出版本锁定 portfolio 和兼容 Markdown view；
8. 在 LIBERO-Pro Long 上对比当前整库加载；
9. 注入 duplicate、stale API 和 conflicting skill 做 maintenance stress test。

这个切片能直接验证三件事：

- 显式最小单元是否降低 context noise；
- dependency/recovery graph 是否提高组合完整性；
- node status/version/evidence 是否使失效和软退役真正可操作。

只有这三项成立后，再引入 principle distillation 和复杂 contextual utility。

---

## 17. 最终建议

采用 Pi 式显式知识模型是可行且有研究价值的，但应把它定位为：

> **ASPIRE discovery engine 之后的 organization、retrieval、composition 和 lifecycle substrate。**

不建议直接把现有 Markdown 标注上 `kind` 就宣称完成分层。真正的破局点是让显式类型产生可测量的系统能力：

- node-level retrieval；
- dependency-complete portfolio；
- contract-aware dedup；
- status/version-aware pruning；
- dependency/API-aware invalidation；
- evidence-driven revalidation；
- held-out-validated principle consolidation。

如果只能优先做一件事，应先实现：

> **`skill + atomic-op` typed registry、evidence ledger、typed dependency/recovery edges 和 portfolio compiler。**

它最接近 ASPIRE 当前数据和执行接口，迁移风险最低，也最容易通过 token、dependency coverage、negative transfer 和 held-out success 做出明确验证。
