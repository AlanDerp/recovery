# ASPIRE Knowledge Layer（AKL）v0.1 工程蓝图（已被取代）

> **方向更新：** 本版采用 `skill + atomic-op` 的 bottom-up MVP，不能直接验证“高层抽象能否抑制知识库爆炸”。当前实施基线已改为 [ASPIRE Skill-code-first 纵向抽象森林工程蓝图 v0.3](./aspire-upward-abstraction-knowledge-graph-blueprint.md)：先积累 skill code，经重复性审计形成 canonical skills，再在纵向能力族内提炼 principle tree。本文件仅保留为历史设计和底层基础设施参考。

> 文档状态：Superseded Blueprint v0.1
> 日期：2026-08-21
> 设计依据：[ASPIRE 显式知识层设计草案](./aspire-explicit-knowledge-layer-design.md)
> 对齐代码：ASPIRE `680bad4df1dabe2463e20069e16f6978630fe1d7`（本地 HEAD 与 `origin/main` 一致）
> 首个落地场景：LIBERO 共享技能库
> 决策口径：先建立可验证的最小闭环，不在 v0.1 一次性实现五层本体和自动知识蒸馏

---

## 0. 执行摘要

首版 AKL 在 ASPIRE 现有的技能发现、Stage 1 验证、coordinator promotion、held-out 隔离和 Git snapshot 机制之后，增加一个显式的知识组织与运行时交付层：

```text
findings / API source
        |
        v
candidate node + evidence
        |
        v
schema / relation / contract gate
        |
        v
versioned registry --------------+
        |                         |
        v                         v
derived SQLite/FTS index     generated Markdown views
        |
        v
retrieve -> hydrate -> gate -> rank -> compile
        |
        v
pinned task portfolio -> ASPIRE actor
        |
        v
usage/outcome evidence -> lifecycle audit
```

v0.1 只实现：

- 两类知识节点：`skill`、`atomic-op`；
- 四类显式关系：`requires`、`alternative-to`、`recovers-with`、`contradicts`；
- 四个受治理状态：`candidate`、`validated`、`deprecated`、`blocked`；
- 基于 SQLite FTS5/BM25 的 lexical retrieval；
- 关系扩展、硬约束过滤、确定性排序和 token budget；
- task-specific、精确版本锁定的 portfolio；
- 现有四个 LIBERO Markdown 文件的无损迁移与兼容视图；
- promotion evidence 导入、AtomicOp 同步、失效检测和软退役；
- `off -> shadow -> retrieve -> enforce` 四档渐进发布。

v0.1 明确不做 dense embedding、图数据库、自动修改 stable knowledge、自动生成 principle/method/strategy、跨机器人自动迁移，也不让 YAML 中的代码直接进入执行命名空间。

成功标准不是“建成了 schema”，而是同时满足：

1. LIBERO 当前四个技能文件可追溯地迁移、可重建为兼容视图；
2. Actor 获得的是 dependency-complete 的小 portfolio，而非四个整文件；
3. 重复、过期 API 和显式冲突能被 CI/审计发现；
4. LIBERO-Long-Pro held-out success 不劣于 flat-library 基线；
5. 输入 token 和错误组合率显著下降；
6. 任意时刻可切回现有 Markdown 路径，不影响实验协议。

---

## 1. 已确认的 ASPIRE 工程边界

本蓝图基于本地代码实查，而不是抽象假设。

### 1.1 运行与依赖

- Python：`>=3.10, <3.13`；
- 已有依赖：Pydantic v2、PyYAML、pytest、ruff、mypy；
- Python 标准库提供 SQLite，无需首版引入数据库服务；
- 主包扫描范围包含 `aspire.sim.*`，运行时代码适合放入 `aspire/sim/cap/knowledge/`；
- CI 测试入口为 `aspire/sim/tests`。

### 1.2 当前共享库

权威的人类可读共享库位于 `aspire/sim/.claude/libero/skills/`：

| 文件 | 正文行数 | 当前混合内容 |
|---|---:|---|
| `localize.md` | 165 | prompt、定位 helper、消歧、放置高度经验 |
| `grasp.md` | 154 | 完整模板、grasp 选择、阈值、几何 helper |
| `transport.md` | 80 | waypoint、IK conditioning、safe transit、release |
| `manipulation.md` | 32 | articulated object、push/slide、参数和注意事项 |

四个正文文件共 431 行、约 15.9 KiB。当前 actor 被要求读取“相关共享技能”，library-size scaling 路径则明确 `cat` 四个整文件。随着库增长，文件是检索单元，也是 context 单元。

### 1.3 当前 promotion 能力

`scripts/libero/record_skill_promotion.py` 已提供：

- per-suite 串行 promotion gate；
- 修改前文件快照；
- before/after SHA256；
- exact unified patch；
- append-only JSONL ledger；
- `begin / finish / verify` 生命周期；
- findings 路径及其 hash。

AKL 必须扩展这个审计链，不能绕过或降低它。

### 1.4 AtomicOp 边界

ASPIRE 的实际可调用原子能力来自 `ApiBase.functions()`。LIBERO 的能力由 `FrankaLiberoApiReducedSkillLibrary.functions()` 和其父类暴露，例如：

- perception：`get_observation`、`segment_sam3_text_prompt`；
- planning/control：`solve_ik`、`goto_pose`、`move_to_joints`；
- grasp：`plan_grasp`、`select_top_down_grasp`；
- geometry：`mask_to_world_points`、`decompose_transform`；
- effectors：`open_gripper`、`close_gripper`。

实例化该 API 会初始化视觉/抓取/IK 组件，因此 CI 的 AtomicOp 同步不能依赖构造真实环境。v0.1 采用静态 AST 同步，并提供真实环境中的可选 runtime verification。

### 1.5 实验不可破坏约束

- Stage 1 development evidence 可以驱动知识变更；
- held-out seeds 只做评估，不能反向修改共享库；
- snapshot/worktree 必须冻结知识版本；
- 禁止 simulator ground-truth API；
- 现有 GPU ownership、task lifecycle 和污染防护不因 AKL 改变。

---

## 2. MVP 边界与核心决策

### 2.1 范围

v0.1 处理一个完整垂直切片：

```text
LIBERO four Markdown files
        +
FrankaLibero API surface
        +
promotion ledgers
        |
        v
skill/atomic-op registry
        |
        v
retrieval + portfolio
        |
        v
LIBERO fix-loop and Long-Pro shadow/retrieve evaluation
```

只有通过该切片验证后，才打开 `strategy`，再考虑 `method` 与 `principle`。数据模型预留五种 kind，但 v0.1 写入 gate 只允许 `skill` 和 `atomic-op`。

### 2.2 关键 ADR

| ADR | 决策 | 理由 |
|---|---|---|
| ADR-001 | YAML/JSONL/Git 是权威数据，SQLite 是可删除、可重建索引 | 保留 diff、review、snapshot、回滚能力 |
| ADR-002 | 节点 revision 不可变，active manifest 选择当前版本 | 避免原地改写历史含义 |
| ADR-003 | 生命周期用 append-only event 表达，节点的 effective status 由 materializer 计算 | 状态变化可审计、可回放 |
| ADR-004 | relation 是一等对象，不把边埋在自然语言或仅放在节点数组里 | 支持边证据、边失效和冲突审查 |
| ADR-005 | portfolio 中所有引用精确 pin 到版本和 content hash | 实验可复现 |
| ADR-006 | v0.1 只用 lexical retrieval + graph hydration | 先验证结构收益，避免 embedding 变量混入因果实验 |
| ADR-007 | contract 使用受限 predicate DSL，不允许任意 Python 表达式 | 安全、确定性、可静态分析 |
| ADR-008 | Markdown 是生成视图；过渡期保留人工编辑入口并检测漂移 | 与现有 prompt、Agent Skills 和人类工作流兼容 |
| ADR-009 | 节点中的 code 仅为示例/模板，不自动 `exec` | 执行能力只来自已注册 AtomicOp/API |
| ADR-010 | 先 shadow，再 retrieve，最后 enforce | 分离数据建模错误与运行时行为风险 |

### 2.3 非目标

- 不引入 Neo4j、Postgres、外部向量服务；
- 不把每个 Markdown 段落机械切成一个 skill；
- 不以文本相似度自动合并节点；
- 不以全局 success rate 取代 task verifier；
- 不让 held-out evidence 改变 active manifest；
- 不在 v0.1 做跨 LIBERO/real-robot 的自动 status 继承；
- 不把现有 `cap/skills/library.py` 的动态 `exec` 机制直接升级为 AKL runtime。

---

## 3. 目标目录结构

建议在 ASPIRE 中形成下列布局：

```text
aspire/sim/
├── cap/
│   └── knowledge/                    # 可安装的 AKL runtime package
│       ├── __init__.py
│       ├── __main__.py               # python -m aspire.sim.cap.knowledge
│       ├── models.py                 # Pydantic domain models
│       ├── ids.py                    # ID/version/ref parsing
│       ├── predicates.py             # 受限 contract DSL evaluator
│       ├── repository.py             # YAML/JSONL repository
│       ├── materialize.py            # effective state + active manifest
│       ├── validation.py             # schema/graph/semantic gates
│       ├── index.py                  # SQLite schema/rebuild
│       ├── retrieval.py              # FTS seed retrieval
│       ├── hydration.py              # relation closure
│       ├── gating.py                 # environment/contract hard gates
│       ├── ranking.py                # deterministic scoring
│       ├── portfolio.py              # compile and serialize portfolio
│       ├── evidence.py               # append/query evidence events
│       ├── lifecycle.py              # stale/conflict/deprecation audit
│       ├── render.py                 # Markdown compatibility views
│       ├── atomic_sync.py            # AST/API sync
│       ├── promotion.py              # promotion transaction adapter
│       ├── telemetry.py              # retrieval/use/outcome events
│       └── cli.py                    # thin command layer
│
├── knowledge/                        # Git-tracked source of truth
│   ├── README.md
│   ├── schema-version.txt
│   ├── manifests/
│   │   ├── libero-active.yaml
│   │   └── libero-baseline-lock.yaml
│   ├── nodes/
│   │   ├── skill/<logical-id>/<version>.yaml
│   │   └── atomic-op/<logical-id>/<version>.yaml
│   ├── relations/<relation-id>/<version>.yaml
│   ├── evidence/
│   │   ├── promotions.jsonl
│   │   ├── validations.jsonl
│   │   └── lifecycle.jsonl
│   ├── migrations/
│   │   └── libero-v0-import-map.yaml
│   └── fixtures/
│       └── task-contexts/
│
├── .claude/libero/skills/             # v0.1 兼容视图；最终由 render 生成
├── scripts/knowledge/
│   ├── import_libero_skills.py
│   ├── sync_atomic_ops.py
│   ├── build_index.py
│   ├── render_views.py
│   ├── audit_library.py
│   └── stress_library.py
├── tests/knowledge/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   └── fixtures/
└── outputs/akl/                       # Git ignored 的派生产物
    ├── index/akl.sqlite3
    ├── portfolios/<run>/<task>.{json,md}
    ├── reports/
    └── telemetry/*.jsonl
```

规则：

- `knowledge/` 中只放可 review 的结构化事实，不放 SQLite、缓存或模型输出；
- `outputs/akl/` 可随时删除并从权威层重建；
- `.claude/libero/skills/*.md` 在 shadow 期仍保持现状，迁移验证通过后改为生成视图；
- 任何生成视图都带 source manifest hash 和“请勿直接编辑”头；
- 所有 path 解析都限制在显式 root 内，禁止 `..` 逃逸。

---

## 4. 领域模型

### 4.1 枚举

```python
KnowledgeNodeKind = Literal[
    "principle", "method", "strategy", "skill", "atomic-op"
]

MvpWritableKind = Literal["skill", "atomic-op"]

KnowledgeStatus = Literal[
    "candidate", "validated", "stable", "deprecated", "blocked"
]

MvpLifecycleStatus = Literal["candidate", "validated", "deprecated", "blocked"]

RelationKind = Literal[
    "requires", "alternative-to", "recovers-with", "contradicts"
]

EvidenceKind = Literal[
    "promotion", "validation", "execution", "failure",
    "api-sync", "lifecycle", "human-review"
]
```

`stable` 被保留但 v0.1 不自动授予，也默认没有写入命令；它需要后续跨任务、跨时间或跨 embodiment 的治理策略。v0.1 的 retrieve/enforce 默认只检索 `validated`；shadow 可额外显示 `candidate`，`deprecated` 和 `blocked` 只参与审计。

### 4.2 ID 与版本规则

逻辑 ID 使用小写 dotted namespace：

```text
skill.libero.perception.multi-prompt-disambiguation
skill.libero.motion.safe-lateral-escape
atomic-op.libero.segment-sam3-text-prompt
atomic-op.libero.solve-ik
relation.libero.safe-lateral-escape.requires.solve-ik
```

约束：

- 正则：`^[a-z0-9]+(?:[.-][a-z0-9]+)*$`；
- ID 表示语义身份，不包含版本；
- 版本使用 SemVer；
- 文案、证据引用修正但不改变 contract：patch；
- contract 兼容扩展：minor；
- goal/effect、调用边界或 applicability 非兼容改变：major；
- 同一 `id@version` 的 canonical content hash 永远不变；
- portfolio 只使用 `id@exact-version#sha256`；
- manifest 可使用版本约束，但 materialize 后必须解析成 exact version。

### 4.3 NodeRevision

```python
class NodeRevision(BaseModel):
    schema_version: Literal[1]
    id: str
    kind: KnowledgeNodeKind
    version: str
    title: str
    summary: str
    body: str | None = None
    tags: list[str] = []
    scope: Scope
    contract: Contract
    artifacts: list[ArtifactRef] = []
    provenance: Provenance
    created_at: datetime
    created_by: ActorRef
```

节点 revision 不直接保存可变的“当前成功率”或“最后使用时间”。这些值来自 evidence materialization，防止每次运行都改 Git 文件。

### 4.4 Scope

```python
class Scope(BaseModel):
    suites: list[str] = []
    task_families: list[str] = []
    embodiments: list[str] = []
    environments: list[str] = []
    api_sets: list[str] = []
    perception_backends: list[str] = []
    labels: dict[str, str] = {}
```

空列表的语义是“未声明限制”，不是“适用于所有”。只有显式 `labels.global: "true"` 才可表达全局声称；v0.1 import 不自动生成全局 claim。

### 4.5 Contract 与 predicate DSL

```python
class Contract(BaseModel):
    goal: str
    inputs: list[PortSpec] = []
    outputs: list[PortSpec] = []
    preconditions: Predicate = Always()
    effects: list[Effect] = []
    validators: list[ValidatorSpec] = []
    failure_modes: list[FailureMode] = []
    contraindications: Predicate = Never()
    token_cost_hint: int | None = None
```

Predicate 只允许以下节点：

```yaml
all:
  - fact: api.available
    op: contains
    value: solve_ik
  - fact: task.object_count
    op: gte
    value: 2
  - any:
      - fact: scene.has_articulated_object
        op: eq
        value: true
      - fact: task.language
        op: regex
        value: "drawer|microwave"
```

允许操作：`eq`、`ne`、`in`、`contains`、`exists`、`gte`、`lte`、`regex`。组合器：`all`、`any`、`not`。规则：

- 无 `eval`、无函数调用、无文件读取；
- 未知 fact 产生 `unknown`，不是自动 `true`；
- hard precondition 遇到 `false` 或 `unknown` 时默认拒绝；
- soft ranking feature 可以显式选择如何处理 `unknown`；
- regex 有长度和执行时间限制；
- predicate 经过 canonical JSON 序列化后参与 hash。

### 4.6 SkillNode

`skill` 表达可复用的状态转换方案，而不是任意知识段落。

```yaml
schema_version: 1
id: skill.libero.motion.safe-lateral-escape
kind: skill
version: 1.0.0
title: Safe lateral escape before lift
summary: Move laterally out of local obstacles before the vertical lift.
tags: [motion, collision-avoidance, transit]
scope:
  suites: [libero]
  embodiments: [franka]
  api_sets: [FrankaLiberoApiReducedSkillLibrary]
contract:
  goal: Move a grasped object from clutter into a collision-safe lift corridor.
  preconditions:
    all:
      - {fact: state.object_grasped, op: eq, value: true}
      - {fact: api.available, op: contains, value: solve_ik}
  effects:
    - fact: state.clearance_for_vertical_lift
      value: true
  validators:
    - type: trace-observation
      signal: close_gripper.gripper_width
      op: gt
      value: 0.06
  failure_modes:
    - id: ik-unreachable
      signature: solve_ik returned None for the lateral waypoint
  contraindications:
    any:
      - {fact: scene.lateral_clearance_known, op: eq, value: false}
artifacts:
  - type: markdown-snippet
    path: knowledge/artifacts/libero/safe-lateral-escape.md
provenance:
  imported_from: .claude/libero/skills/transport.md
  source_anchor: "Safe Transit (Lateral Escape Before Lift)"
created_at: 2026-08-21T00:00:00Z
created_by: {type: migration, id: libero-v0-import}
```

v0.1 validator 要求每个 skill 至少有：`goal`、一个 scope 维度、precondition、effect 或 validator 之一、provenance。

### 4.7 AtomicOpNode

`atomic-op` 映射 ASPIRE 中实际暴露的 API function：

```yaml
schema_version: 1
id: atomic-op.libero.solve-ik
kind: atomic-op
version: 1.0.0
title: solve_ik
summary: Solve robot joint configuration for a target end-effector pose.
scope:
  suites: [libero]
  embodiments: [franka]
  api_sets: [FrankaLiberoApiReducedSkillLibrary]
contract:
  goal: Convert a reachable Cartesian pose into joint angles.
  inputs:
    - {name: position, type: array[3]}
    - {name: quaternion_wxyz, type: array[4]}
  outputs:
    - {name: joints, type: "array[7] | None"}
  preconditions: {fact: api.available, op: contains, value: solve_ik}
  effects: []
  validators:
    - type: return-contract
      any_of:
        - {type: null}
        - {type: array, shape: [7]}
  failure_modes:
    - {id: unreachable, signature: "return is None"}
  contraindications: {fact: api.forbidden, op: contains, value: solve_ik}
binding:
  api_set: FrankaLiberoApiReducedSkillLibrary
  function: solve_ik
  source_path: aspire/sim/cap/integrations/franka/libero_reduced.py
  source_symbol: FrankaLiberoApiReduced.solve_ik
  signature_hash: sha256:...
provenance:
  generated_by: atomic-sync-v1
created_at: 2026-08-21T00:00:00Z
created_by: {type: tool, id: atomic-sync}
```

AtomicOp 的 active status 在 API signature、暴露集合或 forbidden policy 改变时必须重新计算。

### 4.8 RelationRevision

```python
class RelationRevision(BaseModel):
    schema_version: Literal[1]
    id: str
    kind: RelationKind
    version: str
    source: VersionedRef
    target: VersionedRef
    guard: Predicate = Always()
    rationale: str
    provenance: Provenance
    created_at: datetime
    created_by: ActorRef
```

四种关系的语义：

| 关系 | 方向 | 运行时行为 |
|---|---|---|
| `requires` | source → target | source 入选时必须补齐 target；缺失则 source 被拒绝 |
| `alternative-to` | 对称 | 同一 portfolio 默认择优保留一个，除非 recovery policy 需要备选 |
| `recovers-with` | source → target | source 对应 failure signature 命中时补齐 target |
| `contradicts` | 对称或带 guard | 两者 guard 同时成立时不得共同进入 active portfolio |

关系规则：

- `requires` 不允许形成环；
- `alternative-to` 和无 guard 的 `contradicts` 必须规范化为排序后的无向 pair；
- dangling ref 是 CI hard failure；
- relation 的 source/target 使用版本约束，portfolio materialization 解析为精确版本；
- edge 也有 evidence 和 lifecycle，不能假设节点有效就意味着边永久有效。

### 4.9 EvidenceEvent

```python
class EvidenceEvent(BaseModel):
    schema_version: Literal[1]
    event_id: UUID
    kind: EvidenceKind
    subject: VersionedRef
    relation: VersionedRef | None = None
    occurred_at: datetime
    source: EvidenceSource
    partition: Literal["development", "held-out", "production", "static"]
    context: dict[str, JsonValue]
    outcome: EvidenceOutcome
    artifacts: list[ArtifactRef]
    content_hash: str
```

必须保留的字段：suite、task、seed/trial、API set/version、environment version、success/failure、failure signature、源文件 hash。held-out evidence 可进入 evaluation report，但 materializer 必须禁止它触发 node content 或 active status 的自动变化。

### 4.10 LifecycleEvent

```yaml
schema_version: 1
event_id: 019...
subject: skill.libero.motion.safe-lateral-escape@1.0.0
transition: validated->deprecated
reason_code: api-binding-stale
reason: solve_ik signature no longer matches the recorded AtomicOp.
superseded_by: null
occurred_at: 2026-08-21T00:00:00Z
actor: {type: human, id: coordinator}
evidence_refs: [019...]
```

有效状态由“初始 candidate + 按时间回放 lifecycle events”得到。非法 transition、时间倒序和 unknown subject 都是 hard error。

### 4.11 ActiveManifest

```yaml
schema_version: 1
manifest_id: libero-active
revision: 1
generated_at: 2026-08-21T00:00:00Z
nodes:
  skill.libero.motion.safe-lateral-escape: 1.0.0
  atomic-op.libero.solve-ik: 1.0.0
relations:
  relation.libero.safe-lateral-escape.requires.solve-ik: 1.0.0
policy:
  retrievable_statuses: [validated]
  allow_candidate_in_shadow: true
  deny_deprecated: true
  deny_blocked: true
content_hash: sha256:...
```

Manifest 更新必须走 promotion transaction，并在 commit 前验证 graph closure。

### 4.12 TaskContext 与 Portfolio

```python
class TaskContext(BaseModel):
    task_id: str
    suite: str
    task_language: str
    task_family: str | None
    embodiment: str
    environment: str
    api_set: str
    available_atomic_ops: list[str]
    forbidden_atomic_ops: list[str]
    observed_facts: dict[str, JsonValue]
    failure_signatures: list[str] = []
    token_budget: int = 2400
    manifest: str = "libero-active"

class Portfolio(BaseModel):
    schema_version: Literal[1]
    portfolio_id: str
    task_context_hash: str
    manifest_hash: str
    compiler_version: str
    source_epoch: datetime
    selected: list[PinnedNode]
    relations: list[PinnedRelation]
    exclusions: list[Exclusion]
    dependency_coverage: float
    estimated_tokens: int
    content_hash: str
```

Portfolio 同时输出 JSON（机器审计）和 Markdown（actor context）。Markdown 只含入选节点、必要关系、适用条件、失败恢复和 AtomicOp 签名。

`source_epoch` 来自冻结 manifest/Git source epoch，而不是编译时钟；真实编译时间只写 telemetry，不进入 portfolio 内容，确保同输入可 byte-for-byte 重建。

---

## 5. 权威存储、索引与哈希

### 5.1 Canonical serialization

- YAML 只作为可读语法；
- 读取后先经过 Pydantic normalization；
- hash 输入为 UTF-8 canonical JSON：key 排序、无多余空白、datetime 转 UTC ISO-8601、list 保序；
- hash 采用 SHA256；
- YAML comment 不影响语义 hash；
- artifact 文件分别 hash，并将 hash 写入 node canonical content。

### 5.2 SQLite 是派生视图

默认路径：`outputs/akl/index/akl.sqlite3`。每次 rebuild 写入临时文件，完成 `PRAGMA integrity_check` 后原子替换。

核心 DDL：

```sql
CREATE TABLE metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE nodes (
  id TEXT NOT NULL,
  version TEXT NOT NULL,
  kind TEXT NOT NULL,
  effective_status TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  body TEXT,
  scope_json TEXT NOT NULL,
  contract_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  source_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  active INTEGER NOT NULL,
  token_estimate INTEGER NOT NULL,
  PRIMARY KEY (id, version),
  UNIQUE (content_hash)
);

CREATE TABLE relations (
  id TEXT NOT NULL,
  version TEXT NOT NULL,
  kind TEXT NOT NULL,
  source_id TEXT NOT NULL,
  source_version TEXT NOT NULL,
  target_id TEXT NOT NULL,
  target_version TEXT NOT NULL,
  guard_json TEXT NOT NULL,
  rationale TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  active INTEGER NOT NULL,
  PRIMARY KEY (id, version)
);

CREATE TABLE evidence_summary (
  subject_id TEXT NOT NULL,
  subject_version TEXT NOT NULL,
  partition TEXT NOT NULL,
  successes INTEGER NOT NULL DEFAULT 0,
  failures INTEGER NOT NULL DEFAULT 0,
  distinct_tasks INTEGER NOT NULL DEFAULT 0,
  last_observed_at TEXT,
  last_validated_at TEXT,
  failure_signatures_json TEXT NOT NULL,
  PRIMARY KEY (subject_id, subject_version, partition)
);

CREATE TABLE atomic_bindings (
  node_id TEXT NOT NULL,
  node_version TEXT NOT NULL,
  api_set TEXT NOT NULL,
  function_name TEXT NOT NULL,
  source_symbol TEXT NOT NULL,
  signature_hash TEXT NOT NULL,
  currently_exposed INTEGER NOT NULL,
  PRIMARY KEY (node_id, node_version)
);

CREATE VIRTUAL TABLE node_fts USING fts5(
  id UNINDEXED,
  version UNINDEXED,
  title,
  summary,
  body,
  tags,
  goal,
  failure_modes,
  tokenize='unicode61 remove_diacritics 2'
);

CREATE INDEX idx_nodes_active_kind_status
  ON nodes(active, kind, effective_status);
CREATE INDEX idx_rel_source ON relations(source_id, source_version, kind);
CREATE INDEX idx_rel_target ON relations(target_id, target_version, kind);
CREATE INDEX idx_atomic_api_fn ON atomic_bindings(api_set, function_name);
```

若运行环境 SQLite 不支持 FTS5，build 立即失败并给出诊断；不静默退化成不同排序算法，以保证实验一致性。

### 5.3 Rebuild invariant

给定相同 Git tree、manifest 和 CLI 版本：

- active exact refs 相同；
- canonical content hash 相同；
- index logical dump hash 相同；
- 任何加载顺序都不改变检索 tie-break；
- 派生数据库时间戳不进入 logical hash。

---

## 6. 模块接口

### 6.1 Repository

```python
class KnowledgeRepository(Protocol):
    def get_node(self, ref: VersionedRef) -> NodeRevision: ...
    def list_nodes(self, *, kind=None, logical_id=None) -> list[NodeRevision]: ...
    def get_relation(self, ref: VersionedRef) -> RelationRevision: ...
    def iter_evidence(self, *, subject=None, partition=None) -> Iterator[EvidenceEvent]: ...
    def append_evidence(self, event: EvidenceEvent) -> None: ...
    def append_lifecycle(self, event: LifecycleEvent) -> None: ...
    def load_manifest(self, manifest_id: str) -> ActiveManifest: ...
    def write_revision(self, revision: NodeRevision | RelationRevision) -> Path: ...
```

写入必须使用同目录 temporary file + fsync + rename；发现已有相同 ref 但不同 hash 时拒绝。

### 6.2 Validator

```python
class RegistryValidator:
    def validate_revision(self, value) -> ValidationReport: ...
    def validate_manifest(self, manifest) -> ValidationReport: ...
    def validate_graph(self, graph) -> ValidationReport: ...
    def validate_artifacts(self) -> ValidationReport: ...
    def validate_api_bindings(self, snapshot) -> ValidationReport: ...
```

报告区分 `error / warning / info`。CI 中 error 阻断；warning 必须可被稳定排序和机器解析。

### 6.3 Materializer

```python
class RegistryMaterializer:
    def resolve_manifest(self, manifest_id: str) -> ResolvedManifest: ...
    def effective_status(self, ref: VersionedRef) -> KnowledgeStatus: ...
    def build_graph(self, manifest: ResolvedManifest) -> KnowledgeGraph: ...
```

版本解析采用确定性 SemVer，禁止浮动 latest 进入 portfolio。

### 6.4 Retriever

```python
class Retriever:
    def search(
        self,
        query: RetrievalQuery,
        *,
        limit: int = 24,
    ) -> list[ScoredCandidate]: ...
```

检索输入包括 task language、task family、observed facts、failure signatures 和允许 kind。输出保留每项 BM25 命中字段与分数，便于解释。

### 6.5 Hydrator 与 Gate

```python
class GraphHydrator:
    def expand(
        self,
        seeds: list[ScoredCandidate],
        graph: KnowledgeGraph,
        *,
        max_depth: int = 3,
        max_nodes: int = 64,
    ) -> HydratedCandidates: ...

class ContractGate:
    def evaluate(
        self,
        candidate: HydratedNode,
        context: TaskContext,
    ) -> GateDecision: ...
```

Hydrator 对 `requires` 做闭包，对 failure-signature 匹配的 `recovers-with` 做条件扩展；`alternative-to` 和 `contradicts` 交由组合阶段解决。

### 6.6 Ranker 与 PortfolioCompiler

```python
class DeterministicRanker:
    def score(self, candidate, context, evidence) -> ScoreBreakdown: ...

class PortfolioCompiler:
    def compile(
        self,
        context: TaskContext,
        *,
        manifest_id: str,
        token_budget: int,
    ) -> Portfolio: ...
```

Compiler 不允许返回 dependency coverage 小于 1.0 的 portfolio；若预算不足，必须删除整个依赖组件或返回明确的 `budget-unsatisfied`，不能截断 required node。

### 6.7 Renderer

```python
class MarkdownRenderer:
    def render_topic_views(self, manifest: ResolvedManifest, output_root: Path) -> RenderReport: ...
    def render_portfolio(self, portfolio: Portfolio) -> str: ...
```

Topic view 是向后兼容产物；portfolio view 是运行时交付产物。两者使用不同模板，不把所有 node 拼回一个超长文件。

### 6.8 AtomicOpSynchronizer

```python
class AtomicOpSynchronizer:
    def scan_source(self, api_set: str, source_roots: list[Path]) -> ApiSurfaceSnapshot: ...
    def diff(self, snapshot, manifest) -> AtomicSyncReport: ...
    def propose_revisions(self, report) -> list[NodeRevision]: ...
    def verify_runtime(self, api: ApiBase, snapshot) -> RuntimeVerificationReport: ...
```

静态 scanner 使用 AST：

1. 找到 API class 的 `functions()`；
2. 解析 `fns["name"] = self.method`；
3. 沿显式 class inheritance 解析 `super().functions()`；
4. 提取 method signature、docstring、source symbol 和 source hash；
5. 条件暴露（如 Molmo availability）记录为 availability guard；
6. 动态无法解析的构造产生 hard warning，要求人工 adapter；
7. 不 import/instantiate 机器人 API。

---

## 7. 检索与组合算法 v0.1

### 7.1 Context 生成

调用方先构建 TaskContext：

- suite/task/task language；
- API set 和静态 AtomicOp availability；
- forbidden API policy；
- exploration 阶段获得的 observed facts；
- 若在修复循环中，加入 trace failure signatures；
- token budget；
- 冻结 manifest ID/hash。

不允许把 held-out trace 传给 compiler。

### 7.2 Query 生成

v0.1 使用确定性拼接，不调用 LLM 改写：

```text
<task language>
task-family:<family>
observed:<sorted key=value facts>
failures:<sorted signatures>
api:<available op names>
```

对 task language 做 Unicode normalization、snake/camel 分词和受控机器人同义词扩展；同义词表是版本化 fixture。

### 7.3 Seed retrieval

1. 仅查 active manifest 中、允许状态的 `skill`；
2. scope hard prefilter；
3. FTS5 MATCH；
4. 取最多 24 个 seed；
5. BM25 tie-break 依次为：scope specificity、logical ID、version。

AtomicOp 不作为普通语义 seed；它们通过 `requires` 被补齐，避免 API 文档淹没行为策略。

### 7.4 Graph hydration

对每个 seed：

1. DFS/BFS 展开全部 `requires`，最大深度 3；
2. 匹配 failure signature 时展开 `recovers-with`；
3. 收集 `alternative-to` 组；
4. 收集适用 guard 成立的 `contradicts` pair；
5. 超过 64 节点、遇到 requires cycle 或 dangling ref 立即失败。

### 7.5 Hard gate

每个候选组件依次检查：

- effective status；
- suite/embodiment/environment/API scope；
- required AtomicOp 是否暴露且未 forbidden；
- precondition 三值逻辑；
- contraindication；
- stale binding；
- contradiction；
- experiment contamination policy。

blocked/deprecated node 默认不能因关系扩展重新进入 portfolio。

### 7.6 确定性评分

首版分数只用于 gate 后的组件排序：

```text
score =
  0.35 * lexical_relevance
+ 0.20 * scope_match
+ 0.15 * development_evidence
+ 0.10 * cross_task_support
+ 0.10 * freshness
+ 0.10 * failure_signature_match
- 0.15 * uncertainty_penalty
- 0.20 * contradiction_risk
- 0.10 * normalized_token_cost
```

所有分项归一化到 `[0,1]`，权重进入 compiler version。held-out success 不进入 ranking feature。没有证据的 candidate 可在 shadow 输出，但 retrieve/enforce 默认不交付。

### 7.7 Alternative 与 contradiction resolution

- alternative group 默认选最高分组件；
- 分数差小于 0.05 时选 token cost 更低者；
- recovery 专用 alternative 可作为 fallback 保留，但需有匹配 failure signature；
- contradiction 中若一方由更精确 guard 匹配，则保留更精确者；
- 其余无法决策的 conflict 导致组件拒绝，并写入 exclusions，不由 lexical score 强行裁决。

### 7.8 Budget packing

packing 单位是 dependency component，不是单 node。按 `score / estimated_tokens` 排序，在满足以下强制项后装包：

- task language 和 allowed/forbidden API 摘要；
- 每个 skill 的 goal、trigger、关键 procedure；
- requires AtomicOp 的签名；
- validator、failure mode 和 recovery；
- provenance 只输出短 ref，完整 evidence 留在 JSON。

默认 budget：2,400 tokens；实验矩阵使用 1,200 / 2,400 / 4,800。token estimate 由固定 tokenizer adapter 产生；无 tokenizer 时采用版本化近似算法并标注。

### 7.9 Compiler 不变量

- 精确版本与 hash 全部锁定；
- dependency coverage = 1.0；
- 无 blocked/deprecated node；
- 无 unresolved contradiction；
- 无 forbidden AtomicOp；
- 无跨 partition 泄漏；
- 相同输入产生 byte-identical JSON；实际编译时间只存在于 telemetry；
- exclusions 记录每个未入选候选的 reason code。

---

## 8. CLI 设计

统一入口：

```bash
python -m aspire.sim.cap.knowledge <command>
```

并在 `pyproject.toml` 注册等价快捷入口：

```toml
[project.scripts]
akl = "aspire.sim.cap.knowledge.cli:main"
```

### 8.1 Registry 与校验

```bash
akl validate --root knowledge --manifest libero-active
akl index build --root knowledge --output outputs/akl/index/akl.sqlite3
akl index verify --index outputs/akl/index/akl.sqlite3
akl manifest resolve --id libero-active --format json
akl graph inspect --node skill.libero.motion.safe-lateral-escape
```

`akl` 可以作为 `pyproject.toml` console script；测试和文档始终保留 `python -m` 形式。

### 8.2 迁移与渲染

```bash
akl migrate libero-v0 \
  --source .claude/libero/skills \
  --mapping knowledge/migrations/libero-v0-import-map.yaml \
  --dry-run

akl render topic-views \
  --manifest libero-active \
  --output .claude/libero/skills

akl render verify \
  --manifest libero-active \
  --views .claude/libero/skills
```

### 8.3 AtomicOp

```bash
akl atomic scan \
  --api-set FrankaLiberoApiReducedSkillLibrary \
  --class aspire.sim.cap.integrations.franka.libero_reduced_skill_library:FrankaLiberoApiReducedSkillLibrary \
  --output outputs/akl/reports/libero-api-surface.json

akl atomic diff --manifest libero-active --snapshot <path>
akl atomic propose --snapshot <path> --output knowledge/proposals/<id>
```

### 8.4 Retrieval 与 portfolio

```bash
akl retrieve \
  --context outputs/akl/contexts/<task>.json \
  --manifest libero-active \
  --limit 24 \
  --explain

akl portfolio compile \
  --context outputs/akl/contexts/<task>.json \
  --manifest libero-active \
  --budget 2400 \
  --json outputs/akl/portfolios/<run>/<task>.json \
  --markdown outputs/akl/portfolios/<run>/<task>.md

akl portfolio verify outputs/akl/portfolios/<run>/<task>.json
```

### 8.5 Promotion 与 lifecycle

```bash
akl promotion begin --suite "$SUITE" --task "$TASK"
akl candidate scaffold --from-findings "$TASK_DIR/findings.md" --output knowledge/proposals/<id>
akl promotion check --proposal knowledge/proposals/<id>
akl promotion finish --suite "$SUITE" --task "$TASK" --proposal knowledge/proposals/<id>
akl promotion verify --suite "$SUITE" --task "$TASK"

akl audit duplicates --manifest libero-active
akl audit stale --manifest libero-active --api-snapshot <path>
akl audit conflicts --manifest libero-active
akl lifecycle deprecate <exact-ref> --reason-code <code> --evidence <event-id>
```

`candidate scaffold` 只生成待 review 草案，绝不自动激活。任何 manifest 改动必须通过 `promotion finish`。

### 8.6 Exit code

| Code | 含义 |
|---:|---|
| 0 | 成功，无阻断问题 |
| 2 | CLI/输入错误 |
| 3 | schema/semantic validation 失败 |
| 4 | graph closure/conflict 失败 |
| 5 | stale API binding |
| 6 | portfolio 无法满足 budget/contract |
| 7 | generated view drift |
| 8 | contamination/partition policy 失败 |

---

## 9. 与现有 ASPIRE 的接入

### 9.1 Feature mode

通过显式配置而不是隐式环境行为控制：

```yaml
knowledge:
  mode: off       # off | shadow | retrieve | enforce
  manifest: libero-active
  index: outputs/akl/index/akl.sqlite3
  token_budget: 2400
  record_telemetry: true
```

| mode | Actor 看到什么 | 是否阻断任务 |
|---|---|---|
| `off` | 当前 Markdown | 否 |
| `shadow` | 当前 Markdown；后台生成 portfolio 并比较 | 只阻断数据损坏，不改变 actor |
| `retrieve` | portfolio；编译失败自动 fallback 到冻结 Markdown | 否，但记录 fallback |
| `enforce` | portfolio；编译/验证失败即停止任务 | 是 |

默认从 `off` 开始；任何 benchmark tag 必须记录 mode、manifest hash、compiler version 和 portfolio hash。

### 9.2 Fix-loop subagent

当前流程在 Stage 0 exploration 后要求读取相关共享技能。改造为：

1. subagent 探测 task language、对象和 API availability；
2. 写 `TaskContext`；
3. `portfolio compile`；
4. 只读 portfolio Markdown；
5. 运行代码与调试；
6. trace telemetry 关联 portfolio ID。

shadow 模式仍读原 Markdown，同时编译 portfolio，但不把 portfolio 注入 prompt。

### 9.3 Library-size scaling / Long-Pro

当前 one-shot prompt 在 snapshot worktree 中 `cat` 四个共享文件。改造时：

- snapshot tag 同时冻结 `knowledge/` 和 active manifest；
- Step 2 获得真实 task language 和 SAM3 probe 后构建 context；
- compiler 只能读取该 worktree 内的 knowledge root；
- portfolio 输出到该 worktree 的 `outputs/akl/`；
- contamination guard 新增：拒绝外部 index、其他 snapshot manifest 或 portfolio；
- index 必须在 snapshot 内重建，或验证 source tree hash 完全一致。

这样 snapshot-N0/N5/... 的“技能库大小”应改为“active manifest 节点集合”，同时仍可生成旧 Markdown 统计用于历史对比。

### 9.4 Promotion transaction

v0.1 不删除 `record_skill_promotion.py`，先让新 adapter 调用并扩展它：

```text
begin
  -> snapshot active manifest / affected node revisions / relations / views
  -> record findings hash

edit/propose
  -> create immutable candidate revision
  -> attach development evidence
  -> declare relation changes

check
  -> schema + graph + API + partition + generated-view validation
  -> duplicate/conflict report

finish
  -> append evidence/lifecycle events
  -> update manifest atomically
  -> rebuild index
  -> render views
  -> record exact patch + before/after hashes

verify
  -> ledger complete
  -> manifest resolves
  -> index source hash matches
  -> view hash matches
```

每次 transaction 仍按 suite 串行；后续可改为受影响节点级锁，但不是 v0.1 目标。

### 9.5 Findings 入库

现有 findings 已包含 trigger、code、evidence、target skill file。迁移后 coordinator 的职责变为：

- 判断是新 node、现有 revision、alternative、recovery 还是 contradiction；
- 将 task-specific 常量留在 scope/guard，不伪装成 universal knowledge；
- 关联 exact source code path/hash 和 development seeds；
- 不从 held-out 结果修改 contract；
- 对“none”仍写 no-op promotion evidence。

v0.1 可以用 LLM 帮助生成 proposal，但 proposal 始终是 untrusted candidate，必须经过人工 review 和机器 gate。

### 9.6 Runtime telemetry

每个 task 至少产生：

```json
{"event":"portfolio.compiled","portfolio_id":"...","manifest_hash":"...","tokens":1732}
{"event":"node.selected","node":"skill...@1.0.0","reason":"lexical+scope"}
{"event":"node.excluded","node":"skill...@1.0.0","reason":"contraindication"}
{"event":"atomic.called","node":"atomic-op...@1.0.0","count":3}
{"event":"failure.matched","signature":"ik-unreachable","recovery":"skill...@1.0.0"}
{"event":"task.outcome","partition":"development","success":true}
```

telemetry 只追加到 run output；promotion 时才提炼为 evidence event。不得在每次执行后自动改 registry。

---

## 10. LIBERO 首次迁移方案

### 10.1 迁移原则

- 不按 heading 数机械拆分；
- 一个 node 必须有一个清晰 goal/trigger/effect；
- 表格中的单行阈值若不能独立形成状态转换，作为已有 skill 的 parameter/evidence，不单独建 skill；
- API helper 建为 AtomicOp，不重复建成 skill；
- 代码模板可作为 artifact，但依赖必须指向 AtomicOp；
- 无法可靠结构化的内容进入 `unstructured_notes` migration report，不静默丢弃。

### 10.2 Import map

`knowledge/migrations/libero-v0-import-map.yaml` 为人工 review 的映射清单：

```yaml
source: .claude/libero/skills/transport.md
source_sha256: sha256:...
sections:
  - anchor: "Pre-Probe IK Conditioning"
    disposition: node
    target: skill.libero.motion.pre-probe-ik-conditioning@1.0.0
  - anchor: "Physics Settling After Release"
    disposition: merge
    target: skill.libero.placement.release-and-settle@1.0.0
  - anchor: "Placement Approach"
    disposition: split
    targets:
      - skill.libero.placement.vertical-approach@1.0.0
      - skill.libero.placement.release-and-settle@1.0.0
```

每个 source line 必须属于：`node`、`merge`、`split`、`atomic-op`、`view-only` 或 `unstructured`。目标是 source coverage 100%，不是每行都变成节点。

### 10.3 初次状态

- 已有明确 Stage 1 provenance 且仍符合 API 的知识：`validated`；
- 只有正文声称、找不到证据：`candidate`；
- 已知依赖旧 API/旧 scene 的条目：`deprecated` 或收窄 scope；
- 违反当前 forbidden policy：`blocked`；
- 不因出现在共享文件中自动授予 `stable`。

### 10.4 Round-trip 验证

迁移验收分三层：

1. Coverage：import map 覆盖原四文件全部非空正文；
2. Semantic review：每个原 section 都能从 node/artifact/view 中找到；
3. Prompt behavior：生成的 legacy topic views 与原文件做人工/golden review，确认关键代码、阈值、禁忌和 provenance 未丢失。

不要求生成视图 byte-identical，因为目标结构会变化；要求经过批准的 golden diff。

### 10.5 AtomicOp 首次同步

从 `FrankaLiberoApiReducedSkillLibrary` 开始：

- 扫描父类继承的全部暴露函数；
- 输出 API surface snapshot；
- 生成 AtomicOp proposals；
- 标注 conditionally available 的 Molmo；
- 与 subagent prompt 的 ALLOWED/FORBIDDEN 列表做双向 diff；
- 发现“代码已暴露但 prompt 未允许”或“prompt 允许但代码未暴露”都报告，不自动放宽 policy。

---

## 11. 去重、冲突、剪枝与失效

### 11.1 Duplicate audit

v0.1 的 duplicate detector 只给建议，不自动 merge。四层信号：

1. identity：相同 logical ID 或 provenance anchor；
2. lexical：title/goal/body MinHash 或 TF-IDF 相似；
3. contract：precondition/effect/validator canonical overlap；
4. behavior：development evidence 中相同 task/failure/outcome pattern。

输出分类：

- exact duplicate；
- candidate merge；
- alternative；
- specialization（v0.1 先记录，不创建 `is-a`）；
- unrelated lexical collision。

只有 exact duplicate 可由工具生成 merge proposal，最终仍需 review。

### 11.2 Conflict audit

检查：

- 同一 guard 下 effects 互斥；
- 一个 skill requires 被 blocked 的 AtomicOp；
- alternative 同时成为无条件 requires；
- recovery 重新引入 source 的 contradiction；
- scope overlap 中参数或操作方向冲突；
- forbidden policy 与 artifact 示例冲突。

冲突未解决时，相关 node 可以保留在 registry，但不能进入 enforce manifest。

### 11.3 Staleness signals

| Signal | 默认处置 |
|---|---|
| AtomicOp 不再暴露 | dependent skill `blocked` 建议，enforce hard fail |
| signature hash 改变 | 标记 `needs-revalidation`，从 retrieve manifest 移出 |
| perception backend/version 改变 | 收窄 freshness，进入 revalidation queue |
| 连续 development failures | 生成 audit，不直接因低样本 deprecate |
| 长期未使用 | 仅降低 maintenance priority，不视为无用 |
| superseding revision | 旧 revision `deprecated`，保留回滚 |
| forbidden policy 命中 | `blocked` |

`needs-revalidation` 在 v0.1 作为 lifecycle reason 和 materialized flag，不新增公开 status，避免状态爆炸。

### 11.4 Pruning policy

首版只做 soft pruning：

- 从 active manifest 移除；
- status 转 deprecated/blocked；
- 保留 revision、relations、evidence 和 artifact；
- index 中保留但默认不可检索；
- impact report 列出所有 incoming requires/recovery；
- 只有没有 inbound active relation 且通过 archive policy 的历史 revision 才可移到 cold archive；
- v0.1 不物理删除任何已 promotion 的 revision。

---

## 12. 测试蓝图

### 12.1 Unit tests

- ID、SemVer、exact ref parsing；
- Pydantic schema 与 canonical hash；
- predicate 三值逻辑和 regex 限制；
- lifecycle state replay；
- relation symmetry/cycle/dangling validation；
- FTS query normalization 和 tie-break；
- graph hydration depth/node cap；
- gate reason codes；
- score breakdown；
- budget packing 不截断 dependency component；
- Markdown escaping/rendering；
- JSONL partial-line/corruption handling；
- atomic write 与 duplicate ref protection。

### 12.2 Golden tests

- 四个 LIBERO 文件的 import map coverage；
- generated topic views；
- 10–20 个代表性 TaskContext 的 portfolio JSON/Markdown；
- API surface snapshot；
- duplicate/conflict/stale audit reports；
- SQLite logical dump hash。

Golden 更新必须显式 review，不用 `--update` 隐式通过 CI。

### 12.3 Integration tests

1. 从空 outputs rebuild index；
2. compile portfolio；
3. 删除 index 后重建，结果 hash 相同；
4. promotion begin -> proposal -> finish -> verify；
5. generated view drift 能阻断；
6. stale AtomicOp 使 dependent skill 被拒绝；
7. snapshot manifest 不能解析外部路径；
8. held-out evidence 不能触发 materialized status 变化；
9. retrieve mode 编译失败时按配置 fallback；
10. enforce mode 编译失败时停止而非静默全库加载。

### 12.4 Property/invariant tests

- `requires` closure 后 dependency coverage 恒为 1；
- 任何 portfolio 中无 contradicting active pair；
- blocked/deprecated 永不因 hydration 复活；
- active manifest 解析结果与文件遍历顺序无关；
- canonical serialize/parse round-trip 保持 hash；
- 所有 exclusions 有稳定 reason code；
- source coverage 不受 Markdown 换行风格影响。

### 12.5 Stress library

用真实节点复制/变体构建 1x、5x、20x、100x 库，不伪造新的 execution evidence。注入：

- 同义改写 duplicate；
- 只差一个 guard 的近重复；
- obsolete API binding；
- 高 lexical match 但 contraindicated 的 bad skill；
- requires chain；
- contradiction pair；
- recovery edge；
- 大量无关 topic noise。

测量 retrieval recall、dependency coverage、conflict rejection、token、latency和 task success。

### 12.6 ASPIRE-specific evaluation

对照组：

| 组 | 输入 |
|---|---|
| A | 当前四文件整库 |
| B | 仅 BM25 top-k node，无 graph |
| C | BM25 + graph hydration，无 contract hard gate |
| D | 完整 AKL v0.1 |

主场景：LIBERO-Pro Long；补充 fix-loop replay。必须固定 model、prompt、seed、API、snapshot、token budget。报告：

- held-out success rate 及置信区间；
- development-only promotion 指标；
- prompt input tokens；
- portfolio 节点数；
- dependency coverage；
- forbidden/stale/conflict escape rate；
- retrieval recall@k（人工标注 relevant set）；
- duplicate suggestion precision；
- compile p50/p95 latency；
- fallback rate；
- bad-skill negative transfer。

### 12.7 初始验收阈值

进入 retrieve mode 前：

- schema/graph/index tests 100% 通过；
- source mapping coverage 100%；
- dangling refs、requires cycles、unresolved hard conflicts = 0；
- portfolio dependency coverage = 100%；
- forbidden/stale AtomicOp escape = 0；
- deterministic rebuild/golden mismatch = 0；
- 代表性任务 relevant skill recall@10 ≥ 0.90；
- p95 compile latency（本地 warm index）≤ 250 ms；
- 相比 flat-library，portfolio 中位 token 至少下降 40%；
- shadow fallback/compile error < 1%。

进入 enforce mode前：

- LIBERO-Long-Pro held-out success 的 95% CI 不显示相对 flat baseline 下降超过 3 个百分点；
- stale/duplicate/conflict stress 的 hard violation escape = 0；
- 至少完成一次完整 snapshot worktree 复现实验；
- 回滚演练通过。

这些是发布 gate，不是论文结论；实验完成后可基于基线数据修订，但必须版本化记录。

---

## 13. CI、可观测性与安全

### 13.1 Pull request gates

建议顺序：

```text
ruff/mypy
  -> unit tests
  -> akl validate
  -> graph audit
  -> atomic API diff
  -> deterministic index rebuild
  -> generated view drift check
  -> golden portfolios
  -> promotion ledger verification
```

修改 `knowledge/` 却没有对应 promotion/evidence 的 PR 失败；只改生成 view、未改 source 的 PR 失败。

### 13.2 指标

按 run 聚合：

- `akl_compile_total{status}`；
- `akl_compile_latency_ms`；
- `akl_portfolio_tokens`；
- `akl_selected_nodes`；
- `akl_excluded_nodes{reason}`；
- `akl_fallback_total{reason}`；
- `akl_stale_binding_total`；
- `akl_conflict_total`；
- `akl_node_usage_total{node}`；
- `akl_task_outcome{partition,success}`。

v0.1 用 JSONL + 离线汇总，不引入在线 metrics 服务。

### 13.3 安全边界

- node/artifact 中的 Python 只能作为 fenced example；
- AKL 不 `exec` node body；
- runtime function binding 只来自 ApiBase registry；
- renderer 对 Markdown fence、链接和控制字符做转义；
- 所有 artifact/path 必须位于 knowledge root；
- 禁止 YAML custom tags；使用 `safe_load`；
- JSONL 单事件大小、portfolio 大小、regex 长度和 graph expansion 有硬上限；
- AtomicOp policy 单独维护 allowed/forbidden，知识条目不能自行放宽；
- 生成 portfolio 必须记录 manifest/compiler/context hash，避免无来源 prompt 注入；
- 外部/LLM 生成 proposal 视为不可信输入。

---

## 14. 发布、回滚与兼容

### 14.1 发布阶段

```text
R0 scaffold
  -> R1 shadow registry
  -> R2 shadow compiler
  -> R3 retrieve with fallback
  -> R4 enforce for selected suite
  -> R5 default retrieve/enforce decision
```

每阶段只在前一阶段 gate 通过后前进。

### 14.2 回滚

回滚不删除数据：

1. 设置 `knowledge.mode: off`；
2. 使用 snapshot tag 中冻结的 `.claude/libero/skills/*.md`；
3. 保留失败 portfolio、telemetry 和 evidence；
4. active manifest 回指上一 revision；
5. rebuild index/render views；
6. `promotion verify` 记录回滚 transaction。

retrieve mode 的单 task fallback 必须写明原因；enforce mode 禁止隐式 fallback，避免实验条件不一致。

### 14.3 兼容窗口

- R0–R2：Markdown 可人工编辑，但 knowledge import/report 检测差异；
- R3：knowledge 成为权威，Markdown 只由 renderer 写；
- 至少保留一个 release 的 legacy topic views；
- 删除 legacy 读取路径需单独 ADR，不包含在 v0.1。

---

## 15. 实施里程碑与工作包

以下估算为单名熟悉 ASPIRE 的工程师净开发日，不含 GPU benchmark 排队时间。

### M0：契约冻结与脚手架（2–3 人日）

- AKL-001：创建 package/data/test 目录；
- AKL-002：实现 enums、ID、SemVer、canonical hash；
- AKL-003：实现 Node/Relation/Evidence/Manifest Pydantic schema；
- AKL-004：定义 error codes 和 CLI skeleton；
- AKL-005：写 ADR-001 至 ADR-010。

验收：最小 fixture 可 validate、hash、round-trip。

### M1：Registry、lifecycle 与 graph（4–6 人日）

- AKL-101：filesystem repository + atomic writes；
- AKL-102：JSONL evidence/lifecycle reader/writer；
- AKL-103：effective status materializer；
- AKL-104：manifest resolver；
- AKL-105：relation validation、requires cycle 和 dangling checks；
- AKL-106：source/artifact path sandbox。

验收：不可变 revision、状态回放和 graph invariants 全部通过。

### M2：AtomicOp 与 LIBERO 迁移（6–9 人日）

- AKL-201：AST API surface scanner；
- AKL-202：allowed/forbidden policy diff；
- AKL-203：AtomicOp proposal generator；
- AKL-204：四文件 import map；
- AKL-205：skill/artifact revisions；
- AKL-206：topic view renderer；
- AKL-207：round-trip/golden review。

验收：source coverage 100%，API diff 已审查，生成 views 被批准。

### M3：Index、检索与 portfolio（6–8 人日）

- AKL-301：SQLite/FTS schema 和 deterministic rebuild；
- AKL-302：query normalizer；
- AKL-303：seed retriever；
- AKL-304：graph hydrator；
- AKL-305：contract gate；
- AKL-306：deterministic ranker；
- AKL-307：budget packer；
- AKL-308：JSON/Markdown portfolio renderer；
- AKL-309：explain/exclusion report。

验收：golden TaskContext 可重复产生 dependency-complete portfolios。

### M4：ASPIRE 集成与 promotion（5–7 人日）

- AKL-401：feature config 与 mode switch；
- AKL-402：fix-loop context/portfolio 接入；
- AKL-403：Long-Pro snapshot 接入与 contamination guard；
- AKL-404：promotion adapter 扩展现有 ledger；
- AKL-405：telemetry events；
- AKL-406：fallback/enforce behavior；
- AKL-407：端到端 integration tests。

验收：off 行为不变；shadow 不影响 actor；retrieve 可回退；enforce 可正确阻断。

### M5：维护审计与压力实验（5–8 人日 + benchmark）

- AKL-501：duplicate audit；
- AKL-502：conflict audit；
- AKL-503：stale binding/revalidation queue；
- AKL-504：soft deprecation impact report；
- AKL-505：1x/5x/20x/100x stress generator；
- AKL-506：A/B/C/D evaluation harness；
- AKL-507：指标汇总与 go/no-go report；
- AKL-508：rollback drill。

验收：达到第 12.7 节 gate，形成是否进入 enforce 的正式决策。

### 15.1 关键依赖

```text
M0
 ├─> M1 ─────────────┐
 └─> M2(Atomic AST)  │
       └─> M2(import)├─> M3 ─> M4 ─> M5
                     │
promotion adapter ───┘
```

M2 的 import mapping 可与 M1 后半并行；M4 不能在 M3 golden portfolio 稳定前开始改 actor prompt。

### 15.2 首个两周开发切片

如果只批准一个最小 sprint，交付：

1. schema/repository/validator；
2. AtomicOp AST snapshot；
3. 四文件 import map 和至少 8–12 个代表性 skill nodes；
4. SQLite FTS rebuild；
5. 一个离线 `portfolio compile`；
6. 5 个 golden TaskContext；
7. shadow report，不改任何 benchmark prompt。

这能尽早验证 granularity、contract 可写性和结构检索是否值得继续，而不触碰实验主链。

---

## 16. Definition of Done

AKL v0.1 只有同时满足以下条件才算完成：

### 数据与治理

- [ ] 两类节点和四类关系均有 schema、fixture 和文档；
- [ ] 当前 LIBERO 四文件 source coverage 100%；
- [ ] 所有 active skill 都有 provenance 和 exact version；
- [ ] 所有 requires AtomicOp 都与 API surface 对齐；
- [ ] evidence/lifecycle append-only 且能回放；
- [ ] promotion transaction 覆盖 node、relation、manifest、view、index hash；
- [ ] no-op promotion 仍可审计。

### 检索与运行时

- [ ] SQLite index 可从 Git source 一键重建；
- [ ] portfolio deterministic、pinned、dependency complete；
- [ ] blocked/deprecated/forbidden/stale/conflicting knowledge 不会进入 enforce portfolio；
- [ ] token budget 不破坏依赖闭包；
- [ ] Actor prompt 只注入 portfolio view；
- [ ] telemetry 能关联 task、portfolio、node、AtomicOp 和 outcome。

### 兼容与实验

- [ ] off 模式与原流程行为一致；
- [ ] shadow 模式至少跑完一轮代表任务；
- [ ] retrieve/enforce 的 fallback/stop 语义经过 integration test；
- [ ] snapshot worktree 可完全复现 manifest/index/portfolio；
- [ ] A/B/C/D 实验完成并通过发布 gate；
- [ ] 回滚演练完成；
- [ ] README、runbook 和故障排查文档齐全。

---

## 17. 风险登记与应对

| 风险 | 早期信号 | 应对 |
|---|---|---|
| 拆分粒度错误 | node 数暴涨、每个 task 选几十个节点 | 以 state transition/goal 为单元，review import map |
| contract 形式化成本过高 | 大量 `unknown`、body 仍承载全部语义 | 只形式化 gate 所需字段，保留 artifact/body |
| 错误 relation 比没有 graph 更坏 | requires closure 引入无关知识 | edge evidence、shadow ablation、relation review |
| 生成 view 与 source 漂移 | 人工继续直接改 Markdown | R3 起只允许 renderer，CI drift gate |
| API scanner 不完整 | 动态 functions 无法解析 | 显式 adapter + runtime verify，不静默跳过 |
| ranking 对样本噪声过拟合 | 高频 skill 压制关键低频 safety knowledge | hard gate 与 safety policy 不参与频率降权 |
| token 降低但 success 下降 | relevant recall 或 recovery coverage 不足 | 调 budget、improve relation，不直接全库 fallback 做默认 |
| held-out 污染 | evidence partition 缺失或输入 context 含 seed 1–50 | partition mandatory、compiler contamination gate |
| 双权威期冲突 | YAML/Markdown 双边修改 | 明确兼容窗口，promotion transaction 统一写入 |
| 维护系统本身变成技术债 | schema 频繁破坏、CLI 难用 | schema version、migration tests、保持 stdlib/Pydantic 边界 |

---

## 18. 后续演进门

只有 v0.1 证明结构收益后才依次打开：

### Gate A：`strategy`

条件：多个 skills 需要稳定的排序/分支组合，且 portfolio 中重复出现相同组合。新增 `decomposes-to`、`can-follow`，不改变现有 skill identity。

### Gate B：`method`

条件：fix-loop、EvoSearch、revalidation 等工作流需要作为可比较、可版本化的方法节点管理。

### Gate C：`principle`

条件：同一不变量得到跨 task family、跨时间证据，且能明确约束/支持下层节点。原则蒸馏必须是 proposal，不自动覆盖原始证据。

### Gate D：dense retrieval

条件：lexical + graph 的 recall 已成为主要瓶颈，并已有人工 relevant set。dense 模型必须版本锁定，作为额外 seed channel，不取代 hard gate。

### Gate E：real-robot transfer

条件：新增 embodiment adapter、独立 evidence partition 和 revalidation policy。LIBERO 的 validated 不自动等同于 real robot validated。

---

## 19. 最终推荐路径

工程上建议立即启动 M0–M2 的 shadow registry，不直接改 actor：

1. 先冻结 schema、ID、version、predicate 和 relation 语义；
2. 用 AST 同步真实 AtomicOp，建立“知识依赖是否可执行”的底座；
3. 对当前四文件做人工可审查迁移，校准 node granularity；
4. 以 FTS + graph + gate 编译离线 portfolio；
5. shadow 对比“整文件上下文”和“portfolio”覆盖率、token、冲突；
6. 数据门槛通过后才接入 fix-loop/Long-Pro retrieve mode；
7. 最后用 duplicate/stale/conflict stress 和 held-out non-inferiority 决定是否 enforce。

这条路线保留 ASPIRE 最强的部分——从具身失败中学习、严格的 development/held-out 隔离、可审计 promotion 和 snapshot——只替换其最难扩展的部分：以整份 Markdown 作为知识 identity、检索单位和生命周期单位。

首个工程目标不是“做一个知识图谱”，而是让每次机器人任务都能回答五个可审计问题：

1. 为什么选了这些知识？
2. 它们依赖哪些真实可调用能力？
3. 哪些条件下不应该使用？
4. 使用了哪个精确版本、依据什么证据？
5. API、环境或证据变化后，哪些节点必须退出或重验证？

当这五个问题能由代码稳定回答时，Pi 式显式分层才真正从类型设计变成 ASPIRE 技能库规模化的工程解法。
