# LIBERO G17 / CaP-X — Agent Handoff Prompt (from zero)

> **Usage:** Paste this entire document as the system/user context for a new coding agent.  
> **Date:** 2026-08-18  
> **Worktree:** `/home/qshou/capx_g17_observation_integrity`  
> **Outputs:** `/project/peilab/qshou/cap-x-agilex/outputs/`  
> **Frozen helper runtime:** `/home/qshou/capx_causal_ik_replay_81e0acb`  
> **Do not commit unless the user explicitly asks.**

Wash-line deep dive (GT→FG place symbols):  
`docs/LIBERO_G17_WASH_LINE_HANDOFF_20260818.md`  
Method POC anchor: `docs/LIBERO_G17_V254_METHOD_ANCHOR.md`  
Closed set: `docs/LIBERO_G17_CLOSED_SET_SPEC.md`

---

## 0. One-paragraph mission

Build **generalizable Factual Grounding (FG) components** for CaP-X on LIBERO that fix **cognitive** errors (false beliefs about grasp identity and placement), not replace the controller with privileged scripts. Decompose into **GraspCorrect** and **PlaceCorrect**. Skill-library changes (`sample_grasp_pose` instance resolve, `move_tip_xy`) exist so that a corrected high-level plan is **executable**. Privilege (GT) proves upper bounds; **wash GT→FG→PT** so the method uses discrete logical predicates; PT (perception+VLM) is the deployable truth source and future training surface. Current active work: **wash PlaceCorrect’s tipxy law from GT residual-sign to FG RelXY predicates without rate collapse**, under fair-K with frozen π0 (+ grasp replay when needed).

---

## 1. Hard principles (do not violate)

1. **FG ≠ execution coaching.** No “next call `open_gripper` / go left” API advice in public FG. Report eye-checkable **state / gap predicates** only.
2. **FG ≠ continuous GT dump.** Public text uses finite CNL / discrete bands; private BDDL / `_check_success` never action-visible.
3. **Skill changes enable FG plans; they are not the paper claim by themselves.** tipxy / SGP are scaffolding for cognition→action.
4. **Ledger GT → FG → PT must stay separate.**  
   - GT: privileged geometry (`private_poses`, continuous residual).  
   - FG: discrete logical predicates (ideally lossless vs GT’s discrete projection).  
   - PT: Molmo/SAM/`get_object_pose`; gap = **ε_PT**, trainable future work.  
   Performance “should not drop” on **GT_discrete ≡ ideal FG**; drops on PT are expected and attributable.
5. **PG is ablation**, not the main claim path.
6. **Fair-K ≠ full suite.** Prefix fair-K on `spatial_0` trials 1–2 does not extrapolate to no-prefix 10×10.
7. Prefer **minimal diffs** when washing; do not stack inscribed-square / no-Z / extra guards into the “only symbol source changed” claim.

---

## 2. Evaluation protocol (fair-K)

| Item | Value |
|---|---|
| Suite / task | `libero_spatial` **task 0** |
| Cells | K=5 replicates × trials **1–2** → **10 cells** (“x/10”) |
| Pair control | Baseline spatial job **`517994`** (trial1 fail / trial2 success cells) |
| Treatment | Same inits + **frozen π0 code-prefix** |
| Prefix package | `/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json` |
| Model | `qwen3.7-plus`, temp 0, global seed `20260718` |
| Validity | Prefer `valid_prefix_arm`; aggregate **rep-*/LATEST_SUMMARY.json** (arm-level file is last-rep shaped) |
| Metrics | Per-cell `taskcompleted`; also `gain_result` (rescue/guard contract vs control) |
| Launch | `K=5 bash scripts/submit_libero_g17_fair_fg_<arm>_kseed.sh` |

**Prefix meaning:** replay frozen first code block(s) from prior successes (`n_prefix=1` per trial); live FG after π0. Not a prompt prefix.

**GraspNet noise:** historically unfrozen; pollutes place ablations. Grasp replay package (from v256):  
`/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_grasp_replay_v256_20260817/`  
v264 uses `OVERFLOW=repeat_last` (v263’s `miss` was wrong for fair claims).

---

## 3. Problem decomposition

### 3.1 GraspCorrect（抓对）— largely done on fair-K

- **Cognitive:** same-name bowl identity; wrong-grasp beliefs.  
- **Skill:** silent SGP overload / typed instance resolve / keep-stock (no public `which=`).  
- **FG:** wrong-grasp ordinal (often idle if skill prevents `holding_other`).  
- **Anchor:** **v238** → **6/10** (t1 3/5, t2 3/5, joint 2/5), schema `v2.36-ordinal-sgp-overload`.  
- **Ablation:** v239 ordinal off → **3/10** (hurts t2).

### 3.2 PlaceCorrect（放对）— active wash line

- **Cognitive:** On-XY within / sign of payload−receptacle; release readiness.  
- **Skill:** `move_tip_xy` (mm-class, Z/quat locked; never legacy `goto_pose` for this nudge).  
- **Controller laws:** release-ready band-step; near-miss osc **15−3t**; FG predicate wash.  
- **Anchors:** see §4.

### 3.3 FG vs skill (user doctrine)

FG fixes **cognition**. tipxy / SGP changes make corrected plans **executable**. Do not claim “we replaced execution with FG”; claim “false beliefs corrected; skills can carry the plan.”

---

## 4. Version anchors (memorize these)

Scores = unique treatment successes over 10 fair-K cells unless noted. Outputs under `…/outputs/libero_g17_fair_fg_<arm>_YYYYMMDD/`.

| Arm | Role | Rate | Notes |
|---|---|---|---|
| **baseline / v24** | Early FG On/In IFF | fair-K joint **2/5** (v24); suite 10×10 ~**6–9/100** vs ~3–5/100 | No-prefix full spatial does **not** match fair-K hype |
| **v238** | **GraspCorrect anchor** | **6/10** | Ordinal + SGP; XY_LOCK; no tipxy place law |
| **v254** | **Place POC upper bound** | **9/10**, often gain-positive | RR tipxy **3 cm band-step**, residual from **GT** `private_poses` (`exploratory_live`) |
| **v255** | Fair residual ablation of v254 | **6/10** | Student `get_object_pose` residual → **GT→PT** drop |
| **v256** | **Wash reference champion** | **9/10** | Shared-t **15−3t** + tipxy bandwindow; **sign still GT** continuous residual |
| **v260** | **Minimal wash arm** | **7/10** | Same 15−3t; **sign/skip from FG RelXY** (geometry still from `private_poses`) |
| v261/v262 | Inscribed / no-Z side quests | 4/10 → 7/10 | **Not** part of “only symbol source” claim |
| v263 | Grasp replay, overflow=miss | dirty | Do not use for main wash conclusion |
| **v264** | v260 + grasp replay `repeat_last` | **7/10** (complete K=5) | Fair grasp control; **did not recover 9** → gap ≠ only GraspNet |
| v265 | + hold-guard | **4/10** | Regression |

**Open scientific question:** Why does FG RelXY sign (v260/v264) stay at **7/10** while GT sign (v256) is **9/10** despite unit-isomorphism tests? Need same-trial diffs under frozen grasp—not more unrelated guards.

**Invalid suite note:** `libero_g17_v24_spatial_10x10` job **521102** invalid (robot_descriptions git SHA); valid retry **521411** (~9/100 deduped). Robot cache skip-git fix lives in `python_sitecustomize/sitecustomize.py`.

---

## 5. Method definitions (short)

### 5.1 Public On IFF / release-ready

- Private goal: BDDL `On(payload, receptacle)` ↔ `_check_success`.  
- Public: eye-checkable acquisition / placement / On-XY band evidence.  
- Place servo at `open_gripper`: if holding and XY not within → tipxy steps before open.

### 5.2 tipxy

- Module: `capx/integrations/franka/tip_xy_only.py`  
- Closed-loop world tip XY; lock Z & quat; ~mm tolerance.  
- Student expose optional (`CAPX_LIBERO_TIP_XY_ONLY=1`); controllers may call internally.

### 5.3 Wash target law (v256 vs v260)

- Magnitude schedule: **start 0.15 m, step 0.03 m → `mag = 0.15 − t·0.03`** (shared t unless decouple variants).  
- **v256:** sign/skip from GT continuous residual.  
- **v260/v264:** sign/skip from FG c2c RelXY ternary bands; claim: only decision interface changes.  
- Inscribed-square / no-Z: experimental side paths—exclude from minimal-wash conclusions.

### 5.4 Identity / SGP

- Public API: type name only.  
- Internal: `_resolve_typed_instance` + XY_LOCK keep-stock / privileged aim when lock on.  
- Goal-identity rewrite path exists but is **off** on this fair-FG line.

---

## 6. Research assets (what is real)

| Asset | Status |
|---|---|
| Grasp vs Place decomposition | Empirically validated |
| v238 GraspCorrect stack | Fair-K 6/10 anchor |
| `move_tip_xy` skill | Real mm-class primitive + tests |
| v254/v256 PlaceCorrect GT upper bound | 9/10 feasibility proof |
| Fair residual ablations (v255 etc.) | Honest GT→PT accounting |
| Prefix + grasp replay packages | Fairness instrumentation |
| FG RelXY wash implementation (v260) | Exists; **rate gap to 9 open** |
| Closed-set On/In docs + eye catalog | Spec debt mostly paid for pick-place |
| Unit isomorphism FG band ↔ −sign(r) | Supports “discrete ≠ weaker” philosophy |

**Not yet assets:** legal PlaceCorrect at 9/10; PT as live truth source; suite-level generalization; Articulate/Toggle/MultiGoal components.

---

## 7. LIBERO coverage (for planning)

- **Same-spec bowl→plate On:** `libero_spatial` **10/10** + `libero_goal` **1/10** ≈ **11** tasks.  
- **GraspCorrect + PlaceOn** mainly strike **spatial** (identity + On-XY).  
- **Next largest FG+skill family:** **PlaceIn** (`libero_object` ×10 baskets).  
- Later families (cognitive, not trivial coaching):  
  - **Articulate:** Open/Close false beliefs, open-enough, drawer identity, premature close.  
  - **Toggle:** on/off false beliefs, contact≠powered, bind vs On(pot,stove).  
  - **MultiGoal:** goal-set satisfaction + left/right/both binding (orchestration FG; reuse grasp/place skills).

---

## 8. Goal direction (priority order)

1. **Close GT→FG place wash:** explain/fix **7→9** under frozen prefix+grasp (v256 vs v260/v264 same-cell autopsy). Keep minimal symbol-only diff.  
2. **Stabilize PlaceCorrect as a component** (schema/flags frozen; anti-regression vs v256/v260).  
3. **Optional PT path** after FG shell matches GT shell; attribute ε_PT separately.  
4. **Generalize On** across spatial 0–9 (still fair protocol first).  
5. **PlaceInCorrect** for object suite.  
6. Articulate / Toggle / MultiGoal only after On/In place cognition is clean—FG state/gap predicates, not step scripts.

---

## 9. Key code paths

| Path | Why |
|---|---|
| `capx/verification/libero_g17_takeover.py` | Schema resolver, takeover |
| `capx/verification/libero_g17_fg_projection.py` | Public FG / ordinal / On-In |
| `capx/verification/libero_g17_release_ready.py` | RR / band-step / fair residual |
| `capx/verification/libero_g17_nearmiss_osc.py` | 15−3t / wash runners |
| `capx/verification/libero_g17_xy_lock.py` | Same-name / RelXY lock |
| `capx/verification/libero_g17_grasp_replay.py` | Grasp freeze |
| `capx/verification/libero_g17_code_prefix_replay.py` | π0 freeze |
| `capx/integrations/franka/tip_xy_only.py` | tipxy primitive |
| `capx/integrations/franka/libero.py` | SGP, `open_gripper` hooks |
| `python_sitecustomize/sitecustomize.py` | robot_descriptions skip-git, runtime patches |
| `scripts/sbatch_smoke_libero_g17_fair_fg_v*.sh` | Per-arm jobs |
| `scripts/submit_libero_g17_fair_fg_v*_kseed.sh` | K=5 submit |

---

## 10. Immediate next actions for the new agent

1. Read `LIBERO_G17_WASH_LINE_HANDOFF_20260818.md` + `LIBERO_G17_V254_METHOD_ANCHOR.md`.  
2. Treat **v256=9/10** and **v260/v264=7/10** as the live contradiction to resolve (not v265).  
3. Autopsy **same rep/trial** place traces: GT-sign vs FG-sign tipxy decisions under grasp replay.  
4. Do **not** “fix” wash by re-adding inscribed/no-Z/hold-guard without a named hypothesis.  
5. Ask the user before large suite runs, commits, or changing frozen packages.

---

## 11. One-liner

> **GraspCorrect≈done (v238); PlaceCorrect has a GT 9/10 law (v254/v256); washing sign to FG RelXY (v260/v264) still sits at 7/10 under fair grasp freeze—close that gap without trivial coaching, then PT, then PlaceIn / suite.**
