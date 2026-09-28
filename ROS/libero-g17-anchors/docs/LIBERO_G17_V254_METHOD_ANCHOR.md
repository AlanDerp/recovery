# LIBERO G17 v254 Method Anchor (2026-08-17)

**Champion arm:** `v254`  
**Schema:** `capx-libero-g17-fg-takeover-v2.52-ordinal-sgp-release-ready-tipxy-bandstep`  
**Claim class:** `exploratory_live` (fair-K variance estimate; not a `mechanism_verify` table)  
**Task cell:** `libero_spatial` task 0, trials 1–2, K=5 replicates, frozen π0 prefix

## Effectiveness (unique trial success)

Metric: any `taskcompleted_1` under `(rep, trial)` counts as success (same as prior fair-K arms).

| Arm | Rate | Role |
|-----|------|------|
| **v254** | **9/10** | Current champion — RR tipxy band-step |
| v238 | 6/10 | Prior champion (ordinal + SGP; no tipxy place fix) |
| v251 | 5/10 | RR tipxy with **wrong** mm micro-steps |
| v252 | 4/10 | GT near-miss fail-open |
| v253 | 2/10 | Fair `get_object_pose` place-servo (too tight / weak) |
| v250 | 4/10 | start/step oscillator (window blocked entry) |

**Outputs:**  
`/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v254_20260817/`  
Jobs: `522261`–`522265`. Sole fail: `rep-03` trial 02 (`Z abort` then no converge).

**What this does *not* claim:** Ideal-TK, Goal rewrite, or offline IFF mechanism tables. Remote Qwen post-π0 remains live variance.

## Method (what actually worked)

At `open_gripper`, **before** release judgment finishes:

1. If holding and public On XY **not** within band → tip_xy slide using **FG residual** (payload−receptacle XY).
2. Step size = `LIBERO_OBJECT_ONTOP_XY_M` = **0.03 m** (not 2–5 mm).
3. tip_xy tol ≪ step (~1.5 mm) so commands are not no-op’d.
4. Plate `contact` is **not** required to *enter* the slide (window fix). Mid-slide contact flicker continues; open when `acquired ∧ xy_within`.
5. Motion backend: `move_tip_xy` (hand IK, lock Z/quat; **never** legacy `goto_pose` for this nudge).

Failed variants (do not re-enable as default):

- v251 mm steps + `not alignable` refuse when `contact=False`
- v252 GT residual fail-open with long slides / lost contact
- tip-identity / goto TCP patches for micro-XY
- Refuse-open without a real band-step attempt

## Component inventory

### Execution primitive

| Component | Path | Role |
|-----------|------|------|
| `move_tip_xy` | `capx/integrations/franka/tip_xy_only.py` | Closed-loop world tip XY; Z/quat locked; no `goto_pose` |
| Student expose | `CAPX_LIBERO_TIP_XY_ONLY=1` | Optional public API; controllers may call method always |
| Wire | `FrankaLiberoApi.open_gripper` / `move_tip_xy` | `capx/integrations/franka/libero.py` |

### ReleaseReady tipxy band-step (v254)

| Component | Path / flag | Role |
|-----------|-------------|------|
| Gate + law | `capx/verification/libero_g17_release_ready.py` | `next_xy_step_band`, `_TIPXY_STEP_M=0.03` |
| Enable | `CAPX_LIBERO_G17_RELEASE_READY=1` | |
| Tipxy backend | `CAPX_LIBERO_G17_RELEASE_READY_TIPXY=1` | |
| Band-step schema pin | `CAPX_LIBERO_G17_RELEASE_READY_TIPXY_BANDSTEP=1` | Selects v2.52 schema |

### Fair-K stack (unchanged vs v238 base)

| Component | Role |
|-----------|------|
| `CAPX_LIBERO_G17_FG=1` | FG takeover reports |
| `CAPX_LIBERO_G17_XY_LOCK=1` | Empty-hand same-name XY suppress; keep-stock |
| `CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1` | Wrong-grasp ordinal |
| Prefix package | `code_prefix_replay_v1.json` (sha in submit manifest) |
| Public On band | `LIBERO_OBJECT_ONTOP_XY_M=0.03` in `libero_g17_public_iff.py` |
| Schema resolver | `libero_g17_takeover.schema_version()` |

### Supporting (present; not the v254 gain)

| Component | Notes |
|-----------|------|
| `libero_g17_nearmiss_osc.py` | start=R≈0.15, step=0.03 proof schedule; needs same window fix to fire |
| `libero_g17_nearmiss_failopen.py` | v252; GT residual; superseded |
| `libero_g17_fair_place_servo.py` | v253; fair pose; weak on K=5 |
| Ideal-TK / Goal identity / tip-identity goto | Off for v254 |

### Launch

| Script | Role |
|--------|------|
| `scripts/sbatch_smoke_libero_g17_fair_fg_v254.sh` | Single replicate |
| `scripts/submit_libero_g17_fair_fg_v254_kseed.sh` | K=5 submit + preflight |

## Known residual limits

1. **Side grasp:** tip ≠ bowl COM; tip_xy can move tip while c2c stalls; release settle can move COM after open.
2. **Z abort:** rare large `d_tip_z` still refuses that open attempt.
3. **Oscillator proof** (`start=R`, `step=3cm`, ≤5 rounds) remains valid under rigid tip≈COM + entry window; v254 uses residual-capped 3 cm steps (same ≤5-step bound for ‖r‖≤R) rather than the mag schedule.

## Reproduce preflight

```bash
cd /home/qshou/capx_g17_observation_integrity
CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 \
CAPX_LIBERO_G17_RELEASE_READY=1 CAPX_LIBERO_G17_RELEASE_READY_TIPXY=1 \
CAPX_LIBERO_G17_RELEASE_READY_TIPXY_BANDSTEP=1 \
PYTHONPATH=. .venv-or-agilex-python -c \
  "from capx.verification import libero_g17_takeover as t, libero_g17_release_ready as rr; \
   s=t.schema_version(); assert 'v2.52' in s and 'bandstep' in s; \
   assert abs(rr._TIPXY_STEP_M-0.03)<1e-12; print(s)"
```
