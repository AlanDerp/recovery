# Freeze anchor: v332_x native-retry (spatial triad)

Deterministic performance lock for Cap-X / LIBERO G17 **FG-informed native retry**
without adaptive GR / Clean-C.

| Axis | Suite | Score | Expt root |
|------|-------|------:|-----------|
| Canonical | `libero_spatial` | **86/100** | `…/libero_g17_v332_x_native_retry_spatial_10x10_20260829` |
| PRO Pos | `libero_spatial_swap` | **68/100** | `…/libero_g17_v332_x_native_retry_spatial_swap_10x10_20260829` |
| PRO Task | `libero_spatial_task` | **78/100** | `…/libero_g17_v332_x_native_retry_spatial_task_10x10_20260829_retry` |

- **Method git SHA (immutable):** `e20e2ea930bf2a751d07af1f83dc951e0681280d`
- **Tag:** `handoff/v332-x-86-method` (method tree); this folder documents the **PRO triad** on that same SHA
- **Branch tip for this package:** `freeze/v332-x-86-pro-triad`
- **No code-prefix replay / no grasp replay** on scored cells
- **Global seed:** `20260718`

Machine-readable finals live under [`results/`](results/). Source fingerprints under [`manifests/`](manifests/).

**Readable method-code index (by component):** [`COMPONENTS.md`](COMPONENTS.md) — FG takeover, Soft-NR, ReleaseReady, identity, ownership, wiring, submits, with deep links pinned to `e20e2ea`.

---

## Mode in one paragraph

**Composition:** immutable v332 base (live spatial 82/100) **+ audited post-v332 “X”**
(ownership / closed router / inverse escape / FG-native-retry through v354) **− adaptive GR / Clean-C**.
Non-held acquisition endpoints are owned by **FG-informed native retry** (Soft-NR projection + regenerate),
not by adaptive grasp-recovery. RR/place path stays the v332 On/tipxy lineage (unchanged relative to this ablation’s claim).

```text
AcquireNeeded
  -> sample_grasp_pose binds one transaction (ownership single-writer)
  -> canonical pregrasp + ingress + close
  -> FG endpoint classification
       TargetHeld     -> inverse escape -> transport -> RR/On
       non-held       -> open (+ inverse if available) -> FG native retry / regen
  Adaptive GR / Clean-C / True-GR flags: OFF
```

---

## Required flags (ON)

Exact list from the canonical spatial submit / `SOURCE_SHA256`:

```text
CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY=1
CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN=1
CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION=1
CAPX_LIBERO_G17_V313_ACQUISITION_BINDING=1
CAPX_LIBERO_G17_V319_TRANSITION_CLOSURE=1
CAPX_LIBERO_G17_V320_SEMANTIC_TRANSITION=1
CAPX_LIBERO_G17_V331_POSTCLOSE_STATE_MACHINE=1
CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP=1
CAPX_LIBERO_G17_V350G_TOTAL_ACQUISITION_OWNERSHIP=1
CAPX_LIBERO_G17_V350K_TRANSPARENT_NATIVE_TRANSACTION=1
CAPX_LIBERO_G17_V350L_FG_REGENERATION_ON_EXECUTION_FAILURE=1
CAPX_LIBERO_G17_V350M_PROPOSAL_FAILURE_FG_REGENERATION=1
CAPX_LIBERO_G17_V351_CLOSED_TRANSACTION_ROUTER=1
CAPX_LIBERO_G17_V353_COMPLETE_INVERSE_ESCAPE=1
CAPX_LIBERO_G17_V354_FG_NATIVE_RETRY=1
```

PRO Pos/Task runs use the **same flag set**; only `CAPX_LIBERO_SUITE_NAME` switches
(`libero_spatial_swap` / `libero_spatial_task`).

## Explicitly OFF (ablation boundary)

```text
V317_GRASP_RECOVERY_POC
V318_ADAPTIVE_GRASP_RECOVERY
V355_CLEAN_C
V360_TRUE_GR
V361_CORRECTED_TIP_GR
V332_NATIVE_RETRY_ABLATION
CODE_PREFIX_REPLAY / GRASP_REPLAY
later RR tipxy / contact-escape / preGR-closure flags (V333–V358 family as in submit unset block)
```

See [`COMPONENTS.md`](COMPONENTS.md) for the clickable component → code map.
See [`COMPONENT_INVENTORY.md`](COMPONENT_INVENTORY.md) for the denser flag/ablation inventory.

---

## Fingerprinted method files (SHA-256 at `e20e2ea`)

Verified bit-match against the 86-run `SOURCE_SHA256.json`:

| Path | Role |
|------|------|
| `capx/integrations/franka/libero.py` | LIBERO API / acquisition wiring |
| `capx/integrations/franka/acquisition_ownership.py` | Single-writer transaction |
| `capx/integrations/franka/acquisition_extract.py` | Close-time extract |
| `capx/integrations/franka/postclose_acquisition.py` | Post-close route skeleton |
| `capx/integrations/franka/fg_native_retry.py` | Soft-NR hard/soft projection |
| `capx/integrations/franka/grasp_recovery.py` | Present but **not armed** (V318 unset) |
| `capx/integrations/franka/tip_xy_only.py` | Tip-XY / RR support |
| `capx/verification/libero_g17_takeover.py` | FG takeover / student report |
| `scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh` | Job runner |
| `scripts/submit_libero_g17_v332_x_native_retry_spatial_10x10.sh` | Canonical 10×10 submit |

Supporting / interpretable FG surface (same commit, not all in SOURCE ten-file set):

- `capx/verification/libero_g17_fg_lexicon.py` — eye CNL catalog  
- `capx/verification/libero_g17_public_iff.py` — On finish IFF  
- `capx/verification/libero_g17_release_ready.py` — RR  
- `capx/integrations/franka/perceptual_grasp_family.py` — topdown / clearance probe  
- Contracts: `docs/LIBERO_G17_V350_TRANSACTION_OWNERSHIP_CONTRACT.md`, `docs/LIBERO_G17_V354_FG_NATIVE_RETRY_CONTRACT.md`

Unit gate used at submit time:

- `tests/test_libero_g17_acquisition_extract.py`
- `tests/test_libero_g17_v319_transition_closure.py`
- `tests/test_tip_xy_only.py`
- `tests/test_libero_g17_v331_postclose_acquisition.py`
- `tests/test_libero_g17_v350_transaction_ownership.py`
- `tests/test_libero_g17_v354_fg_native_retry.py`

---

## How to check out this method tree

```bash
git clone https://github.com/LiXinchen-ai/libero-g17-anchors.git
cd libero-g17-anchors
git checkout freeze/v332-x-86-pro-triad   # or: git checkout e20e2ea
```

Runtime still requires the separate LIBERO runtime anchor / perception services
used in the original cluster runs (`capx_libero_runtime_anchor_v346_…`); this
repo freezes **method + harness scripts + score manifests**, not the full GPU
cluster environment.

---

## Ablations that preserve claim language

| Ablation | How | Expected use |
|----------|-----|----------------|
| Drop Soft-NR | unset `V354` | Returns toward pre-X / different owner |
| Enable Clean-C GR | set `V355` (+ related) | Distinct claim; not this triad |
| Prefix / grasp replay | set replay envs | Fairness change; not this triad |
| Suite only | `CAPX_LIBERO_SUITE_NAME` | Pos 68 / Task 78 vs canonical 86 |

Do **not** edit fingerprinted files and still call the result “v332_x 86/68/78”.
