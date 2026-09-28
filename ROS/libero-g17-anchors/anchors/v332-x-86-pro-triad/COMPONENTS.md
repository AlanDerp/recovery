# Method component index — v332_x spatial triad (86 / 68 / 78)

Browse the **method code that actually arms this freeze**, grouped by interpretable repair / cognition components.

| | |
|--|--|
| **Scores** | Canonical spatial **86** · PRO Pos **68** · PRO Task **78** |
| **Method SHA** | [`e20e2ea930bf2a751d07af1f83dc951e0681280d`](https://github.com/LiXinchen-ai/libero-g17-anchors/tree/e20e2ea930bf2a751d07af1f83dc951e0681280d) |
| **Tag** | `handoff/v332-x-86-method` |
| **Branch tip (docs + manifests)** | `freeze/v332-x-86-pro-triad` |
| **Tree root** | [repo @ `e20e2ea`](https://github.com/LiXinchen-ai/libero-g17-anchors/tree/e20e2ea930bf2a751d07af1f83dc951e0681280d) |

All links below pin **`e20e2ea`** (immutable method tree). Do not confuse with later worktrees (object Soft-NR, Clean-C, hard-GR, etc.).

```text
AcquireNeeded
  -> ownership binds one transaction
  -> pregrasp + ingress + close
  -> FG endpoint classification
       TargetHeld  -> inverse escape -> transport -> RR / On
       non-held    -> open (+ inverse) -> Soft-NR / regen
```

---

## FG takeover

Student-visible FG report / multiturn carrier replacement.

| Piece | Path @ `e20e2ea` |
|-------|------------------|
| **FG takeover** | [`capx/verification/libero_g17_takeover.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/verification/libero_g17_takeover.py) |
| Adapter / attachment roles | [`capx/verification/libero_g17_adapter.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/verification/libero_g17_adapter.py) |
| FG projection helpers | [`capx/verification/libero_g17_fg_projection.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/verification/libero_g17_fg_projection.py) |

## FG lexicon / eye CNL

| Piece | Path @ `e20e2ea` |
|-------|------------------|
| **Eye-check CNL catalog** | [`capx/verification/libero_g17_fg_lexicon.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/verification/libero_g17_fg_lexicon.py) |

## Public On IFF

| Piece | Path @ `e20e2ea` |
|-------|------------------|
| **Public On finish IFF** | [`capx/verification/libero_g17_public_iff.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/verification/libero_g17_public_iff.py) |

## Native retry / Soft-NR (V354)

Primary non-held owner for this freeze: endpoint kinds → soft projection → regen.

| Piece | Path @ `e20e2ea` |
|-------|------------------|
| **FG-native-retry** | [`capx/integrations/franka/fg_native_retry.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/integrations/franka/fg_native_retry.py) |
| Contract | [`docs/LIBERO_G17_V354_FG_NATIVE_RETRY_CONTRACT.md`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/docs/LIBERO_G17_V354_FG_NATIVE_RETRY_CONTRACT.md) |
| Unit tests | [`tests/test_libero_g17_v354_fg_native_retry.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/tests/test_libero_g17_v354_fg_native_retry.py) |

Armed by: `CAPX_LIBERO_G17_V354_FG_NATIVE_RETRY=1`.

## Post-close routing

| Piece | Path @ `e20e2ea` |
|-------|------------------|
| **Post-close acquisition skeleton** | [`capx/integrations/franka/postclose_acquisition.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/integrations/franka/postclose_acquisition.py) |
| Transition closure | [`capx/integrations/franka/transition_closure.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/integrations/franka/transition_closure.py) |
| Related tests | [`tests/test_libero_g17_v331_postclose_acquisition.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/tests/test_libero_g17_v331_postclose_acquisition.py) · [`tests/test_libero_g17_v319_transition_closure.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/tests/test_libero_g17_v319_transition_closure.py) |

## Release-ready / place

| Piece | Path @ `e20e2ea` |
|-------|------------------|
| **ReleaseReady** | [`capx/verification/libero_g17_release_ready.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/verification/libero_g17_release_ready.py) |
| Tip-XY / place support | [`capx/integrations/franka/tip_xy_only.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/integrations/franka/tip_xy_only.py) |
| Tests | [`tests/test_libero_g17_release_ready.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/tests/test_libero_g17_release_ready.py) · [`tests/test_tip_xy_only.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/tests/test_tip_xy_only.py) |

## Identity / XY lock

| Piece | Path @ `e20e2ea` |
|-------|------------------|
| **XY lock / same-name identity** | [`capx/verification/libero_g17_xy_lock.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/verification/libero_g17_xy_lock.py) |
| Tests | [`tests/test_libero_g17_xy_lock.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/tests/test_libero_g17_xy_lock.py) |

## Ownership (single-writer)

| Piece | Path @ `e20e2ea` |
|-------|------------------|
| **Acquisition ownership** | [`capx/integrations/franka/acquisition_ownership.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/integrations/franka/acquisition_ownership.py) |
| Contract | [`docs/LIBERO_G17_V350_TRANSACTION_OWNERSHIP_CONTRACT.md`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/docs/LIBERO_G17_V350_TRANSACTION_OWNERSHIP_CONTRACT.md) |
| Tests | [`tests/test_libero_g17_v350_transaction_ownership.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/tests/test_libero_g17_v350_transaction_ownership.py) |

Armed by: `V350` / `V350G` / `V350K` / `V350L` / `V350M` (+ `V351` closed router).

## Acquisition extract / close evidence

| Piece | Path @ `e20e2ea` |
|-------|------------------|
| **Close-time FG extract** | [`capx/integrations/franka/acquisition_extract.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/integrations/franka/acquisition_extract.py) |
| Tests | [`tests/test_libero_g17_acquisition_extract.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/tests/test_libero_g17_acquisition_extract.py) |

## Topdown / clearance proposal family

| Piece | Path @ `e20e2ea` |
|-------|------------------|
| **Perceptual grasp family** | [`capx/integrations/franka/perceptual_grasp_family.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/integrations/franka/perceptual_grasp_family.py) |

## LIBERO API wiring (composition root)

Where the components above are hooked into the Franka/LIBERO runtime path.

| Piece | Path @ `e20e2ea` |
|-------|------------------|
| **LIBERO integration** | [`capx/integrations/franka/libero.py`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/capx/integrations/franka/libero.py) |

---

## Submit / runner (how 86 · 68 · 78 were launched)

Method-tree submits @ `e20e2ea`:

| Role | Path |
|------|------|
| Canonical spatial 10×10 | [`scripts/submit_libero_g17_v332_x_native_retry_spatial_10x10.sh`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/scripts/submit_libero_g17_v332_x_native_retry_spatial_10x10.sh) |
| Job runner | [`scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh`](https://github.com/LiXinchen-ai/libero-g17-anchors/blob/e20e2ea930bf2a751d07af1f83dc951e0681280d/scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh) |

PRO suite scripts live on this docs branch (method pin still `e20e2ea`):

| Suite | Path |
|-------|------|
| Canonical (copy) | [`anchors/…/scripts/submit_…_spatial_10x10.sh`](scripts/submit_libero_g17_v332_x_native_retry_spatial_10x10.sh) |
| PRO Pos (`spatial_swap`) | [`scripts/submit_…_spatial_swap_10x10.sh`](scripts/submit_libero_g17_v332_x_native_retry_spatial_swap_10x10.sh) |
| PRO Task (`spatial_task`) | [`scripts/submit_…_spatial_task_10x10.sh`](scripts/submit_libero_g17_v332_x_native_retry_spatial_task_10x10.sh) |

---

## Flag surface for this mode

**ON:** `V309`, `V312`×2, `V313`, `V319`, `V320`, `V331`, `V350`, `V350G`, `V350K`, `V350L`, `V350M`, `V351`, `V353`, **`V354`**.

**OFF for claim:** adaptive GR / Clean-C / True-GR / replay — see [`README.md`](README.md) and [`COMPONENT_INVENTORY.md`](COMPONENT_INVENTORY.md).

Scores / fingerprints: [`results/`](results/) · [`manifests/`](manifests/).

---

## Checkout

```bash
git clone https://github.com/LiXinchen-ai/libero-g17-anchors.git
cd libero-g17-anchors
git checkout e20e2ea          # method code only
# or
git checkout freeze/v332-x-86-pro-triad   # method + this index / manifests
```
