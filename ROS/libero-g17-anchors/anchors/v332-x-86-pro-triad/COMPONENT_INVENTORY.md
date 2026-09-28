# Component inventory — v332_x triad (86 / 68 / 78)

**Prefer the clickable browse page:** [`COMPONENTS.md`](COMPONENTS.md).

All paths relative to git SHA `e20e2ea930bf2a751d07af1f83dc951e0681280d`
unless noted.

## A. Fingerprinted core (SOURCE ten-file set)

| File | Interpretable job | Ablatable via |
|------|-------------------|---------------|
| `capx/integrations/franka/libero.py` | Binds acquisition APIs, Soft-NR hooks, ownership | method edit (breaks freeze) |
| `capx/integrations/franka/acquisition_ownership.py` | Single-writer; canonicalize/suppress goto | `V350` / `V350G` / `V350K` / `V350L` / `V350M` |
| `capx/integrations/franka/acquisition_extract.py` | Close-time FG evidence extract | paired with V309/V312 |
| `capx/integrations/franka/postclose_acquisition.py` | Post-close route enum (GR vs regen skeleton) | `V331`; GR arm requires Clean-C flags **off here** |
| `capx/integrations/franka/fg_native_retry.py` | Endpoint kinds + soft projection + regen owner | **`V354` (primary Soft-NR switch)** |
| `capx/integrations/franka/grasp_recovery.py` | Adaptive GR implementation | **`V318` OFF in this freeze** |
| `capx/integrations/franka/tip_xy_only.py` | Tip-XY / place support | RR-related later flags OFF |
| `capx/verification/libero_g17_takeover.py` | FG report takeover into multiturn carrier | FG master path |
| `scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh` | Per-task sbatch runner | harness |
| `scripts/submit_libero_g17_v332_x_native_retry_spatial_10x10.sh` | Canonical spatial 10×10 submit | harness |

## B. FG / interpretation surface (same commit)

| File | Job |
|------|-----|
| `capx/verification/libero_g17_fg_lexicon.py` | Eye-check CNL catalog |
| `capx/verification/libero_g17_public_iff.py` | Public On finish IFF |
| `capx/verification/libero_g17_release_ready.py` | Release-ready / On placement |
| `capx/verification/libero_g17_xy_lock.py` | Identity / XY lock helpers |
| `capx/integrations/franka/perceptual_grasp_family.py` | Topdown / multi-contact clearance |
| `docs/LIBERO_G17_V350_TRANSACTION_OWNERSHIP_CONTRACT.md` | Ownership contract |
| `docs/LIBERO_G17_V354_FG_NATIVE_RETRY_CONTRACT.md` | Soft-NR contract |

## C. Flag matrix (claim-relevant)

### ON — defines this mode

`V309`, `V312`×2, `V313`, `V319`, `V320`, `V331`, `V350`, `V350G`, `V350K`, `V350L`, `V350M`, `V351`, `V353`, `V354`

### OFF — must stay off to keep 86/68/78 claim

`V317`, `V318`, `V355`, `V360`, `V361`, `V332_NATIVE_RETRY_ABLATION`, prefix/grasp replay, and the submit’s unset RR/preGR family (`V333`…`V358` as listed in submit).

### Suite-only knobs (scores change, method SHA fixed)

| Env | Canonical 86 | Pos 68 | Task 78 |
|-----|--------------|--------|---------|
| `CAPX_LIBERO_SUITE_NAME` | unset / `libero_spatial` | `libero_spatial_swap` | `libero_spatial_task` |

## D. Harness scripts in this anchor package

| Script | Suite |
|--------|-------|
| `anchors/…/scripts/submit_libero_g17_v332_x_native_retry_spatial_10x10.sh` | canonical (from `e20e2ea`) |
| `…/submit_libero_g17_v332_x_native_retry_spatial_swap_10x10.sh` | PRO Pos (suite routing; method pin `e20e2ea`) |
| `…/submit_libero_g17_v332_x_native_retry_spatial_task_10x10.sh` | PRO Task (from post-anchor harness worktree; method pin `e20e2ea`) |

Note: PRO harness `harness_git_head` in submit manifests may differ from method SHA
(e.g. swap submit recorded harness `ee9075a` while **method_anchor remained `e20e2ea`**).
Score claims attach to **method_anchor**, not harness tip.

## E. What is intentionally *not* this anchor

- Object 10×10 / RR-In / Soft-NR history-carry (v380–v393+)
- Clean-C / True-GR / adaptive GR score lines
- Fair-K prefix π0 tables (different protocol)
