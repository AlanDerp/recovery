# LIBERO G17 Fair FG Max-Freeze Protocol

## Goal

Estimate whether public-IFF FG (`v2.4`) improves outcomes versus native baseline
under the strongest practical freeze stack, with **unique treatment delta = FG**.

## Arms

| Arm | `CAPX_LIBERO_G17_FG` | Workgap | Notes |
|-----|----------------------|---------|-------|
| `baseline` | `0` | off | native VLM post-π0 |
| `v24` | `1` | off | public On/In IFF takeover |

Workgap / Ideal-TK / cube takeover are **out of scope** for this contrast.

## Frozen stack

1. M3 baseline fingerprints (sbatch / yaml / `SEED_TABLE`) read-only.
2. Code-prefix package `code_prefix_replay_v1` (π0 / CB0 identical across arms).
3. Same two inits (`libero_spatial_0` trials 1–2); init-sha gate fail-closed.
4. Same local seed stack: `CAPX_GLOBAL_SEED=20260718`, `QWEN_REQUEST_SEED`,
   `CAPX_DETERMINISTIC_TRIAL_SEEDING=1`, `temperature=0`, `num_workers=1`.
5. Same infra endpoints (local Qwen proxy, Molmo, SAM3, GraspNet, Pyroki).

## What cannot be frozen

Remote Qwen at temperature 0 is still nondeterministic across identical
request seeds (same finding as Frozen-A0). Molmo has no response-hash replay
here. Residual variance is estimated by **K identical-stack replicates**.

Do **not** offset post-fork request seeds in the primary contrast; that would
confound the FG delta.

## Metrics (per replicate, then rates over K)

- Trial 1 success / rescue vs control 517994 fail cell
- Trial 2 success / guard vs control 517994 success cell
- Joint positive: t1 success and t2 success
- Hard gates: `valid_prefix_arm`, π0 sha identical across completed arms

## Claim rule

A fair FG gain claim requires valid replicates, identical π0, and a rate
advantage for `v24` over `baseline` on the pre-registered metrics. A single
smoke success is not a claim.
