# LIBERO G17 Layer-1 Sparse + Near-Miss Discrepancy (v2.7)

## Goal

Close remaining Layer-1 readability holes without expanding the bound-pair
ontology or promoting intention obligations:

1. **Sparse endpoint** — emit only the active ME gap subset (Acquire or Place)
   plus the public On/In IFF block.
2. **Near-miss RelX/RelY discrepancy** — when contact holds but On fails on the
   XY band, surface factual axis discrepancies and clarify that
   `supported_by` is not the finish basis.

## Env / schema

| Arm | Env | Schema |
|-----|-----|--------|
| `v26` | `FG=1`, `L1_SPARSE=0` | `…-v2.6-success-witness` |
| `v27` | `FG=1`, `L1_SPARSE=1` | `…-v2.7-sparse-nearmiss` |

`CAPX_LIBERO_G17_WORKGAP_ME` stays **off** (v2.5 not promoted).

## Freeze gate (unchanged)

- Success stop: terminal witness
  `finish_authorized ∧ bddl_atom_true ∧ private_task_completed`
- Gain claims: exact request-hash replay (capture outcomes == exact outcomes)

## What this does **not** do

- Distractors / occlusion algebra
- Active Transition intention layer (`required RelX→IN` obligations)
- Soft finish / workgap-as-IFF

## Exact note (v2.5/v2.6 519406/519407)

Exact-rerun failed with `request_hash_replay_required_miss` (HTTP 409) on the
first post-π0 decision turn — residual Molmo / prompt divergence under
`require-all-request-hash-replay`. Re-capture under this code for v26/v27;
if misses recur, freeze Molmo responses before claiming schema effect.
