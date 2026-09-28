# LIBERO G17 Mechanism Verification Contract

This document is the permanent experiment law. It does not change when a new
FG schema is added. Schema versions change atoms. They do not change classes,
outcomes, or truth values.

The Frozen-A0 protocol already recorded the root cause: remote Qwen at
`temperature=0` returns different response hashes for identical request hashes.
Fair replay that only freezes π0, then calls live Qwen after the fork, is the
same hole with a new name.

## Three legal experiment classes

Set `CAPX_LIBERO_G17_EXPERIMENT_CLASS` to exactly one of:

| Class | Live student LLM | What it may write | What it answers |
|---|---|---|---|
| `mechanism_verify` | forbidden | `success` / `fail` cells | Does schema S emit declared atoms IFF gold? |
| `register_live` | once | a complete replay package | Freeze one treatment trajectory |
| `replay_confirm` | forbidden | `success` / `fail` cells | Does the registered trajectory reproduce? |

Unset defaults to `exploratory_live`. That class includes today's fair-K
max-freeze (shared π0, live post-fork Qwen, K replicates). It may estimate
residual variance. It may not enter a mechanism table.

## Outcomes

A finished experiment cell is `success` or `fail`.

There is no `invalid`, `invalid_infra`, `invalid_prefix_miss`, or
`invalid_fg_not_engaged` in a result file.

If prefix, Molmo, SAM3, GraspNet, Pyroki, sitecustomize, or replay is not
ready, the process exits before any trial and writes no result. That is an
infra abort, not an experimental outcome. Fix the runner once. Do not score it.

## Ground truth

Every public atom takes a value from a declared finite domain. `UNKNOWN`,
`UNDECIDABLE`, `AMBIGUOUS`, and `incomplete` are not domain values.

If witnesses cannot close a predicate, remove the predicate from the schema.
Do not emit a hedge note to the student.

`acquisition_incomplete` / `placement_incomplete` as public wording is a
third value. Mechanism mode rejects that wording. The definite states are
Hold / ¬Hold, On / ¬On, In / ¬In, finish authorized / finish denied.

## What may change when iterating FG

- The declared atom set and IFF rules (`schema_version`).
- Offline IFF fixtures and gold vectors.

## What may not change

- Experiment class vocabulary.
- Result vocabulary (`success` / `fail` only).
- Truth vocabulary (no unknown-shaped tokens).
- A new sbatch copy that re-implements Molmo/prefix/sitecustomize gates.
- Treating K live-after-π0 replicates as a mechanism closed claim.

## Iteration loop

1. Change the schema.
2. Run offline IFF (`mechanism_verify`). Pass or fail.
3. If pass, `register_live` once. Export the full replay. Do not pick a
   response after seeing `task_completed`.
4. `replay_confirm` the registered package. Pass or fail.
5. Only then may `task_completed` appear, as a replay-confirmed outcome,
   not as the mechanism metric.

Live fair-K after step 2 is optional variance, never a substitute for 2–4.
