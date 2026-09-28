# LIBERO G17 v359: v332 + X + clean C asset audit

## Baseline and acceptance rule

The only performance baseline is method commit
`9dade5af9d4800f04503f87faba13193850447b6` (v332). Its fair Spatial
10x10 score is 82/100, with per-task scores
`[10, 10, 7, 8, 2, 10, 7, 9, 10, 9]`.

A post-v332 component enters X only when it is either:

1. a state/ownership defect repair that preserves the v332 native proposal
   and executor; or
2. a locally demonstrated positive transition mechanism with a typed fallback
   that cannot fail-close the native outer loop.

Task-specific candidate ranking, collision certification, new motion planners,
and components whose only evidence is audit visibility do not enter X.

## X: retained assets

### Transaction correctness

- Single writer per acquisition attempt (`e7c3e1b`, `30c2a57`).
- Native proposal completion is recorded, but the transparent native executor
  remains the physical executor (`2f3cf60`, `5638c40`).
- Execution/proposal failure retires the writer and returns to fresh FG
  regeneration instead of leaving stale ownership (`370fb39`, `6c301ae`).
- Every physical close has one post-close classification and successor;
  DefiniteMiss is a retry state, not a terminal exception (`c06c9ab`,
  `61b7d5b`).

These repairs target the unexplained second writer, stale transaction,
pose/quaternion override, and missing-observation bugs seen in v332 task 4.
They do not impose contact as a precondition and do not replace the native
grasp proposal family.

### Recovery interface correctness

- Measured inverse ingress completes before transport admission (`20c369f`).
- Endpoint FG is the sole post-close router and is projected into the next
  regeneration turn (`a22f329`, `9773bae`, `8e7c8d3`).
- Adaptive GR owns only `EmptyTargetNear -> TargetHeld`; every other endpoint
  and every GR failure returns to native retry (`0896bff`).
- Robot-measured tip progress is not discarded when the local tip command
  reports a partial/rejected remainder (`62623f0`). This uses proprioception
  and FK, not continuous object GT.

## Explicitly excluded

- v333-v346 RR command-execution experiments are outside this acquisition
  composition. A result directory labelled v346 also totals 68/100, but its
  acquisition lineage contains the older v320/v332 stack; that aggregate is
  not a paired causal attribution to RR. These experiments are therefore
  neither promoted nor declared falsified here. v332 RR remains unchanged
  until a separate boundary-complete RR audit.
- Contact-aware and connected pregrasp certification (`75c9c5e`, `5a1b9ed`):
  prevention gates with false-negative surface and no benchmark domination.
- Post-v332 identity fallback (`a3696e1`, `45f440a`): useful audit mechanism,
  but the v342 task-4 K10 scored 0/10. v332's already validated identity and
  sample-grasp-pose assets remain.
- Forced blocked connected retry (`4016eee`): paired task-4 K10 was 3/10 versus
  2/10 control, insufficient to justify a new hard motor policy. Soft endpoint
  FG remains; the hard directive is removed from v359.
- v349 dominance shadow, v350 contact/collision policy variants, v356 trusted
  SE(3) replacement, v357 factorized route, and v358 blocked recovery. They are
  forensic/experimental assets, not release assets; none has benchmark-level
  evidence of domination and several have direct frozen-trial regressions.

## Effective state machine

```text
AcquireNeeded
  -> native proposal + native physical attempt (one writer)
  -> close-time finite FG
       TargetHeld
         -> measured inverse escape -> native transport -> v332 RR
       EmptyTargetNear
         -> measured inverse retreat -> one adaptive GR attempt
              TargetHeld -> measured GR escape -> transport -> v332 RR
              otherwise  -> fresh FG native retry
       WrongObjectHeld
         -> open -> inverse retreat if available -> fresh FG native retry
       EmptyBlocked / EmptyClearOffset
         -> open -> inverse retreat if available -> fresh FG native retry
       proposal unavailable / ingress unreached
         -> no fabricated success -> fresh FG native retry
```

The composition is deliberately asymmetric: recovery GR is inserted only
after an observed NearMiss; it never gates the native first attempt. Native
retry is the full-domain fallback. v332 RR is unchanged.

## Required validation

1. Unit contract: no bypass, stale writer, unclassified close, hard connected
   directive, or continuous object GT consumer.
2. Frozen task-4 smoke: one direct-success case and NearMiss/GR cases.
3. Task-4 K10 against v332's exact fixed seeds.
4. Only if K10 is non-regressive, Spatial 10x10 against v332's exact harness
   and runtime anchor. Promotion requires no new task regression hidden by the
   aggregate score.
