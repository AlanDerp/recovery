# PGSVP-v9 AS Materialized PT-GT

## Scope

This version changes only the delivery protocol around the frozen AxisStage
factual kernel. It does not add or remove factual fields, change the verifier
prompt or schema, add next-move planning, or modify task execution.

The raw-delta predecessor is frozen at commit `585e65a`. That version exposed
canonical PT followed by a textual correction list. This version retains PT and
the delta in private audit state, mechanically applies the delta, and exposes
only the resulting worksheet to the action call.

## Per-Regeneration Protocol

For turn `t`:

1. Build the ordinary A1 decision context `C_t`.
2. Replace only its final action request with the twelve-field PT worksheet
   request. Query the same action model to obtain canonical `PT_t`.
3. Give `PT_t` and privileged evidence to the frozen AxisStage verifier. Build
   the complete sanitized verifier state `GT_t`.
4. Compute the changed-field delta privately.
5. Mechanically compute `S_t = apply(PT_t, delta_t)`.
6. Fail closed unless `S_t == GT_t` field by field.
7. Render `S_t` with the same canonical worksheet renderer used for PT.
8. Rebuild the action prompt from `C_t`. Append exactly one assistant worksheet
   containing `S_t`, one minimal state contract, and the original A1
   `REGENERATE/FINISH` request.
9. Persist `GT_t` as the prior worksheet for turn `t + 1`.

The action call never receives raw `PT_t`, `delta_t`, the PT request, the prior
worksheet, closed-value enums, verifier reasoning, or schema-retry history.

## Fixed AxisStage State

The state order remains:

1. `TASK_RELATION`
2. `RUNTIME`
3. `ATTACHMENT`
4. `RELATIVE_X`
5. `RELATIVE_Y`
6. `RELATIVE_Z`
7. `FOOTPRINT`
8. `CONTACT_SUPPORT`
9. `STAGE_POSTCONDITIONS`
10. `EARLIEST_ABSENT_POSTCONDITION`
11. `DOMINANT_ABSENT_FACT`
12. `FALSE_BELIEFS`

The verifier prompt, verifier schema, sanitization rules, deterministic
attachment/spatial overrides, and private GT packet are unchanged from the
raw-delta predecessor.

Frozen AxisStage assets:

- verifier prompt SHA-256:
  `4dc4b4dec1c9dcbef166a8c16af8a3c5a5c564ad329de6cfd60a25b54977c370`
- verifier schema SHA-256:
  `23f165e9ea80179a24c7308f03b7a4587cb24d89fe7306261f872d36d7fc4bad`
- deterministic core extractor SHA-256:
  `28dc73a75d7669936b4d4f6c55dd6f561c0a034a7283c51fdd9551e19bf8d2f9`

## Required Invariants

- The initial student policy is replay-frozen.
- PT is generated from the ordinary action evidence with no action directive.
- `apply(PT, delta) == verified_state`.
- The materialized state equals the verified state.
- Rendering the materialized state equals rendering verified state directly.
- Exactly one materialized state occurs as an action-visible assistant message.
- Raw PT occurs zero times as a separate action-visible message when any field
  was corrected.
- The delta header, full verifier report, PT request, prior worksheet, enums,
  privileged packet, and retry history occur zero times in the action prompt.
- The materialized state contract and action instruction each occur once.
- The previous verified state equals the next PT request's prior state.
- No coordinates, reward/completion labels, hidden IDs, oracle code, action
  advice, or next-move plan are delivered.
- Any replay, protocol, truth-alignment, chain-health, or audit failure
  invalidates the run instead of becoming a task failure.

## Interpretation

This protocol operationalizes PT-GT alignment at an explicit state boundary. It
does not claim to edit hidden model activations. The stateless action call sees
one corrected state snapshot rather than two conflicting textual states.

For the causal ablation, an AS-PT control must use the identical worksheet
renderer, message roles, contract, and action request while substituting
canonical PT for the materialized GT state. The only action-visible difference
between AS-PT and AS-PTGT must then be corrected field values.
