# Ideal TK V4: Canonical State and Ordered Delta

## Experimental intent

Ideal V4 is a strict, additive refinement of the frozen closed-GT takeover
v2.1 treatment (`closed-gt-takeover-v2.1`). It preserves the v2.1 student
interface and factual projection. It does not compose the later ledger,
latest-block-only context, full 51-field public state, or standalone API-effect
section. The rejected composed implementation is preserved only by the
`ideal-tk-v4-composed` tag.

V4 tests whether source-aligned temporal facts improve recovery without adding
advice. Its student-visible report contains:

1. the v2.1 exhaustive, mutually exclusive grasp outcome for each
   executor-object pair;
2. the v2.1 irredundant 44-field current semantic state;
3. the v2.1 endpoint delta and qualitative pose changes; and
4. the v2.1 simulator-step-ordered temporal predicate delta, with the active
   public API name attached inline when an event occurred inside one.

The API annotation is provenance, not a second factual representation. Removing
every annotation must recover the frozen v2.1 report byte for byte.

## Controlled difference

The native M3 visual-difference report is the only replaced request carrier,
and it must occur exactly once. System text, roles, media, task/API contract,
complete executed-code history, stdout, stderr, and decision instruction remain
byte-identical. Replacing the V4 report back with the native report must exactly
recover the original action request.

The treatment does not expose reward, completion, exact coordinates, simulator
IDs, dimensions, continuous action parameters, next move, strategy, causal
diagnosis, oracle code, or task answers.

## State and time semantics

The current-state section is a total assignment over the frozen v2.1
action-visible basis. Changed and unchanged fields are disjoint and exhaustive.
Redundant semantic aliases remain private; the complete 51-state and
57-transition structures are retained in the audit.

Temporal information is a state difference, not a prose history. After every
completed simulator step, V4 compares the new compiled semantic snapshot with
the preceding snapshot and emits only predicates whose enum value changed.
Event batches retain simulator order. The API wrapper maintains a call stack;
the active public API name at the exact observation is attached to that event.
No API effect is inferred after the fact.

The task goal is checked privately as an IFF between the unchanged benchmark
validator and the closed-GT goal definition. Neither the goal formula nor its
truth value is visible to the student.

## Fail-closed invariants

A V4 run is invalid unless all of the following hold:

- the simulator-step observer and public API wrappers were installed;
- each regeneration has exactly one native-report replacement;
- round-tripping that replacement restores the complete native prompt;
- the visible state basis is total, exclusive, and unchanged from v2.1;
- ordered event batches and interval grasp outcomes are structurally valid;
- removing inline API provenance produces the exact v2.1 report hash;
- the report contains no standalone API-effect section;
- the prompt sent to the action API matches the audited takeover prompt;
- M3 prefix replay and deferred semantic replay checks pass; and
- all no-leak and private goal-IFF checks pass.

The audited adapter currently covers `cube_stack` and `cube_restack` only.

## Local operator-effect alignment ablation

`CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT=1` enables a default-off, post-hoc
ablation. It does not ask the student to emit intent and does not inspect prior
regeneration prose. A deterministic AST pass reads only public API calls and
literal public object names from the latest executed code block. The runtime
recorder separately captures the public APIs that were actually invoked.

For a uniquely bound executor-object manipulation interval, the ablation
compares the finite effects implied by the planned public-operator sequence
with GT-observed interval outcomes: bilateral grasp establishment, lift before
release when starting from table support, and grasp retention during motion.
Loss caused by `open_gripper` is an intentional release and is not reported as
a contradiction. Unknown outcomes, ambiguous object binding, and absent
operator evidence fail closed rather than producing a student-visible claim.

Only GT-proven contradictions are appended. When no contradiction exists, the
student-visible report and complete action prompt remain byte-identical to
canonical V4. The appendix contains no next action, repair strategy, numeric
state, reward, completion label, private entity ID, or task-specific plan.
## Local operator-effect alignment ablation

The optional `CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT=1` smoke ablation compares
public operator effects inferred from the latest executed code with the
ordered closed-GT trace. Matching effects are byte-silent. The expression
mode is selected independently with
`CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT_MODE`:

- `first_failure`: expose only the first GT-contradicted effect.
- `verified_prefix`: expose the confirmed prefix and first contradiction.
- `release_support`: emphasize support by a uniquely referenced object after release.
- `grasp_loss_only`: report only GT-proven grasp loss during executed post-grasp
  motion before release; otherwise remain byte-silent.
- `instance_bound_grasp_loss_only`: apply the same report only when loss is
  bound to the last executed close-to-release API-call instance window; losses
  from earlier grasp episodes remain byte-silent. Acquisition may occur on the
  closing call, but a reported loss must occur on a later call before release.
  Missing or ambiguous event-to-call provenance is byte-silent and audited.
- `post_release_support_only`: report only GT-proven failure to establish support
  by a uniquely referenced object after an executed release; otherwise remain
  byte-silent.
- `compact_outcome`: compress confirmed effects and the first contradiction.
- `effect_ledger`: expose each evaluable effect as confirmed or contradicted.

The release reference is bound only when the executed code contains one
unique non-payload object reference between grasp and release. Ambiguous
bindings remain silent. The ablation does not expose coordinates, private
IDs, reward, completion, task answers, or a repair action.
