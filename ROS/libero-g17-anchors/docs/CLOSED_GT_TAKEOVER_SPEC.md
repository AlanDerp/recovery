# Closed-GT Whole-Report Takeover v2.1

## Treatment

This treatment uses native CaP-X M3 through report generation. Immediately
before the stateless action-model request, it replaces exactly the native M3
visual-difference report text with a deterministic controlled-natural-language
rendering of the `closed-gt-v2` semantic basis.

The replacement does not alter the initial policy, executed code, stdout,
stderr, task/API text, image messages, message roles, task configuration,
reward, validator, simulator, or episode state. The native report is generated
but is not action-visible after takeover.

## Public Semantics

The report contains:

1. one mutually exclusive and exhaustive interval grasp outcome for every
   executor-object pair;
2. a 44-field action-visible state projection from the complete 51-field
   validated closure;
3. every changed predicate with its previous and current enum values;
4. an explicit universal statement that every action-visible state predicate
   not listed as changed was unchanged;
5. all three entity pose-motion predicates; and
6. every change in the declared temporal projection, ordered by completed
   simulator step, with simultaneous changes grouped in one event batch.

The full 51-field state and 57-field endpoint transition remain in the private
audit record. The action-visible projection omits holding, payload,
grasp-evidence, and object-semantic aliases and replaces their overlapping
grasp interpretation with one mutually exclusive grasp-contact state per
executor-object pair. This prevents a table-supported grasp contact from being
rendered simultaneously as an unqualified successful grasp.

The interval grasp outcome is a six-value partition over the complete executed
code interval: no executor contact; contact without bilateral grasp; bilateral
grasp without lift then lost; bilateral grasp without lift retained; lift
confirmed then lost; or lift confirmed and retained. It is rendered before the
endpoint closure so that a transient acquisition and later loss cannot be
collapsed into the endpoint alone. The outcome is derived only from the same
validated per-step grasp-contact states and does not name an intended object,
operator, cause, correction, or next move.

The temporal projection covers aperture, executor-object contact, canonical
grasp-contact state, object-object contact and support, table contact, top
access, load bearing, and support stability. It is complete over those
predicates at simulator-step granularity. It states chronology, not causality,
and contains no operator labels or action recommendations.

No action-visible report contains coordinates, dimensions, discretization
thresholds, simulator IDs, reward, success/completion labels, next move,
strategy, code, causal attribution, or camera epistemics.

`top_access` is the currently audited object-access predicate. Generic
`reachable` and `operable` booleans are intentionally absent: the current
controller exposes no audited Cartesian reach envelope, and operability depends
on an action/controller contract. Adding either without such a contract would
turn an exact physical fact into an unvalidated feasibility judgment.

## Fail-Closed Contract

A treatment trial is invalid if:

- closed-GT extraction, validation, transition compilation, or goal-IFF audit
  fails;
- the native M3 report is absent or does not occur exactly once in the action
  request;
- any role, image, or non-text request structure changes;
- the rendered report omits an action-visible field or includes prohibited
  content;
- the post-step observer is absent, fails, or emits a malformed event batch;
- the takeover prompt is not the prompt sent to the action API; or
- paired replay, initial-state, seed, config, dependency, or chain-health
  validation fails.

## Paired M3 Protocol

The A1 arm captures the same closed semantic transitions privately with no
return value and no prompt change. The treatment replays the exact A1 prefix.
If source pixels differ while a normalized M3 request matches, replay is
accepted only when the combined 57-field endpoint transition and ordered event
trace hash matches the private A1 reference. After the first action-request
divergence, replay is blocked for the remainder of that trial.

Smoke tests use one fixed episode seed and run both A1 and treatment in one
allocation. Formal ten-trial runs use the same immutable source snapshot,
model, temperature, global seed, task config, services, and initial responses.
