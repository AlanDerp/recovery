# P1-P7 Failure Attribution Protocol

This protocol applies only after P0 chain-health and paired-control validation
passes. A trial with a replay miss, invalid perception, renderer instability,
terminated-episode traceback, initial-state mismatch, or frozen-A1 drift is P0
invalid and must not be assigned to P1-P6.

## Categories

- **P1 - GT/extractor error:** a verifier predicate is false, undecidable from
  its declared source, bound to the wrong object, or bound to the wrong time.
- **P2 - replacement error:** the selected span is not the minimal false
  factual claim, the replacement changes non-factual meaning, creates a
  contradiction, or is not persisted consistently.
- **P3 - PT coverage error:** the policy relies on a false factual premise that
  is not explicitly represented on the editable native-envelope surface.
- **P4 - planning error:** the corrected factual state is available and
  correctly represented, but regenerated code does not encode a logically
  valid task-state transition.
- **P5 - continuous execution error:** regenerated code encodes a valid logical
  transition, but perception, grasping, IK, contact, dynamics, or control fails
  to realize it despite valid infra outputs.
- **P6 - non-monotone policy use:** P1-P5 and P7 are excluded, yet replacing an
  explicit false premise with its entailed true counterpart causes the fixed
  action model to select a worse logical policy.
- **P7 - budget/cost exhaustion:** the trial reaches its regeneration, action,
  token, wall-clock, or environment horizon before an otherwise admissible
  recovery can be evaluated. The exhausted limit and the immediately
  preceding unresolved transition must both be recorded. P7 is not a generic
  explanation for a bad policy; it applies only when the limit is causal.

## Decision Order

For every failed treatment trial, record each regeneration as:

`GT snapshot -> native claim -> accepted/no-op edit -> action response ->
operator outcome -> next GT snapshot`.

Assign the earliest supported primary cause in P1 through P5 order, then test
P7. P6 is admissible only with explicit evidence excluding all other
categories. A
trial may list secondary causes, but must have one primary category or remain
`unresolved`; unresolved trials may not be counted as P6.

## Required Evidence

Each seed-level report records:

- registered A1 outcome and frozen response hashes;
- treatment outcome, reward, regeneration count, and action response hashes;
- every verifier proposal, contradiction-gate result, and applied patch;
- GT fact IDs and the object/time instance supporting each accepted patch;
- before/after code and the intended logical transition;
- observed API outputs and physical postcondition;
- the first divergence from frozen A1 and its assigned P category.

The runner writes `TRIAL_ATTRIBUTION_EVIDENCE.json`. Structural certificates
may rule out P0 and structural P2 violations, but they do not by themselves
prove that an extractor predicate is semantically true or that the native
report is complete. Those claims require cited turn-level evidence.

Aggregate reports list counts for P0, P1-P7, and unresolved separately. P0
trials invalidate the run; unresolved trials remain method failures but are not
used to claim a P6 counterexample.
