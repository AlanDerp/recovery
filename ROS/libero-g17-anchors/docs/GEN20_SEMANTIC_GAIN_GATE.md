# Gen20 Semantic-Gain Gate

## Purpose

The semantic-gain gate makes factual-grounding extensions conditional on
information gain rather than on task, seed, benchmark version, or observed
success. The frozen parent report is always rendered from the same GT trace.
An extension may change the student request only when it carries a certified,
time-qualified GT fact that the parent report does not explicitly expose.

## Uniform protocol

Each extension follows one protocol:

1. bind an observed public-operator instance to executors and objects;
2. compile its candidate facts from the closed GT snapshot at that event;
3. enumerate the facts explicitly visible in the frozen parent projection;
4. compute the set difference using typed fact identity and time scope; and
5. apply the extension package iff that difference is non-empty.

If the difference is empty, the complete parent action request is preserved
byte for byte. The trace records both the candidate and applied projections,
their hashes, the novel fact identifiers, and a parent-prompt identity
certificate. A fail-closed validator rejects an unproved edit or a changed
no-op.

## First adapter: bound release

The first adapter binds a payload/reference pair from a relational closed goal,
an observed grasp episode, and the executed release API. Its candidate facts
are the payload/reference XYZ bands, support-region and footprint membership,
support at contact loss, and reference orientation/stability at the release
event. Values already emitted as changes by the parent's ordered event report
do not count as gain; unchanged predicates at that event do.

The release snapshot and phase-bound sparse API ledger form one extension
package. The ledger cannot activate itself. This keeps acquisition-only turns
on the exact parent report while retaining action-conditioned release evidence
when it adds information.

## Generality and exclusions

The gate contains no task-name, object-name, seed, coordinate, reward, success,
or repair-strategy branch. It does not inspect trial outcomes. New operator
adapters must use the same protocol and declare their typed event-time facts;
they may not add a separate prompt policy.

The student never sees private IDs, exact values, goal truth, next actions, or
the gain certificate. Full candidate states remain audit-only. K2 comparison,
initial grounding, preflight, rebound, next move, and effect coaching are off
for this ablation.

## Smoke acceptance

The preregistered frozen-replay gate is:

- stack T2: no release-time gain, so preserve the G16 parent request and pass;
- restack T9: retain the release-evidence rescue;
- restack T10: retain the release-evidence rescue.

Passing these cells permits broader guards, not a paper claim. A full 2x10 run
is authorized only after parent-byte identity, replay, chain health, and the
three seed outcomes all pass.
