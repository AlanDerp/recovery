# PGSVP-v9 Frozen Method Contract

## Purpose

This branch evaluates the historical PGSVP-v9 factual-grounding architecture
after correctness and interface repairs. It is not a planner, coach, action
critic, or transition gate.

The method is frozen before outcome evaluation. After the freeze smoke, a
method change requires a new named version and a new experiment series.

## Fixed Public Interface

Every treated regeneration receives exactly one bridge followed by exactly one
fixed-order report with the historical twelve v9 fields:

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

The report describes the current state after the latest execution. It does not
publish a historical transition, next move, operator commitment, exclusion
ledger, action parameter, code fragment, reward, completion label, or oracle
answer.

## Truth Sources

The six closed physical fields are deterministic simulator predicates:

- attachment: `empty`, `holding_target`, or `holding_other`;
- each relative axis: negative offset, aligned band, or positive offset;
- footprint: target center inside or outside the support footprint;
- contact: target contacting or separated from the support.

`TASK_RELATION` comes from the public task definition. `RUNTIME` comes only
from visible stderr. The three stage/absence fields are deterministic,
current-state summaries of the six closed predicates. `FALSE_BELIEFS` lists
only closed physical PT fields whose asserted value directly conflicts with
GT; `unknown` is silence, not an error.

No model-authored public value can override these sources.

## PT-GT Protocol

1. A shadow request asks the student model to fill the closed perceived-truth
   worksheet from its ordinary evidence.
2. The worksheet is projected onto the fixed schema. Missing or invalid closed
   values become `unknown`.
3. A separate verifier request receives PT and privileged evidence for audit.
4. Local deterministic extractors produce the public report and direct PT-GT
   differences.
5. PT and verifier conversations remain outside recursive action history.
6. The bridge establishes only epistemic precedence: reported predicates are
   authoritative current facts, not strategy.

The verifier request cannot fail open into model-authored guidance. Malformed
protocol output receives a bounded retry; a persistent failure invalidates the
run.

## Allowed Repairs

- deterministic GT extraction and enum validation;
- public object-role mapping from the task definition;
- current-state time-scope correction;
- PT isolation and closed PT-GT comparison;
- sanitization and leakage rejection;
- exactly-once bridge/report injection and prompt hashing;
- replay, process, storage, service, and validator infrastructure.

## Forbidden Additions

- `next_move` or a task plan;
- canonical operator or transition selection;
- action commitment or execution gating;
- failed-action exclusion memory;
- strategy, code, coordinates, or control parameters;
- reward, success, completion, or oracle labels;
- task-specific recovery rules.

## Evaluation Freeze

The paper comparison uses the registered ten-trial A1 references for global
seed `20260718`:

- cube stack: code success `1.000`, average reward `0.612`, completed `6`;
- cube restack: code success `1.000`, average reward `0.606`, completed `6`.

Treatment must replay the same ten initial student responses and match the
registered initial-state trajectory. Temperature is zero and worker count is
one. Smoke validates chain health and protocol integrity only; task outcomes
are not used to tune the frozen method.
