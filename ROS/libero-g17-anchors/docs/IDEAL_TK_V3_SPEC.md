# Ideal TK v3.3 Public-Lossless Closure

## Purpose

Ideal TK is a factual-only regeneration treatment over native CaP-X M3. Version
v3.3 tests the public-closure hypothesis: the student-visible report must be a
lossless encoding of the complete declared discrete semantic state, not a
selected projection of a complete private state.

The treatment preserves the initial student action, simulator, task/API
contract, images, stdout, stderr, reward, validator, and action parameters. It
inherits the v3.2.1 parser, provider retry, trace, and replacement-chain fixes.

## Public Semantic Invariant

The declared cube semantic state has fifty-one finite-domain slots: thirty-four
primitive slots and seventeen derived slots. Every regeneration report exposes
exactly one sanitized enum value for every slot. Derived composite slots appear
in a typed authoritative section; primitive slots appear in a separate factual
basis section. A primitive observation is evidence for, not an alternative to,
its derived composite state.

The report also losslessly encodes the previous endpoint and all fifty-seven
transition slots. Ordered within-block events and interval grasp outcomes are
kept separate from endpoint state. An event that occurred during the block does
not imply that the same state holds at the endpoint.

The public renderer is accepted only when deterministic decoding proves:

- decoded current state equals the complete GT current state;
- decoded prior endpoint equals the complete GT prior endpoint;
- decoded transition equals the complete GT transition;
- every slot occurs exactly once and every value belongs to its finite domain;
- sanitized enum mapping is bijective and exposes no private entity ID;
- the benchmark goal evaluated from decoded public state agrees iff with both
  private semantic goal truth and the unmodified task-completed label.

The goal definition and completion result remain private. Public goal-IFF is an
audit of information sufficiency, not a task answer shown to the student.

## Carrier Replacement

The unverified initial visual-description suffix is removed while the original
task/API contract remains byte-identical. The native cumulative execution text
is replaced by the unmodified latest code block and its stdout/stderr. Earlier
code is omitted; historical factual outcomes may re-enter only in a section
explicitly typed as historical and not current. The native M3 report is replaced
with deterministic closed GT for the same block.

The report contains the full current-state closure, complete endpoint delta,
ordered factual events, interval grasp outcomes, and at most two prior distinct
historical outcome summaries. It does not contain a next move, strategy, causal
diagnosis, code, numeric state, coordinates, reward, completion label, task
answer, private object ID, or goal formula.

## Private Audit

The complete closed-GT state, decoded public state, endpoint transition,
simulator-step event trace, API-call boundaries, and both private and public
benchmark-goal IFF checks are stored in the private turn audit. Continuous poses
are never persisted. A five-hundred-word target is recorded for attention
analysis, but it is not an execution gate: completeness cannot be silently
truncated to satisfy a local attention target.

The state compiler and private goal-IFF audit are unchanged from closed GT v2.1.
API wrappers observe before and after snapshots but never alter arguments,
return values, exceptions, or code execution. A capture error invalidates the
treatment at regeneration time.

## Fail-Closed Conditions

A trial is invalid if either textual carrier is absent or non-unique, roles or
media change, temporal capture fails, public decoding is not exact, a slot is
missing or duplicated, an enum is invalid or leaks a private ID, either goal-IFF
audit fails, or the request recorded by the model proxy does not match the
takeover prompt. A provider-reported multimodal download timeout is retried with
the byte-identical request and seed; all attempts are audited. An unmarked
response is recovered only when its complete contents are valid executable
Python.

The audited adapter currently covers only `cube_stack` and `cube_restack`.
