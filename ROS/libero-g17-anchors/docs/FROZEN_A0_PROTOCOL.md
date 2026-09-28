# Frozen A0 Protocol

## Root Cause

The earlier paired runner froze only the ten initial policy responses. It did
not freeze A0 regeneration responses. `temperature=0`, one worker, fixed local
seeds, and identical requests were insufficient to make the remote Qwen
service deterministic.

Historical versus later A0 audits showed:

- cube stack: three exact request hashes occurred in both runs; all three
  returned different response hashes;
- cube restack: eight exact request hashes occurred in both runs; six returned
  different response hashes;
- initial state records, initial request hashes, initial response hashes, task
  configs, model name, temperature, and local seeds matched.

The later A0 values of 5/10 and 7/10 therefore do not satisfy the intended
frozen-A0 comparison. Their treatment results must not be compared against the
historical 6/10 baseline as if A0 were fixed.

## Frozen References

- cube stack: job 496763, commit
  `d07c3cd7ebdb10bc5c39009cce69dd9b168c311e`, A0 result
  `1.000/0.612/6`, 48 model calls;
- cube restack: job 496713, commit
  `b9fb2fea09cb455149c84e5da91275144ab67156`, A0 result
  `1.000/0.606/6`, 41 model calls.

Both source pair validations reported `valid`. Their hashes are recorded in
the committed A0 reference JSON files.

## Fail-Closed Requirements

A run that re-executes frozen A0 is valid only if:

1. every A0 model request is served by exact or image-redacted normalized
   request-hash replay;
2. A0 makes no live model call and no replay miss;
3. A0 consumes the complete historical replay entry set exactly once;
4. all ten initial-state record hashes match the reference;
5. every per-trial sandbox result, reward, and completion value matches;
6. the aggregate A0 summary matches the registered 6/10 reference;
7. treatment initial responses and initial states match A0;
8. infrastructure fingerprint and known chain-blocker checks pass.

The reference-only mode does not re-execute A0. It is valid only if:

1. the source A0 run root is the registered historical artifact;
2. the source `PAIR_VALIDATION.json` hash matches the committed reference;
3. that validation reports the same task, trial count, and `valid` status;
4. the complete A0 replay, all initial-state hashes, per-trial outcomes, and
   aggregate summary match the committed reference;
5. the current run records zero A0 model calls and executes treatment only;
6. treatment initial responses and initial states match the registered A0;
7. current treatment infrastructure and known chain-blocker checks pass.

An obsolete or invalid treatment does not automatically invalidate its A0 arm.
Such an arm may be registered only through an A0-only audit that independently
checks all A0 model calls, initial states, task results, configuration semantics,
visual fingerprints, and known chain blockers. The old pair validation is then
recorded as provenance but is not used as evidence for A0 validity.

Treatment regeneration remains live only inside `register_live`. That live
run is a registration source, not a scored mechanism table. After export,
`replay_confirm` and `mechanism_verify` forbid live student LLM. See
`docs/MECHANISM_VERIFICATION_CONTRACT.md`. Fair-K live-after-π0 is
`exploratory_live` and cannot make a mechanism claim.

## Frozen Treatment Registration

Remote responses remain nondeterministic after the A0 branch diverges. Formal
FGSRP evaluation therefore has two phases:

1. run one preregistered treatment trajectory live and accept it only when the
   paired validator passes;
2. export every request/response pair, register the source validation, code and
   prompt hashes, initial states, per-trial outcomes, and FGSRP turn signatures;
3. re-execute the same committed code with complete treatment replay required;
4. reject the run on any replay miss, live model call, request/response sequence
   change, trace change, initial-state change, or outcome change.

The first valid live run is the registration source. A response may not be
selected after observing success. Seed control is necessary but is not treated
as response determinism.

## M3 Visual-Difference Replay

M3 replay uses the same frozen-policy requirements, with an additional image
binding. Its model requests alternate between a visual-difference report and a
text-only policy request. A normalized replay is valid only when the decoded
RGB fingerprints match the registered source frames in this order:

- query `0`: frame `0`;
- query `2k`: frames `k-1`, `k`;
- query `2k+1`: no image.

The proxy verifies this binding before returning a normalized replay response,
and the audit records that verification. Renderer-only differences may be
deferred to the FGSRP validator only for a pre-divergence M3 report. Such a
report is valid only when its complete qualitative `before`, `after`, and
`transition` GT predicate vector matches the registered source vector.

M3 replay is prefix-locked. The first corrected report changes the following
action request; that action runs live and permanently disables all later native
report/action replay for the trial. This prevents a historical report from
being applied after the treatment trajectory has diverged. One-trial diagnostic
references are allowed when their trial count, global seed, source validation,
complete replay, initial state, semantic vectors, and outcome are registered
exactly like a formal reference.

Deferred visual replay is disabled unless the A0 reference contains an explicit
per-trial, per-turn GT semantic-transition certificate. Without that
certificate, an RGB mismatch is a hard replay failure.
