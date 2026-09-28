# Envelope Replacement v1.2

## Question

Does correcting only explicit false physical-state statements already present
in the most recently executed code block improve recovery without adding a
factual packet, bridge, student worksheet, plan, or action advice?

## Frozen comparison

- A1 and treatment use the same source commit, task YAML, service revisions,
  initial simulator states, initial M2 responses, temperature, worker count,
  seeds, and regeneration limit.
- Registered cube-stack and cube-restack replay banks freeze the initial policy.
- The Slurm submission exports an explicit Git revision. The wrapper
  materializes the shared paired runner from that revision, verifies it still
  equals the clean worktree HEAD when execution starts, archives the runner
  read-only in the pair directory, and records its hash. A queued job therefore
  fails closed rather than silently adopting a later revision.
- The treatment also receives A1's complete request-hash replay bank. A silent
  regeneration returns A1's exact response. Raw-hash replay is preferred. A
  normalized-hash replay is allowed only when all non-image request content is
  identical and the archived A1/treatment RGB observation differs in at most
  32 of 786,432 channel values; the validator records every changed-channel
  count and maximum magnitude. This bounds observed GPU-render pixel noise
  without ignoring a materially different scene. The first factual replacement
  changes the normalized request hash and branches to a live regeneration.
- The treatment runs only after the ordinary A1 regeneration prompt has been
  constructed.

## Editable surface

The private verifier may inspect only exact Python comment tokens from the most
recently executed code block. Earlier code blocks are immutable verified
history. Console stdout is evidence for the student but is not editable.

System text, task goal, API documentation, executable Python tokens, stdout,
stderr, images, decision instructions, message roles, and message count are
immutable.

## Repair protocol

The editor receives latest-block candidate chunks and the private canonical
state catalog measured after that block finished. It may propose a
contradicted chunk only by exact ID, quote the exact smallest contradicted
clause, and select at most four catalog predicates: the smallest sufficient
set. A separate private contradiction gate classifies each proposal as
contradicted, supported/equivalent, compatible, or not decidable. More-specific
GT and missing facts never count as contradictions. A proposal is applied only
when the gate certifies both direct logical contradiction and a minimal complete
clause. It cannot write public text or hashes.

The editor also proposes the smallest grammar-preserving local correction. The
gate must certify that the selected GT both falsifies the original wording and
entails the correction, and that substituting it preserves every other word,
modifier, relation, and clause. The host binds both exact strings to the
original chunk, computes their hashes, verifies the unique source location, and
performs only that substitution. It adds no verifier label, state summary,
packet, bridge, explanation, or instruction. This remains editing of the native
regeneration envelope; it does not construct a closed state representation, PT
worksheet, or world-state ledger.

Each block is verified exactly once, immediately after execution. Its corrected
text is persisted in a host-owned, hash-audited history and is never rechecked
against a later state. This yields an ordered post-block fact chain rather than
rewriting old claims with the newest GT. When no current or prior correction
exists, the action request must be hash-identical to native A1. No full GT
packet, verifier response, schema, bridge, PT worksheet, delta protocol,
reward, success label, coordinate, oracle code, or plan is visible to the
action student.

The visible replacement is strictly qualitative. It may contain object names
and closed-vocabulary predicates, but no coordinates, dimensions, poses,
thresholds, block indices, control parameters, or other numeric data. Numeric
stdout remains byte-identical and is used only as native A1 evidence. Offline
chain-health checks may reject semantically invalid numeric outputs, but they
never repair, hide, or return those values to the student.

## Scope

This first version can repair only explicit textual current-state errors covered
by the cube current-state catalog. An incorrect belief inferred internally from
an image but never externalized in an editable chunk is outside this ablation.

## Fail-closed conditions

The run is invalid if:

- a repair references an unknown chunk, source hash, or fact ID;
- an applied repair lacks an accepted contradiction-gate decision;
- the selected GT does not entail both the negation of the original text and
  the replacement text;
- the substitution changes compatible, supported, or undecidable meaning;
- repair spans overlap;
- a no-op changes the native prompt hash;
- a silent regeneration misses both raw and visually bounded normalized A1
  replay, or a grounded regeneration unexpectedly hits either replay;
- the archived top-level runner is absent, writable, or inconsistent with the
  recorded run commit/hash;
- a repair changes message/media structure, stderr, task/API text, or executable
  code tokens;
- verified block history is missing, reordered, rechecked, or hash-inconsistent;
- a cube OBB API call silently returns a full extent outside the broad
  chain-health domain `[0.005, 0.20]` meters;
- old PGSVP/core-FG visible markers occur in treatment prompts;
- replay, initial-state, infrastructure, source snapshot, or chain-health
  validation fails.
