# FGSRP v1.3

FGSRP v1 remains frozen at Git tag `FGSRP-v1`. Version 1.1 is a strict
increment: carrier surfaces and evidence providers are unchanged, while repair
acceptance and run validation are fail-closed.

Version 1.2 leaves the v1.1 carrier, evidence vocabulary, and action-visible
surface unchanged. It makes truth alignment atomic: a directly contradicted
claim must be repaired or the research run is invalid. This prevents a
known-false claim from surviving merely because a proposed edit included an
adjacent true conjunct.

Version 1.3 leaves the v1.2 evidence, gates, and action-visible replacement
transport unchanged. It replaces the sparse repair-only verifier response with
an exhaustive private claim ledger. The host deterministically enumerates every
native-report sentence; the verifier must classify every atomic claim in every
sentence and map each contradicted claim one-to-one to a local repair. Correct,
compatible, undecidable, and non-editable claims remain byte-identical.

## Name

Factual-Grounded Seamless Report Replacement (FGSRP).

## Research question

Can a private GT-backed or human factual verifier improve multiturn recovery by
correcting false clauses in the policy's native recovery report, without adding
a fact packet, world-state ledger, bridge, plan, or action advice?

## Transformation

For native regeneration request `E_t`, editable native report `R_t`, and private
evidence `G_t`, FGSRP produces `T(E_t, G_t)` subject to:

1. If no clause in `R_t` is proven false, `T(E_t, G_t) = E_t` at the serialized
   action-message level.
2. Every changed original clause is negated by selected evidence in `G_t`.
3. Every replacement clause is entailed by the same selected evidence.
4. The replacement is the smallest grammar-preserving local correction.
5. All bytes outside registered report spans are immutable.
6. Each claim is adjudicated atomically. Supported claims are preserved,
   contradicted claims are replaced, and undecidable claims are preserved.
7. A gate-confirmed contradiction that is not yet a valid minimal edit enters
   private atomic refinement. It may not silently degrade to a valid no-op.

FGSRP is open-world and selective. It does not add omitted facts and does not
claim a complete state representation.

## Components

- `EvidenceProvider`: captures private entity-bound snapshots. The initial
  implementation provides simulator evidence for cube stack/restack. Other
  tasks and a human verifier use separate providers without changing the core.
- `CarrierAdapter`: locates the native report and applies host-owned patches.
- private verifier: exhaustively classifies host-enumerated sentence claims and
  proposes exact contradicted spans and minimal replacements.
- private contradiction gate: independently certifies negation, entailment,
  span minimality, and preservation of all other meaning.
- deterministic host: validates provenance, sanitizes output, applies patches,
  and records request-level hashes.
- deterministic contradiction certificate: accepts a gated proposal only when
  the selected predicate value is explicitly expressed by the replacement and
  directly negates the original claim. Visibility, uncertainty, qualitative
  proximity, and untyped free-text transitions abstain until represented by a
  registered predicate. Because this certificate is a necessary acceptance
  condition, host-rejected proposals are removed before the private
  contradiction-gate request and are recorded as adjudicator-skipped.
- atomic refiner: when the gate proves a proposal contains a contradiction but
  rejects its span or preservation properties, reselects only the false atomic
  substring using the same evidence IDs. The refined edit is independently
  gated and host-certified before application.
- atomic transition override: a gate cannot preserve a pure categorical
  no-change claim when the host certificate proves a registered predicate
  changed and the gate separately confirms claim negation and replacement
  entailment. A global no-change sentence is treated as one complete factual
  clause even if the gate labels it non-atomic. This override does not apply to
  ordinary state claims or mixed-evidence repairs.

## Carriers

### M2 comments

`CAPX_FGSRP_CARRIER=m2_comments` verifies only Python comment tokens in the
latest executed block. Executable tokens, stdout, stderr, images, task text, and
API documentation are immutable. Corrected historical comments retain their
post-block temporal provenance and are never rechecked against a later state.

This carrier has low factual recall because student comments are optional. It
is retained as a carrier ablation.

### M3 visual difference

`CAPX_FGSRP_CARRIER=m3_visual_difference` verifies the native visual
differencing report already present in M3's regeneration request. The raw
report is generated before FGSRP runs. FGSRP changes only directly contradicted
physical-state or transition clauses in that report.

Task-completion judgments, strategy, advice, coordinates, parameters, and
uncertain or unsupported claims are not editable.

## Temporal evidence

FGSRP captures an initial private snapshot after reset and one snapshot after
each executed block. Each verification round receives separately identified:

- immediately-before facts;
- immediately-after facts;
- transition records containing before and after values.

A categorical native claim that no change occurred is contradicted by any
registered predicate whose before and after values differ. Its correction may
state only that qualitative predicate transition; numeric state remains
private. If a verifier instead proposes that correction using only registered
transitions whose values did not change, the host downgrades that claim to
compatible-or-undecidable and removes only its matching repair. Other ledger
claims remain independently auditable; one unsupported proposal cannot discard
an otherwise valid exhaustive ledger.

The transition itself proves the native no-change claim false. A replacement
may state either the complete qualitative transition or only its entailed
after-state; it may not state only the stale before-state.

Categorical no-change language includes both global statements and relational
statements that the same relative positions, configuration, or arrangement
were preserved. A static relation such as one object remaining above another
is not by itself a no-change statement under this rule.

An entity registry binds report names to episode entities. Simulator IDs and
numeric state never enter the action request.

## No-cheating boundary

The action model never receives:

- verifier prompts, schemas, verdicts, or labels;
- complete GT packets or summaries;
- exact coordinates, dimensions, thresholds, or simulator IDs;
- reward, `task_completed`, oracle code, task answers, strategy, or next move.

Private evidence may use simulator state to adjudicate qualitative claims. The
host rejects numeric replacement spans and action/strategy vocabulary.

## Fail-closed conditions

The run is invalid if:

- evidence provider, entity registry, snapshot order, or carrier changes within
  a trial;
- a candidate span or source hash cannot be reproduced exactly;
- an edit is not independently certified as a direct contradiction;
- an edit changes executable tokens, message roles, media, or immutable text;
- a no-op changes the action request hash;
- an accepted repair does not change the registered report surface;
- prompt/schema/source/service/replay provenance is incomplete.
- a treatment child process fails or an episode is restarted after an FGSRP
  exception.

Malformed verifier spans, gate responses, and private verifier/gate transport
failures do not restart an episode. They are recorded and produce a
byte-identical native-request no-op for that turn. A transport failure still
invalidates the paired research run through the independent request audit; the
no-op behavior prevents a service outage from being misreported as a policy or
task failure.

If the contradiction gate establishes that an atomic physical claim is false
but no certified minimal correction is produced, the episode may continue with
a byte-identical request for process safety, but the paired research run is
invalid. Such a turn cannot be counted as a valid no-op or experimental result.

## Negative-result attribution

A treatment failure is analyzed per trial before it is attributed to the
method or hypothesis. The audit order is fixed:

1. replay, seed, request ordering, service, and source provenance;
2. before/after GT capture and entity binding;
3. deterministic sentence segmentation and exhaustive claim-ledger coverage;
4. claim time binding and supported/undecidable/contradicted classification;
5. GT negation and replacement-entailment certificates;
6. exact-span patching and the serialized regeneration request sent to the API;
7. the action response induced after the first corrected request;
8. compilation, physical execution, and task validation.

An invalid or unexplained condition in layers 1-6 is an implementation or
measurement blocker, not evidence against factual grounding. A valid layer-7
policy response followed by a layer-8 control failure is recorded separately
from a factual or replacement failure.

## Baseline terminology

The paired control is the native M2 or M3 policy executed under the exact same
clean runtime commit as treatment. It is not called an "original CaP-X
baseline" when the runtime contains audited perception, determinism, or chain
health patches. Manifests record both `control_runtime_git_rev` and the upstream
`baseline_reference_git_rev`.

Ordinary fallible perception remains part of the native policy. FGSRP does not
alter SAM selection, numeric API output, task configuration, reward, or success
validation.

## Initial experimental matrix

The carriers are evaluated separately:

| Native recovery method | Control | Treatment |
| --- | --- | --- |
| M2 | native M2 | M2 + FGSRP comment replacement |
| M3 | native visual differencing | M3 + FGSRP report replacement |

The registered M2 cube controls remain the historical double-6 references.
M3 requires its own frozen control because visual differencing is a different
native recovery method.

### M3 replay certification

Strict source-image fingerprints remain the first replay check. Renderer-level
pixel differences after the same replayed action do not by themselves establish
a policy-state difference, so M3 A1 additionally records a private qualitative
GT transition at each post-action, pre-report boundary. This passive record is
never serialized into an action request and never calls a model.

For a normalized native-report replay with differing pixels, a run is valid
only when the treatment trace has the identical canonical predicate-transition
hash at the same `(trial, turn)`. The certificate is relative to the registered
FGSRP predicate basis; it is not a claim of full continuous simulator-state
identity. Query 0, action requests, missing transitions, and mismatched hashes
cannot use semantic deferral.

For deterministic replay, M2's initial action is native query offset `0`.
M3's initial visual report is offset `0` and its initial action is offset `1`;
both native M3 requests are replay-required. Private verifier and gate calls
use disjoint explicit seed ranges and do not advance the native query index.
Targeted one-trial smoke runs record `fixed_episode_seed` explicitly. When set,
required native query seeds derive from that fixed seed; otherwise they follow
the global-seed trial schedule.
