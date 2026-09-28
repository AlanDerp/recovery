# FGSRP-v2 Semantic-IR Alignment

FGSRP-v2 corrects false factual content already present in a native
regeneration report. It does not append grounding, expose a verifier protocol,
or add planning advice.

## Alignment Pipeline

1. The native M2/M3 report is divided into exact host-bound sentence units.
2. A private LLM parser receives the report, entity registry, and finite
   predicate ontology. It does not receive current GT values.
3. The parser maps each physical claim to canonical semantic IR:
   `(literal subject, temporal_scope, predicate, asserted value or transition)`.
4. The host evaluates that IR mechanically against synchronized private GT.
   It does not use English phrase rules or an LLM truth judgment.
5. True IR is preserved byte for byte. Unknown or non-editable content is
   preserved. Only GT-contradicted IR is sent to a private repair writer.
6. The writer receives the exact false span and only its corrected qualitative
   GT facts. It returns a grammar-preserving local replacement.
7. The host applies exact substring replacement and verifies that all roles,
   images, code, stdout, stderr, task text, and non-text request structure are
   unchanged.

The action model sees only the corrected native report in its original carrier.
It never sees the parser, IR, GT catalog, correction protocol, or an added
bridge.

## Host Authority

The host may:

- validate exact sentence and claim provenance;
- validate predicate, value, entity, and temporal identifiers;
- compare canonical IR with GT;
- constrain writes to host-proven false spans;
- reject numeric data, privileged identifiers, task answers, and advice;
- preserve the complete native request on abstention.

The host does not interpret arbitrary English with phrase regexes and does not
choose a robot action, plan, strategy, subtask, or next move.

## Truth Semantics

Each evidence provider supplies a finite qualitative ontology. State predicates
have mutually exclusive canonical values. Transition assertions may state
whether a predicate changed and may bind explicit before/after values. A claim
is contradicted exactly when at least one parsed canonical assertion differs
from synchronized GT. The corrected evidence IDs are derived by the host, not
selected by the action model.

Predicate definitions also declare their subject and required entity roles.
The host accepts an assertion only when its quoted grammatical-subject span is
an alias of the declared entity and that entity is valid for the predicate.
Thus executor motion cannot be evaluated as target-object motion. Whole-scene
no-change clauses use an explicit scene subject and are limited to transition
assertions.

Object support is a derived qualitative topology, separate from axis position
and pairwise contact. Claims such as "on top of" and "resting on" map to the
support predicate rather than being reduced to `relative_z`. For cube tasks,
the support value is derived from sanitized axis, footprint, object-contact,
and table-contact facts; it is not read from reward or task-completion state.

Parser errors remain possible because natural-language interpretation is an
LLM operation. They are isolated in the trace as parser output rather than
being conflated with GT evaluation. Invalid or unmapped parser output fails
closed to an action-visible no-op.

## Sanitization

Action-visible replacements contain no coordinates, dimensions, thresholds,
simulator IDs, reward/success labels, oracle code, APIs, parameters, or action
advice. Numeric state stays private. The entity registry exposes only public
role names such as executor, target object, goal object, and support.

## Replay and Audit

Initial action responses remain frozen from A0. Every turn records parser
request/response hashes, normalized claim IR, host GT evaluation, repair-writer
request/response hashes, exact patches, native/corrected request hashes, and
immutable-surface checks. A no-op must be byte-identical to the native action
request.
