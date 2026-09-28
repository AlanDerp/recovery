# FGSRP-NL v1

FGSRP-NL is an independent seamless-replacement treatment. It does not use
the FGSRP semantic IR, entity aliases, a predicate parser, or task-specific
host rules.

At each regeneration turn, a private verifier receives only the native report,
the public task description, and GT-backed natural-language observations of
the immediately preceding execution. It classifies report claims and proposes
minimal exact-span replacements only for directly contradicted physical facts.

The host is deliberately non-semantic. It verifies exact span provenance,
unique occurrence, non-overlap, local size, immutable numeric content, exact
evidence quotation, and absence of code, advice, coordinates, reward labels,
or task answers. Each decision is accepted or rejected independently. Bytes
outside accepted spans, request roles, and media remain unchanged. If no edit
is accepted, the action request is byte-identical to the native request.

The evidence-provider boundary is pluggable. The initial paper implementation
uses the existing GT cube adapter; replacing that adapter does not change the
semantic replacement protocol.
