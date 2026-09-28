# PGSVP L4 Bridge Approval

Status: frozen current-state-only bridge under
`docs/PGSVP_V9_FROZEN_SPEC.md`.

Runtime asset:
`capx/verification/pgsvp_v9_public_bridge.txt`

## Proposed Action-Visible Text

```text
PGSVP GT ALIGNMENT CONTRACT

The attached report is authoritative GT-backed factual context for the current
state after the latest executed code. A conflicting belief about a reported
predicate is replaced by the reported value before regeneration. The report
describes facts, not an action, strategy, or control parameter.
```

The fixed-order `GT_BACKED_VERIFIER_FACTS` report follows this text.

## Scope

The bridge changes epistemic precedence and current-state interpretation only.
It does not add a transition, next-move field, operator sequence, repair plan,
strategy, coordinate, or privileged outcome label.

`FALSE_BELIEFS` names only closed physical worksheet fields whose asserted PT
value directly disagreed with current GT. `unknown` PT values are not treated
as false beliefs.

## Injection Point

The contract and factual report are inserted together after ordinary execution
evidence and before the M2 regeneration decision request. The shadow PT and raw
privileged packet remain outside recursive action history.

## Required Audits

- exact bridge-text SHA-256 recorded in the run manifest and every verifier
  trace;
- exactly one bridge occurrence in each treated regeneration prompt;
- no bridge occurrence in A1 or the frozen initial policy request;
- independent rejection of coordinates, outcome labels, action APIs, action
  advice, and persistent-bias claims;
- explicit field provenance and current-state scope validation.

The runtime audits the fully constructed student decision prompt before the
student request is sent. A missing or duplicated contract or factual report
fails the verifier turn.
