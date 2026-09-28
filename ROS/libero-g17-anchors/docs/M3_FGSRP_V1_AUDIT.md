# M3 FGSRP v1 Audit

## Baseline provenance

The historical M3 result remembered as `cube_restack=3/10` and
`cube_stack=6/10` is job `464753`:

- Git commit `53e9966`, dirty worktree;
- `temperature=0.7`;
- native `_vdm.yaml` task configurations;
- no frozen episode, API-request, or ContactGraspNet seed trajectory.

The FGSRP-v1 formal controls in jobs `500239` and `500242` are different:

- Git commit `8c15b04`, clean worktree;
- `temperature=0`;
- fixed episode, API-request, and ContactGraspNet seeds;
- audited perception and chain-health changes relative to upstream `345af89`.

Their control results (`cube_restack=8/10`, `cube_stack=5/10`) are native M3
under the FGSRP clean runtime. They are not an original-CaP-X M3 reproduction
and cannot be directly compared with job `464753`.

## Formal-run status

Job `500239` must be treated as invalid. During cube_restack trial 7, the
verifier returned an `original_claim` that was not an exact report substring.
`_normalize_repairs()` raised `ValueError`, the trial child process failed, and
the runner restarted the episode. The old paired validator did not reject this
event.

Job `500242` has no corresponding child-process failure. Its aggregate is still
diagnostic rather than final because v1's LLM-only semantic gate accepted
repairs that are not logically certified by the selected predicates.

The A0 arms are separable from those obsolete treatments. A later A0-only
audit found zero failed model calls and zero known chain blockers in both A0
arms, then registered their complete native M3 request trajectories, visual
frame fingerprints, initial states, configs, and per-trial outcomes. This does
not rehabilitate either old treatment arm or its verifier claims.

Pair-local M3 smokes initially failed at the second visual report because the
live A0 proxy audit omitted both the raw request and decoded RGB fingerprints.
The exported replay therefore contained `null` expected fingerprints; this was
missing provenance, not evidence of renderer drift. The proxy now records RGB
fingerprints for every live request, the A0 run retains raw request audit, and
the exporter preserves the direct fingerprint field. A missing fingerprint
continues to fail closed.

## Seed audit

`A1 -> FG` is task completion. `Repairs` is the number accepted by FGSRP-v1.
`v1.1 direct` is the number retained by the deterministic direct-contradiction
certificate.

### Cube restack

| Seed | A1 -> FG | Repairs | v1.1 direct | Finding |
| --- | --- | ---: | ---: | --- |
| 1 | 1 -> 1 | 0 | 0 | No-op |
| 2 | 1 -> 1 | 0 | 0 | No-op |
| 3 | 1 -> 1 | 1 | 0 | `next to` and `contact` are compatible |
| 4 | 0 -> 0 | 2 | 1 | One valid current-height correction; one non-contradictory refinement |
| 5 | 1 -> 1 | 0 | 0 | No-op |
| 6 | 1 -> 1 | 0 | 0 | No-op |
| 7 | 1 -> 0 | 1 | 0 | Invalid support/contact edit plus child-process failure; pair invalid |
| 8 | 1 -> 1 | 0 | 0 | No-op |
| 9 | 0 -> 0 | 7 | 7 | Mostly valid grasp/contact corrections; repeated execution still failed |
| 10 | 1 -> 1 | 2 | 0 | `contact` does not negate `stacked`; `separated` does not negate `side by side` |

### Cube stack

| Seed | A1 -> FG | Repairs | v1.1 direct | Finding |
| --- | --- | ---: | ---: | --- |
| 1 | 1 -> 1 | 2 | 0 | Uncertainty/visibility and untyped distance claims |
| 2 | 1 -> 1 | 2 | 1 | Direct table-contact correction retained; visibility/proximity edit rejected |
| 3 | 0 -> 0 | 3 | 2 | Correct grasp/spatial corrections did not overcome execution failure |
| 4 | 0 -> 1 | 1 | 1 | Direct `lifted above table` -> `contacting table` correction caused regeneration |
| 5 | 0 -> 0 | 0 | 0 | Native report contained no directly contradicted registered claim |
| 6 | 1 -> 1 | 1 | 1 | Direct table-contact correction; no harm |
| 7 | 0 -> 0 | 0 | 0 | Native report contained no directly contradicted registered claim |
| 8 | 1 -> 1 | 0 | 0 | No-op |
| 9 | 1 -> 1 | 0 | 0 | No-op |
| 10 | 0 -> 0 | 1 | 0 | Visibility/proximity claim not decidable from current predicates |

## Root causes

1. Baseline naming conflated an original historical run with a patched clean
   control under different temperature and seed semantics.
2. Invalid verifier output was treated as an episode exception instead of a
   local abstention.
3. The semantic gate was another LLM judgment. It could certify a weak relation
   such as contact as if it negated a stronger support relation.
4. Current evidence has no registered predicates for visibility, qualitative
   distance, or free-text transition semantics. Those claims must abstain rather
   than borrow unrelated predicates.
5. Correct factual repair is not sufficient when the remaining failure is code
   protocol or physical execution. Stack seed 3 and restack seed 9 are the main
   examples for subsequent FG-versus-SL attribution.

## v1.1 acceptance rule

FGSRP-v1.1 preserves the v1 carrier and evidence surface. A repair is applied
only if both LLM stages accept it and the host can certify that:

1. every selected non-transition predicate value is explicitly expressed in
   the replacement;
2. at least one selected value directly negates the original clause;
3. the clause does not rely on unregistered uncertainty, visibility, proximity,
   or free-text transition semantics.

Malformed verifier or gate output records an abstention and sends the native
request byte-for-byte. Any treatment child-process failure invalidates the pair.
