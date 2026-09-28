# Closed Semantic GT v2

## Scope

This module defines the private factual basis used by whole-report takeover.
Continuous
simulator state is private; the action model may receive only finite semantic
predicates rendered without coordinates, dimensions, thresholds, reward,
completion labels, code, or action parameters.

There are two separate IFF claims; they must not be conflated:

1. **State IFF:** completeness is relative to the declared semantic quotient,
   not to continuous simulator state. Two physical states are semantically
   equivalent iff every declared current-state predicate has the same value.
   Two endpoint transitions are equivalent iff every predicate change and
   every declared meaningful pose-change predicate has the same value.
2. **Benchmark-goal IFF:** task success is re-expressed using the exact facts
   consumed by the unmodified benchmark validator. A private live audit must
   establish `goal_atoms <=> task_completed`; the completion label is never
   action-visible.

## Entity Algebra

The schema is generated from ordered executor and object registries:

```text
for executor in executors:
    executor state
    for object in objects:
        executor-object relations

for subject in objects:
    for reference in objects where subject != reference:
        directed object-object relations

for unordered object pair:
    symmetric contact
```

IDs are sanitized semantic IDs, not simulator IDs. Every directed relation key
states subject before reference. Symmetric contact occurs exactly once.
The cube adapter additionally validates that semantic-frame `+Z` is the table
normal; height and orientation enums are invalid rather than silently emitted
when that frame contract does not hold.

## Predicate Layers

Primitive facts are read from continuous state and contact/grasp queries:

- executor aperture, dynamics, and local-Z orientation relative to the table;
- executor-object contact, holding, and relative X/Y/Z bands;
- directed object-object relative X/Y/Z bands;
- directed center/edge/outside support region and volume inclusion;
- symmetric object contact;
- object-table contact, qualitative lift band, and qualitative height in the
  declared workspace frame;
- object dynamics and local-Z orientation relative to the table.

Derived factual predicates are deterministic functions of primitive facts:

- directed support;
- footprint inclusion (derived exactly from support region);
- executor payload;
- grasp evidence;
- one mutually exclusive executor-object grasp-contact state;
- mutually exclusive object semantic state;
- top access;
- whether an object currently supports no, one, or multiple other objects;
- support stability.

Derived facts are retained because they match natural recovery-report concepts,
but each declares dependencies and is recomputed using only fields inside the
closed basis during validation. They are not independent sources of truth and
cannot use hidden continuous values.

For the one-executor, two-object cube registry, v2 has 51 current-state fields:
34 primitive and 17 derived.
Every state field receives an `UNCHANGED/CHANGED` transition field. Executor
and object pose-change fields preserve meaningful motion that stays inside one
qualitative state cell. Holding outcomes and one aggregate change field are
also deterministically derived, giving 57 transition fields.

The default public projection retains the complete 51-field closure. Closure
and contradiction freedom take priority over brevity. A separate optional
34-field canonical projection removes deterministic derived summaries without
information loss; it is an output optimization, not the GT definition. A
future natural-language renderer may keep both a primitive and its derived
summary when that makes the relation less ambiguous, and must record such
duplication rather than silently discard a required fact.

The corresponding optional canonical transition projection contains 34
primitive predicate deltas plus three entity pose-motion facts. The default
transition projection retains all 57 validated fields.

### Grasp-contact partition

For every executor-object pair, exactly one of the following values is true:

- no executor contact;
- executor contact without the simulator's bilateral grasp-contact test;
- bilateral grasp contact while the object is still table-supported;
- bilateral grasp contact while the object is still supported by another
  object;
- bilateral grasp contact without another support but below the lift band; or
- bilateral grasp contact with lift confirmed.

This partition does not equate an instantaneous grasp-contact test with a
durable transport guarantee. In particular, table-supported bilateral contact
is explicitly labeled as transport not established. The older holding,
payload, grasp-evidence, and object-semantic fields remain in the complete
machine-audited closure, but the takeover renderer suppresses those redundant
aliases and exposes this partition as the single canonical grasp relation.

## Conflict And Duplication Rules

Validation fails closed when:

- a field is missing, extra, duplicated, reordered, or outside its finite enum;
- a dependency references an undeclared field;
- a primitive declares dependencies, a derived field lacks dependencies, or
  the dependency graph is cyclic;
- holding is true without executor-object contact;
- both directed support relations are true for one object pair;
- footprint inclusion conflicts with support region or projected X/Y bands;
- support differs from the conjunction of contact, footprint inclusion, and
  positive vertical relation;
- payload conflicts with holding;
- grasp evidence, grasp-contact state, semantic state, top access,
  load-bearing state, or support stability conflicts
  with its primitive basis;
- a transition differs from its before/after snapshots;
- aggregate change conflicts with atomic changes.

GT extraction has no `UNKNOWN` output. Invalid or unavailable raw state is a
chain error, not an `UNKNOWN` fact.

## Goal IFF

Cube task goals are declared as required atoms plus optional forbidden
conjunctions over the factual basis. They
mirror the unmodified Robosuite Stack validator rather than imposing a stricter
human interpretation of stacking:

```text
cube_stack :=
    not HOLDING(red)
    and ABOVE_LIFT_BAND(red)
    and CONTACT(red, green)

cube_restack :=
    cube_stack
    and not (
        ABOVE_WORKSPACE_GUARD_BAND(red)
        and ABOVE_WORKSPACE_GUARD_BAND(green)
    )
```

The separate directional `SUPPORTED_BY` predicate remains available as a
stronger physical fact, but is deliberately not substituted for the benchmark
condition. The workspace-guard clause preserves the restack wrapper's existing
robot-base-frame check exactly; it must not be mislabeled as table-relative
height. No reward or completion label is read when computing semantic goal
truth. A private audit compares that truth with the unmodified environment
label:

```text
semantic_goal_satisfied == private_task_completed
```

Any mismatch invalidates the run and is never shown to the action model. This
is the operational IFF test. It must pass on live stack and restack smoke states
before the basis is frozen.

## Regeneration Coverage

The basis covers the declared physical portions of native M3 recovery reports:

- executor aperture, orientation, location relative to each object, contact,
  motion, holding, and payload;
- object relative position, footprint, volume, contact, support, table
  relation, orientation, motion, dynamics, accessibility, and support
  stability;
- complete endpoint predicate changes and holding outcomes.

`semantic_coverage_manifest()` treats these as seven required factual families
and fails its coverage audit if any required family is absent. This is schema
coverage, not a claim that arbitrary natural-language claim segmentation is
solved. The future takeover renderer must emit only these factual families and
must not silently absorb excluded report content.

The following are deliberately outside the factual state basis:

- action advice and next move;
- reward and task-completion labels;
- oracle code and continuous action parameters;
- camera visibility or corruption;
- causal failure attribution;
- continuous coordinates and events outside the declared temporal predicate
  projection.

Those exclusions must not be silently described as covered. v2 obtains
trajectory evidence by sampling the declared temporal predicate projection
after every completed simulator step. It therefore observes an acquire-then-
drop sequence when at least one sampled semantic state separates those events;
it does not claim continuous-time completeness between simulator steps.

Workspace reachability is intentionally not yet a GT predicate. The current
controller exposes no audited Cartesian reach envelope, and a guessed radius
or an IK failure would conflate physical state with solver behavior. Reach can
be added only through a declared controller domain or a separately validated
feasibility checker. `top_access` and load-bearing topology are retained
because their truth is directly and deterministically defined by the existing
geometry and contact predicates.

## Generality Boundary

The executor/object expansion, directed topology, symmetric contact, support,
motion, and transition algebra are task-independent. A new domain still needs
an audited adapter that binds its semantic entities and physical queries. Tasks
whose success depends on relations absent from this quotient, such as keyed
orientation, articulated joints, liquid state, insertion depth, or deformable
contact, must extend the primitive vocabulary before an IFF claim is allowed.
They must not map those states into an existing enum by approximation.
