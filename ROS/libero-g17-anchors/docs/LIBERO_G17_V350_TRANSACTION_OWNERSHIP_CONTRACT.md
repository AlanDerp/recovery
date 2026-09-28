# LIBERO G17 v350 Transaction Ownership Contract

## Scope

v350 is a bug-fix insertion on the frozen v332 method lineage.  It does not
add a new acquisition policy, grasp-ready gate, candidate ranker, FG
predicate, or recovery controller.  It makes the existing acquisition
transaction the single writer of the physical `AcquireNeeded -> CloseObserved`
transition.

## State contract

```text
AcquireNeeded
  -> proposal bound
  -> canonical pregrasp
  -> canonical ingress endpoint
  -> physical close
  -> fresh FG classification
       FirmAcquired       -> inverse escape -> transport
       RecoverableNearMiss -> inverse escape -> GR/native retry
       DefiniteMiss       -> inverse escape -> native retry
```

`ingress endpoint reached` is an executor fact.  It is not a physical-contact
predicate.  Contact may support the NearMiss classification but is never a
precondition for grasp success.  A close with no pre-close contact observation
may still classify as `FirmAcquired`.

## Single-writer invariant

After `sample_grasp_pose` binds an attempt:

1. Every proposal source, including native GraspNet and frozen replay, has the
   same transaction shape.
2. A pose-only native proposal is completed into one measured pregrasp branch
   and one contact solve seeded exclusively from that branch before motion.
3. Policy-authored xyz/quaternion changes cannot replace the bound continuous
   realization.
4. The first acquisition `goto_pose` executes the canonical transaction once.
5. Additional gotos before close are suppressed; a failed execution is a
   terminal attempt and cannot be replayed.
6. A stale or missing transaction raises a recoverable contract error before
   any physical command and returns to native regeneration.
7. A non-firm close remains `AcquireNeeded`; it never reopens the generic
   Cartesian API between regeneration turns.

The policy remains free before the first proposal and after a firm acquisition
has completed inverse escape.  Ownership is scoped to one transition, not to
the whole robot API.

## Allowed information

The policy selects the task target and requests a skill.  Proposal generation
may use robot perception.  SL may use robot proprioception, FK, IK and its
recorded ingress.  Post-close control receives only the FG predicates defined
by the method.  Continuous simulator object truth is not exposed.

## Acceptance criteria

```text
unowned_acquisition_motion = 0
transaction_bypass = 0
stale_transaction_execution = 0
second_writer_after_ingress = 0
close_without_ingress_endpoint = 0
physical_contact_as_close_precondition = 0
```

Frozen v332 successes must remain executable.  Frozen task-4 failures may
still fail acquisition, but their motion must be attributable to exactly one
proposal, branch, transaction outcome, and recovery transition; elbow-first
motion caused by a policy pose override is not permitted.
