# LIBERO G17 PT Collision Asset Ledger v1

## Purpose

This ledger separates reusable scientific assets from historical collision
policies that must not be inherited by clean adaptive acquisition.  A version
number is not an asset.  Each retained item below has one role, one information
boundary, and one proof obligation.

The central distinction is:

1. **Kinematic transaction integrity** answers whether one proposed grasp has
   a continuous, bound robot realization.
2. **PT collision prediction** estimates risk from RGB-D, robot state, and the
   robot model before execution.
3. **Executed-contact feedback** reports what happened during the commanded
   motion and supports recovery.

Only (1) is a hard command-admission contract.  Current PT collision prediction
is useful evidence, but is not an exact collision-free certificate and must not
be a global hard veto.

## Information boundary

Allowed inputs are observation RGB-D, calibrated camera geometry, measured
robot joints, FK/IK, the robot URDF, the selected target identity, recorded
commands, and post-action FG predicates.  Simulator body poses, hidden contact
truth, and continuous privileged object state are not method inputs.

Execution-time joint/contact reports may be used as robot interaction feedback.
They must be logged separately from pre-action PT prediction, and distilled to
the task-relevant discrete outcome before an outer FG transition consumes them.
Any contact label available only from the simulator is evaluation/audit truth,
not an SL input; promotion requires the same outcome to be derived from
proprioception or PT.

## Atomic asset decisions

| Asset | Historical source | Decision | Permitted role | Forbidden interpretation |
|---|---|---|---|---|
| RGB-D + measured-joint + URDF scene reconstruction | `f9a21d2` | Keep, shadow/soft | Produce per-link distance and path-risk evidence | Exact world geometry or collision truth |
| Swept-path audit with first sample, closest link, object, and timing | `f9a21d2` | Keep, shadow/soft | Audit and rank otherwise executable candidates | A global command veto |
| Multi-contact GraspNet candidate preservation | `a150c1b`, `75c9c5e` | Keep | Preserve candidate diversity instead of collapsing early to one pose | Proof that the highest-scoring pose is executable |
| Multi-seed and redundant IK branch enumeration | `1d4f2f2`, `5a1b9ed` | Keep | Search for a connected realization of each candidate | Task-specific hard-coded arm pose |
| Exact pregrasp/contact branch binding | `2f3cf60` | Keep, hard | Solve contact from the selected pregrasp branch and execute the bound joint states without re-solving | Collision-free proof |
| Transaction ownership and one physical writer | `0d6c30e`, `a150c1b`, v350 ownership contract | Keep, hard | Prevent policy/API/native executors from competing inside one acquisition attempt | A grasp-success guarantee |
| Reached-joint/FK endpoint verification | v350 transaction line | Keep, hard | Distinguish commanded, partial, and reached motion | Object contact or firm acquisition |
| Exact recorded ingress and inverse retreat | `ce24931`, GR transaction line | Keep | Recover the robot branch after an empty/blocked attempt when the inverse is executable | Reversibility of the object/world state |
| Executed non-target block/contact outcome | GR and retry audit line | Keep with source boundary | Typed recovery evidence when robot-observable/PT; simulator-only labels remain offline audit | Permission to expose continuous simulator truth |
| Contact-aware path phases | `75c9c5e` | Keep, shadow/soft | Distinguish free-space pregrasp risk from expected terminal target contact | Treat every predicted target contact as an obstacle |
| PT-clear top-down selection | `ac498d0` | Keep only as proposal/ranking feature | Prefer lower-risk top-down candidates when available | Universal top-down grasp rule or mandatory gate |
| Rejected-alternative audit | `394deb5` | Keep | Calibrate false positives/negatives and explain selection | Runtime physical behavior |
| Proximal-contact heuristic | `c3be5ac` | Keep only as an experimental feature | Candidate proposal or shadow annotation | Core FG predicate or universal grasp condition |
| Hard `predicted_arm_obstacle_path_clear` admission | v296/v350 experiments | Discard from main method | Historical ablation only | Trusted PT collision certificate |
| Coupling “no PT-clear candidate” to `NoCandidate` | v350 completion variants | Discard | Historical ablation only | Absence of an executable kinematic transaction |
| Transparent native executor exemption | `5638c40` | Discard | Historical ablation only | Valid transaction completion |
| Accepting `result.ok == false` as ingress and then closing | `5638c40` line | Discard | None | Recoverable success path |
| Fail-open execution of the original argmax after audit failure | soft/fail-open variants | Discard | None | Candidate diversity or graceful degradation |
| Task-specific clearance distances/heights | POC variants | Discard from generalized SL | Frozen POC comparison only | General collision or escape model |

## Why PT collision is not a hard veto

The task-4 evidence contains both sides of the problem.

- In job `535617`, several catastrophic native attempts were predicted as
  arm-path collisions and later produced cabinet/self contacts.  The predictor
  therefore contains real information and should not be deleted.
- In the v332 task-4 audit, predicted obstruction also appears on trajectories
  that reach NearMiss or succeed, partly because intended target contact,
  target/scene segmentation, point inflation, and interpolation margins are
  imperfect.  A hard veto therefore creates false-negative `NoCandidate`
  outcomes and removes native capability.
- Conversely, some predicted-clear paths still produce actual contact.  PT
  clearance is therefore not sufficient for safety or task success either.

The correct claim is **risk ordering**, not **collision certification**.
Retaining the structured evidence is important: link, object class, phase,
first sample, and margin are more informative than one clear/not-clear bit.

## Clean adaptive-acquisition interface

```text
FG target binding
  -> diverse grasp proposals
  -> kinematic transaction compiler
       candidate pose
       + connected pregrasp branch
       + contact solve seeded from that branch
       + bound ingress/inverse
  -> PT risk ordering among compiled transactions
  -> single owner executes exactly one transaction
  -> close
  -> fresh endpoint classification
       FirmAcquired -> inverse escape -> transport -> RR
       NearMiss     -> inverse retreat -> adaptive GR
       DefiniteMiss -> inverse retreat if executable -> FG-informed retry/regen
       Blocked      -> measured retreat -> FG-informed retry/regen
```

Hard admission may reject a *particular realization* when its kinematic
transaction cannot be compiled.  It must continue enumerating other contacts,
orientations, and branches.  Exhausting the finite candidate set causes a typed,
non-physical regeneration transition; it must not silently execute a rejected
native pose and must not terminate the full outer loop.

PT risk may reorder compiled transactions.  A risky candidate remains available
when no lower-risk executable alternative exists, so prediction uncertainty does
not become a false-negative dead end.  After execution, actual block/contact and
the endpoint FG state—not the prior risk score—determine recovery.

## Non-negotiable invariants

1. One acquisition attempt has one target identity, one proposal id, one bound
   pregrasp/contact branch, one motion owner, and one close.
2. No xyz/quaternion or IK branch is re-solved between transaction compilation
   and execution.
3. No close follows an unexecuted, partial, stale, or failed ingress.
4. PT collision prediction never changes endpoint FG truth.
5. `contact` is not a precondition for success; a no-contact approach may still
   close into FirmAcquired.
6. Intended target contact is distinguished from non-target obstruction and
   self/cabinet contact.
7. Recovery consumes the measured executed prefix.  It does not pretend the
   full requested path happened.
8. A failed candidate is not a failed episode: retry/regeneration remains the
   full-domain fallback.

## Required ablation before promotion

Use identical frozen seeds, proposal sets, model snapshots, and infrastructure:

1. **Kinematic-only:** connected transaction is hard; PT collision is audit-only.
2. **PT soft rank:** same compiled set, PT risk only changes ordering.
3. **Historical hard gate:** same set, PT-clear is mandatory; retain only as the
   negative ablation.

Report by stage rather than reward alone: candidate coverage, connected-branch
coverage, executed ingress, partial/blocked motion, Firm/NearMiss/DefiniteMiss,
identity error, quaternion/branch discontinuity, inverse-retreat completion,
and final task success.  PT soft ranking is promotable only if it reduces
catastrophic non-target contacts without decreasing Firm-or-NearMiss coverage
or introducing new `NoCandidate` terminal failures.

## Mainline decision

The clean-C mainline keeps the perception and collision-audit machinery, but
demotes it from hard truth to calibrated PT evidence.  It keeps kinematic branch
continuity, ownership, exact execution accounting, endpoint FG classification,
and measured recovery as hard structure.  It removes both historical extremes:
hard PT fail-closed and unaudited native fail-open.

This preserves the genuine scientific asset—the ability to predict and explain
collision risk—without allowing an imperfect PT estimate to erase executable
native acquisition capability.
