# G17 Observation-Integrity Ablation

## Frozen parent

- Parent: G17 semantic-gain gate
- Parent commit: `e1be3a5f7c1c22af71a60a3402f22fe9caefc905`
- Parent result: cube stack 10/10, cube restack 7/10
- Parent restack failures: trials 3, 5, and 8

This ablation changes no task, reward, initial response, replay request, seed,
closed-GT predicate, effect-alignment rule, API-ledger rule, or semantic-gain
rule.

## Hypothesis

Trials 3 and 5 contain physically impossible extents returned by an executed
`get_object_pose` call. A student that treats those values as the current rigid
object geometry can build a false spatial model even when the endpoint
predicate ledger is correct.

The ablation verifies one observation-integrity claim: an extent returned for
a bound rigid-object instance must be consistent with the simulator-backed
rigid geometry of that same instance. The comparison is axis-order independent
and uses the existing broad factor-of-three validity bound.

## Public projection

When every observed extent is consistent, the G17 report and regeneration
prompt are byte-identical to the frozen parent. When an inconsistency is proven,
the verifier appends only:

```text
Verified native-observation integrity:
- At least one extent returned by executed `get_object_pose` for <object> was inconsistent with the simulator-backed rigid geometry of that same object.
```

No extent, coordinate, object ID, correction value, action, reward, success
label, task answer, or strategy is exposed.

## Private audit

Each checked extent is bound to an executed API call index and object instance.
The trace stores only its categorical status and SHA-256 signatures of the
observed and expected axis-order-independent extents. The validator proves that
the public object projection equals exactly the instances with an inconsistent
private audit and that removing the appendix reconstructs the G17 parent.

## Smoke decision

- Rescue targets: cube restack trials 3 and 5.
- Non-target failure control: cube restack trial 8.
- Success guards: cube restack trials 9 and 10; cube stack trials 2 and 3.
- Advance only if every success guard remains successful and at least one target
  is rescued after a real observation-integrity contradiction is projected.
- Reject on any replay miss, validator failure, chain failure, leakage, or
  success-guard regression.

Trial 8 is intentionally not addressed by this ablation: its prior diagnosis
was factual-consumption failure after a valid observation, not invalid native
geometry.
