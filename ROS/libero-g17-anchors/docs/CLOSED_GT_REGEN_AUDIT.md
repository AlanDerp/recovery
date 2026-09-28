# Closed GT Regeneration Coverage Audit

## Audited Carrier

The audit uses the frozen M3 A0 replay artifacts that will serve as the
comparison carrier:

- `experiments/fgsrp/replays/cube_stack_qwen37_m3_a0_full_seed20260718_10trial_job500242_v1.json`
- `experiments/fgsrp/replays/cube_restack_qwen37_m3_a0_full_seed20260718_10trial_job500239_v1.json`

A native regeneration report is an audit entry with exactly two image
fingerprints and a non-code response. This identifies 49 reports: 28 stack and
21 restack. All 49 are unique.

The counts below are lexical discovery statistics, not semantic correctness
labels:

| Report concept | Reports mentioning it | Closed-GT coverage |
| --- | ---: | --- |
| relative position or topology | 49 | directed XYZ, footprint, region, volume, contact, support |
| table or physical support | 42 | table contact, lift band, semantic support and stability |
| gripper configuration | 48 | aperture, orientation, dynamics, object-relative XYZ |
| holding or contact | 23 | executor-object contact and mutually exclusive grasp-contact state |
| motion or change | 47 | endpoint deltas, entity pose motion, and ordered temporal events |
| object orientation | 1 | object orientation to table |
| camera visibility | 19 | deliberately excluded |
| task completion or goal judgment | 49 | deliberately excluded from action-visible GT |
| action advice | 6 | deliberately excluded |

## Coverage Contract

Closed GT covers the **physical state and endpoint-transition proposition
roles** of the native report. It does not cover every sentence in the native
report. In particular, the native reporter mixes physical observations with
camera epistemics, task-completion judgments, and action advice.

Therefore a no-cheating takeover cannot honestly claim to reproduce the whole
native report. Its replaceable target must be defined as:

```text
PHYSICAL_REGEN_STATE := current closed-GT predicates
PHYSICAL_REGEN_TRANSITION := previous -> current closed-GT predicate changes
PHYSICAL_REGEN_INTERVAL := mutually exclusive per-pair grasp outcome over the interval
PHYSICAL_REGEN_EVENTS := ordered changes in the declared temporal projection
```

The takeover renderer may replace the native report carrier with those factual
sections. It must not generate or infer:

- task success, reward, or a completion verdict;
- next move, plan, strategy, diagnosis, or causal explanation;
- coordinates, dimensions, thresholds, or action parameters;
- visibility claims or continuous-time facts between simulator steps.

Dropping excluded native material is a treatment choice, not a seamless
byte-preserving edit. It must be declared in the experiment and separately
ablated from local NL replacement.

## IFF Boundary

The closed state has 51 validated fields. Its 34 primitive fields form the
nonredundant canonical basis; the remaining 17 are deterministic report-level
summaries. This basis is complete for its declared semantic quotient and for
the physical proposition roles above. Benchmark success is separately
expressible as an exact Boolean formula over closed facts and privately checked
against the unmodified environment label.

This does not establish a universal manipulation ontology. A task that depends
on insertion depth, keyed rotational alignment, articulation, deformable
state, fluid state, or camera epistemics requires new audited primitives before
the same completeness or IFF claim is made.
