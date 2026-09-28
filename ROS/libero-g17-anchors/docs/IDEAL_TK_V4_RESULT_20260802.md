# Ideal TK V4 Frozen Positive Result (2026-08-02)

## Status

This record freezes the first balanced 10-trial positive result for Ideal TK
V4 on the two closely related M3 cube tasks. The treatment implementation is
commit `a45176fad5c61f362ba4b0032c5023d5b38ac9f8`, tagged
`ideal-tk-v4-canonical-delta-v4.0.1`. The implementation worktree was clean,
and both jobs executed from the same read-only source snapshot.

This is a paper-candidate result, not yet a final paper claim. Cube stack is a
valid 10-trial result. Cube restack contains nine successes and one trial with
an external model-request timeout that the contemporary validator failed to
classify as P0. That trial must be retried under the frozen infra-retry policy
before reporting restack as a fully valid 10-trial estimate.

## Aggregate result

| Task | Frozen A1 | Ideal TK V4 | Net task-completion gain |
| --- | --- | --- | ---: |
| `cube_stack` | `1.000 / 0.533 / 4` | `1.000 / 0.802 / 8` | +4 |
| `cube_restack` | `1.000 / 0.612 / 6` | `0.900 / 0.901 / 9` | +3 raw |

Each tuple is code-generation success rate, average reward, and completed
trials. Across the raw outputs, task completion rises from 10/20 to 17/20.
The restack denominator qualification above remains mandatory.

## Seed-level outcomes

### Cube stack, Slurm job 502911

- Frozen A1 successes: `1, 2, 6, 9`.
- Treatment successes: `1, 3, 4, 6, 7, 8, 9, 10`.
- Recovered seeds: `3, 4, 7, 8, 10`.
- Harm seed: `2`.
- Common failure: `5`.

Stack trial 2 is not supported as a P1 GT error or a P6 non-monotone-policy
counterexample. V4 correctly replaced a false native lift claim with contact
without established transport. The regenerated policy then incurred repeated
continuous grasp/transport failures and exhausted the fixed 1500-step episode
horizon. A1 used 25 `goto_pose` calls across 10 code blocks before succeeding;
V4 used 27 `goto_pose` calls plus three `home_pose` calls across six code
blocks before truncation. The supported attribution is P5 with causal P7
budget coupling.

Stack trial 5 first wasted a regeneration through an invalid two-value unpack
of the three-value `get_object_pose` API. A later attempt transiently reached
centered object support, transitioned to edge support, and then lost support.
Its supported sequence is P4 followed by P5 and P7.

### Cube restack, Slurm job 502915

- Frozen A1 successes: `1, 2, 4, 5, 6, 10`.
- Treatment successes: `1, 2, 3, 4, 5, 6, 8, 9, 10`.
- Recovered seeds: `3, 8, 9`.
- Harm seeds: none.
- P0-censored seed: `7`.

On restack trial 7, native M3 falsely reported task completion after query-only
code and A1 returned `FINISH`. V4 correctly exposed that the green cube was
supported by the red cube, red top access was blocked, and the semantic goal
was unsatisfied. The student regenerated instead of finishing, but initially
ignored the blocked-access precondition and attempted to grasp red. After five
observed transitions, the next student request hung at the local Qwen proxy:
the upstream read timed out after 900 seconds and the trial guard fired after
1800 seconds. The environment was neither terminated nor truncated. This is P0
and cannot be assigned to P1-P7 method behavior.

## Pair controls

Both pair records report:

- `temperature=0`, `num_workers=1`, global seed `20260718`;
- M3 visual-difference carrier and `qwen3.7-plus`;
- frozen A1 reference only, with no new A1 model calls;
- 10/10 replay hits and equal initial response hashes;
- equal initial simulator-state hashes;
- matching infra fingerprints and read-only source snapshots;
- no exact-coordinate, reward, completion, next-move, or oracle exposure from
  the V4 factual report; and
- private task-goal IFF audits passing on every recorded treatment turn.

The pair validator returned `valid` for both jobs. Its current blocker matcher
does not reject a missing model response ending in proxy read timeout. The
restack qualification in this document therefore supersedes the validator's
aggregate `known_chain_blocker_count=0` field for trial 7.

## Frozen artifacts

Cube stack root:

`/home/qshou/capx_paper_verifier_results/results/ideal_tk_v4_canonical_delta_v4_0_1_10trial_20260802/cube_stack/10trial_pair_run-502911`

Cube restack root:

`/home/qshou/capx_paper_verifier_results/results/ideal_tk_v4_canonical_delta_v4_0_1_10trial_20260802/cube_restack/10trial_pair_run-502915`

The machine-readable hashes and outcome sets are preserved in
`manifests/ideal_tk_v4_result_20260802.json`.

## Interpretation boundary

This result supports the empirical claim that a deterministic, sanitized,
GT-backed factual takeover can produce large recovery gains on both stack and
restack while freezing the initial policy. It does not prove strong no-worse:
stack trial 2 remains a budget-coupled harm seed. It also identifies the next
architectural boundary without requiring new task-specific facts: the complete
state basis is present, but operator-precondition consumption and execution
cost are not yet formally controlled.
