# A* Instance-Bound Temporal 10-Trial Result (2026-08-05)

## Status

This record freezes the two-task result for the instance-bound grasp-loss
extension of Ideal-TK V4 A*. The implementation commit is
`5aa413ebc005cd7c10532fe921b70829f8afa9c1`; the worktree was clean. Both
reported runs use M3, `qwen3.7-plus`, temperature 0, one worker, global seed
`20260718`, ten trials, and process-isolated execution.

The A* comparison anchor is 18/20: cube stack 8/10 and the response-frozen
cube-restack result 10/10 (job `505776`). The extension changes only the
event-to-public-API binding for the isolated factual contradiction that an
object-specific grasp was not retained during the associated post-grasp
motion. Canonical A* content is unchanged when that contradiction is absent.

## Aggregate Result

| Task | A* anchor | Instance-bound extension | Net |
| --- | ---: | ---: | ---: |
| `cube_stack` | 8/10 | 9/10 | +1 |
| `cube_restack` | 10/10 | 10/10 | 0 |
| Combined | 18/20 | 19/20 | +1 |

The extension therefore has no observed task-level trade-off on this frozen
20-trial cube suite. Cube-stack average reward is `0.901`; cube-restack average
reward is `1.000`. Code-generation success is `1.000` for both tasks.

## Seed-Level Result

Cube stack succeeds on trials `1,2,3,4,6,7,8,9,10`; trial 5 remains the only
failure. Relative to A*, trial 2 is recovered and no A* success is lost. The
new factual mismatch is active on stack trials 2, 5, and 8. Thus the result
does not imply that every correct contradiction guarantees recovery: trial 5
receives the contradiction but still fails.

Cube restack succeeds on all trials. The mismatch is active on trials 1, 6,
and 7, and does not harm any trial.

## Exact-Replay Audit

The prefix/live discovery runs were followed by full treatment-response
replays. These exact runs are the reported reproducibility records:

- Cube stack job `506350`: 96/96 successful model records are exact
  `request_hash_replay` hits; zero live requests, normalized fallbacks,
  required misses, or known chain blockers.
- Cube restack job `506365`: 76/76 successful model records are exact
  `request_hash_replay` hits; zero live requests, normalized fallbacks,
  required misses, or known chain blockers.

Stack replay SHA-256:
`69a2481e8cc443cdc7241b273fc79a00e25084b838eb6ba90e3b05efe7b30213`.

Restack replay SHA-256:
`512306de6f1cfd5d5869f40db78fac374ef52b5bc467303ff216b0f1b6b9ec91`.

Both registered replay packages use normalization
`image_url_redacted_verifier_frame_removed_v2`, contain complete trial/turn
provenance, and bind every M3 request to RGB frame fingerprints.

## Frozen Artifacts

Cube-stack exact run:

`/home/qshou/capx_paper_verifier_results/results/astar_instance_bound_10trial_exact_20260805/capx-qwen37-m3-10trials-t0/run-506350`

Cube-restack exact run:

`/home/qshou/capx_paper_verifier_results/results/astar_instance_bound_restack_10trial_exact_20260805/capx-qwen37-m3-10trials-t0/run-506365`

## Interpretation Boundary

The exact replays establish deterministic reproducibility of the recorded
treatment trajectories. They do not establish that independent upstream model
sampling is deterministic after a verifier-induced prompt divergence. The
supported empirical statement is that the instance-bound extension improves
the frozen A* suite from 18/20 to 19/20 without observed cross-task harm, and
that the complete resulting trajectories reproduce exactly under request-hash
replay.
