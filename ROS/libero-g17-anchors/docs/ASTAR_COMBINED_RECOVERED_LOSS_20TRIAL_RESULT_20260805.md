# A* Combined Factual-Event 20-Trial Result (2026-08-05)

## Status

This record freezes the first exact-replay 20/20 result for the combined
Ideal-TK V4 factual-event verifier on the two M3 cube tasks. The implementation
commit is `a99256d4e8c072c6bdcd92db70047fa962b7d81c`; the worktree was clean.
Both runs use `qwen3.7-plus`, temperature 0, one worker, global seed
`20260718`, ten trials per task, process-isolated execution, and frozen request
hash replay.

The combined verifier is task-agnostic. It selects among three GT-observed
event classes: grasp loss before release, loss of a previously lift-confirmed
state at an API endpoint, and in-API grasp loss followed by endpoint recovery.
It contains no task-name, trial, seed, object-name, action-strategy, coordinate,
reward, completion-label, or oracle-code branch.

## Aggregate Result

| Task | Frozen A1 | Previous 19/20 anchor | Combined verifier | Net vs A1 |
| --- | ---: | ---: | ---: | ---: |
| `cube_stack` | 4/10 | 9/10 | 10/10 | +6 |
| `cube_restack` | 6/10 | 10/10 | 10/10 | +4 |
| Combined | 10/20 | 19/20 | 20/20 | +10 |

Both treatment runs have code-generation success `1.000`, average reward
`1.000`, and ten completed tasks. Cube stack averages 4.3 code blocks and 3.3
regenerations. Cube restack averages 3.8 code blocks and 2.8 regenerations.

The additive endpoint-regression fact recovers the previous stack failure at
trial 5. The event binder preserves restack's grasp-loss signal when bilateral
grasp is acquired and lost within one public API, including loss followed by
endpoint recovery. Endpoint recovery remains represented in the full ordered
temporal report and in a dedicated audit field; the concise contradiction only
highlights the verified in-motion loss event.

## Exact-Replay Audit

- Cube stack job `506742`: 86/86 successful model records are exact
  `request_hash_replay` hits.
- Cube restack job `506744`: 76/76 successful model records are exact
  `request_hash_replay` hits.
- Both runs have zero live calls, normalized fallbacks, deferred visual
  certificates, required misses, prefix blocks, and failed audit records.
- Both runs contain ten complete Ideal-TK V4 GT trace roots.
- No depth assertion, mask-depth mismatch, Open3D OBB crash, native terminated-
  episode traceback, replay miss, or child-process chain failure was found.

Stack replay package:

`/home/qshou/capx_paper_verifier_results/manifests/astar_transport_only_20260805/stack_anchor_plus_true_lift_regression_job506528_registered.json`

SHA-256:
`c8b40ed5d01da159c81f94bf847825d8ae5673d6b04571e2698fc594e6346285`.

Restack replay package:

`/home/qshou/capx_paper_verifier_results/manifests/astar_instance_bound_temporal_smoke_20260805/restack_instance_bound_job506356_full_replay_v2.json`

SHA-256:
`512306de6f1cfd5d5869f40db78fac374ef52b5bc467303ff216b0f1b6b9ec91`.

## Frozen Artifacts

Cube-stack run:

`/home/qshou/capx_paper_verifier_results/results/astar_combined_recovered_loss_10trial_20260805/capx-qwen37-m3-10trials-t0/run-506742`

Summary SHA-256:
`e700cbffa06bb90ef8acbb529ce3e82e8239bc12d5c786a5dbee3b2b1440a012`.

Cube-restack run:

`/home/qshou/capx_paper_verifier_results/results/astar_combined_recovered_loss_10trial_20260805/capx-qwen37-m3-10trials-t0/run-506744`

Summary SHA-256:
`7e9c2afb12eebc14b62aa794f472c7e2979508507841d5361159206583242ae6`.

## Interpretation Boundary

This result establishes exact reproducibility of the recorded treatment
trajectories and empirical no-harm on the fixed two-task, twenty-trial suite.
It does not by itself establish that independent upstream model sampling is
deterministic, that the method is universally no-worse, or that it generalizes
to unseen tasks. Those claims require fresh response-frozen paired evaluations
on additional task families. The supported claim here is that the three
task-agnostic factual-event components compose without observed trade-off on
this frozen suite and improve its recorded completion from 10/20 to 20/20.
