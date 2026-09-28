# A* Instance-Bound Temporal Smoke (2026-08-05)

## Scope

This experiment changes only the isolated grasp-loss selector. Canonical A*,
the 44-field closed-state report, task files, model configuration, and initial
policy responses remain unchanged. A grasp loss is visible only when its GT
event is bound to the last executed `close_gripper` to `open_gripper` public-API
call window.

- Implementation commit: `dd2001f2834a7c13be2ba22b55be789087967f40`
- Exact-replay runner commit: `0e27f30fbeb9bd7f3a2f809bc4f7ef4eb48d4c70`
- Mode: `instance_bound_grasp_loss_only`
- Task: `cube_stack`
- Registered A* replay SHA-256:
  `f37cb374f587828898576bbc06f74ed162c40689005b6d18d8567ffb60930c1b`

## Results

| Trial | Job | Response control | Result | Mechanism |
| --- | --- | --- | --- | --- |
| 5 | 506211 | registered prefix, then live | failure, reward 0.012 | Valid loss was reported at turn 3; the first live response diverged from the prior successful run despite an identical request and seed. |
| 10 | 506212 | registered prefix, then live | success, reward 1.000 | Historical loss events were bound to `home_pose` call 0 and excluded from the current close-call-4 to open-call-9 window. |
| 5 | 506215 | complete exact response replay | success, reward 1.000 | The valid turn-3 loss remained bound to call 3 inside the close-call-1 to open-call-6 window. All 10 model responses were replay hits. |
| 10 | 506218 | complete exact response replay | success, reward 1.000 | The historical loss remained excluded. All eight model responses were replay hits. |

The exact trial-5 replay was exported from the successful isolated
grasp-loss run 506197:

- replay file:
  `/home/qshou/capx_paper_verifier_results/manifests/astar_instance_bound_temporal_smoke_20260805/stack_trial5_grasp_loss_success_full_replay.json`
- replay SHA-256:
  `5c652ab058424e60ea74be26bff40939e933c0f5c763bae533cb8fdd598b4727`
- audit: 10 records, 10 entries, zero missing responses, zero duplicate
  conflicts; job 506215 consumed all 10 through `request_hash_replay`.

The exact trial-10 replay is
`/home/qshou/capx_paper_verifier_results/manifests/astar_instance_bound_temporal_smoke_20260805/stack_trial10_instance_bound_success_full_replay.json`
with SHA-256
`6ab0f6944f839520f42b833535e93d7064bcd8b6c955d493420534a133b0fc00`.
It contains eight entries with zero missing responses or conflicts; job 506218
consumed all eight through `request_hash_replay`.

## Live Divergence Audit

Jobs 506197 and 506211 had identical request and response hashes for their
first seven calls. Their first live call also had the same request SHA-256
(`44835aa714f9a5f6061af466959abc95fc1be92194d6109da7e318933ea072c6`),
effective request seed (`20265725`), model, and temperature. The remote API
returned different response hashes:

- successful 506197:
  `1541fd50f11c99f2b104a33499ecaeb68278877e9accc84fa3f3db6124215251`
- failed 506211:
  `71a8ea9576eed82c56b35af292c8dc549e408cdb35649c5889c462444b98d50a`

Therefore the live trial-5 failure is not attributable to a changed verifier
prompt or temporal binding. A fixed request seed does not freeze the remote
model response; exact response replay is required for this local ablation.

## Interpretation

This smoke validates the local mechanism:

1. A true loss in the active grasp instance remains visible and can preserve
   the prior successful recovery under identical student outputs.
2. A loss from an earlier grasp instance is no longer misattributed to a later
   plan, removing the observed trial-10 harm.
3. Ambiguous or unindexed event provenance remains byte-silent.

It does not establish aggregate no-worse behavior or a cross-task gain. Those
claims require a frozen response protocol and a larger paired evaluation.
Neither successful run contained the declared depth, mask-index, Open3D OBB,
terminated-episode, or replay-miss chain blockers. The student-generated
`get_object_pose` unpacking error remained a policy/API-usage error and was
recovered within the episode.

## Trial-2 Silent-Replay Blocker

Job 506311 attempted a complete exact replay of the prior byte-silent trial-2
trajectory. It is invalid and is not a task failure: five requests replayed,
then the sixth was rejected with HTTP 409 by the required-replay gate before a
student response was returned.

The old and current takeover reports are byte-identical on both turns. A full
canonical-request diff contains exactly one changed line: the internal
`ideal_tk_v4.py` wrapper traceback line number moved from 1021 to 1211 when the
source file grew. This non-semantic instrumentation provenance changed the raw
and normalized request hashes. Internal verifier traceback frames therefore
need canonicalization or removal before a 10-trial replay can be considered
strictly controlled. The earlier live trial-2 successes remain upstream
response-variance observations, not verifier rescues.
