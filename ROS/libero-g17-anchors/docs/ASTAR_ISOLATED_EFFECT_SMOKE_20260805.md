# A* Isolated Effect Smoke, 2026-08-05

## Frozen setup

- Commit: `bd8f4e46f525c1cca38276f6fb320df061470ef7`.
- Task: `cube_stack`, M3, Qwen `qwen3.7-plus`, temperature zero, one worker.
- Global seed: `20260718`; targeted registered A* trials: 2, 5, and 10.
- Registered prefix replay SHA-256:
  `f37cb374f587828898576bbc06f74ed162c40689005b6d18d8567ffb60930c1b`.
- Each mode is byte-silent unless its one GT-backed effect is contradicted.
- No task YAML, reward, completion test, initial state, action API, physical
  threshold, canonical A* report, or other treatment mode changed.

The two isolated modes are:

- `grasp_loss_only`: loss of bilateral grasp during an executed post-grasp
  motion before release.
- `post_release_support_only`: failure to establish support by the uniquely
  referenced object after an executed release.

Eighty relevant tests passed before submission. The targeted runner rejects a
dirty worktree, a different commit, a different replay hash, unsupported modes,
and trials other than 2, 5, and 10.

## Results

| Mode | Trial 2 | Trial 5 | Trial 10 | Raw total |
| --- | --- | --- | --- | ---: |
| `grasp_loss_only` | S | S | F | 2/3 |
| `post_release_support_only` | S | F | S | 2/3 |

Jobs:

- Grasp loss: `506196`, `506197`, `506198`.
- Post-release support: `506199`, `506200`, `506201`.

All six jobs were chain-valid: the Qwen audit contains no failed model calls,
the source snapshot was clean, and no depth, mask-index, OBB, terminated-
episode, replay-miss, visual-hash, or timeout blocker occurred. User-code
exceptions remain task behavior and are recorded in the trial artifacts.

## Mechanism audit

### Trial 2

Both modes remained byte-silent on both recorded turns. Each run replayed the
first five model calls, then made one live call for the same request SHA-256
`149a24657c0026386c5e0bae3ce0f05716ad58e8ec8c1ed6ad17c0da9f7cf911`.
The two live response hashes differed. Both happened to complete the task.
These successes are evidence of upstream response variance, not either effect
alignment mechanism, and cannot be counted as verifier rescues.

### Trial 5

`grasp_loss_only` first intervened on turn 3 with GT evidence that bilateral
grasp was lost during `goto_pose` before release. It then completed the task.
This is the only positive targeted mechanism result in this smoke.

`post_release_support_only` first intervened on turn 4 and repeated the
GT-backed endpoint fact on turns 5 and 6: after release, the red cube was not
supported by the uniquely referenced green cube and instead had table support.
The student continued to fail and exhausted recovery. Correct support feedback
alone did not rescue this seed.

### Trial 10

`post_release_support_only` intervened on turn 3 with the same support-failure
class and preserved task success. This is a no-harm observation, not a new
recovery, because A* already succeeds on this registered trial.

`grasp_loss_only` first intervened on turn 3 and reported loss during
`home_pose`, then intervened twice more on later `goto_pose` losses. The student
diverged into longer live policies, eventually emitted an invalid two-value
unpack of the three-value `get_object_pose` API, and failed at the horizon.
The first turn-3 event stream contains loss of an earlier grasp followed by a
new acquisition in the same regeneration. The current alignment scans for the
first object loss over the whole interval; it does not bind a loss to the
specific acquisition-to-release segment that generated the expectation. This
is an event-instance binding defect, so `grasp_loss_only` is not ready for a
10-trial claim.

## Decision

Neither mode advances directly to 10 trials from this smoke.

- Grasp loss has one genuine rescue and one genuine harm. Its next admissible
  change is acquisition-instance segmentation: evaluate retention only after
  the bound acquisition event and before its bound release event.
- Post-release support is clean and did not harm the registered success, but it
  produced no demonstrated rescue. A broader 10-trial run is low priority
  unless another fixed failure seed shows a support-specific positive result.
- Trial 2 remains outside both mechanisms in this experiment. Its successful
  live continuations must not be used as evidence for either mode.
