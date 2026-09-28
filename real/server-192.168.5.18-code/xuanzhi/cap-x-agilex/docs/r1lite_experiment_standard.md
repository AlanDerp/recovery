# R1 Lite Root-Cause Experiment Standard

This standard is mandatory for every subsequent real-robot modification and
experiment. An experiment that does not satisfy it is exploratory evidence,
not attribution evidence.

## Hard requirements

1. **Unique run ID**
   - Format: `YYYYMMDD_HHMMSS_<short_hypothesis>`.
   - Raw output paths must contain the run ID.
   - Existing raw files must never be overwritten.

2. **Pre-experiment snapshot**
   - Run `scripts/r1lite_experiment_guard.py snapshot`.
   - Preserve exact critical-file contents and SHA-256 hashes.
   - Record ROS nodes, endpoint GIDs/QoS, processes, tmux sessions, physical
     joint feedback and MOCA `p_des`.
   - Reject the experiment if feedback is unavailable or unexpectedly moving.

3. **Single explicit hypothesis**
   - State the suspected boundary and falsifying result.
   - Change one independent variable at a time.
   - Record target semantics: frame, TCP, quaternion order and whether joints
     are command-space or feedback-space.

4. **Predeclared safety envelope**
   - Define maximum total motion, non-target motion, drift, timeout and return
     error before publishing.
   - A threshold violation stops publication immediately.
   - Never “try the next step” after an abort.

5. **Complete command provenance**
   - Record the exact target published by Cap-X.
   - Record MOCA `p_des`, `/motion_control/control_arm_right` and HDAS feedback.
   - Record FunctionFrame commands and all active publishers/subscribers.

6. **Recovery and rollback**
   - Every changed file must exist in the pre-experiment snapshot.
   - File rollback and physical recovery are separate operations.
   - Physical rollback is never automatic; it requires a newly validated,
     safety-bounded motion plan.
   - A failed return is itself an abort, not permission to retry.

7. **Post-experiment finalization**
   - Run `scripts/r1lite_experiment_guard.py finalize`.
   - Preserve before/after hashes and live-state snapshots.
   - State `passed`, `rejected`, `safety_aborted` or `inconclusive`.
   - Do not label an experiment “passed” if only availability was checked.

## Required commands

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)_short_hypothesis"

python scripts/r1lite_experiment_guard.py snapshot \
  --run-id "${RUN_ID}" \
  --intent "One falsifiable sentence"

# Run exactly one bounded experiment and write outputs beneath:
# outputs/r1lite_audit_runs/${RUN_ID}/experiment/

python scripts/r1lite_experiment_guard.py finalize \
  --run-id "${RUN_ID}" \
  --outcome "passed|rejected|safety_aborted|inconclusive: reason"
```

File rollback, after review:

```bash
python scripts/r1lite_experiment_guard.py restore-files \
  --run-id "${RUN_ID}" \
  --confirm "${RUN_ID}"
```

## Interpretation rule

The first layer whose output violates its own input contract receives the
attribution. Downstream deviations are consequences until independently
proven otherwise.

