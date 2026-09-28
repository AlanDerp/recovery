# PGSVP-v9 AS Materialized PT-GT Paired Experiment

The treatment uses the historical v9 AxisStage factual kernel with a
materialized verification interface. The same action model externalizes PT from
its real regeneration context. The host applies verifier corrections, proves
`PT + delta = verified_state`, and exposes only the resulting canonical state
to the action call. Raw PT and delta remain private audit artifacts. See
`docs/PGSVP_V9_VERIFY_SPEC.md`.

Every paired run archives the clean submission commit into the result
directory. Both A0 and treatment execute from that read-only source tree.
The runner fails closed unless the critical `capx` modules resolve inside the
archive and the source checksums remain unchanged through job exit. Editing the
submission worktree after `sbatch` therefore cannot alter an active run.
Relative runtime artifacts such as perception debug images are written under
the run-specific writable `workdir`, never into the source archive. A read-only
`workdir/capx` link preserves legacy repository-relative resource lookups.

Run a one-trial paired smoke before a formal pair:

```bash
sbatch --time=01:00:00 \
  --export=ALL,PROJECT_DIR=/home/qshou/capx_pgsvp_v9_frozen,CAPX_PAIR_SMOKE=1,TASK_NAME=cube_restack,TOTAL_TRIALS=1,CAPX_TREATMENT=pgsvp_v9_verify \
  experiments/pgsvp_v9_verify/run_paired_task.sbatch
```

The smoke is valid only when `PAIR_VALIDATION.json` reports `status: valid`,
both `RUNTIME_IMPORT_AUDIT.json` files report `status: valid`, and
`PAIR_SMOKE_DONE` exists.

Run cube restack against the registered A1 reference without re-executing A1:

```bash
sbatch --time=03:00:00 \
  --export=ALL,PROJECT_DIR=/home/qshou/capx_pgsvp_v9_frozen,CAPX_TREATMENT=pgsvp_v9_verify,CAPX_SKIP_A1=1,TASK_NAME=cube_restack,TOTAL_TRIALS=10,CAPX_FROZEN_A0_RUN_ROOT=/home/qshou/capx_paper_verifier_results/results/pgsvp_v9_verify_pi0_replayfix_10trial_20260729/cube_restack/10trial_pair_run-496713/a1/capx-qwen37-m2-10trials-t0/run-496713,CAPX_FROZEN_A0_PAIR_VALIDATION=/home/qshou/capx_paper_verifier_results/results/pgsvp_v9_verify_pi0_replayfix_10trial_20260729/cube_restack/10trial_pair_run-496713/PAIR_VALIDATION.json \
  experiments/pgsvp_v9_verify/run_paired_task.sbatch
```

For cube stack, use the corresponding source artifacts:

```text
CAPX_FROZEN_A0_RUN_ROOT=/home/qshou/capx_paper_verifier_results/results/pgsvp_v9_verify_stack_pi0_10trial_20260729/cube_stack/10trial_pair_run-496763/a1/capx-qwen37-m2-10trials-t0/run-496763
CAPX_FROZEN_A0_PAIR_VALIDATION=/home/qshou/capx_paper_verifier_results/results/pgsvp_v9_verify_stack_pi0_10trial_20260729/cube_stack/10trial_pair_run-496763/PAIR_VALIDATION.json
```

For the two registered cube tasks, the runner hash-validates the historical A1
run and validation record, runs no current A1 model calls, runs the treatment
with only its initial response forced to match A1, and validates the pair.
Results are not valid unless
`PAIR_VALIDATION.json` reports `status: valid` and `PAIR_DONE` exists.

The paired runner checks `infra_lock.json` before A0 starts and writes the
matched `INFRA_FINGERPRINT.json` into the pair root. The lock covers the
materialized verifier implementation, frozen AxisStage prompt and schema, and
the action-visible state contract in addition to the shared runtime.

## Frozen A0

The cube-restack A0 reference is the valid historical job `496713` at commit
`b9fb2fea09cb455149c84e5da91275144ab67156`, with result
`1.000/0.606/6`. Its 41-call full replay and reference record are:

```text
experiments/pgsvp_v9_verify/replays/cube_restack_qwen37_m2_a0_full_seed20260718_10trial_v1.json
experiments/pgsvp_v9_verify/replays/cube_restack_a0_reference_seed20260718_10trial_v1.json
```

The cube-stack A0 reference is the valid historical job `496763` at commit
`d07c3cd7ebdb10bc5c39009cce69dd9b168c311e`, with result
`1.000/0.612/6`. Its 48-call full replay and reference record are:

```text
experiments/pgsvp_v9_verify/replays/cube_stack_qwen37_m2_a0_full_seed20260718_10trial_v1.json
experiments/pgsvp_v9_verify/replays/cube_stack_a0_reference_seed20260718_10trial_v1.json
```

Reference-only validation requires the registered source validation hash, all
ten initial-state hashes, historical per-trial results, and aggregate result to
match. It records zero A1 model calls in the current run. The treatment receives
the same frozen initial M2 responses, while regeneration, PT, and verifier
requests remain live experimental outputs.

Replay banks are task-specific. Do not use either cube bank for a different
task.
