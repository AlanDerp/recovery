# Core-FG Paired Experiment

This experiment compares frozen CaP-X M2 A1 with A1 plus deterministic
Core-FG on `cube_stack` and `cube_restack`.

Each Slurm job runs one task pair in a single allocation:

1. run A1;
2. export exact initial responses by canonical request hash;
3. run Core-FG with mandatory initial-response replay.

The treatment uses the same commit, task YAML, seed schedule, model,
temperature, worker count, node, and allocation as its paired A1 run.
Subsequent model calls remain live because their prompts legitimately differ.
Molmo is disabled because the cube task's `FrankaControlApi` has no Molmo
consumer; SAM3, ContactGraspNet, and Pyroki remain enabled.

The Qwen key and workspace endpoint are explicitly paired with the same
Singapore workspace used by the historical clean cube experiments. The key
itself remains outside git and is referenced only by filesystem path.

SAM3 and ContactGraspNet are external source dependencies because upstream
CaP-X excludes `capx/third_party` from the main repository. The paired runner
pins and records their independent git revisions, requires both worktrees to
be clean, validates the model/config SHA-256 values, and fails before
evaluation if any dependency differs.
SAM3, ContactGraspNet, and Pyroki use per-job ports; server startup is
fail-closed.

Smoke:

```bash
SERVICE_SMOKE_ONLY=1 TASK_NAME=cube_stack TOTAL_TRIALS=0 \
  sbatch experiments/core_fg/run_paired_task.sbatch
TASK_NAME=cube_stack TOTAL_TRIALS=1 sbatch experiments/core_fg/run_paired_task.sbatch
TASK_NAME=cube_restack TOTAL_TRIALS=1 sbatch experiments/core_fg/run_paired_task.sbatch
```

Formal:

```bash
TASK_NAME=cube_stack TOTAL_TRIALS=10 sbatch experiments/core_fg/run_paired_task.sbatch
TASK_NAME=cube_restack TOTAL_TRIALS=10 sbatch experiments/core_fg/run_paired_task.sbatch
```
