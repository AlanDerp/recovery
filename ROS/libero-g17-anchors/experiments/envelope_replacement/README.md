# Envelope Replacement v1.3

This treatment intercepts the native A1 regeneration request after ordinary M2
prompt construction. It performs no student PT call and exposes no verifier
protocol, full GT packet, bridge, schema, delta, or extra message to the action
student. Only explicit false current-state claims in model-authored code
comments may receive a minimal grammar-preserving local correction. A private
fail-closed contradiction gate rejects proposals that are supported,
equivalent, merely compatible, not decidable, broader than the smallest false
semantic component, unsupported by the selected GT, or destructive of any
other meaning. This edits the native regeneration information; it does not
create a PT worksheet, closed state, state summary, or visible GT packet.
Stdout, numeric data, code tokens, parameters, and API evidence are immutable.

Silent turns replay the exact A1 response. A raw request-hash hit is preferred.
The only compatibility fallback restores the registered legacy runtime root in
a traceback that is now represented as `<CAPX_RUNTIME>`, then requires the
resulting normalized request hash to match. It does not ignore or rewrite any
other prompt text. Archived visual-feedback files are not replay keys because
older runs could save a later-mutated MuJoCo RGB buffer; current runs copy the
first rendered frame before encoding and saving it.

## Smoke

```bash
CAPX_SUBMISSION_GIT_REV=$(git rev-parse HEAD)
sbatch --time=01:00:00 \
  --export=ALL,PROJECT_DIR=/home/qshou/capx_pgsvp_v9_frozen,CAPX_SUBMISSION_GIT_REV="$CAPX_SUBMISSION_GIT_REV",CAPX_PAIR_SMOKE=1,TASK_NAME=cube_restack,TOTAL_TRIALS=1 \
  experiments/envelope_replacement/run_paired_task.sbatch
```

Run a separate smoke for `cube_stack`. A valid smoke must produce
`PAIR_SMOKE_DONE` and a valid `PAIR_VALIDATION.json`.

## Formal frozen-A1 comparison

Formal runs use `TOTAL_TRIALS=10` and `CAPX_GLOBAL_SEED=20260718`. The runner
automatically selects `frozen_reference_only`: restack references job `496713`
(`0.606/6`) and stack references job `496763` (`0.612/6`). It verifies the
registered reference, full replay, source pair-validation hash, initial states,
and initial-policy responses, but it never resamples or re-executes A1. Only
the treatment arm is executed.

Valid failed trials are analyzed under
[`docs/P1_P6_FAILURE_ATTRIBUTION.md`](../../docs/P1_P6_FAILURE_ATTRIBUTION.md).
P0 chain failures invalidate the run and are never relabeled as method errors.
