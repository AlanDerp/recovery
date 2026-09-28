# FGSRP experiments

FGSRP compares a native recovery method with the same method after private,
minimal correction of directly contradicted facts in its native report.

## M2 comment carrier

The formal ten-trial cube runs use the registered M2 controls (`6/10` for both
cube stack and cube restack) and do not resample A1:

```bash
sbatch --export=ALL,PROJECT_DIR=/home/qshou/capx_pgsvp_v9_frozen,CAPX_FGSRP_CARRIER=m2_comments,TASK_NAME=cube_restack,TOTAL_TRIALS=10 experiments/fgsrp/run_paired_task.sbatch
```

## M3 visual-difference carrier

M3 is a distinct native method and cannot use an M2 replay. A new M3 control
runs first with action-invisible passive GT capture. Its complete request
trajectory, source-frame RGB fingerprints, and qualitative GT transition
hashes are then registered before the treatment arm starts.

Historical clean M3 A0 arms are registered independently with
`register_frozen_a0.py`; this audits only the A0 arm and does not inherit the
validity of an obsolete treatment from the same pair directory.

Historical registered ten-trial M3 controls are diagnostic references:

- cube stack: job `500242`, `1.000/0.599/5`, 76 frozen model calls;
- cube restack: job `500239`, `0.900/0.804/8`, 62 frozen model calls.

They predate passive transition capture and are not selected automatically for
formal FGSRP comparison. A formal M3 pair creates one new preregistered live A1
trajectory in the same allocation. When normalized requests match but source
pixels differ, replay is accepted only if the treatment's exact `(trial, turn)`
predicate-transition hash equals that A1 reference; otherwise validation fails.

```bash
sbatch --export=ALL,PROJECT_DIR=/home/qshou/capx_pgsvp_v9_frozen,CAPX_FGSRP_CARRIER=m3_visual_difference,TASK_NAME=cube_restack,TOTAL_TRIALS=1,CAPX_PAIR_SMOKE=1 experiments/fgsrp/run_paired_task.sbatch
```

Both carriers use `capx/verification/fgsrp.py`, the same verifier and gate
prompts, and the same request-level validator. Only the native report adapter
and native M2/M3 config differ.

## Exact treatment rerun

Register the first valid live treatment pair, then verify that exact trajectory:

```bash
python experiments/core_fg/runtime/tools/register_fgsrp_treatment.py \
  --pair-root "$PAIR_ROOT" \
  --output-replay "$REPLAY" \
  --output-reference "$REFERENCE"

sbatch --export=ALL,PROJECT_DIR=/home/qshou/capx_pgsvp_v9_frozen,\
CAPX_FGSRP_TREATMENT_FULL_REPLAY_FILE="$REPLAY",\
CAPX_FGSRP_TREATMENT_REFERENCE="$REFERENCE",... \
experiments/fgsrp/run_paired_task.sbatch
```

The verification run is invalid if any model request is live or misses the
registered replay. It must also reproduce the complete request/response
sequence, FGSRP turn signatures, initial states, and per-trial outcomes. Every
valid run emits `TRIAL_ATTRIBUTION_EVIDENCE.json`; unresolved failures are not
accepted as evidence for non-monotone harm.

An M2 pair smoke freezes canonical `pi0`, executes a local A1 smoke, exports
that smoke's full request trajectory, and replays matching envelopes in the
FGSRP arm. It does not require post-execution requests to match the historical
ten-trial A1 trajectory. Formal M2 runs still reference the registered double-6
controls without re-executing them.

The registered one-trial M3 restack diagnostic reproduces the failed A1 from
job `500319` (`1.000/0.025/0`) without resampling Qwen:

```text
experiments/fgsrp/replays/cube_restack_qwen37_m3_a1_full_seed6000_1trial_job500319_v1.json
experiments/fgsrp/replays/cube_restack_m3_a1_reference_seed6000_1trial_job500319_v1.json
```
