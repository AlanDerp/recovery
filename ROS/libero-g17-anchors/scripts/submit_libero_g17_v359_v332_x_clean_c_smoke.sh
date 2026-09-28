#!/usr/bin/env bash
# Frozen task-4 mechanism replay for v332 + retained bug fixes + clean C.
set -euo pipefail

G17=/home/qshou/capx_g17_v359_v332_x_clean_c
RUNTIME_ANCHOR=${CAPX_LIBERO_RUNTIME_ANCHOR:-/project/peilab/qshou/capx_libero_runtime_anchor_v346_20260827}
PYTHON="$RUNTIME_ANCHOR/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh"
PREFIX="$G17/experiments/libero_g17_v355/assets/v354_job535684_task4_failures_full_prefix.json"
TRIAL_IDX=${CAPX_V359_SMOKE_TRIAL:-3}
EXPT=${CAPX_V359_SMOKE_EXPT:-/home/qshou/capx_g17_results/libero_g17_v359_v332_x_clean_c_task4_20260828}
HEAD=$(git -C "$G17" rev-parse HEAD)

git -C "$G17" diff --quiet
git -C "$G17" diff --cached --quiet
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/evidence"
bash -n "$RUNNER" "$0"
"$PYTHON" "$G17/scripts/verify_libero_runtime_anchor.py" "$RUNTIME_ANCHOR"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_libero_g17_code_prefix_replay.py" \
  "$G17/tests/test_libero_g17_acquisition_extract.py" \
  "$G17/tests/test_libero_g17_v317_grasp_recovery.py" \
  "$G17/tests/test_libero_g17_v319_transition_closure.py" \
  "$G17/tests/test_tip_xy_only.py" \
  "$G17/tests/test_libero_g17_v350_transaction_ownership.py" \
  "$G17/tests/test_libero_g17_v354_fg_native_retry.py" \
  >"$EXPT/evidence/unit_tests.txt"

export CAPX_G17_WORKTREE_OVERRIDE="$G17"
export CAPX_LIBERO_RUNTIME_ANCHOR="$RUNTIME_ANCHOR"
export CAPX_SPATIAL_10X10_EXPT_OVERRIDE="$EXPT"
export CAPX_SPATIAL_10X10_METHOD_ANCHOR="$HEAD"
export CAPX_SPATIAL_10X10_SOURCE_SCHEMA=capx_libero_g17_v359_v332_x_clean_c_source_v1
export CAPX_SPATIAL_10X10_MODE_NAME=libero_g17_v359_v332_x_clean_c
export CAPX_SPATIAL_10X10_ARM_NAME=v359-v332-x-clean-c
export CAPX_SPATIAL_10X10_STACK_LABEL=v332-plus-retained-bugfixes-plus-clean-C
export CAPX_SPATIAL_10X10_ALLOW_PREFIX_REPLAY=1
export CAPX_SPATIAL_10X10_FROZEN_PREFIX_PACKAGE="$PREFIX"
export CAPX_SPATIAL_10X10_EXTRA_REQUIRED_FLAGS=CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY,CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN,CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION,CAPX_LIBERO_G17_V313_ACQUISITION_BINDING,CAPX_LIBERO_G17_V318_ADAPTIVE_GRASP_RECOVERY,CAPX_LIBERO_G17_V319_TRANSITION_CLOSURE,CAPX_LIBERO_G17_V320_SEMANTIC_TRANSITION,CAPX_LIBERO_G17_V331_POSTCLOSE_STATE_MACHINE,CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP,CAPX_LIBERO_G17_V350G_TOTAL_ACQUISITION_OWNERSHIP,CAPX_LIBERO_G17_V350K_TRANSPARENT_NATIVE_TRANSACTION,CAPX_LIBERO_G17_V350L_FG_REGENERATION_ON_EXECUTION_FAILURE,CAPX_LIBERO_G17_V350M_PROPOSAL_FAILURE_FG_REGENERATION,CAPX_LIBERO_G17_V351_CLOSED_TRANSACTION_ROUTER,CAPX_LIBERO_G17_V353_COMPLETE_INVERSE_ESCAPE,CAPX_LIBERO_G17_V354_FG_NATIVE_RETRY,CAPX_LIBERO_G17_V355_CLEAN_C
export CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY=1
export CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN=1
export CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION=1
export CAPX_LIBERO_G17_V313_ACQUISITION_BINDING=1
export CAPX_LIBERO_G17_V318_ADAPTIVE_GRASP_RECOVERY=1
export CAPX_LIBERO_G17_V319_TRANSITION_CLOSURE=1
export CAPX_LIBERO_G17_V320_SEMANTIC_TRANSITION=1
export CAPX_LIBERO_G17_V331_POSTCLOSE_STATE_MACHINE=1
export CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP=1
export CAPX_LIBERO_G17_V350G_TOTAL_ACQUISITION_OWNERSHIP=1
export CAPX_LIBERO_G17_V350K_TRANSPARENT_NATIVE_TRANSACTION=1
export CAPX_LIBERO_G17_V350L_FG_REGENERATION_ON_EXECUTION_FAILURE=1
export CAPX_LIBERO_G17_V350M_PROPOSAL_FAILURE_FG_REGENERATION=1
export CAPX_LIBERO_G17_V351_CLOSED_TRANSACTION_ROUTER=1
export CAPX_LIBERO_G17_V353_COMPLETE_INVERSE_ESCAPE=1
export CAPX_LIBERO_G17_V354_FG_NATIVE_RETRY=1
export CAPX_LIBERO_G17_V355_CLEAN_C=1
unset CAPX_LIBERO_G17_V332_NATIVE_RETRY_ABLATION
unset CAPX_LIBERO_G17_V349_TRANSACTION_DOMINANCE_SHADOW
unset CAPX_LIBERO_G17_V350D_CONTACT_AWARE_TRANSACTION
unset CAPX_LIBERO_G17_V356_PREGR_EXECUTION_CLOSURE
unset CAPX_LIBERO_G17_V357_TRANSACTION_BLOCKED_RECOVERY
unset CAPX_LIBERO_G17_V358_OBSERVED_BLOCKED_TRANSACTION

"$PYTHON" - <<PY
import hashlib, json, time
from pathlib import Path
root = Path("$G17")
paths = [
    root / "capx/integrations/franka/libero.py",
    root / "capx/integrations/franka/acquisition_ownership.py",
    root / "capx/integrations/franka/acquisition_extract.py",
    root / "capx/integrations/franka/postclose_acquisition.py",
    root / "capx/integrations/franka/fg_native_retry.py",
    root / "capx/integrations/franka/grasp_recovery.py",
    root / "capx/integrations/franka/tip_xy_only.py",
    root / "docs/LIBERO_G17_V359_V332_X_CLEAN_C_ASSET_AUDIT.md",
    root / "scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh",
    root / "scripts/submit_libero_g17_v359_v332_x_clean_c_smoke.sh",
    Path("$PREFIX"),
]
data = {
    "schema": "capx_libero_g17_v359_v332_x_clean_c_source_v1",
    "git_head": "$HEAD", "method_anchor": "$HEAD",
    "base_method_anchor": "9dade5af9d4800f04503f87faba13193850447b6",
    "source_replay": {"version": "v354", "job_id": "535684", "task_id": 4},
    "behavior_effect": "v332_native_first_attempt_plus_clean_NearMiss_GR",
    "hard_connected_retry": False,
    "score_is_fair_live": False,
    "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/SOURCE_SHA256.json").write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\n"
)
PY

# The shared evaluator has no start-index option. TOTAL_TRIALS=N executes
# trials 1..N; the frozen target is trial N. Do not describe this as a
# single-trial run or pass the previously ignored RESUME_IDX variable.
jid=$(sbatch \
  --time=00:30:00 \
  --export=ALL,TASK_ID=4,TOTAL_TRIALS="$TRIAL_IDX",CAPX_SUBMISSION_GIT_REV="$HEAD" \
  --job-name="g17-v359-c-t${TRIAL_IDX}" \
  --output="$EXPT/logs/g17-v359-c-t${TRIAL_IDX}-%j.out" \
  --error="$EXPT/logs/g17-v359-c-t${TRIAL_IDX}-%j.out" \
  "$RUNNER" | awk '{print $4}')

"$PYTHON" - <<PY
import json, time
from pathlib import Path
data = {
    "schema": "capx_libero_g17_v359_v332_x_clean_c_submit_v1",
    "git_head": "$HEAD", "job_id": "$jid", "task_id": 4,
    "trials": [$TRIAL_IDX], "source_job": "535684",
    "score_is_fair_live": False,
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
path = Path("$EXPT/manifests/v359-t${TRIAL_IDX}-${jid}.json")
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(json.dumps(data, indent=2, sort_keys=True))
PY
