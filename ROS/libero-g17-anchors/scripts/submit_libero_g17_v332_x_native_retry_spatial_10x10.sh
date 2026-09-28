#!/usr/bin/env bash
# Fair live Spatial 10x10 ablation: immutable v332 + audited X, with C disabled.
# Acquisition recovery ownership belongs exclusively to FG-informed native retry.
set -euo pipefail

G17=/home/qshou/capx_g17_v359_v332_x_clean_c
RUNTIME_ANCHOR=${CAPX_LIBERO_RUNTIME_ANCHOR:-/project/peilab/qshou/capx_libero_runtime_anchor_v346_20260827}
PYTHON="$RUNTIME_ANCHOR/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh"
SELF="$G17/scripts/submit_libero_g17_v332_x_native_retry_spatial_10x10.sh"
V332=/home/qshou/capx_g17_results/libero_g17_v332_native_retry_spatial_10x10_live_20260825/FINAL_100_TRIAL_SUMMARY.json
EXPT=${CAPX_V332_X_NATIVE_RETRY_10X10_EXPT_OVERRIDE:-/home/qshou/capx_g17_results/libero_g17_v332_x_native_retry_spatial_10x10_20260829}
TOTAL_TRIALS=${TOTAL_TRIALS:-10}
HEAD=$(git -C "$G17" rev-parse HEAD)
JOB_TIME=${CAPX_V332_X_NATIVE_RETRY_10X10_JOB_TIME:-03:00:00}
JOB_PARTITION=${CAPX_V332_X_NATIVE_RETRY_10X10_JOB_PARTITION:-normal}
JOB_QOS=${CAPX_V332_X_NATIVE_RETRY_10X10_JOB_QOS:-normal_qos}

git -C "$G17" diff --quiet
git -C "$G17" diff --cached --quiet
test -f "$V332"
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/evidence" "$EXPT/arms/v332-x-native-retry"
bash -n "$RUNNER" "$SELF"
"$PYTHON" "$G17/scripts/verify_libero_runtime_anchor.py" "$RUNTIME_ANCHOR"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_libero_g17_acquisition_extract.py" \
  "$G17/tests/test_libero_g17_v319_transition_closure.py" \
  "$G17/tests/test_tip_xy_only.py" \
  "$G17/tests/test_libero_g17_v331_postclose_acquisition.py" \
  "$G17/tests/test_libero_g17_v350_transaction_ownership.py" \
  "$G17/tests/test_libero_g17_v354_fg_native_retry.py" \
  >"$EXPT/evidence/unit_tests.txt"

export CAPX_G17_WORKTREE_OVERRIDE="$G17"
export CAPX_LIBERO_RUNTIME_ANCHOR="$RUNTIME_ANCHOR"
export CAPX_SPATIAL_10X10_EXPT_OVERRIDE="$EXPT"
export CAPX_SPATIAL_10X10_METHOD_ANCHOR="$HEAD"
export CAPX_SPATIAL_10X10_SOURCE_SCHEMA=capx_libero_g17_v332_x_native_retry_spatial_10x10_source_v1
export CAPX_SPATIAL_10X10_MODE_NAME=libero_g17_v332_x_native_retry_spatial_10x10
export CAPX_SPATIAL_10X10_ARM_NAME=v332-x-native-retry
export CAPX_SPATIAL_10X10_STACK_LABEL=v332-plus-audited-X-native-retry-no-C
export CAPX_SPATIAL_10X10_EXTRA_REQUIRED_FLAGS=CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY,CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN,CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION,CAPX_LIBERO_G17_V313_ACQUISITION_BINDING,CAPX_LIBERO_G17_V319_TRANSITION_CLOSURE,CAPX_LIBERO_G17_V320_SEMANTIC_TRANSITION,CAPX_LIBERO_G17_V331_POSTCLOSE_STATE_MACHINE,CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP,CAPX_LIBERO_G17_V350G_TOTAL_ACQUISITION_OWNERSHIP,CAPX_LIBERO_G17_V350K_TRANSPARENT_NATIVE_TRANSACTION,CAPX_LIBERO_G17_V350L_FG_REGENERATION_ON_EXECUTION_FAILURE,CAPX_LIBERO_G17_V350M_PROPOSAL_FAILURE_FG_REGENERATION,CAPX_LIBERO_G17_V351_CLOSED_TRANSACTION_ROUTER,CAPX_LIBERO_G17_V353_COMPLETE_INVERSE_ESCAPE,CAPX_LIBERO_G17_V354_FG_NATIVE_RETRY
export CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY=1
export CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN=1
export CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION=1
export CAPX_LIBERO_G17_V313_ACQUISITION_BINDING=1
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

# Define the ablation at the routing boundary, not by relying on dormant flags.
unset CAPX_LIBERO_G17_V317_GRASP_RECOVERY_POC CAPX_LIBERO_G17_V318_ADAPTIVE_GRASP_RECOVERY
unset CAPX_LIBERO_G17_V355_CLEAN_C CAPX_LIBERO_G17_V360_TRUE_GR CAPX_LIBERO_G17_V361_CORRECTED_TIP_GR
unset CAPX_LIBERO_G17_V332_NATIVE_RETRY_ABLATION
unset CAPX_LIBERO_G17_CODE_PREFIX_REPLAY CAPX_LIBERO_G17_GRASP_REPLAY
unset CAPX_LIBERO_G17_V333_RR_COMPLETION CAPX_LIBERO_G17_V337_COMMAND_LEVEL_TIPXY
unset CAPX_LIBERO_G17_V338_WAYPOINT_TIPXY CAPX_LIBERO_G17_V339_PROGRESS_PLANE_TIPXY
unset CAPX_LIBERO_G17_V341_RR_CONTACT_ESCAPE CAPX_LIBERO_G17_V343_RR_SUPPORTED_TOUCHDOWN
unset CAPX_LIBERO_G17_V344_RR_EVENT_XY_EXECUTION CAPX_LIBERO_G17_V346_RR_NOOP_TOLERANCE_REPAIR
unset CAPX_LIBERO_G17_V349_TRANSACTION_DOMINANCE_SHADOW CAPX_LIBERO_G17_V350D_CONTACT_AWARE_TRANSACTION
unset CAPX_LIBERO_G17_V356_PREGR_EXECUTION_CLOSURE CAPX_LIBERO_G17_V357_TRANSACTION_BLOCKED_RECOVERY
unset CAPX_LIBERO_G17_V358_OBSERVED_BLOCKED_TRANSACTION

"$PYTHON" - <<PY
import hashlib, json, time
from pathlib import Path
root = Path("$G17")
rel = [
    "capx/integrations/franka/libero.py",
    "capx/integrations/franka/acquisition_ownership.py",
    "capx/integrations/franka/acquisition_extract.py",
    "capx/integrations/franka/postclose_acquisition.py",
    "capx/integrations/franka/fg_native_retry.py",
    "capx/integrations/franka/grasp_recovery.py",
    "capx/integrations/franka/tip_xy_only.py",
    "capx/verification/libero_g17_takeover.py",
    "scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh",
    "scripts/submit_libero_g17_v332_x_native_retry_spatial_10x10.sh",
]
paths = [root / p for p in rel]
v332 = Path("$V332")
data = {
    "schema": "capx_libero_g17_v332_x_native_retry_spatial_10x10_source_v1",
    "method_anchor": "$HEAD", "harness_git_head": "$HEAD",
    "composition": {
        "base": "immutable v332 (82/100)",
        "X": "audited post-v332 ownership/router/regeneration bug fixes through v354",
        "adaptive_GR_or_clean_C": False,
        "nonheld_endpoint_owner": "FG-informed native retry",
        "RR": "v332 unchanged",
    },
    "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "immutable_v332": {"path": str(v332), "sha256": hashlib.sha256(v332.read_bytes()).hexdigest()},
    "protocol": {"suite": "libero_spatial", "tasks": list(range(10)),
        "trials_per_task": int("$TOTAL_TRIALS"), "global_seed": 20260718,
        "scheduling": "10 parallel live task jobs", "code_prefix_replay": False,
        "grasp_replay": False, "comparison": "v332 immutable 82/100 and v359 clean-C 74/100"},
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/SOURCE_SHA256.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY

job_ids=()
for TASK_ID in $(seq 0 9); do
  jname=$(printf 'g17-v332x-nr-t%02d' "$TASK_ID")
  jid=$(sbatch --account=peilab --partition="$JOB_PARTITION" --qos="$JOB_QOS" \
    --export=ALL,TASK_ID="$TASK_ID",TOTAL_TRIALS="$TOTAL_TRIALS",CAPX_SUBMISSION_GIT_REV="$HEAD" \
    --time="$JOB_TIME" --job-name="$jname" \
    --output="$EXPT/logs/${jname}-%j.out" --error="$EXPT/logs/${jname}-%j.out" \
    "$RUNNER" | awk '{print $4}')
  job_ids+=("$jid")
  echo "submitted task=$TASK_ID job=$jid"
done

"$PYTHON" - <<PY
import json, time
from pathlib import Path
data = {"schema": "capx_libero_g17_v332_x_native_retry_spatial_10x10_submit_v1",
    "method_anchor": "$HEAD", "harness_git_head": "$HEAD",
    "tasks": list(range(10)), "trials_per_task": int("$TOTAL_TRIALS"),
    "job_ids": "${job_ids[*]}".split(), "partition": "$JOB_PARTITION",
    "qos": "$JOB_QOS", "time_limit": "$JOB_TIME", "expt": "$EXPT",
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PY
