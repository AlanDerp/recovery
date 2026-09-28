#!/usr/bin/env bash
# Fair live task-06 ablation against v331 job 530683. Smoke first, then K=10.
set -euo pipefail

G17=/home/qshou/capx_g17_v332_native_retry_ablation
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh"
SUBMIT="$G17/scripts/submit_libero_g17_v332_native_retry_task6_live.sh"
TASK_ID=6
TOTAL_TRIALS=${TOTAL_TRIALS:-1}
if [[ "$TOTAL_TRIALS" -le "2" ]]; then
  PHASE=smoke-k${TOTAL_TRIALS}
  LIMIT=00:30:00
else
  PHASE=k${TOTAL_TRIALS}
  LIMIT=03:00:00
fi
EXPT=${CAPX_V332_EXPT_OVERRIDE:-/home/qshou/capx_g17_results/libero_g17_v332_native_retry_task6_${PHASE}_live_20260825}
HEAD=$(git -C "$G17" rev-parse HEAD)

git -C "$G17" diff --quiet
git -C "$G17" diff --cached --quiet
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/evidence" "$EXPT/arms/v332"
bash -n "$RUNNER" "$SUBMIT"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_libero_g17_v331_postclose_acquisition.py" \
  "$G17/tests/test_libero_g17_v317_grasp_recovery.py" \
  "$G17/tests/test_libero_g17_v319_transition_closure.py" \
  "$G17/tests/test_libero_g17_acquisition_extract.py" \
  "$G17/tests/test_libero_g17_acquisition_audit.py" \
  >"$EXPT/evidence/unit_tests.txt"

export CAPX_G17_WORKTREE_OVERRIDE="$G17"
export CAPX_SPATIAL_10X10_EXPT_OVERRIDE="$EXPT"
export CAPX_SPATIAL_10X10_METHOD_ANCHOR="$HEAD"
export CAPX_SPATIAL_10X10_SOURCE_SCHEMA=capx_libero_g17_v332_native_retry_task6_live_source_v1
export CAPX_SPATIAL_10X10_MODE_NAME=libero_g17_v332_native_retry_task6_${PHASE}_live
export CAPX_SPATIAL_10X10_ARM_NAME=v332
export CAPX_SPATIAL_10X10_STACK_LABEL=v293,v296a,v296b,v302a,v303,v304,v305,v309,v312,v313,v318,v319,v320,v331,v332
export CAPX_SPATIAL_10X10_EXTRA_REQUIRED_FLAGS=CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY,CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN,CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION,CAPX_LIBERO_G17_V313_ACQUISITION_BINDING,CAPX_LIBERO_G17_V318_ADAPTIVE_GRASP_RECOVERY,CAPX_LIBERO_G17_V319_TRANSITION_CLOSURE,CAPX_LIBERO_G17_V320_SEMANTIC_TRANSITION,CAPX_LIBERO_G17_V331_POSTCLOSE_STATE_MACHINE,CAPX_LIBERO_G17_V332_NATIVE_RETRY_ABLATION
export CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY=1
export CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN=1
export CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION=1
export CAPX_LIBERO_G17_V313_ACQUISITION_BINDING=1
export CAPX_LIBERO_G17_V318_ADAPTIVE_GRASP_RECOVERY=1
export CAPX_LIBERO_G17_V319_TRANSITION_CLOSURE=1
export CAPX_LIBERO_G17_V320_SEMANTIC_TRANSITION=1
export CAPX_LIBERO_G17_V331_POSTCLOSE_STATE_MACHINE=1
export CAPX_LIBERO_G17_V332_NATIVE_RETRY_ABLATION=1
unset CAPX_LIBERO_G17_V309_TOPDOWN_SOFT_FALLBACK
unset CAPX_LIBERO_G17_CODE_PREFIX_REPLAY CAPX_LIBERO_G17_GRASP_REPLAY

"$PYTHON" - <<PY
import hashlib, json, time
from pathlib import Path
root = Path("$G17")
rel = [
    "capx/integrations/franka/libero.py",
    "capx/integrations/franka/postclose_acquisition.py",
    "capx/integrations/franka/acquisition_audit.py",
    "capx/integrations/franka/acquisition_extract.py",
    "capx/integrations/franka/grasp_recovery.py",
    "capx/integrations/franka/perceptual_grasp_family.py",
    "capx/integrations/franka/transition_closure.py",
    "capx/integrations/franka/measured_seed_ik.py",
    "capx/verification/libero_g17_takeover.py",
    "capx/verification/libero_g17_nearmiss_osc.py",
    "scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh",
    "scripts/submit_libero_g17_v332_native_retry_task6_live.sh",
]
paths = [root / p for p in rel]
data = {
    "schema": "capx_libero_g17_v332_native_retry_task6_live_source_v1",
    "method_anchor": "$HEAD",
    "harness_git_head": "$HEAD",
    "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "protocol": {
        "suite": "libero_spatial", "task_id": 6,
        "trials": int("$TOTAL_TRIALS"), "global_seed": 20260718,
        "code_prefix_replay": False, "grasp_replay": False,
        "comparison": "v331 task-06 K10 live job-530683",
        "single_delta": "NearMiss bypasses GR after the same inverse escape and enters native regeneration",
    },
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/SOURCE_SHA256.json").write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\n"
)
PY

job=$(sbatch \
  --export=ALL,TASK_ID="$TASK_ID",TOTAL_TRIALS="$TOTAL_TRIALS",CAPX_SUBMISSION_GIT_REV="$HEAD" \
  --time="$LIMIT" \
  --job-name="libero-g17-v332-t6-${PHASE}" \
  --output="$EXPT/logs/libero-g17-v332-t6-${PHASE}-%j.out" \
  --error="$EXPT/logs/libero-g17-v332-t6-${PHASE}-%j.out" \
  "$RUNNER" | awk '{print $4}')

"$PYTHON" - <<PY
import json, time
from pathlib import Path
data = {
    "schema": "capx_libero_g17_v332_native_retry_task6_submit_v1",
    "job_id": "$job", "git_head": "$HEAD", "task_id": 6,
    "trials": int("$TOTAL_TRIALS"), "phase": "$PHASE",
    "v331_control_job": "530683", "v320_reference_job": "529360",
    "expt": "$EXPT",
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(
    json.dumps(data, indent=2) + "\n"
)
print(json.dumps(data, indent=2))
PY
