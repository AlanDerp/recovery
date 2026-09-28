#!/usr/bin/env bash
# v350a: v350 ownership plus total, single-execution native transactions.
set -euo pipefail

G17=/home/qshou/capx_g17_v350_transaction_ownership
RUNTIME_ANCHOR=${CAPX_LIBERO_RUNTIME_ANCHOR:-/project/peilab/qshou/capx_libero_runtime_anchor_v346_20260827}
PYTHON="$RUNTIME_ANCHOR/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh"
EXPT=${CAPX_V350_OWNERSHIP_EXPT_OVERRIDE:-/home/qshou/capx_g17_results/libero_g17_v350a_transaction_total_task4_20260827}
TOTAL_TRIALS=${TOTAL_TRIALS:-2}
HEAD=$(git -C "$G17" rev-parse HEAD)

git -C "$G17" diff --quiet
git -C "$G17" diff --cached --quiet
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/evidence" "$EXPT/arms/v350-ownership"
bash -n "$RUNNER" "$0"
"$PYTHON" "$G17/scripts/verify_libero_runtime_anchor.py" "$RUNTIME_ANCHOR"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_libero_g17_v350_transaction_ownership.py" \
  "$G17/tests/test_libero_g17_v331_postclose_acquisition.py" \
  "$G17/tests/test_libero_g17_v317_grasp_recovery.py" \
  "$G17/tests/test_libero_g17_v319_transition_closure.py" \
  "$G17/tests/test_libero_g17_acquisition_extract.py" \
  "$G17/tests/test_libero_g17_acquisition_audit.py" \
  >"$EXPT/evidence/unit_tests.txt"

export CAPX_G17_WORKTREE_OVERRIDE="$G17"
export CAPX_LIBERO_RUNTIME_ANCHOR="$RUNTIME_ANCHOR"
export CAPX_SPATIAL_10X10_EXPT_OVERRIDE="$EXPT"
export CAPX_SPATIAL_10X10_METHOD_ANCHOR="$HEAD"
export CAPX_SPATIAL_10X10_SOURCE_SCHEMA=capx_libero_g17_v350a_transaction_total_source_v1
export CAPX_SPATIAL_10X10_MODE_NAME=libero_g17_v350a_transaction_total
export CAPX_SPATIAL_10X10_ARM_NAME=v350a-transaction-total
export CAPX_SPATIAL_10X10_STACK_LABEL=v332+v350-transaction-ownership+v350a-native-transaction-total
export CAPX_SPATIAL_10X10_EXTRA_REQUIRED_FLAGS=CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY,CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN,CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION,CAPX_LIBERO_G17_V313_ACQUISITION_BINDING,CAPX_LIBERO_G17_V318_ADAPTIVE_GRASP_RECOVERY,CAPX_LIBERO_G17_V319_TRANSITION_CLOSURE,CAPX_LIBERO_G17_V320_SEMANTIC_TRANSITION,CAPX_LIBERO_G17_V331_POSTCLOSE_STATE_MACHINE,CAPX_LIBERO_G17_V332_NATIVE_RETRY_ABLATION,CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP
export CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY=1
export CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN=1
export CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION=1
export CAPX_LIBERO_G17_V313_ACQUISITION_BINDING=1
export CAPX_LIBERO_G17_V318_ADAPTIVE_GRASP_RECOVERY=1
export CAPX_LIBERO_G17_V319_TRANSITION_CLOSURE=1
export CAPX_LIBERO_G17_V320_SEMANTIC_TRANSITION=1
export CAPX_LIBERO_G17_V331_POSTCLOSE_STATE_MACHINE=1
export CAPX_LIBERO_G17_V332_NATIVE_RETRY_ABLATION=1
export CAPX_LIBERO_G17_V350_TRANSACTION_OWNERSHIP=1
unset CAPX_LIBERO_G17_V309_TOPDOWN_SOFT_FALLBACK
unset CAPX_LIBERO_G17_CODE_PREFIX_REPLAY CAPX_LIBERO_G17_GRASP_REPLAY
unset CAPX_LIBERO_G17_V349_TRANSACTION_DOMINANCE_SHADOW

"$PYTHON" - <<PY
import hashlib, json, time
from pathlib import Path
root = Path("$G17")
runtime_anchor = Path("$RUNTIME_ANCHOR").resolve()
runtime_manifest = runtime_anchor / "CRITICAL_SHA256.json"
paths = [
    root / "capx/integrations/franka/libero.py",
    root / "capx/integrations/franka/acquisition_ownership.py",
    root / "capx/integrations/franka/acquisition_audit.py",
    root / "capx/integrations/franka/acquisition_extract.py",
    root / "capx/integrations/franka/grasp_recovery.py",
    root / "docs/LIBERO_G17_V350_TRANSACTION_OWNERSHIP_CONTRACT.md",
    root / "env_configs/libero/franka_libero_m3.yaml",
    root / "scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh",
    root / "scripts/verify_libero_runtime_anchor.py",
    root / "scripts/submit_libero_g17_v350_task4_ownership.sh",
]
data = {
    "schema": "capx_libero_g17_v350a_transaction_total_source_v1",
    "git_head": "$HEAD",
    "parent": {"version": "v332", "score": "82/100"},
    "task_id": 4,
    "trials": int("$TOTAL_TRIALS"),
    "behavior_effect": "single_writer_single_execution_acquisition_transaction",
    "method_delta": "complete pose-only native proposals into connected pregrasp-contact IK transactions; terminalize execution failure",
    "runtime_anchor": str(runtime_anchor),
    "runtime_manifest_sha256": hashlib.sha256(runtime_manifest.read_bytes()).hexdigest(),
    "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/SOURCE_SHA256.json").write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\n"
)
PY

jid=$(sbatch \
  --time=01:00:00 \
  --export=ALL,TASK_ID=4,TOTAL_TRIALS="$TOTAL_TRIALS",CAPX_SUBMISSION_GIT_REV="$HEAD" \
  --job-name=libero-g17-v350a-t4-txn \
  --output="$EXPT/logs/libero-g17-v350a-t4-txn-%j.out" \
  --error="$EXPT/logs/libero-g17-v350a-t4-txn-%j.out" \
  "$RUNNER" | awk '{print $4}')

"$PYTHON" - <<PY
import json, time
from pathlib import Path
data = {
    "schema": "capx_libero_g17_v350a_transaction_total_submit_v1",
    "git_head": "$HEAD", "job_id": "$jid", "task_id": 4,
    "trials": int("$TOTAL_TRIALS"),
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(data, indent=2, sort_keys=True))
PY
