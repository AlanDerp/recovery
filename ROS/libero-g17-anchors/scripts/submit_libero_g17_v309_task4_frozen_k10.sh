#!/usr/bin/env bash
# v309: formal-v305 CB0 replay with acquisition transaction closure + soft PT fallback.
set -euo pipefail

G17=/home/qshou/capx_g17_v309_acquire_closure
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_libero_g17_v296a_task4_pt_swept_audit.sh"
EXPT="$PROJECT/outputs/libero_g17_v309_task4_acquire_closure_20260821"
PREFIX_DIR="$G17/experiments/libero_g17_v309/assets/task4_job526268_cb0"

mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/evidence"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_code_block_breakpoint.py" \
  "$G17/tests/test_libero_g17_code_prefix_replay.py" \
  "$G17/tests/test_libero_g17_acquisition_extract.py" \
  "$G17/tests/test_libero_g17_perceptual_grasp_family.py" \
  "$G17/tests/test_measured_seed_ik.py" \
  >"$EXPT/evidence/unit_tests.txt"

"$PYTHON" - <<PY
import hashlib, json, subprocess, time
from pathlib import Path
paths = [
    Path("$G17/capx/execution_control.py"),
    Path("$G17/capx/envs/tasks/base.py"),
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/capx/integrations/franka/acquisition_audit.py"),
    Path("$G17/capx/integrations/franka/acquisition_extract.py"),
    Path("$G17/capx/integrations/franka/perceptual_grasp_family.py"),
    Path("$G17/tests/test_code_block_breakpoint.py"),
    Path("$PREFIX_DIR/code_prefix_replay_v1.json"),
    Path("$RUNNER"),
]
data = {
    "schema": "capx_libero_g17_v309_task4_acquire_closure_source_v1",
    "commit": subprocess.check_output(
        ["git", "-C", "$G17", "rev-parse", "HEAD"], text=True
    ).strip(),
    "task_id": 4,
    "trials": list(range(1, 11)),
    "source_prefix": "formal v305 live job-526268 sandboxrc_0 CB0, outcome-blind",
    "behavior_change": [
        "normal code-block boundary after physical close and optional inverse ingress",
        "baseline GraspNet fallback when no PT-clear contact-height topdown exists",
    ],
    "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/SOURCE_SHA256.json").write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\n"
)
PY

export CAPX_LIBERO_G17_V296B_CANDIDATE_PROBE=1
export CAPX_LIBERO_G17_V302A_TOPDOWN_GRASP_PROBE=1
export CAPX_LIBERO_G17_V303_TOPDOWN_GRASP_BEHAVIOR=1
export CAPX_LIBERO_G17_V304_INGRESS_RETREAT=1
export CAPX_LIBERO_G17_V305_FG_EVENT_STOP=1
export CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY=1
export CAPX_LIBERO_G17_V309_TOPDOWN_SOFT_FALLBACK=1
unset CAPX_LIBERO_G17_V306_STABLE_TOPDOWN
unset CAPX_LIBERO_G17_V307_CONTACT_PRESERVING_TOPDOWN
unset CAPX_LIBERO_G17_V308_CONTACT_READY
unset CAPX_LIBERO_G17_V297B_ACQUISITION_RECOVERY
export CAPX_G17_WORKTREE_OVERRIDE="$G17"
export CAPX_V296_EXPT_DIR_OVERRIDE="$EXPT"
export CAPX_LIBERO_G17_PREFIX_DIR_OVERRIDE="$PREFIX_DIR"

jid=$(sbatch \
  --time=02:00:00 \
  --export=ALL,ARM=v286,TASK_ID=4,REPLICATE=0,TOTAL_TRIALS=10 \
  --job-name=libero-g17-v309-task4-k10 \
  --output="$EXPT/logs/libero-g17-v309-task4-k10-%j.out" \
  --error="$EXPT/logs/libero-g17-v309-task4-k10-%j.out" \
  "$RUNNER" | awk '{print $4}')

"$PYTHON" - <<PY
import json, time
from pathlib import Path
data = {
    "schema": "capx_libero_g17_v309_task4_acquire_closure_submit_v1",
    "job_id": "$jid",
    "task_id": 4,
    "trials": list(range(1, 11)),
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(
    json.dumps(data, indent=2) + "\n"
)
print(json.dumps(data, indent=2))
PY
