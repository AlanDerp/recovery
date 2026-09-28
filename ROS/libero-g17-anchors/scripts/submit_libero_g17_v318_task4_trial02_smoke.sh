#!/usr/bin/env bash
# v318 mechanism smoke: frozen v313 trial02, 2cm blocked -> 1cm recovery.
set -euo pipefail

G17=/home/qshou/capx_g17_v318_adaptive_grasp_recovery
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_libero_g17_v296a_task4_pt_swept_audit.sh"
EXPT="$PROJECT/outputs/libero_g17_v318_task4_trial02_adaptive_gr_20260822"
PREFIX_DIR="$G17/experiments/libero_g17_v309/assets/task4_job526268_cb0"

mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/evidence"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_libero_g17_v317_grasp_recovery.py" \
  "$G17/tests/test_libero_g17_acquisition_extract.py" \
  "$G17/tests/test_libero_g17_perceptual_grasp_family.py" \
  >"$EXPT/evidence/unit_tests.txt"

export CAPX_LIBERO_G17_V296B_CANDIDATE_PROBE=1
export CAPX_LIBERO_G17_V302A_TOPDOWN_GRASP_PROBE=1
export CAPX_LIBERO_G17_V303_TOPDOWN_GRASP_BEHAVIOR=1
export CAPX_LIBERO_G17_V304_INGRESS_RETREAT=1
export CAPX_LIBERO_G17_V305_FG_EVENT_STOP=1
export CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY=1
export CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN=1
export CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION=1
export CAPX_LIBERO_G17_V313_ACQUISITION_BINDING=1
export CAPX_LIBERO_G17_V318_ADAPTIVE_GRASP_RECOVERY=1
export CAPX_LIBERO_G17_V317_FROZEN_TRIAL02_TRANSACTION=1
unset CAPX_LIBERO_G17_V317_GRASP_RECOVERY_POC
unset CAPX_LIBERO_G17_V315_GR_SHADOW CAPX_LIBERO_G17_V316_CONTACT_RANKING
unset CAPX_LIBERO_G17_V314_GR_CLOSURE CAPX_LIBERO_G17_V309_TOPDOWN_SOFT_FALLBACK
unset CAPX_LIBERO_G17_V306_STABLE_TOPDOWN CAPX_LIBERO_G17_V307_CONTACT_PRESERVING_TOPDOWN
unset CAPX_LIBERO_G17_V308_CONTACT_READY CAPX_LIBERO_G17_V297B_ACQUISITION_RECOVERY
export CAPX_G17_WORKTREE_OVERRIDE="$G17"
export CAPX_V296_EXPT_DIR_OVERRIDE="$EXPT"
export CAPX_LIBERO_G17_PREFIX_DIR_OVERRIDE="$PREFIX_DIR"

"$PYTHON" - <<PY
import hashlib, json, subprocess, time
from pathlib import Path
paths = [Path("$G17/capx/integrations/franka/grasp_recovery.py"), Path("$G17/capx/integrations/franka/libero.py"), Path("$G17/capx/integrations/franka/tip_xy_only.py"), Path("$RUNNER")]
data = {
    "schema": "capx_libero_g17_v318_task4_trial02_source_v1",
    "commit": subprocess.check_output(["git", "-C", "$G17", "rev-parse", "HEAD"], text=True).strip(),
    "parent": {"version": "v317", "commit": "aecae49"},
    "frozen_trial": 2,
    "near_miss": "empty AND z-aligned AND target-contact",
    "controller": "FK jaw axis; FG discrete sign; intrinsic 20/10/5/2.5mm inward probes",
    "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/SOURCE_SHA256.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY

jid=$(sbatch --time=01:00:00 \
  --export=ALL,ARM=v286,TASK_ID=4,REPLICATE=0,TOTAL_TRIALS=2,START_TRIAL=2 \
  --job-name=libero-g17-v318-t4-smoke \
  --output="$EXPT/logs/libero-g17-v318-t4-smoke-%j.out" \
  --error="$EXPT/logs/libero-g17-v318-t4-smoke-%j.out" \
  "$RUNNER" | awk '{print $4}')

"$PYTHON" - <<PY
import json, subprocess, time
from pathlib import Path
data = {
    "schema": "capx_libero_g17_v318_task4_trial02_submit_v1",
    "job_id": "$jid",
    "commit": subprocess.check_output(["git", "-C", "$G17", "rev-parse", "HEAD"], text=True).strip(),
    "frozen_trial": 2,
    "controller": "z-aligned contact -> FK jaw axis + FG sign + 20/10/5/2.5mm inward probes",
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PY
