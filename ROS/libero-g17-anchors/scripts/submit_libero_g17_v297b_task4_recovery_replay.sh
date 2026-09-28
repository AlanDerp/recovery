#!/usr/bin/env bash
# Recovery-only replay of the frozen task-4 collision trajectory.
set -euo pipefail

G17=/home/qshou/capx_g17_v297b_recovery
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_libero_g17_v296a_task4_pt_swept_audit.sh"
EXPT="$PROJECT/outputs/libero_g17_v297b_acquisition_recovery_20260820"

mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/evidence"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_libero_g17_acquisition_recovery.py" \
  "$G17/tests/test_measured_seed_ik.py" \
  >"$EXPT/evidence/unit_tests.txt"

"$PYTHON" - <<PY
import hashlib, json, time
from pathlib import Path
paths = [
    Path("$G17/capx/integrations/franka/acquisition_recovery.py"),
    Path("$G17/capx/envs/simulators/libero.py"),
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/tests/test_libero_g17_acquisition_recovery.py"),
    Path("$RUNNER"),
]
data = {
    "schema": "capx_libero_g17_v297b_acquisition_recovery_source_v1",
    "parent_commit": "c3be5ac",
    "task_id": 4,
    "trial": 1,
    "behavior_change": "strong-stall stop plus reverse-measured-q rollback",
    "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/SOURCE_SHA256.json").write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\n"
)
PY

unset CAPX_LIBERO_G17_V296B_CANDIDATE_PROBE
unset CAPX_LIBERO_G17_V297A_PROXIMAL_CONTACT_PROBE
export CAPX_LIBERO_G17_V297B_ACQUISITION_RECOVERY=1
export CAPX_G17_WORKTREE_OVERRIDE="$G17"
export CAPX_V296_EXPT_DIR_OVERRIDE="$EXPT"
jid=$(sbatch \
  --export=ALL,ARM=v286,TASK_ID=4,REPLICATE=0,TOTAL_TRIALS=1 \
  --job-name=libero-g17-v297b-task4-recovery \
  --output="$EXPT/logs/libero-g17-v297b-task4-recovery-%j.out" \
  --error="$EXPT/logs/libero-g17-v297b-task4-recovery-%j.out" \
  "$RUNNER" | awk '{print $4}')
"$PYTHON" - <<PY
import json, time
from pathlib import Path
data = {
    "schema": "capx_libero_g17_v297b_acquisition_recovery_submit_v1",
    "job_id": "$jid",
    "task_id": 4,
    "trial": 1,
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(
    json.dumps(data, indent=2) + "\n"
)
print(json.dumps(data, indent=2))
PY
