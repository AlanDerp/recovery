#!/usr/bin/env bash
# Audit lower drawer-entry contact heights on frozen task-4 trial 1.
set -euo pipefail

G17=/home/qshou/capx_g17_v298e_entry_z_probe
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_libero_g17_v296a_task4_pt_swept_audit.sh"
EXPT="$PROJECT/outputs/libero_g17_v298e_entry_z_probe_20260820"

mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/evidence"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_libero_g17_perceptual_grasp_family.py" \
  "$G17/tests/test_libero_g17_perceptual_swept_path.py" \
  "$G17/tests/test_measured_seed_ik.py" \
  >"$EXPT/evidence/unit_tests.txt"

"$PYTHON" - <<PY
import hashlib, json, time
from pathlib import Path
paths = [
    Path("$G17/capx/integrations/franka/perceptual_grasp_family.py"),
    Path("$G17/capx/integrations/franka/perceptual_swept_path.py"),
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/tests/test_libero_g17_perceptual_grasp_family.py"),
    Path("$RUNNER"),
]
data = {
    "schema": "capx_libero_g17_v298e_entry_z_probe_source_v1",
    "commit": "6f1eb42",
    "behavior_effect": "audit_only",
    "azimuth_indices": [4, 5, 6, 7, 8],
    "finger_roll_count": 4,
    "near_side_offset_m": 0.06,
    "tip_z_offsets_m": [0.0, -0.02, -0.04],
    "task_id": 4,
    "trial": 1,
    "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/SOURCE_SHA256.json").write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\n"
)
PY

export CAPX_LIBERO_G17_V296B_CANDIDATE_PROBE=1
unset CAPX_LIBERO_G17_V297A_PROXIMAL_CONTACT_PROBE
unset CAPX_LIBERO_G17_V297B_ACQUISITION_RECOVERY
export CAPX_G17_WORKTREE_OVERRIDE="$G17"
export CAPX_V296_EXPT_DIR_OVERRIDE="$EXPT"
jid=$(sbatch \
  --export=ALL,ARM=v286,TASK_ID=4,REPLICATE=0,TOTAL_TRIALS=1 \
  --job-name=libero-g17-v298e-task4-entry-z \
  --output="$EXPT/logs/libero-g17-v298e-task4-entry-z-%j.out" \
  --error="$EXPT/logs/libero-g17-v298e-task4-entry-z-%j.out" \
  "$RUNNER" | awk '{print $4}')
"$PYTHON" - <<PY
import json, time
from pathlib import Path
data = {
    "schema": "capx_libero_g17_v298e_entry_z_probe_submit_v1",
    "job_id": "$jid",
    "task_id": 4,
    "trial": 1,
    "behavior_effect": "audit_only",
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(
    json.dumps(data, indent=2) + "\n"
)
print(json.dumps(data, indent=2))
PY
