#!/usr/bin/env bash
# Replay task4 with PT top-down grasp plus inverse-ingress extraction.
set -euo pipefail

G17=/home/qshou/capx_g17_v304_ingress_retreat_poc
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_libero_g17_v296a_task4_pt_swept_audit.sh"
EXPT="$PROJECT/outputs/libero_g17_v304_ingress_retreat_poc_20260820"

mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/evidence"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_libero_g17_gt_endpoint_forensic.py" \
  "$G17/tests/test_libero_g17_perceptual_grasp_family.py" \
  "$G17/tests/test_libero_g17_perceptual_swept_path.py" \
  "$G17/tests/test_libero_g17_acquisition_extract.py" \
  "$G17/tests/test_measured_seed_ik.py" \
  >"$EXPT/evidence/unit_tests.txt"

"$PYTHON" - <<PY
import hashlib, json, time
from pathlib import Path
paths = [
    Path("$G17/capx/integrations/franka/gt_endpoint_forensic.py"),
    Path("$G17/capx/integrations/franka/acquisition_extract.py"),
    Path("$G17/capx/integrations/franka/perceptual_grasp_family.py"),
    Path("$G17/capx/integrations/franka/perceptual_swept_path.py"),
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/tests/test_libero_g17_gt_endpoint_forensic.py"),
    Path("$RUNNER"),
]
data = {
    "schema": "capx_libero_g17_v304_ingress_retreat_poc_source_v1",
    "commit": "31db00e",
    "behavior_effect": "PT_topdown_grasp_then_return_to_reached_ingress_endpoint_after_close",
    "candidate_selection": "zero_height_offset_then_maximum_PT_scene_arm_endpoint_clearance",
    "candidate_family": "8_topdown_finger_yaws_x_3_contact_heights_v286_legacy_mapping",
    "comparison": "corrected_frame_PT_vs_MuJoCo_endpoint_contacts",
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
export CAPX_LIBERO_G17_V302A_TOPDOWN_GRASP_PROBE=1
export CAPX_LIBERO_G17_V303_TOPDOWN_GRASP_BEHAVIOR=1
export CAPX_LIBERO_G17_V304_INGRESS_RETREAT=1
unset CAPX_LIBERO_G17_V300A_GT_ENDPOINT_FORENSIC
unset CAPX_LIBERO_G17_V301A_CORRECT_GRASP_FRAME_PROBE
unset CAPX_LIBERO_G17_V299B_DIAGONAL_ENTRY_PROBE
unset CAPX_LIBERO_G17_V299A_NULLSPACE_PROBE
unset CAPX_LIBERO_G17_V297A_PROXIMAL_CONTACT_PROBE
unset CAPX_LIBERO_G17_V297B_ACQUISITION_RECOVERY
export CAPX_G17_WORKTREE_OVERRIDE="$G17"
export CAPX_V296_EXPT_DIR_OVERRIDE="$EXPT"
jid=$(sbatch \
  --export=ALL,ARM=v286,TASK_ID=4,REPLICATE=0,TOTAL_TRIALS=1 \
  --job-name=libero-g17-v304-task4-retreat \
  --output="$EXPT/logs/libero-g17-v304-task4-retreat-%j.out" \
  --error="$EXPT/logs/libero-g17-v304-task4-retreat-%j.out" \
  "$RUNNER" | awk '{print $4}')
"$PYTHON" - <<PY
import json, time
from pathlib import Path
data = {
    "schema": "capx_libero_g17_v304_ingress_retreat_poc_submit_v1",
    "job_id": "$jid",
    "task_id": 4,
    "trial": 1,
    "behavior_effect": "PT_topdown_grasp_then_return_to_reached_ingress_endpoint_after_close",
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(
    json.dumps(data, indent=2) + "\n"
)
print(json.dumps(data, indent=2))
PY
