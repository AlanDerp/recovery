#!/usr/bin/env bash
# Read-only PT swept-path audit over frozen v293 identity + v294 evidence.
set -euo pipefail

G17=/home/qshou/capx_g17_v296a_pt_swept_audit
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_libero_g17_v296a_task4_pt_swept_audit.sh"
EXPT="$PROJECT/outputs/libero_g17_v296a_pt_swept_audit_20260820"
TOTAL_TRIALS=${TOTAL_TRIALS:-2}
REPLICATE=${REPLICATE:-0}
TASK_ID=${TASK_ID:-4}

mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/evidence"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_libero_g17_acquisition_audit.py" \
  "$G17/tests/test_libero_g17_perceptual_swept_path.py" \
  "$G17/tests/test_measured_seed_ik.py" \
  "$G17/tests/test_libero_g17_xy_lock.py::test_publish_and_live_lock_requires_sibling_bands" \
  "$G17/tests/test_libero_g17_xy_lock.py::test_bound_identity_survives_empty_and_relation_flip" \
  "$G17/tests/test_libero_g17_xy_lock.py::test_v291_xyz_match_rejects_xy_alias_at_wrong_height" \
  >"$EXPT/evidence/unit_tests.txt"

IK_EVIDENCE="$PROJECT/outputs/libero_g17_v291_task4_acquisition_20260820/evidence/v272_trial2_tool_pi_ik.json"
test -s "$IK_EVIDENCE"
"$PYTHON" - <<PYAUDIT
import json
from pathlib import Path
d = json.loads(Path("$IK_EVIDENCE").read_text())
assert d.get("schema") == "capx_v282_measured_seed_ik_audit_v1"
assert d.get("contract", {}).get("tool_z_pi_symmetry") == 1
assert d.get("native_execution", {}).get("accepted") is True
PYAUDIT

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time
from pathlib import Path
paths = [
    Path("$G17/capx/integrations/franka/acquisition_audit.py"),
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/capx/integrations/franka/measured_seed_ik.py"),
    Path("$G17/capx/integrations/franka/perceptual_swept_path.py"),
    Path("$G17/capx/envs/simulators/libero.py"),
    Path("$G17/capx/verification/libero_g17_xy_lock.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"),
    Path("$G17/capx/verification/libero_g17_nearmiss_osc.py"),
    Path("$G17/tests/test_libero_g17_acquisition_audit.py"),
    Path("$G17/tests/test_libero_g17_perceptual_swept_path.py"),
    Path("$G17/tests/test_libero_g17_xy_lock.py"),
    Path("$G17/tests/test_measured_seed_ik.py"),
    Path("$RUNNER"),
]
files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
data = {
    "schema": "capx_libero_g17_v296a_pt_swept_audit_source_v1",
    "parent_commit": "db9b88c034535ca9ecb4b8c5d00d670b715b59f5",
    "task_id": int("$TASK_ID"),
    "changes": [
        "private acquisition transaction audit",
        "XYZ-consistent same-name target mask binding",
        "all point-prompt masks considered before rejecting grounding",
        "GraspNet top-k measured-branch candidate selection with fail-open",
        "unreachable approach preview and skip without poisoning contact IK",
        "measured-aligned duplicate goto idempotence",
        "attachment-correct acquisition state transition",
        "GraspNet contact-point geometry audit",
        "no fallback to a mask already rejected by XYZ",
        "task payload identity persists across empty and wrong-grasp regen",
        "RelXY token may refresh while the bound target handle remains stable",
        "private native joint-command exit, residual, progress, and contact evidence",
        "no target, candidate-selection, controller, or action changes from v293",
        "RGB-D plus measured-joint plus URDF swept-path prediction",
        "fixed-shape deterministic voxel scene and target separation",
        "exact native q-goal command-level swept-path audit",
        "exact-command live RGB-D refresh before pre-capture motion audit",
        "separate arm-obstacle risk from expected hand and target contact",
        "audit-only prediction; zero candidate-selection or execution effect",
    ],
    "files": dict(sorted(files.items())),
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/SOURCE_SHA256.json").write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(data, indent=2))
PYMANIFEST

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

jname=$(printf 'libero-g17-v296a-task%02d-audit-r%02d' "$TASK_ID" "$REPLICATE")
log="$EXPT/logs/${jname}-%j.out"
jid=$(sbatch \
  --export=ALL,ARM=v286,TASK_ID="$TASK_ID",REPLICATE="$REPLICATE",TOTAL_TRIALS="$TOTAL_TRIALS" \
  --job-name="$jname" --output="$log" --error="$log" \
  "$RUNNER" | awk '{print $4}')

"$PYTHON" - <<PYREGISTER
import json, time
from pathlib import Path
data = {
    "schema": "capx_libero_g17_v296a_pt_swept_audit_submit_v1",
    "job_id": "$jid",
    "task_id": int("$TASK_ID"),
    "replicate": int("$REPLICATE"),
    "total_trials": int("$TOTAL_TRIALS"),
    "slurm_arm_compatibility_label": "v286",
    "actual_method": "v296a read-only PT swept-path audit over v293 identity + v286 FG+RR",
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(
    json.dumps(data, indent=2) + "\n"
)
print(json.dumps(data, indent=2))
PYREGISTER
echo "submitted $jid task=$TASK_ID rep=$REPLICATE trials=$TOTAL_TRIALS"
