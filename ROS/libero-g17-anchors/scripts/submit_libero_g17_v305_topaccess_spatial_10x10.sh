#!/usr/bin/env bash
# Exact v305 top-access horizontal regression against v286 spatial 10x10.
set -euo pipefail

G17=/home/qshou/capx_g17_v305_topaccess_10x10
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh"
EXPT="$PROJECT/outputs/libero_g17_v305_topaccess_spatial_10x10_retry2_20260821"
V286="$PROJECT/outputs/libero_g17_v286_spatial_10x10_20260819"
TOTAL_TRIALS=${TOTAL_TRIALS:-10}
HEAD=$(git -C "$G17" rev-parse HEAD)

# The harness may live in a child commit, but the method implementation must
# remain byte-for-byte identical to the frozen v305 anchor.
git -C "$G17" diff --quiet 2abec5e -- \
  capx/integrations/franka/libero.py \
  capx/integrations/franka/acquisition_audit.py \
  capx/integrations/franka/acquisition_extract.py \
  capx/integrations/franka/perceptual_grasp_family.py \
  capx/integrations/franka/perceptual_swept_path.py \
  capx/integrations/franka/measured_seed_ik.py \
  capx/verification/libero_g17_takeover.py \
  capx/verification/libero_g17_nearmiss_osc.py

mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/arms/v305-topaccess" "$EXPT/evidence"
bash -n "$RUNNER"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_libero_g17_perceptual_grasp_family.py" \
  "$G17/tests/test_libero_g17_acquisition_extract.py" \
  "$G17/tests/test_libero_g17_perceptual_swept_path.py" \
  "$G17/tests/test_measured_seed_ik.py" \
  "$G17/tests/test_libero_g17_nearmiss_osc.py" \
  "$G17/tests/test_api_server_fail_closed.py" \
  >"$EXPT/evidence/unit_tests.txt"

"$PYTHON" - <<PY
import hashlib, json, time
from pathlib import Path
paths = [
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/capx/integrations/franka/acquisition_audit.py"),
    Path("$G17/capx/integrations/franka/acquisition_extract.py"),
    Path("$G17/capx/integrations/franka/perceptual_grasp_family.py"),
    Path("$G17/capx/integrations/franka/perceptual_swept_path.py"),
    Path("$G17/capx/integrations/franka/measured_seed_ik.py"),
    Path("$G17/capx/envs/runner.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"),
    Path("$G17/capx/verification/libero_g17_nearmiss_osc.py"),
    Path("$RUNNER"),
    Path("$G17/scripts/submit_libero_g17_v305_topaccess_spatial_10x10.sh"),
    Path("$G17/tests/test_api_server_fail_closed.py"),
    Path("$PROJECT/capx/third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet/config.yaml"),
]
data = {
    "schema": "capx_libero_g17_v305_topaccess_spatial_10x10_source_v1",
    "method_anchor": "2abec5e", "harness_git_head": "$HEAD",
    "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "protocol": {
        "suite": "libero_spatial", "tasks": list(range(10)),
        "trials": int("$TOTAL_TRIALS"),
        "scheduling": "10_parallel_jobs_x_1_task_x_10_live_trials",
        "code_prefix_replay": False, "grasp_replay": False,
        "base_stack": "v286 FG+RR",
        "unique_delta": "v293/v296/v302/v303/v304/v305 acquisition top-access",
        "v306_v307_v308": False,
        "comparison": "$V286/manifests/ABSOLUTE_SCORE.json",
    },
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/SOURCE_SHA256.json").write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(data["protocol"], indent=2))
PY

job_ids=()
for TASK_ID in $(seq 0 9); do
  jname=$(printf 'libero-g17-v305-topaccess-s10x10-t%02d' "$TASK_ID")
  log="$EXPT/logs/${jname}-%j.out"
  jid=$(sbatch \
    --export=ALL,TASK_ID="$TASK_ID",TOTAL_TRIALS="$TOTAL_TRIALS",CAPX_SUBMISSION_GIT_REV="$HEAD" \
    --job-name="$jname" --output="$log" --error="$log" \
    "$RUNNER" | awk '{print $4}')
  job_ids+=("$jid")
  echo "submitted task=$TASK_ID job=$jid"
done

"$PYTHON" - <<PY
import json, time
from pathlib import Path
d = {
    "schema": "capx_libero_g17_v305_topaccess_spatial_10x10_submit_v1",
    "method_anchor": "2abec5e", "harness_git_head": "$HEAD",
    "tasks": list(range(10)), "total_trials": int("$TOTAL_TRIALS"),
    "job_ids": """${job_ids[*]}""".split(),
    "expt": "$EXPT", "v286_reference": "$V286",
    "code_prefix_replay": False,
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(json.dumps(d, indent=2) + "\n")
print(json.dumps(d, indent=2))
PY
