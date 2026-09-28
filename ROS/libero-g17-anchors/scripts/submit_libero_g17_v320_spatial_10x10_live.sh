#!/usr/bin/env bash
# Frozen v320 spatial 10x10 live characterization: 10 tasks x 10 trials.
set -euo pipefail

G17=/home/qshou/capx_g17_v320_semantic_transition
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_libero_g17_v305_topaccess_spatial_10x10.sh"
SUBMIT="$G17/scripts/submit_libero_g17_v320_spatial_10x10_live.sh"
EXPT="$PROJECT/outputs/libero_g17_v320_spatial_10x10_live_20260824"
V286="$PROJECT/outputs/libero_g17_v286_spatial_10x10_20260819"
METHOD_ANCHOR=06c78320a67e2723ba08f6fe8fc3a541cd548906
TOTAL_TRIALS=${TOTAL_TRIALS:-10}
HEAD=$(git -C "$G17" rev-parse HEAD)

method_paths=(
  capx/envs/runner.py
  capx/envs/tasks/base.py
  capx/execution_control.py
  capx/integrations/franka/acquisition_audit.py
  capx/integrations/franka/grasp_recovery.py
  capx/integrations/franka/libero.py
  capx/integrations/franka/perceptual_grasp_family.py
  capx/integrations/franka/transition_closure.py
  capx/integrations/franka/acquisition_extract.py
  capx/integrations/franka/perceptual_swept_path.py
  capx/integrations/franka/measured_seed_ik.py
  capx/verification/libero_g17_takeover.py
  capx/verification/libero_g17_nearmiss_osc.py
)
git -C "$G17" diff --quiet "$METHOD_ANCHOR" -- "${method_paths[@]}"

mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/evidence" "$EXPT/arms/v320"
bash -n "$RUNNER" "$SUBMIT"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_libero_g17_v319_transition_closure.py" \
  "$G17/tests/test_libero_g17_v317_grasp_recovery.py" \
  "$G17/tests/test_libero_g17_acquisition_audit.py" \
  "$G17/tests/test_libero_g17_acquisition_extract.py" \
  "$G17/tests/test_libero_g17_perceptual_grasp_family.py" \
  "$G17/tests/test_measured_seed_ik.py" \
  "$G17/tests/test_libero_g17_nearmiss_osc.py" \
  "$G17/tests/test_api_server_fail_closed.py" \
  >"$EXPT/evidence/unit_tests.txt"

export CAPX_G17_WORKTREE_OVERRIDE="$G17"
export CAPX_SPATIAL_10X10_EXPT_OVERRIDE="$EXPT"
export CAPX_SPATIAL_10X10_METHOD_ANCHOR="$METHOD_ANCHOR"
export CAPX_SPATIAL_10X10_SOURCE_SCHEMA=capx_libero_g17_v320_spatial_10x10_source_v1
export CAPX_SPATIAL_10X10_MODE_NAME=libero_g17_v320_spatial_10x10_live
export CAPX_SPATIAL_10X10_ARM_NAME=v320
export CAPX_SPATIAL_10X10_STACK_LABEL=v293,v296a,v296b,v302a,v303,v304,v305,v309,v312,v313,v318,v319,v320
export CAPX_SPATIAL_10X10_EXTRA_REQUIRED_FLAGS=CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY,CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN,CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION,CAPX_LIBERO_G17_V313_ACQUISITION_BINDING,CAPX_LIBERO_G17_V318_ADAPTIVE_GRASP_RECOVERY,CAPX_LIBERO_G17_V319_TRANSITION_CLOSURE,CAPX_LIBERO_G17_V320_SEMANTIC_TRANSITION
export CAPX_LIBERO_G17_V309_ACQUIRE_EVENT_BOUNDARY=1
export CAPX_LIBERO_G17_V312_MULTI_CONTACT_TOPDOWN=1
export CAPX_LIBERO_G17_V312_ACQUISITION_TRANSACTION=1
export CAPX_LIBERO_G17_V313_ACQUISITION_BINDING=1
export CAPX_LIBERO_G17_V318_ADAPTIVE_GRASP_RECOVERY=1
export CAPX_LIBERO_G17_V319_TRANSITION_CLOSURE=1
export CAPX_LIBERO_G17_V320_SEMANTIC_TRANSITION=1
unset CAPX_LIBERO_G17_V317_GRASP_RECOVERY_POC CAPX_LIBERO_G17_V317_FROZEN_TRIAL02_TRANSACTION
unset CAPX_LIBERO_G17_V315_GR_SHADOW CAPX_LIBERO_G17_V316_CONTACT_RANKING
unset CAPX_LIBERO_G17_V314_GR_CLOSURE CAPX_LIBERO_G17_V309_TOPDOWN_SOFT_FALLBACK
unset CAPX_LIBERO_G17_V306_STABLE_TOPDOWN CAPX_LIBERO_G17_V307_CONTACT_PRESERVING_TOPDOWN
unset CAPX_LIBERO_G17_V308_CONTACT_READY CAPX_LIBERO_G17_V297B_ACQUISITION_RECOVERY
unset CAPX_LIBERO_G17_CODE_PREFIX_REPLAY CAPX_LIBERO_G17_GRASP_REPLAY CAPX_LIBERO_G17_ACQUIRE_FACTS

"$PYTHON" - <<PY
import hashlib, json, time
from pathlib import Path
method_paths = [
    "capx/envs/runner.py",
    "capx/envs/tasks/base.py",
    "capx/execution_control.py",
    "capx/integrations/franka/acquisition_audit.py",
    "capx/integrations/franka/grasp_recovery.py",
    "capx/integrations/franka/libero.py",
    "capx/integrations/franka/perceptual_grasp_family.py",
    "capx/integrations/franka/transition_closure.py",
    "capx/integrations/franka/acquisition_extract.py",
    "capx/integrations/franka/perceptual_swept_path.py",
    "capx/integrations/franka/measured_seed_ik.py",
    "capx/verification/libero_g17_takeover.py",
    "capx/verification/libero_g17_nearmiss_osc.py",
]
paths = [Path("$G17") / p for p in method_paths]
paths += [Path("$RUNNER"), Path("$SUBMIT")]
data = {
    "schema": "capx_libero_g17_v320_spatial_10x10_source_v1",
    "method_anchor": "$METHOD_ANCHOR",
    "harness_git_head": "$HEAD",
    "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "protocol": {
        "suite": "libero_spatial", "tasks": list(range(10)),
        "trials_per_task": int("$TOTAL_TRIALS"),
        "scheduling": "10_parallel_jobs_x_1_task_x_10_live_trials",
        "global_seed": 20260718,
        "code_prefix_replay": False, "grasp_replay": False,
        "process_isolated_trials": True,
        "task_local_long_lived_services": True,
        "reference": "$V286/manifests/ABSOLUTE_SCORE.json",
    },
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/SOURCE_SHA256.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(json.dumps(data["protocol"], indent=2))
PY

job_ids=()
for TASK_ID in $(seq 0 9); do
  jname=$(printf 'libero-g17-v320-s10x10-t%02d' "$TASK_ID")
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
    "schema": "capx_libero_g17_v320_spatial_10x10_submit_v1",
    "method_anchor": "$METHOD_ANCHOR", "harness_git_head": "$HEAD",
    "tasks": list(range(10)), "total_trials": int("$TOTAL_TRIALS"),
    "job_ids": """${job_ids[*]}""".split(),
    "expt": "$EXPT", "v286_reference": "$V286",
    "code_prefix_replay": False, "grasp_replay": False,
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(json.dumps(d, indent=2) + "\n")
print(json.dumps(d, indent=2))
PY
