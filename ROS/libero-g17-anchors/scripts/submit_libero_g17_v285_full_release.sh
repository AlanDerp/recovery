#!/usr/bin/env bash
# v285: v283 motion/RR unchanged; finish all open steps before success latching.
set -euo pipefail

G17=/home/qshou/capx_g17_observation_integrity
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT="$PROJECT/outputs/libero_g17_fair_fg_v285_full_release_20260819"
TASK_ID=${TASK_ID:-0}
REPLICATE=${REPLICATE:-1}
TOTAL_TRIALS=${TOTAL_TRIALS:-2}

mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/arms/v285" "$EXPT/evidence"
PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_libero_g17_live_isolation.py" \
  "$G17/tests/test_measured_seed_ik.py" \
  "$G17/tests/test_libero_g17_takeover.py" \
  >"$EXPT/evidence/unit_tests.txt"

CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY=1 PYTHONPATH="$G17" "$PYTHON" \
  "$G17/scripts/audit_v282_measured_seed_ik.py" \
  --output "$EXPT/evidence/v272_trial2_tool_pi_ik.json" >/dev/null

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time, zipfile
from pathlib import Path
paths = {
    Path("$G17/capx/verification/libero_g17_live_isolation.py"),
    Path("$G17/capx/envs/simulators/libero.py"),
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/capx/integrations/franka/measured_seed_ik.py"),
    Path("$G17/capx/integrations/motion/pyroki_snippets/_solve_ik.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"),
    Path("$G17/tests/test_libero_g17_live_isolation.py"),
    Path("$G17/tests/test_measured_seed_ik.py"), Path("$RUNNER"),
    Path("$G17/scripts/submit_libero_g17_v285_full_release.sh"),
}
files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
audit = Path("$EXPT/evidence/v272_trial2_tool_pi_ik.json")
tests = Path("$EXPT/evidence/unit_tests.txt")
data = {
    "schema": "capx_libero_g17_v285_full_release_source_v1",
    "parent": "v284 release-commit candidate; disqualified for latching after the first open step",
    "single_change": "defer task-success latching until the complete 40-step physical gripper-open window finishes",
    "files": dict(sorted(files.items())),
    "audit": str(audit), "audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
    "tests": str(tests), "tests_sha256": hashlib.sha256(tests.read_bytes()).hexdigest(),
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
manifest = Path("$EXPT/manifests/SOURCE_SHA256.json")
manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
with zipfile.ZipFile(Path("$EXPT/manifests/SOURCE_SNAPSHOT.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for p in paths: z.write(p, arcname=str(p).lstrip("/"))
print(json.dumps(data, indent=2))
PYMANIFEST

env \
  CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 \
  CAPX_LIBERO_G17_RELEASE_READY=1 CAPX_LIBERO_G17_NEARMISS_OSC=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH=1 \
  CAPX_LIBERO_G17_V269_MECH=1 CAPX_LIBERO_G17_V269_SYMBOL_SOURCE=fg \
  CAPX_LIBERO_G17_V272_FAIR_LIVE=1 CAPX_LIBERO_G17_V274_MECH_CONTRACT=1 \
  CAPX_LIBERO_G17_V276_COMPLETION_IK=1 CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY=1 \
  CAPX_LIBERO_G17_V285_FULL_RELEASE_WINDOW=1 PYTHONPATH="$G17" "$PYTHON" - <<'PYPREFLIGHT'
from capx.integrations.franka.measured_seed_ik import measured_seed_ik_contract
from capx.verification.libero_g17_takeover import schema_version
s = schema_version(); c = measured_seed_ik_contract()
assert "v2.85" in s and "full-release-window" in s, s
assert c["tool_z_pi_symmetry"] == 1 and c["native_blocking_commands_per_stage"] == 1
assert c["cartesian_waypoints"] == c["joint_delta_gate"] == c["terminal_residual_gate"] == 0
print("v285_full_release_preflight_ok", s, c)
PYPREFLIGHT

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then exit 0; fi
jname=$(printf 'libero-g17-v285-full-release-t%02d-r%02d' "$TASK_ID" "$REPLICATE")
log="$EXPT/logs/${jname}-%j.out"
jid=$(sbatch --export=ALL,ARM=v285,TASK_ID="$TASK_ID",REPLICATE="$REPLICATE",TOTAL_TRIALS="$TOTAL_TRIALS" \
  --job-name="$jname" --output="$log" --error="$log" "$RUNNER" | awk '{print $4}')
"$PYTHON" - <<PYREGISTER
import hashlib, json, time
from pathlib import Path
s=Path("$EXPT/manifests/SOURCE_SHA256.json"); z=Path("$EXPT/manifests/SOURCE_SNAPSHOT.zip")
d={"schema":"capx_libero_g17_v285_full_release_submit_v1","task_id":int("$TASK_ID"),
"replicate":int("$REPLICATE"),"total_trials":int("$TOTAL_TRIALS"),"job_id":"$jid",
"source_manifest_sha256":hashlib.sha256(s.read_bytes()).hexdigest(),
"source_snapshot_sha256":hashlib.sha256(z.read_bytes()).hexdigest(),
"submitted_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(json.dumps(d,indent=2)+"\n")
print(json.dumps(d,indent=2))
PYREGISTER
echo "submitted $jid task=$TASK_ID rep=$REPLICATE trials=$TOTAL_TRIALS"
