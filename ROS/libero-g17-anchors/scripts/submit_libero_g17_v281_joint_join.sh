#!/usr/bin/env bash
# v281: execute the already-proved second joint homotopy from measured neutral.
set -euo pipefail

G17=/home/qshou/capx_g17_observation_integrity
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
SBATCH_SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT="$PROJECT/outputs/libero_g17_fair_fg_v281_joint_join_20260819"
ANCHOR="$PROJECT/outputs/libero_g17_fair_fg_v280_planned_se3_20260819/manifests/SOURCE_SHA256.json"
TASK_ID=${TASK_ID:-0}
REPLICATE=${REPLICATE:-1}
TOTAL_TRIALS=${TOTAL_TRIALS:-2}

[[ -x "$PYTHON" && -f "$SBATCH_SCRIPT" && -f "$ANCHOR" ]]
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/arms/v281" "$EXPT/evidence"

PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_planned_se3.py" \
  "$G17/tests/test_reachable_se3.py" \
  "$G17/tests/test_trusted_se3.py" \
  "$G17/tests/test_tip_xy_only.py" \
  "$G17/tests/test_libero_g17_nearmiss_osc.py" \
  "$G17/tests/test_libero_g17_release_ready.py" \
  >"$EXPT/evidence/unit_tests.txt"

PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/audit_v280_planned_se3.py" \
  --joint-join --output "$EXPT/evidence/v272_trial2_lagged_joint_join.json" >/dev/null

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time, zipfile
from pathlib import Path

anchor = Path("$ANCHOR")
audit_path = Path("$EXPT/evidence/v272_trial2_lagged_joint_join.json")
tests_path = Path("$EXPT/evidence/unit_tests.txt")
audit = json.loads(audit_path.read_text())
if audit.get("schema") != "capx_v281_lagged_joint_join_audit_v1":
    raise SystemExit("v281 audit schema mismatch")
planned = audit.get("planned", {})
if not planned.get("ok") or planned.get("stages", [])[-1:] != ["neutral_1:goal_joint_join"]:
    raise SystemExit("v281 did not execute the proved second joint path")
if planned["max_commanded_arm_delta_rad"] > 0.180001 or planned["max_measured_arm_delta_rad"] > 0.2002:
    raise SystemExit("v281 joint bound failed")

paths = {
    Path("$G17/capx/integrations/franka/planned_se3.py"),
    Path("$G17/capx/integrations/franka/trusted_se3.py"),
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"),
    Path("$SBATCH_SCRIPT"),
    Path("$G17/scripts/audit_v280_planned_se3.py"),
    Path("$G17/tests/test_planned_se3.py"),
    Path("$G17/scripts/submit_libero_g17_v281_joint_join.sh"),
}
files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
data = {
    "schema": "capx_libero_g17_v281_joint_join_source_v1",
    "anchor": "v280 live false-negative evidence and source",
    "anchor_manifest": str(anchor),
    "anchor_manifest_sha256": hashlib.sha256(anchor.read_bytes()).hexdigest(),
    "root_cause_fixed": (
        "v280 discarded a proved second-leg joint homotopy when a redundant "
        "measured-state IK re-solve returned a boundary false negative"
    ),
    "only_runtime_change": (
        "join measured neutral state to the proved second joint waypoint path "
        "using the existing bounded command primitive"
    ),
    "files": dict(sorted(files.items())),
    "lagged_exact_replay": str(audit_path),
    "lagged_exact_replay_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    "unit_tests": str(tests_path),
    "unit_tests_sha256": hashlib.sha256(tests_path.read_bytes()).hexdigest(),
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
manifest = Path("$EXPT/manifests/SOURCE_SHA256.json")
manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
with zipfile.ZipFile(Path("$EXPT/manifests/SOURCE_SNAPSHOT.zip"), "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, arcname=str(path).lstrip("/"))
print(json.dumps(data, indent=2))
PYMANIFEST

env \
  -u CAPX_LIBERO_G17_GRASP_REPLAY \
  -u CAPX_LIBERO_G17_V277_CANONICAL_GRASP \
  -u CAPX_LIBERO_G17_V278_TRUSTED_SE3 \
  -u CAPX_LIBERO_G17_V279_REACHABLE_SE3 \
  -u CAPX_LIBERO_G17_V280_PLANNED_SE3 \
  CAPX_LIBERO_G17_XY_LOCK=1 \
  CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 \
  CAPX_LIBERO_G17_RELEASE_READY=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH=1 \
  CAPX_LIBERO_G17_V269_MECH=1 \
  CAPX_LIBERO_G17_V269_SYMBOL_SOURCE=fg \
  CAPX_LIBERO_G17_V272_FAIR_LIVE=1 \
  CAPX_LIBERO_G17_V274_MECH_CONTRACT=1 \
  CAPX_LIBERO_G17_V276_COMPLETION_IK=1 \
  CAPX_LIBERO_G17_V281_JOINT_JOIN=1 \
  PYTHONPATH="$G17" "$PYTHON" - <<'PYPREFLIGHT'
from capx.integrations.franka.planned_se3 import command_faithful_join_contract, command_faithful_join_enabled
from capx.verification.libero_g17_takeover import schema_version
s = schema_version()
assert "v2.81" in s and "command-faithful-joint-join" in s, s
assert command_faithful_join_enabled()
assert command_faithful_join_contract()["schema"] == "capx_libero_planned_se3_contract_v2"
print("v281_joint_join_preflight_ok", s, command_faithful_join_contract())
PYPREFLIGHT

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "v281 evidence and source manifest ready"
  exit 0
fi

jname=$(printf 'libero-g17-v281-join-t%02d-r%02d' "$TASK_ID" "$REPLICATE")
log="$EXPT/logs/${jname}-%j.out"
jid=$(sbatch \
  --export=ALL,ARM=v281,TASK_ID="$TASK_ID",REPLICATE="$REPLICATE",TOTAL_TRIALS="$TOTAL_TRIALS" \
  --job-name="$jname" --output="$log" --error="$log" \
  "$SBATCH_SCRIPT" | awk '{print $4}')

"$PYTHON" - <<PYREGISTER
import hashlib, json, time
from pathlib import Path
source = Path("$EXPT/manifests/SOURCE_SHA256.json")
snapshot = Path("$EXPT/manifests/SOURCE_SNAPSHOT.zip")
data = {
    "schema": "capx_libero_g17_v281_joint_join_submit_v1",
    "task_id": int("$TASK_ID"), "replicate": int("$REPLICATE"),
    "total_trials": int("$TOTAL_TRIALS"), "job_id": "$jid",
    "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "source_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PYREGISTER

echo "submitted $jid task=$TASK_ID rep=$REPLICATE trials=$TOTAL_TRIALS"
