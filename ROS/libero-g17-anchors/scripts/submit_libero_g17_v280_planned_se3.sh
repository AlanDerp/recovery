#!/usr/bin/env bash
# v280: command-faithful execution of fully preflighted joint homotopies.
set -euo pipefail

G17=/home/qshou/capx_g17_observation_integrity
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
SBATCH_SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT="$PROJECT/outputs/libero_g17_fair_fg_v280_planned_se3_20260819"
ANCHOR="$PROJECT/outputs/libero_g17_fair_fg_v279_reachable_se3_20260819/manifests/SOURCE_SHA256.json"
TASK_ID=${TASK_ID:-0}
REPLICATE=${REPLICATE:-1}
TOTAL_TRIALS=${TOTAL_TRIALS:-2}

[[ -x "$PYTHON" && -f "$SBATCH_SCRIPT" && -f "$ANCHOR" ]]
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/arms/v280" "$EXPT/evidence"

PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_planned_se3.py" \
  "$G17/tests/test_reachable_se3.py" \
  "$G17/tests/test_trusted_se3.py" \
  "$G17/tests/test_tip_xy_only.py" \
  "$G17/tests/test_libero_g17_nearmiss_osc.py" \
  "$G17/tests/test_libero_g17_release_ready.py" \
  >"$EXPT/evidence/unit_tests.txt"

PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/audit_v280_planned_se3.py" \
  --output "$EXPT/evidence/v272_trial2_lagged_planned_se3.json" >/dev/null

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time, zipfile
from pathlib import Path

anchor = Path("$ANCHOR")
audit_path = Path("$EXPT/evidence/v272_trial2_lagged_planned_se3.json")
tests_path = Path("$EXPT/evidence/unit_tests.txt")
audit = json.loads(audit_path.read_text())
if audit.get("schema") != "capx_v280_lagged_planned_se3_audit_v1":
    raise SystemExit("v280 lagged audit schema mismatch")
if audit.get("straight_preflight", {}).get("ok"):
    raise SystemExit("v280 audit did not reproduce the v272 dead end")
if audit["straight_preflight"].get("executed_commands") != 0:
    raise SystemExit("v280 rejected preflight executed commands")
planned = audit.get("planned", {})
if not planned.get("ok") or not str(planned.get("selected_route", "")).startswith("neutral_"):
    raise SystemExit("v280 planned recovery failed")
if planned["max_commanded_arm_delta_rad"] > 0.180001:
    raise SystemExit("v280 commanded joint bound failed")
if planned["max_measured_arm_delta_rad"] > 0.2002:
    raise SystemExit("v280 measured joint bound failed")

paths = {
    Path("$G17/capx/integrations/franka/planned_se3.py"),
    Path("$G17/capx/integrations/franka/trusted_se3.py"),
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"),
    Path("$SBATCH_SCRIPT"),
    Path("$G17/scripts/audit_v280_planned_se3.py"),
    Path("$G17/tests/test_planned_se3.py"),
    Path("$G17/scripts/submit_libero_g17_v280_planned_se3.sh"),
}
files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
data = {
    "schema": "capx_libero_g17_v280_planned_se3_source_v1",
    "anchor": "v279 failure evidence and source",
    "anchor_manifest": str(anchor),
    "anchor_manifest_sha256": hashlib.sha256(anchor.read_bytes()).hexdigest(),
    "root_cause_fixed": (
        "v279 proved joint configurations but re-solved only their Cartesian "
        "endpoint during execution, losing the proved Panda redundancy branch"
    ),
    "runtime_changes": [
        "execute the proved bounded joint waypoint sequence without re-running IK",
        "replan the target leg from measured joint state after the neutral leg",
        "bounded rollback to the exact pre-route joint branch on rejection",
        "v280 schema and runner plumbing",
    ],
    "preserved": [
        "v278/v279 live GraspNet target and quaternion",
        "v276 FG predicates and ReleaseReady law",
        "no grasp replay, canonical grasp, or continuous privileged truth",
    ],
    "files": dict(sorted(files.items())),
    "lagged_exact_replay": str(audit_path),
    "lagged_exact_replay_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    "unit_tests": str(tests_path),
    "unit_tests_sha256": hashlib.sha256(tests_path.read_bytes()).hexdigest(),
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
manifest = Path("$EXPT/manifests/SOURCE_SHA256.json")
manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
snapshot = Path("$EXPT/manifests/SOURCE_SNAPSHOT.zip")
with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, arcname=str(path).lstrip("/"))
print(json.dumps(data, indent=2))
PYMANIFEST

env \
  -u CAPX_LIBERO_G17_GRASP_REPLAY \
  -u CAPX_LIBERO_G17_V277_CANONICAL_GRASP \
  -u CAPX_LIBERO_G17_V278_TRUSTED_SE3 \
  -u CAPX_LIBERO_G17_V279_REACHABLE_SE3 \
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
  CAPX_LIBERO_G17_V280_PLANNED_SE3=1 \
  PYTHONPATH="$G17" "$PYTHON" - <<'PYPREFLIGHT'
from capx.integrations.franka.planned_se3 import planned_se3_contract, planned_se3_enabled
from capx.verification.libero_g17_takeover import schema_version
s = schema_version()
assert "v2.80" in s and "command-faithful-planned-se3" in s, s
assert planned_se3_enabled()
assert planned_se3_contract()["execution_representation"] == "joint_waypoints"
print("v280_planned_se3_preflight_ok", s, planned_se3_contract())
PYPREFLIGHT

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "v280 evidence and source manifest ready"
  exit 0
fi

jname=$(printf 'libero-g17-v280-plan-t%02d-r%02d' "$TASK_ID" "$REPLICATE")
log="$EXPT/logs/${jname}-%j.out"
jid=$(sbatch \
  --export=ALL,ARM=v280,TASK_ID="$TASK_ID",REPLICATE="$REPLICATE",TOTAL_TRIALS="$TOTAL_TRIALS" \
  --job-name="$jname" --output="$log" --error="$log" \
  "$SBATCH_SCRIPT" | awk '{print $4}')

"$PYTHON" - <<PYREGISTER
import hashlib, json, time
from pathlib import Path
source = Path("$EXPT/manifests/SOURCE_SHA256.json")
snapshot = Path("$EXPT/manifests/SOURCE_SNAPSHOT.zip")
data = {
    "schema": "capx_libero_g17_v280_planned_se3_submit_v1",
    "task_id": int("$TASK_ID"),
    "replicate": int("$REPLICATE"),
    "total_trials": int("$TOTAL_TRIALS"),
    "job_id": "$jid",
    "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "source_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
out = Path("$EXPT/manifests")
(out / "LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PYREGISTER

echo "submitted $jid task=$TASK_ID rep=$REPLICATE trials=$TOTAL_TRIALS"
