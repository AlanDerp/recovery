#!/usr/bin/env bash
# v282: measured-seeded, deterministic multistart IK with native-once execution.
set -euo pipefail

G17=/home/qshou/capx_g17_observation_integrity
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
SBATCH_SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT="$PROJECT/outputs/libero_g17_fair_fg_v282_measured_seed_20260819"
ANCHOR="$PROJECT/outputs/libero_g17_fair_fg_v281_joint_join_20260819/manifests/FROZEN_K5_ANCHOR.json"
TASK_ID=${TASK_ID:-0}
REPLICATE=${REPLICATE:-1}
TOTAL_TRIALS=${TOTAL_TRIALS:-2}

[[ -x "$PYTHON" && -f "$SBATCH_SCRIPT" && -f "$ANCHOR" ]]
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/arms/v282" "$EXPT/evidence"

PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_measured_seed_ik.py" \
  "$G17/tests/test_planned_se3.py" \
  "$G17/tests/test_reachable_se3.py" \
  "$G17/tests/test_trusted_se3.py" \
  "$G17/tests/test_tip_xy_only.py" \
  "$G17/tests/test_libero_g17_nearmiss_osc.py" \
  "$G17/tests/test_libero_g17_release_ready.py" \
  >"$EXPT/evidence/unit_tests.txt"

PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/audit_v282_measured_seed_ik.py" \
  --output "$EXPT/evidence/v272_trial2_measured_seed_ik.json" >/dev/null

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time, zipfile
from pathlib import Path

anchor = Path("$ANCHOR")
audit_path = Path("$EXPT/evidence/v272_trial2_measured_seed_ik.json")
tests_path = Path("$EXPT/evidence/unit_tests.txt")
audit = json.loads(audit_path.read_text())
if audit.get("schema") != "capx_v282_measured_seed_ik_audit_v1":
    raise SystemExit("v282 audit schema mismatch")
native = audit.get("native_execution", {})
if not native.get("accepted") or native.get("total_blocking_commands") != 4:
    raise SystemExit("v282 native-once audit failed")
if any(row.get("execution_commands") != 1 for row in native.get("stages", [])):
    raise SystemExit("v282 added execution waypoints")
contract = audit.get("contract", {})
for key in ("cartesian_waypoints", "joint_delta_gate", "terminal_residual_gate", "neutral_route", "rollback"):
    if contract.get(key) != 0:
        raise SystemExit(f"v282 forbidden defensive behavior: {key}={contract.get(key)}")

paths = {
    Path("$G17/capx/integrations/motion/pyroki_snippets/_solve_ik.py"),
    Path("$G17/capx/integrations/motion/pyroki_snippets/__init__.py"),
    Path("$G17/capx/integrations/franka/measured_seed_ik.py"),
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"),
    Path("$SBATCH_SCRIPT"),
    Path("$G17/scripts/audit_v282_measured_seed_ik.py"),
    Path("$G17/tests/test_measured_seed_ik.py"),
    Path("$G17/scripts/submit_libero_g17_v282_measured_seed.sh"),
}
files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
data = {
    "schema": "capx_libero_g17_v282_measured_seed_source_v1",
    "anchor": "frozen v281 FG+RR 9/10; v281 execution backend removed",
    "anchor_manifest": str(anchor),
    "anchor_manifest_sha256": hashlib.sha256(anchor.read_bytes()).hexdigest(),
    "root_cause": (
        "legacy first IK was unseeded and later IK continuity used requested "
        "cfg rather than measured controller state"
    ),
    "runtime_change": (
        "measured optimizer initialization/cache; measured-first deterministic "
        "multistart only on solver residual; one native blocking command per stage"
    ),
    "forbidden": [
        "cartesian waypoint execution", "joint delta gate", "terminal residual gate",
        "executed neutral route", "rollback",
    ],
    "files": dict(sorted(files.items())),
    "counterfactual_audit": str(audit_path),
    "counterfactual_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
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
  -u CAPX_LIBERO_G17_V281_JOINT_JOIN \
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
  CAPX_LIBERO_G17_V282_MEASURED_SEED_IK=1 \
  PYTHONPATH="$G17" "$PYTHON" - <<'PYPREFLIGHT'
from capx.integrations.franka.measured_seed_ik import measured_seed_ik_contract, measured_seed_ik_enabled
from capx.verification.libero_g17_takeover import schema_version
s = schema_version()
contract = measured_seed_ik_contract()
assert "v2.82" in s and "measured-seed-native-once" in s, s
assert measured_seed_ik_enabled()
assert contract["native_blocking_commands_per_stage"] == 1
assert contract["cartesian_waypoints"] == 0
assert contract["joint_delta_gate"] == 0
assert contract["terminal_residual_gate"] == 0
print("v282_measured_seed_preflight_ok", s, contract)
PYPREFLIGHT

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "v282 evidence and source manifest ready"
  exit 0
fi

jname=$(printf 'libero-g17-v282-measured-t%02d-r%02d' "$TASK_ID" "$REPLICATE")
log="$EXPT/logs/${jname}-%j.out"
jid=$(sbatch \
  --export=ALL,ARM=v282,TASK_ID="$TASK_ID",REPLICATE="$REPLICATE",TOTAL_TRIALS="$TOTAL_TRIALS" \
  --job-name="$jname" --output="$log" --error="$log" \
  "$SBATCH_SCRIPT" | awk '{print $4}')

"$PYTHON" - <<PYREGISTER
import hashlib, json, time
from pathlib import Path
source = Path("$EXPT/manifests/SOURCE_SHA256.json")
snapshot = Path("$EXPT/manifests/SOURCE_SNAPSHOT.zip")
data = {
    "schema": "capx_libero_g17_v282_measured_seed_submit_v1",
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
