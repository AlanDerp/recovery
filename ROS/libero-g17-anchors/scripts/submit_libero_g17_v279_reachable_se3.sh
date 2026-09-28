#!/usr/bin/env bash
# v279: frozen v278 FG+RR plus complete-path reachability preflight/recovery.
set -euo pipefail

PHASE=${PHASE:-v272_replay}
case "$PHASE" in v272_replay|task5) ;; *) echo "PHASE must be v272_replay or task5" >&2; exit 2 ;; esac

G17=/home/qshou/capx_g17_observation_integrity
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
SBATCH_SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT="$PROJECT/outputs/libero_g17_fair_fg_v279_reachable_se3_20260819"
V278="$PROJECT/outputs/libero_g17_fair_fg_v278_trusted_se3_20260819"
ANCHOR="$V278/manifests/SOURCE_SHA256.json"
FROZEN="$V278/manifests/FROZEN_ANCHOR.json"

[[ -x "$PYTHON" && -f "$SBATCH_SCRIPT" && -f "$ANCHOR" && -f "$FROZEN" ]]
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/arms/v279" "$EXPT/evidence"

PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_reachable_se3.py" \
  "$G17/tests/test_trusted_se3.py" \
  "$G17/tests/test_tip_xy_only.py" \
  "$G17/tests/test_libero_g17_nearmiss_osc.py" \
  "$G17/tests/test_libero_g17_release_ready.py" \
  >"$EXPT/evidence/unit_tests.txt"

PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/audit_v279_reachable_se3.py" \
  --output "$EXPT/evidence/v272_trial2_reachable_se3.json" >/dev/null

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time, zipfile
from pathlib import Path

anchor_path = Path("$ANCHOR")
frozen_path = Path("$FROZEN")
anchor = json.loads(anchor_path.read_text())
frozen = json.loads(frozen_path.read_text())
if frozen.get("schema") != "capx_libero_g17_v278_frozen_anchor_v1":
    raise SystemExit("v279 requires the sealed v278 anchor")

changed = {
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/capx/integrations/franka/trusted_se3.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"),
    Path("$G17/tests/test_trusted_se3.py"),
    Path("$SBATCH_SCRIPT"),
}
files = {}
for raw_path, expected in anchor["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if path not in changed and actual != expected:
        raise SystemExit(f"v279 unauthorized drift from v278: {path} {actual} != {expected}")
    files[str(path)] = actual

added = {
    Path("$G17/capx/integrations/franka/reachable_se3.py"),
    Path("$G17/scripts/audit_v279_reachable_se3.py"),
    Path("$G17/tests/test_reachable_se3.py"),
    Path("$G17/scripts/submit_libero_g17_v279_reachable_se3.sh"),
}
for path in changed | added:
    if not path.is_file():
        raise SystemExit(f"v279 required source missing: {path}")
    files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

audit_path = Path("$EXPT/evidence/v272_trial2_reachable_se3.json")
audit = json.loads(audit_path.read_text())
if audit.get("schema") != "capx_v279_exact_reachable_se3_audit_v1":
    raise SystemExit("v279 exact replay schema mismatch")
if audit.get("straight_preflight", {}).get("ok"):
    raise SystemExit("v279 audit did not reproduce the straight-path dead end")
if audit["straight_preflight"].get("executed_commands") != 0:
    raise SystemExit("v279 preflight executed a rejected path")
if not audit.get("reachable", {}).get("ok"):
    raise SystemExit("v279 neutral recovery failed")
if not str(audit["reachable"].get("selected_route", "")).startswith("neutral_"):
    raise SystemExit("v279 audit did not select a neutral route")
if audit["reachable"]["max_executed_arm_joint_delta_rad"] > 0.2002:
    raise SystemExit("v279 executed joint trust contract failed")

tests_path = Path("$EXPT/evidence/unit_tests.txt")
data = {
    "schema": "capx_libero_g17_v279_reachable_se3_source_v1",
    "anchor": "frozen v278 trusted-SE3 FG+RR",
    "anchor_manifest": str(anchor_path),
    "anchor_manifest_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
    "frozen_anchor": str(frozen_path),
    "frozen_anchor_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
    "only_runtime_changes": [
        "non-mutating full-path local-IK preflight for large pose rotations",
        "preflighted two-leg route through a neutral Panda configuration",
        "controller-floor classification within the declared terminal tolerance",
        "v279 schema/runner plumbing",
    ],
    "preserved": [
        "v278 live GraspNet target and quaternion",
        "v278 tip-to-hand and z_approach semantics",
        "v278 per-waypoint joint locality and FK validation",
        "v276 FG predicates and ReleaseReady law",
        "no grasp replay, canonical grasp, or continuous privileged truth",
    ],
    "files": dict(sorted(files.items())),
    "exact_replay": str(audit_path),
    "exact_replay_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    "unit_tests": str(tests_path),
    "unit_tests_sha256": hashlib.sha256(tests_path.read_bytes()).hexdigest(),
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
out = Path("$EXPT/manifests/SOURCE_SHA256.json")
out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
snapshot = Path("$EXPT/manifests/SOURCE_SNAPSHOT.zip")
with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for raw_path in files:
        path = Path(raw_path)
        archive.write(path, arcname=str(path).lstrip("/"))
print(json.dumps(data, indent=2))
PYMANIFEST

env \
  -u CAPX_LIBERO_G17_GRASP_REPLAY \
  -u CAPX_LIBERO_G17_V277_CANONICAL_GRASP \
  -u CAPX_LIBERO_G17_V278_TRUSTED_SE3 \
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
  CAPX_LIBERO_G17_V279_REACHABLE_SE3=1 \
  PYTHONPATH="$G17" "$PYTHON" - <<'PYPREFLIGHT'
from capx.integrations.franka.reachable_se3 import reachable_se3_contract, reachable_se3_enabled
from capx.verification.libero_g17_takeover import schema_version
s = schema_version()
assert "v2.79" in s and "trusted-se3-fullpath-neutral-recovery" in s, s
assert reachable_se3_enabled()
assert reachable_se3_contract()["schema"] == "capx_libero_reachable_se3_contract_v1"
print("v279_reachable_se3_preflight_ok", s, reachable_se3_contract())
PYPREFLIGHT

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "v279 source, tests, and exact v272 trial-2 reachability evidence ready"
  exit 0
fi

if [[ "$PHASE" == "v272_replay" ]]; then
  TASK_ID=0; REPLICATE=1
else
  TASK_ID=5; REPLICATE=0
fi
jname=$(printf 'libero-g17-v279-reach-t%02d-r%02d' "$TASK_ID" "$REPLICATE")
log="$EXPT/logs/${jname}-%j.out"
jid=$(sbatch \
  --export=ALL,ARM=v279,TASK_ID="$TASK_ID",REPLICATE="$REPLICATE",TOTAL_TRIALS=2 \
  --job-name="$jname" --output="$log" --error="$log" \
  "$SBATCH_SCRIPT" | awk '{print $4}')

"$PYTHON" - <<PYREGISTER
import hashlib, json, time
from pathlib import Path
source = Path("$EXPT/manifests/SOURCE_SHA256.json")
snapshot = Path("$EXPT/manifests/SOURCE_SNAPSHOT.zip")
data = {
    "schema": "capx_libero_g17_v279_reachable_se3_submit_v1",
    "phase": "$PHASE",
    "task_id": $TASK_ID,
    "replicate": $REPLICATE,
    "total_trials": 2,
    "job_id": "$jid",
    "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "source_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
out = Path("$EXPT/manifests")
(out / f"SUBMIT_{data['phase']}.json").write_text(json.dumps(data, indent=2) + "\n")
(out / "LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PYREGISTER

echo "submitted $jid ($PHASE task=$TASK_ID rep=$REPLICATE)"
