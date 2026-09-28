#!/usr/bin/env bash
# v278: exact v276 FG+RR anchor with trusted Cartesian SE(3) goto execution.
set -euo pipefail

PHASE=${PHASE:-v272_replay}
case "$PHASE" in v272_replay|task5) ;; *) echo "PHASE must be v272_replay or task5" >&2; exit 2 ;; esac

G17=/home/qshou/capx_g17_observation_integrity
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
SBATCH_SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT="$PROJECT/outputs/libero_g17_fair_fg_v278_trusted_se3_20260819"
ANCHOR="$PROJECT/outputs/libero_g17_fair_fg_v276_completion_local_ik_20260819/manifests/SOURCE_SHA256.json"

[[ -x "$PYTHON" && -f "$SBATCH_SCRIPT" && -f "$ANCHOR" ]]
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/arms/v278" "$EXPT/evidence"

if [[ ! -s "$EXPT/evidence/unit_tests.txt" || \
      "$G17/tests/test_trusted_se3.py" -nt "$EXPT/evidence/unit_tests.txt" ]]; then
  PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
    "$G17/tests/test_trusted_se3.py" \
    "$G17/tests/test_tip_xy_only.py" \
    "$G17/tests/test_libero_g17_nearmiss_osc.py" \
    "$G17/tests/test_libero_g17_release_ready.py" \
    >"$EXPT/evidence/unit_tests.txt"
fi

if [[ ! -s "$EXPT/evidence/v272_first_goto_trusted_se3.json" || \
      "$G17/scripts/audit_v272_trusted_se3.py" -nt "$EXPT/evidence/v272_first_goto_trusted_se3.json" || \
      "$G17/capx/integrations/franka/trusted_se3.py" -nt "$EXPT/evidence/v272_first_goto_trusted_se3.json" ]]; then
  PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/audit_v272_trusted_se3.py" \
    --output "$EXPT/evidence/v272_first_goto_trusted_se3.json" >/dev/null
fi

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time, zipfile
from pathlib import Path

anchor_path = Path("$ANCHOR")
anchor = json.loads(anchor_path.read_text())
allowed_anchor_diffs = {
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"),
    Path("$SBATCH_SCRIPT"),
}
files = {}
for raw_path, expected in anchor["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if path not in allowed_anchor_diffs and actual != expected:
        raise SystemExit(f"v278 unauthorized drift from v276: {path} {actual} != {expected}")
    files[str(path)] = actual

added = {
    Path("$G17/capx/integrations/franka/trusted_se3.py"),
    Path("$G17/scripts/audit_v272_trusted_se3.py"),
    Path("$G17/tests/test_trusted_se3.py"),
    Path("$G17/scripts/submit_libero_g17_v278_trusted_se3.sh"),
}
for path in allowed_anchor_diffs | added:
    if not path.is_file():
        raise SystemExit(f"v278 required source missing: {path}")
    files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

audit_path = Path("$EXPT/evidence/v272_first_goto_trusted_se3.json")
audit = json.loads(audit_path.read_text())
if audit.get("schema") != "capx_v272_first_goto_trusted_se3_audit_v1":
    raise SystemExit("v278 exact replay schema mismatch")
if not audit.get("trusted", {}).get("ok"):
    raise SystemExit("v278 exact replay contract failed")
if audit["legacy"]["max_arm_joint_delta_rad"] < 2.0:
    raise SystemExit("v278 audit did not reproduce legacy branch jump")
if audit["trusted"]["max_arm_joint_delta_rad"] > 0.2002:
    raise SystemExit("v278 local joint contract failed")

tests_path = Path("$EXPT/evidence/unit_tests.txt")
data = {
    "schema": "capx_libero_g17_v278_trusted_se3_source_v1",
    "anchor": "v276 completion-local-IK FG+RR",
    "anchor_manifest": str(anchor_path),
    "anchor_manifest_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
    "only_runtime_changes": [
        "goto_pose backend under CAPX_LIBERO_G17_V278_TRUSTED_SE3",
        "v278 schema/runner plumbing",
    ],
    "preserved": [
        "live GraspNet target and quaternion",
        "legacy tip-to-hand target transform",
        "legacy local-axis z_approach target",
        "v276 FG predicates and ReleaseReady law",
        "no grasp replay and no continuous privileged truth",
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
  -u CAPX_LIBERO_G17_V275_TRUSTED_TIPXY \
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
  CAPX_LIBERO_G17_V278_TRUSTED_SE3=1 \
  PYTHONPATH="$G17" "$PYTHON" - <<'PYPREFLIGHT'
from capx.integrations.franka.trusted_se3 import trusted_se3_contract, trusted_se3_enabled
from capx.verification.libero_g17_takeover import schema_version
s = schema_version()
assert "v2.78" in s and "trusted-se3" in s, s
assert trusted_se3_enabled()
assert trusted_se3_contract()["schema"] == "capx_libero_trusted_se3_contract_v1"
print("v278_trusted_se3_preflight_ok", s, trusted_se3_contract())
PYPREFLIGHT

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "v278 source, tests, and exact v272 replay evidence ready"
  exit 0
fi

if [[ "$PHASE" == "v272_replay" ]]; then
  TASK_ID=0; REPLICATE=1
else
  TASK_ID=5; REPLICATE=0
fi
jname=$(printf 'libero-g17-v278-se3-t%02d-r%02d' "$TASK_ID" "$REPLICATE")
log="$EXPT/logs/${jname}-%j.out"
jid=$(sbatch \
  --export=ALL,ARM=v278,TASK_ID="$TASK_ID",REPLICATE="$REPLICATE",TOTAL_TRIALS=2 \
  --job-name="$jname" --output="$log" --error="$log" \
  "$SBATCH_SCRIPT" | awk '{print $4}')

"$PYTHON" - <<PYREGISTER
import hashlib, json, time
from pathlib import Path
source = Path("$EXPT/manifests/SOURCE_SHA256.json")
snapshot = Path("$EXPT/manifests/SOURCE_SNAPSHOT.zip")
data = {
    "schema": "capx_libero_g17_v278_trusted_se3_submit_v1",
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
(out / "LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PYREGISTER
