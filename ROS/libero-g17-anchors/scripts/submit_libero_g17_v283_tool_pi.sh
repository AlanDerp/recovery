#!/usr/bin/env bash
# v283: nearest IK over the exact parallel-jaw q / q*Rz(pi) equivalence class.
set -euo pipefail

G17=/home/qshou/capx_g17_observation_integrity
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
RUNNER="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT="$PROJECT/outputs/libero_g17_fair_fg_v283_tool_pi_20260819"
ANCHOR="$PROJECT/outputs/libero_g17_fair_fg_v281_joint_join_20260819/manifests/FROZEN_K5_ANCHOR.json"
TASK_ID=${TASK_ID:-0}
REPLICATE=${REPLICATE:-1}
TOTAL_TRIALS=${TOTAL_TRIALS:-2}

[[ -x "$PYTHON" && -f "$RUNNER" && -f "$ANCHOR" ]]
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/arms/v283" "$EXPT/evidence"

PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
  "$G17/tests/test_measured_seed_ik.py" \
  "$G17/tests/test_planned_se3.py" \
  "$G17/tests/test_reachable_se3.py" \
  "$G17/tests/test_trusted_se3.py" \
  "$G17/tests/test_tip_xy_only.py" \
  "$G17/tests/test_libero_g17_nearmiss_osc.py" \
  "$G17/tests/test_libero_g17_release_ready.py" \
  >"$EXPT/evidence/unit_tests.txt"

CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY=1 PYTHONPATH="$G17" "$PYTHON" \
  "$G17/scripts/audit_v282_measured_seed_ik.py" \
  --output "$EXPT/evidence/v272_trial2_tool_pi_ik.json" >/dev/null

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time, zipfile
from pathlib import Path
anchor = Path("$ANCHOR")
audit_path = Path("$EXPT/evidence/v272_trial2_tool_pi_ik.json")
tests_path = Path("$EXPT/evidence/unit_tests.txt")
audit = json.loads(audit_path.read_text())
native = audit.get("native_execution", {})
contract = audit.get("contract", {})
if not native.get("accepted") or native.get("total_blocking_commands") != 4:
    raise SystemExit("v283 native-once audit failed")
if contract.get("tool_z_pi_symmetry") != 1:
    raise SystemExit("v283 tool-pi symmetry audit failed")
place = native.get("stages", [])[-1]
if place.get("candidate_arm_delta_rad", 99) >= 1.25:
    raise SystemExit("v283 did not remove the canonical placement branch jump")
paths = {
    Path("$G17/capx/integrations/motion/pyroki_snippets/_solve_ik.py"),
    Path("$G17/capx/integrations/motion/pyroki_snippets/__init__.py"),
    Path("$G17/capx/integrations/franka/measured_seed_ik.py"),
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"), Path("$RUNNER"),
    Path("$G17/scripts/audit_v282_measured_seed_ik.py"),
    Path("$G17/tests/test_measured_seed_ik.py"),
    Path("$G17/scripts/submit_libero_g17_v283_tool_pi.sh"),
}
files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
data = {
    "schema": "capx_libero_g17_v283_tool_pi_source_v1",
    "anchor": "frozen v281 FG+RR 9/10; no v281 execution backend",
    "anchor_manifest": str(anchor),
    "anchor_manifest_sha256": hashlib.sha256(anchor.read_bytes()).hexdigest(),
    "root_fix": (
        "measured-seeded IK over the physical parallel-jaw q/qRz(pi) equivalence "
        "class, nearest exact solution, one native controller command per stage"
    ),
    "files": dict(sorted(files.items())),
    "audit": str(audit_path),
    "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    "tests": str(tests_path),
    "tests_sha256": hashlib.sha256(tests_path.read_bytes()).hexdigest(),
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
manifest = Path("$EXPT/manifests/SOURCE_SHA256.json")
manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
with zipfile.ZipFile(Path("$EXPT/manifests/SOURCE_SNAPSHOT.zip"), "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in paths: z.write(p, arcname=str(p).lstrip("/"))
print(json.dumps(data, indent=2))
PYMANIFEST

env \
  -u CAPX_LIBERO_G17_GRASP_REPLAY -u CAPX_LIBERO_G17_V277_CANONICAL_GRASP \
  -u CAPX_LIBERO_G17_V278_TRUSTED_SE3 -u CAPX_LIBERO_G17_V279_REACHABLE_SE3 \
  -u CAPX_LIBERO_G17_V280_PLANNED_SE3 -u CAPX_LIBERO_G17_V281_JOINT_JOIN \
  -u CAPX_LIBERO_G17_V282_MEASURED_SEED_IK \
  CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 \
  CAPX_LIBERO_G17_RELEASE_READY=1 CAPX_LIBERO_G17_NEARMISS_OSC=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH=1 \
  CAPX_LIBERO_G17_V269_MECH=1 CAPX_LIBERO_G17_V269_SYMBOL_SOURCE=fg \
  CAPX_LIBERO_G17_V272_FAIR_LIVE=1 CAPX_LIBERO_G17_V274_MECH_CONTRACT=1 \
  CAPX_LIBERO_G17_V276_COMPLETION_IK=1 CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY=1 \
  PYTHONPATH="$G17" "$PYTHON" - <<'PYPREFLIGHT'
from capx.integrations.franka.measured_seed_ik import measured_seed_ik_contract
from capx.verification.libero_g17_takeover import schema_version
s = schema_version(); c = measured_seed_ik_contract()
assert "v2.83" in s and "measured-seed-tool-pi-native-once" in s, s
assert c["tool_z_pi_symmetry"] == 1 and c["native_blocking_commands_per_stage"] == 1
assert c["cartesian_waypoints"] == c["joint_delta_gate"] == c["terminal_residual_gate"] == 0
print("v283_tool_pi_preflight_ok", s, c)
PYPREFLIGHT

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then exit 0; fi
jname=$(printf 'libero-g17-v283-tool-pi-t%02d-r%02d' "$TASK_ID" "$REPLICATE")
log="$EXPT/logs/${jname}-%j.out"
jid=$(sbatch --export=ALL,ARM=v283,TASK_ID="$TASK_ID",REPLICATE="$REPLICATE",TOTAL_TRIALS="$TOTAL_TRIALS" \
  --job-name="$jname" --output="$log" --error="$log" "$RUNNER" | awk '{print $4}')
"$PYTHON" - <<PYREGISTER
import hashlib, json, time
from pathlib import Path
s=Path("$EXPT/manifests/SOURCE_SHA256.json"); z=Path("$EXPT/manifests/SOURCE_SNAPSHOT.zip")
d={"schema":"capx_libero_g17_v283_tool_pi_submit_v1","task_id":int("$TASK_ID"),
"replicate":int("$REPLICATE"),"total_trials":int("$TOTAL_TRIALS"),"job_id":"$jid",
"source_manifest_sha256":hashlib.sha256(s.read_bytes()).hexdigest(),
"source_snapshot_sha256":hashlib.sha256(z.read_bytes()).hexdigest(),
"submitted_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
PYREGISTER
echo "submitted $jid task=$TASK_ID rep=$REPLICATE trials=$TOTAL_TRIALS"
