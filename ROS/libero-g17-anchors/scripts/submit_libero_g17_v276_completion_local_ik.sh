#!/usr/bin/env bash
# v276: v274 FG+RR law with hard-local, completion-seeking tip-XY execution.
set -euo pipefail

PHASE=${PHASE:-task5}
case "$PHASE" in task5|anchor) ;; *) echo "PHASE must be task5 or anchor" >&2; exit 2 ;; esac

G17=/home/qshou/capx_g17_observation_integrity
PROJECT_DIR=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT_DIR/.venv-libero/bin/python"
SBATCH_SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT="$PROJECT_DIR/outputs/libero_g17_fair_fg_v276_completion_local_ik_20260819"
ANCHOR_MANIFEST="$PROJECT_DIR/outputs/libero_g17_fair_fg_v275_trusted_tipxy_20260818/manifests/SOURCE_SHA256.json"
PREFIX0="$PROJECT_DIR/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json"
PREFIX5="$PROJECT_DIR/outputs/libero_g17_spatial_prefix_replay_20260818/task_5/code_prefix_replay_v1.json"

[[ -x "$PYTHON" && -f "$SBATCH_SCRIPT" && -f "$ANCHOR_MANIFEST" ]]
[[ -f "$PREFIX0" && -f "$PREFIX5" ]]
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/arms/v276" "$EXPT/evidence"

if [[ "${FORCE_PREPARE:-0}" == "1" || ! -s "$EXPT/evidence/unit_tests.txt" \
      || "$G17/tests/test_tip_xy_only.py" -nt "$EXPT/evidence/unit_tests.txt" \
      || "$G17/tests/test_libero_g17_nearmiss_osc.py" -nt "$EXPT/evidence/unit_tests.txt" \
      || "$G17/capx/verification/libero_g17_nearmiss_osc.py" -nt "$EXPT/evidence/unit_tests.txt" ]]; then
  PYTHONPATH="$G17" "$PYTHON" -m pytest -q \
    "$G17/tests/test_tip_xy_only.py" \
    "$G17/tests/test_libero_g17_nearmiss_osc.py" \
    "$G17/tests/test_libero_g17_release_ready.py" \
    >"$EXPT/evidence/unit_tests.txt"
fi

if [[ "${FORCE_PREPARE:-0}" == "1" || ! -s "$EXPT/evidence/tip_xy_local_solver_audit.json" \
      || "$G17/scripts/audit_tip_xy_solver_seed.py" -nt "$EXPT/evidence/tip_xy_local_solver_audit.json" \
      || "$G17/capx/integrations/motion/pyroki_snippets/_solve_ik_vel_cost.py" -nt "$EXPT/evidence/tip_xy_local_solver_audit.json" ]]; then
  PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/audit_tip_xy_solver_seed.py" \
    --output "$EXPT/evidence/tip_xy_local_solver_audit.json" >/dev/null
fi

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time
from pathlib import Path

anchor_path = Path("$ANCHOR_MANIFEST")
anchor = json.loads(anchor_path.read_text())
allowed = {
    Path("$G17/capx/integrations/franka/tip_xy_only.py"),
    Path("$G17/capx/integrations/motion/pyroki_snippets/__init__.py"),
    Path("$G17/capx/integrations/motion/pyroki_snippets/_solve_ik_vel_cost.py"),
    Path("$G17/capx/verification/libero_g17_nearmiss_osc.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"),
    Path("$G17/tests/test_libero_g17_nearmiss_osc.py"),
    Path("$G17/tests/test_tip_xy_only.py"),
    Path("$G17/scripts/audit_tip_xy_solver_seed.py"),
    Path("$G17/scripts/submit_libero_g17_v276_completion_local_ik.sh"),
    Path("$SBATCH_SCRIPT"),
}
equal = {}
for raw_path, expected in anchor["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if path in allowed:
        continue
    if actual != expected:
        raise SystemExit(f"v276 unauthorized drift from v275: {path} {actual} != {expected}")
    equal[raw_path] = actual

files = dict(equal)
for path in sorted(allowed):
    if not path.is_file():
        raise SystemExit(f"v276 required source missing: {path}")
    files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

audit_path = Path("$EXPT/evidence/tip_xy_local_solver_audit.json")
audit = json.loads(audit_path.read_text())
if audit.get("schema") != "capx_tip_xy_local_solver_audit_v2":
    raise SystemExit("v276 solver audit schema mismatch")
if not audit.get("rows") or not all(row.get("hard_local_contract_ok") for row in audit["rows"]):
    raise SystemExit("v276 hard-local solver audit failed")
tests_path = Path("$EXPT/evidence/unit_tests.txt")
data = {
    "schema": "capx_libero_g17_v276_completion_local_ik_source_v1",
    "anchor": "v275 trusted tipxy task5 diagnostic",
    "anchor_manifest": str(anchor_path),
    "anchor_manifest_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
    "unchanged_anchor_files": len(equal),
    "allowed_runtime_diffs": sorted(str(path) for path in allowed),
    "solver_audit": str(audit_path),
    "solver_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    "unit_tests": str(tests_path),
    "unit_tests_sha256": hashlib.sha256(tests_path.read_bytes()).hexdigest(),
    "files": files,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
out = Path("$EXPT/manifests/SOURCE_SHA256.json")
out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(json.dumps(data, indent=2))
PYMANIFEST

env \
  -u CAPX_LIBERO_G17_GRASP_REPLAY \
  -u CAPX_LIBERO_G17_GRASP_REPLAY_STRICT \
  -u CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW \
  -u CAPX_LIBERO_G17_V270_LIVE_ISOLATION \
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
  PYTHONPATH="$G17" "$PYTHON" - <<'PYPREFLIGHT'
from capx.integrations.franka import tip_xy_only
from capx.integrations.motion import pyroki_snippets
from capx.verification import libero_g17_nearmiss_osc as osc
from capx.verification import libero_g17_takeover as takeover
schema = takeover.schema_version()
contract = tip_xy_only.trusted_tip_xy_contract()
assert "v2.76" in schema and "completion-local-ik" in schema, schema
assert osc.nearmiss_osc_v276_completion_ik_enabled()
assert hasattr(pyroki_snippets, "solve_ik_local")
assert contract["schema"] == "capx_completion_tip_xy_contract_v2"
assert contract["max_arm_joint_delta_rad"] == 0.20
assert contract["max_joint_controller_steps"] == 40
print("v276_completion_local_ik_preflight_ok", schema, contract)
PYPREFLIGHT

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "v276 source manifest, tests, solver evidence, and completion contract are ready"
  exit 0
fi

if [[ "$PHASE" == "task5" ]]; then
  TASK_IDS=(5); REPS=(0)
else
  TASK_IDS=(0); REPS=(0 1 2 3 4)
fi

jobs_tmp=$(mktemp)
printf '[]\n' >"$jobs_tmp"
for task_id in "${TASK_IDS[@]}"; do
  for rep in "${REPS[@]}"; do
    jname=$(printf 'libero-g17-v276-completion-t%02d-r%02d' "$task_id" "$rep")
    log_pattern="$EXPT/logs/${jname}-%j.out"
    jid=$(sbatch \
      --export=ALL,ARM=v276,TASK_ID="$task_id",REPLICATE="$rep",TOTAL_TRIALS=2 \
      --job-name="$jname" --output="$log_pattern" --error="$log_pattern" \
      "$SBATCH_SCRIPT" | awk '{print $4}')
    echo "submitted phase=$PHASE task=$task_id rep=$rep job=$jid"
    "$PYTHON" - <<PYJOB
import json
from pathlib import Path
p = Path("$jobs_tmp")
rows = json.loads(p.read_text())
rows.append({"phase": "$PHASE", "task_id": $task_id, "replicate": $rep, "job_id": "$jid"})
p.write_text(json.dumps(rows))
PYJOB
  done
done

"$PYTHON" - <<PYREGISTER
import hashlib, json, time
from pathlib import Path
source = Path("$EXPT/manifests/SOURCE_SHA256.json")
data = {
    "schema": "capx_libero_g17_v276_completion_local_ik_submit_v1",
    "phase": "$PHASE",
    "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "jobs": json.loads(Path("$jobs_tmp").read_text()),
}
out = Path("$EXPT/manifests")
stamp = time.strftime("%Y%m%d_%H%M%S")
(out / f"submit_{stamp}.json").write_text(json.dumps(data, indent=2) + "\n")
(out / "LATEST_SUBMIT_$PHASE.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PYREGISTER
rm -f "$jobs_tmp"
