#!/usr/bin/env bash
# v274: v272 FG+RR with only inscribed axis completion and acceptance-gated t.
set -euo pipefail

PHASE=${PHASE:-task5}
case "$PHASE" in task5|anchor) ;; *) echo "PHASE must be task5 or anchor" >&2; exit 2 ;; esac

G17=/home/qshou/capx_g17_observation_integrity
PROJECT_DIR=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT_DIR/.venv-libero/bin/python"
SBATCH_SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT="$PROJECT_DIR/outputs/libero_g17_fair_fg_v274_mech_contract_20260818"
ANCHOR_MANIFEST="$PROJECT_DIR/outputs/libero_g17_fair_fg_v272_clean_live_20260818/manifests/SOURCE_SHA256.json"
PREFIX0="$PROJECT_DIR/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json"
PREFIX5="$PROJECT_DIR/outputs/libero_g17_spatial_prefix_replay_20260818/task_5/code_prefix_replay_v1.json"

[[ -x "$PYTHON" && -f "$SBATCH_SCRIPT" && -f "$ANCHOR_MANIFEST" ]]
[[ -f "$PREFIX0" && -f "$PREFIX5" ]]
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/arms/v274"

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time
from pathlib import Path

anchor_path = Path("$ANCHOR_MANIFEST")
anchor = json.loads(anchor_path.read_text())
allowed = {
    Path("$G17/capx/verification/libero_g17_nearmiss_osc.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"),
    Path("$SBATCH_SCRIPT"),
}
equal = {}
for raw_path, expected in anchor["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if path in allowed:
        continue
    if actual != expected:
        raise SystemExit(
            f"v274 unauthorized drift from v272 anchor: {path} {actual} != {expected}"
        )
    equal[raw_path] = actual

files = dict(equal)
for path in sorted(allowed):
    files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
test = Path("$G17/tests/test_libero_g17_nearmiss_osc.py")
files[str(test)] = hashlib.sha256(test.read_bytes()).hexdigest()
data = {
    "schema": "capx_libero_g17_v274_mech_contract_source_v1",
    "anchor": "v272-clean-live 9/10",
    "anchor_manifest": str(anchor_path),
    "anchor_manifest_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
    "unchanged_anchor_files": len(equal),
    "allowed_runtime_diffs": sorted(str(path) for path in allowed),
    "contract": {
        "schedule_m": [0.15, 0.12, 0.09, 0.06, 0.03],
        "axis_completion_band_m": 0.03 / (2 ** 0.5),
        "release_disk_m": 0.03,
        "max_attempts_per_t_axis": 2,
        "advance_rule": "accepted primitive or freshly observed axis predicate",
        "z_abort_m": 0.01,
    },
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
  PYTHONPATH="$G17" "$PYTHON" - <<'PYPREFLIGHT'
from capx.verification import libero_g17_nearmiss_osc as osc
from capx.verification import libero_g17_takeover as takeover
schema = takeover.schema_version()
assert "v2.74" in schema and "fair-live-inscribed-acceptance-gated" in schema, schema
assert osc.nearmiss_osc_v274_mech_contract_enabled()
assert osc.nearmiss_osc_v269_mech_enabled()
assert osc.v269_symbol_source() == "fg"
assert abs(osc.v274_axis_skip_band_m() - 0.03 / (2 ** 0.5)) < 1e-12
assert [osc.axis_magnitude(0.15, 0.03, t) for t in range(5)] == [0.15, 0.12, 0.09, 0.06, 0.03]
print("v274_mechanical_contract_preflight_ok", schema)
PYPREFLIGHT

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "v274 source manifest and mechanical contract are ready"
  exit 0
fi

if [[ "$PHASE" == "task5" ]]; then
  TASK_IDS=(5)
  REPS=(0)
else
  TASK_IDS=(0)
  REPS=(0 1 2 3 4)
fi

jobs_tmp=$(mktemp)
printf '[]\n' >"$jobs_tmp"
for task_id in "${TASK_IDS[@]}"; do
  for rep in "${REPS[@]}"; do
    jname=$(printf 'libero-g17-v274-mech-t%02d-r%02d' "$task_id" "$rep")
    log_pattern="$EXPT/logs/${jname}-%j.out"
    jid=$(sbatch \
      --export=ALL,ARM=v274,TASK_ID="$task_id",REPLICATE="$rep",TOTAL_TRIALS=2 \
      --job-name="$jname" \
      --output="$log_pattern" \
      --error="$log_pattern" \
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
    "schema": "capx_libero_g17_v274_mech_contract_submit_v1",
    "phase": "$PHASE",
    "controller": "v272 + inscribed axis completion + acceptance-gated t-axis only",
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
