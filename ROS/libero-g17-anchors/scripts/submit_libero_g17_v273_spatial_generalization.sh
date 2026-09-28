#!/usr/bin/env bash
# v273: outcome-blind LIBERO-Spatial task1-9 breadth screen.
# Controller is byte-identical to the v272 FG+RR anchor; only task/prefix plumbing changes.
set -euo pipefail

G17=/home/qshou/capx_g17_observation_integrity
PROJECT_DIR=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT_DIR/.venv-libero/bin/python"
SBATCH_SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
BUILDER="$G17/scripts/build_libero_g17_spatial_prefixes.py"
EXPT="$PROJECT_DIR/outputs/libero_g17_fair_fg_v273_spatial_generalization_20260818"
PREFIX_ROOT="$PROJECT_DIR/outputs/libero_g17_spatial_prefix_replay_20260818"
ANCHOR_MANIFEST="$PROJECT_DIR/outputs/libero_g17_fair_fg_v272_clean_live_20260818/manifests/SOURCE_SHA256.json"
TASK_IDS=(1 2 3 4 5 6 7 8 9)
K=${K:-1}

[[ "$K" == "1" ]] || {
  echo "v273 breadth screen is pre-registered for K=1, got K=$K" >&2
  exit 2
}
[[ -x "$PYTHON" && -f "$SBATCH_SCRIPT" && -f "$BUILDER" && -f "$ANCHOR_MANIFEST" ]]
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/arms/v273"

PYTHONPATH="$G17" "$PYTHON" "$BUILDER" --out-root "$PREFIX_ROOT"

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time
from pathlib import Path

anchor_path = Path("$ANCHOR_MANIFEST")
anchor = json.loads(anchor_path.read_text())
runner = Path("$SBATCH_SCRIPT")
builder = Path("$BUILDER")
prefix_index = Path("$PREFIX_ROOT/INDEX.json")

# The shared sbatch runner is generalized in v273.  Every executable controller
# and integration source recorded by v272 must remain byte-identical.
excluded = {str(runner)}
equal = {}
for raw_path, expected in anchor["files"].items():
    if raw_path in excluded:
        continue
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(
            f"v273 controller drift from v272 anchor: {path} {actual} != {expected}"
        )
    equal[raw_path] = actual

files = dict(equal)
for path in (runner, builder):
    files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

data = {
    "schema": "capx_libero_g17_v273_spatial_generalization_source_v1",
    "anchor": "v272-clean-live 9/10",
    "anchor_manifest": str(anchor_path),
    "anchor_manifest_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
    "controller_files_equal_to_anchor": len(equal),
    "runner_only_generalization": True,
    "suite": "libero_spatial",
    "held_out_task_ids": list(range(1, 10)),
    "trials": [1, 2],
    "K": 1,
    "selection": "all held-out spatial tasks; native sandboxrc_0 CB0 consensus; outcome-blind",
    "prefix_index": str(prefix_index),
    "prefix_index_sha256": hashlib.sha256(prefix_index.read_bytes()).hexdigest(),
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
  PYTHONPATH="$G17" "$PYTHON" - <<'PYPREFLIGHT'
import os
from capx.verification import libero_g17_live_isolation as isolation
from capx.verification import libero_g17_nearmiss_osc as osc
from capx.verification import libero_g17_takeover as takeover

schema = takeover.schema_version()
assert "v2.72" in schema and "fair-live-no-replay" in schema, schema
assert osc.nearmiss_osc_v269_mech_enabled()
assert osc.v269_symbol_source() == "fg"
assert isolation.enabled()
for key in (
    "CAPX_LIBERO_G17_GRASP_REPLAY",
    "CAPX_LIBERO_G17_GRASP_REPLAY_STRICT",
    "CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW",
):
    assert not os.environ.get(key), (key, os.environ.get(key))
print("v273_frozen_controller_preflight_ok", schema)
PYPREFLIGHT

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "v273 assets, fingerprints, and frozen-controller preflight are ready"
  exit 0
fi

jobs_tmp=$(mktemp)
printf '[]\n' >"$jobs_tmp"
for task_id in "${TASK_IDS[@]}"; do
  jname=$(printf 'libero-g17-v273-spatial-t%02d-r00' "$task_id")
  log_pattern="$EXPT/logs/${jname}-%j.out"
  jid=$(sbatch \
    --export=ALL,ARM=v273,TASK_ID="$task_id",REPLICATE=0,TOTAL_TRIALS=2 \
    --job-name="$jname" \
    --output="$log_pattern" \
    --error="$log_pattern" \
    "$SBATCH_SCRIPT" | awk '{print $4}')
  echo "submitted task=$task_id rep=0 job=$jid"
  "$PYTHON" - <<PYJOB
import json
from pathlib import Path
p = Path("$jobs_tmp")
rows = json.loads(p.read_text())
rows.append({"task_id": $task_id, "replicate": 0, "job_id": "$jid"})
p.write_text(json.dumps(rows))
PYJOB
done

"$PYTHON" - <<PYREGISTER
import hashlib, json, time
from pathlib import Path
source = Path("$EXPT/manifests/SOURCE_SHA256.json")
prefix_index = Path("$PREFIX_ROOT/INDEX.json")
data = {
    "schema": "capx_libero_g17_v273_spatial_generalization_submit_v1",
    "mode": "v273-spatial-held-out-breadth",
    "suite": "libero_spatial",
    "anchor_task_id": 0,
    "held_out_task_ids": list(range(1, 10)),
    "trials": [1, 2],
    "K": 1,
    "controller": "exact v272 FG+RR/live-isolation; no parameter changes",
    "qwen_policy": "live after task-local frozen CB0",
    "grasp_source": "live sample_grasp_pose",
    "selection": "complete held-out task axis; no failure/success selection",
    "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "prefix_index_sha256": hashlib.sha256(prefix_index.read_bytes()).hexdigest(),
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "jobs": json.loads(Path("$jobs_tmp").read_text()),
}
out = Path("$EXPT/manifests")
stamp = time.strftime("%Y%m%d_%H%M%S")
(out / f"submit_{stamp}.json").write_text(json.dumps(data, indent=2) + "\n")
(out / "LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PYREGISTER
rm -f "$jobs_tmp"
