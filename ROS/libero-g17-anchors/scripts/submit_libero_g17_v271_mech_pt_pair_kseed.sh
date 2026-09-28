#!/usr/bin/env bash
# Paired ablation of v269-mech: ideal FG bands versus PT bands.
set -euo pipefail

K=${K:-5}
G17=/home/qshou/capx_g17_observation_integrity
SBATCH="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_v271_mech_pt_pair_20260818
ANCHOR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v269_mech_20260818
PREFIX=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
GRASP=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_grasp_replay_v256_20260817/grasp_replay_v1.json
PYTHON=/project/peilab/qshou/cap-x-agilex/.venv-libero/bin/python

[[ "$K" == "5" ]] || { echo "v271 PT pair is registered for K=5, got K=$K" >&2; exit 2; }
[[ -f "$SBATCH" && -f "$PREFIX" && -f "$GRASP" ]]
[[ -f "$ANCHOR/manifests/GOLDEN_ANCHOR.json" ]]
[[ -f "$ANCHOR/assets/effective_seed_replay/MANIFEST.json" ]]
mkdir -p "$EXPT/logs" "$EXPT/arms/v271fg" "$EXPT/arms/v271pt" "$EXPT/manifests"

if [[ -f "$EXPT/manifests/LATEST_SUBMIT.json" && "${ALLOW_RESUBMIT:-0}" != "1" ]]; then
  echo "v271 already submitted; set ALLOW_RESUBMIT=1 only for an explicit rerun" >&2
  exit 2
fi

"$PYTHON" - <<PYFREEZE
import hashlib, json, zipfile
from pathlib import Path

paths = [
    Path("$SBATCH"),
    Path("$G17/capx/verification/libero_g17_nearmiss_osc.py"),
    Path("$G17/capx/verification/libero_g17_v271_pt_mech.py"),
    Path("$G17/capx/verification/libero_g17_release_ready.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"),
    Path("$G17/capx/verification/libero_g17_grasp_replay.py"),
    Path("$G17/capx/verification/libero_g17_adapter.py"),
    Path("$G17/capx/verification/libero_g17_band_obligations.py"),
    Path("$G17/capx/verification/libero_g17_code_prefix_replay.py"),
    Path("$G17/capx/verification/libero_g17_fg_projection.py"),
    Path("$G17/capx/verification/libero_g17_public_iff.py"),
    Path("$G17/capx/verification/libero_g17_xy_lock.py"),
    Path("$G17/capx/integrations/franka/tip_xy_only.py"),
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/capx/integrations/vision/molmo.py"),
    Path("$G17/capx/envs/simulators/libero.py"),
    Path("$G17/capx/envs/runner.py"),
    Path("$G17/capx/envs/tasks/base.py"),
    Path("$G17/capx/envs/trial.py"),
    Path("$G17/python_sitecustomize/sitecustomize.py"),
    Path("$G17/scripts/qwen_effective_seed_replay_proxy.py"),
    Path("/home/qshou/capx_causal_ik_replay_81e0acb/experiments/core_fg/runtime/python_sitecustomize/sitecustomize.py"),
    Path("/home/qshou/capx_causal_ik_replay_81e0acb/experiments/core_fg/runtime/qwen_proxy.py"),
]
manifest = {
    "schema": "capx_libero_g17_v271_source_sha256_v1",
    "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
}
out = Path("$EXPT/manifests")
source_manifest = out / "SOURCE_SHA256.json"
source_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
snapshot = out / "SOURCE_SNAPSHOT.zip"
with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, arcname=str(path).lstrip("/"))
print(source_manifest, len(paths))
print(snapshot, hashlib.sha256(snapshot.read_bytes()).hexdigest())
PYFREEZE

for source in fg pt; do
  CAPX_LIBERO_G17_XY_LOCK=1 \
  CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 \
  CAPX_LIBERO_G17_RELEASE_READY=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH=1 \
  CAPX_LIBERO_G17_V269_MECH=1 \
  CAPX_LIBERO_G17_V269_SYMBOL_SOURCE=fg \
  CAPX_LIBERO_G17_V271_PT_PAIR=1 \
  CAPX_LIBERO_G17_V271_SYMBOL_SOURCE="$source" \
  CAPX_LIBERO_G17_GRASP_REPLAY="$GRASP" \
  PYTHONPATH="$G17" "$PYTHON" - <<'PY'
from capx.verification import libero_g17_takeover as takeover
from capx.verification import libero_g17_v271_pt_mech as pt

schema = takeover.schema_version()
source = pt.symbol_source()
assert pt.enabled()
assert "v2.71" in schema and f"{source}-source-pt-paired" in schema, schema
print("v271_preflight_ok", source, schema)
PY
done

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "v271 source snapshot and preflight ready; no jobs submitted"
  exit 0
fi

jobs_tmp=$(mktemp)
echo "[]" >"$jobs_tmp"
for arm in v271fg v271pt; do
  for ((rep=0; rep<K; rep++)); do
    jname=$(printf 'libero-g17-v271-%s-r%02d' "$arm" "$rep")
    log_pattern="$EXPT/logs/${jname}-%j.out"
    jid=$(sbatch \
      --export=ALL,ARM="$arm",REPLICATE="$rep" \
      --job-name="$jname" \
      --output="$log_pattern" \
      --error="$log_pattern" \
      "$SBATCH" | awk '{print $4}')
    echo "submitted arm=$arm replicate=$rep job=$jid"
    "$PYTHON" - <<PYJOB
import json
from pathlib import Path
p = Path("$jobs_tmp")
rows = json.loads(p.read_text())
rows.append({"arm": "$arm", "replicate": $rep, "job_id": "$jid"})
p.write_text(json.dumps(rows))
PYJOB
  done
done

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time
from pathlib import Path

out = Path("$EXPT/manifests")
source_manifest = out / "SOURCE_SHA256.json"
source_snapshot = out / "SOURCE_SNAPSHOT.zip"
anchor = Path("$ANCHOR")
data = {
    "mode": "libero_g17_v271_mech_pt_pair_k5x2",
    "K": 5,
    "arms": ["v271fg", "v271pt"],
    "schema": "capx-libero-g17-fg-takeover-v2.71-ordinal-sgp-nearmiss-osc-tipxy-bandwindow-v256-shared-mech-{fg|pt}-source-pt-paired",
    "mechanism": "v269 policy+grasp replay; one shared v256 controller; both sources materialized; selected action band is the sole arm variable",
    "pt_scope": "get_object_pose c2c ternary action bands only; v269 release predicate remains fixed",
    "anchor_golden_sha256": hashlib.sha256((anchor / "manifests/GOLDEN_ANCHOR.json").read_bytes()).hexdigest(),
    "prefix_sha256": hashlib.sha256(Path("$PREFIX").read_bytes()).hexdigest(),
    "grasp_sha256": hashlib.sha256(Path("$GRASP").read_bytes()).hexdigest(),
    "effective_seed_replay_manifest_sha256": hashlib.sha256((anchor / "assets/effective_seed_replay/MANIFEST.json").read_bytes()).hexdigest(),
    "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
    "source_snapshot_sha256": hashlib.sha256(source_snapshot.read_bytes()).hexdigest(),
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "jobs": json.loads(Path("$jobs_tmp").read_text()),
}
stamp = time.strftime("%Y%m%d_%H%M%S")
(out / f"submit_K5x2_{stamp}.json").write_text(json.dumps(data, indent=2) + "\n")
(out / "LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PYMANIFEST
rm -f "$jobs_tmp"
