#!/usr/bin/env bash
# v270-live-isolation: live Qwen + frozen grasp, with the v269 10/10 RR core.
set -euo pipefail

K=${K:-5}
G17=/home/qshou/capx_g17_observation_integrity
SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v270_live_isolation_20260818
PREFIX=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
PREFIX_SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PACKAGE_SHA256.txt
GRASP=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_grasp_replay_v256_20260817/grasp_replay_v1.json
GRASP_SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_grasp_replay_v256_20260817/PACKAGE_SHA256.txt
PYTHON=/project/peilab/qshou/cap-x-agilex/.venv-libero/bin/python

[[ "$K" == "5" ]] || { echo "v270-live-isolation is registered for K=5, got K=$K" >&2; exit 2; }
[[ -f "$SCRIPT" && -f "$PREFIX" && -f "$PREFIX_SHA" && -f "$GRASP" && -f "$GRASP_SHA" ]]
mkdir -p "$EXPT/logs" "$EXPT/arms/v270" "$EXPT/manifests"

# Freeze every executable source in the live-isolation treatment.  This is a
# fresh manifest: v269's historical manifest must remain immutable.
"$PYTHON" - <<PYHASH
import hashlib, json
from pathlib import Path
paths = [
    Path("$SCRIPT"),
    Path("$G17/capx/verification/libero_g17_nearmiss_osc.py"),
    Path("$G17/capx/verification/libero_g17_release_ready.py"),
    Path("$G17/capx/verification/libero_g17_live_isolation.py"),
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
    Path("$G17/capx/envs/simulators/libero.py"),
    Path("$G17/capx/envs/runner.py"),
    Path("$G17/capx/envs/tasks/base.py"),
    Path("$G17/capx/envs/trial.py"),
    Path("$G17/python_sitecustomize/sitecustomize.py"),
    Path("/home/qshou/capx_causal_ik_replay_81e0acb/experiments/core_fg/runtime/python_sitecustomize/sitecustomize.py"),
    Path("/home/qshou/capx_causal_ik_replay_81e0acb/experiments/core_fg/runtime/qwen_proxy.py"),
]
data = {
    "schema": "capx_libero_g17_v270_source_sha256_v1",
    "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
}
out = Path("$EXPT/manifests/SOURCE_SHA256.json")
out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(out, len(paths))
PYHASH

# No-sim preflight: same v269 controller, FG action symbols, plus v270 guards.
CAPX_LIBERO_G17_XY_LOCK=1 \
CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 \
CAPX_LIBERO_G17_RELEASE_READY=1 \
CAPX_LIBERO_G17_NEARMISS_OSC=1 \
CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW=1 \
CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH=1 \
CAPX_LIBERO_G17_V269_MECH=1 \
CAPX_LIBERO_G17_V269_SYMBOL_SOURCE=fg \
CAPX_LIBERO_G17_V270_LIVE_ISOLATION=1 \
CAPX_LIBERO_G17_GRASP_REPLAY="$GRASP" \
PYTHONPATH="$G17" "$PYTHON" - <<'PY'
from capx.verification import libero_g17_live_isolation as isolation
from capx.verification import libero_g17_nearmiss_osc as osc
from capx.verification import libero_g17_takeover as takeover
s = takeover.schema_version()
assert "v2.70" in s and "v256-shared-mech-fg-source-live-isolation" in s, s
assert osc.nearmiss_osc_v269_mech_enabled()
assert osc.v270_live_isolation_enabled()
assert osc.v269_symbol_source() == "fg"
assert isolation.enabled()
assert abs(osc.axis_magnitude(0.15, 0.03, 4) - 0.03) < 1e-12
print("v270_live_isolation_preflight_ok", s)
PY

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "v270-live-isolation source fingerprints and preflight are ready; no jobs submitted"
  exit 0
fi

jobs_tmp=$(mktemp)
echo "[]" >"$jobs_tmp"
for ((rep=0; rep<K; rep++)); do
  jname=$(printf 'libero-g17-v270-live-r%02d' "$rep")
  log_pattern="$EXPT/logs/${jname}-%j.out"
  jid=$(sbatch \
    --export=ALL,ARM=v270,REPLICATE="$rep" \
    --job-name="$jname" \
    --output="$log_pattern" \
    --error="$log_pattern" \
    "$SCRIPT" | awk '{print $4}')
  echo "submitted arm=v270 replicate=$rep job=$jid"
  "$PYTHON" - <<PYJOB
import json
from pathlib import Path
p=Path("$jobs_tmp"); rows=json.loads(p.read_text())
rows.append({"arm":"v270","replicate":$rep,"job_id":"$jid"})
p.write_text(json.dumps(rows))
PYJOB
done

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time
from pathlib import Path
source_manifest = Path("$EXPT/manifests/SOURCE_SHA256.json")
data = {
    "mode": "libero_g17_fair_fg_v270_live_isolation_k5",
    "K": 5,
    "arm": "v270",
    "schema": "capx-libero-g17-fg-takeover-v2.70-ordinal-sgp-nearmiss-osc-tipxy-bandwindow-v256-shared-mech-fg-source-live-isolation",
    "mechanism": "live Qwen; frozen prefix+v256 grasp; exact v269 RR controller; synchronous release-intent ownership; fail-closed gate; absorbing terminal latch",
    "qwen_policy": "live",
    "prefix_sha256": hashlib.sha256(Path("$PREFIX").read_bytes()).hexdigest(),
    "grasp_sha256": hashlib.sha256(Path("$GRASP").read_bytes()).hexdigest(),
    "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "jobs": json.loads(Path("$jobs_tmp").read_text()),
}
out = Path("$EXPT/manifests")
stamp = time.strftime("%Y%m%d_%H%M%S")
(out / f"submit_K5_{stamp}.json").write_text(json.dumps(data, indent=2) + "\n")
(out / "LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PYMANIFEST
rm -f "$jobs_tmp"
