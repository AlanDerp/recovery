#!/usr/bin/env bash
# v278 trusted-SE3 fair live K=5: libero_spatial task0, 2 trials/seed (10 total).
# Uses the sealed v278 SOURCE fingerprints (restore worktree from freeze if drifted).
set -euo pipefail

K=${K:-5}
G17=/home/qshou/capx_g17_observation_integrity
SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v278_trusted_se3_20260819
KSEED=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v278_trusted_se3_kseed_20260819
PREFIX=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
PREFIX_SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PACKAGE_SHA256.txt
PYTHON=/project/peilab/qshou/cap-x-agilex/.venv-libero/bin/python
FROZEN="$EXPT/manifests/FROZEN_ANCHOR.json"
SOURCE="$EXPT/manifests/SOURCE_SHA256.json"

[[ "$K" == "5" ]] || { echo "v278 kseed is registered for K=5, got K=$K" >&2; exit 2; }
[[ -f "$SCRIPT" && -f "$PREFIX" && -f "$PREFIX_SHA" && -f "$FROZEN" && -f "$SOURCE" ]]
mkdir -p "$EXPT/logs" "$EXPT/arms/v278" "$EXPT/manifests" "$KSEED/logs" "$KSEED/manifests"

"$PYTHON" - <<PYCHECK
import hashlib, json
from pathlib import Path
frozen = json.loads(Path("$FROZEN").read_text())
if frozen.get("schema") != "capx_libero_g17_v278_frozen_anchor_v1":
    raise SystemExit("v278 kseed requires sealed FROZEN_ANCHOR")
src = json.loads(Path("$SOURCE").read_text())
bad = []
for raw, expected in src["files"].items():
    path = Path(raw)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        bad.append(str(path))
if bad:
    raise SystemExit("v278 source drift; restore from SOURCE_SNAPSHOT before kseed: " + ", ".join(bad))
print("v278_kseed_source_ok", len(src["files"]))
PYCHECK

env \
  -u CAPX_LIBERO_G17_GRASP_REPLAY \
  -u CAPX_LIBERO_G17_GRASP_REPLAY_STRICT \
  -u CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW \
  -u CAPX_LIBERO_G17_V270_LIVE_ISOLATION \
  -u CAPX_LIBERO_G17_V277_CANONICAL_GRASP \
  -u CAPX_LIBERO_G17_V275_TRUSTED_TIPXY \
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
  CAPX_LIBERO_G17_V278_TRUSTED_SE3=1 \
  PYTHONPATH="$G17" "$PYTHON" - <<'PY'
import os
from capx.verification import libero_g17_nearmiss_osc as osc
from capx.verification import libero_g17_takeover as takeover
from capx.integrations.franka.trusted_se3 import trusted_se3_enabled, trusted_se3_contract
s = takeover.schema_version()
assert "v2.78" in s and "trusted-se3" in s, s
assert osc.nearmiss_osc_v269_mech_enabled()
assert osc.v269_symbol_source() == "fg"
assert trusted_se3_enabled()
assert trusted_se3_contract()["schema"] == "capx_libero_trusted_se3_contract_v1"
for key in (
    "CAPX_LIBERO_G17_GRASP_REPLAY",
    "CAPX_LIBERO_G17_V277_CANONICAL_GRASP",
    "CAPX_LIBERO_G17_V279_REACHABLE_SE3",
):
    assert not os.environ.get(key), (key, os.environ.get(key))
print("v278_kseed_preflight_ok", s)
PY

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "v278 kseed fingerprints and preflight are ready; no jobs submitted"
  exit 0
fi

jobs_tmp=$(mktemp)
echo "[]" >"$jobs_tmp"
for ((rep=0; rep<K; rep++)); do
  jname=$(printf 'libero-g17-v278-se3-t00-r%02d' "$rep")
  log_pattern="$EXPT/logs/${jname}-%j.out"
  jid=$(sbatch \
    --export=ALL,ARM=v278,TASK_ID=0,REPLICATE="$rep",TOTAL_TRIALS=2 \
    --job-name="$jname" \
    --output="$log_pattern" \
    --error="$log_pattern" \
    "$SCRIPT" | awk '{print $4}')
  echo "submitted arm=v278 task=0 replicate=$rep job=$jid"
  "$PYTHON" - <<PYJOB
import json
from pathlib import Path
p=Path("$jobs_tmp"); rows=json.loads(p.read_text())
rows.append({"arm":"v278","task_id":0,"replicate":$rep,"job_id":"$jid","total_trials":2})
p.write_text(json.dumps(rows))
PYJOB
done

"$PYTHON" - <<PYMANIFEST
import hashlib, json, time
from pathlib import Path
source_manifest = Path("$SOURCE")
frozen = Path("$FROZEN")
data = {
    "mode": "libero_g17_fair_fg_v278_trusted_se3_k5",
    "K": 5,
    "arm": "v278",
    "task_id": 0,
    "suite": "libero_spatial",
    "total_trials_per_seed": 2,
    "schema": "capx-libero-g17-fg-takeover-v2.78-ordinal-sgp-nearmiss-osc-tipxy-bandwindow-v256-shared-mech-fg-source-fair-live-inscribed-completion-local-ik-trusted-se3",
    "mechanism": "live Qwen+Molmo+GraspNet; sealed v278 trusted SE3 goto; v276 completion-local-IK FG+RR; no grasp/effective-seed replay",
    "qwen_policy": "live",
    "grasp_source": "live_sample_grasp_pose",
    "grasp_replay": False,
    "trusted_se3": True,
    "frozen_anchor": str(frozen),
    "frozen_anchor_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
    "prefix_sha256": hashlib.sha256(Path("$PREFIX").read_bytes()).hexdigest(),
    "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
    "note": "K5 live score under sealed v278; WIP files restored from SOURCE_SNAPSHOT for fingerprint gate",
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "jobs": json.loads(Path("$jobs_tmp").read_text()),
}
stamp = time.strftime("%Y%m%d_%H%M%S")
for out in (Path("$EXPT/manifests"), Path("$KSEED/manifests")):
    (out / f"submit_K5_{stamp}.json").write_text(json.dumps(data, indent=2) + "\n")
    (out / "LATEST_SUBMIT_K5.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PYMANIFEST
rm -f "$jobs_tmp"
