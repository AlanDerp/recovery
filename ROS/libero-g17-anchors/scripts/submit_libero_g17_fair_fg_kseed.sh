#!/usr/bin/env bash
# Submit max-freeze fair FG K-replicate arms (baseline vs v24).
# Usage: K=5 ./scripts/submit_libero_g17_fair_fg_kseed.sh
set -euo pipefail

K=${K:-5}
G17=/home/qshou/capx_g17_observation_integrity
SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg.sh"
EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_20260815
PKG=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PACKAGE_SHA256.txt
AUDIT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PREFIX_REPLAY_AUDIT.json

if ! [[ "$K" =~ ^[1-9][0-9]*$ ]]; then
  echo "K must be a positive integer, got: $K" >&2
  exit 2
fi

[[ -f "$SCRIPT" && -f "$PKG" && -f "$SHA" && -f "$AUDIT" ]]
python3 - <<PY
import hashlib, json
from pathlib import Path
want = Path("$SHA").read_text().strip()
got = hashlib.sha256(Path("$PKG").read_bytes()).hexdigest()
assert got == want, (got, want)
audit = json.loads(Path("$AUDIT").read_text())
assert audit.get("ok") is True, audit
print("fairness_proof_ok", got)
PY

mkdir -p "$EXPT/logs" "$EXPT/arms/baseline" "$EXPT/arms/v24" "$EXPT/manifests"

manifest="$EXPT/manifests/submit_K${K}_$(date +%Y%m%d_%H%M%S).json"
python3 - <<PY >"$manifest"
import hashlib, json, time
from pathlib import Path
pkg = Path("$PKG")
print(json.dumps({
    "mode": "libero_g17_fair_fg_max_freeze",
    "K": int("$K"),
    "arms": ["baseline", "v24"],
    "unique_delta": "CAPX_LIBERO_G17_FG",
    "workgap": False,
    "prefix_package": str(pkg),
    "prefix_sha256": hashlib.sha256(pkg.read_bytes()).hexdigest(),
    "global_seed": 20260718,
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "jobs": [],
}, indent=2))
PY

jobs_tmp=$(mktemp)
echo "[]" >"$jobs_tmp"

for arm in baseline v24; do
  for ((rep=0; rep<K; rep++)); do
    jname=$(printf 'libero-g17-fair-%s-r%02d' "$arm" "$rep")
    jid=$(sbatch --export=ALL,ARM="$arm",REPLICATE="$rep" --job-name="$jname" "$SCRIPT" | awk '{print $4}')
    echo "submitted arm=$arm replicate=$rep job=$jid name=$jname"
    python3 - <<PY
import json
from pathlib import Path
path = Path("$jobs_tmp")
rows = json.loads(path.read_text())
rows.append({"arm": "$arm", "replicate": int("$rep"), "job_id": "$jid", "job_name": "$jname"})
path.write_text(json.dumps(rows))
PY
  done
done

python3 - <<PY
import json
from pathlib import Path
manifest = Path("$manifest")
data = json.loads(manifest.read_text())
data["jobs"] = json.loads(Path("$jobs_tmp").read_text())
manifest.write_text(json.dumps(data, indent=2) + "\n")
Path("$EXPT/manifests/LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2) + "\n")
print("manifest", manifest)
print("n_jobs", len(data["jobs"]))
PY
rm -f "$jobs_tmp"
