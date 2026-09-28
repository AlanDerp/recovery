#!/usr/bin/env bash
# Submit v26 success-witness K replicates (treatment only; reuse prior baseline).
# Usage: K=5 ./scripts/submit_libero_g17_fair_fg_v26_kseed.sh
set -euo pipefail

K=${K:-5}
G17=/home/qshou/capx_g17_observation_integrity
SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v26.sh"
EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v26_20260815
BASELINE_FAIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_20260815
PKG=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PACKAGE_SHA256.txt
DRY="$EXPT/DRY_RUN_SUCCESS_WITNESS.json"

if ! [[ "$K" =~ ^[1-9][0-9]*$ ]]; then
  echo "K must be a positive integer, got: $K" >&2
  exit 2
fi

[[ -f "$SCRIPT" && -f "$PKG" && -f "$SHA" ]]
mkdir -p "$EXPT/logs" "$EXPT/arms/v26" "$EXPT/manifests"

# Require dry-run pass
"$G17/../cap-x-agilex/.venv-libero/bin/python" "$G17/scripts/dry_run_libero_g17_success_witness.py"
python3 - <<PY
import json
from pathlib import Path
dry=json.loads(Path("$DRY").read_text())
assert dry.get("ok") is True, dry
n=0
for p in Path("$BASELINE_FAIR/arms/baseline").glob("rep-*/LATEST_SUMMARY.json"):
    s=json.loads(p.read_text())
    if s.get("validity")=="valid_prefix_arm":
        n+=1
assert n==5, n
print("reuse_baseline_ok", n)
PY

jobs_tmp=$(mktemp)
echo "[]" >"$jobs_tmp"
for ((rep=0; rep<K; rep++)); do
  jname=$(printf 'libero-g17-v26-r%02d' "$rep")
  jid=$(sbatch --export=ALL,ARM=v26,REPLICATE="$rep" --job-name="$jname" "$SCRIPT" | awk '{print $4}')
  echo "submitted arm=v26 replicate=$rep job=$jid name=$jname"
  python3 - <<PY
import json
from pathlib import Path
path=Path("$jobs_tmp")
rows=json.loads(path.read_text())
rows.append({"arm":"v26","replicate":int("$rep"),"job_id":"$jid","job_name":"$jname"})
path.write_text(json.dumps(rows))
PY
done

python3 - <<PY
import hashlib, json, time
from pathlib import Path
pkg=Path("$PKG")
data={
  "mode":"libero_g17_fair_fg_v26_success_witness",
  "K":int("$K"),
  "arms":["v26"],
  "reuse_baseline_from":"$BASELINE_FAIR",
  "unique_delta_vs_v24":"terminal_success_witness + near-miss XY note",
  "schema":"capx-libero-g17-fg-takeover-v2.6-success-witness",
  "prefix_sha256":hashlib.sha256(pkg.read_bytes()).hexdigest(),
  "submitted_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "jobs":json.loads(Path("$jobs_tmp").read_text()),
}
out=Path("$EXPT/manifests")
out.mkdir(parents=True, exist_ok=True)
p=out/f"submit_K{data['K']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
p.write_text(json.dumps(data, indent=2)+"\n")
(out/"LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2)+"\n")
print("manifest", p)
print("n_jobs", len(data["jobs"]))
PY
rm -f "$jobs_tmp"
