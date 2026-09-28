#!/usr/bin/env bash
# Submit v2.6 + v2.7 CAPTURE jobs (raw Qwen audit + Molmo audit) under shared π0.
set -euo pipefail
G17=/home/qshou/capx_g17_observation_integrity
SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_exact_26_27.sh"
EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_exact_26_27_20260815
PKG=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PACKAGE_SHA256.txt

[[ -f "$SCRIPT" && -f "$PKG" && -f "$SHA" ]]
python3 - <<PY
import hashlib
from pathlib import Path
want=Path("$SHA").read_text().strip()
got=hashlib.sha256(Path("$PKG").read_bytes()).hexdigest()
assert got==want, (got, want)
print("prefix_ok", got)
PY

mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/packages"
jobs_tmp=$(mktemp)
echo "[]" >"$jobs_tmp"
for arm in v26 v27; do
  jname="libero-g17-c2627-${arm}-r00"
  jid=$(sbatch --export=ALL,ARM="$arm",MODE=capture,REPLICATE=0 --job-name="$jname" "$SCRIPT" | awk '{print $4}')
  echo "submitted capture arm=$arm job=$jid"
  python3 - <<PY
import json
from pathlib import Path
p=Path("$jobs_tmp")
rows=json.loads(p.read_text())
rows.append({"arm":"$arm","mode":"capture","replicate":0,"job_id":"$jid","job_name":"$jname"})
p.write_text(json.dumps(rows))
PY
done

python3 - <<PY
import hashlib, json, time
from pathlib import Path
pkg=Path("$PKG")
data={
  "phase":"capture",
  "mode":"libero_g17_exact_26_27",
  "arms":["v26","v27"],
  "unique_delta":"schema v2.6-success-witness vs v2.7-sparse-nearmiss",
  "prefix_sha256":hashlib.sha256(pkg.read_bytes()).hexdigest(),
  "note":"Recapture with CAPX_MOLMO_AUDIT_FILE for Molmo response freeze before exact.",
  "molmo_audit": True,
  "submitted_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "jobs":json.loads(Path("$jobs_tmp").read_text()),
}
out=Path("$EXPT/manifests")
p=out/f"capture_{time.strftime('%Y%m%d_%H%M%S')}.json"
p.write_text(json.dumps(data, indent=2)+"\n")
(out/"LATEST_CAPTURE_SUBMIT.json").write_text(json.dumps(data, indent=2)+"\n")
print(json.dumps(data, indent=2))
PY
rm -f "$jobs_tmp"
