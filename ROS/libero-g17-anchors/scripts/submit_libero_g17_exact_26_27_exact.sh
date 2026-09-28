#!/usr/bin/env bash
# After capture jobs finish: export Qwen + Molmo replay packages and submit exact-reruns.
set -euo pipefail
G17=/home/qshou/capx_g17_observation_integrity
EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_exact_26_27_20260815
SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_exact_26_27.sh"
EXPORT_QWEN="$G17/experiments/core_fg/runtime/tools/export_request_hash_replay.py"
EXPORT_MOLMO="$G17/experiments/core_fg/runtime/tools/export_molmo_replay.py"
PYTHON=/project/peilab/qshou/cap-x-agilex/.venv-libero/bin/python
MANIFEST="$EXPT/manifests/LATEST_CAPTURE_SUBMIT.json"

[[ -f "$MANIFEST" ]]
mkdir -p "$EXPT/packages" "$EXPT/manifests"

jobs_tmp=$(mktemp)
echo "[]" >"$jobs_tmp"

python3 - <<'PY' >"$EXPT/manifests/_capture_roots.json"
import json
from pathlib import Path
man = json.loads(Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_exact_26_27_20260815/manifests/LATEST_CAPTURE_SUBMIT.json").read_text())
expt = Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_exact_26_27_20260815")
roots = {}
for job in man["jobs"]:
    arm = job["arm"]
    jid = job["job_id"]
    cands = list((expt / "runs" / "libero_spatial" / "mode-capture" / f"arm-{arm}").glob(f"**/job-{jid}"))
    if not cands:
        cands = list(expt.glob(f"**/arm-{arm}/**/job-{jid}"))
    if not cands:
        raise SystemExit(f"capture root missing for {arm} job-{jid}")
    root = cands[0]
    audit = root / "qwen_proxy_audit.jsonl"
    molmo_audit = root / "molmo_audit.jsonl"
    if not audit.is_file():
        raise SystemExit(f"missing audit: {audit}")
    if not molmo_audit.is_file():
        raise SystemExit(f"missing Molmo audit: {molmo_audit}")
    has_raw = False
    for line in audit.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("ok") and o.get("response_content"):
            has_raw = True
            break
    if not has_raw:
        raise SystemExit(f"capture audit lacks response_content (raw audit missing?): {audit}")
    roots[arm] = {
        "job_id": jid,
        "suite_root": str(root),
        "molmo_audit": str(molmo_audit),
    }
print(json.dumps(roots, indent=2))
PY

for arm in v26 v27; do
  root=$("$PYTHON" - <<PY
import json
from pathlib import Path
print(json.loads(Path("$EXPT/manifests/_capture_roots.json").read_text())["$arm"]["suite_root"])
PY
)
  molmo_audit=$("$PYTHON" - <<PY
import json
from pathlib import Path
print(json.loads(Path("$EXPT/manifests/_capture_roots.json").read_text())["$arm"]["molmo_audit"])
PY
)
  pkg="$EXPT/packages/${arm}_request_hash_replay.json"
  molmo_pkg="$EXPT/packages/${arm}_molmo_response_replay.json"
  echo "[export] arm=$arm root=$root"
  "$PYTHON" "$EXPORT_QWEN" --run-root "$root" --output "$pkg"
  echo "[export-molmo] arm=$arm audit=$molmo_audit"
  "$PYTHON" "$EXPORT_MOLMO" --audit "$molmo_audit" --output "$molmo_pkg"
  jname="libero-g17-e2627-${arm}-r00"
  jid=$(sbatch --export=ALL,ARM="$arm",MODE=exact,REPLICATE=0,CAPX_REQUEST_HASH_REPLAY="$pkg",CAPX_MOLMO_RESPONSE_REPLAY_FILE="$molmo_pkg" \
    --job-name="$jname" "$SCRIPT" | awk '{print $4}')
  echo "submitted exact arm=$arm job=$jid qwen_pkg=$pkg molmo_pkg=$molmo_pkg"
  python3 - <<PY
import json
from pathlib import Path
p=Path("$jobs_tmp")
rows=json.loads(p.read_text())
rows.append({
  "arm":"$arm","mode":"exact","replicate":0,"job_id":"$jid","job_name":"$jname",
  "replay_pkg":"$pkg","molmo_replay_pkg":"$molmo_pkg",
})
p.write_text(json.dumps(rows))
PY
done

python3 - <<PY
import json, time
from pathlib import Path
data={
  "phase":"exact_rerun",
  "mode":"libero_g17_exact_26_27",
  "molmo_freeze": True,
  "capture_roots":json.loads(Path("$EXPT/manifests/_capture_roots.json").read_text()),
  "submitted_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "jobs":json.loads(Path("$jobs_tmp").read_text()),
}
out=Path("$EXPT/manifests")
p=out/f"exact_{time.strftime('%Y%m%d_%H%M%S')}.json"
p.write_text(json.dumps(data, indent=2)+"\n")
(out/"LATEST_EXACT_SUBMIT.json").write_text(json.dumps(data, indent=2)+"\n")
print(json.dumps(data, indent=2))
PY
rm -f "$jobs_tmp"
