#!/usr/bin/env bash
# Complete v286 K=5 using frozen rep1 and fresh rep0/2/3/4, without changing behavior.
set -euo pipefail

G17=/home/qshou/capx_g17_observation_integrity
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
SBATCH_SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT="$PROJECT/outputs/libero_g17_fair_fg_v286_final_fg_20260819"
SOURCE="$EXPT/manifests/SOURCE_SHA256.json"
SNAPSHOT="$EXPT/manifests/SOURCE_SNAPSHOT.zip"
FROZEN="$EXPT/manifests/FROZEN_FOCUSED_ANCHOR.json"

[[ -x "$PYTHON" && -f "$SBATCH_SCRIPT" && -f "$SOURCE" && -f "$SNAPSHOT" && -f "$FROZEN" ]]

"$PYTHON" - <<PYVERIFY
import hashlib, json
from pathlib import Path
source = json.loads(Path("$SOURCE").read_text())
frozen = json.loads(Path("$FROZEN").read_text())
if frozen.get("schema") != "capx_libero_g17_v286_focused_frozen_anchor_v1":
    raise SystemExit("v286 focused freeze missing")
if frozen.get("score") != {"successes": 2, "trials": 2}:
    raise SystemExit("v286 focused freeze is not 2/2")
for raw, expected in source["files"].items():
    path = Path(raw)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v286 source drift: {path}: {actual} != {expected}")
print("v286_k5_source_and_freeze_ok", len(source["files"]))
PYVERIFY

jobs_json="["
separator=""
for rep in 0 2 3 4; do
  jname=$(printf 'libero-g17-v286-final-fg-t00-r%02d' "$rep")
  log="$EXPT/logs/${jname}-%j.out"
  jid=$(sbatch \
    --export=ALL,ARM=v286,TASK_ID=0,REPLICATE="$rep",TOTAL_TRIALS=2 \
    --job-name="$jname" --output="$log" --error="$log" \
    "$SBATCH_SCRIPT" | awk '{print $4}')
  jobs_json+="$separator{\"replicate\":$rep,\"job_id\":\"$jid\"}"
  separator=","
  echo "submitted rep=$rep job=$jid"
done
jobs_json+="]"

"$PYTHON" - <<PYREGISTER
import hashlib, json, time
from pathlib import Path
source = Path("$SOURCE")
snapshot = Path("$SNAPSHOT")
frozen = Path("$FROZEN")
fresh = json.loads('$jobs_json')
data = {
    "schema": "capx_libero_g17_v286_k5_live_submit_v1",
    "task_id": 0,
    "total_trials_per_replicate": 2,
    "expected_successes": "9-10/10",
    "frozen_existing": {"replicate": 1, "job_id": "523865", "successes": 2},
    "fresh_jobs": fresh,
    "behavior_change_from_focused": "none",
    "jitter_policy": "observe and classify separately; no smoothing or gate added during K5",
    "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "source_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    "focused_freeze_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
out = Path("$EXPT/manifests")
stamp = time.strftime("%Y%m%d_%H%M%S")
(out / f"SUBMIT_K5_{stamp}.json").write_text(json.dumps(data, indent=2) + "\n")
(out / "LATEST_SUBMIT_K5.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PYREGISTER
