#!/usr/bin/env bash
# Submit v244: v242 stack + ReleaseReady XY nudge (no LLM / no post-miss clear).
# XY_LOCK=1, WRONG_GRASP_ORDINAL=1, RELEASE_READY=1; BAND_OBLIGATIONS unset.
# Skips v244 reps that already have a valid LATEST_SUMMARY.
# Usage: K=5 ./scripts/submit_libero_g17_fair_fg_v244_kseed.sh
set -euo pipefail

K=${K:-5}
G17=/home/qshou/capx_g17_observation_integrity
SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v244.sh"
EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v244_20260817
BASELINE_FAIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_20260815
PKG=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PACKAGE_SHA256.txt
PYTHON=/project/peilab/qshou/cap-x-agilex/.venv-libero/bin/python

if ! [[ "$K" =~ ^[1-9][0-9]*$ ]]; then
  echo "K must be a positive integer, got: $K" >&2
  exit 2
fi

[[ -f "$SCRIPT" && -f "$PKG" && -f "$SHA" ]]
mkdir -p "$EXPT/logs" "$EXPT/arms/v244" "$EXPT/manifests"

# Dry-run + freeze gates (baseline + early positive v24).
PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_placement_precision_groups.py"
CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_placement_precision_groups.py"
CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_task_gap_align.py"
CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_interaction_referents.py"
CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 CAPX_LIBERO_G17_RELEASE_READY=1 PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_release_ready.py"
python3 - <<PY
import hashlib, json
from pathlib import Path

want = Path("$SHA").read_text().strip()
got = hashlib.sha256(Path("$PKG").read_bytes()).hexdigest()
assert got == want, (got, want)

def n_valid(arm: str) -> int:
    root = Path("$BASELINE_FAIR") / "arms" / arm
    n = 0
    for p in root.glob("rep-*/LATEST_SUMMARY.json"):
        s = json.loads(p.read_text())
        if s.get("validity") == "valid_prefix_arm":
            n += 1
    return n

nb, nv24 = n_valid("baseline"), n_valid("v24")
assert nb == 5, nb
assert nv24 == 5, nv24
print("reuse_baseline_ok", nb)
print("reuse_v24_early_positive_ok", nv24)
print("prefix_sha_ok", got)
PY

jobs_tmp=$(mktemp)
echo "[]" >"$jobs_tmp"
skipped_tmp=$(mktemp)
echo "[]" >"$skipped_tmp"

for ((rep=0; rep<K; rep++)); do
  latest="$EXPT/arms/v244/rep-$(printf '%02d' "$rep")/LATEST_SUMMARY.json"
  if [[ -f "$latest" ]]; then
    valid=$(python3 - <<PY
import json
from pathlib import Path
s=json.loads(Path("$latest").read_text())
print("1" if s.get("validity")=="valid_prefix_arm" else "0")
PY
)
    if [[ "$valid" == "1" ]]; then
      echo "skip arm=v244 replicate=$rep (valid LATEST_SUMMARY exists)"
      python3 - <<PY
import json
from pathlib import Path
path=Path("$skipped_tmp")
rows=json.loads(path.read_text())
rows.append({"arm":"v244","replicate":int("$rep"),"reason":"reuse_valid_summary","path":"$latest"})
path.write_text(json.dumps(rows))
PY
      continue
    fi
  fi
  jname=$(printf 'libero-g17-v244-r%02d' "$rep")
  jid=$(sbatch --export=ALL,ARM=v244,REPLICATE="$rep" --job-name="$jname" "$SCRIPT" | awk '{print $4}')
  echo "submitted arm=v244 replicate=$rep job=$jid name=$jname"
  python3 - <<PY
import json
from pathlib import Path
path=Path("$jobs_tmp")
rows=json.loads(path.read_text())
rows.append({"arm":"v244","replicate":int("$rep"),"job_id":"$jid","job_name":"$jname"})
path.write_text(json.dumps(rows))
PY
done

python3 - <<PY
import hashlib, json, time
from pathlib import Path
pkg=Path("$PKG")
data={
  "mode":"libero_g17_fair_fg_v244_k5_frozen_triple",
  "K":int("$K"),
  "arms":["baseline","v24","v244"],
  "reuse_baseline_from":"$BASELINE_FAIR/arms/baseline",
  "reuse_early_positive_from":"$BASELINE_FAIR/arms/v24",
  "early_positive_note":"v24 is the first fair-FG K=5 treatment with clear joint-positive gain; v26 was weaker on joint/t2",
  "schema":"capx-libero-g17-fg-takeover-v2.42-ordinal-sgp-release-ready-xy",
  "xy_lock":True,
  "release_ready":True,
  "band_obligations":False,
  "post_miss_clear":False,
  "skill_relxy_which":False,
  "segmenter_lock":False,
  "mechanism":"v242 stack + open_gripper ReleaseReady: hold+GT c2c XY nudge, no LLM/lift/regrasp",
  "goal_referent_takeover":False,
  "prefix_sha256":hashlib.sha256(pkg.read_bytes()).hexdigest(),
  "submitted_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "jobs":json.loads(Path("$jobs_tmp").read_text()),
  "skipped":json.loads(Path("$skipped_tmp").read_text()),
}
out=Path("$EXPT/manifests")
out.mkdir(parents=True, exist_ok=True)
p=out/f"submit_K{data['K']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
p.write_text(json.dumps(data, indent=2)+"\n")
(out/"LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2)+"\n")
print("manifest", p)
print("n_submitted", len(data["jobs"]))
print("n_skipped", len(data["skipped"]))
PY
rm -f "$jobs_tmp" "$skipped_tmp"
