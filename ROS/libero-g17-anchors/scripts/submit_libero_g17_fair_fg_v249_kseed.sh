#!/usr/bin/env bash
# Submit v249: tip↔goto identity + ReleaseReady XY (root TCP fix).
set -euo pipefail
K=${K:-5}
G17=/home/qshou/capx_g17_observation_integrity
SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v249.sh"
EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v249_20260817
BASELINE_FAIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_20260815
PKG=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PACKAGE_SHA256.txt
PYTHON=/project/peilab/qshou/cap-x-agilex/.venv-libero/bin/python
[[ -f "$SCRIPT" && -f "$PKG" && -f "$SHA" ]]
mkdir -p "$EXPT/logs" "$EXPT/arms/v249" "$EXPT/manifests"
PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_placement_precision_groups.py"
CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_placement_precision_groups.py"
CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_task_gap_align.py"
CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_interaction_referents.py"
CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 CAPX_LIBERO_G17_RELEASE_READY=1 CAPX_LIBERO_G17_RELEASE_READY_GOTO=1 CAPX_LIBERO_G17_GOTO_TIP_IDENTITY=1 PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_release_ready.py"
python3 - <<PY
import hashlib, json
from pathlib import Path
want=Path("$SHA").read_text().strip(); got=hashlib.sha256(Path("$PKG").read_bytes()).hexdigest(); assert got==want
def n_valid(arm):
  n=0
  for p in (Path("$BASELINE_FAIR")/"arms"/arm).glob("rep-*/LATEST_SUMMARY.json"):
    if json.loads(p.read_text()).get("validity")=="valid_prefix_arm": n+=1
  return n
assert n_valid("baseline")==5 and n_valid("v24")==5
print("prefix_ok", got)
PY
jobs_tmp=$(mktemp); echo "[]">"$jobs_tmp"
for ((rep=0;rep<K;rep++)); do
  jname=$(printf 'libero-g17-v249-r%02d' "$rep")
  jid=$(sbatch --export=ALL,ARM=v249,REPLICATE="$rep" --job-name="$jname" "$SCRIPT" | awk '{print $4}')
  echo "submitted arm=v249 replicate=$rep job=$jid"
  python3 - <<PY
import json
from pathlib import Path
p=Path("$jobs_tmp"); r=json.loads(p.read_text()); r.append({"arm":"v249","replicate":int("$rep"),"job_id":"$jid"}); p.write_text(json.dumps(r))
PY
done
python3 - <<PY
import hashlib,json,time
from pathlib import Path
data={"mode":"libero_g17_fair_fg_v249_k5","K":int("$K"),"arms":["baseline","v24","v249"],
"schema":"capx-libero-g17-fg-takeover-v2.47-ordinal-sgp-release-ready-goto",
"mechanism":"obs TCP inverse in goto_pose (CAPX_LIBERO_G17_GOTO_TIP_IDENTITY) + ReleaseReady tip XY nudge",
"prefix_sha256":hashlib.sha256(Path("$PKG").read_bytes()).hexdigest(),
"submitted_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
"jobs":json.loads(Path("$jobs_tmp").read_text())}
out=Path("$EXPT/manifests"); out.mkdir(parents=True,exist_ok=True)
p=out/f"submit_K{data['K']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
p.write_text(json.dumps(data,indent=2)+"\n"); (out/"LATEST_SUBMIT.json").write_text(json.dumps(data,indent=2)+"\n")
print("manifest",p,"n",len(data["jobs"]))
PY
rm -f "$jobs_tmp"
