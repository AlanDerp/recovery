#!/usr/bin/env bash
# Submit v257: v256 osc + student get_object_pose residual ablation.
set -euo pipefail
K=${K:-5}
G17=/home/qshou/capx_g17_observation_integrity
SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v257.sh"
EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v257_20260817
BASELINE_FAIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_20260815
PKG=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PACKAGE_SHA256.txt
PYTHON=/project/peilab/qshou/cap-x-agilex/.venv-libero/bin/python
[[ -f "$SCRIPT" && -f "$PKG" && -f "$SHA" ]]
mkdir -p "$EXPT/logs" "$EXPT/arms/v257" "$EXPT/manifests"
PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_tip_xy_only.py"
PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_nearmiss_osc.py"
CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 \
CAPX_LIBERO_G17_RELEASE_READY=1 CAPX_LIBERO_G17_NEARMISS_OSC=1 \
CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW=1 \
CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL=1 PYTHONPATH="$G17" "$PYTHON" - <<'PY'
from capx.verification import libero_g17_takeover as t
from capx.verification import libero_g17_release_ready as rr
s=t.schema_version()
assert "v2.55" in s and "bandwindow-fair-residual" in s, s
from capx.verification import libero_g17_nearmiss_osc as osc
assert osc.nearmiss_osc_bandwindow_enabled()
assert osc.nearmiss_osc_fair_residual()
assert abs(osc._PROOF_START_M - 0.15) < 1e-12
print("schema_ok", s, "start", osc._PROOF_START_M)
PY
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
  jname=$(printf 'libero-g17-v257-r%02d' "$rep")
  jid=$(sbatch --export=ALL,ARM=v257,REPLICATE="$rep" --job-name="$jname" "$SCRIPT" | awk '{print $4}')
  echo "submitted arm=v257 replicate=$rep job=$jid"
  python3 - <<PY
import json
from pathlib import Path
p=Path("$jobs_tmp"); r=json.loads(p.read_text()); r.append({"arm":"v257","replicate":int("$rep"),"job_id":"$jid"}); p.write_text(json.dumps(r))
PY
done
python3 - <<PY
import hashlib,json,time
from pathlib import Path
data={"mode":"libero_g17_fair_fg_v257_k5","K":int("$K"),"arms":["baseline","v24","v256","v257"],
"schema":"capx-libero-g17-fg-takeover-v2.55-ordinal-sgp-nearmiss-osc-tipxy-bandwindow-fair-residual",
"mechanism":"RR osc 15-3t + bandwindow + student get_object_pose residual (ablation vs v256 GT sign)",
"prefix_sha256":hashlib.sha256(Path("$PKG").read_bytes()).hexdigest(),
"submitted_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
"jobs":json.loads(Path("$jobs_tmp").read_text())}
out=Path("$EXPT/manifests"); out.mkdir(parents=True,exist_ok=True)
p=out/f"submit_K{data['K']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
p.write_text(json.dumps(data,indent=2)+"\n"); (out/"LATEST_SUBMIT.json").write_text(json.dumps(data,indent=2)+"\n")
print("manifest",p,"n",len(data["jobs"]))
PY
rm -f "$jobs_tmp"
