#!/usr/bin/env bash
# Submit v260: v256 FG-predicate wash (shared-t; RelXY band sign+skip).
set -euo pipefail
K=${K:-5}
G17=/home/qshou/capx_g17_observation_integrity
SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v260.sh"
EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v260_20260817
BASELINE_FAIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_20260815
PKG=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PACKAGE_SHA256.txt
PYTHON=/project/peilab/qshou/cap-x-agilex/.venv-libero/bin/python
[[ -f "$SCRIPT" && -f "$PKG" && -f "$SHA" ]]
mkdir -p "$EXPT/logs" "$EXPT/arms/v260" "$EXPT/manifests"
PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_tip_xy_only.py"
PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_nearmiss_osc.py"
CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 \
CAPX_LIBERO_G17_RELEASE_READY=1 CAPX_LIBERO_G17_NEARMISS_OSC=1 \
CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW=1 \
CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH=1 PYTHONPATH="$G17" "$PYTHON" - <<'PY'
from capx.verification import libero_g17_takeover as t
s=t.schema_version()
assert "v2.58" in s and "fg-predicate-wash" in s, s
from capx.verification import libero_g17_nearmiss_osc as osc
assert osc.nearmiss_osc_bandwindow_enabled()
assert osc.nearmiss_osc_fg_predicate_wash_enabled()
assert not osc.nearmiss_osc_xy_decouple_enabled()
assert abs(osc._PROOF_START_M - 0.15) < 1e-12
from capx.verification.libero_g17_band_obligations import residual_to_band
for r in (-0.05, -0.01, 0.0, 0.01, 0.05):
    b = residual_to_band(axis='x', residual=r, band_m=0.03)
    assert abs(osc.axis_delta_from_fg_band(b, 0.15) - osc.axis_delta(r, 0.15, 0.03)) < 1e-12
print("schema_ok", s, "isomorphic_wash")
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
  jname=$(printf 'libero-g17-v260-r%02d' "$rep")
  jid=$(sbatch --export=ALL,ARM=v260,REPLICATE="$rep" --job-name="$jname" "$SCRIPT" | awk '{print $4}')
  echo "submitted arm=v260 replicate=$rep job=$jid"
  python3 - <<PY
import json
from pathlib import Path
p=Path("$jobs_tmp"); r=json.loads(p.read_text()); r.append({"arm":"v260","replicate":int("$rep"),"job_id":"$jid"}); p.write_text(json.dumps(r))
PY
done
python3 - <<PY
import hashlib,json,time
from pathlib import Path
data={"mode":"libero_g17_fair_fg_v260_k5","K":int("$K"),"arms":["baseline","v24","v256","v260"],
"schema":"capx-libero-g17-fg-takeover-v2.58-ordinal-sgp-nearmiss-osc-tipxy-bandwindow-fg-predicate-wash",
"mechanism":"v256 wash: shared-t 15-3t; sign+skip=FG RelXY bands; open=xy_within; no XY_DECOUPLE",
"prefix_sha256":hashlib.sha256(Path("$PKG").read_bytes()).hexdigest(),
"submitted_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
"jobs":json.loads(Path("$jobs_tmp").read_text())}
out=Path("$EXPT/manifests"); out.mkdir(parents=True,exist_ok=True)
p=out/f"submit_K{data['K']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
p.write_text(json.dumps(data,indent=2)+"\n"); (out/"LATEST_SUBMIT.json").write_text(json.dumps(data,indent=2)+"\n")
print("manifest",p,"n",len(data["jobs"]))
PY
rm -f "$jobs_tmp"
