#!/usr/bin/env bash
# Submit v266: v264 + silent-allow close (soft-held promote; no tipxy contact gate).
set -euo pipefail
K=${K:-5}
G17=/home/qshou/capx_g17_observation_integrity
SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v266.sh"
EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v266_20260818
BASELINE_FAIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_20260815
PKG=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PACKAGE_SHA256.txt
GRASP_PKG=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_grasp_replay_v256_20260817/grasp_replay_v1.json
GRASP_SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_grasp_replay_v256_20260817/PACKAGE_SHA256.txt
PYTHON=/project/peilab/qshou/cap-x-agilex/.venv-libero/bin/python
[[ -f "$SCRIPT" && -f "$PKG" && -f "$SHA" && -f "$GRASP_PKG" && -f "$GRASP_SHA" ]]
mkdir -p "$EXPT/logs" "$EXPT/arms/v266" "$EXPT/manifests"
PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_tip_xy_only.py"
PYTHONPATH="$G17" "$PYTHON" "$G17/scripts/dry_run_libero_g17_nearmiss_osc.py"
CAPX_LIBERO_G17_XY_LOCK=1 CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 \
CAPX_LIBERO_G17_RELEASE_READY=1 CAPX_LIBERO_G17_NEARMISS_OSC=1 \
CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW=1 \
CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH=1 \
CAPX_LIBERO_G17_GRASP_REPLAY="$GRASP_PKG" \
CAPX_LIBERO_G17_GRASP_REPLAY_STRICT=1 \
CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW=repeat_last \
CAPX_LIBERO_G17_RELEASE_READY_NO_SILENT_ALLOW=1 \
REPLICATE=0 CAPX_ACTIVE_TRIAL=1 \
PYTHONPATH="$G17" "$PYTHON" - <<'PY'
from capx.verification import libero_g17_takeover as t
s=t.schema_version()
assert "v2.64" in s and "silent-allow-close" in s and "grasp-replay" in s, s
assert "hold-guard" not in s
assert "inscribed" not in s
from capx.verification import libero_g17_release_ready as rr
assert hasattr(rr, "promote_soft_held")
from capx.verification import libero_g17_nearmiss_osc as osc
assert osc.nearmiss_osc_fg_predicate_wash_enabled()
assert not osc.nearmiss_osc_inscribed_square_enabled()
assert not osc.nearmiss_osc_no_z_abort_enabled()
start, step = osc.schedule_scalars_m()
assert abs(start - 0.15) < 1e-12 and abs(step - 0.03) < 1e-12
from capx.verification import libero_g17_grasp_replay as gr
assert gr.enabled() and gr.overflow_mode() == "repeat_last"
p0 = gr.next_pose(object_name="akita_black_bowl")
assert p0 is not None and p0.call_idx == 0
print("schema_ok", s, "start", start, "step", step, "grasp0", p0.position.tolist())
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
wantg=Path("$GRASP_SHA").read_text().strip(); gotg=hashlib.sha256(Path("$GRASP_PKG").read_bytes()).hexdigest(); assert gotg==wantg
print("grasp_replay_ok", gotg)
PY
jobs_tmp=$(mktemp); echo "[]">"$jobs_tmp"
for ((rep=0;rep<K;rep++)); do
  jname=$(printf 'libero-g17-v266-r%02d' "$rep")
  jid=$(sbatch --export=ALL,ARM=v266,REPLICATE="$rep" --job-name="$jname" "$SCRIPT" | awk '{print $4}')
  echo "submitted arm=v266 replicate=$rep job=$jid"
  python3 - <<PY
import json
from pathlib import Path
p=Path("$jobs_tmp"); r=json.loads(p.read_text()); r.append({"arm":"v266","replicate":int("$rep"),"job_id":"$jid"}); p.write_text(json.dumps(r))
PY
done
python3 - <<PY
import hashlib,json,time
from pathlib import Path
data={"mode":"libero_g17_fair_fg_v266_k5","K":int("$K"),"arms":["baseline","v24","v256","v264","v265","v266"],
"schema":"capx-libero-g17-fg-takeover-v2.64-ordinal-sgp-nearmiss-osc-tipxy-bandwindow-fg-predicate-wash-grasp-replay-silent-allow-close",
"mechanism":"v264+silent-allow-close: soft-held promote into tipxy; no tipxy contact gate (v265 rollback)",
"prefix_sha256":hashlib.sha256(Path("$PKG").read_bytes()).hexdigest(),
"grasp_sha256":hashlib.sha256(Path("$GRASP_PKG").read_bytes()).hexdigest(),
"submitted_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
"jobs":json.loads(Path("$jobs_tmp").read_text())}
out=Path("$EXPT/manifests"); out.mkdir(parents=True,exist_ok=True)
p=out/f"submit_K{data['K']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
p.write_text(json.dumps(data,indent=2)+"\n"); (out/"LATEST_SUBMIT.json").write_text(json.dumps(data,indent=2)+"\n")
print("manifest",p,"n",len(data["jobs"]))
PY
rm -f "$jobs_tmp"
