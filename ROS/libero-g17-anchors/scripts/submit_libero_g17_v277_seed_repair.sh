#!/usr/bin/env bash
# Single-seed counterfactual: canonical top-down grasp + world-Z pregrasp + FGRR.
set -euo pipefail

G17=/home/qshou/capx_g17_observation_integrity
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
SBATCH="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT="$PROJECT/outputs/libero_g17_v277_seed_repair_20260818"
GRASP="$EXPT/assets/v272_rep01_trial02_first_grasp.json"
GRASP_SHA="$EXPT/assets/FIRST_GRASP_SHA256.txt"
PREFIX="$PROJECT/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json"

[[ -x "$PYTHON" && -f "$SBATCH" && -f "$GRASP" && -f "$GRASP_SHA" && -f "$PREFIX" ]]
mkdir -p "$EXPT/logs" "$EXPT/manifests" "$EXPT/arms/v277/task-00"

"$PYTHON" - <<PY
import hashlib, json, math, zipfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

source_q = np.array([-0.34075639, 0.60183579, -0.62495569, -0.36208998])
q = np.array([0.0, 0.0, 1.0, 0.0])
p = np.array([0.64733738, 0.19962308, 0.02439878])
rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
hand = p + rot.apply(np.array([0.0, 0.0, -0.1]))
pregrasp = hand + np.array([0.0, 0.0, 0.15])
assert np.allclose(pregrasp[:2], hand[:2])
assert math.isclose(float(pregrasp[2] - hand[2]), 0.15, abs_tol=1e-12)
assert hand[2] > p[2]

paths = [
    Path("$SBATCH"),
    Path("$G17/capx/integrations/franka/libero.py"),
    Path("$G17/capx/integrations/franka/common.py"),
    Path("$G17/capx/integrations/franka/tip_xy_only.py"),
    Path("$G17/capx/verification/libero_g17_nearmiss_osc.py"),
    Path("$G17/capx/verification/libero_g17_release_ready.py"),
    Path("$G17/capx/verification/libero_g17_takeover.py"),
    Path("$G17/capx/verification/libero_g17_live_isolation.py"),
    Path("$G17/capx/verification/libero_g17_grasp_replay.py"),
    Path("$G17/capx/verification/libero_g17_xy_lock.py"),
    Path("$G17/capx/verification/libero_g17_fg_projection.py"),
    Path("$G17/capx/envs/simulators/libero.py"),
    Path("$G17/capx/envs/trial.py"),
    Path("$G17/python_sitecustomize/sitecustomize.py"),
]
manifest = {
    "schema": "capx_libero_g17_v277_seed_repair_source_v1",
    "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    "counterfactual": {
        "source_job": "523227",
        "replicate": 1,
        "trial": 2,
        "init_state_idx": 1,
        "trial_seed": 20260720,
        "first_grasp_sha256": hashlib.sha256(Path("$GRASP").read_bytes()).hexdigest(),
        "source_grasp_quaternion_wxyz": source_q.tolist(),
        "executed_grasp_quaternion_wxyz": q.tolist(),
        "grasp_contract": "canonical top-down quaternion before goto_pose",
        "pregrasp_contract": "legacy grasp-center offset plus [0,0,+z_approach]_world"
    },
}
out = Path("$EXPT/manifests")
source = out / "SOURCE_SHA256.json"
source.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
snapshot = out / "SOURCE_SNAPSHOT.zip"
with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, arcname=str(path).lstrip("/"))
(out / "FRAME_PREFLIGHT.json").write_text(json.dumps({
    "tip_position": p.tolist(),
    "source_grasp_quaternion_wxyz": source_q.tolist(),
    "executed_grasp_quaternion_wxyz": q.tolist(),
    "grasp_hand_position": hand.tolist(),
    "world_z_pregrasp_hand_position": pregrasp.tolist(),
    "xy_drift_m": float(np.linalg.norm(pregrasp[:2] - hand[:2])),
    "z_rise_m": float(pregrasp[2] - hand[2]),
}, indent=2) + "\n")
print("v277_frame_preflight_ok", hand, pregrasp)
PY

env \
  CAPX_LIBERO_G17_XY_LOCK=1 \
  CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1 \
  CAPX_LIBERO_G17_RELEASE_READY=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW=1 \
  CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH=1 \
  CAPX_LIBERO_G17_V269_MECH=1 \
  CAPX_LIBERO_G17_V269_SYMBOL_SOURCE=fg \
  CAPX_LIBERO_G17_V272_FAIR_LIVE=1 \
  CAPX_LIBERO_G17_V277_CANONICAL_GRASP=1 \
  CAPX_LIBERO_G17_GRASP_REPLAY="$GRASP" \
  CAPX_LIBERO_G17_GRASP_REPLAY_STRICT=0 \
  CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW=live \
  PYTHONPATH="$G17" "$PYTHON" - <<'PY'
from capx.verification import libero_g17_takeover as takeover
s = takeover.schema_version()
assert "v2.77" in s and "single-seed-canonical-world-z-grasp" in s, s
print("v277_schema_preflight_ok", s)
PY

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "v277 prepared; no job submitted"
  exit 0
fi

jname=libero-g17-v277-seed-repair-r01
log="$EXPT/logs/${jname}-%j.out"
jid=$(sbatch \
  --export=ALL,ARM=v277,TASK_ID=0,REPLICATE=1,TOTAL_TRIALS=2 \
  --job-name="$jname" --output="$log" --error="$log" \
  "$SBATCH" | awk '{print $4}')

"$PYTHON" - <<PY
import hashlib, json, time
from pathlib import Path
out = Path("$EXPT/manifests")
source = out / "SOURCE_SHA256.json"
snapshot = out / "SOURCE_SNAPSHOT.zip"
data = {
    "schema": "capx_libero_g17_v277_seed_repair_submit_v1",
    "source_job": "523227",
    "target_cell": {"task_id": 0, "replicate": 1, "trial": 2, "init_state_idx": 1},
    "mechanism": "same registered prefix, first grasp position and seed; canonical top-down quaternion; world-Z pregrasp; live continuation; v269 FG ReleaseReady",
    "job_id": "$jid",
    "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "source_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
(out / "LATEST_SUBMIT.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PY
