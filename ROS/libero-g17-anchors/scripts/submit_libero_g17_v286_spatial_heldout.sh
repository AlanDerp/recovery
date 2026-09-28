#!/usr/bin/env bash
# Run frozen v286 unchanged on the exact v273 task1-9, trials1-2 held-out breadth set.
set -euo pipefail

G17=/home/qshou/capx_g17_observation_integrity
PROJECT=/project/peilab/qshou/cap-x-agilex
PYTHON="$PROJECT/.venv-libero/bin/python"
SBATCH_SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh"
EXPT="$PROJECT/outputs/libero_g17_fair_fg_v286_final_fg_20260819"
SOURCE="$EXPT/manifests/SOURCE_SHA256.json"
FROZEN="$EXPT/manifests/FROZEN_K5_RESULT.json"
OLD="$PROJECT/outputs/libero_g17_fair_fg_v273_spatial_generalization_20260818"
OLD_SUBMIT="$OLD/manifests/LATEST_SUBMIT.json"
PREFIX_INDEX="$PROJECT/outputs/libero_g17_spatial_prefix_replay_20260818/INDEX.json"
TASK_IDS=(1 2 3 4 5 6 7 8 9)

[[ -x "$PYTHON" && -f "$SBATCH_SCRIPT" && -f "$SOURCE" && -f "$FROZEN" ]]
[[ -f "$OLD_SUBMIT" && -f "$PREFIX_INDEX" ]]

"$PYTHON" - <<PYVERIFY
import hashlib, json
from pathlib import Path

source = json.loads(Path("$SOURCE").read_text())
frozen = json.loads(Path("$FROZEN").read_text())
old = json.loads(Path("$OLD_SUBMIT").read_text())
index = json.loads(Path("$PREFIX_INDEX").read_text())

if frozen.get("schema") != "capx_libero_g17_v286_k5_frozen_result_v1":
    raise SystemExit("v286 frozen K5 anchor missing")
score = frozen.get("score", {})
if score.get("successes") != 10 or score.get("trials") != 10 or score.get("replicates") != 5:
    raise SystemExit(f"v286 frozen anchor is not K5 10/10: {score}")
mechanical = frozen.get("mechanical", {})
if mechanical.get("max_candidate_arm_delta_rad", 99) > 1.25:
    raise SystemExit("v286 frozen anchor exceeds clean-motion bound")
if any(mechanical.get(key) != 0 for key in
       ("cartesian_waypoints", "joint_delta_gate", "terminal_residual_gate", "rollback")):
    raise SystemExit("v286 frozen anchor changed the command-faithful mechanical contract")
release = frozen.get("release_semantics", {})
if release.get("terminal_latched_after_full_release_window") != 10:
    raise SystemExit("v286 frozen anchor lacks full release-window evidence")
for raw, expected in source["files"].items():
    path = Path(raw)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v286 source drift: {path}: {actual} != {expected}")

expected_tasks = list(range(1, 10))
if old.get("held_out_task_ids") != expected_tasks or old.get("trials") != [1, 2]:
    raise SystemExit("old v273 held-out definition drift")
if old.get("K") != 1 or len(old.get("jobs", [])) != 9:
    raise SystemExit("old v273 held-out cardinality drift")
index_tasks = sorted(int(task_id) for task_id in index.get("tasks", {}))
if index_tasks != expected_tasks:
    raise SystemExit(f"prefix index task axis drift: {index_tasks}")
print("v286_heldout_source_and_anchor_ok", len(source["files"]))
print("v286_heldout_axis_ok", expected_tasks, "trials", old["trials"])
PYVERIFY

jobs_tmp=$(mktemp)
printf '[]\n' >"$jobs_tmp"
for task_id in "${TASK_IDS[@]}"; do
  jname=$(printf 'libero-g17-v286-heldout-t%02d-r00' "$task_id")
  log="$EXPT/logs/${jname}-%j.out"
  jid=$(sbatch \
    --export=ALL,ARM=v286,TASK_ID="$task_id",REPLICATE=0,TOTAL_TRIALS=2 \
    --job-name="$jname" --output="$log" --error="$log" \
    "$SBATCH_SCRIPT" | awk '{print $4}')
  echo "submitted task=$task_id rep=0 job=$jid"
  "$PYTHON" - <<PYJOB
import json
from pathlib import Path
p = Path("$jobs_tmp")
rows = json.loads(p.read_text())
rows.append({"task_id": $task_id, "replicate": 0, "job_id": "$jid"})
p.write_text(json.dumps(rows))
PYJOB
done

"$PYTHON" - <<PYREGISTER
import hashlib, json, time
from pathlib import Path

source = Path("$SOURCE")
frozen = Path("$FROZEN")
old_submit = Path("$OLD_SUBMIT")
prefix_index = Path("$PREFIX_INDEX")
data = {
    "schema": "capx_libero_g17_v286_spatial_heldout_submit_v1",
    "mode": "v286-frozen-on-exact-v273-heldout-breadth",
    "suite": "libero_spatial",
    "anchor_task_id": 0,
    "held_out_task_ids": list(range(1, 10)),
    "trials": [1, 2],
    "K": 1,
    "controller": "frozen v286 FG+RR measured-seed/tool-pi/full-release/final-FG; unchanged",
    "comparison": "exact v273 task/rep/trial axis previously scoring 10/18",
    "qwen_policy": "live after identical task-local frozen CB0",
    "grasp_source": "live sample_grasp_pose",
    "selection": "complete held-out task axis; no failure/success selection",
    "behavior_change_from_frozen_anchor": "none",
    "jitter_policy": "observe and classify separately; no smoothing or gate added",
    "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "frozen_k5_anchor_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
    "v273_submit_sha256": hashlib.sha256(old_submit.read_bytes()).hexdigest(),
    "prefix_index_sha256": hashlib.sha256(prefix_index.read_bytes()).hexdigest(),
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "jobs": json.loads(Path("$jobs_tmp").read_text()),
}
out = Path("$EXPT/manifests")
stamp = time.strftime("%Y%m%d_%H%M%S")
(out / f"SUBMIT_SPATIAL_HELDOUT_{stamp}.json").write_text(json.dumps(data, indent=2) + "\n")
(out / "LATEST_SUBMIT_SPATIAL_HELDOUT.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PYREGISTER
rm -f "$jobs_tmp"
