#!/usr/bin/env bash
# Submit the three fair prefix-replay arms (baseline / v24 / next).
set -euo pipefail
G17=/home/qshou/capx_g17_observation_integrity
SCRIPT="$G17/scripts/sbatch_smoke_libero_g17_prefix_arms.sh"
PKG=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
SHA=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PACKAGE_SHA256.txt
AUDIT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PREFIX_REPLAY_AUDIT.json

[[ -f "$PKG" && -f "$SHA" && -f "$AUDIT" ]]
python3 - <<PY
import json
from pathlib import Path
audit=json.loads(Path("$AUDIT").read_text())
assert audit.get("ok") is True, audit
print("fairness_proof_ok", audit["package_sha256"])
PY

mkdir -p /project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_arms_20260815/logs
for arm in baseline v24 next; do
  jid=$(sbatch --export=ALL,ARM="$arm" --job-name="libero-g17-pfx-${arm}" "$SCRIPT" | awk '{print $4}')
  echo "submitted arm=$arm job=$jid"
done
