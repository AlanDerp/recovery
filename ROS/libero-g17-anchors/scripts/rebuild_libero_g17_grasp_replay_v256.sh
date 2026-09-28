#!/usr/bin/env bash
# Rebuild grasp_replay_v1.json from v256 fair-FG logs (donor freeze package).
set -euo pipefail
PYTHON=${PYTHON:-/project/peilab/qshou/cap-x-agilex/.venv-libero/bin/python}
G17=${G17:-/home/qshou/capx_g17_observation_integrity}
"$PYTHON" "$G17/scripts/build_libero_g17_grasp_replay_v256.py" "$@"
