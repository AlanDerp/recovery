#!/usr/bin/env bash
#SBATCH --job-name=libero-g17-iff
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_iff_smoke_20260814/logs/%x-%j.out
#SBATCH --error=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_iff_smoke_20260814/logs/%x-%j.out

set -euo pipefail
G17=/home/qshou/capx_g17_observation_integrity
CAPX=/project/peilab/qshou/cap-x-agilex
OUT=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_iff_smoke_20260814
mkdir -p "$OUT/logs"
export LIBERO_G17_SMOKE_OUT="$OUT"
export LIBERO_CONFIG_PATH="$CAPX/.libero"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$G17:$CAPX:${PYTHONPATH:-}"
export TMPDIR="$OUT/tmp-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"

echo "host=$(hostname) job=$SLURM_JOB_ID date=$(date -Is)"
nvidia-smi -L | head -3
source "$CAPX/.venv-libero/bin/activate"
"$CAPX/.venv-libero/bin/python" "$G17/scripts/smoke_libero_g17_iff.py"
