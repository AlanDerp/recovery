#!/usr/bin/env bash
#SBATCH --job-name=tip-xy-only-probe
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/project/peilab/qshou/cap-x-agilex/outputs/libero_tip_xy_only_probe_20260817/logs/%x-%j.out
#SBATCH --error=/project/peilab/qshou/cap-x-agilex/outputs/libero_tip_xy_only_probe_20260817/logs/%x-%j.out

# Isolated sim probe: tip XY-only primitive vs legacy goto (no fair FG / no LLM).
set -euo pipefail
G17=/home/qshou/capx_g17_observation_integrity
PROJECT_DIR=/project/peilab/qshou/cap-x-agilex
OUT=/project/peilab/qshou/cap-x-agilex/outputs/libero_tip_xy_only_probe_20260817
PYTHON="$PROJECT_DIR/.venv-libero/bin/python"
mkdir -p "$OUT/logs"
export TIP_XY_PROBE_OUT="$OUT"
export TIP_XY_PROBE_STEP_M=0.03
export TMPDIR="$OUT/tmp"
export MPLCONFIGDIR="$OUT/mpl"
export UV_CACHE_DIR=/scratch/peilab/qshou/uv-cache
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/.venv-libero"
export LIBERO_CONFIG_PATH="$PROJECT_DIR/.libero"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$G17${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
"$PYTHON" "$G17/scripts/dry_run_tip_xy_only.py"
"$PYTHON" "$G17/scripts/smoke_tip_xy_only_libero.py"
