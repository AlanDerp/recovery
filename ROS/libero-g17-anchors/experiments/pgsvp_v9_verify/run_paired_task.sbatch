#!/usr/bin/env bash
#SBATCH --job-name=capx-pgsvp-v9-verify
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/qshou/capx_paper_verifier_results/logs/pgsvp-v9-verify-%j.out
#SBATCH --error=/home/qshou/capx_paper_verifier_results/logs/pgsvp-v9-verify-%j.out

set -euo pipefail

project=${PROJECT_DIR:-/home/qshou/capx_pgsvp_v9_verify_clean}
export CAPX_TREATMENT=pgsvp_v9_verify
exec bash "$project/experiments/core_fg/run_paired_task.sbatch"

