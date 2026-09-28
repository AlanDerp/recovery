#!/usr/bin/env bash
#SBATCH --job-name=capx-fgsrp
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/qshou/capx_paper_verifier_results/logs/fgsrp-%j.out
#SBATCH --error=/home/qshou/capx_paper_verifier_results/logs/fgsrp-%j.out

set -euo pipefail

project=${PROJECT_DIR:-/home/qshou/capx_pgsvp_v9_frozen}
submission_rev=${CAPX_SUBMISSION_GIT_REV:-$(git -C "$project" rev-parse HEAD)}
submission_rev=$(git -C "$project" rev-parse "${submission_rev}^{commit}")
export CAPX_SUBMISSION_GIT_REV="$submission_rev"
runner_snapshot="${TMPDIR:-/tmp}/capx-fgsrp-core-runner-${SLURM_JOB_ID:-$$}.sbatch"
git -C "$project" show \
  "${submission_rev}:experiments/core_fg/run_paired_task.sbatch" \
  > "$runner_snapshot"
chmod a-w "$runner_snapshot"
export CAPX_SUBMISSION_RUNNER_GIT_REV="$submission_rev"
export CAPX_SUBMISSION_RUNNER_SNAPSHOT="$runner_snapshot"
export CAPX_SUBMISSION_RUNNER_SHA256
CAPX_SUBMISSION_RUNNER_SHA256=$(sha256sum "$runner_snapshot" | cut -d' ' -f1)
export CAPX_TREATMENT=fgsrp
exec bash "$runner_snapshot"
