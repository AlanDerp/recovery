#!/usr/bin/env bash
#SBATCH --job-name=astar-effect-smoke
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/qshou/capx_paper_verifier_results/logs/astar-effect-smoke-%j.out
#SBATCH --error=/home/qshou/capx_paper_verifier_results/logs/astar-effect-smoke-%j.out

set -euo pipefail

project=${PROJECT_DIR:-/home/qshou/capx_astar_effect_alignment}
mode=${CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT_MODE:?Set the isolated effect mode}
target_trial=${CAPX_TARGET_TRIAL:?Set CAPX_TARGET_TRIAL from 1 through 10}
task_name=${TASK_NAME:-cube_stack}
submission_rev=${CAPX_SUBMISSION_GIT_REV:?Set CAPX_SUBMISSION_GIT_REV}
replay=${CAPX_ASTAR_STACK_REPLAY_FILE:-/home/qshou/capx_paper_verifier_results/manifests/astar_effect_alignment_smoke_20260805/astar_v4_stack_registered_prefix_replay.json}
expected_replay_sha=${CAPX_ASTAR_STACK_REPLAY_SHA256:-f37cb374f587828898576bbc06f74ed162c40689005b6d18d8567ffb60930c1b}
require_all_replay=${CAPX_REQUIRE_ALL_REQUEST_HASH_REPLAY:-0}
enable_normalized_replay=${CAPX_ENABLE_NORMALIZED_REPLAY:-1}
allow_deferred_semantic_visual_match=${CAPX_ALLOW_DEFERRED_SEMANTIC_VISUAL_MATCH:-0}
helper="$project/experiments/core_fg/runtime"
python_bin=${CAPX_PYTHON:-/project/peilab/quanxin/cap-x/.venv-main/bin/python}
result_root=${RESULT_ROOT:-/home/qshou/capx_paper_verifier_results/results/astar_isolated_effect_smoke_20260805}
key_file=${QWEN_KEY_FILE:-/scratch/peilab/qshou/RoboTwin_Sandbox/cap-x/.qwenkey_sg_tmp}

case "$mode" in
  grasp_loss_only|instance_bound_grasp_loss_only|instance_bound_grasp_outcome_only|instance_bound_transport_failure_only|instance_bound_off_table_lift_regression_only|instance_bound_grasp_loss_plus_off_table_lift_regression|post_release_support_only) ;;
  *)
    echo "Unsupported isolated effect mode: $mode" >&2
    exit 2
    ;;
esac
if [[ ! "$target_trial" =~ ^([1-9]|10)$ ]]; then
  echo "Target trial must be an integer from 1 through 10" >&2
  exit 2
fi
case "$task_name" in
  cube_stack|cube_restack) ;;
  *)
    echo "TASK_NAME must be cube_stack or cube_restack" >&2
    exit 2
    ;;
esac
case "$require_all_replay" in
  0|1) ;;
  *)
    echo "CAPX_REQUIRE_ALL_REQUEST_HASH_REPLAY must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$enable_normalized_replay" in
  0|1) ;;
  *)
    echo "CAPX_ENABLE_NORMALIZED_REPLAY must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$allow_deferred_semantic_visual_match" in
  0|1) ;;
  *)
    echo "CAPX_ALLOW_DEFERRED_SEMANTIC_VISUAL_MATCH must be 0 or 1" >&2
    exit 2
    ;;
esac

actual_rev=$(git -C "$project" rev-parse HEAD)
if [[ "$actual_rev" != "$submission_rev" ]]; then
  echo "Submission revision mismatch: expected $submission_rev, found $actual_rev" >&2
  exit 3
fi
if [[ -n "$(git -C "$project" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty worktree" >&2
  exit 3
fi
if [[ ! -f "$replay" ]]; then
  echo "Missing registered A* replay: $replay" >&2
  exit 4
fi
actual_replay_sha=$(sha256sum "$replay" | cut -d' ' -f1)
if [[ "$actual_replay_sha" != "$expected_replay_sha" ]]; then
  echo "Registered A* replay hash mismatch" >&2
  exit 4
fi
if [[ ! -s "$key_file" ]]; then
  echo "Missing Qwen credential file: $key_file" >&2
  exit 4
fi

query_seed_base=$((20260718 + target_trial * 1000))
required_replay_seeds="${query_seed_base}+$((query_seed_base + 1))"
output_base="$result_root/${mode}_trial${target_trial}"
mkdir -p "$output_base" /home/qshou/capx_paper_verifier_results/logs

env \
  PROJECT_DIR="$project" \
  QWEN_HELPER_DIR="$helper" \
  CAPX_FROZEN_GIT_REV="$submission_rev" \
  CAPX_RUNTIME_SOURCE_ROOT="$project" \
  CAPX_PYTHON="$python_bin" \
  QWEN_KEY_FILE="$key_file" \
  QWEN_BASE_URL=https://ws-vpxr6jfor4t48uyo.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1 \
  QWEN_OUTPUT_BASE="$output_base" \
  MODE=M3 \
  MODEL=qwen3.7-plus \
  TEMPERATURE=0 \
  TOTAL_TRIALS="$target_trial" \
  CAPX_RESUME_IDX="$target_trial" \
  NUM_WORKERS=1 \
  TASK_FILTER="$task_name" \
  CAPX_GLOBAL_SEED=20260718 \
  QWEN_REQUEST_SEED=20260718 \
  QWEN_ENABLE_THINKING=false \
  QWEN_UPSTREAM_CONCURRENCY=1 \
  CAPX_SEED_TORCH=1 \
  CAPX_DETERMINISTIC_TRIAL_SEEDING=1 \
  CAPX_DETERMINISTIC_GRASPNET=1 \
  CAPX_START_MOLMO=0 \
  CAPX_RESET_API_EPISODE_STATE=1 \
  CAPX_PROCESS_ISOLATED_TRIALS=1 \
  CAPX_PROCESS_BOOTSTRAP_RETRIES=1 \
  CAPX_MAX_TRIAL_RETRIES=0 \
  CAPX_TRIAL_TIMEOUT_SECONDS=1800 \
  CAPX_REPRO_AUDIT=1 \
  CAPX_MULTITURN_LIMIT=10 \
  QWEN_REQUEST_HASH_REPLAY_FILE="$replay" \
  QWEN_REQUIRED_REPLAY_SEEDS="$required_replay_seeds" \
  QWEN_ENABLE_NORMALIZED_REPLAY="$enable_normalized_replay" \
  QWEN_REQUIRE_ALL_REQUEST_HASH_REPLAY="$require_all_replay" \
  QWEN_REQUIRE_NORMALIZED_REPLAY_VISUAL_MATCH=1 \
  QWEN_M3_PREFIX_REPLAY=1 \
  QWEN_ALLOW_DEFERRED_SEMANTIC_VISUAL_MATCH="$allow_deferred_semantic_visual_match" \
  QWEN_AUDIT_RAW_CONTENT=1 \
  QWEN_AUDIT_RAW_REQUEST=1 \
  QWEN_AUDIT_RAW_REQUEST_ON_MISS=1 \
  CAPX_CORE_FG=0 \
  CAPX_PGSVP_V9_VERIFY=0 \
  CAPX_ENVELOPE_REPLACEMENT=0 \
  CAPX_FGSRP=0 \
  CAPX_FGSRP_NL=0 \
  CAPX_CLOSED_GT_TAKEOVER=0 \
  CAPX_IDEAL_TK=0 \
  CAPX_IDEAL_TK_V4=1 \
  CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT=1 \
  CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT_MODE="$mode" \
  CAPX_PRIVILEGED_COACH=0 \
  CAPX_PRIVILEGED_COACH_BRIDGE=0 \
  CAPX_NODE_COACH=0 \
  CAPX_SEGMENTED_RECOVERY=0 \
  CAPX_CONTRACT_REPAIR=0 \
  CAPX_C1_NO_WORSE=0 \
  bash "$helper/run_qwen_multiturn.sbatch"
