#!/usr/bin/env bash
#SBATCH --job-name=astar-instance-bound-10t
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/qshou/capx_paper_verifier_results/logs/astar-instance-bound-10t-%j.out
#SBATCH --error=/home/qshou/capx_paper_verifier_results/logs/astar-instance-bound-10t-%j.out

set -euo pipefail

project=${PROJECT_DIR:-/home/qshou/capx_astar_effect_alignment}
submission_rev=${CAPX_SUBMISSION_GIT_REV:?Set CAPX_SUBMISSION_GIT_REV}
replay=${CAPX_ASTAR_STACK_REPLAY_FILE:-/home/qshou/capx_paper_verifier_results/manifests/astar_instance_bound_temporal_smoke_20260805/astar_v4_stack_registered_prefix_replay_v2.json}
expected_replay_sha=${CAPX_ASTAR_STACK_REPLAY_SHA256:-5e51a20b4ee508c0093964ff77128bd0124fe9aad6cb0fead77074917c599e17}
require_all_replay=${CAPX_REQUIRE_ALL_REQUEST_HASH_REPLAY:-0}
enable_normalized_replay=${CAPX_ENABLE_NORMALIZED_REPLAY:-1}
allow_deferred_semantic_visual_match=${CAPX_ALLOW_DEFERRED_SEMANTIC_VISUAL_MATCH:-0}
m3_prefix_replay=${CAPX_M3_PREFIX_REPLAY:-1}
task_name=${TASK_NAME:-cube_stack}
effect_mode=${CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT_MODE:-instance_bound_grasp_loss_only}
helper="$project/experiments/core_fg/runtime"
python_bin=${CAPX_PYTHON:-/project/peilab/quanxin/cap-x/.venv-main/bin/python}
result_root=${RESULT_ROOT:-/home/qshou/capx_paper_verifier_results/results/astar_instance_bound_10trial_20260805}
key_file=${QWEN_KEY_FILE:-/scratch/peilab/qshou/RoboTwin_Sandbox/cap-x/.qwenkey_sg_tmp}
global_seed=20260718
trials=10

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
case "$m3_prefix_replay" in
  0|1) ;;
  *)
    echo "CAPX_M3_PREFIX_REPLAY must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$task_name" in
  cube_stack|cube_restack) ;;
  *)
    echo "TASK_NAME must be cube_stack or cube_restack" >&2
    exit 2
    ;;
esac
case "$effect_mode" in
  instance_bound_grasp_loss_only|instance_bound_grasp_outcome_only|instance_bound_transport_failure_only|instance_bound_off_table_lift_regression_only|instance_bound_grasp_loss_plus_off_table_lift_regression) ;;
  *)
    echo "Unsupported instance-bound effect mode: $effect_mode" >&2
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
"$python_bin" - "$replay" "$global_seed" "$trials" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
global_seed = int(sys.argv[2])
trials = int(sys.argv[3])
package = json.loads(path.read_text(encoding="utf-8"))
entries = package.get("entries")
if (
    package.get("schema") != "capx_request_hash_replay_v1"
    or package.get("request_normalization_version")
    != "image_url_redacted_verifier_frame_removed_v2"
    or package.get("visual_fingerprint_schema")
    != "rgb_shape_bytes_sha256_v1"
    or package.get("visual_fingerprint_native_mode") != "M3"
    or package.get("visual_fingerprint_global_seed") != global_seed
    or package.get("visual_fingerprint_total_trials") != trials
    or package.get("missing_count") != 0
    or package.get("duplicate_request_conflicts")
    or not isinstance(entries, dict)
    or package.get("entries_count") != len(entries)
):
    raise SystemExit("Registered A* replay package identity is invalid")
for entry in entries.values():
    seed = entry.get("effective_request_seed")
    trial = entry.get("trial")
    turn = entry.get("turn")
    if (
        not isinstance(seed, int)
        or not isinstance(trial, int)
        or trial not in range(1, trials + 1)
        or not isinstance(turn, int)
        or turn < 0
        or trial != (seed - global_seed) // 1000
        or turn != seed - (global_seed + trial * 1000)
        or not isinstance(entry.get("request_image_rgb_sha256"), list)
    ):
        raise SystemExit("Registered A* replay entry provenance is invalid")
PY
if [[ ! -s "$key_file" ]]; then
  echo "Missing Qwen credential file: $key_file" >&2
  exit 4
fi

required_replay_seeds=""
for trial in $(seq 1 "$trials"); do
  query_seed_base=$((global_seed + trial * 1000))
  replay_pair="${query_seed_base}+$((query_seed_base + 1))"
  if [[ -z "$required_replay_seeds" ]]; then
    required_replay_seeds="$replay_pair"
  else
    required_replay_seeds="${required_replay_seeds}+${replay_pair}"
  fi
done

mkdir -p "$result_root" /home/qshou/capx_paper_verifier_results/logs

env \
  PROJECT_DIR="$project" \
  QWEN_HELPER_DIR="$helper" \
  CAPX_FROZEN_GIT_REV="$submission_rev" \
  CAPX_RUNTIME_SOURCE_ROOT="$project" \
  CAPX_PYTHON="$python_bin" \
  QWEN_KEY_FILE="$key_file" \
  QWEN_BASE_URL=https://ws-vpxr6jfor4t48uyo.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1 \
  QWEN_OUTPUT_BASE="$result_root" \
  MODE=M3 \
  MODEL=qwen3.7-plus \
  TEMPERATURE=0 \
  TOTAL_TRIALS="$trials" \
  NUM_WORKERS=1 \
  TASK_FILTER="$task_name" \
  CAPX_GLOBAL_SEED="$global_seed" \
  QWEN_REQUEST_SEED="$global_seed" \
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
  QWEN_M3_PREFIX_REPLAY="$m3_prefix_replay" \
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
  CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT_MODE="$effect_mode" \
  CAPX_PRIVILEGED_COACH=0 \
  CAPX_PRIVILEGED_COACH_BRIDGE=0 \
  CAPX_NODE_COACH=0 \
  CAPX_SEGMENTED_RECOVERY=0 \
  CAPX_CONTRACT_REPAIR=0 \
  CAPX_C1_NO_WORSE=0 \
  bash "$helper/run_qwen_multiturn.sbatch"
