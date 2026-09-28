#!/usr/bin/env bash
#SBATCH --job-name=capx-qwen-mt
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/qshou/RoboTwin_Sandbox/cap-x/qwen_runs_output/logs/%x-%j.out
#SBATCH --error=/home/qshou/RoboTwin_Sandbox/cap-x/qwen_runs_output/logs/%x-%j.out

set -euo pipefail

project=${PROJECT_DIR:-/project/peilab/quanxin/cap-x}
python_bin=${CAPX_PYTHON:-/project/peilab/quanxin/cap-x/.venv-main/bin/python}
helper_dir=${QWEN_HELPER_DIR:-/home/qshou/RoboTwin_Sandbox/cap-x/qwen_runs}
key_file=${QWEN_KEY_FILE:-/project/peilab/quanxin/cap-x/.qwenkey}
base_url=${QWEN_BASE_URL:-https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1}
model=${MODEL:-qwen3.7-plus}
mode=${MODE:?Set MODE=M1, M2, M3, or M4}
temperature=${TEMPERATURE:-0.7}
max_tokens=${CAPX_MAX_TOKENS:-}
total_trials=${TOTAL_TRIALS:-3}
num_workers=${NUM_WORKERS:-1}
upstream_concurrency=${QWEN_UPSTREAM_CONCURRENCY:-1}
enable_thinking=${QWEN_ENABLE_THINKING:-false}
global_seed=${CAPX_GLOBAL_SEED:-12345}
request_seed=${QWEN_REQUEST_SEED:-$global_seed}
replay_file=${QWEN_REPLAY_FILE:-}
request_hash_replay_file=${QWEN_REQUEST_HASH_REPLAY_FILE:-}
required_replay_seeds=${QWEN_REQUIRED_REPLAY_SEEDS:-}
require_all_request_hash_replay=${QWEN_REQUIRE_ALL_REQUEST_HASH_REPLAY:-0}
replay_legacy_runtime_roots=${QWEN_REPLAY_LEGACY_RUNTIME_ROOTS:-}
deterministic_graspnet=${CAPX_DETERMINISTIC_GRASPNET:-0}
if [[ -n "${CAPX_FIXED_EPISODE_SEED:-}" && -n "${CAPX_EPISODE_SEED_BASE:-}" ]]; then
  echo "CAPX_FIXED_EPISODE_SEED and CAPX_EPISODE_SEED_BASE are mutually exclusive" >&2
  exit 2
fi
if [[ -n "${QWEN_PROXY_PORT:-}" ]]; then
  llm_port=$QWEN_PROXY_PORT
else
  llm_port=$("$python_bin" - <<'PYPORT'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PYPORT
  )
fi
sam3_port=${CAPX_SAM3_PORT:-$((20000 + SLURM_JOB_ID % 10000))}
graspnet_port=${CAPX_GRASPNET_PORT:-$((30000 + SLURM_JOB_ID % 10000))}
pyroki_port=${CAPX_PYROKI_PORT:-$((40000 + SLURM_JOB_ID % 10000))}

molmo_root=/scratch/peilab/qshou/capx-molmo
molmo_venv="$molmo_root/venv"
molmo_model_path=/scratch/peilab/qshou/capx-molmo/cache/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b
sam3_checkpoint=/scratch/peilab/qshou/models/sam3/sam3.pt
sam3_source_root=${CAPX_SAM3_SOURCE_ROOT:-/scratch/peilab/qshou/capx_paper_verifier/repo/capx/third_party/sam3}
contact_graspnet_source_root=${CAPX_CONTACT_GRASPNET_SOURCE_ROOT:-/scratch/peilab/qshou/capx_paper_verifier/repo/capx/third_party/contact_graspnet_pytorch}
owlvit_model_path=${OWLVIT_MODEL_PATH:-/scratch/peilab/qshou/capx-services/hf-no-xet/models--google--owlv2-large-patch14-ensemble/snapshots/95e26936e865f87db1742128404b3c035d47d89d}
sam2_model_path=${SAM2_MODEL_PATH:-/scratch/peilab/qshou/capx-deepseek-official-s3-100-t0/run-458994/hf-no-xet/models--facebook--sam2.1-hiera-large/snapshots/665f8e2ad61cf5f53d65644ff27c8ee525124610}

mode_lc=$(echo "$mode" | tr '[:upper:]' '[:lower:]')
temp_tag=$(echo "$temperature" | sed 's/\./p/g')
trial_tag="${total_trials}trials"
output_base=${QWEN_OUTPUT_BASE:-/home/qshou/RoboTwin_Sandbox/cap-x/qwen_runs_output}
resume_from_job_id=${RESUME_FROM_JOB_ID:-}
capx_resume_idx=${CAPX_RESUME_IDX:-}
task_filter=${TASK_FILTER:-}
config_override=${CONFIG_OVERRIDE:-}
task_name_override=${TASK_NAME_OVERRIDE:-custom_task}
if [[ -z "$task_filter" && -n "${TASK_FILTER_PLUS:-}" ]]; then
  task_filter=${TASK_FILTER_PLUS//+/,}
fi
lift_docfix=${LIFT_DOCFIX:-0}
lift_m3_finish_guard=${LIFT_M3_FINISH_GUARD:-0}
run_root="$output_base/capx-qwen37-${mode_lc}-${trial_tag}-t${temp_tag}/run-${SLURM_JOB_ID}"
log_root="$output_base/logs"
runtime_workdir="$run_root/workdir"

mkdir -p "$run_root" "$runtime_workdir" "$log_root" "$molmo_root/logs" "$molmo_root/cache" "$molmo_root/uv-cache"
cd "$project"

# Preserve the exact intervention runtime used by this job.  This deliberately
# excludes credentials and model caches.
runtime_snapshot="$run_root/runtime_snapshot"
mkdir -p "$runtime_snapshot/python_sitecustomize"
cp "$helper_dir/coach_runtime.py" "$runtime_snapshot/coach_runtime.py"
cp "$helper_dir/python_sitecustomize/sitecustomize.py" \
  "$runtime_snapshot/python_sitecustomize/sitecustomize.py"
for runtime_file in \
  run_qwen_multiturn.sbatch qwen_proxy.py capx_reproducibility.py \
  deterministic_contact_graspnet_server.py; do
  if [[ -f "$helper_dir/$runtime_file" ]]; then
    cp "$helper_dir/$runtime_file" "$runtime_snapshot/$runtime_file"
  fi
done
coach_prompt_source=${CAPX_COACH_PROMPT_PATH:-$helper_dir/coach_prompt.md}
coach_schema_source=${CAPX_COACH_SCHEMA_PATH:-$helper_dir/coach_schema.json}
node_prompt_source=${CAPX_NODE_COACH_PROMPT_PATH:-$helper_dir/node_coach_prompt_privileged_cheating.md}
successor_prompt_source=${CAPX_NODE_COACH_SUCCESSOR_PROMPT_PATH:-$helper_dir/node_coach_prompt_successor_transaction.md}
cp "$coach_prompt_source" "$runtime_snapshot/coach_prompt.md"
cp "$coach_schema_source" "$runtime_snapshot/coach_schema.json"
cp "$node_prompt_source" "$runtime_snapshot/node_coach_prompt.md"
cp "$successor_prompt_source" "$runtime_snapshot/node_coach_successor_prompt.md"
export CAPX_COACH_PROMPT_PATH="$runtime_snapshot/coach_prompt.md"
export CAPX_COACH_SCHEMA_PATH="$runtime_snapshot/coach_schema.json"
export CAPX_NODE_COACH_PROMPT_PATH="$runtime_snapshot/node_coach_prompt.md"
export CAPX_NODE_COACH_SUCCESSOR_PROMPT_PATH="$runtime_snapshot/node_coach_successor_prompt.md"
if [[ "${CAPX_PGSVP_V9_VERIFY:-0}" == "1" ]]; then
  pgsvp_v9_prompt_source=${CAPX_PGSVP_V9_PROMPT_PATH:?}
  pgsvp_v9_schema_source=${CAPX_PGSVP_V9_SCHEMA_PATH:?}
  pgsvp_v9_bridge_source=${CAPX_PGSVP_V9_BRIDGE_PATH:?}
  cp "$pgsvp_v9_prompt_source" "$runtime_snapshot/pgsvp_v9_verifier_prompt.md"
  cp "$pgsvp_v9_schema_source" "$runtime_snapshot/pgsvp_v9_verifier_schema.json"
  cp "$pgsvp_v9_bridge_source" "$runtime_snapshot/pgsvp_v9_public_bridge.txt"
  export CAPX_PGSVP_V9_PROMPT_PATH="$runtime_snapshot/pgsvp_v9_verifier_prompt.md"
  export CAPX_PGSVP_V9_SCHEMA_PATH="$runtime_snapshot/pgsvp_v9_verifier_schema.json"
  export CAPX_PGSVP_V9_BRIDGE_PATH="$runtime_snapshot/pgsvp_v9_public_bridge.txt"
fi
if [[ "${CAPX_ENVELOPE_REPLACEMENT:-0}" == "1" ]]; then
  envelope_prompt_source=${CAPX_ENVELOPE_REPLACEMENT_PROMPT_PATH:?}
  envelope_schema_source=${CAPX_ENVELOPE_REPLACEMENT_SCHEMA_PATH:?}
  envelope_adjudicator_prompt_source=${CAPX_ENVELOPE_ADJUDICATOR_PROMPT_PATH:?}
  envelope_adjudicator_schema_source=${CAPX_ENVELOPE_ADJUDICATOR_SCHEMA_PATH:?}
  cp "$envelope_prompt_source" "$runtime_snapshot/envelope_replacement_verifier_prompt.md"
  cp "$envelope_schema_source" "$runtime_snapshot/envelope_replacement_verifier_schema.json"
  cp "$envelope_adjudicator_prompt_source" "$runtime_snapshot/envelope_replacement_adjudicator_prompt.md"
  cp "$envelope_adjudicator_schema_source" "$runtime_snapshot/envelope_replacement_adjudicator_schema.json"
  export CAPX_ENVELOPE_REPLACEMENT_PROMPT_PATH="$runtime_snapshot/envelope_replacement_verifier_prompt.md"
  export CAPX_ENVELOPE_REPLACEMENT_SCHEMA_PATH="$runtime_snapshot/envelope_replacement_verifier_schema.json"
  export CAPX_ENVELOPE_ADJUDICATOR_PROMPT_PATH="$runtime_snapshot/envelope_replacement_adjudicator_prompt.md"
  export CAPX_ENVELOPE_ADJUDICATOR_SCHEMA_PATH="$runtime_snapshot/envelope_replacement_adjudicator_schema.json"
fi
if [[ "${CAPX_FGSRP:-0}" == "1" ]]; then
  fgsrp_parser_prompt_source=${CAPX_FGSRP_IR_PARSER_PROMPT_PATH:?}
  fgsrp_parser_schema_source=${CAPX_FGSRP_IR_PARSER_SCHEMA_PATH:?}
  fgsrp_rewriter_prompt_source=${CAPX_FGSRP_IR_REWRITER_PROMPT_PATH:?}
  fgsrp_rewriter_schema_source=${CAPX_FGSRP_IR_REWRITER_SCHEMA_PATH:?}
  cp "$fgsrp_parser_prompt_source" "$runtime_snapshot/fgsrp_ir_parser_prompt.md"
  cp "$fgsrp_parser_schema_source" "$runtime_snapshot/fgsrp_ir_parser_schema.json"
  cp "$fgsrp_rewriter_prompt_source" "$runtime_snapshot/fgsrp_ir_rewriter_prompt.md"
  cp "$fgsrp_rewriter_schema_source" "$runtime_snapshot/fgsrp_ir_rewriter_schema.json"
  export CAPX_FGSRP_IR_PARSER_PROMPT_PATH="$runtime_snapshot/fgsrp_ir_parser_prompt.md"
  export CAPX_FGSRP_IR_PARSER_SCHEMA_PATH="$runtime_snapshot/fgsrp_ir_parser_schema.json"
  export CAPX_FGSRP_IR_REWRITER_PROMPT_PATH="$runtime_snapshot/fgsrp_ir_rewriter_prompt.md"
  export CAPX_FGSRP_IR_REWRITER_SCHEMA_PATH="$runtime_snapshot/fgsrp_ir_rewriter_schema.json"
fi
if [[ "${CAPX_FGSRP_NL:-0}" == "1" ]]; then
  fgsrp_nl_prompt_source=${CAPX_FGSRP_NL_PROMPT_PATH:?}
  fgsrp_nl_schema_source=${CAPX_FGSRP_NL_SCHEMA_PATH:?}
  cp "$fgsrp_nl_prompt_source" "$runtime_snapshot/fgsrp_nl_prompt.md"
  cp "$fgsrp_nl_schema_source" "$runtime_snapshot/fgsrp_nl_schema.json"
  export CAPX_FGSRP_NL_PROMPT_PATH="$runtime_snapshot/fgsrp_nl_prompt.md"
  export CAPX_FGSRP_NL_SCHEMA_PATH="$runtime_snapshot/fgsrp_nl_schema.json"
fi
if [[ -n "$request_hash_replay_file" ]]; then
  cp "$request_hash_replay_file" "$runtime_snapshot/request_hash_replay.json"
  request_hash_replay_file="$runtime_snapshot/request_hash_replay.json"
fi
find "$runtime_snapshot" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$run_root/RUNTIME_SHA256SUMS.txt"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$global_seed"
export CAPX_GLOBAL_SEED="$global_seed"
export QWEN_REQUEST_SEED="$request_seed"
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export CAPX_MODEL_QUERY_TIMEOUT_SECONDS=${CAPX_MODEL_QUERY_TIMEOUT_SECONDS:-900}
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TMPDIR="$run_root/tmp"
export XDG_CACHE_HOME="$run_root/cache"
export MPLCONFIGDIR="$run_root/cache/matplotlib"
export HF_HOME="$run_root/hf-no-xet"
export TRANSFORMERS_CACHE="$run_root/hf-no-xet"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SAM3_CHECKPOINT_PATH="$sam3_checkpoint"
export CAPX_SAM3_SOURCE_ROOT="$sam3_source_root"
export CAPX_CONTACT_GRASPNET_SOURCE_ROOT="$contact_graspnet_source_root"
export PYTHONPATH="$runtime_snapshot/python_sitecustomize:$runtime_snapshot:$project:$helper_dir:$sam3_source_root:$contact_graspnet_source_root:$contact_graspnet_source_root/Pointnet_Pointnet2_pytorch:${CAPX_EXTRA_PYTHONPATH:-}"
export CAPX_COACH_SERVER_URL="http://127.0.0.1:$llm_port/chat/completions"
export CAPX_NODE_COACH_SERVER_URL="${CAPX_NODE_COACH_SERVER_URL:-http://127.0.0.1:$llm_port/chat/completions}"
export CAPX_NODE_COACH_TRACE_DIR="${CAPX_NODE_COACH_TRACE_DIR:-$run_root/node_coach_traces}"
export CAPX_PYROKI_PORT="$pyroki_port"
export CAPX_PYROKI_URL="http://127.0.0.1:$pyroki_port"
export SAM3_SERVICE_URL="http://127.0.0.1:$sam3_port"
export GRASPNET_SERVICE_URL="http://127.0.0.1:$graspnet_port"
export CAPX_REPRO_AUDIT_DIR="${CAPX_REPRO_AUDIT_DIR:-$run_root/repro_audit}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$HF_HOME"

pids=()
molmo_pid=""
cleanup_all() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  if [[ -n "${molmo_pid:-}" ]]; then kill "$molmo_pid" 2>/dev/null || true; fi
  for pid in "${pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  if [[ -n "${molmo_pid:-}" ]]; then wait "$molmo_pid" 2>/dev/null || true; fi
}
trap cleanup_all EXIT INT TERM

run_date=$(date -Is)
status_file="$run_root/git_status_porcelain.txt"
if [[ -n "${CAPX_FROZEN_GIT_REV:-}" ]]; then
  short_head=${CAPX_FROZEN_GIT_REV:0:7}
  : > "$status_file"
else
  short_head=$(git rev-parse --short HEAD)
  git status --porcelain > "$status_file"
fi

runtime_modules=(
  capx
  capx.envs.runner
  capx.envs.trial
  capx.utils.launch_utils
)
if [[ "${CAPX_PGSVP_V9_VERIFY:-0}" == "1" ]]; then
  runtime_modules+=(capx.verification.pgsvp_v9_verify)
fi
if [[ "${CAPX_ENVELOPE_REPLACEMENT:-0}" == "1" ]]; then
  runtime_modules+=(capx.verification.envelope_replacement)
fi
if [[ "${CAPX_FGSRP:-0}" == "1" ]]; then
  runtime_modules+=(capx.verification.fgsrp)
fi
if [[ "${CAPX_FGSRP_NL:-0}" == "1" ]]; then
  runtime_modules+=(capx.verification.fgsrp_nl)
fi
if [[
  "${CAPX_CLOSED_GT_TAKEOVER:-0}" == "1"
  || "${CAPX_CLOSED_GT_PASSIVE_CAPTURE:-0}" == "1"
]]; then
  runtime_modules+=(capx.verification.closed_gt capx.verification.closed_gt_takeover)
fi
if [[ "${CAPX_IDEAL_TK:-0}" == "1" ]]; then
  runtime_modules+=(capx.verification.closed_gt capx.verification.ideal_tk)
fi
if [[ "${CAPX_IDEAL_TK_V4:-0}" == "1" ]]; then
  runtime_modules+=(
    capx.verification.closed_gt
    capx.verification.closed_gt_takeover
    capx.verification.ideal_tk_v4
  )
fi
"$python_bin" "$helper_dir/tools/verify_runtime_imports.py" \
  --project "${CAPX_RUNTIME_SOURCE_ROOT:-$project}" \
  --output "$run_root/RUNTIME_IMPORT_AUDIT.json" \
  "${runtime_modules[@]}"
ln -s "$project/capx" "$runtime_workdir/capx"
cd "$runtime_workdir"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID"
echo "[$(date '+%F %T')] project=$project"
echo "[$(date '+%F %T')] runtime_workdir=$runtime_workdir"
echo "[$(date '+%F %T')] run_root=$run_root"
echo "[$(date '+%F %T')] mode=$mode model=$model temperature=$temperature total_trials=$total_trials num_workers=$num_workers upstream_concurrency=$upstream_concurrency"
echo "[$(date '+%F %T')] qwen_enable_thinking=$enable_thinking"
echo "[$(date '+%F %T')] git=$short_head dirty_lines=$(wc -l < "$status_file")"

cat > "$run_root/RUN_CONTEXT.txt" <<CTX
run_date: $run_date
slurm_job_id: $SLURM_JOB_ID
node: $(hostname)
project: $project
git_rev_short: $short_head
git_status_porcelain_file: $status_file
provider: qwen_dashscope_compatible
base_url: $base_url
model: $model
mode: $mode
temperature: $temperature
max_tokens: ${max_tokens:-default}
total_trials: $total_trials
num_workers: $num_workers
upstream_concurrency: $upstream_concurrency
qwen_enable_thinking: $enable_thinking
qwen_proxy_port: $llm_port
qwen_audit_raw_content: ${QWEN_AUDIT_RAW_CONTENT:-0}
qwen_audit_raw_request: ${QWEN_AUDIT_RAW_REQUEST:-0}
qwen_audit_raw_request_on_miss: ${QWEN_AUDIT_RAW_REQUEST_ON_MISS:-0}
qwen_enable_normalized_replay: ${QWEN_ENABLE_NORMALIZED_REPLAY:-0}
qwen_require_all_request_hash_replay: $require_all_request_hash_replay
qwen_require_normalized_replay_visual_match: ${QWEN_REQUIRE_NORMALIZED_REPLAY_VISUAL_MATCH:-0}
qwen_replay_legacy_runtime_roots: ${replay_legacy_runtime_roots:-none}
runtime_snapshot: $runtime_snapshot
runtime_frozen_imports: true
runtime_source_root: ${CAPX_RUNTIME_SOURCE_ROOT:-$project}
runtime_workdir: $runtime_workdir
runtime_resource_link: $runtime_workdir/capx
runtime_import_audit: $run_root/RUNTIME_IMPORT_AUDIT.json
runtime_sha256s: $run_root/RUNTIME_SHA256SUMS.txt
capx_model_query_timeout_seconds: $CAPX_MODEL_QUERY_TIMEOUT_SECONDS
capx_trial_timeout_seconds: ${CAPX_TRIAL_TIMEOUT_SECONDS:-1000}
capx_process_isolated_trials: ${CAPX_PROCESS_ISOLATED_TRIALS:-0}
capx_process_bootstrap_retries: ${CAPX_PROCESS_BOOTSTRAP_RETRIES:-1}
capx_max_trial_retries: ${CAPX_MAX_TRIAL_RETRIES:-legacy}
capx_global_seed: $global_seed
capx_seed_torch: ${CAPX_SEED_TORCH:-0}
capx_deterministic_trial_seeding: ${CAPX_DETERMINISTIC_TRIAL_SEEDING:-0}
capx_repro_audit: ${CAPX_REPRO_AUDIT:-0}
capx_repro_audit_dir: $CAPX_REPRO_AUDIT_DIR
capx_fixed_episode_seed: ${CAPX_FIXED_EPISODE_SEED:-}
capx_episode_seed_base: ${CAPX_EPISODE_SEED_BASE:-}
capx_deterministic_graspnet: $deterministic_graspnet
capx_sam3_port: $sam3_port
capx_graspnet_port: $graspnet_port
capx_pyroki_port: $pyroki_port
capx_start_molmo: ${CAPX_START_MOLMO:-1}
capx_sam3_source_root: $sam3_source_root
capx_contact_graspnet_source_root: $contact_graspnet_source_root
capx_reset_api_episode_state: ${CAPX_RESET_API_EPISODE_STATE:-0}
qwen_request_seed: $request_seed
qwen_replay_file: ${replay_file:-none}
qwen_request_hash_replay_file: ${request_hash_replay_file:-none}
qwen_required_replay_seeds: ${required_replay_seeds:-none}
capx_preclose_gripper_feedback: ${CAPX_PRECLOSE_GRIPPER_FEEDBACK:-0}
capx_gripper_checkpoint_feedback: ${CAPX_GRIPPER_CHECKPOINT_FEEDBACK:-0}
capx_segmented_recovery: ${CAPX_SEGMENTED_RECOVERY:-0}
capx_segmented_recovery_json: ${CAPX_SEGMENTED_RECOVERY_JSON:-0}
capx_segmented_recovery_patch_mode: ${CAPX_SEGMENTED_RECOVERY_PATCH_MODE:-}
capx_multiturn_limit: ${CAPX_MULTITURN_LIMIT:-10}
capx_contract_debug: ${CAPX_CONTRACT_DEBUG:-0}
capx_contract_repair: ${CAPX_CONTRACT_REPAIR:-0}
capx_contract_primitive_poc: ${CAPX_CONTRACT_PRIMITIVE_POC:-0}
capx_memory_prior_path: ${CAPX_MEMORY_PRIOR_PATH:-}
capx_core_fg: ${CAPX_CORE_FG:-0}
capx_pgsvp_v9_verify: ${CAPX_PGSVP_V9_VERIFY:-0}
capx_pgsvp_v9_prompt_path: ${CAPX_PGSVP_V9_PROMPT_PATH:-none}
capx_pgsvp_v9_schema_path: ${CAPX_PGSVP_V9_SCHEMA_PATH:-none}
capx_pgsvp_v9_bridge_path: ${CAPX_PGSVP_V9_BRIDGE_PATH:-none}
capx_pgsvp_v9_schema_attempts: ${CAPX_PGSVP_V9_SCHEMA_ATTEMPTS:-2}
capx_envelope_replacement: ${CAPX_ENVELOPE_REPLACEMENT:-0}
capx_envelope_replacement_prompt_path: ${CAPX_ENVELOPE_REPLACEMENT_PROMPT_PATH:-none}
capx_envelope_replacement_schema_path: ${CAPX_ENVELOPE_REPLACEMENT_SCHEMA_PATH:-none}
capx_envelope_replacement_schema_attempts: ${CAPX_ENVELOPE_REPLACEMENT_SCHEMA_ATTEMPTS:-2}
capx_fgsrp: ${CAPX_FGSRP:-0}
capx_fgsrp_carrier: ${CAPX_FGSRP_CARRIER:-none}
capx_fgsrp_ir_parser_prompt_path: ${CAPX_FGSRP_IR_PARSER_PROMPT_PATH:-none}
capx_fgsrp_ir_parser_schema_path: ${CAPX_FGSRP_IR_PARSER_SCHEMA_PATH:-none}
capx_fgsrp_ir_rewriter_prompt_path: ${CAPX_FGSRP_IR_REWRITER_PROMPT_PATH:-none}
capx_fgsrp_ir_rewriter_schema_path: ${CAPX_FGSRP_IR_REWRITER_SCHEMA_PATH:-none}
capx_fgsrp_schema_attempts: ${CAPX_FGSRP_SCHEMA_ATTEMPTS:-2}
capx_fgsrp_gate_attempts: ${CAPX_FGSRP_GATE_ATTEMPTS:-2}
capx_fgsrp_nl: ${CAPX_FGSRP_NL:-0}
capx_fgsrp_nl_prompt_path: ${CAPX_FGSRP_NL_PROMPT_PATH:-none}
capx_fgsrp_nl_schema_path: ${CAPX_FGSRP_NL_SCHEMA_PATH:-none}
capx_fgsrp_nl_schema_attempts: ${CAPX_FGSRP_NL_SCHEMA_ATTEMPTS:-2}
capx_closed_gt_takeover: ${CAPX_CLOSED_GT_TAKEOVER:-0}
capx_closed_gt_passive_capture: ${CAPX_CLOSED_GT_PASSIVE_CAPTURE:-0}
capx_ideal_tk: ${CAPX_IDEAL_TK:-0}
capx_ideal_tk_v4: ${CAPX_IDEAL_TK_V4:-0}
capx_ideal_tk_v4_effect_alignment: ${CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT:-0}
capx_ideal_tk_v4_effect_alignment_mode: ${CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT_MODE:-contradictions}
capx_privileged_coach: ${CAPX_PRIVILEGED_COACH:-0}
capx_privileged_coach_bridge: ${CAPX_PRIVILEGED_COACH_BRIDGE:-0}
capx_coach_model: ${CAPX_COACH_MODEL:-$model}
capx_coach_prompt_path: ${CAPX_COACH_PROMPT_PATH:-$helper_dir/coach_prompt.md}
capx_coach_schema_path: ${CAPX_COACH_SCHEMA_PATH:-$helper_dir/coach_schema.json}
capx_coach_cheating: ${CAPX_COACH_CHEATING:-0}
capx_coach_memory_freeze: ${CAPX_COACH_MEMORY_FREEZE:-0}
capx_coach_privileged_state_mode: ${CAPX_COACH_PRIVILEGED_STATE_MODE:-subset}
capx_coach_temperature: ${CAPX_COACH_TEMPERATURE:-0}
capx_coach_max_tokens: ${CAPX_COACH_MAX_TOKENS:-1200}
capx_node_coach: ${CAPX_NODE_COACH:-0}
capx_node_coach_model: ${CAPX_NODE_COACH_MODEL:-${CAPX_COACH_MODEL:-$model}}
capx_node_coach_prompt_path: ${CAPX_NODE_COACH_PROMPT_PATH:-$helper_dir/node_coach_prompt_privileged_cheating.md}
capx_node_coach_max_calls: ${CAPX_NODE_COACH_MAX_CALLS:-20}
capx_node_coach_trace_dir: $CAPX_NODE_COACH_TRACE_DIR
capx_node_coach_action_pattern: ${CAPX_NODE_COACH_ACTION_PATTERN:-default}
capx_node_coach_temperature: ${CAPX_NODE_COACH_TEMPERATURE:-0}
capx_node_coach_max_tokens: ${CAPX_NODE_COACH_MAX_TOKENS:-1500}
capx_node_coach_compact_packet: ${CAPX_NODE_COACH_COMPACT_PACKET:-0}
capx_node_coach_parse_retries: ${CAPX_NODE_COACH_PARSE_RETRIES:-0}
capx_node_coach_inline_resume: ${CAPX_NODE_COACH_INLINE_RESUME:-0}
capx_node_resume_max_repairs: ${CAPX_NODE_RESUME_MAX_REPAIRS:-10}
capx_c1_success_freeze: ${CAPX_C1_SUCCESS_FREEZE:-0}
capx_c1_transaction_ledger: ${CAPX_C1_TRANSACTION_LEDGER:-0}
capx_local_repair_model: ${CAPX_LOCAL_REPAIR_MODEL:-$model}
capx_local_repair_temperature: ${CAPX_LOCAL_REPAIR_TEMPERATURE:-0}
capx_local_repair_max_tokens: ${CAPX_LOCAL_REPAIR_MAX_TOKENS:-1600}
capx_local_repair_parse_retries: ${CAPX_LOCAL_REPAIR_PARSE_RETRIES:-2}
capx_c1_motion_observe_only: ${CAPX_C1_MOTION_OBSERVE_ONLY:-0}
capx_c1_adaptive_motion_gate: ${CAPX_C1_ADAPTIVE_MOTION_GATE:-0}
capx_c1_causal_post_close_gate: ${CAPX_C1_CAUSAL_POST_CLOSE_GATE:-0}
capx_c1_abort_unverified_close: ${CAPX_C1_ABORT_UNVERIFIED_CLOSE:-0}
capx_c1_counterfactual_close_preview: ${CAPX_C1_COUNTERFACTUAL_CLOSE_PREVIEW:-0}
capx_c1_counterfactual_release_preview: ${CAPX_C1_COUNTERFACTUAL_RELEASE_PREVIEW:-0}
capx_c1_counterfactual_protected_motion_preview: ${CAPX_C1_COUNTERFACTUAL_PROTECTED_MOTION_PREVIEW:-0}
capx_c1_no_worse: ${CAPX_C1_NO_WORSE:-0}
capx_c1_successor_transaction_prompt: ${CAPX_C1_SUCCESSOR_TRANSACTION_PROMPT:-0}
capx_c1_local_retry_transaction_lock: ${CAPX_C1_LOCAL_RETRY_TRANSACTION_LOCK:-0}
capx_node_coach_successor_prompt_path: ${CAPX_NODE_COACH_SUCCESSOR_PROMPT_PATH:-$helper_dir/node_coach_prompt_successor_transaction.md}
capx_handover_cheating_initial_guidance: ${CAPX_HANDOVER_CHEATING_INITIAL_GUIDANCE:-0}
capx_enable_depth_patch: ${CAPX_ENABLE_DEPTH_PATCH:-0}
capx_patch_pointcloud_alignment: ${CAPX_PATCH_POINTCLOUD_ALIGNMENT:-0}
capx_robust_mask_obb: ${CAPX_ROBUST_MASK_OBB:-0}
record_video: true
output_root: $run_root
resume_from_job_id: ${resume_from_job_id:-none}
capx_resume_idx: ${capx_resume_idx:-none}
task_filter: ${task_filter:-all}
lift_docfix: $lift_docfix
lift_m3_finish_guard: $lift_m3_finish_guard
owlvit_model_path: $owlvit_model_path
sam2_model_path: $sam2_model_path
CTX

start_qwen_proxy() {
  if [[ -n "$replay_file" || -n "$request_hash_replay_file" ]]; then
    echo "[$(date '+%F %T')] Replay mode: skipping upstream Qwen preflight replay_file=${replay_file:-none} request_hash_replay_file=${request_hash_replay_file:-none}"
  else
  echo "[$(date '+%F %T')] Preflight: checking Qwen API connectivity"
  QWEN_KEY_FILE="$key_file" QWEN_BASE_URL="$base_url" MODEL="$model" QWEN_REQUEST_SEED="$request_seed" "$python_bin" - <<'PYPROBE'
from openai import OpenAI
import os
from pathlib import Path
key = next(line.strip() for line in Path(os.environ["QWEN_KEY_FILE"]).read_text().splitlines() if line.strip() and not line.startswith("#"))
client = OpenAI(api_key=key, base_url=os.environ["QWEN_BASE_URL"])
kwargs = dict(
    model=os.environ["MODEL"],
    messages=[{"role": "user", "content": "Return exactly: OK"}],
    max_tokens=16,
    temperature=0,
    timeout=120,
    extra_body={"enable_thinking": False},
)
seed = os.environ.get("QWEN_REQUEST_SEED")
if seed and int(seed) >= 0:
    kwargs["seed"] = int(seed)
r = client.chat.completions.create(**kwargs)
msg = r.choices[0].message.content
if not msg:
    raise RuntimeError(f"Qwen preflight returned empty content; finish_reason={r.choices[0].finish_reason}")
print(f"Qwen preflight OK: {msg[:32]!r}")
PYPROBE
  fi

  echo "[$(date '+%F %T')] Starting Qwen proxy on 127.0.0.1:$llm_port"
  proxy_args=(
    "$runtime_snapshot/qwen_proxy.py"
    --key-file "$key_file" \
    --base-url "$base_url" \
    --audit-log "$run_root/qwen_proxy_audit.jsonl" \
    --host 127.0.0.1 \
    --port "$llm_port" \
    --upstream-concurrency "$upstream_concurrency" \
    --enable-thinking "$enable_thinking" \
    --request-seed "$request_seed"
  )
  if [[ -n "$replay_file" ]]; then
    proxy_args+=(--replay-file "$replay_file")
  fi
  if [[ -n "$request_hash_replay_file" ]]; then
    proxy_args+=(--request-hash-replay-file "$request_hash_replay_file")
  fi
  if [[ -n "$required_replay_seeds" ]]; then
    proxy_args+=(--required-replay-seeds "$required_replay_seeds")
  fi
  if [[ "${QWEN_AUDIT_RAW_CONTENT:-0}" == "1" ]]; then
    proxy_args+=(--audit-raw-content)
  fi
  if [[ "${QWEN_AUDIT_RAW_REQUEST:-0}" == "1" ]]; then
    proxy_args+=(--audit-raw-request)
  fi
  if [[ "${QWEN_AUDIT_RAW_REQUEST_ON_MISS:-0}" == "1" ]]; then
    proxy_args+=(--audit-raw-request-on-miss)
  fi
  if [[ "${QWEN_ENABLE_NORMALIZED_REPLAY:-0}" == "1" ]]; then
    proxy_args+=(--enable-normalized-replay)
  fi
  if [[ "$require_all_request_hash_replay" == "1" ]]; then
    proxy_args+=(--require-all-request-hash-replay)
  fi
  if [[ "${QWEN_REQUIRE_NORMALIZED_REPLAY_VISUAL_MATCH:-0}" == "1" ]]; then
    proxy_args+=(--require-normalized-replay-visual-match)
  fi
  if [[ "${QWEN_M3_PREFIX_REPLAY:-0}" == "1" ]]; then
    proxy_args+=(--m3-prefix-replay)
  fi
  if [[ "${QWEN_ALLOW_DEFERRED_SEMANTIC_VISUAL_MATCH:-0}" == "1" ]]; then
    proxy_args+=(--allow-deferred-semantic-visual-match)
  fi
  if [[ -n "$replay_legacy_runtime_roots" ]]; then
    proxy_args+=(--replay-legacy-runtime-roots "$replay_legacy_runtime_roots")
  fi
  "$python_bin" "${proxy_args[@]}" \
    >"$run_root/qwen_proxy.log" 2>&1 &
  local proxy_pid=$!
  local proxy_ready=0
  pids+=("$proxy_pid")
  for _ in $(seq 1 60); do
    if ! kill -0 "$proxy_pid" 2>/dev/null; then
      echo "Qwen proxy exited before becoming ready" >&2
      cat "$run_root/qwen_proxy.log" >&2
      return 1
    fi
    if curl --noproxy '*' --fail --silent "http://127.0.0.1:$llm_port/health" >/dev/null; then
      proxy_ready=1
      break
    fi
    sleep 2
  done
  if [[ "$proxy_ready" != "1" ]]; then
    echo "Qwen proxy did not become ready" >&2
    cat "$run_root/qwen_proxy.log" >&2
    return 1
  fi
  sleep 0.2
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    echo "Qwen proxy exited after a foreign health endpoint answered" >&2
    cat "$run_root/qwen_proxy.log" >&2
    return 1
  fi
}

start_molmo() {
  if [[ -n "${molmo_pid:-}" ]] && kill -0 "$molmo_pid" 2>/dev/null; then return 0; fi
  local molmo_tmp="/tmp/capx-molmo-${SLURM_JOB_ID}"
  local molmo_ready_wait_seconds="${MOLMO_READY_WAIT_SECONDS:-900}"
  local molmo_ready_poll_seconds=2
  local molmo_ready_attempts=$(( (molmo_ready_wait_seconds + molmo_ready_poll_seconds - 1) / molmo_ready_poll_seconds ))
  mkdir -p "$molmo_tmp"
  echo "[$(date '+%F %T')] Starting Molmo on 127.0.0.1:8122 model=$molmo_model_path ready_timeout=${molmo_ready_wait_seconds}s"
  HF_HOME="$molmo_root/cache" HUGGINGFACE_HUB_CACHE="$molmo_root/cache/hub" HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$molmo_tmp" \
  "$molmo_venv/bin/python" -m capx.serving.vllm_server \
    --model "$molmo_model_path" \
    --served-model-name allenai/Molmo2-8B \
    --host 127.0.0.1 \
    --port 8122 \
    --max-model-len 16384 \
    --gpu-memory-utilization "${MOLMO_GPU_UTIL:-0.55}" \
    >"$run_root/molmo.log" 2>&1 &
  molmo_pid=$!
  for i in $(seq 1 "$molmo_ready_attempts"); do
    if curl --noproxy '*' --fail --silent http://127.0.0.1:8122/v1/models >/dev/null; then
      echo "[$(date '+%F %T')] Molmo ready"
      return 0
    fi
    if ! kill -0 "$molmo_pid" 2>/dev/null; then
      echo "Molmo exited during startup"
      tail -n 200 "$run_root/molmo.log" || true
      return 1
    fi
    if (( i % 30 == 0 )); then
      local waited=$(( i * molmo_ready_poll_seconds ))
      echo "[$(date '+%F %T')] waiting for Molmo ${waited}/${molmo_ready_wait_seconds}s"
      tail -n 20 "$run_root/molmo.log" || true
    fi
    sleep "$molmo_ready_poll_seconds"
  done
  echo "Molmo did not become ready within ${molmo_ready_wait_seconds}s"
  tail -n 200 "$run_root/molmo.log" || true
  kill "$molmo_pid" 2>/dev/null || true
  wait "$molmo_pid" 2>/dev/null || true
  molmo_pid=""
  return 1
}

write_manifest() {
  local name="$1" config="$2" out="$3" manifest="$4"
  cat > "$manifest" <<M
run_date: $(date -Is)
slurm_job_id: $SLURM_JOB_ID
node: $(hostname)
git_rev_short: $short_head
git_status_porcelain_file: $status_file
provider: qwen_dashscope_compatible
base_url: $base_url
model: $model
mode: $mode
temperature: $temperature
max_tokens: ${max_tokens:-default}
config_path: $config
total_trials: $total_trials
num_workers: $num_workers
upstream_concurrency: $upstream_concurrency
qwen_enable_thinking: $enable_thinking
capx_model_query_timeout_seconds: $CAPX_MODEL_QUERY_TIMEOUT_SECONDS
capx_global_seed: $global_seed
capx_seed_torch: ${CAPX_SEED_TORCH:-0}
capx_deterministic_trial_seeding: ${CAPX_DETERMINISTIC_TRIAL_SEEDING:-0}
capx_episode_seed_base: ${CAPX_EPISODE_SEED_BASE:-}
qwen_request_seed: $request_seed
qwen_replay_file: ${replay_file:-none}
qwen_request_hash_replay_file: ${request_hash_replay_file:-none}
qwen_required_replay_seeds: ${required_replay_seeds:-none}
capx_deterministic_graspnet: $deterministic_graspnet
capx_sam3_port: $sam3_port
capx_graspnet_port: $graspnet_port
capx_pyroki_port: $pyroki_port
capx_start_molmo: ${CAPX_START_MOLMO:-1}
capx_sam3_source_root: $sam3_source_root
capx_contact_graspnet_source_root: $contact_graspnet_source_root
capx_reset_api_episode_state: ${CAPX_RESET_API_EPISODE_STATE:-0}
capx_preclose_gripper_feedback: ${CAPX_PRECLOSE_GRIPPER_FEEDBACK:-0}
capx_gripper_checkpoint_feedback: ${CAPX_GRIPPER_CHECKPOINT_FEEDBACK:-0}
capx_segmented_recovery: ${CAPX_SEGMENTED_RECOVERY:-0}
capx_segmented_recovery_json: ${CAPX_SEGMENTED_RECOVERY_JSON:-0}
capx_segmented_recovery_patch_mode: ${CAPX_SEGMENTED_RECOVERY_PATCH_MODE:-}
capx_multiturn_limit: ${CAPX_MULTITURN_LIMIT:-10}
capx_contract_debug: ${CAPX_CONTRACT_DEBUG:-0}
capx_memory_prior_path: ${CAPX_MEMORY_PRIOR_PATH:-}
capx_core_fg: ${CAPX_CORE_FG:-0}
capx_pgsvp_v9_verify: ${CAPX_PGSVP_V9_VERIFY:-0}
capx_pgsvp_v9_prompt_path: ${CAPX_PGSVP_V9_PROMPT_PATH:-none}
capx_pgsvp_v9_schema_path: ${CAPX_PGSVP_V9_SCHEMA_PATH:-none}
capx_pgsvp_v9_schema_attempts: ${CAPX_PGSVP_V9_SCHEMA_ATTEMPTS:-2}
capx_envelope_replacement: ${CAPX_ENVELOPE_REPLACEMENT:-0}
capx_envelope_replacement_prompt_path: ${CAPX_ENVELOPE_REPLACEMENT_PROMPT_PATH:-none}
capx_envelope_replacement_schema_path: ${CAPX_ENVELOPE_REPLACEMENT_SCHEMA_PATH:-none}
capx_envelope_replacement_schema_attempts: ${CAPX_ENVELOPE_REPLACEMENT_SCHEMA_ATTEMPTS:-2}
capx_fgsrp: ${CAPX_FGSRP:-0}
capx_fgsrp_carrier: ${CAPX_FGSRP_CARRIER:-none}
capx_fgsrp_ir_parser_prompt_path: ${CAPX_FGSRP_IR_PARSER_PROMPT_PATH:-none}
capx_fgsrp_ir_parser_schema_path: ${CAPX_FGSRP_IR_PARSER_SCHEMA_PATH:-none}
capx_fgsrp_ir_rewriter_prompt_path: ${CAPX_FGSRP_IR_REWRITER_PROMPT_PATH:-none}
capx_fgsrp_ir_rewriter_schema_path: ${CAPX_FGSRP_IR_REWRITER_SCHEMA_PATH:-none}
capx_fgsrp_schema_attempts: ${CAPX_FGSRP_SCHEMA_ATTEMPTS:-2}
capx_fgsrp_gate_attempts: ${CAPX_FGSRP_GATE_ATTEMPTS:-2}
capx_fgsrp_nl: ${CAPX_FGSRP_NL:-0}
capx_fgsrp_nl_prompt_path: ${CAPX_FGSRP_NL_PROMPT_PATH:-none}
capx_fgsrp_nl_schema_path: ${CAPX_FGSRP_NL_SCHEMA_PATH:-none}
capx_fgsrp_nl_schema_attempts: ${CAPX_FGSRP_NL_SCHEMA_ATTEMPTS:-2}
capx_closed_gt_takeover: ${CAPX_CLOSED_GT_TAKEOVER:-0}
capx_closed_gt_passive_capture: ${CAPX_CLOSED_GT_PASSIVE_CAPTURE:-0}
capx_ideal_tk: ${CAPX_IDEAL_TK:-0}
capx_ideal_tk_v4: ${CAPX_IDEAL_TK_V4:-0}
capx_ideal_tk_v4_effect_alignment: ${CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT:-0}
capx_ideal_tk_v4_effect_alignment_mode: ${CAPX_IDEAL_TK_V4_EFFECT_ALIGNMENT_MODE:-contradictions}
capx_privileged_coach: ${CAPX_PRIVILEGED_COACH:-0}
capx_privileged_coach_bridge: ${CAPX_PRIVILEGED_COACH_BRIDGE:-0}
capx_coach_model: ${CAPX_COACH_MODEL:-$model}
capx_coach_prompt_path: ${CAPX_COACH_PROMPT_PATH:-$helper_dir/coach_prompt.md}
capx_coach_schema_path: ${CAPX_COACH_SCHEMA_PATH:-$helper_dir/coach_schema.json}
capx_coach_cheating: ${CAPX_COACH_CHEATING:-0}
capx_coach_memory_freeze: ${CAPX_COACH_MEMORY_FREEZE:-0}
capx_coach_privileged_state_mode: ${CAPX_COACH_PRIVILEGED_STATE_MODE:-subset}
capx_coach_temperature: ${CAPX_COACH_TEMPERATURE:-0}
capx_coach_max_tokens: ${CAPX_COACH_MAX_TOKENS:-1200}
capx_node_coach: ${CAPX_NODE_COACH:-0}
capx_node_coach_model: ${CAPX_NODE_COACH_MODEL:-${CAPX_COACH_MODEL:-$model}}
capx_node_coach_prompt_path: ${CAPX_NODE_COACH_PROMPT_PATH:-$helper_dir/node_coach_prompt_privileged_cheating.md}
capx_node_coach_max_calls: ${CAPX_NODE_COACH_MAX_CALLS:-20}
capx_node_coach_trace_dir: $CAPX_NODE_COACH_TRACE_DIR
capx_node_coach_action_pattern: ${CAPX_NODE_COACH_ACTION_PATTERN:-default}
capx_node_coach_temperature: ${CAPX_NODE_COACH_TEMPERATURE:-0}
capx_node_coach_max_tokens: ${CAPX_NODE_COACH_MAX_TOKENS:-1500}
capx_node_coach_compact_packet: ${CAPX_NODE_COACH_COMPACT_PACKET:-0}
capx_node_coach_parse_retries: ${CAPX_NODE_COACH_PARSE_RETRIES:-0}
capx_c1_motion_observe_only: ${CAPX_C1_MOTION_OBSERVE_ONLY:-0}
capx_c1_adaptive_motion_gate: ${CAPX_C1_ADAPTIVE_MOTION_GATE:-0}
capx_c1_causal_post_close_gate: ${CAPX_C1_CAUSAL_POST_CLOSE_GATE:-0}
capx_c1_abort_unverified_close: ${CAPX_C1_ABORT_UNVERIFIED_CLOSE:-0}
capx_c1_counterfactual_close_preview: ${CAPX_C1_COUNTERFACTUAL_CLOSE_PREVIEW:-0}
capx_c1_counterfactual_release_preview: ${CAPX_C1_COUNTERFACTUAL_RELEASE_PREVIEW:-0}
capx_c1_counterfactual_protected_motion_preview: ${CAPX_C1_COUNTERFACTUAL_PROTECTED_MOTION_PREVIEW:-0}
capx_c1_no_worse: ${CAPX_C1_NO_WORSE:-0}
capx_c1_successor_transaction_prompt: ${CAPX_C1_SUCCESSOR_TRANSACTION_PROMPT:-0}
capx_c1_local_retry_transaction_lock: ${CAPX_C1_LOCAL_RETRY_TRANSACTION_LOCK:-0}
capx_node_coach_successor_prompt_path: ${CAPX_NODE_COACH_SUCCESSOR_PROMPT_PATH:-$helper_dir/node_coach_prompt_successor_transaction.md}
capx_handover_cheating_initial_guidance: ${CAPX_HANDOVER_CHEATING_INITIAL_GUIDANCE:-0}
capx_enable_depth_patch: ${CAPX_ENABLE_DEPTH_PATCH:-0}
capx_patch_pointcloud_alignment: ${CAPX_PATCH_POINTCLOUD_ALIGNMENT:-0}
output_dir: $out
record_video: true
owlvit_model_path: $owlvit_model_path
sam2_model_path: $sam2_model_path
lift_docfix: $lift_docfix
lift_m3_finish_guard: $lift_m3_finish_guard
M
}

materialize_config() {
  local config="$1"
  local src="$project/$config"
  if [[ "$config" = /* ]]; then
    src="$config"
  fi
  local dst="$run_root/materialized_configs/$config"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  if [[ "$deterministic_graspnet" == "1" ]]; then
    sed -i \
      "s|capx.serving.launch_contact_graspnet_server.main|deterministic_contact_graspnet_server.main|g" \
      "$dst"
    echo "[$(date '+%F %T')] Enabled request-deterministic ContactGraspNet in $dst" >&2
  fi
  # Isolate every local API service per Slurm job. Fixed ports allow unrelated
  # jobs placed on the same node to silently share a server.
  sed -i -E "s|^([[:space:]]*port:[[:space:]]*)8114([[:space:]]*)$|\1${sam3_port}\2|" "$dst"
  sed -i -E "s|^([[:space:]]*port:[[:space:]]*)8115([[:space:]]*)$|\1${graspnet_port}\2|" "$dst"
  sed -i -E "s|^([[:space:]]*port:[[:space:]]*)8116([[:space:]]*)$|\1${pyroki_port}\2|" "$dst"
  if [[ "${FORCE_USE_ORACLE_CODE:-0}" == "1" ]]; then
    CONFIG_DST="$dst" "$python_bin" - <<'PYORACLEMODE'
from pathlib import Path
import os
import re

path = Path(os.environ["CONFIG_DST"])
text = path.read_text()
if re.search(r"(?m)^use_oracle_code:\s*", text):
    text = re.sub(r"(?m)^use_oracle_code:\s*.*$", "use_oracle_code: true", text)
else:
    text += "\nuse_oracle_code: true\n"
path.write_text(text)
PYORACLEMODE
    echo "[$(date '+%F %T')] Enabled declared env.oracle_code in $dst" >&2
  fi
  if [[ -n "${ORACLE_CODE_PATH:-}" ]]; then
    if [[ ! -f "$ORACLE_CODE_PATH" ]]; then
      echo "ORACLE_CODE_PATH does not exist: $ORACLE_CODE_PATH" >&2
      return 2
    fi
    CONFIG_DST="$dst" ORACLE_CODE_PATH="$ORACLE_CODE_PATH" "$python_bin" - <<'PYORACLECODE'
from pathlib import Path
import os

path = Path(os.environ["CONFIG_DST"])
code = Path(os.environ["ORACLE_CODE_PATH"]).read_text().rstrip()
text = path.read_text()
needle = "  cfg:\n"
if needle not in text:
    raise SystemExit(f"CodeExecEnvConfig cfg block not found in {path}")
block = "    oracle_code: |\n" + "\n".join("      " + line for line in code.splitlines()) + "\n"
text = text.replace(needle, needle + block, 1)
path.write_text(text)
PYORACLECODE
    echo "[$(date '+%F %T')] Injected ORACLE_CODE_PATH=$ORACLE_CODE_PATH into $dst" >&2
  fi
  if [[ -n "$capx_resume_idx" ]]; then
    CONFIG_DST="$dst" CAPX_RESUME_IDX="$capx_resume_idx" "$python_bin" - <<'PYRESUMEIDX'
from pathlib import Path
import os
import re

path = Path(os.environ["CONFIG_DST"])
resume_idx = int(os.environ["CAPX_RESUME_IDX"])
text = path.read_text()
if re.search(r"(?m)^resume_idx:\s*", text):
    text = re.sub(r"(?m)^resume_idx:\s*.*$", f"resume_idx: {resume_idx}", text)
else:
    text += f"\nresume_idx: {resume_idx}\n"
path.write_text(text)
PYRESUMEIDX
    echo "[$(date '+%F %T')] Applied CAPX_RESUME_IDX=$capx_resume_idx in $dst" >&2
  fi
  if [[ -d "$owlvit_model_path" ]]; then
    sed -i "s|model_name: google/owlv2-large-patch14-ensemble|model_name: $owlvit_model_path|g" "$dst"
  else
    echo "[$(date '+%F %T')] WARNING OWLVIT_MODEL_PATH missing: $owlvit_model_path; using original OWLv2 HF id" >&2
  fi
  if [[ -d "$sam2_model_path" ]]; then
    sed -i "s|model_name: facebook/sam2.1-hiera-large|model_name: $sam2_model_path|g" "$dst"
  else
    echo "[$(date '+%F %T')] WARNING SAM2_MODEL_PATH missing: $sam2_model_path; using original SAM2 HF id" >&2
  fi
  if [[ "$config" == *"two_arm_lift"* ]] && grep -q "capx.serving.launch_sam3_server.main" "$dst"; then
    echo "[$(date '+%F %T')] Rewriting two_arm_lift segmentation server from SAM3 to SAM2.1 in $dst" >&2
    "$python_bin" - "$dst" "$sam2_model_path" <<'PYSAM2'
from pathlib import Path
import sys

path = Path(sys.argv[1])
sam2_model_path = sys.argv[2]
text = path.read_text()
old = """  - _target_: capx.serving.launch_sam3_server.main
    device: cuda
    port: 8114
    host: 127.0.0.1"""
new = f"""  - _target_: capx.serving.launch_sam2_server.main
    device: cuda
    port: 8113
    host: 127.0.0.1
    model_name: {sam2_model_path}"""
if old not in text:
    raise SystemExit(f"Expected SAM3 server block not found in {path}")
path.write_text(text.replace(old, new))
PYSAM2
  fi
  if [[ "$config" == *"two_arm_lift"* && "$config" == *"reduced"* ]] && grep -q "FrankaControlApiReducedBimanual" "$dst"; then
    echo "[$(date '+%F %T')] Rewriting two_arm_lift reduced API to SAM2-backed FrankaTwoArmLiftApiReduced in $dst" >&2
    sed -i "s|FrankaControlApiReducedBimanual|FrankaTwoArmLiftApiReduced|g" "$dst"
  fi
  if [[ "$lift_docfix" == "1" && "$config" == *"two_arm_lift"* && "$config" != *"reduced"* ]]; then
    CONFIG_DST="$dst" "$python_bin" - <<'PYLIFTDOC'
from pathlib import Path
import os

path = Path(os.environ["CONFIG_DST"])
text = path.read_text()
needle = "Important grasp convention for this task"
if needle not in text:
    insert = """      - Important grasp convention for this task: `get_handle0_pos()` and `get_handle1_pos()` return visual bbox centers, not final pregrasp poses.
      - For this high-level handle-center API, use the following side-grasp strategy: set `gripper0_wxyz = np.array([0.7071, 0.7071, 0, 0])` and `gripper1_wxyz = np.array([0.7071, -0.7071, 0, 0])`.
      - Before closing the grippers, first approach handle0 with `handle0_pos[1] += 0.15` and handle1 with `handle1_pos[1] -= 0.15`, then move back to the original bbox centers and close both grippers.
      - Lift both arms simultaneously after closing the grippers.
"""
    marker = "      You may write python code comments for reasoning"
    if marker not in text:
        raise SystemExit(f"Could not find prompt insertion marker in {path}")
    text = text.replace(marker, insert + "\n" + marker, 1)
    path.write_text(text)
PYLIFTDOC
    echo "[$(date '+%F %T')] Applied two_arm_lift multiturn prompt consistency fix in $dst" >&2
  fi
  if [[ "${CAPX_SEGMENTED_RECOVERY:-0}" == "1" ]]; then
    CONFIG_DST="$dst" "$python_bin" - <<'PYSEGREC'
from pathlib import Path
import os

path = Path(os.environ["CONFIG_DST"])
text = path.read_text()
marker = "CAPX_SEGMENTED_RECOVERY_PROMPT_V1"
if marker not in text:
    prompt_addition = f"""      # {marker}
      Segmented recovery discipline for this experiment:
      - Treat the task as one or more manipulation sections.
      - At the start, the gripper should be open. If it may be closed, explicitly open it before approaching.
      - Each section must follow this narrative structure in comments and code:
        # SECTION <n>: <short subtask name>
        # SEGMENT <n>.1 align_grasp
        <move/rotate/approach until the gripper is correctly aligned with the target object before grasping>
        # END SEGMENT <n>.1 align_grasp
        # SEGMENT <n>.2 grasp
        <close the gripper only after align_grasp is complete>
        # END SEGMENT <n>.2 grasp
        # SEGMENT <n>.3 align_release
        <move/rotate/approach until the held object is correctly aligned with the release/place/insert/handover target>
        # END SEGMENT <n>.3 align_release
        # SEGMENT <n>.4 release
        <open/release the gripper only after align_release is complete>
        # END SEGMENT <n>.4 release
      - The SECTION, SEGMENT, and END SEGMENT labels must be Python comments beginning with `#`; never write them as bare text.
      - Heuristic: always approach first; move to the correct position before grasping or placing; then close/open the gripper.
"""
    regen_addition = f"""
      # {marker}
      Segmented recovery decision protocol:
      - First identify the earliest failed section and segment from the visual feedback, stdout/stderr, and checkpoint images.
      - In your reasoning text before the final decision, include:
        FAILED_SECTION: <section id or unknown>
        FAILED_SEGMENT: align_grasp | grasp | align_release | release | unknown
        FAILURE_REASON: <one concise sentence>
        PATCH_SCOPE: only the failed segment unless an earlier segment must be corrected
      - If regenerating, preserve the section/segment labels and only modify the earliest failed segment or its necessary setup.
      - In segment patch mode, return only the target segment replacement code, not a full script. The runner will replace only that segment.
      - Do not rewrite unrelated later sections when an earlier segment is wrong; later sections depend on earlier sections being correct.
      - Continue the repeat-until-success process until the task is complete or the configured multi-turn limit is reached.
"""
    prompt_key = "    prompt: |"
    mt_key = "    multi_turn_prompt: |"
    if prompt_key not in text or mt_key not in text:
        raise SystemExit(f"Expected prompt keys not found in {path}")
    text = text.replace(prompt_key, prompt_key + "\n" + prompt_addition, 1)
    text = text.replace(mt_key, mt_key + "\n" + regen_addition, 1)
    if "save_multiturn_prompts:" not in text:
        text += "\nsave_multiturn_prompts: true\n"
    path.write_text(text)
PYSEGREC
    echo "[$(date '+%F %T')] Applied segmented recovery prompt v1 in $dst" >&2
  fi
  if [[ "${CAPX_CONTRACT_DEBUG:-0}" == "1" ]]; then
    CONFIG_DST="$dst" "$python_bin" - <<'PYCONTRACTCFG'
from pathlib import Path
import os

path = Path(os.environ["CONFIG_DST"])
text = path.read_text()
if "save_multiturn_prompts:" not in text:
    text += "\nsave_multiturn_prompts: true\n"
path.write_text(text)
PYCONTRACTCFG
    echo "[$(date '+%F %T')] Enabled contract debug artifact saving in $dst" >&2
  fi
  if [[ -n "${CAPX_MEMORY_PRIOR_PATH:-}" ]]; then
    if [[ ! -f "${CAPX_MEMORY_PRIOR_PATH}" ]]; then
      echo "CAPX_MEMORY_PRIOR_PATH does not exist: ${CAPX_MEMORY_PRIOR_PATH}" >&2
      return 2
    fi
    CONFIG_DST="$dst" CAPX_MEMORY_PRIOR_PATH="${CAPX_MEMORY_PRIOR_PATH}" "$python_bin" - <<'PYMEMORYPRIOR'
from pathlib import Path
import os

path = Path(os.environ["CONFIG_DST"])
prior_path = Path(os.environ["CAPX_MEMORY_PRIOR_PATH"])
prior = prior_path.read_text(errors="ignore").strip()
if not prior:
    raise SystemExit(f"Empty CAPX_MEMORY_PRIOR_PATH: {prior_path}")

text = path.read_text()
marker = "CAPX_SANITIZED_STRATEGY_PRIOR_V1"
if marker not in text:
    prior_block = f"""      # {marker}
      Sanitized strategy prior from previous audited counterfactual experience:
      The following memory has been filtered to remove trial-specific coordinates, rewards, paths, raw code, and hard-coded spoilers.
      Read it carefully before writing or regenerating code. Use it only as general decision logic and recovery bias, not as a source of copied code or constants.
"""
    for line in prior.splitlines():
        prior_block += "      " + line + "\n"
    prior_block += "\n"

    regen_block = f"""      # {marker}
      Sanitized strategy prior for multiturn recovery:
      The following memory has been filtered to remove trial-specific coordinates, rewards, paths, raw code, and hard-coded spoilers.
      Read it carefully before deciding FINISH vs REGENERATE. Use it to identify the earliest failed manipulation stage and make local, observation-grounded corrections.
"""
    for line in prior.splitlines():
        regen_block += "      " + line + "\n"
    regen_block += "\n"

    prompt_key = "    prompt: |"
    mt_key = "    multi_turn_prompt: |"
    if prompt_key not in text or mt_key not in text:
        raise SystemExit(f"Expected prompt keys not found in {path}")
    text = text.replace(prompt_key, prompt_key + "\n" + prior_block, 1)
    text = text.replace(mt_key, mt_key + "\n" + regen_block, 1)
    if "save_multiturn_prompts:" not in text:
        text += "\nsave_multiturn_prompts: true\n"
    path.write_text(text)
PYMEMORYPRIOR
    echo "[$(date '+%F %T')] Injected sanitized memory prior from ${CAPX_MEMORY_PRIOR_PATH} into $dst" >&2
  fi
  if [[ "${CAPX_PRIVILEGED_COACH_BRIDGE:-0}" == "1" ]]; then
    CONFIG_DST="$dst" "$python_bin" - <<'PYCOACHBRIDGE'
from pathlib import Path
import os

path = Path(os.environ["CONFIG_DST"])
text = path.read_text()
marker = "CAPX_PRIVILEGED_COACH_BRIDGE_V1"
if marker not in text:
    bridge = f"""      # {marker}
      If the console stdout contains a section named SANITIZED_PRIVILEGED_COACH_GUIDANCE, treat it as authoritative diagnostic feedback from a privileged recovery coach.
      Purpose: the coach guidance exists to help you complete the task successfully through correct failure attribution, correct reasoning, and targeted recovery. Do not use it for blind trial-and-error.
      How to use it:
      - Use the coach guidance as the primary explanation of what went wrong in the previous attempt.
      - Prioritize the coach's earliest-failed-stage diagnosis and recovery scope.
      - Preserve the task goal, object identities, and any earlier stage that was already correct.
      - Modify only the failed stage and the directly dependent next step unless the coach explicitly says otherwise.
      - Translate the guidance into executable Python using only the public APIs listed in this prompt.
      - Re-query public perception APIs when current object state is needed. Do not hard-code numeric positions, poses, quaternions, rewards, or simulator state copied from stdout.
      - Do not use hidden simulator internals, private evaluator signals, or oracle-only fields.
      - Do not copy the coach text as code. Convert it into a clear action sequence.
      - Before FINISH, verify through visual feedback or public perception APIs that the task goal appears satisfied.
      - When the previous code has an incorrect manipulation order, write a complete corrected recovery block for the failed stage instead of making a tiny local edit that preserves the bad order.
      - For manipulation subtasks, use public API-level stage contracts: re-query needed object poses, approach/align before closing the gripper, close only at the grasp pose, lift or retreat safely before transport, align above the target before descending, open only at the intended release pose, retreat, then verify before FINISH.
      - If grasp sampling or perception fails, do not invent hidden state or hard-coded constants; recover by re-querying public perception, using a safe approach, and retrying the public API-level manipulation sequence.
      - If stderr mentions an episode limit, truncation, or executing an action in a terminated episode, do not write another long exploratory sequence. Write a minimal recovery block with only the necessary public API calls, avoid repeated full restarts, and verify the next irreversible action before taking it.
      If the coach guidance is brief or underspecified, still follow its failure attribution, but implement a complete safe recovery sequence: identify object identity, approach/align, grasp or release as appropriate, verify, transport/align, release/finish.

"""
    mt_key = "    multi_turn_prompt: |"
    if mt_key not in text:
        raise SystemExit(f"Expected multi_turn_prompt key not found in {path}")
    text = text.replace(mt_key, mt_key + "\n" + bridge, 1)
    if "save_multiturn_prompts:" not in text:
        text += "\nsave_multiturn_prompts: true\n"
    path.write_text(text)
PYCOACHBRIDGE
    echo "[$(date '+%F %T')] Injected privileged coach bridge into $dst" >&2
  fi
  if [[ "${CAPX_COACH_CHEATING:-0}" == "1" && "${CAPX_HANDOVER_CHEATING_INITIAL_GUIDANCE:-0}" == "1" && "$dst" == *"two_arm_handover"* ]]; then
    CONFIG_DST="$dst" "$python_bin" - <<'PYCHEATINGINIT'
from pathlib import Path
import os

path = Path(os.environ["CONFIG_DST"])
text = path.read_text()
marker = "CAPX_HANDOVER_PRIVILEGED_CHEATING_INITIAL_GUIDANCE_V1"
if marker not in text:
    block = f"""      # {marker}
      PRIVILEGED CHEATING UPPER-BOUND DIAGNOSTIC MODE:
      This run is not a clean baseline. Use the following verified handover recipe from the first code block; do not wait for failure recovery.
      Canonical successful sequence:
      - Query hammer_pos, hammer_quat, bbox_extent = get_object_pose("hammer", return_bbox_extent=True).
      - The hammer is initially on the table, approximately aligned along Y. Do NOT have Arm0 grasp the +Y handle end; Arm1 must later grasp the handle.
      - Arm0 should grasp slightly toward the head / negative-Y side from the hammer center: grasp_pos = hammer_pos + [0, -0.25 * bbox_extent[1], 0].
      - Arm0 grasp orientation: gripper facing down, opening along X: np.array([0, 0.707, 0.707, 0]).
      - Open Arm0, approach above, descend to grasp_pos, close Arm0, then lift about 0.10-0.15 m above the initial hammer z.
      - Move Arm0 stepwise to handover_x = 0.81, handover_y = 0.0. Keep safe intermediate height before lowering.
      - At handover, rotate Arm0 to gripper facing down opening along Y: np.array([0, 1, 0, 0]); this presents the handle along +X toward Arm1.
      - Handover z should be about 0.175 in the API coordinate frame.
      - Arm1 should grasp at handover_x + 0.10, handover_y, handover_z with quat np.array([0, 0, 1, 0]).
      - Open Arm1 before approach, close Arm1 at the grasp pose, wait briefly, then open Arm0.
      - Retract Arm0 away toward lower x and upward. Then lift Arm1 slightly. This exact order is important: Arm1 close before Arm0 release.
      - Do not write exploratory retries in the first block. Execute the complete recipe once, stepwise and concisely.

"""
    prompt_key = "    prompt: |"
    if prompt_key not in text:
        raise SystemExit(f"Expected prompt key not found in {path}")
    text = text.replace(prompt_key, prompt_key + "\n" + block, 1)
    if "save_multiturn_prompts:" not in text:
        text += "\nsave_multiturn_prompts: true\n"
    path.write_text(text)
PYCHEATINGINIT
    echo "[$(date '+%F %T')] Injected handover privileged cheating initial guidance into $dst" >&2
  fi
  echo "$dst"
}

run_task() {
  local name="$1" config="$2"
  if [[ -n "$resume_from_job_id" ]]; then
    local source_root="$output_base/capx-qwen37-${mode_lc}-${trial_tag}-t${temp_tag}/run-${resume_from_job_id}/qwen3.7-plus/$name"
    if [[ -f "$source_root/summaries.txt" ]]; then
      echo "[$(date '+%F %T')] SKIP $name: existing summary from run-$resume_from_job_id: $source_root/summaries.txt"
      return 0
    fi
  fi
  local out="$run_root/$name"
  local task_log="$run_root/${name}.log"
  local manifest="$run_root/${name}_manifest.txt"
  local launch_config
  launch_config=$(materialize_config "$config")
  mkdir -p "$out"
  write_manifest "$name" "$launch_config" "$out" "$manifest"
  echo "[$(date '+%F %T')] START $name config=$launch_config original_config=$config out=$out"
  local max_token_args=()
  if [[ -n "$max_tokens" ]]; then
    max_token_args=(--max-tokens "$max_tokens")
  fi
  set +e
  CAPX_CURRENT_TASK_NAME="$name" "$python_bin" -m capx.envs.launch \
    --config-path "$launch_config" \
    --model "$model" \
    --server-url "http://127.0.0.1:$llm_port/chat/completions" \
    --visual-differencing-model "$model" \
    --visual-differencing-model-server-url "http://127.0.0.1:$llm_port/chat/completions" \
    --temperature "$temperature" \
    "${max_token_args[@]}" \
    --total-trials "$total_trials" \
    --num-workers "$num_workers" \
    --record-video True \
    --output-dir "$out" \
    2>&1 | tee "$task_log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "[$(date '+%F %T')] END $name rc=$rc"
  local summary
  summary=$(find "$run_root" -maxdepth 8 -path "*/$name/summaries.txt" -print | sort | tail -1 || true)
  if [[ -n "$summary" ]]; then
    echo "summary_path: $summary" >> "$manifest"
    echo "[$(date '+%F %T')] SUMMARY $name $summary"
    sed -n '1,12p' "$summary"
  else
    echo "summary_path: MISSING" >> "$manifest"
    echo "[$(date '+%F %T')] WARNING $name produced no summaries.txt under $out"
  fi
  return "$rc"
}

case "$mode" in
  M1)
    tasks=(
      "cube_lift env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn.yaml"
      "cube_stack env_configs/cube_stack/franka_robosuite_cube_stack_multiturn.yaml"
      "spill_wipe env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn.yaml"
      "peg_insertion env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn.yaml"
      "cube_restack env_configs/cube_restack/franka_robosuite_cube_restack_multiturn.yaml"
      "two_arm_lift env_configs/two_arm_lift/franka_robosuite_two_arm_lift_multiturn.yaml"
      "two_arm_handover env_configs/two_arm_handover/two_arm_handover_multiturn.yaml"
    )
    ;;
  M2)
    tasks=(
      "cube_lift env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn_vf.yaml"
      "cube_stack env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vf.yaml"
      "spill_wipe env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vf.yaml"
      "peg_insertion env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vf.yaml"
      "cube_restack env_configs/cube_restack/franka_robosuite_cube_restack_multiturn_vf.yaml"
      "two_arm_lift env_configs/two_arm_lift/franka_robosuite_two_arm_lift_multiturn_vf.yaml"
      "two_arm_handover env_configs/two_arm_handover/two_arm_handover_multiturn_vf.yaml"
    )
    ;;
  M3)
    tasks=(
      "cube_lift env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn_vdm.yaml"
      "cube_stack env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml"
      "spill_wipe env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vdm.yaml"
      "peg_insertion env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vdm.yaml"
      "cube_restack env_configs/cube_restack/franka_robosuite_cube_restack_multiturn_vdm.yaml"
      "two_arm_lift env_configs/two_arm_lift/franka_robosuite_two_arm_lift_multiturn_vdm.yaml"
      "two_arm_handover env_configs/two_arm_handover/two_arm_handover_multiturn_vdm.yaml"
    )
    ;;
  M4)
    tasks=(
      "cube_lift env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn_vdm_reduced_api.yaml"
      "cube_stack env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_reduced_api.yaml"
      "spill_wipe env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vdm_reduced_api.yaml"
      "peg_insertion env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vdm_reduced_api.yaml"
      "cube_restack env_configs/cube_restack/franka_robosuite_cube_restack_multiturn_vdm_reduced_api.yaml"
      "two_arm_lift env_configs/two_arm_lift/franka_robosuite_two_arm_lift_multiturn_vdm_reduced_api.yaml"
      "two_arm_handover env_configs/two_arm_handover/hillclimb/two_arm_handover_multiturn_vdm_reduced_api.yaml"
    )
    ;;
  *)
    echo "Unsupported MODE=$mode; use M1, M2, M3, or M4" >&2
    exit 2
    ;;
esac

if [[ -n "$config_override" ]]; then
  tasks=("$task_name_override $config_override")
  echo "[$(date '+%F %T')] CONFIG_OVERRIDE=$config_override TASK_NAME_OVERRIDE=$task_name_override"
fi

if [[ -n "$task_filter" ]]; then
  filtered=()
  for item in "${tasks[@]}"; do
    name=${item%% *}
    if [[ ",$task_filter," == *",$name,"* ]]; then
      filtered+=("$item")
    fi
  done
  tasks=("${filtered[@]}")
  echo "[$(date '+%F %T')] TASK_FILTER=$task_filter selected ${#tasks[@]} task(s)"
fi

start_qwen_proxy
if [[ "${CAPX_START_MOLMO:-1}" == "1" ]]; then
  start_molmo
else
  echo "[$(date '+%F %T')] Molmo disabled by CAPX_START_MOLMO=0"
fi

failures=0
for item in "${tasks[@]}"; do
  name=${item%% *}
  config=${item#* }
  run_task "$name" "$config" || failures=$((failures+1))
done

echo "[$(date '+%F %T')] all Qwen $mode multiturn tasks finished failures=$failures"
find "$run_root" -maxdepth 6 -name summaries.txt -print | sort
exit "$failures"
