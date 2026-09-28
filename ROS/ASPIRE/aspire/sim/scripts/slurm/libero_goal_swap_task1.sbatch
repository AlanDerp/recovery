#!/usr/bin/env bash
#SBATCH --job-name=libero-gs-task1
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/home/jtangbx/ASPIRE/aspire/sim/slurm_logs/%x-%j.log

set -euo pipefail

ASPIRE_ROOT=/home/jtangbx/ASPIRE/aspire/sim
PYTHON_ROOT=/home/jtangbx/ASPIRE
TASK=open_the_middle_drawer_of_the_cabinet
SUITE=libero_goal_swap
TASK_DIR="$ASPIRE_ROOT/outputs/libero_fix_loop/$SUITE/$TASK"
FIX_CODE="$TASK_DIR/fix_code.py"
PROMPT_FILE="$ASPIRE_ROOT/scripts/slurm/libero_goal_swap_task1.prompt.md"
CLAUDE_BIN=/home/jtangbx/.local/bin/claude
PERCEPTION_LOG_DIR="$ASPIRE_ROOT/slurm_logs/perception-$SLURM_JOB_ID"

cd "$ASPIRE_ROOT"
mkdir -p "$PERCEPTION_LOG_DIR"

export ASPIRE_ROOT PYTHON_ROOT
export PYTHONPATH="$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ASPIRE_PERCEPTION_PYTHON=.venv-libero/bin/python3
# In non-interactive print mode, Claude otherwise stops background subagents
# after its default 600-second wait ceiling.
export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0

SERVER_PIDS=()

cleanup() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${SERVER_PIDS[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

remaining_seconds() {
  local now
  now=$(date +%s)
  echo $((SLURM_JOB_END_TIME - now - 300))
}

run_before_deadline() {
  local remaining
  remaining=$(remaining_seconds)
  if (( remaining < 60 )); then
    echo "Less than 60 seconds remain before the cleanup window; resubmit the job."
    return 124
  fi
  timeout --signal=TERM --kill-after=30s "${remaining}s" "$@"
}

echo "Job $SLURM_JOB_ID on $(hostname), started $(date --iso-8601=seconds)"
echo "Repository commit: $(git rev-parse HEAD)"
echo "Allocated CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

.venv-libero/bin/python3 -c \
  'import torch, libero, robosuite, sam3, contact_graspnet_pytorch; assert torch.cuda.is_available(); print("LIBERO environment and CUDA are ready:", torch.__version__, torch.version.cuda)'

if [[ ! -f "$HOME/.cache/huggingface/token" ]]; then
  echo "Missing Hugging Face token file."
  exit 2
fi
if ! "$CLAUDE_BIN" auth status >/dev/null 2>&1; then
  echo "Claude Code authentication is unavailable."
  exit 2
fi

echo "Starting SAM3 and GraspNet on the allocated logical GPU 0; PyRoKi runs on CPU."
CUDA_VISIBLE_DEVICES=0 .venv-libero/bin/python3 -u \
  cap/serving/launch_sam3_server.py --device cuda --port 8114 --host 127.0.0.1 \
  >"$PERCEPTION_LOG_DIR/sam3.log" 2>&1 &
SERVER_PIDS+=("$!")

CUDA_VISIBLE_DEVICES=0 .venv-libero/bin/python3 -u \
  cap/serving/launch_contact_graspnet_server.py --port 8115 --host 127.0.0.1 \
  >"$PERCEPTION_LOG_DIR/graspnet.log" 2>&1 &
SERVER_PIDS+=("$!")

.venv-libero/bin/python3 -u \
  cap/serving/launch_pyroki_server.py --port 8116 --host 127.0.0.1 \
  --robot panda_description --target-link panda_hand \
  >"$PERCEPTION_LOG_DIR/pyroki.log" 2>&1 &
SERVER_PIDS+=("$!")

for attempt in $(seq 1 60); do
  ready=true
  for port in 8114 8115 8116; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
      "http://127.0.0.1:$port/health" 2>/dev/null || true)
    [[ -n "$code" && "$code" != 000 ]] || ready=false
  done
  if "$ready"; then
    break
  fi
  if (( attempt == 60 )); then
    echo "Perception services did not become ready."
    tail -30 "$PERCEPTION_LOG_DIR"/*.log
    exit 2
  fi
  sleep 5
done
echo "Perception services are ready."

if [[ ! -f "$FIX_CODE" ]]; then
  echo "Stage 1 is pending or partial; launching the Task 1 coordinator."
  set +e
  run_before_deadline "$CLAUDE_BIN" -p "$(<"$PROMPT_FILE")" \
    --model sonnet --effort high --dangerously-skip-permissions \
    --output-format stream-json --verbose \
    2>&1 | tee "$ASPIRE_ROOT/slurm_logs/claude-task1-$SLURM_JOB_ID.jsonl"
  agent_rc=${PIPESTATUS[0]}
  set -e
  echo "Coordinator exit code: $agent_rc"
fi

if [[ ! -f "$FIX_CODE" ]]; then
  echo "Stage 1 has not produced fix_code.py. Resubmit this same script to continue."
  exit 3
fi

if ! .venv/bin/python3 scripts/libero/record_skill_promotion.py verify \
  --suite "$SUITE" --task "$TASK"; then
  echo "Stage 1 exists but its skill-promotion record is not verified."
  echo "Resubmit so the coordinator can complete the Stage 1 handoff."
  exit 4
fi

echo "Stage 1 and promotion are complete; starting or resuming held-out seeds 1-50."
set +e
run_before_deadline .venv-libero/bin/python3 scripts/libero/run_fix_loop_validation.py \
  --suite "$SUITE" --task "$TASK" --gpu 0 \
  --fix-code "$FIX_CODE" \
  --output-dir outputs/libero_fix_loop_eval \
  --seeds $(seq 1 50) --resume
validation_rc=$?
set -e

echo "Validation exit code: $validation_rc"
.venv-libero/bin/python3 scripts/libero/gen_progress.py --print | \
  sed -n '/## libero_goal_swap/,/## libero_goal_task/p'

if (( validation_rc == 124 || validation_rc == 137 )); then
  echo "The two-hour slice ended before validation completed; resubmit this script."
  exit 0
fi
exit "$validation_rc"
