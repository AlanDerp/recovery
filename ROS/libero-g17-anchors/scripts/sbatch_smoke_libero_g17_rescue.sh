#!/usr/bin/env bash
#SBATCH --job-name=libero-g17-rescue
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_rescue_smoke_20260814/logs/%x-%j.out
#SBATCH --error=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_rescue_smoke_20260814/logs/%x-%j.out

# Paired rescue gain smoke vs frozen M3 baseline (job 517994 spatial_0).
# Treatment only: CAPX_LIBERO_G17_FG=1. Does NOT rewrite baseline sbatch/yaml/SEED.
set -euo pipefail

G17=/home/qshou/capx_g17_observation_integrity
PROJECT_DIR=/project/peilab/qshou/cap-x-agilex
BASELINE_EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813
EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_rescue_smoke_20260814
HELPER=/home/qshou/capx_causal_ik_replay_81e0acb/experiments/core_fg/runtime
PYTHON="$PROJECT_DIR/.venv-libero/bin/python"
KEY=/scratch/peilab/qshou/RoboTwin_Sandbox/cap-x/.qwenkey_sg_tmp
BASE_URL=https://ws-vpxr6jfor4t48uyo.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
MODEL=qwen3.7-plus
TOTAL_TRIALS=2
NUM_WORKERS=1
TEMPERATURE=0
GLOBAL_SEED=20260718
MOLMO_ROOT=/scratch/peilab/qshou/capx-molmo
MOLMO_VENV="$MOLMO_ROOT/venv"
MOLMO_MODEL=/scratch/peilab/qshou/capx-molmo/cache/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b
MOLMO_GPU_UTIL=${MOLMO_GPU_UTIL:-0.55}

suite_name=libero_spatial
task_id=0
task_name="${suite_name}_${task_id}"
suite_root="$EXPT_DIR/runs/${suite_name}/job-${SLURM_JOB_ID}"
mkdir -p "$suite_root" "$EXPT_DIR/logs"

# Baseline fingerprint gate (read-only).
"$PYTHON" - <<'PY'
import hashlib
from pathlib import Path
expected = {
    Path("/project/peilab/qshou/cap-x-agilex/scripts/sbatch_libero_m3_3x10x10.sh"):
        "1728358749e5e2fe916011e4578e81b15c1ffda55ec07c116c36f0a2b1794ef1",
    Path("/project/peilab/qshou/cap-x-agilex/env_configs/libero/franka_libero_m3.yaml"):
        "befb7ff390ad619ac4c6a0201c470ae884376206de666dd7cfba2bb3787a4939",
    Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813/SEED_TABLE.json"):
        "4fec7669be41d3c6fbff096f1ca587052a7253c7af6a9e0a619eb0854f65c24d",
}
for path, want in expected.items():
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != want:
        raise SystemExit(f"baseline drift: {path} got {h}")
print("baseline fingerprints ok")
PY

export TMPDIR="$suite_root/tmp"
export MPLCONFIGDIR="$suite_root/mpl"
export UV_CACHE_DIR=/scratch/peilab/qshou/uv-cache
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/.venv-libero"
export LIBERO_CONFIG_PATH="$PROJECT_DIR/.libero"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$GLOBAL_SEED"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export SAM3_CHECKPOINT_PATH=/scratch/peilab/qshou/models/sam3/sam3.pt
export CAPX_GLOBAL_SEED="$GLOBAL_SEED"
export QWEN_REQUEST_SEED="$GLOBAL_SEED"
export CAPX_DETERMINISTIC_TRIAL_SEEDING=1
export CAPX_RESET_API_EPISODE_STATE=1
export CAPX_PROCESS_ISOLATED_TRIALS=1
export CAPX_PROCESS_BOOTSTRAP_RETRIES=1
export CAPX_MAX_TRIAL_RETRIES=0
export CAPX_TRIAL_TIMEOUT_SECONDS=1000
export CAPX_REPRO_AUDIT=1
export CAPX_REPRO_AUDIT_DIR="$suite_root/repro_audit"
export CAPX_SEED_TORCH=1
export CAPX_CURRENT_TASK_NAME="$task_name"
# Treatment: LIBERO G17 FG only. Keep Ideal-TK / cube takeover off.
export CAPX_LIBERO_G17_FG=1
export CAPX_IDEAL_TK_V4=0
export CAPX_IDEAL_TK=0
export CAPX_CLOSED_GT_TAKEOVER=0
export CAPX_FGSRP=0
export CAPX_FGSRP_NL=0
export CAPX_CORE_FG=0
unset PYOPENGL_PLATFORM
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CAPX_REPRO_AUDIT_DIR"

# G17 worktree first so trial.py + verification takeover win; still uses .venv-libero.
export PYTHONPATH="$G17:$HELPER/python_sitecustomize:$HELPER:$PROJECT_DIR:${PYTHONPATH:-}"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv-libero/bin/activate"

[[ -x "$PYTHON" ]]
[[ -s "$KEY" ]]
[[ -s "$SAM3_CHECKPOINT_PATH" ]]
[[ -f "$HELPER/qwen_proxy.py" ]]
[[ -x "$MOLMO_VENV/bin/python" ]]
[[ -d "$MOLMO_MODEL" ]]
[[ -f "$BASELINE_EXPT/SEED_TABLE.json" ]]

sam3_port=$((20000 + SLURM_JOB_ID % 10000))
graspnet_port=$((30000 + SLURM_JOB_ID % 10000))
pyroki_port=$((40000 + SLURM_JOB_ID % 10000))
molmo_port=$((50000 + SLURM_JOB_ID % 10000))
export SAM3_SERVICE_URL="http://127.0.0.1:${sam3_port}"
export GRASPNET_SERVICE_URL="http://127.0.0.1:${graspnet_port}"
export CAPX_PYROKI_URL="http://127.0.0.1:${pyroki_port}"
export MOLMO_SERVICE_URL="http://127.0.0.1:${molmo_port}/v1"

llm_port=$("$PYTHON" - <<'PYPORT'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PYPORT
)

cleanup() {
  for pid_var in proxy_pid molmo_pid; do
    pid="${!pid_var:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  rm -rf "/tmp/capx-molmo-${SLURM_JOB_ID:-}" 2>/dev/null || true
}
trap cleanup EXIT

echo "host=$(hostname) job=$SLURM_JOB_ID date=$(date -Is)"
echo "treatment=CAPX_LIBERO_G17_FG suite=$suite_name task=$task_id trials=1..$TOTAL_TRIALS"
echo "pair_control=baseline job-517994 libero_spatial_0 (trial1 fail / trial2 success)"
nvidia-smi -L | head -3

echo "[$(date '+%F %T')] Qwen preflight"
QWEN_KEY_FILE="$KEY" QWEN_BASE_URL="$BASE_URL" MODEL="$MODEL" "$PYTHON" - <<'PYPROBE'
from openai import OpenAI
import os
from pathlib import Path
key = next(line.strip() for line in Path(os.environ["QWEN_KEY_FILE"]).read_text().splitlines() if line.strip() and not line.startswith("#"))
client = OpenAI(api_key=key, base_url=os.environ["QWEN_BASE_URL"])
r = client.chat.completions.create(model=os.environ["MODEL"], messages=[{"role":"user","content":"ping"}], max_tokens=8, temperature=0)
print("qwen_ok", r.choices[0].message.content)
PYPROBE

echo "[$(date '+%F %T')] start qwen proxy on $llm_port"
QWEN_KEY_FILE="$KEY" QWEN_BASE_URL="$BASE_URL" \
"$PYTHON" "$HELPER/qwen_proxy.py" --port "$llm_port" --model "$MODEL" \
  >"$suite_root/qwen_proxy.log" 2>&1 &
proxy_pid=$!
for i in $(seq 1 60); do
  if curl --noproxy '*' --fail --silent "http://127.0.0.1:$llm_port/healthz" >/dev/null 2>&1 \
     || curl --noproxy '*' --fail --silent "http://127.0.0.1:$llm_port/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Start local API servers via launch's config ports — same as M3 sbatch pattern.
echo "[$(date '+%F %T')] start Molmo"
molmo_tmp="/tmp/capx-molmo-${SLURM_JOB_ID}"
mkdir -p "$molmo_tmp"
pushd "$molmo_tmp" >/dev/null
HF_HOME="$MOLMO_ROOT/cache" \
HUGGINGFACE_HUB_CACHE="$MOLMO_ROOT/cache/hub" \
HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TMPDIR="$molmo_tmp" PYTHONPATH=/project/peilab/quanxin/cap-x \
"$MOLMO_VENV/bin/python" /project/peilab/quanxin/cap-x/capx/serving/vllm_server.py \
  --model "$MOLMO_MODEL" \
  --served-model-name allenai/Molmo2-8B \
  --host 127.0.0.1 \
  --port "$molmo_port" \
  --max-model-len 16384 \
  --gpu-memory-utilization "$MOLMO_GPU_UTIL" \
  >"$suite_root/molmo.log" 2>&1 &
molmo_pid=$!
popd >/dev/null

molmo_ready=0
for i in $(seq 1 450); do
  if ! kill -0 "$molmo_pid" 2>/dev/null; then
    echo "Molmo exited during startup" >&2
    tail -n 200 "$suite_root/molmo.log" >&2 || true
    exit 1
  fi
  if curl --noproxy '*' --fail --silent "http://127.0.0.1:$molmo_port/v1/models" >/dev/null; then
    molmo_ready=1
    break
  fi
  sleep 2
done
if [[ "$molmo_ready" != "1" ]]; then
  echo "Molmo did not become ready" >&2
  exit 1
fi
echo "[$(date '+%F %T')] Molmo ready"

task_root="$suite_root/$task_name"
mkdir -p "$task_root" "$task_root/repro_audit"
export CAPX_REPRO_AUDIT_DIR="$task_root/repro_audit"
config_path="$task_root/config.yaml"
"$PYTHON" - <<PYCFG
from pathlib import Path
src = Path("$PROJECT_DIR/env_configs/libero/franka_libero_m3.yaml")
dst = Path("$config_path")
text = src.read_text()
text = text.replace("suite_name: libero_spatial", "suite_name: $suite_name", 1)
text = text.replace("task_id: 0", "task_id: $task_id", 1)
text = text.replace("port: 8116", "port: $pyroki_port", 1)
text = text.replace("port: 8115", "port: $graspnet_port", 1)
text = text.replace("port: 8114", "port: $sam3_port", 1)
text = text.replace("output_dir: ./outputs/franka_libero_m3", f"output_dir: {dst.parent / 'eval'}")
text = text.replace("trials: 10", "trials: $TOTAL_TRIALS", 1)
dst.write_text(text)
print(dst)
PYCFG

cat > "$suite_root/RUN_CONTEXT.txt" <<CTX
mode: libero_g17_fg_rescue_smoke
pair_control: $BASELINE_EXPT/runs/libero_spatial/job-517994/libero_spatial_0
suite: $suite_name
task_id: $task_id
trials: 1..$TOTAL_TRIALS
rescue_target: trial 1 (M3 fail)
success_guard: trial 2 (M3 success)
treatment: CAPX_LIBERO_G17_FG=1
ideal_tk: off
baseline_sbatch_yaml_seed: fingerprinted_read_only
g17_worktree: $G17
model: $MODEL
temperature: $TEMPERATURE
global_seed: $GLOBAL_SEED
CTX

echo "[$(date '+%F %T')] START $task_name treatment FG"
set +e
"$PYTHON" -m capx.envs.launch \
  --config-path "$config_path" \
  --model "$MODEL" \
  --server-url "http://127.0.0.1:$llm_port/chat/completions" \
  --visual-differencing-model "$MODEL" \
  --visual-differencing-model-server-url "http://127.0.0.1:$llm_port/chat/completions" \
  --use-img-differencing True \
  --use-legacy-multi-turn-decision-prompt True \
  --temperature "$TEMPERATURE" \
  --total-trials "$TOTAL_TRIALS" \
  --num-workers "$NUM_WORKERS" \
  --record-video True \
  --output-dir "$task_root/eval"
rc=$?
set -e
echo "[$(date '+%F %T')] END $task_name rc=$rc"

# Pairing summary vs frozen baseline (read-only).
"$PYTHON" - <<'PYSUM'
import json, re
from pathlib import Path
out = Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_rescue_smoke_20260814")
# find latest job dir
runs = sorted((out / "runs" / "libero_spatial").glob("job-*"))
if not runs:
    raise SystemExit("no treatment runs")
job = runs[-1]
eval_dir = job / "libero_spatial_0" / "qwen3.7-plus" / "eval"
if not eval_dir.exists():
    # model folder may vary
    cands = list((job / "libero_spatial_0").glob("*/eval"))
    eval_dir = cands[0] if cands else eval_dir
baseline = Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813/runs/libero_spatial/job-517994/libero_spatial_0/qwen3.7-plus/eval")

def outcomes(root: Path):
    rows = {}
    if not root.exists():
        return rows
    for p in root.glob("trial_*"):
        m = re.match(r"trial_(\d+)_sandboxrc_(\d+)_reward_([0-9.]+)_taskcompleted_(\d+)", p.name)
        if not m or m.group(2) != "0":
            continue
        tid = int(m.group(1))
        rows[tid] = {"completed": int(m.group(4)), "reward": float(m.group(3)), "path": p.name}
    return rows

ctrl = outcomes(baseline)
treat = outcomes(eval_dir)
# gain traces
traces = {}
trace_root = eval_dir.parent.parent / "eval" / "libero_g17_fg_traces"
# traces live under output_dir
trace_root = job / "libero_spatial_0" / "eval" / "libero_g17_fg_traces"
if not trace_root.exists():
    # alternate: under qwen folder's parent eval
    alt = list(job.glob("**/libero_g17_fg_traces"))
    trace_root = alt[0] if alt else trace_root
for trial_dir in sorted(trace_root.glob("trial_*")) if trace_root.exists() else []:
    tid = int(trial_dir.name.split("_")[1])
    turns = []
    for turn in sorted(trial_dir.glob("turn_*/record.json")):
        turns.append(json.loads(turn.read_text()))
    traces[tid] = turns

summary = {
    "control_job": "517994",
    "treatment_job": job.name,
    "cells": {
        "trial_1_rescue_target": {
            "control_completed": ctrl.get(1, {}).get("completed"),
            "treatment_completed": treat.get(1, {}).get("completed"),
            "rescued": (ctrl.get(1, {}).get("completed") == 0 and treat.get(1, {}).get("completed") == 1),
        },
        "trial_2_success_guard": {
            "control_completed": ctrl.get(2, {}).get("completed"),
            "treatment_completed": treat.get(2, {}).get("completed"),
            "guard_held": (ctrl.get(2, {}).get("completed") == 1 and treat.get(2, {}).get("completed") == 1),
        },
    },
    "control_outcomes": ctrl,
    "treatment_outcomes": treat,
    "fg_turn_records": {
        str(k): [{"turn": r.get("turn"), "gain_status": r.get("gain_status"), "novel_fact_count": r.get("novel_fact_count")} for r in v]
        for k, v in traces.items()
    },
}
(out / "RESCUE_PAIR_SUMMARY.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PYSUM

# Re-check baseline fingerprints after run
"$PYTHON" - <<'PY'
import hashlib
from pathlib import Path
expected = {
    Path("/project/peilab/qshou/cap-x-agilex/scripts/sbatch_libero_m3_3x10x10.sh"):
        "1728358749e5e2fe916011e4578e81b15c1ffda55ec07c116c36f0a2b1794ef1",
    Path("/project/peilab/qshou/cap-x-agilex/env_configs/libero/franka_libero_m3.yaml"):
        "befb7ff390ad619ac4c6a0201c470ae884376206de666dd7cfba2bb3787a4939",
    Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813/SEED_TABLE.json"):
        "4fec7669be41d3c6fbff096f1ca587052a7253c7af6a9e0a619eb0854f65c24d",
}
for path, want in expected.items():
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    assert h == want, (path, h)
print("baseline fingerprints still ok after treatment")
PY

exit "$rc"
