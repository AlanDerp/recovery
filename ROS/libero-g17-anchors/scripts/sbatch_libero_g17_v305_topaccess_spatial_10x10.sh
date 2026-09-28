#!/usr/bin/env bash
#SBATCH --job-name=libero-g17-v305-topaccess-spatial-10x10
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_v305_topaccess_spatial_10x10_retry2_20260821/logs/%x-%j.out
#SBATCH --error=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_v305_topaccess_spatial_10x10_retry2_20260821/logs/%x-%j.out

# v305 top-access spatial 10x10 live (no π0 code-prefix). One task × 10 trials per job.
# Exact method anchor 2abec5e.  Unique delta from the v286 live protocol is
# the v305 acquisition/top-access stack; v306+ and all replay are disabled.
set -euo pipefail

# Login-node loopback proxies are not services on a Slurm compute node.  Since
# submission uses --export=ALL, sanitize them before any Python/OpenAI client
# or job-local HTTP service starts.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

TASK_ID=${TASK_ID:?TASK_ID required 0..9}
TOTAL_TRIALS=${TOTAL_TRIALS:-10}
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 9 )); then
  echo "TASK_ID must be 0..9, got: $TASK_ID" >&2
  exit 2
fi

G17=${CAPX_G17_WORKTREE_OVERRIDE:-/home/qshou/capx_g17_v305_topaccess_10x10}
PROJECT_DIR=${CAPX_LIBERO_RUNTIME_ANCHOR:-/project/peilab/qshou/capx_libero_runtime_anchor_v346_20260827}
BASELINE_EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813
EXPT_DIR=${CAPX_SPATIAL_10X10_EXPT_OVERRIDE:-/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_v305_topaccess_spatial_10x10_retry2_20260821}
METHOD_ANCHOR=${CAPX_SPATIAL_10X10_METHOD_ANCHOR:-2abec5e}
SOURCE_SCHEMA=${CAPX_SPATIAL_10X10_SOURCE_SCHEMA:-capx_libero_g17_v305_topaccess_spatial_10x10_source_v1}
MODE_NAME=${CAPX_SPATIAL_10X10_MODE_NAME:-libero_g17_v305_topaccess_spatial_10x10}
ARM_NAME=${CAPX_SPATIAL_10X10_ARM_NAME:-v305-topaccess}
export G17 EXPT_DIR METHOD_ANCHOR SOURCE_SCHEMA MODE_NAME ARM_NAME
export CAPX_EXPECTED_G17_ROOT="$G17"
export CAPX_SPATIAL_10X10_MODE_NAME="$MODE_NAME"
export CAPX_SPATIAL_10X10_ARM_NAME="$ARM_NAME"
HELPER=/home/qshou/capx_causal_ik_replay_81e0acb/experiments/core_fg/runtime
PYTHON="$PROJECT_DIR/.venv-libero/bin/python"
KEY=/scratch/peilab/qshou/RoboTwin_Sandbox/cap-x/.qwenkey_sg_tmp
BASE_URL=https://ws-vpxr6jfor4t48uyo.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
MODEL=qwen3.7-plus
NUM_WORKERS=1
TEMPERATURE=0
GLOBAL_SEED=20260718
MOLMO_ROOT=/scratch/peilab/qshou/capx-molmo
MOLMO_VENV="$MOLMO_ROOT/venv"
MOLMO_MODEL=/scratch/peilab/qshou/capx-molmo/cache/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b
MOLMO_GPU_UTIL=${MOLMO_GPU_UTIL:-0.55}

suite_name=libero_spatial
task_name="${suite_name}_${TASK_ID}"
suite_root="$EXPT_DIR/runs/${suite_name}/job-${SLURM_JOB_ID}"
task_root="$suite_root/${task_name}"
mkdir -p "$suite_root" "$task_root" "$EXPT_DIR/logs" "$EXPT_DIR/arms/$ARM_NAME" "$EXPT_DIR/evidence"

# Baseline + submitted-source fingerprint gates (read-only).
"$PYTHON" - <<PY
import hashlib, json, os, subprocess
from pathlib import Path
expected = {
    Path("$G17/env_configs/libero/franka_libero_m3.yaml"):
        "1a3ecd58f6e20fdc339c6abe9c11dec8f10da9523dacdd8788520ba6dc178dfa",
    Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813/SEED_TABLE.json"):
        "4fec7669be41d3c6fbff096f1ca587052a7253c7af6a9e0a619eb0854f65c24d",
}
for path, want in expected.items():
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != want:
        raise SystemExit(f"baseline drift: {path} got {h}")
print("baseline fingerprints ok")
source = Path("$EXPT_DIR/manifests/SOURCE_SHA256.json")
data = json.loads(source.read_text())
if data.get("schema") != "$SOURCE_SCHEMA":
    raise SystemExit(f"source schema mismatch: {data.get('schema')!r}")
for raw, want in data["files"].items():
    path = Path(raw)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"v305 top-access source drift: {path} {got} != {want}")
print("v305_topaccess_fingerprints_ok", len(data["files"]))
head = subprocess.check_output(["git", "-C", "$G17", "rev-parse", "HEAD"], text=True).strip()
want_head = os.environ["CAPX_SUBMISSION_GIT_REV"]
if head != want_head:
    raise SystemExit(f"worktree HEAD must be submitted harness {want_head}, got {head}")
method_paths = [
    "capx/integrations/franka/libero.py",
    "capx/integrations/franka/postclose_acquisition.py",
    "capx/integrations/franka/acquisition_audit.py",
    "capx/integrations/franka/acquisition_extract.py",
    "capx/integrations/franka/perceptual_grasp_family.py",
    "capx/integrations/franka/perceptual_swept_path.py",
    "capx/integrations/franka/measured_seed_ik.py",
    "capx/verification/libero_g17_takeover.py",
    "capx/verification/libero_g17_nearmiss_osc.py",
]
method_diff = subprocess.run(
    ["git", "-C", "$G17", "diff", "--quiet", "$METHOD_ANCHOR", "--", *method_paths]
)
if method_diff.returncode != 0:
    raise SystemExit("method files differ from frozen method anchor $METHOD_ANCHOR")
print("git_head_ok", head)
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
export CAPX_REPRO_AUDIT_DIR="$task_root/repro_audit"
export CAPX_SEED_TORCH=1
export CAPX_CURRENT_TASK_NAME="$task_name"
export CAPX_CONTACT_GRASPNET_SOURCE_ROOT="$PROJECT_DIR/capx/third_party/contact_graspnet_pytorch"
export CAPX_REQUIRE_API_SERVERS_READY=1
export CAPX_API_SERVER_WAIT_TIMEOUT_SECONDS=600

# v286 FG+RR base stack, plus the exact v305 top-access acquisition stack.
export CAPX_LIBERO_G17_FG=1
export CAPX_LIBERO_G17_XY_LOCK=1
export CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1
export CAPX_LIBERO_G17_RELEASE_READY=1
export CAPX_LIBERO_G17_NEARMISS_OSC=1
export CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW=1
export CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH=1
export CAPX_LIBERO_G17_V269_MECH=1
export CAPX_LIBERO_G17_V269_SYMBOL_SOURCE=fg
export CAPX_LIBERO_G17_V272_FAIR_LIVE=1
export CAPX_LIBERO_G17_V274_MECH_CONTRACT=1
export CAPX_LIBERO_G17_V276_COMPLETION_IK=1
export CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY=1
export CAPX_LIBERO_G17_V285_FULL_RELEASE_WINDOW=1
export CAPX_LIBERO_G17_V286_FINAL_FG_CHECK=1
export CAPX_LIBERO_G17_WORKGAP_ME=0
export CAPX_LIBERO_G17_V293_ACQUISITION=1
export CAPX_LIBERO_G17_V293_REQUIRE_FEASIBLE_GRASP=0
export CAPX_LIBERO_G17_V296A_PT_SWEPT_AUDIT=1
export CAPX_LIBERO_G17_V296B_CANDIDATE_PROBE=1
export CAPX_LIBERO_G17_V302A_TOPDOWN_GRASP_PROBE=1
export CAPX_LIBERO_G17_V303_TOPDOWN_GRASP_BEHAVIOR=1
export CAPX_LIBERO_G17_V304_INGRESS_RETREAT=1
export CAPX_LIBERO_G17_V305_FG_EVENT_STOP=1
unset CAPX_LIBERO_G17_V297B_ACQUISITION_RECOVERY || true
unset CAPX_LIBERO_G17_V306_STABLE_TOPDOWN || true
unset CAPX_LIBERO_G17_V307_CONTACT_PRESERVING_TOPDOWN || true
unset CAPX_LIBERO_G17_V308_CONTACT_READY || true
unset CAPX_LIBERO_G17_CODE_PREFIX_REPLAY || true
unset CAPX_LIBERO_G17_GRASP_REPLAY || true
unset CAPX_LIBERO_G17_ACQUIRE_FACTS || true
unset CAPX_LIBERO_G17_V287_RELEASE_EPOCH || true
unset CAPX_LIBERO_G17_V287_SETTLE_STEPS || true
unset CAPX_LIBERO_G17_V288_ACCEPTANCE || true
unset CAPX_LIBERO_G17_V289_PARTIAL_TRANSACTION || true
unset CAPX_LIBERO_G17_V24_PIN || true
unset CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER || true
unset CAPX_LIBERO_G17_GOAL_IDENTITY_VERIFY || true
export CAPX_IDEAL_TK_V4=0
export CAPX_IDEAL_TK=0
export CAPX_CLOSED_GT_TAKEOVER=0
export CAPX_FGSRP=0
export CAPX_FGSRP_NL=0
export CAPX_CORE_FG=0
unset PYOPENGL_PLATFORM
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CAPX_REPRO_AUDIT_DIR"

ROBOT_DESC_SRC=${ROBOT_DESC_SRC:-/scratch/peilab/qshou/robot_descriptions_cache_g17_v24}
ROBOT_DESC_JOB="$suite_root/robot_descriptions_cache"
mkdir -p "$ROBOT_DESC_JOB"
if [[ ! -f "$ROBOT_DESC_SRC/example-robot-data/robots/panda_description/urdf/panda.urdf" ]]; then
  echo "INFRA INVALID: missing panda.urdf in $ROBOT_DESC_SRC" >&2
  exit 2
fi
rsync -a --delete "$ROBOT_DESC_SRC/" "$ROBOT_DESC_JOB/"
export ROBOT_DESCRIPTIONS_CACHE="$ROBOT_DESC_JOB"

export PYTHONPATH="$G17/python_sitecustomize:$G17:$HELPER/python_sitecustomize:$HELPER:$PROJECT_DIR:${PYTHONPATH:-}"
cd "$G17"
source "$PROJECT_DIR/.venv-libero/bin/activate"

"$PYTHON" - <<'PYROBOSUITE'
import robosuite
import robosuite.macros
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

print(
    "robosuite_provenance_ok "
    f"package={robosuite.__file__} macros={robosuite.macros.__file__} "
    f"single_arm={SingleArmEnv.__module__}"
)
PYROBOSUITE

[[ -x "$PYTHON" ]]
[[ -s "$KEY" ]]
[[ -s "$SAM3_CHECKPOINT_PATH" ]]
[[ -f "$HELPER/qwen_proxy.py" ]]
[[ -x "$MOLMO_VENV/bin/python" ]]
[[ -d "$MOLMO_MODEL" ]]

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
echo "mode=$MODE_NAME base=v286 no_prefix method_anchor=$METHOD_ANCHOR"
echo "suite=$suite_name task=$TASK_ID trials=$TOTAL_TRIALS"
echo "baseline_control=$BASELINE_EXPT/runs/libero_spatial/job-517994"
echo "ports llm=$llm_port sam3=$sam3_port graspnet=$graspnet_port pyroki=$pyroki_port molmo=$molmo_port"
nvidia-smi -L | head -3

echo "[$(date '+%F %T')] prove v305 top-access + no prefix + no v306+"
"$PYTHON" - <<'PYPROOF'
import os
import capx.envs.trial as trial
path = trial.__file__
if os.environ["CAPX_EXPECTED_G17_ROOT"] not in path:
    raise SystemExit(f"FG INVALID: trial.py not from G17 worktree: {path}")
import capx.verification.libero_g17_takeover as takeover
from capx.integrations.franka.measured_seed_ik import measured_seed_ik_contract
if os.environ.get("CAPX_LIBERO_G17_FG") != "1":
    raise SystemExit("FG INVALID: CAPX_LIBERO_G17_FG!=1")
for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    if os.environ.get(key):
        raise SystemExit(f"INFRA INVALID: inherited proxy {key}={os.environ[key]!r}")
if os.environ.get("CAPX_LIBERO_G17_CODE_PREFIX_REPLAY"):
    raise SystemExit("FG INVALID: code prefix must be unset for suite 10x10")
if os.environ.get("CAPX_LIBERO_G17_ACQUIRE_FACTS", "0") == "1":
    raise SystemExit("FG INVALID: acquire-facts overlay must be off")
if os.environ.get("CAPX_LIBERO_G17_V287_RELEASE_EPOCH", "0") == "1":
    raise SystemExit("FG INVALID: v287+ overlay must be off")
required = {
    "CAPX_LIBERO_G17_XY_LOCK": "1",
    "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
    "CAPX_LIBERO_G17_V269_MECH": "1",
    "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
    "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
    "CAPX_LIBERO_G17_V276_COMPLETION_IK": "1",
    "CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY": "1",
    "CAPX_LIBERO_G17_V285_FULL_RELEASE_WINDOW": "1",
    "CAPX_LIBERO_G17_V286_FINAL_FG_CHECK": "1",
    "CAPX_LIBERO_G17_V293_ACQUISITION": "1",
    "CAPX_LIBERO_G17_V296A_PT_SWEPT_AUDIT": "1",
    "CAPX_LIBERO_G17_V296B_CANDIDATE_PROBE": "1",
    "CAPX_LIBERO_G17_V302A_TOPDOWN_GRASP_PROBE": "1",
    "CAPX_LIBERO_G17_V303_TOPDOWN_GRASP_BEHAVIOR": "1",
    "CAPX_LIBERO_G17_V304_INGRESS_RETREAT": "1",
    "CAPX_LIBERO_G17_V305_FG_EVENT_STOP": "1",
}
for key in filter(None, os.environ.get("CAPX_SPATIAL_10X10_EXTRA_REQUIRED_FLAGS", "").split(",")):
    required[key] = "1"
for key, want in required.items():
    if os.environ.get(key) != want:
        raise SystemExit(f"FG INVALID: {key}!={want!r}")
sv = takeover.schema_version()
print("takeover_schema", sv)
v354_schema = (
    os.environ.get("CAPX_LIBERO_G17_V354_FG_NATIVE_RETRY", "0") == "1"
    and sv == "capx-libero-g17-fg-takeover-v3.54-native-retry-dual-projection"
)
v286_schema = (
    "v2.86" in sv and "final-fg-axis-check" in sv and "fair-live" in sv
)
if not (v286_schema or v354_schema):
    raise SystemExit(f"FG INVALID: expected v286 fair-live schema, got {sv!r}")
c = measured_seed_ik_contract()
if c["native_blocking_commands_per_stage"] != 1 or c["tool_z_pi_symmetry"] != 1:
    raise SystemExit(f"FG INVALID: measured-seed contract drift {c}")
print("v305_topaccess_spatial_10x10_proof_ok")
PYPROOF

echo "[$(date '+%F %T')] infra asset preflight"
"$PYTHON" - <<PYINFRA
from pathlib import Path
g17 = Path("$G17")
proj = Path("$PROJECT_DIR")
def resolve(rel: str) -> Path:
    local = g17 / rel
    if local.exists():
        return local
    fallback = proj / rel
    if fallback.exists():
        return fallback
    return local
checks = {
    "graspnet_source_root": Path("$CAPX_CONTACT_GRASPNET_SOURCE_ROOT"),
    "graspnet_config": Path("$CAPX_CONTACT_GRASPNET_SOURCE_ROOT/checkpoints/contact_graspnet/config.yaml"),
    "graspnet_weights_dir": Path("$CAPX_CONTACT_GRASPNET_SOURCE_ROOT/checkpoints/contact_graspnet/checkpoints"),
    "libero_pro": resolve("capx/third_party/LIBERO-PRO/libero"),
    "sam3_ckpt": Path("$SAM3_CHECKPOINT_PATH"),
    "molmo_model": Path("$MOLMO_MODEL"),
    "qwen_key": Path("$KEY"),
}
missing = [name for name, path in checks.items() if not path.exists()]
for name, path in checks.items():
    print(f"{name}: {'OK' if path.exists() else 'MISSING'} ({path})")
if missing:
    raise SystemExit(f"infra preflight failed: {missing}")
print("graspnet_config_resolved", checks["graspnet_config"].resolve())
print("infra_preflight_ok")
PYINFRA

echo "[$(date '+%F %T')] Qwen preflight"
QWEN_KEY_FILE="$KEY" QWEN_BASE_URL="$BASE_URL" MODEL="$MODEL" "$PYTHON" - <<'PYPROBE'
from openai import OpenAI
import os
from pathlib import Path
key = next(line.strip() for line in Path(os.environ["QWEN_KEY_FILE"]).read_text().splitlines() if line.strip() and not line.startswith("#"))
client = OpenAI(api_key=key, base_url=os.environ["QWEN_BASE_URL"])
r = client.chat.completions.create(
    model=os.environ["MODEL"],
    messages=[{"role": "user", "content": "Return exactly: OK"}],
    max_tokens=16,
    temperature=0,
    timeout=120,
    extra_body={"enable_thinking": False},
)
msg = r.choices[0].message.content
if not msg:
    raise RuntimeError(f"Qwen preflight empty; finish_reason={r.choices[0].finish_reason}")
print(f"Qwen preflight OK: {msg[:32]!r}")
PYPROBE

echo "[$(date '+%F %T')] Starting Qwen proxy on 127.0.0.1:$llm_port"
"$PYTHON" "$HELPER/qwen_proxy.py" \
  --key-file "$KEY" \
  --base-url "$BASE_URL" \
  --audit-log "$suite_root/qwen_proxy_audit.jsonl" \
  --host 127.0.0.1 \
  --port "$llm_port" \
  --upstream-concurrency 1 \
  --enable-thinking false \
  --request-seed "$GLOBAL_SEED" \
  >"$suite_root/qwen_proxy.log" 2>&1 &
proxy_pid=$!
proxy_ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    echo "Qwen proxy exited before ready" >&2
    cat "$suite_root/qwen_proxy.log" >&2
    exit 1
  fi
  if curl --noproxy '*' --fail --silent "http://127.0.0.1:$llm_port/health" >/dev/null; then
    proxy_ready=1
    break
  fi
  sleep 2
done
if [[ "$proxy_ready" != "1" ]]; then
  echo "Qwen proxy did not become ready" >&2
  cat "$suite_root/qwen_proxy.log" >&2
  exit 1
fi
echo "[$(date '+%F %T')] Qwen proxy ready"

molmo_tmp="/tmp/capx-molmo-${SLURM_JOB_ID}"
mkdir -p "$molmo_tmp"
echo "[$(date '+%F %T')] Starting Molmo on 127.0.0.1:$molmo_port"
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
  tail -n 200 "$suite_root/molmo.log" >&2 || true
  exit 1
fi
echo "[$(date '+%F %T')] Molmo ready"

cat > "$suite_root/RUN_CONTEXT.txt" <<CTX
mode: $MODE_NAME
git_head: $CAPX_SUBMISSION_GIT_REV
code_prefix_replay: unset
grasp_replay: unset
base_stack: v286 FG+RR
top_access_switch: CAPX_LIBERO_G17_V303_TOPDOWN_GRASP_BEHAVIOR=1
acquisition_stack: ${CAPX_SPATIAL_10X10_STACK_LABEL:-v293,v296a,v296b,v302a,v303,v304,v305}
v306_v307_v308: disabled
CAPX_LIBERO_G17_XY_LOCK: 1
CAPX_LIBERO_G17_V286_FINAL_FG_CHECK: 1
suite: $suite_name
task_id: $TASK_ID
total_trials: $TOTAL_TRIALS
model: $MODEL
temperature: $TEMPERATURE
global_seed: $GLOBAL_SEED
baseline_control: $BASELINE_EXPT/runs/libero_spatial/job-517994
g17_worktree: $G17
CTX

config_path="$task_root/config.yaml"
"$PYTHON" - <<PYCFG
from pathlib import Path
src = Path("$G17/env_configs/libero/franka_libero_m3.yaml")
dst = Path("$config_path")
text = src.read_text()
text = text.replace("suite_name: libero_spatial", "suite_name: $suite_name", 1)
text = text.replace("task_id: 0", "task_id: $TASK_ID", 1)
text = text.replace("port: 8116", "port: $pyroki_port", 1)
text = text.replace("port: 8115", "port: $graspnet_port", 1)
text = text.replace("port: 8114", "port: $sam3_port", 1)
text = text.replace("output_dir: ./outputs/franka_libero_m3", f"output_dir: {dst.parent / 'eval'}")
text = text.replace("trials: 10", "trials: $TOTAL_TRIALS", 1)
dst.write_text(text)
print(dst)
PYCFG

echo "[$(date '+%F %T')] START $task_name"
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

# A nominal launch exit is not a valid method result if any required API
# server was unavailable during a trial.  Fail closed instead of scoring a
# fallback path as v305.
job_log="$EXPT_DIR/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
if rg -q \
  "API server .* (exited before ready|not ready after)|Failed to communicate with (GraspNet|SAM3|PyRoKi) service" \
  "$job_log"; then
  echo "INFRA INVALID: required API service unavailable; refusing task score" >&2
  exit 3
fi

# Absolute score for this task; suite aggregation compares against frozen v286.
export TASK_ROOT_FOR_SCORE="$task_root"
export TASK_ID_FOR_SCORE="$TASK_ID"
export JOB_ID_FOR_SCORE="$SLURM_JOB_ID"
export EXPT_DIR_FOR_SCORE="$EXPT_DIR"
"$PYTHON" - <<'PYSCORE'
import json, os, re
from pathlib import Path

def score(eval_dir: Path) -> tuple[int, int, dict[int, int]]:
    best: dict[int, tuple] = {}
    if not eval_dir.is_dir():
        return 0, 0, {}
    for d in eval_dir.glob("trial_*"):
        m = re.match(r"trial_(\d+)_sandboxrc_(\d+)_.*taskcompleted_(\d+)", d.name)
        if not m:
            continue
        tid, sb, ok = int(m.group(1)), int(m.group(2)), int(m.group(3))
        key = (sb, -ok, -d.stat().st_mtime)
        if tid not in best or key < best[tid][0]:
            best[tid] = (key, ok)
    outcomes = {tid: int(ok) for tid, (_, ok) in best.items()}
    return sum(outcomes.values()), len(outcomes), outcomes

task_id = int(os.environ["TASK_ID_FOR_SCORE"])
job_id = os.environ["JOB_ID_FOR_SCORE"]
treat = Path(os.environ["TASK_ROOT_FOR_SCORE"])
expt = Path(os.environ["EXPT_DIR_FOR_SCORE"])
t_eval = treat / "qwen3.7-plus" / "eval"
if not t_eval.is_dir():
    cands = list(treat.glob("*/eval"))
    t_eval = cands[0] if cands else t_eval
th, tn, outcomes = score(t_eval)
out = {
    "schema": "capx_libero_g17_spatial_10x10_task_score_v1",
    "mode": os.environ["CAPX_SPATIAL_10X10_MODE_NAME"],
    "git_head": os.environ["CAPX_SUBMISSION_GIT_REV"],
    "task_id": task_id,
    "job_id": job_id,
    "score": {"successes": th, "trials": tn},
    "outcomes": {str(k): v for k, v in sorted(outcomes.items())},
    "code_prefix_replay": False,
    "task_root": str(treat),
}
arm = expt / "arms" / os.environ["CAPX_SPATIAL_10X10_ARM_NAME"] / f"task-{task_id:02d}"
arm.mkdir(parents=True, exist_ok=True)
path = arm / f"SUMMARY_job-{job_id}.json"
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PYSCORE

exit "$rc"
