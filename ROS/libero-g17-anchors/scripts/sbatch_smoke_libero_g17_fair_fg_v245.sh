#!/usr/bin/env bash
#SBATCH --job-name=libero-g17-v245
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v245_20260817/logs/%x-%j.out
#SBATCH --error=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v245_20260817/logs/%x-%j.out

# Fair FG v245: v242 stack + ReleaseReady full conjuncts + EEF XY lock-Z (no TCP round-trip).
# Empty-hand same-name XY suppressed; wrong-grasp ordinal retained; skill keep-stock.
# Reuses frozen baseline reps from libero_g17_fair_fg_20260815 (do not re-run baseline).
# Set REPLICATE=0..K-1:
#   sbatch --export=ALL,ARM=v245,REPLICATE=0 scripts/sbatch_smoke_libero_g17_fair_fg_v245.sh
set -euo pipefail

ARM=${ARM:-v245}
REPLICATE=${REPLICATE:-0}
export ARM REPLICATE
case "$ARM" in
  v245) ;;
  *)
    echo "ARM must be v245 (ReleaseReady full-conjunct + lock-Z on v242 stack), got: $ARM" >&2
    exit 2
    ;;
esac
if ! [[ "$REPLICATE" =~ ^[0-9]+$ ]]; then
  echo "REPLICATE must be a non-negative integer, got: $REPLICATE" >&2
  exit 2
fi

G17=/home/qshou/capx_g17_observation_integrity
PROJECT_DIR=/project/peilab/qshou/cap-x-agilex
BASELINE_EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813
EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v245_20260817
PREFIX_PKG=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/code_prefix_replay_v1.json
PREFIX_SHA_FILE=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815/PACKAGE_SHA256.txt
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
suite_root="$EXPT_DIR/runs/${suite_name}/arm-${ARM}/rep-$(printf '%02d' "$REPLICATE")/job-${SLURM_JOB_ID}"
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
# Arm-specific FG / workgap. Ideal-TK / cube takeover stay off.
export CAPX_LIBERO_G17_CODE_PREFIX_REPLAY="$PREFIX_PKG"
export CAPX_IDEAL_TK_V4=0
export CAPX_IDEAL_TK=0
export CAPX_CLOSED_GT_TAKEOVER=0
export CAPX_FGSRP=0
export CAPX_FGSRP_NL=0
export CAPX_CORE_FG=0
case "$ARM" in
  v245)
    export CAPX_LIBERO_G17_FG=1
    export CAPX_LIBERO_G17_WORKGAP_ME=0
    export CAPX_LIBERO_G17_XY_LOCK=1
    export CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1
    export CAPX_LIBERO_G17_RELEASE_READY=1
    unset CAPX_LIBERO_G17_BAND_OBLIGATIONS
    unset CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE
    unset CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER
    unset CAPX_LIBERO_G17_GOAL_IDENTITY_VERIFY
    unset CAPX_LIBERO_G17_TARGET_GRASP_OBJECT
    ;;
esac
unset PYOPENGL_PLATFORM
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CAPX_REPRO_AUDIT_DIR"

# Prefix package fingerprint gate (frozen package sha + per-block code sha).
"$PYTHON" - <<PYPREFIX
import hashlib, json
from pathlib import Path
pkg = Path("$PREFIX_PKG")
want = Path("$PREFIX_SHA_FILE").read_text().strip()
got = hashlib.sha256(pkg.read_bytes()).hexdigest()
if got != want:
    raise SystemExit(f"prefix package drift: {got} != {want}")
data = json.loads(pkg.read_text())
# Fail closed if any registered CB0 code hash drifts from package content.
for tid, entry in data["trials"].items():
    blocks = entry["code_blocks"]
    digests = entry.get("code_block_sha256") or []
    if len(digests) != len(blocks):
        raise SystemExit(f"trial {tid}: code_block_sha256 length mismatch")
    for i, (block, digest) in enumerate(zip(blocks, digests)):
        h = hashlib.sha256(block.strip().encode()).hexdigest()
        if h != digest:
            raise SystemExit(f"trial {tid} block {i} code sha drift: {h} != {digest}")
print("prefix_package_ok", got)
print("prefix_code_block_sha_ok", {k: v.get("code_block_sha256") for k,v in data["trials"].items()})
PYPREFIX

# G17 overlay sitecustomize FIRST: disables editable agilex capx finder, then
# chains the HELPER reproducibility sitecustomize.
export PYTHONPATH="$G17/python_sitecustomize:$G17:$HELPER/python_sitecustomize:$HELPER:$PROJECT_DIR:${PYTHONPATH:-}"

cd "$G17"
# Keep agilex as CAPX project for configs/venv only; sources come from G17 overlay.
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
echo "arm=$ARM replicate=$REPLICATE treatment_FG=$CAPX_LIBERO_G17_FG workgap=$CAPX_LIBERO_G17_WORKGAP_ME"
echo "prefix_pkg=$PREFIX_PKG"
echo "suite=$suite_name task=$task_id trials=1..$TOTAL_TRIALS"
echo "pair_control=baseline job-517994 libero_spatial_0 (trial1 fail / trial2 success)"
echo "ports llm=$llm_port sam3=$sam3_port graspnet=$graspnet_port pyroki=$pyroki_port molmo=$molmo_port"
echo "MOLMO_SERVICE_URL=$MOLMO_SERVICE_URL"
nvidia-smi -L | head -3

echo "[$(date '+%F %T')] prove G17 trial.py wins over editable agilex"
"$PYTHON" - <<'PYPROOF'
import os
import capx.envs.trial as trial
path = trial.__file__
print("trial_module", path)
if "capx_g17_observation_integrity" not in path:
    raise SystemExit(f"FG INVALID: trial.py not from G17 worktree: {path}")
if "CAPX_LIBERO_G17_CODE_PREFIX_REPLAY" not in open(path, encoding="utf-8").read():
    raise SystemExit(f"PREFIX INVALID: trial.py lacks code-prefix hook: {path}")
from capx.verification import libero_g17_code_prefix_replay as code_prefix
import capx.verification.libero_g17_takeover as takeover
import capx.verification.libero_g17_fg_projection as proj
import capx.integrations.vision.molmo as molmo
print("prefix_module", code_prefix.__file__)
print("takeover_module", takeover.__file__)
print("projection_module", proj.__file__)
want = os.environ.get("MOLMO_SERVICE_URL", "")
print("molmo_SERVICE_URL_env", want)
print("molmo_SERVICE_URL_module", molmo.SERVICE_URL)
if not want:
    raise SystemExit("INFRA INVALID: MOLMO_SERVICE_URL unset")
if molmo.SERVICE_URL.rstrip("/") != want.rstrip("/"):
    raise SystemExit(
        f"INFRA INVALID: molmo.SERVICE_URL={molmo.SERVICE_URL!r} != env {want!r}"
    )
arm = os.environ.get("ARM", "")
fg = os.environ.get("CAPX_LIBERO_G17_FG", "0")
workgap = os.environ.get("CAPX_LIBERO_G17_WORKGAP_ME", "0")
replicate = os.environ.get("REPLICATE", "0")
src = open(proj.__file__, encoding="utf-8").read()
if workgap not in ("0", "", "false", "False"):
    raise SystemExit(f"FAIR FG INVALID: workgap must be off, got {workgap!r}")
if fg == "1":
    if "libero_g17_public_iff" not in src:
        raise SystemExit("FG INVALID: projection does not import public On IFF module")
    schema = takeover.schema_version()
    if "ordinal-sgp-release-ready-full" not in schema:
        raise SystemExit(
            f"FG INVALID: v245 expected ordinal-sgp-release-ready-full schema, got {schema!r}"
        )
    if "goal-identity" in schema or "target-grasp-object" in schema:
        raise SystemExit(f"FG INVALID: v245 must not use Goal/identity schema, got {schema!r}")
    if os.environ.get("CAPX_LIBERO_G17_XY_LOCK", "0") != "1":
        raise SystemExit("FG INVALID: v245 requires CAPX_LIBERO_G17_XY_LOCK=1")
    if os.environ.get("CAPX_LIBERO_G17_RELEASE_READY", "0") != "1":
        raise SystemExit("FG INVALID: v245 requires RELEASE_READY=1")
    if os.environ.get("CAPX_LIBERO_G17_BAND_OBLIGATIONS", "0") == "1":
        raise SystemExit("FG INVALID: v245 must not enable BAND_OBLIGATIONS")
    if os.environ.get("CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE", "0") == "1":
        raise SystemExit("FG INVALID: v245 must not enable POST_MISS_CLEAR_CACHE")
    _ctg = open(proj.__file__, encoding="utf-8").read().split("def classify_task_gap")[1].split("\ndef ")[0]
    if "acquired = _payload_acquired(after)" not in _ctg or "or acquired" not in _ctg:
        raise SystemExit("FG INVALID: v245 classify_task_gap must use _payload_acquired")
    if "def assert_task_gap_consistency" not in open(proj.__file__, encoding="utf-8").read():
        raise SystemExit("FG INVALID: v245 must assert_task_gap_consistency")
    rr_src = open(__import__("capx.verification.libero_g17_release_ready", fromlist=["x"]).__file__, encoding="utf-8").read()
    if "maybe_xy_align_before_release" not in rr_src:
        raise SystemExit("FG INVALID: v245 missing release_ready module")
    if "maybe_xy_align_before_release" not in open(__import__("capx.integrations.franka.libero", fromlist=["x"]).__file__, encoding="utf-8").read():
        raise SystemExit("FG INVALID: v245 open_gripper must call release_ready")
    if os.environ.get("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER", "0") == "1":
        raise SystemExit("FG INVALID: v245 must not rewrite Goal")
    from capx.verification import libero_g17_xy_lock as xy_lock
    if not xy_lock.xy_lock_enabled():
        raise SystemExit("FG INVALID: xy_lock.xy_lock_enabled() is false")
    libero_src = open(__import__("capx.integrations.franka.libero", fromlist=["x"]).__file__, encoding="utf-8").read()
    if "keep_stock" not in libero_src:
        raise SystemExit("FG INVALID: v245 requires keep-stock path")
    if 'points_3d") or []' in libero_src or "points_3d') or []" in libero_src:
        raise SystemExit("FG INVALID: v245 must not use points_3d or [] (numpy truth-value)")
    if "body_world_to_robot_xyz" not in libero_src:
        raise SystemExit("FG INVALID: v245 requires robot-base projection")
    sgp = libero_src.split("def sample_grasp_pose")[1].split("def ")[0]
    if "_resolve_typed_instance(" not in sgp:
        raise SystemExit("FG INVALID: sample_grasp_pose must call _resolve_typed_instance")
    sgp_sig = libero_src.split("def sample_grasp_pose")[1].split(":")[0]
    if "which" in sgp_sig:
        raise SystemExit("FG INVALID: v245 sample_grasp_pose must not expose which=")
    if "live FG RelXY token" in libero_src.split("def sample_grasp_pose")[1].split("def ")[0]:
        raise SystemExit("FG INVALID: v245 must not docstring-coach RelXY/which")
    proj_src = open(proj.__file__, encoding="utf-8").read()
    if "allow_empty_sibling_xy=not xy_lock_enabled()" not in proj_src:
        raise SystemExit(
            "FG INVALID: v245 must suppress empty-hand XY while keeping wrong-grasp ordinal"
        )
    if "pass which=" in libero_src:
        raise SystemExit("FG INVALID: v245 must not coach which= in skill errors")

    if "sibling_probe_pixels" in libero_src or "_lock_second_stock_on_agentview" in libero_src:
        raise SystemExit("FG INVALID: v245 must not use SAM3 second-stock probes")
    if "capture_terminal_witness" not in open(__import__("capx.verification.libero_g17_takeover", fromlist=["x"]).__file__, encoding="utf-8").read():
        raise SystemExit("FG INVALID: takeover lacks capture_terminal_witness")
    trial_src = open(__import__("capx.envs.trial", fromlist=["x"]).__file__, encoding="utf-8").read()
    if "capture_terminal_witness" not in trial_src:
        raise SystemExit("FG INVALID: trial.py lacks terminal witness hook")
elif arm != "v245":
    raise SystemExit(f"v245 arm expected CAPX_LIBERO_G17_FG=1, arm={arm}")
print("fair_fg_replicate", replicate)
print("layer1_keep_stock_xy_lock_ok")
print("keep_stock_path_ok")
print("xy_lock_ndarray_fix_ok")
print("prefix_hook_proof_ok")
print("molmo_url_proof_ok")
print("world_fg_endpoint_proof_ok")
# Infra: prefix env + audit dir must be live for init-sha fallback.
if not os.environ.get("CAPX_LIBERO_G17_CODE_PREFIX_REPLAY"):
    raise SystemExit("INFRA INVALID: CAPX_LIBERO_G17_CODE_PREFIX_REPLAY unset")
if not os.environ.get("CAPX_REPRO_AUDIT_DIR"):
    raise SystemExit("INFRA INVALID: CAPX_REPRO_AUDIT_DIR unset")
from pathlib import Path
Path(os.environ["CAPX_REPRO_AUDIT_DIR"]).mkdir(parents=True, exist_ok=True)
print("prefix_infra_ok", os.environ["CAPX_LIBERO_G17_CODE_PREFIX_REPLAY"])
print("arm", arm, "fg", fg, "workgap", workgap)
print("takeover_schema", takeover.schema_version())
PYPROOF

echo "[$(date '+%F %T')] infra asset preflight"
"$PYTHON" - <<PYINFRA
from pathlib import Path
checks = {
    "graspnet_config": Path("$G17/capx/third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet/config.yaml"),
    "graspnet_weights_dir": Path("$G17/capx/third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints"),
    "libero_pro": Path("$G17/capx/third_party/LIBERO-PRO/libero"),
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
QWEN_KEY_FILE="$KEY" QWEN_BASE_URL="$BASE_URL" MODEL="$MODEL" "$PYTHON" - <<'PYPROBE'
from openai import OpenAI
import os
from pathlib import Path
key = next(line.strip() for line in Path(os.environ["QWEN_KEY_FILE"]).read_text().splitlines() if line.strip() and not line.startswith("#"))
client = OpenAI(api_key=key, base_url=os.environ["QWEN_BASE_URL"])
r = client.chat.completions.create(model=os.environ["MODEL"], messages=[{"role":"user","content":"ping"}], max_tokens=8, temperature=0)
print("qwen_ok", r.choices[0].message.content)
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
mode: libero_g17_fair_fg_v245_v24_contract_head_reports
arm: $ARM
replicate: $REPLICATE
prefix_pkg: $PREFIX_PKG
pair_control: $BASELINE_EXPT/runs/libero_spatial/job-517994/libero_spatial_0
suite: $suite_name
task_id: $task_id
trials: 1..$TOTAL_TRIALS
CAPX_LIBERO_G17_FG: $CAPX_LIBERO_G17_FG
CAPX_LIBERO_G17_WORKGAP_ME: $CAPX_LIBERO_G17_WORKGAP_ME
CAPX_LIBERO_G17_XY_LOCK: $CAPX_LIBERO_G17_XY_LOCK
CAPX_LIBERO_G17_RELEASE_READY: ${CAPX_LIBERO_G17_RELEASE_READY:-0}
ideal_tk: off
baseline_sbatch_yaml_seed: fingerprinted_read_only
g17_worktree: $G17
model: $MODEL
temperature: $TEMPERATURE
global_seed: $GLOBAL_SEED
CTX

echo "[$(date '+%F %T')] START $task_name arm=$ARM"
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
echo "[$(date '+%F %T')] END $task_name arm=$ARM rc=$rc"

export SUITE_ROOT="$suite_root"
# Arm summary vs frozen baseline (read-only).
"$PYTHON" - <<'PYSUM'
import json, os, re, sys
from pathlib import Path
arm = os.environ["ARM"]
out = Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v245_20260817")
job = Path(os.environ["SUITE_ROOT"]).resolve()
eval_dir = job / "libero_spatial_0" / "qwen3.7-plus" / "eval"
if not eval_dir.exists():
    cands = list((job / "libero_spatial_0").glob("*/eval"))
    eval_dir = cands[0] if cands else eval_dir
baseline = Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813/runs/libero_spatial/job-517994/libero_spatial_0/qwen3.7-plus/eval")
job_id = job.name.split("-")[-1]
# Logs are named libero-g17-v245-{arm}-{jobid}.out (sbatch --job-name), not pfx-{jobid}.
rep = int(os.environ.get("REPLICATE", "0"))
log_candidates = [
    out / "logs" / f"libero-g17-v245-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v245-{arm}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v245-{arm}-{job_id}.out",
    out / "logs" / f"libero-g17-v245-{job_id}.out",
]
log = next((p for p in log_candidates if p.exists()), log_candidates[0])
molmo_log = job / "molmo.log"

def outcomes(root: Path):
    rows = {}
    if not root.exists():
        return rows
    for p in root.glob("trial_*"):
        m = re.match(r"trial_(\d+)_sandboxrc_(\d+)_reward_([0-9.]+)_taskcompleted_(\d+)", p.name)
        if not m:
            continue
        tid = int(m.group(1))
        completed = int(m.group(4))
        reward = float(m.group(3))
        prev = rows.get(tid)
        if prev is None or completed > prev["completed"] or (
            completed == prev["completed"] and reward > prev["reward"]
        ):
            rows[tid] = {
                "completed": completed,
                "reward": reward,
                "path": p.name,
                "sandboxrc": int(m.group(2)),
            }
    return rows

ctrl = outcomes(baseline)
treat = outcomes(eval_dir)

prefix_hits = []
if eval_dir.exists():
    prefix_hits.extend(eval_dir.glob("**/libero_g17_code_prefix_replay/trial_*_turn_*.json"))
repro = job / "libero_spatial_0" / "repro_audit" / "libero_g17_code_prefix_replay"
if repro.exists():
    prefix_hits.extend(repro.glob("trial_*_turn_*.json"))
# de-dupe by name
seen_names = set()
uniq = []
for p in prefix_hits:
    if p.name in seen_names:
        continue
    seen_names.add(p.name)
    uniq.append(p)
prefix_hits = uniq
prefix_hit_rows = []
for p in prefix_hits:
    try:
        prefix_hit_rows.append(json.loads(p.read_text()))
    except Exception:
        pass
prefix_ok = len(prefix_hit_rows) >= 2  # trial1 turn0 + trial2 turn0

trace_roots = list(job.glob("**/libero_g17_fg_traces"))
engagement = {}
turn_records = {}
mechanism_violations = []
for trace_root in trace_roots:
    for trial_dir in sorted(trace_root.glob("trial_*")):
        tid = int(trial_dir.name.split("_")[1])
        eng = trial_dir / "engagement.json"
        if eng.exists():
            engagement[tid] = json.loads(eng.read_text())
        turns = []
        for turn_dir in sorted(trial_dir.glob("turn_*")):
            rec_path = turn_dir / "record.json"
            if not rec_path.exists():
                continue
            rec = json.loads(rec_path.read_text())
            turns.append(rec)
        turn_records[tid] = turns

fg = os.environ.get("CAPX_LIBERO_G17_FG", "0") == "1"
fg_engaged = bool(engagement) if fg else True
schemas = {str(k): (v or {}).get("schema_version") for k, v in engagement.items()}

log_text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
molmo_text = molmo_log.read_text(encoding="utf-8", errors="ignore") if molmo_log.exists() else ""
refused = len(re.findall(r"Connection refused", log_text))
wrong_port_8122 = len(re.findall(r"127\.0\.0\.1:8122", log_text))
molmo_chat_hits = len(re.findall(r"/v1/chat/completions", molmo_text))
molmo_url_proof = "molmo_url_proof_ok" in log_text
prefix_proof = "prefix_hook_proof_ok" in log_text
infra_ok = molmo_url_proof and prefix_proof and refused == 0 and wrong_port_8122 == 0 and molmo_chat_hits > 0

if not prefix_ok:
    validity = "invalid_prefix_miss"
elif fg and not fg_engaged:
    validity = "invalid_fg_not_engaged"
elif not infra_ok:
    validity = "invalid_infra"
else:
    validity = "valid_prefix_arm"

rescued = None
guard_held = None
gain = None
if validity == "valid_prefix_arm":
    rescued = ctrl.get(1, {}).get("completed") == 0 and treat.get(1, {}).get("completed") == 1
    guard_held = ctrl.get(2, {}).get("completed") == 1 and treat.get(2, {}).get("completed") == 1
    if rescued and guard_held:
        gain = "positive"
    elif (not rescued) and guard_held:
        gain = "negative_no_rescue"
    else:
        gain = "negative_guard_lost_or_incomplete"

# Mechanism carrier in first post-fork prompt (multi_turn_prompt_00 after CB0)
carrier = {}
for tid, row in treat.items():
    trial_dirs = list(eval_dir.glob(f"trial_{tid:02d}_*"))
    if not trial_dirs:
        continue
    tdir = sorted(trial_dirs, key=lambda p: ("taskcompleted_1" not in p.name, p.name))[0]
    prompts = sorted((tdir / "prompts_and_responses").glob("multi_turn_prompt_*.txt")) if (tdir / "prompts_and_responses").exists() else []
    text = prompts[0].read_text(encoding="utf-8", errors="ignore") if prompts else ""
    carrier[str(tid)] = {
        "has_ontop_iff": "Verified LIBERO on-top IFF basis" in text,
        "has_placement_incomplete_note": "placement evidence is incomplete" in text,
        "has_acquisition_incomplete_note": "acquisition evidence is incomplete" in text,
        "prompt_file": str(prompts[0]) if prompts else None,
    }

replicate = int(os.environ.get("REPLICATE", "0"))
summary = {
    "mode": "libero_g17_fair_fg_v245_release_ready_full",
    "arm": arm,
    "replicate": replicate,
    "validity": validity,
    "gain_result": gain,
    "prefix_ok": prefix_ok,
    "prefix_hits": len(prefix_hit_rows),
    "fg_enabled": fg,
    "fg_engaged": bool(engagement),
    "workgap_me": False,
    "schemas": schemas,
    "infra_ok": infra_ok,
    "xy_lock_enabled": os.environ.get("CAPX_LIBERO_G17_XY_LOCK", "0") == "1",
    "release_ready_enabled": os.environ.get("CAPX_LIBERO_G17_RELEASE_READY", "0") == "1",
    "release_ready_log_hits": log_text.count("[release-ready]"),
    "xy_lock_keep_stock": log_text.count("[xy-lock] keep-stock"),
    "xy_lock_point_correct": log_text.count("[xy-lock] point-correct"),
    "xy_lock_stock_mismatch": log_text.count("stock-mismatch"),
    "xy_lock_log_hits": log_text.count("[xy-lock]"),
    "xy_lock_n_complete": [int(x) for x in re.findall(r"n_complete=(\d+)", log_text)],
    "xy_lock_n_probe": [int(x) for x in re.findall(r"n_probe=(\d+)", log_text)],
    "molmo_chat_completions_logged": molmo_chat_hits,
    "control_job": "517994",
    "treatment_job": job.name,
    "suite_root": str(job),
    "cells": {
        "trial_1_rescue_target": {
            "control_completed": ctrl.get(1, {}).get("completed"),
            "treatment_completed": treat.get(1, {}).get("completed"),
            "rescued": rescued,
            "treatment_success": treat.get(1, {}).get("completed") == 1,
        },
        "trial_2_success_guard": {
            "control_completed": ctrl.get(2, {}).get("completed"),
            "treatment_completed": treat.get(2, {}).get("completed"),
            "guard_held": guard_held,
            "treatment_success": treat.get(2, {}).get("completed") == 1,
        },
    },
    "control_outcomes": ctrl,
    "treatment_outcomes": treat,
    "post_fork_carrier": carrier,
    "prefix_hit_records": prefix_hit_rows,
}
arm_dir = out / "arms" / arm / f"rep-{replicate:02d}"
arm_dir.mkdir(parents=True, exist_ok=True)
(arm_dir / f"SUMMARY_job-{job_id}.json").write_text(json.dumps(summary, indent=2))
(arm_dir / "LATEST_SUMMARY.json").write_text(json.dumps(summary, indent=2))
(out / "arms" / arm / "LATEST_SUMMARY.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
if validity != "valid_prefix_arm":
    sys.exit(2)
PYSUM
summary_rc=$?

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

# Exit: launch failure OR FG-invalid summary both fail-closed.
if [[ "$rc" != "0" ]]; then
  exit "$rc"
fi
exit "$summary_rc"
