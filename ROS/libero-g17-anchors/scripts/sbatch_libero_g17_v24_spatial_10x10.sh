#!/usr/bin/env bash
#SBATCH --job-name=libero-g17-v24-spatial-10x10
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_v24_spatial_10x10_20260816/logs/%x-%j.out
#SBATCH --error=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_v24_spatial_10x10_20260816/logs/%x-%j.out

# Frozen fair-K v24 FG on libero_spatial 10 tasks × 10 trials.
# Matches M3 baseline infra (job-517994) with unique delta = FG v2.4-public-in-iff.
# No π0 code-prefix (full suite, not fair-K fork). XY_LOCK / Goal rewrite off.
set -euo pipefail

G17=/home/qshou/capx_g17_observation_integrity
PROJECT_DIR=/project/peilab/qshou/cap-x-agilex
BASELINE_EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813
EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_v24_spatial_10x10_20260816
HELPER=/home/qshou/capx_causal_ik_replay_81e0acb/experiments/core_fg/runtime
PYTHON="$PROJECT_DIR/.venv-libero/bin/python"
KEY=/scratch/peilab/qshou/RoboTwin_Sandbox/cap-x/.qwenkey_sg_tmp
BASE_URL=https://ws-vpxr6jfor4t48uyo.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
MODEL=qwen3.7-plus
TOTAL_TRIALS=10
NUM_WORKERS=1
TEMPERATURE=0
GLOBAL_SEED=20260718
MOLMO_ROOT=/scratch/peilab/qshou/capx-molmo
MOLMO_VENV="$MOLMO_ROOT/venv"
MOLMO_MODEL=/scratch/peilab/qshou/capx-molmo/cache/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b
MOLMO_GPU_UTIL=${MOLMO_GPU_UTIL:-0.55}

suite_name=libero_spatial
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

# v24 FG treatment (fair-K winner contract).
export CAPX_LIBERO_G17_FG=1
export CAPX_LIBERO_G17_V24_PIN=1
export CAPX_LIBERO_G17_WORKGAP_ME=0
export CAPX_LIBERO_G17_XY_LOCK=0
unset CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER || true
unset CAPX_LIBERO_G17_GOAL_IDENTITY_VERIFY || true
unset CAPX_LIBERO_G17_TARGET_GRASP_OBJECT || true
unset CAPX_LIBERO_G17_CODE_PREFIX_REPLAY || true
export CAPX_IDEAL_TK_V4=0
export CAPX_IDEAL_TK=0
export CAPX_CLOSED_GT_TAKEOVER=0
export CAPX_FGSRP=0
export CAPX_FGSRP_NL=0
export CAPX_CORE_FG=0
unset PYOPENGL_PLATFORM
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CAPX_REPRO_AUDIT_DIR"

# Job-local robot_descriptions cache (521102 root-cause: shallow/NFS git SHA miss).
ROBOT_DESC_SRC=${ROBOT_DESC_SRC:-/scratch/peilab/qshou/robot_descriptions_cache_g17_v24}
ROBOT_DESC_JOB="$suite_root/robot_descriptions_cache"
mkdir -p "$ROBOT_DESC_JOB"
if [[ ! -f "$ROBOT_DESC_SRC/example-robot-data/robots/panda_description/urdf/panda.urdf" ]]; then
  echo "INFRA INVALID: missing panda.urdf in $ROBOT_DESC_SRC" >&2
  exit 2
fi
rsync -a --delete "$ROBOT_DESC_SRC/" "$ROBOT_DESC_JOB/"
export ROBOT_DESCRIPTIONS_CACHE="$ROBOT_DESC_JOB"
echo "ROBOT_DESCRIPTIONS_CACHE=$ROBOT_DESCRIPTIONS_CACHE"

# G17 overlay sitecustomize FIRST.
export PYTHONPATH="$G17/python_sitecustomize:$G17:$HELPER/python_sitecustomize:$HELPER:$PROJECT_DIR:${PYTHONPATH:-}"

cd "$G17"
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
echo "mode=libero_g17_v24_spatial_10x10 FG=1 V24_PIN=1 XY_LOCK=0"
echo "suite=$suite_name tasks=0-9 trials=$TOTAL_TRIALS"
echo "baseline_control=$BASELINE_EXPT/runs/libero_spatial/job-517994"
echo "ports llm=$llm_port sam3=$sam3_port graspnet=$graspnet_port pyroki=$pyroki_port molmo=$molmo_port"
nvidia-smi -L | head -3

echo "[$(date '+%F %T')] prove G17 + v24 pin + robot_descriptions"
"$PYTHON" - <<'PYPROOF'
import os
import capx.envs.trial as trial
path = trial.__file__
print("trial_module", path)
if "capx_g17_observation_integrity" not in path:
    raise SystemExit(f"FG INVALID: trial.py not from G17 worktree: {path}")
import capx.verification.libero_g17_takeover as takeover
import capx.verification.libero_g17_fg_projection as proj
print("takeover_module", takeover.__file__)
print("projection_module", proj.__file__)
if os.environ.get("CAPX_LIBERO_G17_FG") != "1":
    raise SystemExit("FG INVALID: CAPX_LIBERO_G17_FG!=1")
if os.environ.get("CAPX_LIBERO_G17_V24_PIN") != "1":
    raise SystemExit("FG INVALID: CAPX_LIBERO_G17_V24_PIN!=1")
if os.environ.get("CAPX_LIBERO_G17_XY_LOCK", "0") not in ("0", "", "false", "False"):
    raise SystemExit("FG INVALID: XY_LOCK must be off for v24 suite")
if os.environ.get("CAPX_LIBERO_G17_CODE_PREFIX_REPLAY"):
    raise SystemExit("FG INVALID: code prefix must be unset for full suite")
sv = takeover.schema_version()
print("takeover_schema", sv)
if sv != "capx-libero-g17-fg-takeover-v2.4-public-in-iff":
    raise SystemExit(f"FG INVALID: expected v2.4-public-in-iff, got {sv!r}")
if not proj.v24_pin_enabled():
    raise SystemExit("FG INVALID: projection v24_pin_enabled() is False")
want = os.environ.get("MOLMO_SERVICE_URL", "")
import capx.integrations.vision.molmo as molmo
if molmo.SERVICE_URL.rstrip("/") != want.rstrip("/"):
    raise SystemExit(f"INFRA INVALID: molmo URL mismatch {molmo.SERVICE_URL!r} vs {want!r}")

# Hard gate: panda URDF load must not hit flaky git (521102 failure mode).
cache = os.environ.get("ROBOT_DESCRIPTIONS_CACHE", "")
urdf = os.path.join(
    cache, "example-robot-data", "robots", "panda_description", "urdf", "panda.urdf"
)
if not os.path.isfile(urdf):
    raise SystemExit(f"INFRA INVALID: panda.urdf missing at {urdf}")
from robot_descriptions.loaders.yourdfpy import load_robot_description
import robot_descriptions._cache as _rd_cache
if not getattr(_rd_cache, "_capx_skip_git_if_cached", False):
    raise SystemExit("INFRA INVALID: robot_descriptions skip-git patch not applied")
loaded = load_robot_description("panda_description")
print("panda_load_ok", getattr(loaded, "base_link", None), "cache", cache)
from capx.integrations.motion.pyroki_context import get_pyroki_context
ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
print("pyroki_ctx_ok", type(ctx.robot).__name__)

# Child-process gate (mirrors CAPX_PROCESS_ISOLATED_TRIALS spawn).
import subprocess, sys, textwrap
child_src = textwrap.dedent(
    """
    import os
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    import robot_descriptions._cache as c
    assert getattr(c, "_capx_skip_git_if_cached", False), "child missing skip-git patch"
    u = load_robot_description("panda_description")
    print("child_ok", u.base_link)
    """
)
env = os.environ.copy()
for i in range(4):
    p = subprocess.run(
        [sys.executable, "-c", child_src],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode != 0 or "child_ok" not in p.stdout:
        raise SystemExit(
            f"INFRA INVALID: child panda load failed i={i} "
            f"rc={p.returncode} stdout={p.stdout!r} stderr={p.stderr[-500:]!r}"
        )
print("panda_child_ok", 4)
print("v24_spatial_10x10_proof_ok")
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
  if (( i % 30 == 0 )); then
    echo "[$(date '+%F %T')] waiting for Molmo $((i * 2))/900s"
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
mode: libero_g17_v24_spatial_10x10
schema: capx-libero-g17-fg-takeover-v2.4-public-in-iff
CAPX_LIBERO_G17_FG: 1
CAPX_LIBERO_G17_V24_PIN: 1
CAPX_LIBERO_G17_XY_LOCK: 0
CAPX_LIBERO_G17_WORKGAP_ME: 0
code_prefix_replay: unset
suite: $suite_name
task_ids: 0-9
total_trials: $TOTAL_TRIALS
model: $MODEL
temperature: $TEMPERATURE
global_seed: $GLOBAL_SEED
baseline_control: $BASELINE_EXPT/runs/libero_spatial/job-517994
baseline_sbatch_yaml_seed: fingerprinted_read_only
g17_worktree: $G17
ROBOT_DESCRIPTIONS_CACHE: $ROBOT_DESCRIPTIONS_CACHE
robot_desc_fix: skip_git_when_urdf_present
CTX

failures=0
for task_id in $(seq 0 9); do
  task_name="${suite_name}_${task_id}"
  task_root="$suite_root/$task_name"
  mkdir -p "$task_root" "$task_root/repro_audit"
  export CAPX_REPRO_AUDIT_DIR="$task_root/repro_audit"
  export CAPX_CURRENT_TASK_NAME="$task_name"
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
  if [[ "$rc" != "0" ]]; then
    failures=$((failures + 1))
  fi
done

echo "[$(date '+%F %T')] DONE suite=$suite_name failures=$failures"
find "$suite_root" -name summaries.txt -print | sort || true

# Compact suite scorecard vs frozen baseline spatial (job-517994).
export SUITE_ROOT_FOR_SCORE="$suite_root"
"$PYTHON" - <<'PYSCORE'
import json, os, re
from pathlib import Path

def score(eval_dir: Path) -> tuple[int, int]:
    # Dedup by trial index. Prefer sandboxrc=0, then success, then newest mtime.
    # (Lexicographic dir names wrongly prefer sandboxrc_1 over sandboxrc_0.)
    best: dict[int, tuple] = {}
    if not eval_dir.is_dir():
        return 0, 0
    for d in eval_dir.glob("trial_*"):
        m = re.match(r"trial_(\d+)_sandboxrc_(\d+)_.*taskcompleted_(\d+)", d.name)
        if not m:
            continue
        tid, sb, ok = int(m.group(1)), int(m.group(2)), int(m.group(3))
        key = (sb, -ok, -d.stat().st_mtime)
        if tid not in best or key < best[tid][0]:
            best[tid] = (key, ok)
    if not best:
        return 0, 0
    return sum(v[1] for v in best.values()), len(best)

base = Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813/runs/libero_spatial/job-517994")
treat = Path(os.environ["SUITE_ROOT_FOR_SCORE"])
rows = []
b_tot = t_tot = 0
b_n = t_n = 0
for tid in range(10):
    b_eval = base / f"libero_spatial_{tid}" / "qwen3.7-plus" / "eval"
    if not b_eval.is_dir():
        cands = list((base / f"libero_spatial_{tid}").glob("*/eval"))
        b_eval = cands[0] if cands else b_eval
    t_eval = treat / f"libero_spatial_{tid}" / "qwen3.7-plus" / "eval"
    if not t_eval.is_dir():
        cands = list((treat / f"libero_spatial_{tid}").glob("*/eval"))
        t_eval = cands[0] if cands else t_eval
    bh, bn = score(b_eval)
    th, tn = score(t_eval)
    rows.append({"task": tid, "baseline": f"{bh}/{bn}", "v24": f"{th}/{tn}"})
    b_tot += bh; b_n += bn; t_tot += th; t_n += tn
out = {
    "mode": "libero_g17_v24_spatial_10x10",
    "schema": "capx-libero-g17-fg-takeover-v2.4-public-in-iff",
    "baseline_job": "517994",
    "treatment_suite_root": str(treat),
    "per_task": rows,
    "baseline_total": f"{b_tot}/{b_n}",
    "v24_total": f"{t_tot}/{t_n}",
}
Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_v24_spatial_10x10_20260816/SUITE_SCORECARD.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
print(json.dumps(out, indent=2))
PYSCORE

exit "$failures"
