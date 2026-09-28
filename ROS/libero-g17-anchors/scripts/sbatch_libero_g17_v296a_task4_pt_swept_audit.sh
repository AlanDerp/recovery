#!/usr/bin/env bash
#SBATCH --job-name=libero-g17-v296a-task4-audit
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v256_20260817/logs/%x-%j.out
#SBATCH --error=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v256_20260817/logs/%x-%j.out

# v296a: behavior-identical v294 replay with robot-observation-only swept-path
# prediction attached to each GraspNet candidate.
# RELEASE_READY + NEARMISS_OSC + OSC_BANDWINDOW; start=0.15 step=0.03; no residual-vector snap.
# RELEASE_READY=1 + RELEASE_READY_TIPXY=1 (no goto / no tip-identity).
# Empty-hand same-name XY suppressed; wrong-grasp ordinal retained; skill keep-stock.
# Reuses frozen baseline reps from libero_g17_fair_fg_20260815 (do not re-run baseline).
# Set REPLICATE=0..K-1:
#   sbatch --export=ALL,ARM=v256,REPLICATE=0 scripts/sbatch_smoke_libero_g17_fair_fg_v256.sh
set -euo pipefail

# Slurm inherits the submitter environment.  Login-node localhost proxies are
# not present on compute nodes and make httpx/OpenAI connect to a dead
# 127.0.0.1:18080 instead of the Qwen endpoint.  All services used here are
# either directly reachable or job-local, so sanitize proxies before any
# Python client or local service starts.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

ARM=${ARM:-v256}
REPLICATE=${REPLICATE:-0}
TASK_ID=${TASK_ID:-0}
export ARM REPLICATE TASK_ID
case "$ARM" in
  v256|v269|v270|v271fg|v271pt|v272|v273|v274|v275|v276|v277|v278|v279|v280|v281|v282|v283|v284|v285|v286) ;;
  *)
    echo "unsupported ARM: $ARM" >&2
    exit 2
    ;;
esac
if ! [[ "$REPLICATE" =~ ^[0-9]+$ ]]; then
  echo "REPLICATE must be a non-negative integer, got: $REPLICATE" >&2
  exit 2
fi
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 9 )); then
  echo "TASK_ID must be an integer in 0..9, got: $TASK_ID" >&2
  exit 2
fi

G17=${CAPX_G17_WORKTREE_OVERRIDE:-/home/qshou/capx_g17_v296a_pt_swept_audit}
PROJECT_DIR=/project/peilab/qshou/cap-x-agilex
BASELINE_EXPT=/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813
if [[ "$ARM" == "v269" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v269_mech_20260818
  RUN_MODE=libero_g17_fair_fg_v269_mech
elif [[ "$ARM" == "v270" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v270_live_isolation_20260818
  RUN_MODE=libero_g17_fair_fg_v270_live_isolation
elif [[ "$ARM" == "v271fg" || "$ARM" == "v271pt" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_v271_mech_pt_pair_20260818
  RUN_MODE=libero_g17_v271_mech_pt_pair
elif [[ "$ARM" == "v272" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v272_clean_live_20260818
  RUN_MODE=libero_g17_fair_fg_v272_clean_live
elif [[ "$ARM" == "v273" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v273_spatial_generalization_20260818
  RUN_MODE=libero_g17_fair_fg_v273_spatial_generalization
elif [[ "$ARM" == "v274" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v274_mech_contract_20260818
  RUN_MODE=libero_g17_fair_fg_v274_mech_contract
elif [[ "$ARM" == "v275" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v275_trusted_tipxy_20260818
  RUN_MODE=libero_g17_fair_fg_v275_trusted_tipxy
elif [[ "$ARM" == "v276" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v276_completion_local_ik_20260819
  RUN_MODE=libero_g17_fair_fg_v276_completion_local_ik
elif [[ "$ARM" == "v277" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_v277_seed_repair_20260818
  RUN_MODE=libero_g17_v277_seed_repair
elif [[ "$ARM" == "v278" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v278_trusted_se3_20260819
  RUN_MODE=libero_g17_fair_fg_v278_trusted_se3
elif [[ "$ARM" == "v279" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v279_reachable_se3_20260819
  RUN_MODE=libero_g17_fair_fg_v279_reachable_se3
elif [[ "$ARM" == "v280" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v280_planned_se3_20260819
  RUN_MODE=libero_g17_fair_fg_v280_planned_se3
elif [[ "$ARM" == "v281" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v281_joint_join_20260819
  RUN_MODE=libero_g17_fair_fg_v281_joint_join
elif [[ "$ARM" == "v282" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v282_measured_seed_20260819
  RUN_MODE=libero_g17_fair_fg_v282_measured_seed
elif [[ "$ARM" == "v283" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v283_tool_pi_20260819
  RUN_MODE=libero_g17_fair_fg_v283_tool_pi
elif [[ "$ARM" == "v284" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v284_release_commit_20260819
  RUN_MODE=libero_g17_fair_fg_v284_release_commit
elif [[ "$ARM" == "v285" ]]; then
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v285_full_release_20260819
  RUN_MODE=libero_g17_fair_fg_v285_full_release
elif [[ "$ARM" == "v286" ]]; then
  EXPT_DIR=${CAPX_V296_EXPT_DIR_OVERRIDE:-/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_v296a_pt_swept_audit_20260820}
  RUN_MODE=libero_g17_v296a_pt_swept_audit
else
  EXPT_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v256_20260817
  RUN_MODE=libero_g17_fair_fg_v256_v24_contract_head_reports
fi
export EXPT_DIR RUN_MODE
export CAPX_LIBERO_G17_V293_ACQUISITION=1
export CAPX_LIBERO_G17_V293_REQUIRE_FEASIBLE_GRASP=0
export CAPX_LIBERO_G17_V296A_PT_SWEPT_AUDIT=1
if [[ "$ARM" == "v273" || ( ( "$ARM" == "v274" || "$ARM" == "v275" || "$ARM" == "v276" || "$ARM" == "v278" || "$ARM" == "v279" || "$ARM" == "v280" || "$ARM" == "v281" || "$ARM" == "v282" || "$ARM" == "v283" || "$ARM" == "v284" || "$ARM" == "v285" || "$ARM" == "v286" ) && "$TASK_ID" != "0" ) ]]; then
  PREFIX_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_spatial_prefix_replay_20260818/task_${TASK_ID}
else
  PREFIX_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_prefix_replay_20260815
fi
PREFIX_DIR=${CAPX_LIBERO_G17_PREFIX_DIR_OVERRIDE:-$PREFIX_DIR}
PREFIX_PKG="$PREFIX_DIR/code_prefix_replay_v1.json"
PREFIX_SHA_FILE="$PREFIX_DIR/PACKAGE_SHA256.txt"
GRASP_REPLAY_PKG=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_grasp_replay_v256_20260817/grasp_replay_v1.json
GRASP_REPLAY_SHA_FILE=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_grasp_replay_v256_20260817/PACKAGE_SHA256.txt
if [[ "$ARM" == "v277" ]]; then
  GRASP_REPLAY_PKG="$EXPT_DIR/assets/v272_rep01_trial02_first_grasp.json"
  GRASP_REPLAY_SHA_FILE="$EXPT_DIR/assets/FIRST_GRASP_SHA256.txt"
fi
if [[ "$ARM" == "v271fg" || "$ARM" == "v271pt" ]]; then
  V269_REPLAY_DIR=/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v269_mech_20260818/assets/effective_seed_replay
else
  V269_REPLAY_DIR="$EXPT_DIR/assets/effective_seed_replay"
fi
V269_REPLAY_PKG="$V269_REPLAY_DIR/v256_rep_$(printf '%02d' "$REPLICATE").json"
V269_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V270_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V271_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V272_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V273_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V274_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V275_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V276_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V277_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V278_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V279_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V280_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V281_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V282_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V283_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V284_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V285_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
V286_SOURCE_SHA="$EXPT_DIR/manifests/SOURCE_SHA256.json"
HELPER=/home/qshou/capx_causal_ik_replay_81e0acb/experiments/core_fg/runtime
PYTHON="$PROJECT_DIR/.venv-libero/bin/python"
KEY=/scratch/peilab/qshou/RoboTwin_Sandbox/cap-x/.qwenkey_sg_tmp
BASE_URL=https://ws-vpxr6jfor4t48uyo.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
MODEL=qwen3.7-plus
TOTAL_TRIALS=${TOTAL_TRIALS:-2}
START_TRIAL=${START_TRIAL:-1}
NUM_WORKERS=1
TEMPERATURE=0
GLOBAL_SEED=20260718
MOLMO_ROOT=/scratch/peilab/qshou/capx-molmo
MOLMO_VENV="$MOLMO_ROOT/venv"
MOLMO_MODEL=/scratch/peilab/qshou/capx-molmo/cache/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b
MOLMO_GPU_UTIL=${MOLMO_GPU_UTIL:-0.55}

suite_name=libero_spatial
task_id=$TASK_ID
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
# Use the frozen, complete Panda asset tree.  The default shared home cache can
# be observed mid-clone by child processes and then has no valid Git HEAD.
export ROBOT_DESCRIPTIONS_CACHE=/scratch/peilab/qshou/robot_descriptions_cache_g17_v24
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
  v256)
    export CAPX_LIBERO_G17_FG=1
    export CAPX_LIBERO_G17_WORKGAP_ME=0
    export CAPX_LIBERO_G17_XY_LOCK=1
    export CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1
    export CAPX_LIBERO_G17_RELEASE_READY=1
    export CAPX_LIBERO_G17_NEARMISS_OSC=1
    export CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW=1
    unset CAPX_LIBERO_G17_V269_MECH
    unset CAPX_LIBERO_G17_V269_SYMBOL_SOURCE
    unset CAPX_LIBERO_G17_V270_LIVE_ISOLATION
    unset CAPX_LIBERO_G17_V271_PT_PAIR
    unset CAPX_LIBERO_G17_V271_SYMBOL_SOURCE
    unset CAPX_LIBERO_G17_V272_FAIR_LIVE
    unset CAPX_LIBERO_G17_V274_MECH_CONTRACT
    unset CAPX_LIBERO_G17_V275_TRUSTED_TIPXY
    unset CAPX_LIBERO_G17_V276_COMPLETION_IK
    unset CAPX_LIBERO_G17_V277_CANONICAL_GRASP
    unset CAPX_LIBERO_G17_V278_TRUSTED_SE3
    unset CAPX_LIBERO_G17_V279_REACHABLE_SE3
    unset CAPX_LIBERO_G17_V280_PLANNED_SE3
    unset CAPX_LIBERO_G17_V281_JOINT_JOIN
    unset CAPX_LIBERO_G17_V282_MEASURED_SEED_IK
    unset CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY
    unset CAPX_LIBERO_G17_V284_RELEASE_COMMIT_LATCH
    unset CAPX_LIBERO_G17_V285_FULL_RELEASE_WINDOW
    unset CAPX_LIBERO_G17_V286_FINAL_FG_CHECK
    unset CAPX_LIBERO_G17_RELEASE_READY_TIPXY
    unset CAPX_LIBERO_G17_RELEASE_READY_TIPXY_BANDSTEP
    unset CAPX_LIBERO_G17_RELEASE_READY_TIPXY_FAIR_RESIDUAL
    unset CAPX_LIBERO_G17_NEARMISS_FAILOPEN
    unset CAPX_LIBERO_G17_FAIR_PLACE_SERVO
    unset CAPX_LIBERO_G17_RELEASE_READY_GOTO
    unset CAPX_LIBERO_G17_GOTO_TIP_IDENTITY
    unset CAPX_LIBERO_G17_BAND_OBLIGATIONS
    unset CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE
    unset CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER
    unset CAPX_LIBERO_G17_GOAL_IDENTITY_VERIFY
    unset CAPX_LIBERO_G17_TARGET_GRASP_OBJECT
    ;;
  v269|v270|v271fg|v271pt|v272|v273|v274|v275|v276|v277|v278|v279|v280|v281|v282|v283|v284|v285|v286)
    export CAPX_LIBERO_G17_FG=1
    export CAPX_LIBERO_G17_WORKGAP_ME=0
    export CAPX_LIBERO_G17_XY_LOCK=1
    export CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL=1
    export CAPX_LIBERO_G17_RELEASE_READY=1
    export CAPX_LIBERO_G17_NEARMISS_OSC=1
    export CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW=1
    export CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH=1
    export CAPX_LIBERO_G17_V269_MECH=1
    export CAPX_LIBERO_G17_V269_SYMBOL_SOURCE=fg
    if [[ "$ARM" == "v270" ]]; then
      export CAPX_LIBERO_G17_V270_LIVE_ISOLATION=1
    else
      unset CAPX_LIBERO_G17_V270_LIVE_ISOLATION
    fi
    if [[ "$ARM" == "v272" || "$ARM" == "v273" || "$ARM" == "v274" || "$ARM" == "v275" || "$ARM" == "v276" || "$ARM" == "v277" || "$ARM" == "v278" || "$ARM" == "v279" || "$ARM" == "v280" || "$ARM" == "v281" || "$ARM" == "v282" || "$ARM" == "v283" || "$ARM" == "v284" || "$ARM" == "v285" || "$ARM" == "v286" ]]; then
      export CAPX_LIBERO_G17_V272_FAIR_LIVE=1
    else
      unset CAPX_LIBERO_G17_V272_FAIR_LIVE
    fi
    if [[ "$ARM" == "v271fg" || "$ARM" == "v271pt" ]]; then
      export CAPX_LIBERO_G17_V271_PT_PAIR=1
      if [[ "$ARM" == "v271fg" ]]; then
        export CAPX_LIBERO_G17_V271_SYMBOL_SOURCE=fg
      else
        export CAPX_LIBERO_G17_V271_SYMBOL_SOURCE=pt
      fi
    else
      unset CAPX_LIBERO_G17_V271_PT_PAIR
      unset CAPX_LIBERO_G17_V271_SYMBOL_SOURCE
    fi
    if [[ "$ARM" == "v272" || "$ARM" == "v273" || "$ARM" == "v274" || "$ARM" == "v275" || "$ARM" == "v276" || "$ARM" == "v278" || "$ARM" == "v279" || "$ARM" == "v280" || "$ARM" == "v281" || "$ARM" == "v282" || "$ARM" == "v283" || "$ARM" == "v284" || "$ARM" == "v285" || "$ARM" == "v286" ]]; then
      unset CAPX_LIBERO_G17_GRASP_REPLAY
      unset CAPX_LIBERO_G17_GRASP_REPLAY_STRICT
      unset CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW
      unset CAPX_LIBERO_G17_GRASP_REPLAY_VERBOSE
    else
      export CAPX_LIBERO_G17_GRASP_REPLAY="$GRASP_REPLAY_PKG"
      if [[ "$ARM" == "v277" ]]; then
        export CAPX_LIBERO_G17_GRASP_REPLAY_STRICT=0
        export CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW=live
      else
        export CAPX_LIBERO_G17_GRASP_REPLAY_STRICT=1
        export CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW=repeat_last
      fi
      export CAPX_LIBERO_G17_GRASP_REPLAY_VERBOSE=0
    fi
    if [[ "$ARM" == "v274" || "$ARM" == "v275" || "$ARM" == "v276" || "$ARM" == "v278" || "$ARM" == "v279" || "$ARM" == "v280" || "$ARM" == "v281" || "$ARM" == "v282" || "$ARM" == "v283" || "$ARM" == "v284" || "$ARM" == "v285" || "$ARM" == "v286" ]]; then
      export CAPX_LIBERO_G17_V274_MECH_CONTRACT=1
    else
      unset CAPX_LIBERO_G17_V274_MECH_CONTRACT
    fi
    if [[ "$ARM" == "v275" ]]; then
      export CAPX_LIBERO_G17_V275_TRUSTED_TIPXY=1
    else
      unset CAPX_LIBERO_G17_V275_TRUSTED_TIPXY
    fi
    if [[ "$ARM" == "v276" || "$ARM" == "v278" || "$ARM" == "v279" || "$ARM" == "v280" || "$ARM" == "v281" || "$ARM" == "v282" || "$ARM" == "v283" || "$ARM" == "v284" || "$ARM" == "v285" || "$ARM" == "v286" ]]; then
      export CAPX_LIBERO_G17_V276_COMPLETION_IK=1
    else
      unset CAPX_LIBERO_G17_V276_COMPLETION_IK
    fi
    if [[ "$ARM" == "v277" ]]; then
      export CAPX_LIBERO_G17_V277_CANONICAL_GRASP=1
    else
      unset CAPX_LIBERO_G17_V277_CANONICAL_GRASP
    fi
    if [[ "$ARM" == "v278" ]]; then
      export CAPX_LIBERO_G17_V278_TRUSTED_SE3=1
    else
      unset CAPX_LIBERO_G17_V278_TRUSTED_SE3
    fi
    if [[ "$ARM" == "v279" ]]; then
      export CAPX_LIBERO_G17_V279_REACHABLE_SE3=1
    else
      unset CAPX_LIBERO_G17_V279_REACHABLE_SE3
    fi
    if [[ "$ARM" == "v280" ]]; then
      export CAPX_LIBERO_G17_V280_PLANNED_SE3=1
    else
      unset CAPX_LIBERO_G17_V280_PLANNED_SE3
    fi
    if [[ "$ARM" == "v281" ]]; then
      export CAPX_LIBERO_G17_V281_JOINT_JOIN=1
    else
      unset CAPX_LIBERO_G17_V281_JOINT_JOIN
    fi
    if [[ "$ARM" == "v282" ]]; then
      export CAPX_LIBERO_G17_V282_MEASURED_SEED_IK=1
    else
      unset CAPX_LIBERO_G17_V282_MEASURED_SEED_IK
    fi
    if [[ "$ARM" == "v283" || "$ARM" == "v284" || "$ARM" == "v285" || "$ARM" == "v286" ]]; then
      export CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY=1
    else
      unset CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY
    fi
    if [[ "$ARM" == "v284" ]]; then
      export CAPX_LIBERO_G17_V284_RELEASE_COMMIT_LATCH=1
    else
      unset CAPX_LIBERO_G17_V284_RELEASE_COMMIT_LATCH
    fi
    if [[ "$ARM" == "v285" || "$ARM" == "v286" ]]; then
      export CAPX_LIBERO_G17_V285_FULL_RELEASE_WINDOW=1
    else
      unset CAPX_LIBERO_G17_V285_FULL_RELEASE_WINDOW
    fi
    if [[ "$ARM" == "v286" ]]; then
      export CAPX_LIBERO_G17_V286_FINAL_FG_CHECK=1
    else
      unset CAPX_LIBERO_G17_V286_FINAL_FG_CHECK
    fi
    unset CAPX_LIBERO_G17_RELEASE_READY_NO_SILENT_ALLOW
    unset CAPX_LIBERO_G17_RELEASE_READY_NEARMISS_GATE
    unset CAPX_LIBERO_G17_RELEASE_READY_HOLD_GUARD
    unset CAPX_LIBERO_G17_NEARMISS_OSC_INSCRIBED_SQUARE
    unset CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT
    unset CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE
    unset CAPX_LIBERO_G17_NEARMISS_OSC_STUDENT_SIGN
    unset CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL
    unset CAPX_LIBERO_G17_RELEASE_READY_TIPXY
    unset CAPX_LIBERO_G17_RELEASE_READY_TIPXY_BANDSTEP
    unset CAPX_LIBERO_G17_RELEASE_READY_TIPXY_FAIR_RESIDUAL
    unset CAPX_LIBERO_G17_NEARMISS_FAILOPEN
    unset CAPX_LIBERO_G17_FAIR_PLACE_SERVO
    unset CAPX_LIBERO_G17_RELEASE_READY_GOTO
    unset CAPX_LIBERO_G17_GOTO_TIP_IDENTITY
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

if [[ "$ARM" == "v269" ]]; then
  # Freeze mechanism assets and executable sources at submission time.
  "$PYTHON" - <<PYV269ASSET
import hashlib, json
from pathlib import Path

grasp = Path("$GRASP_REPLAY_PKG")
if hashlib.sha256(grasp.read_bytes()).hexdigest() != Path("$GRASP_REPLAY_SHA_FILE").read_text().strip():
    raise SystemExit("v269 grasp replay package drift")
replay = Path("$V269_REPLAY_PKG")
package = json.loads(replay.read_text())
if package.get("schema") != "capx_libero_g17_v269_effective_seed_replay_v1":
    raise SystemExit(f"v269 replay schema mismatch: {package.get('schema')!r}")
source_manifest = json.loads(Path("$V269_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v269 source drift: {path} {actual} != {expected}")
print("v269_assets_ok", replay, package["entries_count"])
print("v269_source_fingerprints_ok", len(source_manifest["files"]))
PYV269ASSET
fi

if [[ "$ARM" == "v270" ]]; then
  # v270 keeps the v269 mechanical controller and grasp asset, but the student
  # Qwen path is live.  Freeze all executable sources before scheduling.
  "$PYTHON" - <<PYV270ASSET
import hashlib, json
from pathlib import Path

grasp = Path("$GRASP_REPLAY_PKG")
if hashlib.sha256(grasp.read_bytes()).hexdigest() != Path("$GRASP_REPLAY_SHA_FILE").read_text().strip():
    raise SystemExit("v270 grasp replay package drift")
source_manifest = json.loads(Path("$V270_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v270 source drift: {path} {actual} != {expected}")
print("v270_source_fingerprints_ok", len(source_manifest["files"]))
PYV270ASSET
fi

if [[ "$ARM" == "v271fg" || "$ARM" == "v271pt" ]]; then
  "$PYTHON" - <<PYV271ASSET
import hashlib, json
from pathlib import Path

grasp = Path("$GRASP_REPLAY_PKG")
if hashlib.sha256(grasp.read_bytes()).hexdigest() != Path("$GRASP_REPLAY_SHA_FILE").read_text().strip():
    raise SystemExit("v271 grasp replay package drift")
replay = Path("$V269_REPLAY_PKG")
package = json.loads(replay.read_text())
if package.get("schema") != "capx_libero_g17_v269_effective_seed_replay_v1":
    raise SystemExit(f"v271 replay schema mismatch: {package.get('schema')!r}")
source_manifest = json.loads(Path("$V271_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v271 source drift: {path} {actual} != {expected}")
print("v271_assets_ok", replay, package["entries_count"])
print("v271_source_fingerprints_ok", len(source_manifest["files"]))
PYV271ASSET
fi

if [[ "$ARM" == "v272" ]]; then
  "$PYTHON" - <<PYV272ASSET
import hashlib, json
from pathlib import Path
source_manifest = json.loads(Path("$V272_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v272 source drift: {path} {actual} != {expected}")
print("v272_source_fingerprints_ok", len(source_manifest["files"]))
PYV272ASSET
fi

if [[ "$ARM" == "v273" ]]; then
  "$PYTHON" - <<PYV273ASSET
import hashlib, json
from pathlib import Path
source_manifest = json.loads(Path("$V273_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v273 source drift: {path} {actual} != {expected}")
package = json.loads(Path("$PREFIX_PKG").read_text())
if package.get("suite_name") != "libero_spatial" or int(package.get("task_id", -1)) != int("$TASK_ID"):
    raise SystemExit(f"v273 task-local prefix mismatch: task={package.get('task_id')} expected=$TASK_ID")
print("v273_source_fingerprints_ok", len(source_manifest["files"]))
print("v273_task_local_prefix_ok", package["task_id"])
PYV273ASSET
fi

if [[ "$ARM" == "v274" ]]; then
  "$PYTHON" - <<PYV274ASSET
import hashlib, json
from pathlib import Path
source_manifest = json.loads(Path("$V274_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v274 source drift: {path} {actual} != {expected}")
package = json.loads(Path("$PREFIX_PKG").read_text())
if package.get("suite_name") != "libero_spatial" or int(package.get("task_id", -1)) != int("$TASK_ID"):
    raise SystemExit(f"v274 task-local prefix mismatch: task={package.get('task_id')} expected=$TASK_ID")
print("v274_source_fingerprints_ok", len(source_manifest["files"]))
print("v274_task_local_prefix_ok", package["task_id"])
PYV274ASSET
fi

if [[ "$ARM" == "v275" ]]; then
  "$PYTHON" - <<PYV275ASSET
import hashlib, json
from pathlib import Path
source_manifest = json.loads(Path("$V275_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v275 source drift: {path} {actual} != {expected}")
package = json.loads(Path("$PREFIX_PKG").read_text())
if package.get("suite_name") != "libero_spatial" or int(package.get("task_id", -1)) != int("$TASK_ID"):
    raise SystemExit(f"v275 task-local prefix mismatch: task={package.get('task_id')} expected=$TASK_ID")
print("v275_source_fingerprints_ok", len(source_manifest["files"]))
print("v275_task_local_prefix_ok", package["task_id"])
PYV275ASSET
fi

if [[ "$ARM" == "v276" ]]; then
  "$PYTHON" - <<PYV276ASSET
import hashlib, json
from pathlib import Path
source_manifest = json.loads(Path("$V276_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v276 source drift: {path} {actual} != {expected}")
package = json.loads(Path("$PREFIX_PKG").read_text())
if package.get("suite_name") != "libero_spatial" or int(package.get("task_id", -1)) != int("$TASK_ID"):
    raise SystemExit(f"v276 task-local prefix mismatch: task={package.get('task_id')} expected=$TASK_ID")
print("v276_source_fingerprints_ok", len(source_manifest["files"]))
print("v276_task_local_prefix_ok", package["task_id"])
PYV276ASSET
fi

if [[ "$ARM" == "v277" ]]; then
  "$PYTHON" - <<PYV277ASSET
import hashlib, json
from pathlib import Path
source_manifest = json.loads(Path("$V277_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v277 source drift: {path} {actual} != {expected}")
grasp = Path("$GRASP_REPLAY_PKG")
if hashlib.sha256(grasp.read_bytes()).hexdigest() != Path("$GRASP_REPLAY_SHA_FILE").read_text().strip():
    raise SystemExit("v277 first-grasp package drift")
package = json.loads(grasp.read_text())
if package.get("schema") != "capx_libero_g17_grasp_replay_v1":
    raise SystemExit("v277 first-grasp package schema mismatch")
print("v277_source_fingerprints_ok", len(source_manifest["files"]))
print("v277_first_grasp_ok", grasp)
PYV277ASSET
fi

if [[ "$ARM" == "v278" ]]; then
  "$PYTHON" - <<PYV278ASSET
import hashlib, json
from pathlib import Path
source_manifest = json.loads(Path("$V278_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v278 source drift: {path} {actual} != {expected}")
package = json.loads(Path("$PREFIX_PKG").read_text())
if package.get("suite_name") != "libero_spatial" or int(package.get("task_id", -1)) != int("$TASK_ID"):
    raise SystemExit(f"v278 task-local prefix mismatch: task={package.get('task_id')} expected=$TASK_ID")
print("v278_source_fingerprints_ok", len(source_manifest["files"]))
print("v278_task_local_prefix_ok", package["task_id"])
PYV278ASSET
fi

if [[ "$ARM" == "v279" ]]; then
  "$PYTHON" - <<PYV279ASSET
import hashlib, json
from pathlib import Path
source_manifest = json.loads(Path("$V279_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v279 source drift: {path} {actual} != {expected}")
package = json.loads(Path("$PREFIX_PKG").read_text())
if package.get("suite_name") != "libero_spatial" or int(package.get("task_id", -1)) != int("$TASK_ID"):
    raise SystemExit(f"v279 task-local prefix mismatch: task={package.get('task_id')} expected=$TASK_ID")
audit = json.loads(Path("$EXPT_DIR/evidence/v272_trial2_reachable_se3.json").read_text())
if audit.get("schema") != "capx_v279_exact_reachable_se3_audit_v1" or not audit.get("reachable", {}).get("ok"):
    raise SystemExit("v279 exact mechanical audit missing or invalid")
print("v279_source_fingerprints_ok", len(source_manifest["files"]))
print("v279_task_local_prefix_ok", package["task_id"])
print("v279_exact_reachability_audit_ok", audit["reachable"]["selected_route"])
PYV279ASSET
fi

if [[ "$ARM" == "v280" ]]; then
  "$PYTHON" - <<PYV280ASSET
import hashlib, json
from pathlib import Path
source_manifest = json.loads(Path("$V280_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v280 source drift: {path} {actual} != {expected}")
package = json.loads(Path("$PREFIX_PKG").read_text())
if package.get("suite_name") != "libero_spatial" or int(package.get("task_id", -1)) != int("$TASK_ID"):
    raise SystemExit(f"v280 task-local prefix mismatch: task={package.get('task_id')} expected=$TASK_ID")
audit = json.loads(Path("$EXPT_DIR/evidence/v272_trial2_lagged_planned_se3.json").read_text())
if audit.get("schema") != "capx_v280_lagged_planned_se3_audit_v1" or not audit.get("planned", {}).get("ok"):
    raise SystemExit("v280 lagged mechanical audit missing or invalid")
print("v280_source_fingerprints_ok", len(source_manifest["files"]))
print("v280_task_local_prefix_ok", package["task_id"])
print("v280_lagged_planned_audit_ok", audit["planned"]["selected_route"])
PYV280ASSET
fi

if [[ "$ARM" == "v281" ]]; then
  "$PYTHON" - <<PYV281ASSET
import hashlib, json
from pathlib import Path
source_manifest = json.loads(Path("$V281_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v281 source drift: {path} {actual} != {expected}")
package = json.loads(Path("$PREFIX_PKG").read_text())
if package.get("suite_name") != "libero_spatial" or int(package.get("task_id", -1)) != int("$TASK_ID"):
    raise SystemExit(f"v281 task-local prefix mismatch: task={package.get('task_id')} expected=$TASK_ID")
audit = json.loads(Path("$EXPT_DIR/evidence/v272_trial2_lagged_joint_join.json").read_text())
if audit.get("schema") != "capx_v281_lagged_joint_join_audit_v1" or not audit.get("planned", {}).get("ok"):
    raise SystemExit("v281 lagged joint-join audit missing or invalid")
print("v281_source_fingerprints_ok", len(source_manifest["files"]))
print("v281_task_local_prefix_ok", package["task_id"])
print("v281_lagged_joint_join_audit_ok", audit["planned"]["selected_route"])
PYV281ASSET
fi

if [[ "$ARM" == "v282" ]]; then
  "$PYTHON" - <<PYV282ASSET
import hashlib, json
from pathlib import Path
source_manifest = json.loads(Path("$V282_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v282 source drift: {path} {actual} != {expected}")
package = json.loads(Path("$PREFIX_PKG").read_text())
if package.get("suite_name") != "libero_spatial" or int(package.get("task_id", -1)) != int("$TASK_ID"):
    raise SystemExit(f"v282 task-local prefix mismatch: task={package.get('task_id')} expected=$TASK_ID")
audit = json.loads(Path("$EXPT_DIR/evidence/v272_trial2_measured_seed_ik.json").read_text())
if audit.get("schema") != "capx_v282_measured_seed_ik_audit_v1" or not audit.get("native_execution", {}).get("accepted"):
    raise SystemExit("v282 measured-seed IK audit missing or invalid")
print("v282_source_fingerprints_ok", len(source_manifest["files"]))
print("v282_task_local_prefix_ok", package["task_id"])
print("v282_measured_seed_audit_ok", audit["native_execution"]["total_blocking_commands"])
PYV282ASSET
fi

if [[ "$ARM" == "v283" || "$ARM" == "v284" || "$ARM" == "v285" || "$ARM" == "v286" ]]; then
  "$PYTHON" - <<PYV283ASSET
import hashlib, json
from pathlib import Path
source_manifest = json.loads(Path("$V283_SOURCE_SHA").read_text())
for raw_path, expected in source_manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v283 source drift: {path} {actual} != {expected}")
package = json.loads(Path("$PREFIX_PKG").read_text())
if package.get("suite_name") != "libero_spatial" or int(package.get("task_id", -1)) != int("$TASK_ID"):
    raise SystemExit(f"v283 task-local prefix mismatch: task={package.get('task_id')} expected=$TASK_ID")
audit = json.loads(Path(
    "/project/peilab/qshou/cap-x-agilex/outputs/"
    "libero_g17_v291_task4_acquisition_20260820/evidence/"
    "v272_trial2_tool_pi_ik.json"
).read_text())
if audit.get("schema") != "capx_v282_measured_seed_ik_audit_v1" or not audit.get("native_execution", {}).get("accepted"):
    raise SystemExit("v283 tool-pi IK audit missing or invalid")
if audit.get("contract", {}).get("tool_z_pi_symmetry") != 1:
    raise SystemExit("v283 tool-pi symmetry contract missing")
print("v283_source_fingerprints_ok", len(source_manifest["files"]))
print("v283_task_local_prefix_ok", package["task_id"])
print("v283_tool_pi_audit_ok", audit["native_execution"]["total_blocking_commands"])
PYV283ASSET
fi

# G17 overlay sitecustomize FIRST: disables editable agilex capx finder, then
# chains the HELPER reproducibility sitecustomize.
export PYTHONPATH="$G17/python_sitecustomize:$G17:$HELPER/python_sitecustomize:$HELPER:$PROJECT_DIR:${PYTHONPATH:-}"

cd "$G17"
# Keep agilex as CAPX project for configs/venv only; sources come from G17 overlay.
source "$PROJECT_DIR/.venv-libero/bin/activate"

[[ -x "$PYTHON" ]]
if [[ "$ARM" != "v269" ]]; then
  [[ -s "$KEY" ]]
fi
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
echo "pair_control=baseline job-517994 ${task_name} trials=1..$TOTAL_TRIALS"
echo "ports llm=$llm_port sam3=$sam3_port graspnet=$graspnet_port pyroki=$pyroki_port molmo=$molmo_port"
echo "MOLMO_SERVICE_URL=$MOLMO_SERVICE_URL"
nvidia-smi -L | head -3

echo "[$(date '+%F %T')] prove G17 trial.py wins over editable agilex"
"$PYTHON" - <<'PYPROOF'
import os
import capx.envs.trial as trial
path = trial.__file__
print("trial_module", path)
expected_root = os.path.realpath(
    os.environ.get(
        "CAPX_G17_WORKTREE_OVERRIDE",
        "/home/qshou/capx_g17_v296a_pt_swept_audit",
    )
)
if os.path.commonpath([os.path.realpath(path), expected_root]) != expected_root:
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
    if arm == "v269":
        if "v2.69" not in schema or "v256-shared-mech-fg-source" not in schema:
            raise SystemExit(f"FG INVALID: v269 shared-mech schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v269 {key} != {expected!r}")
        forbidden = (
            "CAPX_LIBERO_G17_RELEASE_READY_NO_SILENT_ALLOW",
            "CAPX_LIBERO_G17_RELEASE_READY_NEARMISS_GATE",
            "CAPX_LIBERO_G17_RELEASE_READY_HOLD_GUARD",
            "CAPX_LIBERO_G17_NEARMISS_OSC_INSCRIBED_SQUARE",
            "CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT",
            "CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE",
        )
        for key in forbidden:
            if os.environ.get(key, "0") == "1":
                raise SystemExit(f"FG INVALID: v269 forbids {key}")
    elif arm == "v270":
        if "v2.70" not in schema or "live-isolation" not in schema:
            raise SystemExit(f"FG INVALID: v270 live-isolation schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V270_LIVE_ISOLATION": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v270 {key} != {expected!r}")
    elif arm in ("v271fg", "v271pt"):
        selected = "fg" if arm == "v271fg" else "pt"
        if "v2.71" not in schema or f"{selected}-source-pt-paired" not in schema:
            raise SystemExit(f"FG INVALID: v271 paired schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_V271_PT_PAIR": "1",
            "CAPX_LIBERO_G17_V271_SYMBOL_SOURCE": selected,
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v271 {key} != {expected!r}")
        if os.environ.get("CAPX_LIBERO_G17_V270_LIVE_ISOLATION", "0") == "1":
            raise SystemExit("FG INVALID: v271 must not enable v270 live isolation")
    elif arm == "v286":
        if "v2.86" not in schema or "final-fg-axis-check" not in schema:
            raise SystemExit(f"FG INVALID: v286 final-FG schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
            "CAPX_LIBERO_G17_V276_COMPLETION_IK": "1",
            "CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY": "1",
            "CAPX_LIBERO_G17_V285_FULL_RELEASE_WINDOW": "1",
            "CAPX_LIBERO_G17_V286_FINAL_FG_CHECK": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v286 {key} != {expected!r}")
        from capx.integrations.franka import measured_seed_ik
        contract = measured_seed_ik.measured_seed_ik_contract()
        if contract["native_blocking_commands_per_stage"] != 1 or contract["tool_z_pi_symmetry"] != 1:
            raise SystemExit("FG INVALID: v286 changed v283 IK contract")
    elif arm == "v285":
        if "v2.85" not in schema or "full-release-window" not in schema:
            raise SystemExit(f"FG INVALID: v285 full-release schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
            "CAPX_LIBERO_G17_V276_COMPLETION_IK": "1",
            "CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY": "1",
            "CAPX_LIBERO_G17_V285_FULL_RELEASE_WINDOW": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v285 {key} != {expected!r}")
        if os.environ.get("CAPX_LIBERO_G17_V284_RELEASE_COMMIT_LATCH", "0") == "1":
            raise SystemExit("FG INVALID: v285 must isolate v284 flag")
        from capx.integrations.franka import measured_seed_ik
        contract = measured_seed_ik.measured_seed_ik_contract()
        if (
            contract["native_blocking_commands_per_stage"] != 1
            or contract["tool_z_pi_symmetry"] != 1
            or contract["cartesian_waypoints"] != 0
            or contract["joint_delta_gate"] != 0
            or contract["terminal_residual_gate"] != 0
        ):
            raise SystemExit("FG INVALID: v285 changed v283 IK contract")
    elif arm == "v284":
        if "v2.84" not in schema or "release-commit-latch" not in schema:
            raise SystemExit(f"FG INVALID: v284 release-commit schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
            "CAPX_LIBERO_G17_V276_COMPLETION_IK": "1",
            "CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY": "1",
            "CAPX_LIBERO_G17_V284_RELEASE_COMMIT_LATCH": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v284 {key} != {expected!r}")
        from capx.integrations.franka import measured_seed_ik
        contract = measured_seed_ik.measured_seed_ik_contract()
        if (
            contract["native_blocking_commands_per_stage"] != 1
            or contract["tool_z_pi_symmetry"] != 1
            or contract["cartesian_waypoints"] != 0
            or contract["joint_delta_gate"] != 0
            or contract["terminal_residual_gate"] != 0
        ):
            raise SystemExit("FG INVALID: v284 changed v283 IK contract")
    elif arm == "v283":
        if "v2.83" not in schema or "measured-seed-tool-pi-native-once" not in schema:
            raise SystemExit(f"FG INVALID: v283 tool-pi schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
            "CAPX_LIBERO_G17_V276_COMPLETION_IK": "1",
            "CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v283 {key} != {expected!r}")
        for key in (
            "CAPX_LIBERO_G17_GRASP_REPLAY", "CAPX_LIBERO_G17_V277_CANONICAL_GRASP",
            "CAPX_LIBERO_G17_V275_TRUSTED_TIPXY", "CAPX_LIBERO_G17_V278_TRUSTED_SE3",
            "CAPX_LIBERO_G17_V279_REACHABLE_SE3", "CAPX_LIBERO_G17_V280_PLANNED_SE3",
            "CAPX_LIBERO_G17_V281_JOINT_JOIN", "CAPX_LIBERO_G17_V282_MEASURED_SEED_IK",
        ):
            if os.environ.get(key, "").strip():
                raise SystemExit(f"FG INVALID: v283 forbids {key}")
        from capx.integrations.franka import measured_seed_ik
        contract = measured_seed_ik.measured_seed_ik_contract()
        if (
            contract["native_blocking_commands_per_stage"] != 1
            or contract["tool_z_pi_symmetry"] != 1
            or contract["cartesian_waypoints"] != 0
            or contract["joint_delta_gate"] != 0
            or contract["terminal_residual_gate"] != 0
        ):
            raise SystemExit("FG INVALID: v283 tool-pi contract mismatch")
    elif arm == "v282":
        if "v2.82" not in schema or "measured-seed-native-once" not in schema:
            raise SystemExit(f"FG INVALID: v282 measured-seed schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
            "CAPX_LIBERO_G17_V276_COMPLETION_IK": "1",
            "CAPX_LIBERO_G17_V282_MEASURED_SEED_IK": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v282 {key} != {expected!r}")
        for key in (
            "CAPX_LIBERO_G17_GRASP_REPLAY",
            "CAPX_LIBERO_G17_V277_CANONICAL_GRASP",
            "CAPX_LIBERO_G17_V275_TRUSTED_TIPXY",
            "CAPX_LIBERO_G17_V278_TRUSTED_SE3",
            "CAPX_LIBERO_G17_V279_REACHABLE_SE3",
            "CAPX_LIBERO_G17_V280_PLANNED_SE3",
            "CAPX_LIBERO_G17_V281_JOINT_JOIN",
        ):
            if os.environ.get(key, "").strip():
                raise SystemExit(f"FG INVALID: v282 forbids {key}")
        from capx.integrations.franka import measured_seed_ik
        contract = measured_seed_ik.measured_seed_ik_contract()
        if (
            contract["schema"] != "capx_libero_measured_seed_ik_contract_v1"
            or contract["native_blocking_commands_per_stage"] != 1
            or contract["cartesian_waypoints"] != 0
            or contract["joint_delta_gate"] != 0
            or contract["terminal_residual_gate"] != 0
        ):
            raise SystemExit("FG INVALID: v282 measured-seed contract mismatch")
    elif arm == "v281":
        if "v2.81" not in schema or "command-faithful-joint-join" not in schema:
            raise SystemExit(f"FG INVALID: v281 joint join schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
            "CAPX_LIBERO_G17_V276_COMPLETION_IK": "1",
            "CAPX_LIBERO_G17_V281_JOINT_JOIN": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v281 {key} != {expected!r}")
        for key in (
            "CAPX_LIBERO_G17_GRASP_REPLAY",
            "CAPX_LIBERO_G17_V277_CANONICAL_GRASP",
            "CAPX_LIBERO_G17_V275_TRUSTED_TIPXY",
            "CAPX_LIBERO_G17_V278_TRUSTED_SE3",
            "CAPX_LIBERO_G17_V279_REACHABLE_SE3",
            "CAPX_LIBERO_G17_V280_PLANNED_SE3",
        ):
            if os.environ.get(key, "").strip():
                raise SystemExit(f"FG INVALID: v281 forbids {key}")
        from capx.integrations.franka import planned_se3
        if planned_se3.command_faithful_join_contract()["schema"] != "capx_libero_planned_se3_contract_v2":
            raise SystemExit("FG INVALID: v281 joint join contract mismatch")
    elif arm == "v280":
        if "v2.80" not in schema or "command-faithful-planned-se3" not in schema:
            raise SystemExit(f"FG INVALID: v280 planned SE3 schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
            "CAPX_LIBERO_G17_V276_COMPLETION_IK": "1",
            "CAPX_LIBERO_G17_V280_PLANNED_SE3": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v280 {key} != {expected!r}")
        for key in (
            "CAPX_LIBERO_G17_GRASP_REPLAY",
            "CAPX_LIBERO_G17_V277_CANONICAL_GRASP",
            "CAPX_LIBERO_G17_V275_TRUSTED_TIPXY",
            "CAPX_LIBERO_G17_V278_TRUSTED_SE3",
            "CAPX_LIBERO_G17_V279_REACHABLE_SE3",
        ):
            if os.environ.get(key, "").strip():
                raise SystemExit(f"FG INVALID: v280 forbids {key}")
        from capx.integrations.franka import planned_se3
        if planned_se3.planned_se3_contract()["schema"] != "capx_libero_planned_se3_contract_v1":
            raise SystemExit("FG INVALID: v280 planned SE3 contract mismatch")
    elif arm == "v279":
        if "v2.79" not in schema or "trusted-se3-fullpath-neutral-recovery" not in schema:
            raise SystemExit(f"FG INVALID: v279 reachable SE3 schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
            "CAPX_LIBERO_G17_V276_COMPLETION_IK": "1",
            "CAPX_LIBERO_G17_V279_REACHABLE_SE3": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v279 {key} != {expected!r}")
        for key in (
            "CAPX_LIBERO_G17_GRASP_REPLAY",
            "CAPX_LIBERO_G17_V277_CANONICAL_GRASP",
            "CAPX_LIBERO_G17_V275_TRUSTED_TIPXY",
            "CAPX_LIBERO_G17_V278_TRUSTED_SE3",
        ):
            if os.environ.get(key, "").strip():
                raise SystemExit(f"FG INVALID: v279 forbids {key}")
        from capx.integrations.franka import reachable_se3
        if reachable_se3.reachable_se3_contract()["schema"] != "capx_libero_reachable_se3_contract_v1":
            raise SystemExit("FG INVALID: v279 reachable SE3 contract mismatch")
    elif arm == "v278":
        if "v2.78" not in schema or "completion-local-ik-trusted-se3" not in schema:
            raise SystemExit(f"FG INVALID: v278 trusted SE3 schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
            "CAPX_LIBERO_G17_V276_COMPLETION_IK": "1",
            "CAPX_LIBERO_G17_V278_TRUSTED_SE3": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v278 {key} != {expected!r}")
        for key in (
            "CAPX_LIBERO_G17_GRASP_REPLAY",
            "CAPX_LIBERO_G17_V277_CANONICAL_GRASP",
            "CAPX_LIBERO_G17_V275_TRUSTED_TIPXY",
        ):
            if os.environ.get(key, "").strip():
                raise SystemExit(f"FG INVALID: v278 forbids {key}")
        from capx.integrations.franka import trusted_se3
        if trusted_se3.trusted_se3_contract()["schema"] != "capx_libero_trusted_se3_contract_v1":
            raise SystemExit("FG INVALID: v278 trusted SE3 contract mismatch")
    elif arm == "v277":
        if "v2.77" not in schema or "single-seed-canonical-world-z-grasp" not in schema:
            raise SystemExit(f"FG INVALID: v277 grasp repair schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V277_CANONICAL_GRASP": "1",
            "CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW": "live",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v277 {key} != {expected!r}")
        if not os.environ.get("CAPX_LIBERO_G17_GRASP_REPLAY", "").strip():
            raise SystemExit("FG INVALID: v277 requires frozen first grasp")
    elif arm == "v276":
        if "v2.76" not in schema or "fair-live-inscribed-completion-local-ik" not in schema:
            raise SystemExit(f"FG INVALID: v276 completion IK schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
            "CAPX_LIBERO_G17_V276_COMPLETION_IK": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v276 {key} != {expected!r}")
        if os.environ.get("CAPX_LIBERO_G17_V275_TRUSTED_TIPXY", "0") == "1":
            raise SystemExit("FG INVALID: v276 must not enable v275 flag")
        for key in (
            "CAPX_LIBERO_G17_GRASP_REPLAY",
            "CAPX_LIBERO_G17_GRASP_REPLAY_STRICT",
            "CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW",
            "CAPX_LIBERO_G17_V270_LIVE_ISOLATION",
            "CAPX_LIBERO_G17_V271_PT_PAIR",
        ):
            if os.environ.get(key, "").strip():
                raise SystemExit(f"FG INVALID: v276 forbids {key}")
        from capx.integrations.franka import tip_xy_only
        if not hasattr(tip_xy_only, "move_tip_xy"):
            raise SystemExit("FG INVALID: v276 completion primitive missing")
        if tip_xy_only.trusted_tip_xy_contract()["schema"] != "capx_completion_tip_xy_contract_v2":
            raise SystemExit("FG INVALID: v276 completion contract mismatch")
    elif arm == "v275":
        if "v2.75" not in schema or "fair-live-inscribed-transactional-tipxy" not in schema:
            raise SystemExit(f"FG INVALID: v275 trusted tipxy schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
            "CAPX_LIBERO_G17_V275_TRUSTED_TIPXY": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v275 {key} != {expected!r}")
        for key in (
            "CAPX_LIBERO_G17_GRASP_REPLAY",
            "CAPX_LIBERO_G17_GRASP_REPLAY_STRICT",
            "CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW",
            "CAPX_LIBERO_G17_V270_LIVE_ISOLATION",
            "CAPX_LIBERO_G17_V271_PT_PAIR",
        ):
            if os.environ.get(key, "").strip():
                raise SystemExit(f"FG INVALID: v275 forbids {key}")
    elif arm == "v274":
        if "v2.74" not in schema or "fair-live-inscribed-acceptance-gated" not in schema:
            raise SystemExit(f"FG INVALID: v274 mechanical contract schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
            "CAPX_LIBERO_G17_V274_MECH_CONTRACT": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: v274 {key} != {expected!r}")
        for key in (
            "CAPX_LIBERO_G17_GRASP_REPLAY",
            "CAPX_LIBERO_G17_GRASP_REPLAY_STRICT",
            "CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW",
            "CAPX_LIBERO_G17_V270_LIVE_ISOLATION",
            "CAPX_LIBERO_G17_V271_PT_PAIR",
        ):
            if os.environ.get(key, "").strip():
                raise SystemExit(f"FG INVALID: v274 forbids {key}")
    elif arm in ("v272", "v273"):
        if "v2.72" not in schema or "fair-live-no-replay" not in schema:
            raise SystemExit(f"FG INVALID: v272 clean-live schema mismatch: {schema!r}")
        required = {
            "CAPX_LIBERO_G17_V269_MECH": "1",
            "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE": "fg",
            "CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH": "1",
            "CAPX_LIBERO_G17_V272_FAIR_LIVE": "1",
        }
        for key, expected in required.items():
            if os.environ.get(key) != expected:
                raise SystemExit(f"FG INVALID: {arm} {key} != {expected!r}")
        forbidden = (
            "CAPX_LIBERO_G17_GRASP_REPLAY",
            "CAPX_LIBERO_G17_GRASP_REPLAY_STRICT",
            "CAPX_LIBERO_G17_GRASP_REPLAY_OVERFLOW",
            "CAPX_LIBERO_G17_V270_LIVE_ISOLATION",
            "CAPX_LIBERO_G17_V271_PT_PAIR",
        )
        for key in forbidden:
            if os.environ.get(key, "").strip():
                raise SystemExit(f"FG INVALID: {arm} forbids {key}")
    elif "nearmiss-osc-tipxy-bandwindow" not in schema or "v2.54" not in schema:
        raise SystemExit(
            f"FG INVALID: v256 expected v2.54 nearmiss-osc-tipxy-bandwindow schema, got {schema!r}"
        )
    if "goal-identity" in schema or "target-grasp-object" in schema:
        raise SystemExit(f"FG INVALID: v256 must not use Goal/identity schema, got {schema!r}")
    if os.environ.get("CAPX_LIBERO_G17_XY_LOCK", "0") != "1":
        raise SystemExit("FG INVALID: v256 requires CAPX_LIBERO_G17_XY_LOCK=1")
    if os.environ.get("CAPX_LIBERO_G17_RELEASE_READY", "0") != "1":
        raise SystemExit("FG INVALID: v256 requires RELEASE_READY=1")
    if os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC", "0") != "1":
        raise SystemExit("FG INVALID: v256 requires NEARMISS_OSC=1")
    if os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW", "0") != "1":
        raise SystemExit("FG INVALID: v256 requires NEARMISS_OSC_BANDWINDOW=1")
    if os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_TIPXY", "0") == "1":
        raise SystemExit("FG INVALID: v256 must not set RELEASE_READY_TIPXY (osc dispatch)")
    if os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_GOTO", "0") == "1":
        raise SystemExit("FG INVALID: v256 must not use RELEASE_READY_GOTO")
    if os.environ.get("CAPX_LIBERO_G17_GOTO_TIP_IDENTITY", "0") == "1":
        raise SystemExit("FG INVALID: v256 must not use GOTO_TIP_IDENTITY")
    osc_src = open(__import__("capx.verification.libero_g17_nearmiss_osc", fromlist=["x"]).__file__, encoding="utf-8").read()
    if "nearmiss_osc_bandwindow_enabled" not in osc_src or "no residual snap" not in osc_src:
        raise SystemExit("FG INVALID: v256 missing osc bandwindow / no-snap law")
    if "_PROOF_START_M" not in osc_src or "0.15" not in osc_src:
        raise SystemExit("FG INVALID: v256 missing 15cm proof start")
    if "run_nearmiss_osc" not in open(__import__("capx.verification.libero_g17_release_ready", fromlist=["x"]).__file__, encoding="utf-8").read():
        raise SystemExit("FG INVALID: v256 release_ready must dispatch nearmiss_osc")
    if "move_tip_xy" not in open(__import__("capx.integrations.franka.tip_xy_only", fromlist=["x"]).__file__, encoding="utf-8").read():
        raise SystemExit("FG INVALID: v256 missing tip_xy_only primitive")
    if os.environ.get("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER", "0") == "1":
        raise SystemExit("FG INVALID: v256 must not rewrite Goal")
    from capx.verification import libero_g17_xy_lock as xy_lock
    if not xy_lock.xy_lock_enabled():
        raise SystemExit("FG INVALID: xy_lock.xy_lock_enabled() is false")
    libero_src = open(__import__("capx.integrations.franka.libero", fromlist=["x"]).__file__, encoding="utf-8").read()
    if "keep_stock" not in libero_src:
        raise SystemExit("FG INVALID: v256 requires keep-stock path")
    if 'points_3d") or []' in libero_src or "points_3d') or []" in libero_src:
        raise SystemExit("FG INVALID: v256 must not use points_3d or [] (numpy truth-value)")
    if "body_world_to_robot_xyz" not in libero_src:
        raise SystemExit("FG INVALID: v256 requires robot-base projection")
    sgp = libero_src.split("def sample_grasp_pose")[1].split("def ")[0]
    if "_resolve_typed_instance(" not in sgp:
        raise SystemExit("FG INVALID: sample_grasp_pose must call _resolve_typed_instance")
    sgp_sig = libero_src.split("def sample_grasp_pose")[1].split(":")[0]
    if "which" in sgp_sig:
        raise SystemExit("FG INVALID: v256 sample_grasp_pose must not expose which=")
    if "live FG RelXY token" in libero_src.split("def sample_grasp_pose")[1].split("def ")[0]:
        raise SystemExit("FG INVALID: v256 must not docstring-coach RelXY/which")
    proj_src = open(proj.__file__, encoding="utf-8").read()
    if "allow_empty_sibling_xy=not xy_lock_enabled()" not in proj_src:
        raise SystemExit(
            "FG INVALID: v256 must suppress empty-hand XY while keeping wrong-grasp ordinal"
        )
    if "pass which=" in libero_src:
        raise SystemExit("FG INVALID: v256 must not coach which= in skill errors")

    if "sibling_probe_pixels" in libero_src or "_lock_second_stock_on_agentview" in libero_src:
        raise SystemExit("FG INVALID: v256 must not use SAM3 second-stock probes")
    if "capture_terminal_witness" not in open(__import__("capx.verification.libero_g17_takeover", fromlist=["x"]).__file__, encoding="utf-8").read():
        raise SystemExit("FG INVALID: takeover lacks capture_terminal_witness")
    trial_src = open(__import__("capx.envs.trial", fromlist=["x"]).__file__, encoding="utf-8").read()
    if "capture_terminal_witness" not in trial_src:
        raise SystemExit("FG INVALID: trial.py lacks terminal witness hook")
elif arm not in ("v256", "v269", "v270", "v271fg", "v271pt", "v272", "v273", "v274", "v275", "v276", "v277", "v278", "v279", "v280", "v281", "v282", "v283", "v284", "v285", "v286"):
    raise SystemExit(f"v256/v269 arm expected CAPX_LIBERO_G17_FG=1, arm={arm}")
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
    "panda_urdf": Path("$ROBOT_DESCRIPTIONS_CACHE/example-robot-data/robots/panda_description/urdf/panda.urdf"),
}
if "$ARM" != "v269":
    checks["qwen_key"] = Path("$KEY")
missing = [name for name, path in checks.items() if not path.exists()]
for name, path in checks.items():
    print(f"{name}: {'OK' if path.exists() else 'MISSING'} ({path})")
if missing:
    raise SystemExit(f"infra preflight failed: {missing}")
print("graspnet_config_resolved", checks["graspnet_config"].resolve())
print("infra_preflight_ok")
PYINFRA
if [[ "$ARM" != "v269" ]]; then
QWEN_KEY_FILE="$KEY" QWEN_BASE_URL="$BASE_URL" MODEL="$MODEL" "$PYTHON" - <<'PYPROBE'
from openai import OpenAI
import os
import time
from pathlib import Path
for name in (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
):
    if os.environ.get(name):
        raise SystemExit(f"proxy sanitation failed: {name}={os.environ[name]!r}")
key = next(line.strip() for line in Path(os.environ["QWEN_KEY_FILE"]).read_text().splitlines() if line.strip() and not line.startswith("#"))
client = OpenAI(
    api_key=key,
    base_url=os.environ["QWEN_BASE_URL"],
    timeout=30.0,
    max_retries=0,
)
last = None
for attempt in range(1, 31):
    try:
        r = client.chat.completions.create(
            model=os.environ["MODEL"],
            messages=[{"role":"user","content":"ping"}],
            max_tokens=8,
            temperature=0,
        )
        print("qwen_ok", r.choices[0].message.content, "attempt", attempt)
        break
    except Exception as exc:
        last = exc
        print(f"qwen_probe_retry attempt={attempt}/30 error={exc!r}", flush=True)
        if attempt < 30:
            time.sleep(10)
else:
    raise RuntimeError("qwen upstream unavailable after 30 probes") from last
PYPROBE
fi

echo "[$(date '+%F %T')] Starting Qwen proxy on 127.0.0.1:$llm_port"
if [[ "$ARM" == "v269" || "$ARM" == "v271fg" || "$ARM" == "v271pt" ]]; then
"$PYTHON" "$G17/scripts/qwen_effective_seed_replay_proxy.py" \
  --package "$V269_REPLAY_PKG" \
  --audit-log "$suite_root/qwen_proxy_audit.jsonl" \
  --host 127.0.0.1 \
  --port "$llm_port" \
  >"$suite_root/qwen_proxy.log" 2>&1 &
else
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
fi
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
if int("$START_TRIAL") > 1:
    text += "\nresume_idx: $START_TRIAL\n"
dst.write_text(text)
print(dst)
PYCFG

cat > "$suite_root/RUN_CONTEXT.txt" <<CTX
mode: $RUN_MODE
arm: $ARM
replicate: $REPLICATE
prefix_pkg: $PREFIX_PKG
pair_control: $BASELINE_EXPT/runs/libero_spatial/job-517994/$task_name
suite: $suite_name
task_id: $task_id
trials: 1..$TOTAL_TRIALS
CAPX_LIBERO_G17_FG: $CAPX_LIBERO_G17_FG
CAPX_LIBERO_G17_WORKGAP_ME: $CAPX_LIBERO_G17_WORKGAP_ME
CAPX_LIBERO_G17_XY_LOCK: $CAPX_LIBERO_G17_XY_LOCK
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
out = Path(os.environ["EXPT_DIR"])
job = Path(os.environ["SUITE_ROOT"]).resolve()
task_id = int(os.environ["TASK_ID"])
task_name = f"libero_spatial_{task_id}"
eval_dir = job / task_name / "qwen3.7-plus" / "eval"
if not eval_dir.exists():
    cands = list((job / task_name).glob("*/eval"))
    eval_dir = cands[0] if cands else eval_dir
baseline = Path("/project/peilab/qshou/cap-x-agilex/outputs/libero_m3_3x10x10_20260813/runs/libero_spatial/job-517994") / task_name / "qwen3.7-plus" / "eval"
job_id = job.name.split("-")[-1]
rep = int(os.environ.get("REPLICATE", "0"))
log_candidates = [
    out / "logs" / f"libero-g17-v286-final-fg-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v285-full-release-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v284-release-commit-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v283-tool-pi-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v282-measured-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v281-join-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v280-plan-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v279-reach-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v278-se3-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v277-seed-repair-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v272-clean-live-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v273-spatial-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v274-mech-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v275-tipxy-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v276-completion-t{task_id:02d}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v271-{arm}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v270-live-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v269-mech-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v256-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v256-{arm}-r{rep:02d}-{job_id}.out",
    out / "logs" / f"libero-g17-v256-{arm}-{job_id}.out",
    out / "logs" / f"libero-g17-v256-{job_id}.out",
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
repro = job / task_name / "repro_audit" / "libero_g17_code_prefix_replay"
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

mech_trace = job / task_name / "repro_audit" / "libero_g17_v269_mech" / "trace.jsonl"
pt_trace = job / task_name / "repro_audit" / "libero_g17_v271_pt_pair" / "trace.jsonl"
mech_rows = []
if mech_trace.exists():
    mech_rows = [json.loads(line) for line in mech_trace.read_text().splitlines() if line.strip()]
mech_mismatches = [row for row in mech_rows if row.get("event") == "symbol_mismatch"]
pt_rows = []
if pt_trace.exists():
    pt_rows = [json.loads(line) for line in pt_trace.read_text().splitlines() if line.strip()]
pt_symbol_failures = [
    row for row in pt_rows
    if row.get("event") == "symbol_unavailable"
]
replay_audit = job / "qwen_proxy_audit.jsonl"
replay_rows = []
if replay_audit.exists():
    replay_rows = [json.loads(line) for line in replay_audit.read_text().splitlines() if line.strip()]
mech_replay_ok = bool(replay_rows) and all(
    row.get("ok") and row.get("source") == "v269_effective_seed_replay"
    for row in replay_rows
)
if arm == "v269" and (not mech_rows or mech_mismatches or not mech_replay_ok):
    infra_ok = False
if arm == "v272" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v273" and mech_mismatches:
    infra_ok = False
if arm == "v274" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v275" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v276" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v277" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v278" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v279" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v280" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v281" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v282" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v283" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v284" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v285" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm == "v286" and (not mech_rows or mech_mismatches):
    infra_ok = False
if arm in ("v271fg", "v271pt") and (
    not pt_rows or pt_symbol_failures or not mech_replay_ok
):
    infra_ok = False

if not prefix_ok:
    validity = "invalid_prefix_miss"
elif fg and not fg_engaged and arm != "v273":
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
    "mode": os.environ["RUN_MODE"],
    "arm": arm,
    "replicate": replicate,
    "task_id": task_id,
    "task_name": task_name,
    "validity": validity,
    "gain_result": gain,
    "prefix_ok": prefix_ok,
    "prefix_hits": len(prefix_hit_rows),
    "fg_enabled": fg,
    "fg_engaged": bool(engagement),
    "fg_engaged_trials": sorted(engagement),
    "fg_noop_valid": arm == "v273" and not bool(engagement),
    "workgap_me": False,
    "schemas": schemas,
    "infra_ok": infra_ok,
    "v269_mech_trace_rows": len(mech_rows),
    "v269_mech_mismatches": mech_mismatches,
    "v269_effective_seed_replay_ok": mech_replay_ok if arm in ("v269", "v271fg", "v271pt") else None,
    "v271_pt_trace_rows": len(pt_rows),
    "v271_pt_symbol_failures": pt_symbol_failures,
    "xy_lock_enabled": os.environ.get("CAPX_LIBERO_G17_XY_LOCK", "0") == "1",
    "xy_lock_keep_stock": log_text.count("[xy-lock] keep-stock"),
    "xy_lock_point_correct": log_text.count("[xy-lock] point-correct"),
    "xy_lock_stock_mismatch": log_text.count("stock-mismatch"),
    "xy_lock_log_hits": log_text.count("[xy-lock]"),
    "xy_lock_n_complete": [int(x) for x in re.findall(r"n_complete=(\d+)", log_text)],
    "xy_lock_n_probe": [int(x) for x in re.findall(r"n_probe=(\d+)", log_text)],
    "molmo_chat_completions_logged": molmo_chat_hits,
    "control_job": "517994",
    "control_task": task_name,
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
arm_dir = out / "arms" / arm
if arm in ("v273", "v274", "v275", "v276", "v277", "v278", "v279", "v280", "v281", "v282", "v283", "v284", "v285", "v286"):
    arm_dir = arm_dir / f"task-{task_id:02d}"
arm_dir = arm_dir / f"rep-{replicate:02d}"
arm_dir.mkdir(parents=True, exist_ok=True)
(arm_dir / f"SUMMARY_job-{job_id}.json").write_text(json.dumps(summary, indent=2))
(arm_dir / "LATEST_SUMMARY.json").write_text(json.dumps(summary, indent=2))
if arm not in ("v273", "v274", "v275", "v276", "v277", "v278", "v279", "v280", "v281", "v282", "v283", "v284", "v285", "v286"):
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

if [[ "$ARM" == "v269" ]]; then
  "$PYTHON" - <<PYV269POST
import hashlib, json
from pathlib import Path
manifest = json.loads(Path("$V269_SOURCE_SHA").read_text())
for raw_path, expected in manifest["files"].items():
    path = Path(raw_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v269 source changed during run: {path} {actual} != {expected}")
print("v269 source fingerprints still ok after treatment")
PYV269POST
fi

# Exit: launch failure OR FG-invalid summary both fail-closed.
if [[ "$rc" != "0" ]]; then
  exit "$rc"
fi
exit "$summary_rc"
