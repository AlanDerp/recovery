#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/nvidia/xuanzhi/cap-x-agilex"
CAPX_OPENAI_ENV="${CAPX_OPENAI_ENV:-/home/nvidia/.config/capx/openai.env}"
CAPX_LLM_SERVER_URL="${CAPX_LLM_SERVER_URL:-https://usezyla.com/v1/chat/completions}"

if [[ ! -r "${CAPX_OPENAI_ENV}" ]]; then
  echo "[error] missing API environment file: ${CAPX_OPENAI_ENV}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${CAPX_OPENAI_ENV}"
set +a

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[error] OPENAI_API_KEY is not set in ${CAPX_OPENAI_ENV}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
export LLM_QUERY_MAX_RETRIES="${LLM_QUERY_MAX_RETRIES:-2}"

set +u
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/local_setup.bash
set -u

cd "${REPO_ROOT}"
"${REPO_ROOT}/scripts/start_r1lite_capx_services.sh"

cleanup_task() {
  "${REPO_ROOT}/scripts/stop_r1lite_moca_isolated.sh" || true
}
trap cleanup_task EXIT INT TERM

python3 capx/envs/launch.py \
  --config-path env_configs/r1lite/r1lite_tennis_ball_in_box_head_only_live.yaml \
  --server-url "${CAPX_LLM_SERVER_URL}" \
  --model gpt-5.5 \
  --reasoning-effort low \
  --max-tokens 8192 \
  --use-oracle-code False \
  --use-parallel-ensemble False \
  --debug False \
  --web-ui False \
  --record-video False \
  --total-trials 1 \
  --num-workers 1 \
  --output-dir ./outputs/r1lite_capx_full_place_live
