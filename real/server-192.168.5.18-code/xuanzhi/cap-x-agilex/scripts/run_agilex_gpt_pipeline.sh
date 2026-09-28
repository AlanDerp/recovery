#!/usr/bin/env bash
# Standalone GPT-only Agilex pipeline.
# This script intentionally does not use or modify .openrouterkey / port 8110.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

CONFIG_PATH="${CONFIG_PATH:-env_configs/agilex/agilex_real_apple_tennis_ball_peach_in_box_gpt_single_turn.yaml}"
GPT_MODEL="${GPT_MODEL:-gpt-5.5}"
GPT_SERVER_URL="${GPT_SERVER_URL:-https://usezyla.com/v1/chat/completions}"
GPT_KEY_FILE="${GPT_KEY_FILE:-.gpt_api_key}"

if [[ -z "${GPT_API_KEY:-}" ]]; then
  if [[ -f "${GPT_KEY_FILE}" ]]; then
    GPT_API_KEY="$(grep -v '^[[:space:]]*#' "${GPT_KEY_FILE}" | awk 'NF {print; exit}')"
  fi
fi

if [[ -z "${GPT_API_KEY:-}" ]]; then
  echo "[error] GPT_API_KEY is not set and ${GPT_KEY_FILE} has no usable key." >&2
  echo "Set one of:" >&2
  echo "  export GPT_API_KEY='your-real-gpt-api-key'" >&2
  echo "  printf '%s\n' 'your-real-gpt-api-key' > ${GPT_KEY_FILE}" >&2
  exit 1
fi

export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:10808}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:10808}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
uv run --no-sync --active capx/envs/launch.py \
  --config-path "${CONFIG_PATH}" \
  --model "${GPT_MODEL}" \
  --server-url "${GPT_SERVER_URL}" \
  --api-key "${GPT_API_KEY}"
