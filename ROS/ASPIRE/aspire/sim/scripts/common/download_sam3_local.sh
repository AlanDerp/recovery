#!/usr/bin/env bash
# Download the gated facebook/sam3 snapshot with an already-authenticated Conda env.

set -euo pipefail

REPO_ID="facebook/sam3"
ENV_SPEC=""
CACHE_DIR=""
CHECK_ONLY=false

usage() {
  cat <<'EOF'
Usage:
  bash scripts/common/download_sam3_local.sh [options]

Options:
  --env NAME_OR_PREFIX  Check only this Conda environment.
  --cache-dir PATH      Override the Hugging Face cache directory.
  --check-only          Verify login and SAM3 access without downloading weights.
  -h, --help            Show this help.

The script never prints or copies a Hugging Face token. It requires an existing
Conda environment containing huggingface_hub and a cached login authorized for
facebook/sam3. Without --cache-dir, files use the standard Hugging Face cache.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      [[ $# -ge 2 ]] || { echo "ERROR: --env requires a value" >&2; exit 2; }
      ENV_SPEC="$2"
      shift 2
      ;;
    --cache-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --cache-dir requires a value" >&2; exit 2; }
      CACHE_DIR="$2"
      shift 2
      ;;
    --check-only)
      CHECK_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available on PATH." >&2
  exit 2
fi

resolve_env_prefix() {
  local spec="$1"
  if [[ -d "$spec" && -x "$spec/bin/python" ]]; then
    printf '%s\n' "$spec"
    return 0
  fi

  conda env list | awk -v wanted="$spec" '
    $0 !~ /^#/ && NF > 0 && $1 == wanted { print $NF; found=1 }
    END { if (!found) exit 1 }
  '
}

list_env_prefixes() {
  conda env list | awk '$0 !~ /^#/ && NF > 0 { print $NF }'
}

check_access() {
  local prefix="$1"
  local python_bin="$prefix/bin/python"

  [[ -x "$python_bin" ]] || return 1
  "$python_bin" - "$REPO_ID" <<'PY'
import sys

repo_id = sys.argv[1]
try:
    from huggingface_hub import HfApi
except Exception as exc:
    print(f"missing huggingface_hub: {exc}", file=sys.stderr)
    raise SystemExit(1)

try:
    info = HfApi().model_info(repo_id, token=True)
except Exception as exc:
    print(f"access check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"authorized revision: {info.sha}")
PY
}

SELECTED_PREFIX=""

if [[ -n "$ENV_SPEC" ]]; then
  if ! SELECTED_PREFIX=$(resolve_env_prefix "$ENV_SPEC"); then
    echo "ERROR: Conda environment not found: $ENV_SPEC" >&2
    exit 2
  fi
  echo "Checking Conda environment: $SELECTED_PREFIX"
  if ! check_access "$SELECTED_PREFIX"; then
    echo "ERROR: this environment is not logged in with access to $REPO_ID." >&2
    exit 3
  fi
else
  while IFS= read -r prefix; do
    [[ -n "$prefix" ]] || continue
    echo "Checking Conda environment: $prefix"
    if check_access "$prefix"; then
      SELECTED_PREFIX="$prefix"
      break
    fi
  done < <(list_env_prefixes)

  if [[ -z "$SELECTED_PREFIX" ]]; then
    cat >&2 <<'EOF'
ERROR: no Conda environment is both:
  1. installed with huggingface_hub, and
  2. logged in with permission to access facebook/sam3.

Create or select an environment, then run:
  conda activate ENV_NAME
  python -m pip install -U huggingface_hub
  hf auth login

After login, rerun this script.
EOF
    exit 3
  fi
fi

echo "Using Conda environment: $SELECTED_PREFIX"
if $CHECK_ONLY; then
  echo "Access check passed; download skipped (--check-only)."
  exit 0
fi

PYTHON_BIN="$SELECTED_PREFIX/bin/python"
"$PYTHON_BIN" - "$REPO_ID" "$CACHE_DIR" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
cache_arg = sys.argv[2]
cache_dir = str(Path(cache_arg).expanduser().resolve()) if cache_arg else None

snapshot = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    cache_dir=cache_dir,
    token=True,
)
print(f"SAM3 snapshot ready: {snapshot}")
PY
