#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 <command> [args ...]" >&2
  exit 2
fi

export R1LITE_RIGHT_JOINT2_CONTROL_OFFSET_DEG="${R1LITE_RIGHT_JOINT2_CONTROL_OFFSET_DEG:-9.0}"

exec "$@"
