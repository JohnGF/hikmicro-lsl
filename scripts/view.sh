#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

args=()
if [ $# -ge 1 ]; then
    args+=(--device "$1")
    shift
fi
exec uv run hikmicro-lsl view "${args[@]}" "$@"
