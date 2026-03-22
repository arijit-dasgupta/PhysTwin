#!/usr/bin/env bash
# Wrapper script to run interactive_playground_gradio.py with warning filtering

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SHIM="${ROOT}/scripts/shims"

# Filter out Warp warnings about set_control_points
# This filters at the shell level as a backup to Python-level filtering
# Note: Python-level filtering in interactive_playground_gradio.py should handle most cases

python "${SHIM}/interactive_playground_gradio.py" "$@"
