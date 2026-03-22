#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SHIM="${ROOT}/scripts/shims"
python "${SHIM}/evaluate_chamfer.py"
python "${SHIM}/evaluate_track.py"
python gaussian_splatting/evaluate_render.py