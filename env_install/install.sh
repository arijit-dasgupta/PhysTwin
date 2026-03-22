#!/usr/bin/env bash
# UV-based environment install for PhysTwin (Linux).
# Replaces legacy conda flows in env_install.sh (kept for reference only).
#
# Prerequisites: install UV — https://docs.astral.sh/uv/
#
# CUDA PyTorch: after `uv sync`, install torch from the PyTorch CUDA wheel index, e.g.:
#   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv python install 3.10
uv sync --python 3.10 --extra dev "$@"

echo "Done. Activate: source .venv/bin/activate  (or use: uv run <command>)"
echo "Verify: uv run python -c \"import qqtt, rerun_viz\""
