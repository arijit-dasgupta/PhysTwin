# Repository layout (top level)

This file complements `AGENTS.md`: what belongs at the repository root versus subtrees, and where entrypoint implementations live.

## Root (minimal)

| Item | Role |
|------|------|
| `pyproject.toml`, `uv.lock` | Project metadata and lockfile |
| `README.md`, `AGENTS.md` | Human and agent entry docs |
| `Dockerfile`, `.dockerignore` | Container build |
| `config/` | Small CSV/text inputs used by batch scripts (e.g. `config/data_config.csv`, `config/data_spec_minimal.txt`) — paths are relative to the repo root when those scripts run |
| `src/` | Installable packages `qqtt`, `rerun_viz` |
| `tests/` | Pytest |
| `benchmarks/` | Packaged benchmark code |
| `scripts/` | `bench_spring_mass_gpu.py`, `_compat_shim.py`, **`shims/`** (thin CLI wrappers), **`entrypoints/`** (real script bodies), **`shell/`** (bash sources) |
| `docs/` | Documentation, including this file and `docs/agents/` |
| `gaussian_splatting/`, `data_process/` | Vendored / pipeline trees (not top-level PyPI packages) |
| `assets/`, `configs/` (YAML) | Static assets and experiment YAML used by trainers |
| `env_install/` | Legacy or auxiliary install scripts |

**There are no** `train_warp.py`, `gs_run.sh`, or similar **at the repo root** — use `scripts/shims/` and `scripts/shell/` instead.

## `scripts/shims/` (thin CLI files)

Each file is a few lines delegating to `scripts/entrypoints/...` via `scripts._compat_shim.run_entrypoint`. Run from the repo root:

```bash
python scripts/shims/train_warp.py --help
```

For imports like `from gs_render import ...` (used by `qqtt.engine.trainer_warp` and Gaussian scripts), put **`scripts/shims`** on `PYTHONPATH` **before** the repo root (see `README.md` and `imports-and-packages.md`).

## `scripts/entrypoints/` (by role)

| Subdir | Contents (examples) |
|--------|---------------------|
| `train/` | `train_warp.py`, `script_train.py` |
| `inference/` | `inference_warp.py`, `script_inference.py` |
| `optimize/` | `optimize_cma.py`, `script_optimize.py` |
| `data/` | `process_data.py`, `script_process_data.py` |
| `gaussian/` | `gs_train.py`, `gs_render.py`, `gs_render_dynamics.py`, `export_*.py` |
| `eval/` | `evaluate_*.py`, `visualize_*.py` |
| `playground/` | `interactive_playground*.py`, `run_playground_gradio.py` |

## Imports

- **`from gs_render import ...`** resolves when **`scripts/shims`** is on `PYTHONPATH` (the shim module `gs_render.py` lives there).
- **`from interactive_playground_gradio import ...`** (e.g. from `run_playground_gradio`) works when the shim is run as `python scripts/shims/run_playground_gradio.py` (same directory on `sys.path`).

## CI

Ruff and format checks include `scripts/` (see `.github/workflows/ci.yml`). The large **legacy** bodies under `scripts/entrypoints/` are **excluded** from Ruff; `scripts/` shims, `_compat_shim.py`, and `scripts/bench_spring_mass_gpu.py` are linted.
