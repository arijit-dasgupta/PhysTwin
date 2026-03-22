# PhysTwin — agent map

Read this before large edits. Deeper notes live under `docs/agents/`.

## Areas

| Area | Role |
|------|------|
| `src/qqtt/` | Core library: differentiable spring–mass (Warp), trainers, data loaders, utils. |
| `src/rerun_viz/` | Rerun visualization, replay CLI (`python -m rerun_viz.replay_recorded`), helpers. |
| `benchmarks/` | Spring–mass GPU benchmarks and reports (importable package). |
| `gaussian_splatting/` | Vendored 3D Gaussian / rendering code; local `submodules/` builds (`simple-knn`, etc.). |
| `data_process/` | Data pipelines; may clone external tools (e.g. TRELLIS). |
| `scripts/shims/*.py` | Thin compatibility shims (delegate to `scripts/entrypoints/`); **`scripts/shell/*.sh`** for bash entrypoints — see `docs/agents/repo-layout.md`. The **repo root stays minimal** (no loose `train_warp.py` / `gs_run.sh` at top level). |
| `config/` | Small data-driven inputs for batch scripts (e.g. `data_config.csv`). |

## Before editing

- Skim `docs/agents/architecture.md` for how simulation + trainer fit together.
- Skim `docs/agents/imports-and-packages.md` for `src/` layout, editable install, and `PYTHONPATH` vs `gaussian_splatting`.
- Skim `docs/agents/testing.md` for pytest markers (`gpu`) and CI scope.
- Skim `docs/agents/repo-layout.md` for root vs `scripts/entrypoints/` layout.

## When to update docs

If you change public import paths, packaging, or default CLIs, update `docs/agents/` and the root `README.md` in the same change (see `.cursor/rules/`).
