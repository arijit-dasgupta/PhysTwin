# Imports and packages

## Layout

- **`src/qqtt/`**, **`src/rerun_viz/`**, and **`src/downsampling/`** are installable packages (`import qqtt`, `import rerun_viz`, `import downsampling`).
- **`benchmarks/`** at the repository root is also packaged (for imports like `benchmarks.spring_mass_gpu`).
- **`gaussian_splatting/`** is **not** a published PyPI package; it lives beside `src/` and is imported as `gaussian_splatting....` when the **repository root** is on `PYTHONPATH` (or when working directory is the repo root for many scripts).
- **CLI shims** (`scripts/shims/*.py`) delegate to **`scripts/entrypoints/`** (see `repo-layout.md`). **`qqtt.engine.trainer_warp`** prepends **`scripts/shims`** to `sys.path` when that directory exists (repo checkout), so `from gs_render import ...` works without manually exporting `PYTHONPATH`. For shell one-offs you can still set **`PYTHONPATH="$(pwd)/scripts/shims:$(pwd)"`**.

## Editable install

```bash
uv sync --extra dev
# or
pip install -e ".[dev]"
```

Verify:

```bash
python -c "import qqtt, rerun_viz, downsampling"
```

`qqtt` uses lazy exports in `qqtt/__init__.py` (PEP 562): `import qqtt` does not immediately load `InvPhyTrainerWarp`; accessing `qqtt.InvPhyTrainerWarp` loads the engine (and thus may import `gaussian_splatting`).

## Case setup helper

`rerun_viz.case_setup` defers `from qqtt.utils import cfg` to functions that need it, so importing path helpers (`config_yaml_for_case`, `optimal_params_path`) does not pull the full trainer stack.

## PyTorch / CUDA

Runtime PyTorch may be installed from PyTorch’s CUDA wheel index after `uv sync`; see `README.md` and `docs/UV_AND_PIXI.md`.
