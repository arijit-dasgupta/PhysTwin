# UV (primary) and Pixi (optional)

## UV

- Install dependencies from the repo root: `uv sync --extra dev` (uses `uv.lock`).
- Editable package: `qqtt`, `rerun_viz`, and `benchmarks` are declared in `pyproject.toml`; `PYTHONPATH` should include the repository root when running scripts that import `gaussian_splatting` (vendored, not on PyPI).

### PyTorch with CUDA

PyTorch CUDA wheels are published on a separate index from PyPI. After `uv sync`, install the stack that matches your driver/CUDA, for example:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for the correct index URL.

## Pixi

If you need a conda-forge base (e.g. system OpenGL / Qt) while still using the same Python project, install [Pixi](https://pixi.sh/) and run `pixi install` from the repo root. The `pixi.toml` here provides a minimal Python 3.10 shell; run `uv` inside that shell for dependency resolution, or install `uv` into the Pixi environment.

This is **optional**; the default developer path is UV only.
