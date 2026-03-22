# Testing

## Commands

```bash
uv run pytest -q -m "not gpu"    # CPU subset (default CI)
uv run pytest -q                  # full suite; GPU tests may skip without CUDA
```

## Markers

- **`gpu`** — needs a CUDA GPU (excluded from default CI). Example: `tests/test_spring_mass_gpu_benchmark_smoke.py::test_gpu_report_sanity` when CUDA is available.

## Packaging / imports

- `tests/test_packaging.py` uses `importlib.util.find_spec` for `qqtt` where full `import qqtt` would require built `gaussian_splatting` extensions.
- Optional: `tests/test_packaging.py::test_import_qqtt_when_simple_knn_available` runs full `import qqtt` when `simple_knn` is installed.

## Lint

```bash
uv run ruff check .
uv run ruff format --check .
```
