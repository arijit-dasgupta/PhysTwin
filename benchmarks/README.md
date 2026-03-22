# Benchmarks

## Spring–mass GPU (`spring_mass_gpu/`)

End-to-end benchmark of forward physics for real PhysTwin cases: topology stats, parallel trainers until CUDA OOM, Markdown + PNG reports.

**Run (GPU + data + checkpoints):**

```bash
python scripts/bench_spring_mass_gpu.py --output_dir benchmarks/reports/run_001
```

**Smoke / unit tests (CPU-safe):**

```bash
uv run pytest -q tests/test_spring_mass_gpu_benchmark_smoke.py tests/test_parallel_oom.py -m "not gpu"
```

**Integration (optional, GPU):** set `RUN_GPU_BENCH=1` and see `tests/test_bench_spring_mass_gpu_integration.py`.

## Reports

Historical runs may live under `benchmarks/reports/` (generated artifacts; not all are committed).
