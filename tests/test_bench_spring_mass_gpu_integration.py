"""GPU integration test for `scripts/bench_spring_mass_gpu.py` (not run in default CI).

Run locally before claiming the benchmark works::

    RUN_GPU_BENCH=1 pytest tests/test_bench_spring_mass_gpu_integration.py -v

Requires CUDA, `./data/different_types/.../final_data.pkl`, and matching `experiments/.../train/best_*.pth`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_GPU_BENCH") != "1",
    reason="Set RUN_GPU_BENCH=1 to run the real GPU benchmark subprocess (slow; catches Warp compile issues).",
)
def test_bench_spring_mass_segment_profile_one_case() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from benchmarks.spring_mass_gpu.discover import discover_cases

    cases = discover_cases(str(REPO / "data" / "different_types"))
    if not cases:
        pytest.skip("No discoverable cases (need final_data.pkl + experiments/.../best_*.pth)")

    out_dir = REPO / "benchmarks" / "reports" / "_pytest_gpu_bench_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "bench_spring_mass_gpu.py"),
        "--output_dir",
        str(out_dir),
        "--cases",
        cases[0],
        "--segment-profile",
        "--max_parallel",
        "2",
    ]
    r = subprocess.run(
        cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    report = out_dir / "REPORT.md"
    assert report.is_file(), f"missing {report}"
