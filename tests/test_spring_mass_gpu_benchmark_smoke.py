"""Smoke tests for spring-mass GPU benchmark helpers (no full benchmark run in CI)."""

from __future__ import annotations

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_discover_cases_empty_for_missing_dir():
    from benchmarks.spring_mass_gpu.discover import discover_cases

    assert discover_cases("/nonexistent_path_abc_123") == []


def test_timing_stats_shape():
    from benchmarks.spring_mass_gpu.timing import time_block

    n = [0]

    def fn() -> None:
        n[0] += 1

    st = time_block(fn, warmup=2, repeats=5)
    assert st.n_samples == 5
    assert st.mean_ms >= 0


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA not available",
)
def test_gpu_report_sanity():
    from benchmarks.spring_mass_gpu.gpu_info import collect_gpu_report

    r = collect_gpu_report()
    assert r.device_name
    assert r.properties_total_memory_bytes > 0
    assert r.mem_get_info_total_bytes > 0


def test_physics_step_import():
    from benchmarks.spring_mass_gpu.physics import step_forward_once

    assert callable(step_forward_once)
