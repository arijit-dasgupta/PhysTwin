"""Unit tests for pure benchmark report helpers."""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_derived_outer_ms_per_substep():
    from benchmarks.spring_mass_gpu.profile_segments import derived_outer_ms_per_substep

    assert derived_outer_ms_per_substep(10.0, 5) == pytest.approx(2.0)
    assert math.isnan(derived_outer_ms_per_substep(1.0, 0))


def test_write_report_minimal(tmp_path):
    from types import SimpleNamespace

    from benchmarks.spring_mass_gpu.report import write_report

    gpu = SimpleNamespace(
        device_index=0,
        device_name="test",
        properties_total_memory_bytes=1,
        mem_get_info_total_bytes=1,
        mem_get_info_free_bytes=1,
        sanity_total_mem_match=True,
        cuda_version="x",
        nvidia_smi_line=None,
    )
    p = write_report(
        str(tmp_path),
        gpu=gpu,
        topology_rows=[],
        parallel_rows=None,
        trainers_count=0,
        oom_at_n=None,
        parallel_case="x",
        base_path="/tmp",
        warmup=1,
        repeats=1,
    )
    assert pathlib.Path(p).is_file()
    text = pathlib.Path(p).read_text(encoding="utf-8")
    assert "Outer step" in text or "outer" in text.lower()
