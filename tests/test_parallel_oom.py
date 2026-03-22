"""Unit tests for CUDA OOM message matching in spring-mass GPU parallel helpers."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("CUDA out of memory", True),
        ("cuda out of memory", True),
        ("RuntimeError: Out of memory", True),
        ("Failed to allocate 12345 bytes on device", True),
        ("failed to allocate", True),
        ("Warp CUDA error: something", True),
        ("warp cuda error", True),
        ("Some other RuntimeError", False),
        ("division by zero", False),
    ],
)
def test_is_cuda_oom_runtime_error(message: str, expected: bool) -> None:
    from benchmarks.spring_mass_gpu.oom_match import _is_cuda_oom_runtime_error

    assert _is_cuda_oom_runtime_error(RuntimeError(message)) is expected
