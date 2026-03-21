"""Warmup, CUDA sync, and simple timing stats."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class TimingStats:
    mean_ms: float
    std_ms: float
    n_samples: int


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_block(fn, *, warmup: int, repeats: int) -> TimingStats:
    """Run ``fn()`` ``warmup`` + ``repeats`` times; return mean/std of *repeats* in ms."""
    for _ in range(warmup):
        fn()
    cuda_sync()
    samples: list[float] = []
    for _ in range(repeats):
        cuda_sync()
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / max(len(samples) - 1, 1)
    std = var**0.5
    return TimingStats(mean_ms=mean, std_ms=std, n_samples=len(samples))
