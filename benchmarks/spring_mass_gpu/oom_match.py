"""CUDA / Warp OOM message heuristics (no trainer or heavy deps)."""

from __future__ import annotations

_OOM_SUBSTRINGS = (
    "out of memory",
    "failed to allocate",
    "cuda out of memory",
    "warp cuda error",
)


def _is_cuda_oom_runtime_error(exc: BaseException) -> bool:
    """True if *exc* looks like a CUDA / Warp OOM (case-insensitive message match)."""
    msg = str(exc).lower()
    return any(s in msg for s in _OOM_SUBSTRINGS)
