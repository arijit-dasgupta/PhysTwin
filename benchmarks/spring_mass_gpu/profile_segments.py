"""Helpers for eager-only CUDA segment profiling inside ``SpringMassSystemWarp.step()``."""

from __future__ import annotations

from contextlib import contextmanager

from qqtt.utils import cfg


@contextmanager
def eager_graph_for_segment_profile():
    """Temporarily set ``cfg.use_graph = False`` so ``step()`` runs eagerly (required for segments).

    Segment timings are **not** comparable to a single ``wp.capture_launch(forward_graph)`` wall time;
    see report methodology.
    """
    prev = cfg.use_graph
    cfg.use_graph = False
    try:
        yield
    finally:
        cfg.use_graph = prev


@contextmanager
def benchmark_step_segment_profile(simulator):
    """Enable per-segment CUDA events on *simulator* for the duration of the context."""
    prev = simulator._benchmark_profile_step
    simulator._benchmark_profile_step = True
    try:
        yield
    finally:
        simulator._benchmark_profile_step = prev


@contextmanager
def profile_one_outer_step(simulator):
    """Combine eager graph + segment profiling flags for one ``step()`` execution path."""
    with eager_graph_for_segment_profile(), benchmark_step_segment_profile(simulator):
        yield


def derived_outer_ms_per_substep(mean_outer_ms: float, num_substeps: int) -> float:
    """Average wall time per inner substep index, derived from one outer step mean (no extra timer)."""
    if num_substeps <= 0:
        return float("nan")
    return mean_outer_ms / float(num_substeps)
