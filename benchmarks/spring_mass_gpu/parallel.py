"""Load multiple trainers until OOM; time stepping N instances in one process."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import torch

from benchmarks.spring_mass_gpu.load_case import load_trainer_same_config
from benchmarks.spring_mass_gpu.oom_match import _is_cuda_oom_runtime_error
from benchmarks.spring_mass_gpu.physics import step_and_advance_state, step_forward_once
from benchmarks.spring_mass_gpu.profile_segments import profile_one_outer_step
from benchmarks.spring_mass_gpu.timing import TimingStats, cuda_sync, time_block
from rerun_viz.replay_core import load_config_and_camera, set_all_seeds


@dataclass
class ParallelScalingRow:
    n_instances: int
    step_all_ms: float
    step_all_std_ms: float
    per_instance_ms: float
    segment_ms: dict[str, float] | None = None


def safe_frame_idx(simulator) -> int:
    cp = simulator.controller_points
    assert cp is not None
    t = int(cp.shape[0])
    return min(1, max(0, t - 1))


def trainers_until_oom(
    base_path: str,
    case_name: str,
    *,
    max_instances: int = 256,
    seed_base: int = 42,
    on_progress: Callable[[list], None] | None = None,
) -> tuple[list, int]:
    """
    Load independent trainers (same case) until first CUDA OOM on load.

    Returns (trainers, oom_at_n) where ``oom_at_n`` is the 1-based index of the load
    that failed (len(trainers) + 1), or ``len(trainers) + 1`` if stopped at ``max_instances``.
    """
    set_all_seeds(seed_base)
    load_config_and_camera(base_path, case_name)
    trainers: list = []
    oom_at = max_instances + 1
    while len(trainers) < max_instances:
        try:
            torch.cuda.empty_cache()
            t = load_trainer_same_config(base_path, case_name, seed=seed_base + len(trainers))
            trainers.append(t)
            if on_progress is not None:
                on_progress(list(trainers))
        except RuntimeError as e:
            if _is_cuda_oom_runtime_error(e):
                oom_at = len(trainers) + 1
                break
            raise
    else:
        oom_at = len(trainers) + 1
    return trainers, oom_at


def benchmark_step_k_parallel(
    trainers: list,
    k: int,
    *,
    warmup: int,
    repeats: int,
) -> TimingStats:
    """Time one sequential pass: step each of the first ``k`` trainers once."""
    subset = trainers[:k]
    frame_idxs = [safe_frame_idx(tr.simulator) for tr in subset]

    def one_round() -> None:
        for tr, fi in zip(subset, frame_idxs, strict=True):
            sim = tr.simulator
            sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities, pure_inference=True)
            step_and_advance_state(sim, fi)

    return time_block(one_round, warmup=warmup, repeats=repeats)


def _segment_sums_for_n(trainers: list, n: int) -> dict[str, float]:
    """Sum of per-simulator segment times for one sequential round (``n`` simulators × one outer step each)."""
    subset = trainers[:n]
    frame_idxs = [safe_frame_idx(tr.simulator) for tr in subset]
    total: dict[str, float] = defaultdict(float)
    for tr, fi in zip(subset, frame_idxs, strict=True):
        sim = tr.simulator
        sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities, pure_inference=True)
        with profile_one_outer_step(sim):
            step_forward_once(sim, fi)
        for k, v in sim._last_step_segment_ms.items():
            total[k] += float(v)
    return dict(total)


def rows_for_all_n(
    trainers: list,
    *,
    warmup: int,
    repeats: int,
    segment_profile: bool = False,
) -> tuple[list[ParallelScalingRow], bool]:
    """One timing row per n from 1..len(trainers).

    On CUDA OOM during timing, returns rows completed so far and ``timing_complete`` False.
    """
    out: list[ParallelScalingRow] = []
    for n in range(1, len(trainers) + 1):
        try:
            cuda_sync()
            torch.cuda.reset_peak_memory_stats()
            st = benchmark_step_k_parallel(trainers, n, warmup=warmup, repeats=repeats)
            per = st.mean_ms / n
            seg = _segment_sums_for_n(trainers, n) if segment_profile else None
            out.append(
                ParallelScalingRow(
                    n_instances=n,
                    step_all_ms=st.mean_ms,
                    step_all_std_ms=st.std_ms,
                    per_instance_ms=per,
                    segment_ms=seg,
                )
            )
        except RuntimeError as e:
            if _is_cuda_oom_runtime_error(e):
                break
            raise
    timing_complete = len(out) == len(trainers)
    return out, timing_complete
