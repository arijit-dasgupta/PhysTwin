"""Track (B): per-real-case forward step timing + model size."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from benchmarks.spring_mass_gpu.load_case import load_trainer_for_case
from benchmarks.spring_mass_gpu.parallel import safe_frame_idx
from benchmarks.spring_mass_gpu.physics import step_and_advance_state, step_forward_once
from benchmarks.spring_mass_gpu.profile_segments import profile_one_outer_step
from benchmarks.spring_mass_gpu.sizing import ModelSizeRecord, collect_model_size
from benchmarks.spring_mass_gpu.timing import TimingStats, time_block


@dataclass
class TopologyRow:
    case_name: str
    timing: TimingStats
    frame_idx_used: int
    size: ModelSizeRecord
    peak_mem_mb: float
    segment_ms: dict[str, float] | None = None


def benchmark_one_case(
    base_path: str,
    case_name: str,
    *,
    warmup: int,
    repeats: int,
    segment_profile: bool = False,
) -> TopologyRow:
    torch.cuda.reset_peak_memory_stats()
    trainer = load_trainer_for_case(base_path, case_name)
    sim = trainer.simulator
    fi = safe_frame_idx(sim)

    def one_round() -> None:
        sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities, pure_inference=True)
        step_and_advance_state(sim, fi)

    st = time_block(one_round, warmup=warmup, repeats=repeats)
    size = collect_model_size(case_name, sim)
    peak = torch.cuda.max_memory_allocated() / (1024.0**2)
    seg: dict[str, float] | None = None
    if segment_profile:
        sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities, pure_inference=True)
        with profile_one_outer_step(sim):
            step_forward_once(sim, fi)
        seg = dict(sim._last_step_segment_ms)
    return TopologyRow(
        case_name=case_name,
        timing=st,
        frame_idx_used=fi,
        size=size,
        peak_mem_mb=peak,
        segment_ms=seg,
    )
