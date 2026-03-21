"""Topology, parameter footprint, and memory metrics for SpringMassSystemWarp."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp

from qqtt.utils import cfg


@dataclass
class ModelSizeRecord:
    case_name: str
    n_vertices: int
    num_object_points: int
    n_springs: int
    num_control_points: int
    num_substeps: int
    object_collision_flag: int
    self_collision_cfg: bool
    collision_dist: float | None
    spring_Y_numel: int
    spring_Y_bytes: int
    rest_lengths_numel: int
    masses_numel: int
    memory_peak_allocated_mb: float


def _wp_numel_bytes(wp_arr: wp.array) -> tuple[int, int]:
    t = wp.to_torch(wp_arr, requires_grad=False)
    n = t.numel()
    return n, n * t.element_size()


def collect_model_size(case_name: str, simulator) -> ModelSizeRecord:
    """``simulator`` is ``InvPhyTrainerWarp.simulator`` (SpringMassSystemWarp)."""
    spring_Y = wp.to_torch(simulator.wp_spring_Y, requires_grad=False)
    rl = wp.to_torch(simulator.wp_rest_lengths, requires_grad=False)
    masses = wp.to_torch(simulator.wp_masses, requires_grad=False)

    cd = float(simulator.collision_dist) if simulator.object_collision_flag else None

    mem_after = torch.cuda.max_memory_allocated()

    return ModelSizeRecord(
        case_name=case_name,
        n_vertices=int(simulator.n_vertices),
        num_object_points=int(simulator.num_object_points),
        n_springs=int(simulator.n_springs),
        num_control_points=int(simulator.num_control_points),
        num_substeps=int(simulator.num_substeps),
        object_collision_flag=int(simulator.object_collision_flag),
        self_collision_cfg=bool(cfg.self_collision),
        collision_dist=cd,
        spring_Y_numel=int(spring_Y.numel()),
        spring_Y_bytes=int(spring_Y.numel() * spring_Y.element_size()),
        rest_lengths_numel=int(rl.numel()),
        masses_numel=int(masses.numel()),
        memory_peak_allocated_mb=mem_after / (1024.0**2),
    )
