"""Rerun paths for multi-resolution compare: shared controls + per-variant meshes."""

from __future__ import annotations

import re

import numpy as np
import rerun as rr

from rerun_viz.spring_mass_logging import build_spring_strips

COMPARE_ROOT = "compare"
COMPARE_SHARED = f"{COMPARE_ROOT}/shared"


def variant_path_segment(label: str) -> str:
    if label == "full_res":
        return "full_res"
    safe = re.sub(r"[^\w\-.]", "_", label.strip())
    return safe or "variant"


def mesh_entity_path(label: str) -> str:
    return f"{COMPARE_ROOT}/objects/{variant_path_segment(label)}/mesh"


def log_shared_controls_frame(
    *,
    object_positions: np.ndarray,
    controller_positions: np.ndarray | None,
    springs: np.ndarray,
    frame_idx: int,
    timeline: str = "frame",
) -> None:
    """Log controller points and control springs only (no object–object springs)."""
    rr.set_time(timeline, sequence=int(frame_idx))
    num_obj = int(object_positions.shape[0])

    if controller_positions is not None and controller_positions.size > 0:
        rr.log(
            f"{COMPARE_SHARED}/controls/points",
            rr.Points3D(
                positions=controller_positions,
                radii=0.004,
                colors=[0, 255, 120, 255],
            ),
        )

    if springs.size == 0:
        return

    strips = build_spring_strips(
        object_positions=object_positions,
        controller_positions=controller_positions,
        springs=springs,
        num_object_points=num_obj,
    )

    ctrl_obj_idx: list[int] = []
    ctrl_ctrl_idx: list[int] = []
    for i in range(springs.shape[0]):
        a, b = int(springs[i, 0]), int(springs[i, 1])
        a_obj = a < num_obj
        b_obj = b < num_obj
        if a_obj and b_obj:
            continue
        if (a_obj and not b_obj) or (not a_obj and b_obj):
            ctrl_obj_idx.append(i)
        else:
            ctrl_ctrl_idx.append(i)

    if ctrl_obj_idx:
        s = [strips[i] for i in ctrl_obj_idx]
        col = np.array([(255, 220, 0, 255)] * len(s), dtype=np.uint8)
        rr.log(f"{COMPARE_SHARED}/controls/springs_control_object", rr.LineStrips3D(strips=s, colors=col))
    if ctrl_ctrl_idx:
        s = [strips[i] for i in ctrl_ctrl_idx]
        col = np.array([(255, 140, 0, 255)] * len(s), dtype=np.uint8)
        rr.log(f"{COMPARE_SHARED}/controls/springs_control_control", rr.LineStrips3D(strips=s, colors=col))


def log_mesh_variant_frame(
    *,
    label: str,
    vertex_positions: np.ndarray,
    triangle_indices: np.ndarray,
    rgb: tuple[int, int, int],
    opacity: float,
    frame_idx: int,
    timeline: str = "frame",
) -> None:
    """Log one deformed mesh for a resolution variant.

    Rerun 0.25+ documents ``Mesh3D`` transparency via **albedo_factor** (see
    https://rerun.io/blog/release-0.25). Per-vertex ``vertex_colors`` use opaque alpha (1.0);
    translucency is ``albedo_factor=(255, 255, 255, a_byte)`` with ``a_byte = round(opacity * 255)``.
    """
    rr.set_time(timeline, sequence=int(frame_idx))
    n = int(vertex_positions.shape[0])
    a = float(np.clip(opacity, 0.0, 1.0))
    a_byte = int(round(a * 255.0))
    r, g, b = rgb
    vc = np.empty((n, 4), dtype=np.float32)
    vc[:, 0] = r / 255.0
    vc[:, 1] = g / 255.0
    vc[:, 2] = b / 255.0
    vc[:, 3] = 1.0
    path = mesh_entity_path(label)
    rr.log(
        path,
        rr.Mesh3D(
            vertex_positions=vertex_positions.astype(np.float32),
            triangle_indices=triangle_indices.astype(np.uint32),
            vertex_colors=vc,
            albedo_factor=(255, 255, 255, a_byte),
        ),
    )


def log_compare_legend_once(text: str) -> None:
    rr.log(f"{COMPARE_SHARED}/legend_compare", rr.TextLog(text))
