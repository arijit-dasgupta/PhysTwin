"""Coarse spring reconstruction from a fine partition (see docs/downsampling_spring_mass.md)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def active_linear_stiffness(k_linear: float, y_min: float, y_max: float) -> float:
    """
    Match Warp eval_springs: inactive if exp(Y) <= y_min with Y=log(k); else clamp(k, y_min, y_max).
    Checkpoint stores linear stiffness k.
    """
    if not math.isfinite(k_linear) or k_linear <= float(y_min):
        return 0.0
    return float(np.clip(k_linear, float(y_min), float(y_max)))


def effective_stiffness_linear_checkpoint_row(k_linear: float, y_min: float, y_max: float) -> float:
    """Alias for :func:`active_linear_stiffness` (naming from coarsening plan / docs)."""
    return active_linear_stiffness(k_linear, y_min, y_max)


def coarse_log_stiffness_from_linear_sum(k_sum: float, y_min: float) -> float | None:
    """Return log(k_sum) for coarse spring_Y internal log storage, or None if inactive."""
    if not math.isfinite(k_sum) or k_sum <= float(y_min):
        return None
    return float(math.log(k_sum))


def build_fine_per_frame_positions(
    object_points_tx3: NDArray[np.float64],
    structure_static: NDArray[np.float64],
    num_original_points: int,
    num_surface_points: int,
    num_all_points: int,
) -> NDArray[np.float64]:
    """(T, N, 3) fine world positions: tracked from object_points, rest static from structure."""
    if structure_static.shape[0] != num_all_points:
        raise ValueError(
            f"structure row count {structure_static.shape[0]} != num_all_points {num_all_points}"
        )
    t = int(object_points_tx3.shape[0])
    if object_points_tx3.shape[1] != num_original_points:
        raise ValueError("object_points second dim must equal num_original_points")
    out = np.zeros((t, num_all_points, 3), dtype=np.float64)
    out[:, :num_original_points, :] = object_points_tx3
    static = structure_static[num_original_points:num_all_points, :]
    if static.size:
        out[:, num_original_points:num_all_points, :] = static[np.newaxis, :, :]
    return out


def rebuild_coarse_trajectories(
    fine_pos: NDArray[np.float64],
    partition: NDArray[np.int32],
    masses: NDArray[np.float64],
    K: int,
) -> NDArray[np.float64]:
    """Mass-weighted centroid trajectory per coarse cluster, shape (T, K, 3)."""
    t, n, _three = fine_pos.shape
    if partition.shape[0] != n:
        raise ValueError("partition length must match N")
    out = np.zeros((t, K, 3), dtype=np.float64)
    mass_per = np.bincount(partition, weights=masses, minlength=K).astype(np.float64)
    if np.any(mass_per <= 0):
        raise ValueError("Empty coarse cluster (no mass)")
    for ti in range(t):
        pts = fine_pos[ti]
        for k in range(K):
            m = masses[partition == k]
            p = pts[partition == k]
            out[ti, k] = (p * m[:, None]).sum(axis=0) / m.sum()
    return out


def rest_length_centroid(
    X0: NDArray[np.float64], a: int, b: int, eps: float = 1e-8
) -> float:
    d = np.linalg.norm(X0[a] - X0[b])
    return float(max(d, eps))


def _bucket_key_oo(alpha: int, beta: int) -> tuple[int, int]:
    if alpha == beta:
        raise ValueError("oo bucket requires alpha != beta")
    return (alpha, beta) if alpha < beta else (beta, alpha)


def build_coarse_springs_and_rest(
    springs: NDArray[np.int32],
    rest_lengths_fine: NDArray[np.float64],
    k_linear_fine: NDArray[np.float64],
    partition: NDArray[np.int32],
    num_object_points: int,
    K: int,
    X0_object: NDArray[np.float64],
    controller0: NDArray[np.float64],
    y_min: float,
    y_max: float,
    rest_mode: str,
    stiffness_mode: str,
    stiffness_scale: float,
    eps_rest: float = 1e-8,
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.float64], int]:
    """
    Returns coarse (springs, rest_lengths, k_linear_coarse, num_object_springs).

    Control springs: trainer uses [num_object_points + ctrl_idx, obj_idx] with object idx fine.
    """
    n_obj = int(num_object_points)
    # Buckets: oo (a,b) canon; oc (alpha, c) with alpha coarse obj, c control index
    oo_rest_terms: dict[tuple[int, int], list[float]] = defaultdict(list)
    oo_k_terms: dict[tuple[int, int], list[float]] = defaultdict(list)
    oc_rest_terms: dict[tuple[int, int], list[float]] = defaultdict(list)
    oc_k_terms: dict[tuple[int, int], list[float]] = defaultdict(list)

    def eff(k: float) -> float:
        return active_linear_stiffness(float(k), y_min, y_max)

    for e in range(springs.shape[0]):
        i1, i2 = int(springs[e, 0]), int(springs[e, 1])
        rl = float(rest_lengths_fine[e])
        kf = float(k_linear_fine[e])
        ke = eff(kf)
        if ke <= 0.0:
            continue
        if i1 < n_obj and i2 < n_obj:
            a, b = int(partition[i1]), int(partition[i2])
            if a == b:
                continue
            key = _bucket_key_oo(a, b)
            oo_rest_terms[key].append(rl)
            oo_k_terms[key].append(ke)
        elif i1 >= n_obj and i2 < n_obj:
            c = i1 - n_obj
            j = i2
            alpha = int(partition[j])
            key = (alpha, c)
            oc_rest_terms[key].append(rl)
            oc_k_terms[key].append(ke)
        elif i2 >= n_obj and i1 < n_obj:
            c = i2 - n_obj
            j = i1
            alpha = int(partition[j])
            key = (alpha, c)
            oc_rest_terms[key].append(rl)
            oc_k_terms[key].append(ke)
        else:
            # control-control: skip
            continue

    def aggregate_k(vals: list[float]) -> float:
        if stiffness_mode == "sum":
            return stiffness_scale * float(np.sum(vals))
        if stiffness_mode == "mean":
            return stiffness_scale * float(np.mean(vals)) if vals else 0.0
        if stiffness_mode == "scaled_sum":
            return stiffness_scale * float(np.sum(vals))
        raise ValueError(f"unknown stiffness_mode {stiffness_mode}")

    def aggregate_rest(vals: list[float], kvals: list[float]) -> float:
        if rest_mode == "mean_fine_rest":
            return float(np.mean(vals)) if vals else eps_rest
        if rest_mode == "stiffness_weighted_rest":
            w = np.array(kvals, dtype=np.float64)
            v = np.array(vals, dtype=np.float64)
            s = w.sum()
            if s <= 0:
                return float(np.mean(vals)) if vals else eps_rest
            return float((w * v).sum() / s)
        # centroid_distance filled later
        return float(np.mean(vals)) if vals else eps_rest

    coarse_rows: list[list[int]] = []
    coarse_rest: list[float] = []
    coarse_k: list[float] = []

    for (a, b), klist in oo_k_terms.items():
        if not klist:
            continue
        k_agg = aggregate_k(klist)
        if k_agg <= float(y_min):
            continue
        rlr = oo_rest_terms[(a, b)]
        if rest_mode == "centroid_distance":
            rl0 = rest_length_centroid(X0_object, a, b, eps_rest)
        else:
            rl0 = max(aggregate_rest(rlr, klist), eps_rest)
        # endpoint order: lower coarse index first (deterministic)
        i, j = (a, b)
        coarse_rows.append([i, j])
        coarse_rest.append(rl0)
        coarse_k.append(k_agg)

    n_oo = len(coarse_rows)

    for (alpha, c), klist in oc_k_terms.items():
        if not klist:
            continue
        k_agg = aggregate_k(klist)
        if k_agg <= float(y_min):
            continue
        rlr = oc_rest_terms[(alpha, c)]
        if rest_mode == "centroid_distance":
            d = np.linalg.norm(X0_object[alpha] - controller0[c])
            rl0 = float(max(d, eps_rest))
        else:
            rl0 = max(aggregate_rest(rlr, klist), eps_rest)
        # control first: [K + c, alpha]
        coarse_rows.append([K + c, alpha])
        coarse_rest.append(rl0)
        coarse_k.append(k_agg)

    if not coarse_rows:
        springs_out = np.zeros((0, 2), dtype=np.int32)
        rest_out = np.zeros((0,), dtype=np.float64)
        k_out = np.zeros((0,), dtype=np.float64)
        return springs_out, rest_out, k_out, 0

    springs_out = np.array(coarse_rows, dtype=np.int32)
    rest_out = np.array(coarse_rest, dtype=np.float64)
    k_out = np.array(coarse_k, dtype=np.float64)
    return springs_out, rest_out, k_out, int(n_oo)
