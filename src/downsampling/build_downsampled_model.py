"""Build downsampled spring–mass artifacts for a real-data case (see docs/downsampling_spring_mass.md)."""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp

from downsampling.checkpoint import build_downsampled_checkpoint
from downsampling.io import (
    artifact_paths,
    ensure_dir,
    save_coarse_npz,
    save_final_data_pickle,
    save_meta_json,
    save_partition_map,
)
from downsampling.partition_graph import partition_graph_coarsen
from downsampling.partition_kmeans import CoarseningFeasibilityError, partition_kmeans
from downsampling.reconstruct import (
    build_coarse_springs_and_rest,
    build_fine_per_frame_positions,
    rebuild_coarse_trajectories,
)
from downsampling.types import PartitionMethod, RestLengthMode, StiffnessMode
from downsampling.validate import (
    DownsampledBundleError,
    allocate_unique_downsample_tag,
    downsampled_bundle_dir,
    downsampled_bundle_exists,
    validate_downsampled_bundle,
)
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg
from rerun_viz.case_setup import load_case_yaml_and_optimal


def _format_r_for_tag(r: float) -> str:
    if not math.isfinite(r):
        raise ValueError("r must be finite")
    ri = round(r)
    if abs(r - ri) < 1e-9:
        return str(int(ri))
    s = f"{r:.12f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "m")


def default_auto_tag_base(effective_method: PartitionMethod, r: float) -> str:
    """Default folder tag: ``{method}_r{formatted_r}`` (e.g. ``kmeans_r2``, ``graph_r2p5``)."""
    base = f"{effective_method.value}_r{_format_r_for_tag(r)}"
    return re.sub(r"[^\w\-.]", "_", base)


def target_coarse_cluster_count(n_fine: int, r: float) -> int:
    """
    Number of coarse object clusters ``K = max(1, ceil(N/r))``.

    Raises ``CoarseningFeasibilityError`` if ``K > n_fine`` (invalid coarsening / r too small).
    """
    if r <= 0 or not math.isfinite(r):
        raise ValueError("r must be positive and finite")
    if n_fine < 1:
        raise ValueError("n_fine must be >= 1")
    K = max(1, int(math.ceil(n_fine / r)))
    if K > n_fine:
        raise CoarseningFeasibilityError(
            f"target K={K} exceeds fine vertices N={n_fine} (r={r} implies more coarse than fine). "
            "Increase r so ceil(N/r) <= N (typically r >= 1)."
        )
    return K


def _coarse_colors(coarse_traj_tk3: np.ndarray) -> np.ndarray:
    t, k, _ = coarse_traj_tk3.shape
    y_min, y_max = np.min(coarse_traj_tk3[0, :, 1]), np.max(coarse_traj_tk3[0, :, 1])
    span = float(y_max - y_min)
    if span < 1e-12:
        y_norm = np.zeros(k, dtype=np.float64)
    else:
        y_norm = (coarse_traj_tk3[0, :, 1] - y_min) / span
    rainbow = plt.cm.rainbow(y_norm)[:, :3].astype(np.float32)
    return np.broadcast_to(rainbow[np.newaxis, :, :], (t, k, 3)).copy()


def _aggregate_bool_by_partition(
    fine_txn: np.ndarray, partition: np.ndarray, K: int, *, reduce_all: bool
) -> np.ndarray:
    """fine_txn: (T, N) bool; partition: (N,) -> (T, K)."""
    t, n = fine_txn.shape
    out = np.zeros((t, K), dtype=bool)
    for k in range(K):
        ix = np.where(partition == k)[0]
        if ix.size == 0:
            out[:, k] = True
        elif reduce_all:
            out[:, k] = np.all(fine_txn[:, ix], axis=1)
        else:
            out[:, k] = np.any(fine_txn[:, ix], axis=1)
    return out


def run_build(
    *,
    base_path: str,
    case_name: str,
    tag: str | None,
    r: float,
    method: PartitionMethod,
    rest_length_mode: RestLengthMode,
    stiffness_mode: StiffnessMode,
    stiffness_scale: float,
    checkpoint_path: str | None,
    kmeans_random_state: int,
    kmeans_n_init: int,
    skip_validate: bool,
    device: str,
) -> dict[str, Any]:
    if r <= 0 or not math.isfinite(r):
        raise ValueError("r must be positive and finite")
    load_case_yaml_and_optimal(case_name)
    cfg.device = device
    wp.set_device(device)

    exp_dir = f"experiments/{case_name}"
    final_pkl = f"{base_path}/{case_name}/final_data.pkl"
    if not os.path.isfile(final_pkl):
        raise FileNotFoundError(final_pkl)

    if checkpoint_path is None:
        cands = sorted(glob.glob(f"{exp_dir}/train/best_*.pth"))
        if not cands:
            raise FileNotFoundError(f"no best_*.pth under {exp_dir}/train")
        checkpoint_path = cands[0]
    ckpt = torch.load(checkpoint_path, map_location=device)

    effective_method = method
    if cfg.self_collision and method == PartitionMethod.kmeans:
        print(
            "[downsample] self_collision=True: using graph coarsening (k-means is unsupported for "
            "unique-per-vertex collision IDs); pass --method graph explicitly to silence this.",
            file=sys.stderr,
        )
        effective_method = PartitionMethod.graph

    if tag is None:
        base = default_auto_tag_base(effective_method, r)
        tag = allocate_unique_downsample_tag(base_path, case_name, base)
        print(f"[downsample] auto tag: {tag!r} (base {base!r})", file=sys.stderr)
    else:
        tag = tag.strip()
        if not tag:
            raise ValueError("--tag must be non-empty when provided")
        if downsampled_bundle_exists(base_path, case_name, tag):
            d = downsampled_bundle_dir(base_path, case_name, tag)
            raise FileExistsError(
                f"downsampled bundle already exists for this case and tag: {d} "
                "(remove it, pick another --tag, or omit --tag for an auto tag)"
            )

    trainer = InvPhyTrainerWarp(
        data_path=final_pkl,
        base_dir=exp_dir,
        pure_inference_mode=True,
        device=device,
    )

    N = int(trainer.num_all_points)
    K = target_coarse_cluster_count(N, r)
    springs_f = wp.to_torch(trainer.simulator.wp_springs).cpu().numpy().astype(np.int32)
    rest_f = wp.to_torch(trainer.simulator.wp_rest_lengths).cpu().numpy().astype(np.float64)
    k_linear = ckpt["spring_Y"]
    k_np = k_linear.detach().cpu().numpy().astype(np.float64)
    if k_np.shape[0] != springs_f.shape[0]:
        raise RuntimeError("checkpoint spring_Y length does not match simulator springs")

    X0 = trainer.structure_points.detach().cpu().numpy().astype(np.float64)
    if X0.ndim != 2 or X0.shape[1] != 3:
        raise RuntimeError(
            f"structure_points must be (N, 3) rest layout, got shape {X0.shape} "
            "(do not index with [0] — that takes a single vertex)."
        )
    if X0.shape[0] != N:
        raise RuntimeError(
            f"structure_points rows {X0.shape[0]} != trainer.num_all_points {N}"
        )
    masses = np.ones(N, dtype=np.float64)

    protected = np.zeros(N, dtype=bool)
    for e in range(springs_f.shape[0]):
        i1, i2 = int(springs_f[e, 0]), int(springs_f[e, 1])
        if i1 >= N > i2:
            protected[i2] = True
        elif i2 >= N > i1:
            protected[i1] = True

    fine_masks = None
    if effective_method == PartitionMethod.kmeans:
        pi = partition_kmeans(
            X0,
            masses,
            K,
            protected=protected,
            fine_masks=fine_masks,
            random_state=kmeans_random_state,
            n_init=kmeans_n_init,
        )
    else:
        pi = partition_graph_coarsen(
            X0,
            springs_f,
            N,
            K,
            protected=protected,
            fine_masks=fine_masks,
        )
    if pi.shape[0] != N:
        raise RuntimeError(f"partition length {pi.shape[0]} != N={N}")

    object_np = trainer.object_points.detach().cpu().numpy().astype(np.float64)
    structure_np = trainer.structure_points.detach().cpu().numpy().astype(np.float64)
    n_orig = int(trainer.num_original_points)
    n_surf = int(trainer.num_surface_points)
    n_all = int(trainer.num_all_points)
    fine_pos = build_fine_per_frame_positions(object_np, structure_np, n_orig, n_surf, n_all)
    coarse_traj = rebuild_coarse_trajectories(fine_pos, pi, masses, K)
    X0_coarse = coarse_traj[0].astype(np.float64)
    ctrl0 = trainer.controller_points[0].detach().cpu().numpy().astype(np.float64)

    y_min = float(cfg.spring_Y_min)
    y_max = float(cfg.spring_Y_max)
    c_springs, c_rest, c_k, n_oo = build_coarse_springs_and_rest(
        springs_f,
        rest_f,
        k_np,
        pi,
        N,
        K,
        X0_coarse,
        ctrl0,
        y_min,
        y_max,
        rest_length_mode.value,
        stiffness_mode.value,
        float(stiffness_scale),
    )

    coarse_masses = np.bincount(pi, weights=masses, minlength=K).astype(np.float64)
    C = int(ctrl0.shape[0])
    init_vertices = np.vstack([X0_coarse, ctrl0]).astype(np.float64)
    init_masses = np.concatenate([coarse_masses, np.ones(C, dtype=np.float64)])

    paths = artifact_paths(base_path, case_name, tag)
    ensure_dir(paths["root"])

    save_coarse_npz(
        paths["coarse_npz"],
        init_vertices=init_vertices,
        init_springs=c_springs,
        init_rest_lengths=c_rest,
        init_masses=init_masses,
        num_object_springs=n_oo,
        num_all_points=K,
    )
    save_partition_map(paths["partition"], pi)

    fine_vis = np.ones((fine_pos.shape[0], N), dtype=bool)
    fine_vis[:, :n_orig] = trainer.object_visibilities.detach().cpu().numpy()
    fine_mv = np.ones((fine_pos.shape[0], N), dtype=bool)
    fine_mv[:, :n_orig] = trainer.object_motions_valid.detach().cpu().numpy()
    coarse_vis = _aggregate_bool_by_partition(fine_vis, pi, K, reduce_all=False)
    coarse_mv = _aggregate_bool_by_partition(fine_mv, pi, K, reduce_all=True)

    ctrl_all = trainer.controller_points.detach().cpu().numpy().astype(np.float32)
    final_data = {
        "object_points": coarse_traj.astype(np.float32),
        "object_colors": _coarse_colors(coarse_traj.astype(np.float32)),
        "object_visibilities": coarse_vis,
        "object_motions_valid": coarse_mv,
        "controller_points": ctrl_all,
        "surface_points": np.zeros((0, 3), dtype=np.float32),
        "interior_points": np.zeros((0, 3), dtype=np.float32),
    }
    save_final_data_pickle(paths["final_data"], final_data)

    coarse_ckpt = build_downsampled_checkpoint(
        c_k.astype(np.float64),
        n_oo,
        ckpt,
        epoch=int(ckpt.get("epoch", 0)),
    )
    torch.save(coarse_ckpt, paths["checkpoint"])

    meta = {
        "version": 1,
        "case_name": case_name,
        "tag": tag,
        "method": effective_method.value,
        "method_requested": method.value,
        "r": r,
        "K": K,
        "N_fine": N,
        "rest_length_mode": rest_length_mode.value,
        "stiffness_mode": stiffness_mode.value,
        "stiffness_scale": float(stiffness_scale),
        "kmeans_random_state": kmeans_random_state,
        "kmeans_n_init": kmeans_n_init,
        "fine_checkpoint": os.path.abspath(checkpoint_path),
        "spring_reduction": f"{springs_f.shape[0]}->{c_springs.shape[0]}",
    }
    if effective_method != method:
        meta["method_override_note"] = (
            "requested_kmeans_self_collision_used_graph_coarsening"
        )
    if effective_method == PartitionMethod.graph:
        meta["graph_merge_policy"] = {
            "edge_cost": "squared_euclidean_distance_in_rest_pose_X0",
            "eligible_edges": "object_object_springs_only_i2_lt_num_all_points",
            "edge_sort_key": "ascending (cost, i_lo, i_hi) with i_lo=min(i1,i2), i_hi=max(i1,i2)",
            "merge_order": "scan_sorted_edges_first_admissible_union_forest_per_step",
            "tie_break_note": "deterministic lexicographic on (cost, lower_index, higher_index)",
        }
    save_meta_json(paths["meta"], meta)

    if not skip_validate:
        validate_downsampled_bundle(
            base_path,
            case_name,
            tag,
            partition_N_fine=N,
        )

    del trainer
    return {"paths": paths, "meta": meta}


def main() -> None:
    p = argparse.ArgumentParser(description="Build downsampled spring–mass bundle for a case.")
    p.add_argument("--base_path", type=str, required=True)
    p.add_argument("--case_name", type=str, required=True)
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help=(
            "Folder under downsampled/<tag>/. Default: auto from effective --method and --r "
            "(e.g. kmeans_r2); if that tag exists, uses suffix _2, _3, ... "
            "If you pass --tag, the run fails when that bundle already exists."
        ),
    )
    p.add_argument("--r", type=float, required=True, help="Approximate downsampling factor (K=ceil(N/r)).")
    p.add_argument(
        "--method",
        type=str,
        choices=("kmeans", "graph"),
        default="kmeans",
    )
    p.add_argument(
        "--rest_length",
        type=str,
        default=RestLengthMode.centroid_distance.value,
        choices=tuple(m.value for m in RestLengthMode),
    )
    p.add_argument(
        "--stiffness",
        type=str,
        default=StiffnessMode.sum.value,
        choices=tuple(m.value for m in StiffnessMode),
    )
    p.add_argument("--stiffness-scale", type=float, default=1.0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--kmeans-random-state", type=int, default=0)
    p.add_argument("--kmeans-n-init", type=int, default=10)
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()
    try:
        out = run_build(
            base_path=args.base_path,
            case_name=args.case_name,
            tag=args.tag,
            r=args.r,
            method=PartitionMethod(args.method),
            rest_length_mode=RestLengthMode(args.rest_length),
            stiffness_mode=StiffnessMode(args.stiffness),
            stiffness_scale=args.stiffness_scale,
            checkpoint_path=args.checkpoint,
            kmeans_random_state=args.kmeans_random_state,
            kmeans_n_init=args.kmeans_n_init,
            skip_validate=args.skip_validate,
            device=args.device,
        )
    except CoarseningFeasibilityError as e:
        print(f"[downsample] coarsening failed: {e}", file=sys.stderr)
        sys.exit(2)
    except FileExistsError as e:
        print(f"[downsample] tag collision: {e}", file=sys.stderr)
        sys.exit(4)
    except DownsampledBundleError as e:
        print(f"[downsample] validation failed: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"[downsample] error: {e}", file=sys.stderr)
        sys.exit(1)

    meta = out["meta"]
    paths = out["paths"]
    print("[downsample] wrote bundle:")
    for k in ("final_data", "coarse_npz", "partition", "meta", "checkpoint"):
        print(f"  {k}: {paths[k]}")
    print(
        f"[downsample] tag={meta['tag']!r} N_fine={meta['N_fine']} -> K={meta['K']} "
        f"springs {meta['spring_reduction']}"
    )


if __name__ == "__main__":
    main()
