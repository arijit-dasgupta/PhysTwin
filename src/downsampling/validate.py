"""Shared validation for downsampled artifact bundles (builder + replay)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

CANONICAL_FINAL_DATA_NAME = "final_data_downsampled.pkl"
CANONICAL_NPZ_NAME = "coarse_model.npz"
CANONICAL_PARTITION_NAME = "partition_map.npy"
CANONICAL_META_NAME = "downsample_meta.json"
CANONICAL_CHECKPOINT_NAME = "best_downsampled.pth"


class DownsampledBundleError(RuntimeError):
    """Artifact missing, wrong shape, or inconsistent with coarse metadata."""


def downsampled_bundle_dir(base_path: str, case_name: str, tag: str) -> str:
    return str(Path(base_path) / case_name / "downsampled" / tag)


def downsampled_bundle_exists(base_path: str, case_name: str, tag: str) -> bool:
    """True if a bundle was written under ``tag`` (canonical ``downsample_meta.json`` present)."""
    meta = Path(downsampled_bundle_dir(base_path, case_name, tag)) / CANONICAL_META_NAME
    return meta.is_file()


def allocate_unique_downsample_tag(base_path: str, case_name: str, base_tag: str) -> str:
    """
    Return ``base_tag`` if no bundle exists there; else ``base_tag_2``, ``base_tag_3``, ...
    until a free tag is found.
    """
    if not downsampled_bundle_exists(base_path, case_name, base_tag):
        return base_tag
    for n in range(2, 10_001):
        cand = f"{base_tag}_{n}"
        if not downsampled_bundle_exists(base_path, case_name, cand):
            return cand
    raise RuntimeError(f"could not allocate a free downsample tag for base {base_tag!r}")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise DownsampledBundleError(msg)


def assert_realdata_pickle_layout(data: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Return (T, N_orig, N_surf, N_all, N_ctrl) from a ``final_data``-style dict."""
    for k in (
        "object_points",
        "controller_points",
        "surface_points",
        "interior_points",
        "object_visibilities",
        "object_motions_valid",
        "object_colors",
    ):
        _require(k in data, f"final_data missing key {k!r}")
    op = data["object_points"]
    _require(isinstance(op, np.ndarray) and op.ndim == 3 and op.shape[2] == 3, "object_points")
    ctrl = data["controller_points"]
    _require(
        isinstance(ctrl, np.ndarray) and ctrl.ndim == 3 and ctrl.shape[2] == 3,
        "controller_points",
    )
    T, n_orig, _ = op.shape
    _require(ctrl.shape[0] == T, "controller_points T mismatch object_points")
    surf = data["surface_points"]
    interior = data["interior_points"]
    _require(isinstance(surf, np.ndarray) and surf.ndim == 2 and surf.shape[1] == 3, "surface_points")
    _require(
        isinstance(interior, np.ndarray) and interior.ndim == 2 and interior.shape[1] == 3,
        "interior_points",
    )
    n_surf = n_orig + int(surf.shape[0])
    n_all = n_surf + int(interior.shape[0])
    vis = data["object_visibilities"]
    mv = data["object_motions_valid"]
    _require(vis.shape == (T, n_orig), "object_visibilities shape")
    _require(mv.shape == (T, n_orig), "object_motions_valid shape")
    oc = data["object_colors"]
    _require(oc.ndim in (2, 3), "object_colors rank")
    if oc.ndim == 2:
        _require(oc.shape == (n_orig, 3), "object_colors (N,3)")
    else:
        _require(oc.shape[:2] == (T, n_orig), "object_colors (T,N,3)")
    n_ctrl = int(ctrl.shape[1])
    _require(n_orig <= n_surf <= n_all, "need num_original_points <= num_surface_points <= num_all_points")
    return T, n_orig, n_surf, n_all, n_ctrl


# Alias matching plan / external docs
assert_realdata_layout = assert_realdata_pickle_layout

# Maximum endpoint index for Warp vec2i / int32 spring rows (Auditable upper bound.)
_SPRING_INDEX_INT32_MAX = int(np.iinfo(np.int32).max)


def assert_spring_indices_int32_safe(K: int, C: int, springs: NDArray[np.integer]) -> None:
    """
    Fail if ``K+C`` or any spring endpoint cannot be represented as a non‑negative int32 index
    (plan: spring rows are vec2i).
    """
    nvert = int(K) + int(C)
    _require(nvert >= 1, "need at least one vertex")
    _require(
        nvert <= _SPRING_INDEX_INT32_MAX,
        f"K+C={nvert} exceeds int32 max ({_SPRING_INDEX_INT32_MAX}); spring graph not representable",
    )
    smax = int(np.max(springs)) if springs.size else -1
    _require(
        smax < nvert,
        f"spring index max={smax} must be < K+C={nvert}",
    )
    _require(
        smax <= _SPRING_INDEX_INT32_MAX - 1,
        f"spring endpoint {smax} out of int32-safe range for vec2i",
    )


def assert_coarse_final_data_loads_in_realdata(
    coarse_final_data_path: str,
    *,
    K: int,
    device: str = "cpu",
) -> None:
    """
    Load coarse ``final_data_downsampled.pkl`` with ``RealData`` (same entry path as replay).

    Asserts the coarse contract: ``N_orig == N_surf == N_all == K`` and ``structure_points``
    row order matches ``concat(object_points[0], surface_points, interior_points)`` (numpy
    comparison to the tensor materialization on ``device``).
    """
    from qqtt.data import RealData
    from qqtt.utils import cfg

    path = str(Path(coarse_final_data_path).resolve())
    snap = {a: getattr(cfg, a, None) for a in ("data_path", "base_dir", "device", "data_type")}
    try:
        cfg.data_path = path
        cfg.base_dir = str(Path(path).parent)
        cfg.device = device
        cfg.data_type = "real"
        ds = RealData(visualize=False, save_gt=False)
        _require(
            ds.num_original_points == K == ds.num_surface_points == ds.num_all_points,
            "RealData: coarse case must have num_original_points == num_surface_points "
            f"== num_all_points == K (got N_orig={ds.num_original_points}, "
            f"N_surf={ds.num_surface_points}, N_all={ds.num_all_points}, K={K})",
        )
        struct = ds.structure_points.detach().cpu().numpy()
        frame0 = ds.object_points[0].detach().cpu().numpy()
        if not np.allclose(struct, frame0, rtol=0.0, atol=1e-5):
            raise DownsampledBundleError(
                "RealData.structure_points must equal object_points[0] for coarse bundle "
                "(empty surface/interior); ordering bug otherwise"
            )
    finally:
        for attr, val in snap.items():
            setattr(cfg, attr, val)


def assert_spring_checkpoint_alignment(
    springs: NDArray[np.int32],
    spring_Y: torch.Tensor | np.ndarray,
    num_object_springs: int,
    K: int,
    C: int,
) -> None:
    _require(springs.ndim == 2 and springs.shape[1] == 2, "springs must be (E, 2)")
    E = springs.shape[0]
    sy = spring_Y.cpu().numpy() if isinstance(spring_Y, torch.Tensor) else np.asarray(spring_Y)
    _require(sy.size == E, f"len(spring_Y)={sy.size} != n_springs={E}")
    bound = K + C
    n_oo = 0
    for e in range(E):
        i, j = int(springs[e, 0]), int(springs[e, 1])
        _require(i >= 0 and j >= 0, "negative spring index")
        _require(i < bound and j < bound, f"spring {e} endpoint out of range for K+C={bound}: ({i},{j})")
        if i < K and j < K:
            n_oo += 1
            _require(i < j, f"object–object spring row {e} must have i<j, got ({i},{j})")
        elif i >= K:
            _require(
                i < K + C,
                f"control index {i} must be in [K, K+C) with K={K}, C={C}",
            )
            _require(j < K, "control–object spring must be [K+c, alpha] with alpha < K")
        else:
            raise DownsampledBundleError(
                f"spring {e}: invalid convention ({i}, {j}) for K={K} (control must be first)"
            )
    _require(n_oo == int(num_object_springs), "num_object_springs mismatch recomputed OO count")


def assert_coarse_npz_bundle(
    z: dict[str, np.ndarray],
    *,
    K: int,
    C: int,
) -> None:
    for k in ("init_vertices", "init_springs", "init_rest_lengths", "init_masses"):
        _require(k in z, f"coarse_model.npz missing {k!r}")
    v = z["init_vertices"]
    sp = z["init_springs"]
    rl = z["init_rest_lengths"]
    ms = z["init_masses"]
    _require(v.ndim == 2 and v.shape[1] == 3, "init_vertices")
    _require(v.shape[0] == K + C, "init_vertices rows must be K + C")
    _require(sp.dtype in (np.int32, np.int64), "init_springs dtype")
    _require(rl.shape == (sp.shape[0],), "init_rest_lengths")
    _require(np.all(rl > 0), "rest lengths must be > 0")
    _require(ms.shape[0] == K + C, "init_masses length")
    if "num_all_points" in z:
        _require(int(z["num_all_points"].item()) == K, "num_all_points in npz must match K")


def validate_downsampled_bundle(
    base_path: str,
    case_name: str,
    tag: str,
    *,
    partition_N_fine: int | None = None,
    verify_realdata_loader: bool = True,
) -> dict[str, Any]:
    """
    Validate on-disk layout under ``<base>/<case>/downsampled/<tag>/``.

    ``partition_N_fine`` should be the fine ``num_all_points`` when available (from meta);
    if omitted, only partition length vs meta cross-check is skipped.

    If ``verify_realdata_loader``, load the coarse pickle through ``qqtt.data.RealData`` to
    catch count / ``structure_points`` ordering bugs.
    """
    ddir = downsampled_bundle_dir(base_path, case_name, tag)
    _require(os.path.isdir(ddir), f"missing downsample directory: {ddir}")

    p_final = os.path.join(ddir, CANONICAL_FINAL_DATA_NAME)
    p_npz = os.path.join(ddir, CANONICAL_NPZ_NAME)
    p_part = os.path.join(ddir, CANONICAL_PARTITION_NAME)
    p_meta = os.path.join(ddir, CANONICAL_META_NAME)
    p_ckpt = os.path.join(ddir, CANONICAL_CHECKPOINT_NAME)
    for p in (p_final, p_npz, p_part, p_meta, p_ckpt):
        _require(os.path.isfile(p), f"missing required artifact: {p}")

    with open(p_meta) as f:
        meta = json.load(f)
    for k in ("version", "K", "method", "r"):
        _require(k in meta, f"downsample_meta.json missing {k!r}")
    K = int(meta["K"])
    if partition_N_fine is None and "N_fine" in meta:
        partition_N_fine = int(meta["N_fine"])

    pm = np.load(p_part)
    _require(pm.ndim == 1 and pm.dtype in (np.int32, np.int64), "partition_map")
    if partition_N_fine is not None:
        _require(pm.shape[0] == partition_N_fine, "partition_map length != N_fine")
    _require(int(pm.min()) >= 0 and int(pm.max()) == K - 1, "partition labels must be 0..K-1")

    z = np.load(p_npz)
    try:
        with open(p_final, "rb") as f:
            import pickle

            data = pickle.load(f)
        T, n_orig, n_surf, n_all, n_ctrl = assert_realdata_pickle_layout(data)
        _require(
            n_orig == n_surf == n_all == K,
            "coarse final_data must satisfy N_orig == N_surf == N_all == K for current writer",
        )

        ckpt = torch.load(p_ckpt, map_location="cpu")
        for key in (
            "spring_Y",
            "collide_elas",
            "collide_fric",
            "collide_object_elas",
            "collide_object_fric",
            "num_object_springs",
        ):
            _require(key in ckpt, f"checkpoint missing {key!r}")

        springs = np.asarray(z["init_springs"], dtype=np.int32)
        assert_coarse_npz_bundle(z, K=K, C=n_ctrl)
        v_init = np.asarray(z["init_vertices"], dtype=np.float64)
        obj0 = np.asarray(data["object_points"][0], dtype=np.float64)
        _require(
            v_init.shape[0] >= K and obj0.shape == (K, 3),
            "init_vertices / object_points[0] shape mismatch",
        )
        if not np.allclose(v_init[:K], obj0, rtol=0.0, atol=1e-4):
            raise DownsampledBundleError(
                "coarse init_vertices[:K] must match final_data object_points[0] (rest pose / structure)"
            )
        ctrl0 = np.asarray(data["controller_points"][0], dtype=np.float64)
        _require(
            np.allclose(v_init[K : K + n_ctrl], ctrl0, rtol=0.0, atol=1e-4),
            "init_vertices[K:K+C] must match controller_points[0]",
        )
        assert_spring_indices_int32_safe(K, n_ctrl, springs)
        assert_spring_checkpoint_alignment(
            springs,
            ckpt["spring_Y"],
            int(ckpt["num_object_springs"]),
            K,
            n_ctrl,
        )
        if verify_realdata_loader:
            assert_coarse_final_data_loads_in_realdata(p_final, K=K, device="cpu")
    finally:
        z.close()

    return {
        "dir": ddir,
        "meta": meta,
        "K": K,
        "final_data": p_final,
        "coarse_npz": p_npz,
        "checkpoint": p_ckpt,
    }
