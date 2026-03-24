"""Unit tests for spring–mass downsampling (no GPU)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from downsampling.build_downsampled_model import (
    _coarse_colors,
    _format_r_for_tag,
    default_auto_tag_base,
    target_coarse_cluster_count,
)
from downsampling.partition_kmeans import CoarseningFeasibilityError
from downsampling.reconstruct import (
    active_linear_stiffness,
    coarse_log_stiffness_from_linear_sum,
    effective_stiffness_linear_checkpoint_row,
)
from downsampling.types import PartitionMethod
from downsampling.validate import (
    CANONICAL_META_NAME,
    DownsampledBundleError,
    allocate_unique_downsample_tag,
    assert_coarse_final_data_loads_in_realdata,
    assert_spring_checkpoint_alignment,
    assert_spring_indices_int32_safe,
    downsampled_bundle_dir,
    downsampled_bundle_exists,
    validate_downsampled_bundle,
)


def test_active_linear_stiffness_below_min_is_zero() -> None:
    assert active_linear_stiffness(1.0, y_min=10.0, y_max=100.0) == 0.0


def test_active_linear_stiffness_clamp() -> None:
    assert active_linear_stiffness(500.0, y_min=10.0, y_max=100.0) == 100.0
    assert active_linear_stiffness(50.0, y_min=10.0, y_max=100.0) == 50.0


def test_coarse_log_stiffness_inactive() -> None:
    assert coarse_log_stiffness_from_linear_sum(0.0, y_min=1.0) is None


def test_effective_stiffness_at_y_min_is_inactive() -> None:
    """
    Match Warp ``eval_springs``: branch is ``exp(Y) > spring_Y_min`` — at equality the spring
    does not apply, consistent with inactive treatment for checkpoint linear k <= y_min.
    """
    y_min, y_max = 10.0, 100.0
    assert effective_stiffness_linear_checkpoint_row(10.0, y_min, y_max) == 0.0
    assert effective_stiffness_linear_checkpoint_row(10.0 + 1e-12, y_min, y_max) == y_min + 1e-12


def test_effective_stiffness_above_y_max_clamped() -> None:
    assert effective_stiffness_linear_checkpoint_row(500.0, y_min=10.0, y_max=100.0) == 100.0


def test_effective_stiffness_nan_inf_are_inactive() -> None:
    y_min, y_max = 1.0, 10.0
    assert effective_stiffness_linear_checkpoint_row(float("nan"), y_min, y_max) == 0.0
    assert effective_stiffness_linear_checkpoint_row(float("inf"), y_min, y_max) == 0.0
    assert effective_stiffness_linear_checkpoint_row(float("-inf"), y_min, y_max) == 0.0


def test_effective_stiffness_alias_matches_active() -> None:
    assert effective_stiffness_linear_checkpoint_row(50.0, 10.0, 100.0) == active_linear_stiffness(
        50.0, 10.0, 100.0
    )


def test_assert_spring_indices_int32_safe_rejects_gap_index() -> None:
    with pytest.raises(DownsampledBundleError, match="spring index max"):
        assert_spring_indices_int32_safe(2, 1, np.array([[0, 10]], dtype=np.int32))


def test_coarse_final_data_realdata_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib.pyplot")  # RealData rainbow path
    K, T, C = 4, 3, 1
    obj = np.random.randn(T, K, 3).astype(np.float32)
    ctrl = np.random.randn(T, C, 3).astype(np.float32)
    pkl = tmp_path / "final_data_downsampled.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(
            {
                "object_points": obj,
                "object_colors": np.ones((T, K, 3), dtype=np.float32),
                "object_visibilities": np.ones((T, K), dtype=bool),
                "object_motions_valid": np.ones((T, K), dtype=bool),
                "controller_points": ctrl,
                "surface_points": np.zeros((0, 3), dtype=np.float32),
                "interior_points": np.zeros((0, 3), dtype=np.float32),
            },
            f,
        )
    assert_coarse_final_data_loads_in_realdata(str(pkl), K=K, device="cpu")


def test_validate_fails_when_required_artifact_removed(tmp_path: Path) -> None:
    case, tag = "rm", "art"
    base = tmp_path
    root = Path(downsampled_bundle_dir(str(base), case, tag))
    root.mkdir(parents=True)
    K, C, N_fine, T = 2, 1, 3, 2
    obj = np.random.randn(T, K, 3).astype(np.float32)
    ctrl = np.random.randn(T, C, 3).astype(np.float32)
    init_vertices = np.vstack([obj[0].astype(np.float64), ctrl[0].astype(np.float64)])
    springs = np.array([[0, 1], [K + 0, 0]], dtype=np.int32)
    E = springs.shape[0]
    np.savez(
        root / "coarse_model.npz",
        init_vertices=init_vertices,
        init_springs=springs,
        init_rest_lengths=np.full(E, 0.1, dtype=np.float64),
        init_masses=np.ones(K + C, dtype=np.float64),
        num_object_springs=np.int32(1),
        num_all_points=np.int32(K),
    )
    np.save(root / "partition_map.npy", np.array([0, 1, 0], dtype=np.int32))
    (root / "downsample_meta.json").write_text(
        json.dumps({"version": 1, "K": K, "method": "kmeans", "r": 2.0, "N_fine": N_fine})
    )
    with open(root / "final_data_downsampled.pkl", "wb") as f:
        pickle.dump(
            {
                "object_points": obj,
                "object_colors": np.ones((T, K, 3), dtype=np.float32),
                "object_visibilities": np.ones((T, K), dtype=bool),
                "object_motions_valid": np.ones((T, K), dtype=bool),
                "controller_points": ctrl,
                "surface_points": np.zeros((0, 3), dtype=np.float32),
                "interior_points": np.zeros((0, 3), dtype=np.float32),
            },
            f,
        )
    torch.save(
        {
            "epoch": 0,
            "num_object_springs": 1,
            "spring_Y": torch.ones(E, dtype=torch.float32) * 1e4,
            "collide_elas": torch.tensor(0.1),
            "collide_fric": torch.tensor(0.2),
            "collide_object_elas": torch.tensor(0.3),
            "collide_object_fric": torch.tensor(0.4),
        },
        root / "best_downsampled.pth",
    )
    validate_downsampled_bundle(str(base), case, tag, partition_N_fine=N_fine, verify_realdata_loader=True)
    (root / "best_downsampled.pth").unlink()
    with pytest.raises(DownsampledBundleError, match="missing required artifact"):
        validate_downsampled_bundle(str(base), case, tag, partition_N_fine=N_fine)


def test_validate_bundle_missing_dir() -> None:
    with pytest.raises(DownsampledBundleError, match="missing downsample"):
        validate_downsampled_bundle("/nonexistent", "c", "t", partition_N_fine=10)


def test_coarse_colors_constant_y_no_nan() -> None:
    """All vertices same Y → rainbow norm must not produce NaNs."""
    traj = np.zeros((2, 4, 3), dtype=np.float32)
    traj[..., 1] = 3.0
    c = _coarse_colors(traj)
    assert c.shape == (2, 4, 3) and np.all(np.isfinite(c))


def test_target_coarse_cluster_count_rejects_k_gt_n() -> None:
    with pytest.raises(CoarseningFeasibilityError, match="K="):
        target_coarse_cluster_count(10, 0.25)


def test_target_coarse_cluster_count_normal() -> None:
    assert target_coarse_cluster_count(100, 2.0) == 50
    assert target_coarse_cluster_count(100, 2.5) == 40


def test_format_r_for_tag() -> None:
    assert _format_r_for_tag(2.0) == "2"
    assert _format_r_for_tag(2.5) == "2p5"
    assert _format_r_for_tag(10) == "10"


def test_default_auto_tag_base() -> None:
    assert default_auto_tag_base(PartitionMethod.kmeans, 2.0) == "kmeans_r2"
    assert default_auto_tag_base(PartitionMethod.graph, 2.5) == "graph_r2p5"


def test_downsampled_bundle_exists_and_allocate_unique(tmp_path: Path) -> None:
    base = str(tmp_path)
    case = "c1"
    assert not downsampled_bundle_exists(base, case, "kmeans_r2")
    assert allocate_unique_downsample_tag(base, case, "kmeans_r2") == "kmeans_r2"
    root = Path(downsampled_bundle_dir(base, case, "kmeans_r2"))
    root.mkdir(parents=True)
    (root / CANONICAL_META_NAME).write_text("{}")
    assert downsampled_bundle_exists(base, case, "kmeans_r2")
    assert allocate_unique_downsample_tag(base, case, "kmeans_r2") == "kmeans_r2_2"


def test_validate_minimal_bundle_round_trip(tmp_path: Path) -> None:
    case = "fake_case"
    tag = "testtag"
    base = tmp_path
    root = Path(downsampled_bundle_dir(str(base), case, tag))
    root.mkdir(parents=True)

    K, C, N_fine, T = 3, 2, 7, 4
    obj = np.random.randn(T, K, 3).astype(np.float32)
    ctrl = np.random.randn(T, C, 3).astype(np.float32)
    init_vertices = np.vstack([obj[0].astype(np.float64), ctrl[0].astype(np.float64)])
    springs = np.array(
        [
            [0, 1],
            [1, 2],
            [K + 0, 0],
            [K + 1, 1],
        ],
        dtype=np.int32,
    )
    E = springs.shape[0]
    rest = np.full(E, 0.05, dtype=np.float64)
    masses = np.ones(K + C, dtype=np.float64)
    np.savez(
        root / "coarse_model.npz",
        init_vertices=init_vertices,
        init_springs=springs,
        init_rest_lengths=rest,
        init_masses=masses,
        num_object_springs=np.int32(2),
        num_all_points=np.int32(K),
    )

    partition = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int32)
    np.save(root / "partition_map.npy", partition)

    meta = {
        "version": 1,
        "K": K,
        "method": "kmeans",
        "r": 2.5,
        "N_fine": N_fine,
    }
    (root / "downsample_meta.json").write_text(json.dumps(meta))

    final_data = {
        "object_points": obj,
        "object_colors": np.ones((T, K, 3), dtype=np.float32) * 0.5,
        "object_visibilities": np.ones((T, K), dtype=bool),
        "object_motions_valid": np.ones((T, K), dtype=bool),
        "controller_points": ctrl,
        "surface_points": np.zeros((0, 3), dtype=np.float32),
        "interior_points": np.zeros((0, 3), dtype=np.float32),
    }
    with open(root / "final_data_downsampled.pkl", "wb") as f:
        pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    ckpt = {
        "epoch": 0,
        "num_object_springs": 2,
        "spring_Y": torch.ones(E, dtype=torch.float32) * 1000.0,
        "collide_elas": torch.tensor(0.1),
        "collide_fric": torch.tensor(0.2),
        "collide_object_elas": torch.tensor(0.3),
        "collide_object_fric": torch.tensor(0.4),
    }
    torch.save(ckpt, root / "best_downsampled.pth")

    info = validate_downsampled_bundle(str(base), case, tag, partition_N_fine=N_fine)
    assert info["K"] == K
    assert (Path(info["dir"]) / "best_downsampled.pth").is_file()


def test_spring_alignment_rejects_wrong_oo_order() -> None:
    K = 3
    C = 0
    springs = np.array([[1, 0]], dtype=np.int32)  # should be i < j
    y = torch.ones(1)
    with pytest.raises(DownsampledBundleError, match="i<j"):
        assert_spring_checkpoint_alignment(springs, y, 1, K, C)


def test_spring_alignment_rejects_index_out_of_range() -> None:
    K, C = 2, 1
    springs = np.array([[K + C, 0]], dtype=np.int32)  # invalid control index
    y = torch.ones(1)
    with pytest.raises(DownsampledBundleError, match="out of range"):
        assert_spring_checkpoint_alignment(springs, y, 0, K, C)


def test_validate_bundle_rejects_geometry_mismatch(tmp_path: Path) -> None:
    """init_vertices must match object_points[0] and controller_points[0]."""
    case, tag = "c", "t"
    base = tmp_path
    root = Path(downsampled_bundle_dir(str(base), case, tag))
    root.mkdir(parents=True)
    K, C, N_fine, T, E = 2, 1, 4, 2, 1
    bad_vertices = np.zeros((K + C, 3), dtype=np.float64)
    bad_vertices[K:] = 1.0
    np.savez(
        root / "coarse_model.npz",
        init_vertices=bad_vertices,
        init_springs=np.array([[0, 1]], dtype=np.int32),
        init_rest_lengths=np.array([0.1], dtype=np.float64),
        init_masses=np.ones(K + C, dtype=np.float64),
        num_object_springs=np.int32(1),
        num_all_points=np.int32(K),
    )
    np.save(root / "partition_map.npy", np.array([0, 1, 0, 1], dtype=np.int32))
    (root / "downsample_meta.json").write_text(
        json.dumps({"version": 1, "K": K, "method": "k", "r": 1.0, "N_fine": N_fine})
    )
    obj = np.zeros((T, K, 3), dtype=np.float32)
    ctrl = np.zeros((T, C, 3), dtype=np.float32)
    with open(root / "final_data_downsampled.pkl", "wb") as f:
        pickle.dump(
            {
                "object_points": obj,
                "object_colors": np.ones((T, K, 3), dtype=np.float32),
                "object_visibilities": np.ones((T, K), dtype=bool),
                "object_motions_valid": np.ones((T, K), dtype=bool),
                "controller_points": ctrl,
                "surface_points": np.zeros((0, 3), dtype=np.float32),
                "interior_points": np.zeros((0, 3), dtype=np.float32),
            },
            f,
        )
    torch.save(
        {
            "epoch": 0,
            "num_object_springs": 1,
            "spring_Y": torch.ones(E) * 1e4,
            "collide_elas": torch.tensor(0.1),
            "collide_fric": torch.tensor(0.2),
            "collide_object_elas": torch.tensor(0.3),
            "collide_object_fric": torch.tensor(0.4),
        },
        root / "best_downsampled.pth",
    )
    with pytest.raises(DownsampledBundleError, match="controller_points"):
        validate_downsampled_bundle(str(base), case, tag, partition_N_fine=N_fine)


def test_partition_kmeans_small() -> None:
    from downsampling.partition_kmeans import partition_kmeans

    X = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float64)
    m = np.ones(3, dtype=np.float64)
    pi = partition_kmeans(X, m, 2, random_state=0, n_init=5)
    assert pi.shape == (3,)
    assert set(np.unique(pi).tolist()) == {0, 1}


def test_build_coarse_merges_duplicate_oo_bucket() -> None:
    from downsampling.reconstruct import build_coarse_springs_and_rest

    N, K = 4, 2
    partition = np.array([0, 0, 1, 1], dtype=np.int32)
    springs = np.array([[0, 2], [2, 0]], dtype=np.int32)
    rest = np.array([0.5, 0.6], dtype=np.float64)
    kf = np.array([1000.0, 1000.0], dtype=np.float64)
    X0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    ctrl0 = np.array([[5.0, 0.0, 0.0]], dtype=np.float64)
    cs, cr, ck, n_oo = build_coarse_springs_and_rest(
        springs,
        rest,
        kf,
        partition,
        N,
        K,
        X0,
        ctrl0,
        y_min=1.0,
        y_max=1e6,
        rest_mode="centroid_distance",
        stiffness_mode="sum",
        stiffness_scale=1.0,
    )
    assert n_oo == 1 and cs.shape[0] == 1
    assert ck[0] > 0


def test_build_coarse_all_inactive_springs_empty() -> None:
    from downsampling.reconstruct import build_coarse_springs_and_rest

    N, K = 2, 2
    partition = np.array([0, 1], dtype=np.int32)
    springs = np.array([[0, 1]], dtype=np.int32)
    rest = np.array([0.5], dtype=np.float64)
    kf = np.array([0.5], dtype=np.float64)
    X0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    ctrl0 = np.zeros((0, 3), dtype=np.float64)
    cs, _, _, n_oo = build_coarse_springs_and_rest(
        springs,
        rest,
        kf,
        partition,
        N,
        K,
        X0,
        ctrl0,
        y_min=10.0,
        y_max=1e6,
        rest_mode="centroid_distance",
        stiffness_mode="sum",
        stiffness_scale=1.0,
    )
    assert cs.shape[0] == 0 and n_oo == 0


def test_partition_graph_two_components() -> None:
    from downsampling.partition_graph import partition_graph_coarsen

    X0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [101.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    springs = np.array([[0, 1], [2, 3]], dtype=np.int32)
    pi = partition_graph_coarsen(X0, springs, 4, 2, protected=None, fine_masks=None)
    assert len(np.unique(pi)) == 2
    assert pi[0] == pi[1] and pi[2] == pi[3] and pi[0] != pi[2]


def test_partition_graph_forbids_cross_mask_merge() -> None:
    from downsampling.partition_graph import partition_graph_coarsen

    X0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    springs = np.array([[0, 1], [1, 2]], dtype=np.int32)
    masks = np.array([0, 0, 1], dtype=np.int32)
    pi = partition_graph_coarsen(X0, springs, 3, 2, protected=None, fine_masks=masks)
    assert pi[2] != pi[0] and pi[2] != pi[1]
