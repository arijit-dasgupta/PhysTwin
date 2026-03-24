"""Tests for spring-graph surface meshes (3-cliques)."""

from __future__ import annotations

import numpy as np
import pytest

from rerun_viz.surface_mesh import (
    SurfaceMeshError,
    build_surface_mesh,
    subsample_triangles_to_budget,
    triangles_from_object_springs,
)


def test_square_with_diagonal_two_triangles() -> None:
    # z=0 square (0,0),(1,0),(1,1),(0,1) + diagonal 0-2
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    springs = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 2],
        ],
        dtype=np.int64,
    )
    n = 4
    tri = triangles_from_object_springs(pts, springs, n, spring_max_edge_factor=100.0)
    assert tri.shape[0] == 2
    assert tri.shape[1] == 3


def test_edges_only_raises() -> None:
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    springs = np.array([[0, 1], [1, 2]], dtype=np.int64)
    with pytest.raises(SurfaceMeshError):
        triangles_from_object_springs(pts, springs, 3, spring_max_edge_factor=10.0)


def test_build_surface_mesh_returns_verts_slice() -> None:
    pts = np.zeros((10, 3), dtype=np.float64)
    pts[:4, :2] = [[0, 0], [1, 0], [1, 1], [0, 1]]
    springs = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.int64)
    v, tri = build_surface_mesh(pts, springs, 4, spring_max_edge_factor=100.0)
    assert v.shape == (4, 3)
    assert tri.shape[0] == 2


def test_subsample_triangles_to_budget_noop_when_under_cap() -> None:
    pts = np.zeros((4, 3), dtype=np.float64)
    tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    out = subsample_triangles_to_budget(pts, tri, max_triangles=10)
    assert out.shape == tri.shape


def test_subsample_triangles_to_budget_keeps_largest() -> None:
    # Two triangles: one large, one tiny
    pts = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.01]],
        dtype=np.float64,
    )
    tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    out = subsample_triangles_to_budget(pts, tri, max_triangles=1)
    assert out.shape[0] == 1
    # Large triangle is 0,1,2
    assert np.array_equal(out[0], tri[0])
