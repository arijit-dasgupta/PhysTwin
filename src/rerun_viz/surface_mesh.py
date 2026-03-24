"""Surface mesh from object–object spring graph: triangle = 3-clique (3-cycle)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SurfaceMeshError(RuntimeError):
    """Could not derive any triangles from the spring graph."""


def _undirected_oo_edges(springs: NDArray[np.integer], n_mesh: int) -> NDArray[np.int64]:
    """Rows (u, v) with u < v, both in [0, n_mesh)."""
    s = np.asarray(springs, dtype=np.int64)
    if s.ndim != 2 or s.shape[1] != 2:
        raise ValueError("springs must be (M, 2)")
    a, b = s[:, 0], s[:, 1]
    mask = (a >= 0) & (b >= 0) & (a < n_mesh) & (b < n_mesh)
    s = s[mask]
    if s.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    lo = np.minimum(s[:, 0], s[:, 1])
    hi = np.maximum(s[:, 0], s[:, 1])
    edges = np.stack([lo, hi], axis=1)
    edges = np.unique(edges, axis=0)
    return edges[edges[:, 0] != edges[:, 1]]


def _enumerate_triangles_from_adj(adj: list[set[int]], n: int) -> list[tuple[int, int, int]]:
    tris: list[tuple[int, int, int]] = []
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u].intersection(adj[v]):
                if w > v:
                    tris.append((u, v, w))
    return tris


def _triangle_filter(
    positions: NDArray[np.floating],
    tris: list[tuple[int, int, int]],
    max_edge: float,
) -> NDArray[np.int32]:
    if not tris:
        return np.zeros((0, 3), dtype=np.int32)
    p = positions.astype(np.float64)
    out: list[tuple[int, int, int]] = []
    for a, b, c in tris:
        e0 = float(np.linalg.norm(p[a] - p[b]))
        e1 = float(np.linalg.norm(p[b] - p[c]))
        e2 = float(np.linalg.norm(p[c] - p[a]))
        if max(e0, e1, e2) <= max_edge:
            out.append((a, b, c))
    if not out:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(out, dtype=np.int32)


def triangles_from_object_springs(
    positions: NDArray[np.floating],
    springs: NDArray[np.integer],
    n_mesh: int,
    spring_max_edge_factor: float,
) -> NDArray[np.int32]:
    """
    Build triangle indices from 3-cliques in the OO spring graph on ``[0, n_mesh)``.

    Drops triangles whose longest edge exceeds ``spring_max_edge_factor * median_oo_edge``.
    If that removes every triangle, retries without edge filtering.
    """
    if n_mesh < 3:
        raise SurfaceMeshError(f"need at least 3 mesh vertices, got n_mesh={n_mesh}")
    pos = np.asarray(positions, dtype=np.float64)
    if pos.shape[0] < n_mesh or pos.shape[1] != 3:
        raise ValueError("positions must be (>=n_mesh, 3)")

    edges = _undirected_oo_edges(springs, n_mesh)
    if edges.shape[0] == 0:
        raise SurfaceMeshError("no object–object springs in [0, n_mesh)")

    adj: list[set[int]] = [set() for _ in range(n_mesh)]
    lens: list[float] = []
    for u, v in edges:
        u, v = int(u), int(v)
        adj[u].add(v)
        adj[v].add(u)
        lens.append(float(np.linalg.norm(pos[u] - pos[v])))
    median = float(np.median(lens)) if lens else 1.0
    if median < 1e-12:
        median = 1.0

    tris = _enumerate_triangles_from_adj(adj, n_mesh)
    if not tris:
        raise SurfaceMeshError("spring graph has no 3-cliques (triangles)")

    max_edge = spring_max_edge_factor * median
    tri_arr = _triangle_filter(pos, tris, max_edge)
    if tri_arr.shape[0] == 0:
        tri_arr = np.asarray(tris, dtype=np.int32)
    return tri_arr


def build_surface_mesh(
    points: NDArray[np.floating],
    springs: NDArray[np.integer],
    n_object_vertices: int,
    spring_max_edge_factor: float = 6.0,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Rest-pose mesh: vertices are ``points[:n_object_vertices]``, triangles from 3-cliques.

    Returns:
        vertices (N,3) float32, triangle_indices (T,3) int32
    """
    tri = triangles_from_object_springs(
        points, springs, n_object_vertices, spring_max_edge_factor
    )
    verts = np.asarray(points[:n_object_vertices], dtype=np.float32)
    return verts, tri


def subsample_triangles_to_budget(
    vertices: NDArray[np.floating],
    triangle_indices: NDArray[np.integer],
    max_triangles: int,
) -> NDArray[np.int32]:
    """
    Cap triangle count for visualization / smaller ``.rrd`` files.

    We **keep vertex indices 0..N-1 unchanged** so per-frame ``simulator`` positions still align
    with ``Mesh3D`` vertices (quadric decimation would remap vertices and break that).

    At **rest** positions, triangles are sorted by **descending area**; we keep the
    ``max_triangles`` largest faces first so bulk shape is preserved while dropping many
    small faces (typical of dense spring meshes).

    Args:
        vertices: (N, 3) positions (rest pose).
        triangle_indices: (T, 3) int indices into vertices.
        max_triangles: Upper bound on output triangles; ``<= 0`` means no limit.

    Returns:
        (T', 3) int32 with ``T' <= max_triangles`` (or unchanged if already under budget).
    """
    tri = np.asarray(triangle_indices, dtype=np.int64)
    if max_triangles <= 0 or tri.shape[0] <= max_triangles:
        return tri.astype(np.int32)
    p = np.asarray(vertices, dtype=np.float64)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    pa, pb, pc = p[a], p[b], p[c]
    areas = 0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa), axis=1)
    order = np.argsort(-areas)
    keep = order[:max_triangles]
    return tri[keep].astype(np.int32)
