"""Greedy local graph coarsening on object-object springs (geometric merge cost)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from downsampling.partition_kmeans import CoarseningFeasibilityError


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = np.arange(n, dtype=np.int32)
        self.r = np.zeros(n, dtype=np.int32)

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = int(self.p[x])
        return int(x)

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def partition_graph_coarsen(
    X0: NDArray[np.float64],
    springs: NDArray[np.int32],
    num_object_points: int,
    K: int,
    *,
    protected: NDArray[np.bool_] | None = None,
    fine_masks: NDArray[np.int32] | None = None,
) -> NDArray[np.int32]:
    """
    Repeatedly merge the lowest-cost admissible object-object edge until K supernodes.

    Cost = ||x_i^0 - x_j^0||^2. Protected nodes never merge with anything.
    Merges across different fine_masks (when provided) are forbidden.
    """
    n = int(num_object_points)
    if K > n or K < 1:
        raise CoarseningFeasibilityError(f"invalid K={K} for N={n}")
    if protected is None:
        protected = np.zeros(n, dtype=np.bool_)
    if fine_masks is not None and fine_masks.shape[0] != n:
        raise ValueError("fine_masks mismatch")

    uf = UnionFind(n)
    active = n

    def root_mask(r: int) -> int | None:
        if fine_masks is None:
            return None
        members = np.where(np.array([uf.find(i) for i in range(n)], dtype=np.int32) == r)[0]
        if members.size == 0:
            return None
        ms = int(fine_masks[members[0]])
        if not np.all(fine_masks[members] == ms):
            return -1  # inconsistent (should not happen)
        return ms

    def can_merge(ra: int, rb: int) -> bool:
        if ra == rb:
            return False
        for r in (ra, rb):
            members = [i for i in range(n) if uf.find(i) == r]
            if any(protected[i] for i in members):
                return False
        if fine_masks is not None:
            ma = root_mask(ra)
            mb = root_mask(rb)
            if ma is not None and mb is not None and ma != mb:
                return False
        return True

    # collect edges
    edges: list[tuple[float, int, int]] = []
    n_spr = int(springs.shape[0])
    _spr_iter = range(n_spr)
    if n_spr >= 5000:
        _spr_iter = tqdm(
            _spr_iter,
            desc="graph coarsen collect",
            unit="spr",
            leave=False,
            ncols=88,
        )
    for e in _spr_iter:
        i1, i2 = int(springs[e, 0]), int(springs[e, 1])
        if i1 >= num_object_points or i2 >= num_object_points:
            continue
        if i1 == i2:
            continue
        d = float(np.sum((X0[i1] - X0[i2]) ** 2))
        a, b = (i1, i2) if i1 < i2 else (i2, i1)
        edges.append((d, a, b))
    edges.sort(key=lambda t: (t[0], t[1], t[2]))

    merges_needed = n - K
    ei = 0
    pbar = tqdm(
        total=merges_needed,
        desc="graph coarsen merge",
        unit="merge",
        ncols=88,
        disable=merges_needed <= 0,
    )
    try:
        while active > K and ei < len(edges):
            d, a, b = edges[ei]
            ei += 1
            ra, rb = uf.find(a), uf.find(b)
            if not can_merge(ra, rb):
                continue
            if uf.union(a, b):
                active -= 1
                pbar.update(1)
                pbar.set_postfix(active=active, edges=ei, refresh=False)
    finally:
        pbar.close()

    if active > K:
        raise CoarseningFeasibilityError(
            f"graph coarsening stuck at {active} supernodes (> K={K}); "
            "try kmeans, lower r, or relax constraints"
        )

    # label components 0..K-1
    roots = np.unique([uf.find(i) for i in range(n)])
    if roots.size != K:
        raise CoarseningFeasibilityError(
            f"internal error: expected {K} roots, got {roots.size}"
        )
    r_sorted = sorted(int(r) for r in roots.tolist())
    mp = {r: i for i, r in enumerate(r_sorted)}
    _lab_iter = range(n)
    if n >= 50000:
        _lab_iter = tqdm(_lab_iter, desc="graph coarsen labels", unit="v", leave=False, ncols=88)
    out = np.empty(n, dtype=np.int32)
    for i in _lab_iter:
        out[i] = mp[uf.find(i)]
    return out
