"""Mass-weighted K-means partition on rest positions (see docs/downsampling_spring_mass.md)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


class CoarseningFeasibilityError(RuntimeError):
    pass


def _allocate_clusters_proportional(sizes: list[int], k_total: int) -> list[int]:
    """Non-negative integers summing to k_total; active groups get >= 1."""
    g = len(sizes)
    if g == 0:
        return []
    n_act = sum(1 for s in sizes if s > 0)
    if k_total < n_act:
        raise CoarseningFeasibilityError(
            f"need at least one cluster per mask group ({n_act} groups), got k_total={k_total}"
        )
    tot_pts = sum(sizes)
    if tot_pts <= 0:
        raise CoarseningFeasibilityError("no free points for k-means")
    alloc: list[int] = []
    for s in sizes:
        if s == 0:
            alloc.append(0)
        else:
            alloc.append(max(1, int(round(k_total * s / tot_pts))))
    delta = k_total - sum(alloc)
    while delta < 0:
        j = max(range(g), key=lambda i: alloc[i])
        alloc[j] -= 1
        delta += 1
    idx = 0
    while delta > 0:
        if sizes[idx % g] > 0:
            alloc[idx % g] += 1
            delta -= 1
        idx += 1
    return alloc


def partition_kmeans(
    X0: NDArray[np.float64],
    masses: NDArray[np.float64],
    K: int,
    *,
    protected: NDArray[np.bool_] | None = None,
    fine_masks: NDArray[np.int32] | None = None,
    random_state: int = 0,
    n_init: int = 10,
) -> NDArray[np.int32]:
    """
    Return partition map pi: N -> K clusters (labels 0..K-1).

    Protected vertices get distinct labels K-P .. K-1.
    """
    n = X0.shape[0]
    if K > n or K < 1:
        raise CoarseningFeasibilityError(f"invalid K={K} for N={n}")
    if protected is None:
        protected = np.zeros(n, dtype=np.bool_)
    if protected.shape[0] != n:
        raise ValueError("protected shape mismatch")
    prot_idx = np.where(protected)[0]
    p_n = int(prot_idx.size)
    if p_n > K:
        raise CoarseningFeasibilityError(f"protected count {p_n} exceeds target K={K}")
    if fine_masks is not None and fine_masks.shape[0] != n:
        raise ValueError("fine_masks length mismatch")

    labels = np.full(n, -1, dtype=np.int32)
    if p_n == n:
        for i, idx in enumerate(sorted(prot_idx.tolist())):
            labels[idx] = i
        return labels

    free_idx = np.where(~protected)[0]
    k_free = K - p_n
    if k_free < 1:
        raise CoarseningFeasibilityError("no room for kmeans after reserving protected labels")

    xf = X0[free_idx]
    mf = masses[free_idx]
    lab_free = np.zeros(free_idx.shape[0], dtype=np.int32)

    if fine_masks is None:
        km = KMeans(n_clusters=k_free, random_state=random_state, n_init=n_init)
        lab_free = km.fit_predict(xf, sample_weight=mf).astype(np.int32)
    else:
        fm = fine_masks[free_idx]
        uniq = np.unique(fm)
        if uniq.size == 1:
            km = KMeans(n_clusters=k_free, random_state=random_state, n_init=n_init)
            lab_free = km.fit_predict(xf, sample_weight=mf).astype(np.int32)
        else:
            groups = [np.where(fm == u)[0] for u in uniq]
            sizes = [int(len(ix)) for ix in groups]
            alloc = _allocate_clusters_proportional(sizes, k_free)
            cur = 0
            for gix, local_ix in enumerate(groups):
                k_sub = alloc[gix]
                if k_sub <= 0:
                    continue
                sub_x = xf[local_ix]
                sub_w = mf[local_ix]
                if len(local_ix) < k_sub:
                    raise CoarseningFeasibilityError(
                        f"mask group has {len(local_ix)} points but needs {k_sub} clusters"
                    )
                km = KMeans(
                    n_clusters=k_sub,
                    random_state=random_state + 1000 * gix,
                    n_init=n_init,
                )
                sub_lab = km.fit_predict(sub_x, sample_weight=sub_w).astype(np.int32)
                lab_free[local_ix] = sub_lab + cur
                cur += k_sub
            uniq_l = np.unique(lab_free)
            mp = {int(old): i for i, old in enumerate(sorted(uniq_l.tolist()))}
            lab_free = np.array([mp[int(v)] for v in lab_free], dtype=np.int32)

    labels[free_idx] = lab_free
    for j, idx in enumerate(sorted(prot_idx.tolist())):
        labels[idx] = K - p_n + j

    if np.any(labels < 0) or int(labels.max()) >= K:
        raise RuntimeError("invalid label assignment")
    if np.unique(labels).size != K:
        raise CoarseningFeasibilityError(
            f"expected {K} unique labels, got {np.unique(labels).size}"
        )
    return labels
