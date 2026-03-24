"""Filesystem paths and numpy/pickle IO for downsampled artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from downsampling.validate import (
    CANONICAL_CHECKPOINT_NAME,
    CANONICAL_FINAL_DATA_NAME,
    CANONICAL_META_NAME,
    CANONICAL_NPZ_NAME,
    CANONICAL_PARTITION_NAME,
    downsampled_bundle_dir,
)


def artifact_paths(base_path: str, case_name: str, tag: str) -> dict[str, str]:
    root = downsampled_bundle_dir(base_path, case_name, tag)
    return {
        "root": root,
        "final_data": str(Path(root) / CANONICAL_FINAL_DATA_NAME),
        "coarse_npz": str(Path(root) / CANONICAL_NPZ_NAME),
        "partition": str(Path(root) / CANONICAL_PARTITION_NAME),
        "meta": str(Path(root) / CANONICAL_META_NAME),
        "checkpoint": str(Path(root) / CANONICAL_CHECKPOINT_NAME),
    }


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_final_data_pickle(path: str, data: dict[str, Any]) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_coarse_npz(
    path: str,
    *,
    init_vertices: NDArray[np.float64],
    init_springs: NDArray[np.int32],
    init_rest_lengths: NDArray[np.float64],
    init_masses: NDArray[np.float64],
    num_object_springs: int,
    num_all_points: int,
) -> None:
    np.savez(
        path,
        init_vertices=init_vertices.astype(np.float64),
        init_springs=init_springs.astype(np.int32),
        init_rest_lengths=init_rest_lengths.astype(np.float64),
        init_masses=init_masses.astype(np.float64),
        num_object_springs=np.int32(num_object_springs),
        num_all_points=np.int32(num_all_points),
    )


def load_coarse_npz(path: str) -> dict[str, np.ndarray]:
    z = np.load(path)
    try:
        return {k: z[k] for k in z.files}
    finally:
        z.close()


def save_meta_json(path: str, meta: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def save_partition_map(path: str, partition: NDArray[np.int32]) -> None:
    np.save(path, partition.astype(np.int32))


__all__ = [
    "artifact_paths",
    "ensure_dir",
    "save_final_data_pickle",
    "save_coarse_npz",
    "load_coarse_npz",
    "save_meta_json",
    "save_partition_map",
]
