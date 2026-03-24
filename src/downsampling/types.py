"""Enums and small types for coarsening."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PartitionMethod(str, Enum):
    kmeans = "kmeans"
    graph = "graph"


class RestLengthMode(str, Enum):
    """How to set coarse rest length per bucket (baseline from doc)."""

    centroid_distance = "centroid_distance"
    mean_fine_rest = "mean_fine_rest"
    stiffness_weighted_rest = "stiffness_weighted_rest"


class StiffnessMode(str, Enum):
    sum = "sum"
    mean = "mean"
    scaled_sum = "scaled_sum"


@dataclass(frozen=True)
class CoarseningParams:
    """User-facing knobs for one coarsening run."""

    method: PartitionMethod
    r: float
    rest_length_mode: RestLengthMode = RestLengthMode.centroid_distance
    stiffness_mode: StiffnessMode = StiffnessMode.sum
    stiffness_scale: float = 1.0
    kmeans_random_state: int = 0
    kmeans_n_init: int = 10
