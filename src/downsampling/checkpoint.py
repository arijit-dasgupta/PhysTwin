"""Build coarse ``best_downsampled.pth`` compatible with replay loaders."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


def build_downsampled_checkpoint(
    spring_Y_linear: NDArray[np.float64],
    num_object_springs: int,
    fine_checkpoint: dict[str, Any],
    *,
    epoch: int = 0,
) -> dict[str, torch.Tensor | int]:
    """Linear per-spring stiffness; collision tensors copied from fine checkpoint."""
    k = torch.tensor(spring_Y_linear, dtype=torch.float32)
    return {
        "epoch": epoch,
        "num_object_springs": int(num_object_springs),
        "spring_Y": k,
        "collide_elas": fine_checkpoint["collide_elas"].detach().clone()
        if isinstance(fine_checkpoint["collide_elas"], torch.Tensor)
        else torch.tensor(fine_checkpoint["collide_elas"], dtype=torch.float32),
        "collide_fric": fine_checkpoint["collide_fric"].detach().clone()
        if isinstance(fine_checkpoint["collide_fric"], torch.Tensor)
        else torch.tensor(fine_checkpoint["collide_fric"], dtype=torch.float32),
        "collide_object_elas": fine_checkpoint["collide_object_elas"].detach().clone()
        if isinstance(fine_checkpoint["collide_object_elas"], torch.Tensor)
        else torch.tensor(fine_checkpoint["collide_object_elas"], dtype=torch.float32),
        "collide_object_fric": fine_checkpoint["collide_object_fric"].detach().clone()
        if isinstance(fine_checkpoint["collide_object_fric"], torch.Tensor)
        else torch.tensor(fine_checkpoint["collide_object_fric"], dtype=torch.float32),
    }
