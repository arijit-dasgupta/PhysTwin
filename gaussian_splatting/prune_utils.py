"""Lightweight Gaussian pruning helpers (no pytorch3d / full render CLI deps)."""

from __future__ import annotations

import copy

import numpy as np
import torch
from tqdm import tqdm


def remove_gaussians_with_low_opacity(gaussians, opacity_threshold: float = 0.1):
    """Drop Gaussians whose opacity is below ``opacity_threshold``."""
    opacity = gaussians.get_opacity.squeeze(-1)
    mask3d = opacity > opacity_threshold
    print(f"Removing {len(mask3d) - mask3d.sum()} gaussians with opacity < {opacity_threshold}")

    new_gaussians = copy.deepcopy(gaussians)
    new_gaussians._xyz = gaussians._xyz[mask3d]
    new_gaussians._features_dc = gaussians._features_dc[mask3d]
    new_gaussians._features_rest = gaussians._features_rest[mask3d]
    new_gaussians._scaling = gaussians._scaling[mask3d]
    new_gaussians._rotation = gaussians._rotation[mask3d]
    new_gaussians._opacity = gaussians._opacity[mask3d]

    return new_gaussians


def remove_gaussians_with_mask(gaussians, views):
    gaussians_xyz = gaussians._xyz.detach()
    gaussians_view_counter = torch.zeros(gaussians_xyz.shape[0], dtype=torch.int32, device="cuda")
    with torch.no_grad():
        for _idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            H, W = view.image_height, view.image_width
            K = view.K
            R, T = view.R, view.T

            # Create the World-to-Camera transformation matrix
            W2C = np.zeros((4, 4))
            W2C[:3, :3] = R.transpose()
            W2C[:3, 3] = T
            W2C[3, 3] = 1.0
            W2C = torch.tensor(W2C, dtype=torch.float32, device="cuda")

            # Transform gaussians' xyz coordinates to the camera space
            xyz = torch.cat(
                [gaussians_xyz, torch.ones(gaussians_xyz.size(0), 1, device="cuda")], dim=1
            )
            xyz = torch.matmul(xyz, W2C.T)
            xyz = xyz[:, :3]
            xyz = xyz / xyz[:, 2].unsqueeze(1)  # Normalize by z-coordinate

            # Project to image plane
            uv = torch.matmul(xyz, torch.FloatTensor(K).to("cuda").T)
            uv = uv[:, :2].round().long()  # Convert to integer pixel coordinates

            # Check if (u, v) coordinates are within the image bounds
            alpha_mask = view.alpha_mask.squeeze(
                0
            )  # Assuming mask is a 2D tensor on CUDA with shape [H, W]
            valid_uv = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

            # Filter valid coordinates and check mask values
            for i, (u, v) in enumerate(uv):
                if (
                    valid_uv[i] and alpha_mask[v, u] > 0
                ):  # Mask value > 0 implies it lies within the mask region
                    gaussians_view_counter[i] += 1

        # Remove the gaussians that are visible in a frequency of less than 50% of the views
        VIEW_THRESHOLD = 1.0
        mask3d = gaussians_view_counter >= len(views) * VIEW_THRESHOLD
        print(
            f"Removing {len(mask3d) - mask3d.sum()} gaussians not visible in {VIEW_THRESHOLD * 100}% of the views"
        )
        new_gaussians = copy.deepcopy(gaussians)
        new_gaussians._xyz = gaussians._xyz[mask3d]
        new_gaussians._features_dc = gaussians._features_dc[mask3d]
        new_gaussians._features_rest = gaussians._features_rest[mask3d]
        new_gaussians._scaling = gaussians._scaling[mask3d]
        new_gaussians._rotation = gaussians._rotation[mask3d]
        new_gaussians._opacity = gaussians._opacity[mask3d]

    return new_gaussians
