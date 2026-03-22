#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.gaussian_renderer import GaussianModel, render
from gaussian_splatting.scene import Scene
from gaussian_splatting.utils.general_utils import safe_state
from tqdm import tqdm

try:

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import copy

import numpy as np
import pytorch3d.ops as ops
from kornia import create_meshgrid

from gaussian_splatting.prune_utils import (
    remove_gaussians_with_low_opacity,
    remove_gaussians_with_mask,
)


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    disable_sh=False,
):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    # TODO: temporary debug for demo
    # scene_name = model_path.split('/')[-2]
    # render_path = os.path.join('./output_tmp_for_sydney', scene_name, "renders")
    # gts_path = os.path.join('./output_tmp_for_sydney', scene_name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if disable_sh:
            override_color = gaussians.get_features_dc.squeeze()
            results = render(
                view,
                gaussians,
                pipeline,
                background,
                override_color=override_color,
                use_trained_exp=train_test_exp,
                separate_sh=separate_sh,
            )
        else:
            results = render(
                view,
                gaussians,
                pipeline,
                background,
                use_trained_exp=train_test_exp,
                separate_sh=separate_sh,
            )
        rendering = results["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2 :]
            gt = gt[..., gt.shape[-1] // 2 :]

        torchvision.utils.save_image(
            rendering, os.path.join(render_path, f"{idx:05d}" + ".png")
        )
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}" + ".png"))


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    separate_sh: bool,
    remove_gaussians: bool = False,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # remove gaussians that are outside the mask
        if remove_gaussians:
            gaussians = remove_gaussians_with_mask(gaussians, scene.getTrainCameras())

        # remove gaussians that are low opacity
        gaussians = remove_gaussians_with_low_opacity(gaussians)

        # TODO: quick demo purpose (remove later)
        # # sub-sample the gaussians
        # n_subsample = 1000
        # idx = torch.randperm(gaussians._xyz.size(0))[:n_subsample]
        # gaussians._xyz = gaussians._xyz[idx]
        # gaussians._features_dc = gaussians._features_dc[idx]
        # gaussians._features_rest = gaussians._features_rest[idx]
        # gaussians._scaling = gaussians._scaling[idx]
        # gaussians._rotation = gaussians._rotation[idx]
        # gaussians._opacity = gaussians._opacity[idx]
        # # set the scale of the gaussians
        # scale = 0.01
        # gaussians._scaling = gaussians.scaling_inverse_activation(torch.ones_like(gaussians._scaling) * scale)

        # remove gaussians that are far from the mesh
        # gaussians = remove_gaussians_with_point_mesh_distance(gaussians, scene.mesh_sampled_points, dist_threshold=0.01)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                separate_sh,
                disable_sh=dataset.disable_sh,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                separate_sh,
                disable_sh=dataset.disable_sh,
            )


def get_ray_directions(
    H, W, K, device="cuda", random=False, return_uv=False, flatten=True, anti_aliasing_factor=1.0
):
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    if anti_aliasing_factor > 1.0:
        H = int(H * anti_aliasing_factor)
        W = int(W * anti_aliasing_factor)
        K *= anti_aliasing_factor
        K[2, 2] = 1
    grid = create_meshgrid(H, W, False, device=device)[0]  # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if random:
        directions = torch.stack(
            [
                (u - cx + torch.rand_like(u)) / fx,
                (v - cy + torch.rand_like(v)) / fy,
                torch.ones_like(u),
            ],
            -1,
        )
    else:  # pass by the center
        directions = torch.stack([(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)
    if return_uv:
        return directions, grid
    return directions


def remove_gaussians_with_point_mesh_distance(gaussians, mesh_sampled_points, dist_threshold=0.1):
    """
    Remove gaussians that are far from the mesh

    Args:
        gaussians (GaussianModel): Gaussian model
        mesh_sampled_points (Tensor): Sampled points from the mesh
        dist_threshold (float): Distance threshold (in meters) to remove the gaussians
    """

    gaussians_xyz = gaussians._xyz.detach()
    # dists_knn = ops.knn_points(gaussians_xyz.unsqueeze(0), mesh_sampled_points.unsqueeze(0), K=1, norm=2)
    dists_bq = ops.ball_query(
        gaussians_xyz.unsqueeze(0), mesh_sampled_points.unsqueeze(0), K=1, radius=dist_threshold
    )
    mask3d = (dists_bq[1].squeeze(0) != -1).squeeze(-1)
    print(f"Removing {len(mask3d) - mask3d.sum()} gaussians with distance < {dist_threshold}")

    new_gaussians = copy.deepcopy(gaussians)
    new_gaussians._xyz = gaussians._xyz[mask3d]
    new_gaussians._features_dc = gaussians._features_dc[mask3d]
    new_gaussians._features_rest = gaussians._features_rest[mask3d]
    new_gaussians._scaling = gaussians._scaling[mask3d]
    new_gaussians._rotation = gaussians._rotation[mask3d]
    new_gaussians._opacity = gaussians._opacity[mask3d]

    return new_gaussians


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--remove_gaussians", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        SPARSE_ADAM_AVAILABLE,
        args.remove_gaussians,
    )
