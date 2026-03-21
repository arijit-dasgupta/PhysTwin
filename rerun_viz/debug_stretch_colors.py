#!/usr/bin/env python3
"""Debug script: run a few frames of double_lift_cloth_3 and inspect stretch color pipeline."""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import warp as wp

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger
from rerun_viz.spring_mass_logging import (
    build_spring_strips,
    compute_spring_colors_from_stretch,
    compute_stretch_ratios,
    compute_global_color_ranges,
    _normalize_percentile,
    _value_to_color_stretch,
)


def load_case(case_name: str = "double_lift_cloth_3"):
    cfg.load_from_yaml("configs/cloth.yaml")
    base_path = "./data/different_types"
    import pickle
    with open(f"experiments_optimization/{case_name}/optimal_params.pkl", "rb") as f:
        cfg.set_optimal_params(pickle.load(f))
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c) for c in c2ws])
    with open(f"{base_path}/{case_name}/metadata.json") as f:
        import json
        d = json.load(f)
    cfg.intrinsics = np.array(d["intrinsics"])
    cfg.WH = d["WH"]

    import glob
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=f"experiments/{case_name}",
        pure_inference_mode=True,
    )
    ckpt = torch.load(glob.glob(f"experiments/{case_name}/train/best_*.pth")[0], map_location=cfg.device)
    trainer.simulator.set_spring_Y(torch.log(ckpt["spring_Y"]).detach().clone())
    trainer.simulator.set_collide(ckpt["collide_elas"].detach().clone(), ckpt["collide_fric"].detach().clone())
    trainer.simulator.set_collide_object(
        ckpt["collide_object_elas"].detach().clone(),
        ckpt["collide_object_fric"].detach().clone(),
    )
    trainer.simulator.set_init_state(
        trainer.simulator.wp_init_vertices,
        trainer.simulator.wp_init_velocities,
        pure_inference=True,
    )
    return trainer


def main():
    print("Loading double_lift_cloth_3...")
    trainer = load_case("double_lift_cloth_3")
    sim = trainer.simulator
    controller_points = sim.controller_points

    springs = wp.to_torch(sim.wp_springs, requires_grad=False).detach().cpu().numpy()
    rest_lengths = wp.to_torch(sim.wp_rest_lengths, requires_grad=False).detach().cpu().numpy()
    num_obj = sim.n_vertices  # object points

    print(f"n_vertices={sim.n_vertices}, n_springs={sim.n_springs}")
    print(f"rest_lengths: min={rest_lengths.min():.6f}, max={rest_lengths.max():.6f}, nan={np.isnan(rest_lengths).sum()}")
    print()

    # Object-object filter
    keep = [i for i in range(springs.shape[0]) if int(springs[i, 0]) < num_obj and int(springs[i, 1]) < num_obj]
    print(f"Object-object springs: {len(keep)} / {springs.shape[0]}")

    # Run 2 frames
    for frame_idx in [1, 2]:
        print(f"\n--- Frame {frame_idx} ---")
        sim.set_controller_target(frame_idx, pure_inference=True)
        if sim.object_collision_flag:
            sim.update_collision_graph()
        wp.capture_launch(sim.forward_graph)

        obj_pos = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).detach().cpu().numpy()
        ctrl_pos = controller_points[frame_idx].detach().cpu().numpy()

        print(f"obj_pos: shape={obj_pos.shape}, nan={np.isnan(obj_pos).sum()}, finite={np.isfinite(obj_pos).all()}")
        print(f"obj_pos range: x=[{obj_pos[:, 0].min():.4f},{obj_pos[:, 0].max():.4f}], y=[{obj_pos[:, 1].min():.4f},{obj_pos[:, 1].max():.4f}], z=[{obj_pos[:, 2].min():.4f},{obj_pos[:, 2].max():.4f}]")

        springs_sub = springs[keep]
        rest_sub = rest_lengths[keep]
        strips = build_spring_strips(obj_pos, ctrl_pos, springs_sub, num_obj)

        ratios = np.zeros(len(keep), dtype=np.float32)
        for j, i in enumerate(keep):
            rest = float(rest_lengths[i])
            if rest > 1e-8:
                current_len = float(np.linalg.norm(strips[j, 1] - strips[j, 0]))
                ratios[j] = current_len / rest
            else:
                ratios[j] = 1.0

        print(f"ratios: min={ratios.min():.6f}, max={ratios.max():.6f}, mean={ratios.mean():.6f}, nan={np.isnan(ratios).sum()}")

        # Per-frame normalization (no global)
        x = _normalize_percentile(ratios, v_low=None, v_high=None)
        print(f"x (normalized): min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, nan={np.isnan(x).sum()}")

        colors = _value_to_color_stretch(x)
        print(f"colors: shape={colors.shape}, dtype={colors.dtype}")
        print(f"colors sample [0] (RGBA): {colors[0]} (alpha must be 255, not 1!)")
        print(f"colors min/max per channel: r=[{colors[:, 0].min()},{colors[:, 0].max()}], g=[{colors[:, 1].min()},{colors[:, 1].max()}], b=[{colors[:, 2].min()},{colors[:, 2].max()}]")
        black_count = np.sum(np.all(colors[:, :3] < 10, axis=1))
        print(f"Springs with near-black color (r,g,b<10): {black_count} / {len(colors)}")

        # Test with global range (like pre-pass)
        gr = compute_global_color_ranges(
            np.exp(wp.to_torch(sim.wp_spring_Y, requires_grad=False).detach().cpu().numpy()),
            np.concatenate([ratios]),  # single frame for now
            wp.to_torch(sim.wp_masses, requires_grad=False).detach().cpu().numpy(),
        )
        print(f"Global stretch range: {gr.stretch}")
        if gr.stretch:
            x_glob = _normalize_percentile(ratios, v_low=gr.stretch[0], v_high=gr.stretch[1])
            colors_glob = _value_to_color_stretch(x_glob)
            print(f"With global range - colors sample [0]: {colors_glob[0]}")
            print(f"With global - black count: {np.sum(np.all(colors_glob[:, :3] < 10, axis=1))}")

        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True)

    print("\n--- Testing actual compute_spring_colors_from_stretch ---")
    # Simulate what log_springs_by_stretch does
    ratios_test = ratios  # from last frame
    stretch_range = gr.stretch if gr else None
    v_low, v_high = stretch_range if stretch_range else (None, None)
    colors_final = compute_spring_colors_from_stretch(ratios_test, v_low=v_low, v_high=v_high)
    print(f"Final colors sample: {colors_final[0]}")
    print(f"Final black count: {np.sum(np.all(colors_final[:, :3] < 10, axis=1))}")

    # Check Rerun format - how does stiffness pass colors?
    print("\n--- Checking Rerun LineStrips3D format ---")
    print("Our colors shape:", colors_final.shape)
    print("Our colors dtype:", colors_final.dtype)
    print("First 3 strips colors:\n", colors_final[:3])


if __name__ == "__main__":
    main()
