#!/usr/bin/env python3
"""Print empirical distribution of masses and spring stiffness for a PhysTwin case.

Uses the same config/checkpoint requirements as ``replay_recorded`` (optimal params + best checkpoint).

Usage:
    python -m rerun_viz.inspect_case_stats --case_name double_lift_cloth_3
"""

from __future__ import annotations

import glob

import numpy as np
import torch

from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger
from rerun_viz.case_setup import load_case_yaml_and_optimal


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--case_name", required=True)
    p.add_argument("--base_path", default="./data/different_types")
    args = p.parse_args()

    load_case_yaml_and_optimal(args.case_name)

    base_dir = f"experiments/{args.case_name}"
    logger.set_log_file(path=base_dir, name="inspect_stats_log")

    trainer = InvPhyTrainerWarp(
        data_path=f"{args.base_path}/{args.case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    candidates = glob.glob(f"{base_dir}/train/best_*.pth")
    assert len(candidates) > 0, (
        f"No best_*.pth checkpoint found under {base_dir}/train; did you run training?"
    )
    ckpt = torch.load(candidates[0], map_location="cpu")

    import warp as wp

    masses = wp.to_torch(trainer.simulator.wp_masses).detach().cpu().numpy()
    stiffness = ckpt["spring_Y"].detach().cpu().numpy().astype(np.float64)
    stiffness_finite = stiffness[np.isfinite(stiffness)]

    print("\n" + "=" * 55)
    print(f"  CASE: {args.case_name}")
    print("=" * 55)
    print("\n  MASSES (per vertex)")
    print(f"    count:  {len(masses)}")
    print(f"    min:    {masses.min():.6f}")
    print(f"    max:    {masses.max():.6f}")
    print(f"    mean:   {masses.mean():.6f}")
    print(f"    std:    {masses.std():.6f}")
    print(f"    unique: {len(np.unique(masses))}")
    if len(np.unique(masses)) <= 5:
        print(f"    values: {np.unique(masses)}")

    print("\n  SPRING STIFFNESS (linear, from checkpoint)")
    print(f"    count:     {len(stiffness)}")
    n_inf = np.sum(~np.isfinite(stiffness))
    if n_inf > 0:
        print(f"    non-finite: {n_inf} (excluded from stats)")
    if len(stiffness_finite) > 0:
        print(f"    min:       {stiffness_finite.min():.4f}")
        print(f"    max:       {stiffness_finite.max():.4f}")
        print(f"    mean:      {stiffness_finite.mean():.4f}")
        print(f"    std:       {stiffness_finite.std():.4f}")
        print(f"    p2:        {np.percentile(stiffness_finite, 2):.4f}")
        print(f"    p50:       {np.percentile(stiffness_finite, 50):.4f}")
        print(f"    p98:       {np.percentile(stiffness_finite, 98):.4f}")
    else:
        print("    (all inf/nan)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
