"""Shared PhysTwin case configuration (yaml + optimal params + optional camera metadata).

Used by replay and diagnostic scripts so behavior and error messages stay consistent.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np

from qqtt.utils import cfg


def config_yaml_for_case(case_name: str) -> str:
    """Return path to the yaml config used for this case (cloth vs real)."""
    if "cloth" in case_name or "package" in case_name:
        return "configs/cloth.yaml"
    return "configs/real.yaml"


def optimal_params_path(case_name: str) -> str:
    """Path to first-stage optimized parameters for ``case_name``."""
    return f"experiments_optimization/{case_name}/optimal_params.pkl"


def load_case_yaml_and_optimal(case_name: str) -> None:
    """Load yaml into ``cfg`` and apply ``optimal_params.pkl`` (same as replay)."""
    cfg.load_from_yaml(config_yaml_for_case(case_name))
    path = optimal_params_path(case_name)
    assert os.path.exists(path), f"{case_name}: Optimal parameters not found: {path}"
    with open(path, "rb") as f:
        cfg.set_optimal_params(pickle.load(f))


def load_camera_and_intrinsics(base_path: str, case_name: str) -> None:
    """Load camera calibrate + metadata into ``cfg`` (replay / interactive scripts)."""
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
