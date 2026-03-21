"""Discover PhysTwin cases under ``base_path`` that have replay-style artifacts."""

from __future__ import annotations

import glob
import os

# Repo root (parent of `benchmarks/`)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def discover_cases(base_path: str) -> list[str]:
    """Return case names with ``final_data.pkl`` and ``experiments/<case>/train/best_*.pth``."""
    if not os.path.isdir(base_path):
        return []
    names: list[str] = []
    for name in sorted(os.listdir(base_path)):
        case_dir = os.path.join(base_path, name)
        if not os.path.isdir(case_dir):
            continue
        if not os.path.isfile(os.path.join(case_dir, "final_data.pkl")):
            continue
        train_pat = os.path.join(_REPO_ROOT, "experiments", name, "train", "best_*.pth")
        if not glob.glob(train_pat):
            continue
        names.append(name)
    return names
