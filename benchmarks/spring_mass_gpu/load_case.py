"""Load ``InvPhyTrainerWarp`` the same way as ``rerun_viz.replay_core``."""

from __future__ import annotations

import os
import sys

# Project root on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qqtt import InvPhyTrainerWarp
from rerun_viz.replay_core import load_config_and_camera, load_trainer_and_model, set_all_seeds


def load_trainer_for_case(base_path: str, case_name: str, *, seed: int = 42) -> InvPhyTrainerWarp:
    """Configure ``cfg`` + camera and load checkpoint like replay."""
    set_all_seeds(seed)
    load_config_and_camera(base_path, case_name)
    return load_trainer_and_model(base_path, case_name)


def load_trainer_same_config(
    base_path: str, case_name: str, *, seed: int = 42
) -> InvPhyTrainerWarp:
    """Assume ``load_config_and_camera`` was already called for ``case_name``."""
    set_all_seeds(seed)
    return load_trainer_and_model(base_path, case_name)
