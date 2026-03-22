"""PhysTwin core package. Engine imports load on first attribute access (see PEP 562)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import InvPhyTrainerWarp, OptimizerCMA
    from .model import SpringMassSystemWarp

__all__ = ["SpringMassSystemWarp", "InvPhyTrainerWarp", "OptimizerCMA"]


def __getattr__(name: str):
    if name == "SpringMassSystemWarp":
        from .model import SpringMassSystemWarp

        return SpringMassSystemWarp
    if name == "InvPhyTrainerWarp":
        from .engine import InvPhyTrainerWarp

        return InvPhyTrainerWarp
    if name == "OptimizerCMA":
        from .engine import OptimizerCMA

        return OptimizerCMA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
