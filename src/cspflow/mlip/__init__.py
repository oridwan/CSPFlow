"""MLIP engines."""

from __future__ import annotations

from .base import MLIP, BatchStats, RelaxResult, validate_structure
from .mattersim_engine import MatterSimEngine, apply_ase_compat_shim
from .chgnet_engine import CHGNetEngine

__all__ = ["MLIP", "BatchStats", "CHGNetEngine", "MatterSimEngine", "RelaxResult",
           "apply_ase_compat_shim", "for_config"]


def for_config(screen) -> MLIP:
    """The engine a campaign's `screen:` block asks for."""
    if screen.mlip == "mattersim":
        return MatterSimEngine(
            model=screen.mattersim.model,
            fmax=screen.mattersim.fmax,
            max_steps=screen.mattersim.max_steps,
        )
    if screen.mlip == "chgnet":
        return CHGNetEngine(
            model=screen.chgnet.model,
            fmax=screen.chgnet.fmax,
            max_steps=screen.chgnet.max_steps,
        )
    raise NotImplementedError(
        f"screen.mlip={screen.mlip!r} is accepted by the schema but has no engine yet. "
        f"Implemented: mattersim, chgnet."
    )
