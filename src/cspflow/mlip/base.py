"""What an MLIP engine is, and the gates every structure passes before one runs.

The interface is small on purpose: a single point, a relaxation, and a batch of
relaxations. Everything interesting is in what `RelaxResult` refuses to
conflate.

**Convergence is computed here, not asked of the optimizer.** ASE 3.27 changed
`Optimizer.converged()` to require a `gradient` argument, so the obvious call
raises `TypeError` -- but the better reason to own the definition is that this
whole project turns on the difference between "the process finished" and "the
answer is usable". A relaxation that stops at the step limit is exactly the MLIP
counterpart of a VASP run that hits `NSW`, which was 61% of `redo-new-ter-mag`
(D027), and it is recorded the same way: it ran, it is not converged, and its
geometry is not a minimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

# Pre-validation bounds. Ported from `prescreen.py`'s `validate_structure`,
# which exists because a generative model can and does emit geometrically
# impossible cells -- and an MLIP will happily return a number for one.
MIN_LATTICE_A = 1.0        # Angstrom
MAX_LATTICE_A = 50.0
MIN_ANGLE_DEG = 10.0
MAX_ANGLE_DEG = 170.0
MIN_INTERATOMIC_A = 0.5


@dataclass
class RelaxResult:
    """One MLIP relaxation or single point.

    `converged` and `ok` are separate, deliberately. `ok` says the engine
    produced a number at all; `converged` says the geometry reached the force
    criterion. A structure that stopped at the step limit is `ok=True,
    converged=False` -- not a failure, and emphatically not a result to put on a
    hull without knowing which it is.
    """

    atoms: Any = None
    energy: float | None = None
    e_per_atom: float | None = None
    converged: bool = False
    n_steps: int = 0
    fmax: float | None = None
    volume_before: float | None = None
    volume_after: float | None = None
    error: str = ""
    engine: str = ""

    # Per-site moments, when the engine predicts them. MatterSim does not and
    # leaves these None -- that is the honest record, not a zero [D4].
    #
    # READ THIS BEFORE USING THEM. CHGNet predicts moment MAGNITUDES, |m|. It
    # carries no sign, so `m_total` is the sum of magnitudes: the SATURATION
    # value, assuming every site aligns. For a ferromagnet that is the moment.
    # For a ferrimagnet it is an upper bound and can be badly high. Never
    # report it as "the" moment, and never conclude an ordering from it [D3].
    magmoms: list[float] | None = None
    m_total: float | None = None

    @property
    def ok(self) -> bool:
        return not self.error and self.energy is not None

    @property
    def volume_drift(self) -> float | None:
        """Fractional cell-volume change. Kept separate from the energy error.

        Stage 4 uses it as its own diagnostic: an MLIP can land the energy and
        still move the cell, and averaging the two into one "accuracy" number
        hides which of them went wrong.
        """
        if not self.volume_before or self.volume_after is None:
            return None
        return (self.volume_after - self.volume_before) / self.volume_before

    @property
    def exit_reason(self) -> str:
        if self.error:
            return self.error
        if not self.converged:
            return "step_limit"
        return ""


@dataclass
class BatchStats:
    total: int = 0
    converged: int = 0
    step_limited: int = 0
    rejected: int = 0
    failed: int = 0
    seconds: float = 0.0
    reasons: dict[str, int] = field(default_factory=dict)

    def note(self, result: RelaxResult) -> None:
        self.total += 1
        if result.error:
            key = "rejected" if result.error.startswith("rejected:") else "failed"
            setattr(self, key, getattr(self, key) + 1)
            self.reasons[result.error] = self.reasons.get(result.error, 0) + 1
        elif result.converged:
            self.converged += 1
        else:
            self.step_limited += 1

    def render(self) -> str:
        lines = [
            f"structures      {self.total}",
            f"  converged     {self.converged}",
            f"  step-limited  {self.step_limited}"
            + ("   <- ran, but the geometry is not a minimum" if self.step_limited else ""),
            f"  rejected      {self.rejected}",
            f"  failed        {self.failed}",
        ]
        if self.seconds:
            lines.append(f"  {self.seconds:.1f}s ({self.seconds / max(1, self.total):.2f}s each)")
        for reason, n in sorted(self.reasons.items(), key=lambda kv: -kv[1])[:5]:
            lines.append(f"    {n:>5}  {reason}")
        return "\n".join(lines)


def validate_structure(atoms, *, max_lattice: float | None = None) -> str:
    """Reasons not to hand this cell to an MLIP.  Empty string means it is fine.

    Every one of these is cheap and every one of them is something a generative
    model actually produces. Checking first matters because an MLIP does not
    refuse a nonsense cell -- it returns an energy for it, and that energy then
    looks exactly like a real one on a hull.

    `max_lattice` raises the length cap for callers whose structures are known
    good.  The default 50 A exists to catch a generated cell with a 200 A axis;
    applied to a Materials Project phase it also rejects real long-period
    stacking polytypes.  Cu27Se20 (mp-684606) and Cu29Se19 (mp-685189) are
    genuinely c = 65 A after Niggli reduction, and were silently dropped from
    the MatterSim reference until this was separable.
    """
    import numpy as np

    cap = MAX_LATTICE_A if max_lattice is None else max_lattice

    if len(atoms) == 0:
        return "rejected: empty cell"

    lengths = atoms.cell.lengths()
    if not np.all(np.isfinite(lengths)):
        return "rejected: non-finite lattice"
    if any(a < MIN_LATTICE_A or a > cap for a in lengths):
        return (f"rejected: lattice parameter outside "
                f"[{MIN_LATTICE_A}, {cap}] A ({lengths.round(2).tolist()})")

    angles = atoms.cell.angles()
    if not np.all(np.isfinite(angles)):
        return "rejected: non-finite cell angles"
    if any(a < MIN_ANGLE_DEG or a > MAX_ANGLE_DEG for a in angles):
        return (f"rejected: cell angle outside [{MIN_ANGLE_DEG}, {MAX_ANGLE_DEG}] deg "
                f"({angles.round(1).tolist()})")

    if atoms.cell.volume <= 0:
        return "rejected: non-positive cell volume"

    if len(atoms) > 1:
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        closest = float(distances.min())
        if closest < MIN_INTERATOMIC_A:
            return f"rejected: atoms {closest:.3f} A apart (minimum {MIN_INTERATOMIC_A})"

    return ""


class MLIP(Protocol):
    """The whole interface the screen stage needs."""

    name: str

    def single_point(self, atoms) -> RelaxResult: ...
    def relax(self, atoms) -> RelaxResult: ...
    def relax_many(self, structures: Sequence[Any]) -> list[RelaxResult]: ...
