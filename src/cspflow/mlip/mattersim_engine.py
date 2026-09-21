"""MatterSim.

Deliberately does **not** use `mattersim.applications.batch_relax.BatchRelaxer`,
which is what `prescreen.py` uses. Measured on this machine, in both the new
`cspflow` environment and the user's pre-existing `mattersim` one, it is broken
against the installed ASE in two independent places:

    from mattersim.applications.batch_relax import BatchRelaxer
    ImportError: cannot import name 'Filter' from 'ase.constraints'
        # ASE moved Filter (and the cell filters) to ase.filters

    # after shimming that:
    File ".../mattersim/applications/batch_relax.py", line 122, in step_batch
        if opt.converged():
    TypeError: Optimizer.converged() missing 1 required positional argument

The first is a one-line compatibility shim. The second is inside mattersim's own
loop and cannot be fixed from outside without patching its internals, which is
further than a compatibility layer should reach. `BatchRelaxer` in this version
also accepts no step limit at all -- `max_n_steps` is not a parameter, though
`prescreen.py` passes one -- so an unconvergeable structure has nothing to stop
it.

So relaxation runs through ASE's own optimizer against `MatterSimCalculator`,
which is fully supported by the installed ASE, honours `max_steps`, and lets us
own the convergence test. The cost is losing MatterSim's GPU batching; the
benefit is a relaxation that stops when told to and reports honestly whether it
converged. Measured on CPU (login node): 0.8 s for a 2-atom cell, 10.8 s for 54
atoms, ~60-230 ms per optimizer step.

Physical sanity, same run:

    Fe bcc  a = 2.8369 A  (expt 2.87)   E/atom = -8.47789 eV
    Co fcc  a = 3.5160 A  (expt 3.545)  E/atom = -7.06160 eV
    Ni fcc  a = 3.5064 A  (expt 3.524)  E/atom = -5.77280 eV

and a 16-atom and a 54-atom Fe supercell relax to the same E/atom to 5 decimal
places, which is the internal consistency check worth having.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Sequence

from .base import BatchStats, RelaxResult, validate_structure

DEFAULT_MODEL = "MatterSim-v1.0.0-5M.pth"


def apply_ase_compat_shim() -> list[str]:
    """Restore names ASE relocated, for third-party code that still imports them.

    ASE moved `Filter`, `ExpCellFilter`, `UnitCellFilter`, `StrainFilter` and
    `FrechetCellFilter` from `ase.constraints` to `ase.filters`. Several
    packages -- mattersim among them -- still import from the old location.
    Restoring the alias is narrow, reversible and additive: nothing is
    overwritten, only missing names are filled in.

    Returns the names it added, so `csp doctor` can report that a shim is in
    effect rather than leaving it invisible.
    """
    import ase.constraints
    import ase.filters

    added = []
    for name in ("Filter", "ExpCellFilter", "UnitCellFilter", "StrainFilter",
                 "FrechetCellFilter"):
        if not hasattr(ase.constraints, name) and hasattr(ase.filters, name):
            setattr(ase.constraints, name, getattr(ase.filters, name))
            added.append(name)
    return added


class MatterSimEngine:
    """MatterSim as an `MLIP`."""

    name = "mattersim"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        fmax: float = 0.01,
        max_steps: int = 500,
        device: str | None = None,
        optimizer: str = "FIRE",
        relax_cell: bool = True,
        max_lattice: float | None = None,
    ) -> None:
        # `max_lattice` is None for campaign work, where the 50 A default guards
        # against generated nonsense.  The reference store passes a larger value
        # because its structures come from MP and a long axis there is a real
        # long-period polytype, not a defect.  See `_admit`.
        self.max_lattice = max_lattice
        self.model = model
        self.fmax = fmax
        self.max_steps = max_steps
        self.optimizer = optimizer
        self.relax_cell = relax_cell
        self._device = device
        self._calc = None

    # -- lazy setup --------------------------------------------------------

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def calc(self):
        """Loaded once and reused.

        Reloading the checkpoint per structure would dominate the runtime -- the
        model load is seconds, a small relaxation is under one.
        """
        if self._calc is None:
            apply_ase_compat_shim()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from mattersim.forcefield import MatterSimCalculator

                self._calc = MatterSimCalculator(load_path=self.model, device=self.device)
        return self._calc

    # -- work --------------------------------------------------------------

    def single_point(self, atoms) -> RelaxResult:
        atoms, reason = self._admit(atoms)
        if reason:
            return RelaxResult(error=reason, engine=self.name)

        work = atoms.copy()
        work.calc = self.calc
        try:
            energy = float(work.get_potential_energy())
            fmax = _max_force(work.get_forces())
        except Exception as exc:
            return RelaxResult(error=f"failed: {type(exc).__name__}: {exc}", engine=self.name)

        if not _finite(energy):
            return RelaxResult(error=f"failed: non-finite energy ({energy})", engine=self.name)

        return RelaxResult(
            atoms=work, energy=energy, e_per_atom=energy / len(work),
            converged=fmax <= self.fmax, n_steps=0, fmax=fmax,
            volume_before=work.cell.volume, volume_after=work.cell.volume,
            engine=self.name,
        )

    # -- admission ---------------------------------------------------------

    def _admit(self, atoms):
        """Validate, and if only the CELL SETTING is at fault, reduce it and retry.

        Two different things get rejected by the same gate and only one of them
        is a real problem:

        * a nonsense cell from a generative model -- reject it, that is the point
        * a real structure written in a badly skewed setting -- the same atoms,
          the same volume, the same energy, just an unreduced basis

        Materials Project serves plenty of the second kind.  mp-673643
        (Ce17O32) arrives with cell angles of 18/19/5 degrees and axes of
        66/41/10 A; Niggli-reduced it is 8.7/8.7/10.3 A at 105/95/107 degrees --
        identical volume, and it passes.  Before this, four store phases were
        dropped from the MatterSim reference for no physical reason, which is
        what blocked the MatterSim hull in five chemical systems.

        Reduction is applied unconditionally, not only as a rescue.  It is a
        change of lattice BASIS, not of structure: the atoms, the volume and the
        energy are identical, and the reduced basis is the canonical one.  Doing
        it always also keeps a cell like mp-684627 (angles 28/30/14 degrees,
        which passes the length check but is a poor neighbour list) from
        reaching the optimizer in that setting.

        Returns (atoms_to_use, reason).  A non-empty reason means genuinely
        rejected, after reduction was tried.
        """
        try:
            from ase.build import niggli_reduce

            reduced = atoms.copy()
            niggli_reduce(reduced)
        except Exception:                                          # noqa: BLE001
            reduced = None

        if reduced is not None and not validate_structure(
                reduced, max_lattice=self.max_lattice):
            return reduced, ""
        # Reduction did not help (or failed).  Report the ORIGINAL cell's verdict,
        # because that is the cell the caller handed us and the one to debug.
        return atoms, validate_structure(atoms, max_lattice=self.max_lattice)

    def relax(self, atoms) -> RelaxResult:
        atoms, reason = self._admit(atoms)
        if reason:
            return RelaxResult(error=reason, engine=self.name)

        work = atoms.copy()
        work.calc = self.calc
        volume_before = float(work.cell.volume)

        try:
            target = self._filtered(work)
            optimizer = self._optimizer(target)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                optimizer.run(fmax=self.fmax, steps=self.max_steps)
            # Convergence is decided here, from the forces, rather than by
            # asking the optimizer -- see the module docstring.
            fmax = _max_force(target.get_forces())
            energy = float(work.get_potential_energy())
            n_steps = int(optimizer.get_number_of_steps())
        except Exception as exc:
            return RelaxResult(error=f"failed: {type(exc).__name__}: {exc}",
                               engine=self.name, volume_before=volume_before)

        if not _finite(energy) or not _finite(fmax):
            return RelaxResult(
                error=f"failed: non-finite result (E={energy}, fmax={fmax})",
                engine=self.name, volume_before=volume_before,
            )

        return RelaxResult(
            atoms=work, energy=energy, e_per_atom=energy / len(work),
            converged=bool(fmax <= self.fmax), n_steps=n_steps, fmax=fmax,
            volume_before=volume_before, volume_after=float(work.cell.volume),
            engine=self.name,
        )

    def relax_many(self, structures: Sequence[Any]) -> list[RelaxResult]:
        """One at a time, sharing the loaded model.

        Not batched: see the module docstring for why MatterSim's own
        `BatchRelaxer` is unusable against the installed ASE. Sharing the
        calculator is most of the win in any case -- the checkpoint load is the
        expensive part, and it happens once.
        """
        return [self.relax(atoms) for atoms in structures]

    def relax_with_stats(self, structures: Sequence[Any]) -> tuple[list[RelaxResult], BatchStats]:
        stats = BatchStats()
        started = time.monotonic()
        results = []
        for atoms in structures:
            result = self.relax(atoms)
            results.append(result)
            stats.note(result)
        stats.seconds = time.monotonic() - started
        return results, stats

    # -- internals ---------------------------------------------------------

    def _filtered(self, atoms):
        """Relax the cell too, unless told not to.

        `FrechetCellFilter` in preference to `ExpCellFilter`: ASE deprecated the
        latter because its strain parameterisation makes the optimizer take a
        path that is not the steepest descent, which costs steps and can stall.
        """
        if not self.relax_cell:
            return atoms
        from ase.filters import FrechetCellFilter

        return FrechetCellFilter(atoms)

    def _optimizer(self, target):
        from ase import optimize

        cls = getattr(optimize, self.optimizer, None)
        if cls is None:
            known = [n for n in dir(optimize) if n[0].isupper()]
            raise ValueError(f"unknown ASE optimizer {self.optimizer!r}; known: {known}")
        return cls(target, logfile=None)


def _max_force(forces) -> float:
    import numpy as np

    if forces is None or len(forces) == 0:
        return 0.0
    return float(np.sqrt((np.asarray(forces) ** 2).sum(axis=1)).max())


def _finite(value: float | None) -> bool:
    import math

    return value is not None and math.isfinite(value)
