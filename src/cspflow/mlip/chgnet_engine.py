"""CHGNet.

WHY THIS ENGINE EXISTS AT ALL, GIVEN MATTERSIM ALREADY DOES RELAXATION
    MatterSim is a non-magnetic potential: it returns energy, forces and stress
    and knows nothing about moments [D4]. CHGNet was trained on the Materials
    Project's *magnetic* relaxation trajectories and predicts a per-site moment
    alongside the energy. That single extra output is the reason to carry a
    second engine -- it turns "how strong is this magnet" from a DFT-only
    question into something screenable.

    Measured here 2026-09-15, Ce2Fe14B (68 atoms, experimental cell):

        Fe sublattice   128.70 muB   (mean 2.298, range 2.02-2.60)
        Ce              1.87 muB     (mean 0.233)
        B               0.28 muB
        total           130.85 muB  ->  1.667 T
        our DFT         125.06 muB  ->  1.593 T      (+4.6% high)

    2.9 s on one CPU core, against roughly 20 core-HOURS for the VASP number.

THE LIMITATION THAT DECIDES HOW IT MAY BE USED
    CHGNet predicts |m|, the MAGNITUDE. There is no sign. So the sum over sites
    is the moment the structure would have IF every site aligned -- the
    saturation value, an UPPER BOUND, not a prediction of the ground state.

    For Ce2Fe14B, where the Fe sublattices are parallel, the bound is tight and
    the number above is meaningful. For a candidate that orders ferrimagnetically
    it is wrong, and wrong in the dangerous direction: it reads HIGH. A search
    rewarded for large predicted moment will therefore drift preferentially
    towards exactly the structures where this engine is least trustworthy.

    So: the quantity is named `m_total` and documented as saturation; anything
    downstream calls it J_s_max; and no ordering claim may rest on it. FM vs AFM
    stays a DFT comparison [D3].

RARE EARTHS
    CHGNet inherits MP's f-in-valence convention, so its Ce and Gd moments are
    not the moments our frozen-f campaigns report [D1]. Ce here comes out at
    0.23 muB against an experimental ~0.1, which is small enough not to matter
    for Ce2Fe14B -- masking Ce changes 1.667 T to 1.650 T. It is NOT small for a
    Gd or Tb compound. `moments()` returns the per-site array so the caller can
    mask deliberately rather than inheriting a convention by accident.

VERSION
    Pinned to the 0.3.x line on purpose. chgnet 0.4.2 requires torch>=2.4.1;
    this environment runs torch 2.2.1+cu118 because MatterSim's stack
    (torchvision, torchaudio, e3nn, torch_geometric) is built against it.
    Installing 0.4.2 would have pulled torch 2.14 and CUDA 13 and broken
    MatterSim. 0.3.8 requires only torch>=1.11 and installs additively.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

from .base import BatchStats, RelaxResult, validate_structure


class CHGNetEngine:
    """CHGNet as an `MLIP`, plus the moments MatterSim cannot give."""

    name = "chgnet"

    def __init__(
        self,
        model: str = "",
        *,
        fmax: float = 0.05,
        max_steps: int = 500,
        device: str | None = None,
        optimizer: str = "FIRE",
        relax_cell: bool = True,
        max_lattice: float | None = None,
    ) -> None:
        self.model = model            # "" -> CHGNet.load(), the shipped weights
        self.fmax = fmax
        self.max_steps = max_steps
        self.optimizer = optimizer
        self.relax_cell = relax_cell
        self.max_lattice = max_lattice
        self._device = device
        self._net = None
        self._calc = None

    # -- lazy setup --------------------------------------------------------

    @property
    def device(self) -> str:
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def net(self):
        """The raw model, for `predict_structure`. Loaded once."""
        if self._net is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from chgnet.model import CHGNet
                self._net = (CHGNet.from_file(self.model) if self.model
                             else CHGNet.load(verbose=False))
        return self._net

    @property
    def calc(self):
        """ASE calculator, for relaxation. Shares the loaded model."""
        if self._calc is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from chgnet.model.dynamics import CHGNetCalculator
                self._calc = CHGNetCalculator(model=self.net, use_device=self.device)
        return self._calc

    # -- work --------------------------------------------------------------

    def _admit(self, atoms):
        return atoms, validate_structure(atoms, max_lattice=self.max_lattice)

    def moments(self, atoms) -> tuple[list[float] | None, float | None, str]:
        """Per-site |m| and their sum, in Bohr magnetons.

        Returns (magmoms, m_total, error). `m_total` is a SATURATION value --
        the sum of magnitudes, valid as the moment only if every site aligns.
        See the module docstring.
        """
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            import numpy as np
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = self.net.predict_structure(AseAtomsAdaptor.get_structure(atoms))
            m = np.asarray(pred["m"]).flatten().astype(float)
            if m.size != len(atoms) or not np.all(np.isfinite(m)):
                return None, None, f"failed: bad magmom array (size {m.size})"
            return m.tolist(), float(np.abs(m).sum()), ""
        except Exception as exc:
            return None, None, f"failed: {type(exc).__name__}: {exc}"

    def single_point(self, atoms) -> RelaxResult:
        atoms, reason = self._admit(atoms)
        if reason:
            return RelaxResult(error=reason, engine=self.name)
        work = atoms.copy()
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            import numpy as np
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = self.net.predict_structure(AseAtomsAdaptor.get_structure(work))
            # CHGNet's 'e' is eV per ATOM, unlike MatterSim's total energy.
            # Getting this backwards is a silent factor of len(atoms).
            e_per_atom = float(pred["e"])
            energy = e_per_atom * len(work)
            m = np.asarray(pred["m"]).flatten().astype(float)
        except Exception as exc:
            return RelaxResult(error=f"failed: {type(exc).__name__}: {exc}", engine=self.name)
        if not _finite(energy):
            return RelaxResult(error=f"failed: non-finite energy ({energy})", engine=self.name)
        return RelaxResult(
            atoms=work, energy=energy, e_per_atom=e_per_atom, converged=True,
            n_steps=0, volume_before=float(work.cell.volume),
            volume_after=float(work.cell.volume), engine=self.name,
            magmoms=m.tolist() if m.size == len(work) else None,
            m_total=float(abs(m).sum()) if m.size == len(work) else None,
        )

    def relax(self, atoms) -> RelaxResult:
        atoms, reason = self._admit(atoms)
        if reason:
            return RelaxResult(error=reason, engine=self.name)
        work = atoms.copy()
        work.calc = self.calc
        volume_before = float(work.cell.volume)
        try:
            from ase.optimize import FIRE, BFGS
            from ase.filters import FrechetCellFilter
            target = FrechetCellFilter(work) if self.relax_cell else work
            opt_cls = {"FIRE": FIRE, "BFGS": BFGS}.get(self.optimizer, FIRE)
            optimizer = opt_cls(target, logfile=None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                optimizer.run(fmax=self.fmax, steps=self.max_steps)
            # Convergence decided from the forces here, not asked of the
            # optimizer -- same rule as the MatterSim engine.
            fmax = _max_force(target.get_forces())
            energy = float(work.get_potential_energy())
            n_steps = int(optimizer.get_number_of_steps())
        except Exception as exc:
            return RelaxResult(error=f"failed: {type(exc).__name__}: {exc}",
                               engine=self.name, volume_before=volume_before)
        if not _finite(energy) or not _finite(fmax):
            return RelaxResult(error=f"failed: non-finite result (E={energy}, fmax={fmax})",
                               engine=self.name, volume_before=volume_before)
        magmoms, m_total, _ = self.moments(work)
        return RelaxResult(
            atoms=work, energy=energy, e_per_atom=energy / len(work),
            converged=bool(fmax <= self.fmax), n_steps=n_steps, fmax=fmax,
            volume_before=volume_before, volume_after=float(work.cell.volume),
            engine=self.name, magmoms=magmoms, m_total=m_total,
        )

    def relax_many(self, structures: Sequence[Any]) -> list[RelaxResult]:
        return [self.relax(a) for a in structures]

    def relax_with_stats(self, structures: Sequence[Any]) -> tuple[list[RelaxResult], BatchStats]:
        stats = BatchStats()
        out = []
        for atoms in structures:
            r = self.relax(atoms)
            stats.note(r)
            out.append(r)
        return out, stats


def _max_force(forces) -> float:
    import numpy as np
    return float(np.sqrt((np.asarray(forces) ** 2).sum(axis=1).max()))


def _finite(value: float | None) -> bool:
    import math
    return value is not None and math.isfinite(value)
