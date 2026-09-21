"""KPOINTS.

Three schemes, and the reason there are three: a hull compares energies across
compositions and cell sizes, so the k-point sampling has to be *comparable*
across them, not merely fine enough for each.

    reciprocal_density   grid chosen so k-point density in reciprocal space is
                         constant -- a big cell gets a coarse grid, a small one
                         a fine grid, and the two are comparable. This is the
                         default and the one the legacy campaign used.
    kspacing             a target spacing in A^-1; VASP's own KSPACING tag
                         expresses the same idea, and this writes the explicit
                         grid so the file records what was actually used.
    explicit             a literal grid, for when you know what you want.

`gamma: true` throughout. A Gamma-centred grid preserves the crystal's point
symmetry; a Monkhorst-Pack grid with an even subdivision does not, and for
hexagonal cells -- which most RE-TM magnets are -- that silently breaks the
symmetry VASP then tries to use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..recipe import Kpoints


class KpointsError(Exception):
    pass


def canonical_cell(atoms):
    """The same crystal, in a basis the k-point policy can be applied to.

    A lattice has no unique set of basis vectors, so any grid chosen from
    `a, b, c` is chosen from an arbitrary description rather than from the
    crystal. Materials Project ships some structures in near-degenerate
    settings -- mp-1210207 has cell angles 169, 169, 15 degrees, three edges of
    23.94 A enclosing 460 A^3 -- and `reciprocal_density` applied to those edges
    returns a 1x1x1 grid for a metal whose reduced cell needs 5x5x1.

    Measured over the 2,746-phase MP reference recompute: re-describing a
    crystal in a random equivalent basis reproduced the same grid 3.8% of the
    time without this reduction and 100% of the time with it. 324 structures
    were sampled on a starved grid as a result, of which 228 were measurably
    wrong -- energies off by up to 5.1 eV/atom, and one cell that ran away by a
    factor of 677 because the stress tensor at a single k-point is meaningless.

    Niggli reduction is a change of basis, so the crystal, its volume and its
    energy are untouched; only the description changes. Returns a copy.
    """
    from ase.build import niggli_reduce

    reduced = atoms.copy()
    try:
        niggli_reduce(reduced)
    except Exception:
        # A cell too degenerate even to reduce is better left alone than
        # silently mangled; the caller still gets a usable structure.
        return atoms.copy()
    return reduced


@dataclass(frozen=True)
class KpointGrid:
    a: int
    b: int
    c: int
    scheme: str
    gamma: bool = True
    comment: str = ""

    @property
    def total(self) -> int:
        return self.a * self.b * self.c

    def render(self) -> str:
        head = self.comment or f"cspflow {self.scheme}"
        style = "Gamma" if self.gamma else "Monkhorst-Pack"
        return f"{head}\n0\n{style}\n{self.a} {self.b} {self.c}\n"


def grid_for(cell_lengths: Sequence[float], kpoints: Kpoints,
             n_atoms: int = 1) -> KpointGrid:
    """The grid this recipe stage asks for, given the cell."""
    if kpoints.scheme == "explicit":
        value = kpoints.value
        if not (isinstance(value, (list, tuple)) and len(value) == 3):
            raise KpointsError(
                f"kpoints.scheme='explicit' needs a three-element grid, got {value!r}"
            )
        return KpointGrid(*(int(v) for v in value), scheme="explicit",
                          comment="cspflow explicit grid")

    if kpoints.scheme == "kspacing":
        spacing = float(kpoints.value)
        if spacing <= 0:
            raise KpointsError(f"kpoints.value must be > 0 for kspacing, got {spacing}")
        divisions = [max(1, int(math.ceil(2 * math.pi / (length * spacing))))
                     for length in cell_lengths]
        return KpointGrid(*divisions, scheme="kspacing",
                          comment=f"cspflow KSPACING {spacing} A^-1")

    if kpoints.scheme == "reciprocal_density":
        density = float(kpoints.value)
        if density <= 0:
            raise KpointsError(
                f"kpoints.value must be > 0 for reciprocal_density, got {density}")
        # pymatgen's `automatic_density_by_vol`, reproduced rather than imported
        # so the grid cannot move under a pymatgen upgrade:
        #
        #     kppa = kppvol * V_recip * n_atoms
        #     mult = (kppa / n_atoms * a*b*c) ** (1/3)
        #     n_i  = floor(max(mult / l_i, 1))
        #
        # V_recip * V_cell is (2*pi)^3 identically, and n_atoms cancels, so the
        # whole thing collapses to `mult = 2*pi * kppvol^(1/3)` -- independent of
        # both cell size and atom count. That independence is the point: it is
        # what makes the sampling comparable across the different cell sizes a
        # hull has to compare.
        #
        # Verified against the legacy campaign: Gd1Co10Cr2_s020 has
        # a,b,c = 4.6399, 8.1658, 8.2433 A, and both this and its own KPOINTS
        # file give `5 3 3`.
        # FLOOR, which is pymatgen's own behaviour. Validated against 500 of the
        # legacy campaign's own KPOINTS files:
        #
        #     exact                 402/500  (80.4%)
        #     within 1 per axis     500/500  (100%)
        #
        # Every disagreement is a length sitting within ~0.1% of an integer
        # boundary. Switching to round fixes those and breaks more than it fixes
        # (29.8% exact against 80.4%), which is the signature of the POSCAR no
        # longer being the cell its KPOINTS were generated from -- the legacy
        # restart scripts overwrite POSCAR from CONTCAR and do not regenerate
        # KPOINTS. cspflow writes both together and keeps the originals.
        #
        # That is not the same as the question being unreachable, and an earlier
        # version of this comment claimed it was. A RESUME rewrites POSCAR from
        # the previous attempt's CONTCAR and did regenerate KPOINTS from it, so
        # a relaxation whose cell drifted across one of these boundaries changed
        # its own sampling halfway through. See `carry_grid` below, and D149.
        mult = 2 * math.pi * (density ** (1.0 / 3.0))
        divisions = [max(1, int(math.floor(mult / max(float(length), 1e-9))))
                     for length in cell_lengths]
        return KpointGrid(*divisions, scheme="reciprocal_density",
                          comment=f"cspflow reciprocal_density {density:g}")

    raise KpointsError(
        f"unknown kpoints.scheme {kpoints.scheme!r}; known: reciprocal_density, "
        f"kspacing, explicit"
    )




def carry_grid(previous: KpointGrid,
               source_lengths: Sequence[float],
               canonical_lengths: Sequence[float],
               tol: float = 1e-6) -> KpointGrid | None:
    """The previous attempt's grid, expressed in this attempt's basis.

    A relaxation that resumes from its own CONTCAR must keep the grid it was
    already running, not re-derive one from the cell it has reached. The grid
    is a floor() of `mult / length`, so a cell that drifts across an integer
    boundary during the relaxation changes its own sampling mid-flight, and the
    resumed run is then minimising a different energy surface from the one its
    starting geometry was nearly converged on.

    Measured on structure 2498 of RE-magnets-CHGNet (Fe56Nd8B4, Nd2Fe14B type):
    99 ionic steps relaxed c from 12.5437 to 12.6264 A, crossing the 2 -> 1
    boundary at mult/2 = 12.5664 A by 0.05%. The resume halved the c* sampling,
    the geometry stopped being a minimum, forces climbed from 0.069 to 0.244
    eV/A over 185 further steps, and the run was killed at walltime having
    consumed 560 core-hours. Across the campaign, 31 of 120 retried runs had
    their grid change this way, and they took a median 87 ionic steps against
    44 for the runs whose grid was kept.

    Niggli reduction runs again on the resumed cell, and it may permute the
    axes -- so the grid is mapped through that permutation rather than copied
    positionally. A permutation preserves lengths exactly, so the match is
    strict; anything else means the basis genuinely changed and the old grid no
    longer describes the same sampling. Returns None in that case, and the
    caller falls back to deriving the grid, which is what it did before.
    """
    source = [float(x) for x in source_lengths]
    canonical = [float(x) for x in canonical_lengths]
    if len(source) != 3 or len(canonical) != 3:
        return None

    divisions = (previous.a, previous.b, previous.c)
    mapped: list[int] = []
    taken: set[int] = set()
    for length in canonical:
        match = None
        for j, other in enumerate(source):
            if j in taken:
                continue
            if abs(length - other) <= tol * max(abs(length), abs(other), 1.0):
                match = j
                break
        if match is None:
            return None
        taken.add(match)
        mapped.append(int(divisions[match]))

    return KpointGrid(*mapped, scheme=previous.scheme, gamma=previous.gamma,
                      comment=previous.comment or f"cspflow {previous.scheme}")
