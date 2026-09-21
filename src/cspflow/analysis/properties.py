"""Everything one finished relaxation is worth recording.

The rule this module exists to enforce is that a derived number is never
returned under a computed number's name.  Three pairs are easy to confuse and
all three are kept apart:

*   `m_dft_raw` (cell magnetisation) against `m_s_reconstructed` (a Hund's-rule
    model layered on the transition-metal sublattice).
*   the cell magnetisation against the sum over PAW spheres, which differ by a
    few percent because the spheres do not tile the cell.
*   the *relaxed* geometry against the input one -- a job that stopped at the
    ionic step limit has a CONTCAR, and it is not a relaxed structure.

Spacegroup is symmetry-analysed from the CONTCAR at a stated tolerance, never
inherited from whatever produced the input.  A generated structure is P1 by
construction and its relaxed form usually is not; carrying the input's symmetry
forward would report every candidate as triclinic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..chem import RARE_EARTHS, canonical_formula
from ..dft.vasp.parse import read_job_directory
from .hund import counts_of, reconstruct
from .moments import MomentReport, read_site_moments

# The symmetry tolerance properties are reported at. Loose enough that a relaxed
# cell is not shattered into P1 by numerical noise, tight enough not to invent
# symmetry that is not there. Reported alongside the number, because a
# spacegroup without its tolerance is not reproducible.
SPACEGROUP_SYMPREC = 0.1


class PropertyError(Exception):
    pass


@dataclass
class StructureProperties:
    """One relaxed structure, as numbers."""

    structure_id: int | None = None
    formula: str = ""
    z: int = 1
    n_atoms: int = 0

    energy: float | None = None            # E0, eV, total
    e_per_atom: float | None = None
    converged: bool = False

    volume: float | None = None            # A^3, relaxed cell
    volume_per_atom: float | None = None
    spacegroup_number: int | None = None
    spacegroup_symbol: str = ""
    symprec: float = SPACEGROUP_SYMPREC

    m_dft_raw: float | None = None         # cell magnetisation, mu_B
    m_spheres: float | None = None         # sum over PAW spheres
    m_s_reconstructed: float | None = None  # Hund's-rule model
    m_per_formula_unit: float | None = None
    m_per_volume: float | None = None      # mu_B / A^3
    sublattice: dict[str, float] = field(default_factory=dict)
    # Every ion's projected moment, by channel.  Kept because it used to be
    # parsed and thrown away: `read_site_moments` builds one `SiteMoment` per
    # ion and only the two sublattice sums survived into the database, so the
    # per-site table -- the thing that says WHICH Fe carries the magnet --
    # existed nowhere afterwards and could only be recovered by going back to
    # the OUTCAR on scratch.
    site_moments: list[dict[str, Any]] = field(default_factory=list)
    f_treatment: str = "frozen"

    warnings: list[str] = field(default_factory=list)

    def as_data(self) -> dict[str, Any]:
        """The structured part, for ASE's `data` blob rather than its key-values.

        `key_value_pairs` holds only str/int/float/bool, so a 68-row table
        cannot go there.  `data` round-trips JSON and is not queryable, which
        is the right trade: nothing selects on an individual ion's moment, and
        everything that wants to read one wants all of them at once.
        """
        out: dict[str, Any] = {}
        if self.site_moments:
            out["site_moments"] = self.site_moments
        if self.sublattice:
            out["sublattice"] = self.sublattice
        if self.m_spheres is not None:
            out["m_spheres"] = self.m_spheres
        return out

    def as_kv(self) -> dict[str, Any]:
        """The subset that goes into the database as key-value pairs.

        `None` is dropped rather than stored as 0: a property that could not be
        computed and a property that is zero are different facts, and ASE has no
        null.
        """
        out = {
            "volume": self.volume, "volume_per_atom": self.volume_per_atom,
            "spacegroup": self.spacegroup_number,
            "m_dft_raw": self.m_dft_raw,
            "m_s_reconstructed": self.m_s_reconstructed,
            "m_per_formula_unit": self.m_per_formula_unit,
            "m_per_volume": self.m_per_volume,
            "f_treatment": self.f_treatment,
        }
        return {k: v for k, v in out.items() if v is not None}


def symbols_of(atoms) -> list[str]:
    return list(atoms.get_chemical_symbols())


def spacegroup_of(atoms, symprec: float = SPACEGROUP_SYMPREC) -> tuple[int | None, str]:
    """(number, symbol) at `symprec`, or (None, reason) if it cannot be found."""
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:                                      # pragma: no cover
        return None, "pymatgen not available"
    try:
        analyzer = SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(atoms),
                                      symprec=symprec)
        return int(analyzer.get_space_group_number()), str(analyzer.get_space_group_symbol())
    except Exception as exc:                                 # spglib is not gentle
        return None, f"symmetry analysis failed: {type(exc).__name__}: {exc}"


def formula_units(symbols: list[str]) -> int:
    """How many formula units the cell contains: the gcd of its element counts.

    Derived from the structure rather than looked up, because the alternative
    was a `z` key on the structure row that nothing ever wrote -- so
    `m_per_formula_unit` divided by 1 and was the cell magnetisation under
    another name. For `Gd2Cr4Co20` this returns 2, and the per-formula-unit
    moment is half the cell value.
    """
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return math.gcd(*counts.values()) if counts else 1


def extract(job_dir: Path, *, structure_id: int | None = None, z: int | None = None,
            f_treatment: str = "frozen") -> StructureProperties:
    """Read one relaxation directory into a property record.

    Uses the CONTCAR, not the POSCAR: the properties describe the relaxed
    geometry.  When the job stopped at the ionic step limit the CONTCAR is the
    last step rather than a minimum, and `converged` says so -- the numbers are
    still extracted, because a table of near-misses is useful, but nothing here
    silently promotes them.
    """
    import ase.io

    job_dir = Path(job_dir)
    outcome = read_job_directory(job_dir)
    props = StructureProperties(structure_id=structure_id, z=z or 1,
                                f_treatment=f_treatment,
                                energy=outcome.energy, e_per_atom=outcome.e_per_atom,
                                converged=outcome.converged)
    if not outcome.converged:
        props.warnings.append(
            "not converged: the geometry is the last ionic step, not a minimum")

    contcar = job_dir / "CONTCAR"
    if not contcar.is_file() or contcar.stat().st_size == 0:
        props.warnings.append(f"no usable CONTCAR in {job_dir}")
        return props

    atoms = ase.io.read(str(contcar), format="vasp")
    symbols = symbols_of(atoms)
    # Z from the cell itself unless the caller insists otherwise.
    if z is None:
        props.z = formula_units(symbols)
    props.n_atoms = len(symbols)
    props.formula = canonical_formula(counts_of(symbols))
    props.volume = float(atoms.get_volume())
    props.volume_per_atom = props.volume / props.n_atoms

    number, symbol = spacegroup_of(atoms, SPACEGROUP_SYMPREC)
    props.spacegroup_number = number
    props.spacegroup_symbol = symbol if number is not None else ""
    if number is None:
        props.warnings.append(symbol)

    moments = read_site_moments(job_dir / "OUTCAR", symbols)
    _apply_moments(props, moments, symbols)
    return props


def _apply_moments(props: StructureProperties, moments: MomentReport,
                   symbols: list[str]) -> None:
    props.m_dft_raw = moments.m_cell
    props.m_spheres = moments.m_spheres
    if moments.note:
        props.warnings.append(moments.note)

    if moments.m_cell is not None:
        props.m_per_formula_unit = moments.per_formula_unit(props.z)
        if props.volume:
            props.m_per_volume = moments.per_volume(props.volume)

    if not moments.sites:
        props.warnings.append(
            "no sublattice split and no m_s_reconstructed: LORBIT=11 is needed "
            "for the site-projected moments this is built from")
        return

    props.sublattice = {"rare_earth": moments.rare_earth,
                        "transition_metal": moments.transition_metal}
    props.site_moments = [
        {"i": site.index, "el": site.element, "tot": round(site.total, 4),
         **{k: round(v, 4) for k, v in site.channels.items()}}
        for site in moments.sites
    ]
    rare_earth_counts = counts_of(symbols, RARE_EARTHS)
    result = reconstruct(moments.transition_metal, rare_earth_counts,
                         f_treatment=props.f_treatment)
    props.m_s_reconstructed = result.m_s
    props.warnings.extend(result.warnings)
