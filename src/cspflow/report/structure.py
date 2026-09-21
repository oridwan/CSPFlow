"""What one finished structure looks like, and what every ion in it is doing.

WHAT THIS MODULE DOES
    Assembles the per-structure detail the report draws a card from:

        * the geometry, as cell vectors plus fractional coordinates
        * every ion's projected moment, by s/p/d/f channel
        * the per-element sphere sums, with the spread across inequivalent sites
        * the magnetisation in four units

WHICH GEOMETRY, AND WHY IT IS NOT THE ONE IN THE DATABASE
    `campaign.db` holds the **MLIP-relaxed** cell.  `screen_stage` writes it
    there deliberately, so the DFT relaxation starts from a good guess rather
    than from the seed as supplied.  It is not the answer: the answer is the
    CONTCAR the VASP relaxation ended on, and it is on scratch under the row's
    `dft_dir` key.

    So this module reads `dft_dir` first and falls back to the database only
    when that directory is gone -- and `geometry_source` says which one you are
    looking at, every time, because a picture of the wrong cell is
    indistinguishable from a picture of the right one.

WHY THE MOMENTS ARE RE-READ RATHER THAN LOOKED UP
    `analysis/moments.py` already parses the full `LORBIT = 11` table into one
    `SiteMoment` per ion.  `analysis/properties.py` then keeps two sublattice
    sums out of it and discards the rest, so the per-ion numbers exist nowhere
    in the database.  Re-reading the OUTCAR is the only way to get them without
    re-running the campaign.  It is cheap: `tail_text` reads the last 400 kB,
    and the report does this for the detail cards only, not for all 2,000 rows.

INPUTS   a `Store`, and a structure id
OUTPUTS  a `StructureDetail`; `payload()` gives the JSON the page embeds
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..db.store import Store
from .magnetics import MagneticSummary, SiteRow, by_element, site_rows, summarise

# Which key on the row points at the finished VASP directory.
DIR_KEY = "dft_dir"

# Jmol's element colours and covalent radii, for the drawing.  Only the elements
# a report actually contains are emitted, so the table in the page stays small.
_FALLBACK_COLOUR = "#ff1493"
_FALLBACK_RADIUS = 1.5


@dataclass
class StructureDetail:
    """One structure, as the report card needs it."""

    structure_id: int
    formula: str
    n_atoms: int = 0
    cell: list[list[float]] = field(default_factory=list)      # 3x3, angstrom
    frac: list[list[float]] = field(default_factory=list)      # n x 3
    symbols: list[str] = field(default_factory=list)
    lattice: dict[str, float] = field(default_factory=dict)    # a b c alpha beta gamma
    volume: float | None = None
    spacegroup: str = ""
    geometry_source: str = ""        # 'dft-contcar' | 'database-mlip' | ''
    source_path: str = ""            # the seed this structure entered as
    sites: list[SiteRow] = field(default_factory=list)
    elements: list[dict[str, Any]] = field(default_factory=list)
    m_cell: float | None = None      # cell magnetisation, mu_B
    m_spheres: float | None = None   # sphere sum, mu_B
    magnetics: MagneticSummary | None = None
    m_s_reconstructed: float | None = None
    f_treatment: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def sphere_deficit(self) -> float | None:
        """Cell minus spheres: the moment the PAW projection does not see."""
        if self.m_cell is None or self.m_spheres is None:
            return None
        return self.m_cell - self.m_spheres

    def payload(self) -> dict[str, Any]:
        """The compact JSON the in-page viewer reads.

        Fractional coordinates rather than cartesian: the viewer draws periodic
        images of the cell edge atoms, which needs the fractional form anyway,
        and it is the shorter of the two to serialise.
        """
        return {
            "id": self.structure_id,
            "formula": self.formula,
            "cell": [[round(v, 6) for v in row] for row in self.cell],
            "symbols": self.symbols,
            "frac": [[round(v, 6) for v in row] for row in self.frac],
            "moments": [round(s.total, 3) for s in self.sites] if self.sites else [],
        }


def element_table(symbols: set[str]) -> dict[str, dict[str, Any]]:
    """Colour and covalent radius for each element, from ASE's own tables."""
    try:
        from ase.data import atomic_numbers, covalent_radii
        from ase.data.colors import jmol_colors
    except ImportError:                                          # pragma: no cover
        return {s: {"c": _FALLBACK_COLOUR, "r": _FALLBACK_RADIUS} for s in symbols}

    out: dict[str, dict[str, Any]] = {}
    for symbol in sorted(symbols):
        number = atomic_numbers.get(symbol)
        if number is None:
            out[symbol] = {"c": _FALLBACK_COLOUR, "r": _FALLBACK_RADIUS}
            continue
        r, g, b = (int(round(255 * v)) for v in jmol_colors[number])
        out[symbol] = {"c": f"#{r:02x}{g:02x}{b:02x}",
                       "r": float(covalent_radii[number])}
    return out


def _lattice_parameters(cell: list[list[float]]) -> dict[str, float]:
    def norm(v):
        return math.sqrt(sum(x * x for x in v))

    def angle(u, v):
        d = sum(x * y for x, y in zip(u, v)) / (norm(u) * norm(v))
        return math.degrees(math.acos(max(-1.0, min(1.0, d))))

    a, b, c = cell
    return {"a": norm(a), "b": norm(b), "c": norm(c),
            "alpha": angle(b, c), "beta": angle(a, c), "gamma": angle(a, b)}


def detail(store: Store, sid: int, *, read_outcar: bool = True) -> StructureDetail:
    """Everything the card for structure `sid` shows.

    Never raises for a missing file: a structure whose scratch directory has
    been cleaned still gets a card, drawn from the database geometry and saying
    so.  Losing the whole report because one run was tidied away would be the
    worse failure.
    """
    import ase.io

    row = store.get_structure(sid)
    kv = row.key_value_pairs
    out = StructureDetail(
        structure_id=sid,
        formula=str(kv.get("reduced_formula") or row.formula),
        spacegroup=(f"{kv.get('spacegroup_symbol', '')} "
                    f"({int(kv['spacegroup'])})").strip()
        if kv.get("spacegroup") is not None else "",
        m_s_reconstructed=kv.get("m_s_reconstructed"),
        f_treatment=str(kv.get("f_treatment") or ""),
        source_path=str(kv.get("source_path") or ""),
    )

    directory = kv.get(DIR_KEY)
    atoms = None
    if directory and Path(directory).is_dir():
        contcar = Path(directory) / "CONTCAR"
        if contcar.is_file() and contcar.stat().st_size:
            try:
                atoms = ase.io.read(str(contcar), format="vasp")
                out.geometry_source = "dft-contcar"
            except Exception as exc:                             # noqa: BLE001
                out.notes.append(f"CONTCAR unreadable ({type(exc).__name__}); "
                                 f"showing the database geometry instead")
    if atoms is None:
        atoms = row.toatoms()
        out.geometry_source = out.geometry_source or "database-mlip"
        if not out.notes and directory:
            out.notes.append(f"no CONTCAR under {directory}; the cell shown is "
                             f"the MLIP-relaxed one from the database, not the "
                             f"DFT-relaxed one")

    out.symbols = list(atoms.get_chemical_symbols())
    out.n_atoms = len(out.symbols)
    out.cell = [[float(x) for x in vector] for vector in atoms.get_cell()]
    out.frac = [[float(x) for x in p] for p in atoms.get_scaled_positions(wrap=True)]
    out.volume = float(atoms.get_volume())
    out.lattice = _lattice_parameters(out.cell)

    if read_outcar and directory and Path(directory, "OUTCAR").is_file():
        out = _apply_outcar(out, Path(directory) / "OUTCAR")
    if not out.sites:
        # The OUTCAR is gone, or was never readable. `analyze` stores the same
        # table in the row's `data` blob, so a report written after scratch has
        # been cleaned still has per-ion moments. Older campaigns analysed
        # before that was added have no blob and simply show no site table.
        _apply_stored(out, row)

    # The cell magnetisation from the database is the authority when the OUTCAR
    # is gone: `analyze` read it from the same file at the time.
    if out.m_cell is None and kv.get("m_dft_raw") is not None:
        out.m_cell = float(kv["m_dft_raw"])
    out.magnetics = summarise(out.m_cell, out.volume, source="m_dft_raw",
                              z=_formula_units(out.symbols))
    return out


def _apply_stored(out: StructureDetail, row) -> None:
    """Fall back to the per-site table `analyze` wrote into the row's blob."""
    try:
        blob = dict(getattr(row, "data", {}) or {})
    except Exception:                                            # noqa: BLE001
        return
    stored = blob.get("site_moments")
    if not stored:
        return

    class _Stored:
        """Just enough of `moments.SiteMoment` for `site_rows` to read."""

        def __init__(self, record: dict[str, Any]) -> None:
            self.index = int(record.get("i", 0))
            self.element = str(record.get("el", ""))
            self.total = float(record.get("tot", 0.0))
            self.channels = {k: float(v) for k, v in record.items()
                             if k in ("s", "p", "d", "f")}

    out.sites = site_rows(_Stored(r) for r in stored)
    out.elements = by_element(out.sites)
    if out.m_spheres is None and blob.get("m_spheres") is not None:
        out.m_spheres = float(blob["m_spheres"])
    out.notes.append("per-site moments read from the database, not from the "
                     "OUTCAR: the VASP directory is no longer readable")


def _apply_outcar(out: StructureDetail, outcar: Path) -> StructureDetail:
    from ..analysis.moments import MomentError, read_site_moments

    try:
        report = read_site_moments(outcar, out.symbols)
    except MomentError as exc:
        # The projected table and the structure disagree on ion count, which
        # means the OUTCAR and the CONTCAR are not the same calculation. Say so
        # rather than drawing a table whose rows are mislabelled.
        out.notes.append(f"per-site moments not shown: {exc}")
        return out
    except Exception as exc:                                     # noqa: BLE001
        out.notes.append(f"OUTCAR unreadable: {type(exc).__name__}: {exc}")
        return out

    out.m_cell = report.m_cell
    out.m_spheres = report.m_spheres
    out.sites = site_rows(report.sites)
    out.elements = by_element(out.sites)
    if report.note:
        out.notes.append(report.note)
    return out


def _formula_units(symbols: list[str]) -> int:
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return math.gcd(*counts.values()) if counts else 1


def cif_text(detail_: StructureDetail) -> str:
    """A P1 CIF of the cell shown, for download and for JSmol.

    P1 on purpose: the symmetry reported elsewhere on the page was found at a
    stated tolerance from a relaxed cell, and baking a symmetry group into the
    file would assert positions the calculation did not actually constrain.
    """
    p = detail_.lattice
    lines = [
        f"# {detail_.formula}, cspflow structure {detail_.structure_id}",
        f"# geometry: {detail_.geometry_source}",
        f"data_cspflow_{detail_.structure_id}",
        f"_cell_length_a {p['a']:.6f}",
        f"_cell_length_b {p['b']:.6f}",
        f"_cell_length_c {p['c']:.6f}",
        f"_cell_angle_alpha {p['alpha']:.4f}",
        f"_cell_angle_beta {p['beta']:.4f}",
        f"_cell_angle_gamma {p['gamma']:.4f}",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "loop_",
        " _symmetry_equiv_pos_as_xyz",
        "  'x, y, z'",
        "loop_",
        " _atom_site_label",
        " _atom_site_type_symbol",
        " _atom_site_fract_x",
        " _atom_site_fract_y",
        " _atom_site_fract_z",
    ]
    for i, (symbol, (x, y, z)) in enumerate(zip(detail_.symbols, detail_.frac), 1):
        lines.append(f"  {symbol}{i} {symbol} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines) + "\n"
