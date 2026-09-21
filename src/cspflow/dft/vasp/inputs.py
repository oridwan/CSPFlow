"""Assemble a complete VASP job directory.

`csp dft --dry-run` calls exactly this and prints what it produced, which is the
property that makes the dry run worth having: there is no second code path that
"would" write the files.

Four files, and one manifest:

    INCAR     the recipe's literal tags plus the computed ones (incar.py)
    POSCAR    the structure, species-sorted
    KPOINTS   from the recipe's scheme and the cell (kpoints.py)
    POTCAR    concatenated in POSCAR species order
    inputs.json  what was resolved, and the hashes that pin it

The manifest is the part that is easy to skip and expensive to have skipped. It
records the resolved INCAR, the POTCAR `md5_header_hash` per element, and the
recipe stage -- so `settings_hash` describes what was actually written rather
than what the config said, and two jobs can be shown to be comparable without
re-reading their inputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ...config.schema import Dft, Machine
from ..recipe import RecipeStage, tag_value
from . import potcar as pc
from .incar import IncarContext, build_incar, render_incar
from .kpoints import KpointGrid, canonical_cell, carry_grid, grid_for
from .parallel import irreducible_kpoints
from .parallel import plan as parallel_plan

# VASP's own floor for ISMEAR=-5; below it the run aborts.
TETRAHEDRON_MIN_KPOINTS = 4


class InputError(Exception):
    pass


@dataclass
class ResolvedInputs:
    """Everything that will be written, before anything is."""

    stage: str
    incar: dict[str, Any]
    grid: KpointGrid
    symbols: list[str]                 # POSCAR species order
    potcars: list[pc.PotcarInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # The cell the grid was actually chosen from -- see kpoints.canonical_cell.
    # write_inputs must use this and not the caller's copy, or POSCAR and
    # KPOINTS describe different bases.
    atoms: Any = None
    parallel: Any = None

    @property
    def settings_hash(self) -> str:
        """A hash over what will actually be computed.

        Deliberately includes the POTCAR hashes: two jobs with identical INCARs
        and different pseudopotentials are not comparable, and nothing else in
        the record would show it.
        """
        payload = {
            "stage": self.stage,
            "incar": {k: _hashable(v) for k, v in sorted(self.incar.items())},
            "kpoints": [self.grid.a, self.grid.b, self.grid.c, self.grid.scheme],
            "potcars": sorted((p.element, p.symbol, p.md5_header_hash)
                              for p in self.potcars),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.settings_hash[:12]

    def render(self) -> str:
        lines = [f"stage {self.stage}   settings_hash {self.short_hash}", "",
                 "INCAR", *(f"  {line}" for line in
                            render_incar(self.incar).rstrip().splitlines()),
                 "", "KPOINTS",
                 *(f"  {line}" for line in self.grid.render().rstrip().splitlines()),
                 "", "POTCAR"]
        for p in self.potcars:
            lines.append(f"  {p.element:<4} {p.symbol:<6} {p.titel:<28} "
                         f"ZVAL {p.zval:6.1f}  {p.short_hash}")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


def resolve_inputs(
    atoms,
    stage: RecipeStage,
    dft: Dft,
    machine: Machine,
    *,
    ntasks: int | None = None,
    carried_grid: KpointGrid | None = None,
) -> ResolvedInputs:
    """Work out every input file's contents.  Writes nothing.

    `carried_grid` is the grid a previous attempt of this same step actually
    ran with. Passing it keeps a resumed relaxation on the sampling it was
    already using instead of re-deriving one from the cell it has reached --
    see `kpoints.carry_grid` for why that distinction is not cosmetic.
    """
    symbols = _species_order(atoms)
    infos, errors = pc.resolve_all(
        sorted(set(symbols)), machine,
        tree=dft.potcar.tree, f_treatment=dft.rare_earth.f_treatment,
        overrides=dft.potcar.overrides if hasattr(dft.potcar, "overrides") else None,
    )
    if errors:
        raise InputError(
            "cannot resolve every POTCAR for this structure:\n  " + "\n  ".join(errors)
        )
    pc.assert_one_f_convention(infos)

    by_element = {p.element: p for p in infos}
    zvals = {e: p.zval for e, p in by_element.items()}
    f_in_valence = any(p.n_f_valence for p in infos)

    context = IncarContext(
        symbols=list(atoms.get_chemical_symbols()),
        formula=atoms.get_chemical_formula(),
        zvals=zvals,
        f_in_valence=f_in_valence,
    )
    incar = build_incar(
        stage.incar, context,
        magnetism=dft.magnetism, rare_earth=dft.rare_earth, ldau=dft.ldau,
        nbands=dft.nbands, overrides=dft.incar_overrides,
    )

    warnings = []
    encut = incar.get("ENCUT")
    max_enmax = pc.max_enmax(infos)
    if encut is not None and float(encut) < max_enmax:
        warnings.append(
            f"ENCUT {encut} eV is below the largest ENMAX in this structure's POTCARs "
            f"({max_enmax:.1f} eV). VASP will run, and the basis is under-converged "
            f"for that species."
        )

    # The grid must come from a canonical basis, not from however the source
    # happened to write the lattice down.  See kpoints.canonical_cell.
    source_lengths = list(atoms.cell.lengths())
    atoms = canonical_cell(atoms)
    grid = grid_for(list(atoms.cell.lengths()), stage.kpoints, n_atoms=len(atoms))

    # A resume keeps the grid it was already running.  The derived grid is
    # still computed, because the interesting case is the one where the two
    # disagree: that is a relaxation whose cell drifted across a floor()
    # boundary, and it is recorded here so the manifest shows it happened
    # rather than leaving two attempts silently incomparable.
    if carried_grid is not None:
        remapped = carry_grid(carried_grid, source_lengths,
                              list(atoms.cell.lengths()))
        if remapped is None:
            warnings.append(
                f"could not carry the previous attempt's "
                f"{carried_grid.a}x{carried_grid.b}x{carried_grid.c} grid onto this "
                f"cell -- the canonical basis changed by more than a permutation. "
                f"Falling back to the derived {grid.a}x{grid.b}x{grid.c} grid; the "
                f"two attempts are not sampled alike."
            )
        else:
            if (remapped.a, remapped.b, remapped.c) != (grid.a, grid.b, grid.c):
                warnings.append(
                    f"k-point grid carried from the previous attempt: "
                    f"{remapped.a}x{remapped.b}x{remapped.c}. The cell reached by "
                    f"that attempt would have derived "
                    f"{grid.a}x{grid.b}x{grid.c} instead -- re-deriving it would "
                    f"move the energy surface under a geometry that is already "
                    f"near its minimum."
                )
            grid = remapped

    # The tetrahedron method needs at least four k-points.  Below that VASP does
    # not approximate -- it aborts ("Tetrahedron method fails for NKPT<4"), and
    # it does so in the STATIC step, after the relaxation has already been paid
    # for.  Measured on this reference set: 35 of 2,746 MP phases land on a grid
    # product below 4 at reciprocal_density 64, 29 of them Gamma-only.
    #
    # The fallback is Gaussian smearing, not a denser grid.  A cell whose grid
    # collapses to Gamma is a large cell with a small Brillouin zone, where
    # Gamma-only sampling is already adequate and forcing more k-points would
    # cost far more than the few meV of smearing difference.
    #
    # This is a per-STRUCTURE resolution, so it lands in `settings_hash` and not
    # in `recipe_id`: the recipe still says ISMEAR -5, and the cache stays one
    # cache.  The substitution is recorded here so it is visible in the manifest
    # rather than being an unexplained difference between two INCARs.
    # The IRREDUCIBLE count, not the grid product.  VASP folds the mesh by
    # symmetry before BZINTS counts, so a 2x2x2 grid -- product 8, comfortably
    # over the limit -- can present VASP with 3 k-points and abort:
    #     VERY BAD NEWS! internal error in subroutine BZINTS:
    #     Tetrahedron method fails (number of k-points < 4) 3
    # Measured 2026-09-10 on mp-1192814-Ce3Si3Pd102 and three others, all of
    # which the old grid-product test waved through.  spglib supplies the same
    # number the parallel planner below already asks it for.
    nkpts = irreducible_kpoints(
        atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers(),
        (grid.a, grid.b, grid.c),
    )
    if nkpts is None:               # spglib absent: the grid product is a
        nkpts = grid.total          # valid upper bound
    if str(tag_value(incar, "ISMEAR", 0)).strip() == "-5":
        n_kpoints = nkpts
        if n_kpoints < TETRAHEDRON_MIN_KPOINTS:
            incar["ISMEAR"] = 0
            incar.setdefault("SIGMA", 0.05)
            warnings.append(
                f"ISMEAR -5 replaced with 0 (SIGMA {incar['SIGMA']}): the "
                f"{grid.a}x{grid.b}x{grid.c} grid folds to {n_kpoints} "
                f"irreducible k-point(s) and "
                f"the tetrahedron method needs at least {TETRAHEDRON_MIN_KPOINTS}. "
                f"VASP would abort rather than approximate."
            )

    ordered = [by_element[s] for s in symbols]
    # --- how the job will be divided across ranks -------------------------
    # NCORE and KPAR change only how the work is split, never the converged
    # answer, so they land in settings_hash and never in recipe_id.  A fixed
    # NCORE across a campaign is wrong whenever the k-point count varies: this
    # reference set spanned 1 to 232 irreducible k-points on one NCORE = 8.
    par = None
    if ntasks:
        try:
            par = parallel_plan(
                nkpts=nkpts,
                nbands=int(tag_value(incar, "NBANDS", 0) or 0),
                ntasks=int(ntasks),
                cores_per_socket=getattr(machine, "cores_per_socket", None),
            )
            incar.update(par.incar_tags())
        except Exception as exc:        # never let a tuning choice stop a run
            warnings.append(f"parallelization left at recipe defaults: {exc}")

    return ResolvedInputs(stage=stage.name, incar=incar, grid=grid,
                          symbols=symbols, potcars=ordered, warnings=warnings,
                          atoms=atoms, parallel=par)


def write_inputs(resolved: ResolvedInputs, atoms, directory: Path) -> Path:
    """Write the four files and the manifest.  Returns the directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "INCAR").write_text(
        render_incar(resolved.incar,
                     comment=f"cspflow {resolved.stage}  {resolved.short_hash}")
    )
    (directory / "KPOINTS").write_text(resolved.grid.render())
    # the canonical cell the grid was chosen from, never the caller's copy
    _write_poscar(resolved.atoms if resolved.atoms is not None else atoms,
                  resolved.symbols, directory / "POSCAR")
    _concat_potcars(resolved.potcars, directory / "POTCAR")

    (directory / "inputs.json").write_text(json.dumps({
        "stage": resolved.stage,
        "settings_hash": resolved.settings_hash,
        "incar": {k: _hashable(v) for k, v in sorted(resolved.incar.items())},
        "kpoints": {"a": resolved.grid.a, "b": resolved.grid.b, "c": resolved.grid.c,
                    "scheme": resolved.grid.scheme, "gamma": resolved.grid.gamma},
        "potcars": [{"element": p.element, "symbol": p.symbol, "titel": p.titel,
                     "zval": p.zval, "enmax": p.enmax,
                     "md5_header_hash": p.md5_header_hash} for p in resolved.potcars],
        "warnings": resolved.warnings,
    }, indent=2, sort_keys=True))
    return directory


# --------------------------------------------------------------------------


def _species_order(atoms) -> list[str]:
    """First appearance, not alphabetical.

    POTCAR concatenation order and the positional LDAU arrays both key on this,
    so getting it wrong applies the wrong pseudopotential to the wrong element --
    and VASP will not complain, because the file is well-formed.
    """
    seen: list[str] = []
    for symbol in atoms.get_chemical_symbols():
        if symbol not in seen:
            seen.append(symbol)
    return seen


def _write_poscar(atoms, symbols: Sequence[str], path: Path) -> None:
    """Species-sorted POSCAR with an explicit element line.

    The element line is not optional here for the same reason Stage 0.3 refuses
    a POSCAR without one: a VASP-4 file has no species names and every reader
    that accepts it has to invent them.
    """
    from ase.io import write

    ordered = atoms.copy()
    order = sorted(range(len(ordered)),
                   key=lambda i: symbols.index(ordered.get_chemical_symbols()[i]))
    ordered = ordered[order]
    write(str(path), ordered, format="vasp", direct=True, sort=False)

    lines = path.read_text().splitlines()
    if len(lines) > 5 and all(tok.lstrip("+-").isdigit() for tok in lines[5].split()):
        # Older ASE omits the species line; put it back rather than shipping a
        # file we would ourselves refuse to read.
        counts: list[str] = []
        current, n = None, 0
        for s in ordered.get_chemical_symbols():
            if s != current:
                if current is not None:
                    counts.append(str(n))
                current, n = s, 1
            else:
                n += 1
        counts.append(str(n))
        lines.insert(5, " ".join(symbols))
        path.write_text("\n".join(lines) + "\n")


def _concat_potcars(infos: Sequence[pc.PotcarInfo], path: Path) -> None:
    with path.open("wb") as out:
        for info in infos:
            out.write(Path(info.path).read_bytes())


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_hashable(v) for v in value]
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    return value
