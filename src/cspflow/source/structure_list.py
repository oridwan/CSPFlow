"""Mode 3: structures in, no generation.

MatterGen is skipped entirely; files are ingested, composition is *derived* from
each file rather than given, and the rows enter the funnel at Stage 2.

This is the one place in the pipeline that reads files nobody in the pipeline
wrote, so it is where malformed input is likely, and every gate below is a hard
error at ingest rather than a warning.  The reason is uniform: each of these
failures otherwise surfaces after GPU or DFT time has been spent, or -- worse --
never surfaces at all, and a wrong number reaches the results table looking
exactly like a right one.

**Why both pymatgen and ASE parse every file.**  Measured on this machine, and
this is the finding that shaped the module: neither library is safe alone, and
their blind spots do not overlap.

    file                                pymatgen              ASE
    --------------------------------    ------------------    ------------------
    VASP-4 POSCAR, comment 'FeCo test'  H1 He1 (+warning)     CoFe
    VASP-4 POSCAR, comment 'my struct'  H1 He1 (+warning)     ParseError
    CIF with 0.5/0.5 mixed Ti/Fe site   Ti0.5 Fe1.5,          Fe2, NO WARNING
                                        is_ordered=False      (Ti silently gone)

pymatgen *invents* hydrogen and helium for a POSCAR with no element line, and
does so with nothing but a warning on stderr -- in a batch loop, H and He then
proceed into MatterSim and VASP.  ASE refuses that file, which is the right
answer, but silently discards the minority species of a partial-occupancy CIF
and hands back a clean-looking ordered cell that is not the material in the
file.  A pipeline built on either one alone inherits its blind spot.

So both read every file and must agree on the composition.  The cost is
milliseconds per seed; the failure it prevents is a published number for a
material that was never computed.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any, Iterable

from ..chem import ChemError, canonical_formula, chemsys, n_atoms as _n_atoms, reduce_counts
from ..config.schema import Source
from .base import EmittedStructure, SourceError, SourceResult

# Extensions we will hand to a structure parser.  A directory in `paths` is
# walked for these; anything else in it is ignored rather than guessed at, so
# dropping a README next to your seeds is not an error.
STRUCTURE_SUFFIXES = frozenset(
    {".vasp", ".poscar", ".contcar", ".cif", ".xyz", ".extxyz", ".res", ".json"}
)
STRUCTURE_STEMS = ("POSCAR", "CONTCAR")


def expand_structure_list(
    source: Source,
    base_dir: Path | None = None,
    *,
    default_max_atoms: int | None = None,
) -> SourceResult:
    block = source.structure_list
    if block is None:                                   # pragma: no cover - schema guards
        raise ValueError("expand_structure_list called on a source without a block")

    result = SourceResult(name=source.name, mode=source.mode.value)
    cap = block.max_atoms or default_max_atoms or source.defaults.max_atoms

    files = _collect_files(block.paths, base_dir)
    if not files:
        raise SourceError(
            f"source '{source.name}': structure_list.paths matched no files. "
            f"Patterns tried: {block.paths}"
        )

    for path in files:
        try:
            atoms, counts = read_seed(path)
        except SeedError as exc:
            # A curated input list is something the user wrote, so a file in it
            # that cannot be read is a mistake they want to hear about now.
            raise SourceError(f"source '{source.name}': {exc}") from exc

        total = _n_atoms(counts)
        if total > cap:
            result.rejected.add(f"cell over max_atoms ({cap})", f"{path.name}: {total} atoms")
            continue

        reduced, z = reduce_counts(counts)
        result.structures.append(
            EmittedStructure(
                atoms=atoms,
                formula=canonical_formula(reduced),
                chemsys=chemsys(reduced),
                counts=reduced,
                z=z,
                n_atoms=total,
                path=str(path),
                content_hash=_sha256(path),
                source_name=source.name,
                source_mode=source.mode.value,
                relax=block.relax,
            )
        )

    _dedup_within_set(result, block.dedup)

    if not block.relax:
        result.warnings.append(
            "relax: false -- DFT will run at the geometry as given, so a seed that "
            "is not already at a minimum will report a stressed energy"
        )
    return result


# --------------------------------------------------------------------------
# Reading one seed
# --------------------------------------------------------------------------


class SeedError(SourceError):
    """One input file that cannot be trusted, with the remedy named."""


def read_seed(path: Path) -> tuple[Any, dict[str, int]]:
    """Read one structure file through every gate.  Returns ``(Atoms, counts)``."""
    if path.suffix.lower() in {".vasp", ".poscar", ".contcar"} or path.name.startswith(
        STRUCTURE_STEMS
    ):
        _gate_poscar_has_symbols(path)

    structure = _read_pymatgen(path)
    _gate_ordered(path, structure)

    counts = {str(el): int(round(n)) for el, n in structure.composition.get_el_amt_dict().items()}
    if not counts:
        raise SeedError(f"{path}: parsed to an empty composition")

    atoms = _to_atoms(structure)
    _gate_parsers_agree(path, counts, atoms)
    return atoms, counts


def _gate_poscar_has_symbols(path: Path) -> None:
    """Refuse a VASP-4 POSCAR before any parser gets the chance to invent elements.

    This runs on the raw text rather than on a parsed object on purpose: it is
    the only gate here that does not depend on which libraries are installed or
    on their version, and it names the exact fix.
    """
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        raise SeedError(f"{path}: cannot be read ({exc})") from exc
    if len(lines) < 7:
        raise SeedError(f"{path}: too short to be a POSCAR ({len(lines)} lines)")

    symbol_line = lines[5].split()
    if not symbol_line:
        raise SeedError(f"{path}: line 6 is empty; a POSCAR needs a species line there")
    if all(tok.lstrip("+-").isdigit() for tok in symbol_line):
        raise SeedError(
            f"{path}: line 6 is counts ({' '.join(symbol_line)}) with no element line "
            f"above it -- this is VASP-4 format. pymatgen does not refuse such a file; "
            f"it invents elements (an FeCo POSCAR parses as H1 He1 with only a "
            f"BadPoscarWarning). Insert the element symbols as line 6, e.g. 'Fe Co'."
        )


def _read_pymatgen(path: Path):
    try:
        from pymatgen.core import Structure
    except ImportError as exc:                          # pragma: no cover
        raise SeedError(
            f"{path}: reading seed structures needs pymatgen, which is not importable "
            f"in this environment ({exc}). Run `csp doctor` for the environment report."
        ) from exc

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            structure = Structure.from_file(str(path))
        except Exception as exc:
            raise SeedError(f"{path}: pymatgen could not parse this file ({exc})") from exc

    for w in caught:
        name = type(w.message).__name__
        if name == "BadPoscarWarning":
            raise SeedError(
                f"{path}: {w.message} -- refused. These are not the elements in your "
                f"file; they are pymatgen's placeholders for a POSCAR with no species "
                f"line, and they would proceed into MatterSim and VASP as real atoms."
            )
    return structure


def _gate_ordered(path: Path, structure) -> None:
    """Partial occupancy fails late and in the wrong place; fail it here instead.

    A mixed-site CIF parses cleanly, yields a fractional composition, survives
    composition assignment, and only dies at DFT input generation -- one full
    stage after the MLIP time was spent -- with
    ``ValueError: Disordered structure with partial occupancies cannot be
    converted into POSCAR!``
    """
    if getattr(structure, "is_ordered", True):
        return
    sites = [
        f"site {i} = {dict(site.species.as_dict())}"
        for i, site in enumerate(structure)
        if not site.is_ordered
    ]
    raise SeedError(
        f"{path}: partial occupancies ({structure.composition}). "
        f"{'; '.join(sites[:3])}. Order it first -- e.g. pymatgen's "
        f"OrderDisorderedStructureTransformation -- and supply the ordered cells. "
        f"Left as is, this survives ingestion and MLIP relaxation and only fails at "
        f"POSCAR generation, after the GPU time is spent."
    )


def _to_atoms(structure):
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(structure)


def _gate_parsers_agree(path: Path, counts: dict[str, int], atoms) -> None:
    """Require pymatgen and ASE to read the same composition.  See module docstring."""
    try:
        from ase.io import read as ase_read
    except ImportError:                                 # pragma: no cover
        return

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            independent = ase_read(str(path))
        except Exception as exc:
            # ASE refusing a file pymatgen accepted is informative but not always
            # decisive -- it also happens for formats ASE simply does not support.
            # The POSCAR case, which is the dangerous one, is already caught by
            # `_gate_poscar_has_symbols` before we get here.
            raise SeedError(
                f"{path}: pymatgen read this as {canonical_formula(counts)} but ASE "
                f"refused it ({type(exc).__name__}: {exc}). Two parsers disagreeing "
                f"about whether a file is readable is not a file to compute on."
            ) from exc

    ase_counts: dict[str, int] = {}
    for symbol in independent.get_chemical_symbols():
        ase_counts[symbol] = ase_counts.get(symbol, 0) + 1

    if ase_counts != counts:
        raise SeedError(
            f"{path}: parsers disagree. pymatgen reads {canonical_formula(counts)}, "
            f"ASE reads {canonical_formula(ase_counts)}. This is how a partial-occupancy "
            f"CIF looks: ASE drops the minority species without a warning and hands back "
            f"a clean ordered cell that is not the material in the file. Resolve the file "
            f"before computing on it."
        )


# --------------------------------------------------------------------------
# Collecting and deduplicating
# --------------------------------------------------------------------------


def _collect_files(patterns: Iterable[str], base_dir: Path | None) -> list[Path]:
    """Expand paths, globs and directories into a sorted, deduplicated file list."""
    found: list[Path] = []
    seen: set[Path] = set()

    for pattern in patterns:
        expanded = Path(pattern).expanduser()
        if not expanded.is_absolute() and base_dir is not None:
            root, rel = base_dir, str(expanded)
        else:
            root, rel = None, str(expanded)

        if any(ch in rel for ch in "*?["):
            matches = sorted(root.glob(rel)) if root else sorted(Path("/").glob(rel.lstrip("/")))
        else:
            p = (root / rel) if root else Path(rel)
            matches = [p]

        for match in matches:
            if match.is_dir():
                candidates = sorted(q for q in match.rglob("*") if _looks_structural(q))
            elif match.exists():
                candidates = [match]
            else:
                raise SourceError(f"structure_list path does not exist: {match}")
            for c in candidates:
                resolved = c.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(c)
    return found


def _looks_structural(path: Path) -> bool:
    return path.is_file() and (
        path.suffix.lower() in STRUCTURE_SUFFIXES or path.name.startswith(STRUCTURE_STEMS)
    )


def _dedup_within_set(result: SourceResult, policy: str) -> None:
    """Report structural duplicates among the seeds; drop them only if asked.

    `warn` is the default here and that is the opposite of Stage 2's default,
    deliberately.  For generated structures, dropping duplicates is the entire
    point of screening.  For a list the user curated it is a hazard: a relaxed
    and an unrelaxed copy of one prototype, or two settings of the same cell,
    would be silently merged and you would never learn which one survived.
    """
    if policy == "off" or len(result.structures) < 2:
        return

    by_formula: dict[str, list[int]] = {}
    for i, s in enumerate(result.structures):
        by_formula.setdefault(s.formula, []).append(i)

    pairs: list[tuple[str, int, int]] = []
    for formula, indices in by_formula.items():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                pairs.append((formula, i, j))
    if not pairs:
        return

    verdict = _compare_pairs(result.structures, pairs)

    drop: set[int] = set()
    for formula, i, j in pairs:
        if i in drop or j in drop or not verdict[(i, j)]:
            continue
        a, b = result.structures[i].path, result.structures[j].path
        if policy == "drop":
            drop.add(j)
            result.warnings.append(f"{b} duplicates {a} ({formula}); dropped")
        else:
            result.warnings.append(
                f"{b} is structurally the same as {a} ({formula}); BOTH kept "
                f"(dedup: warn). Set dedup: drop, or dedup: off to skip the "
                f"comparison entirely, if that is not what you want."
            )
    if drop:
        result.structures = [s for i, s in enumerate(result.structures) if i not in drop]


# Below this many pairs, a process pool costs more to start than it saves.
_PARALLEL_MIN_PAIRS = 16
_MAX_WORKERS = 16


def _reduce_one(struct):
    """The primitive+Niggli reduction StructureMatcher would do internally.

    Hoisted out so it happens once per structure instead of once per pair --
    see `_compare_pairs`.  Module level so a process pool can pickle it.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher

    try:
        return StructureMatcher._get_reduced_structure(
            struct, primitive_cell=True, niggli=True)
    except Exception:                                          # noqa: BLE001
        return None


def _fit_pair(args):
    """One comparison between two ALREADY-REDUCED structures.

    `primitive_cell=False` because the reduction has been done: with it left at
    the default True, the matcher redoes it on both sides of every pair.
    Measured on 739 CePdGe pairs, the two give identical verdicts (61 matches
    either way) and this is 2.4x faster.
    """
    a, b = args
    if a is None or b is None:
        return False
    from pymatgen.analysis.structure_matcher import StructureMatcher

    try:
        return bool(StructureMatcher(primitive_cell=False).fit(a, b))
    except Exception:                                          # noqa: BLE001
        return False


def _worker_count(n_tasks: int) -> int:
    """How many processes we may actually use.

    `os.cpu_count()` reports the MACHINE, not the allocation. On the Orion login
    node it says 48 while the cgroup permits 2, and under `sbatch
    --cpus-per-task=N` it reports the whole node. Oversubscribing a 2-core
    cpuset with 16 workers is slower than staying serial.
    `sched_getaffinity` is the number actually schedulable.
    """
    import os

    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:                                     # pragma: no cover
        usable = os.cpu_count() or 1
    return max(1, min(n_tasks, usable, _MAX_WORKERS))


def _compare_pairs(structures, pairs) -> dict[tuple[int, int], bool]:
    """Decide `same structure?` for every candidate pair.

    Two things make this worth its own function rather than a nested loop.

    **Convert each structure ONCE.**  The comparison used to call
    `AseAtomsAdaptor.get_structure` on both sides of every pair, so a seed in a
    group of 14 was converted 13 times over.  232 CePdGe seeds meant 1,478
    conversions where 232 would do.

    **Reduce each structure ONCE too.**  `StructureMatcher.fit` with its default
    `primitive_cell=True` re-derives the primitive, Niggli-reduced cell of BOTH
    inputs on every call, so a seed in a 14-member group had its reduction
    recomputed 13 times.  Hoisting it out and comparing with
    `primitive_cell=False` gives identical verdicts -- 61 matches either way on
    the 739 CePdGe pairs -- and is 2.4x faster on its own (139 s -> 58 s).  That
    is a bigger win than the parallelism below, and the two multiply.

    **Run what is left in parallel.**  The fits are independent and pure, so
    they parallelise exactly; the drop bookkeeping stays sequential afterwards,
    which keeps the result and the warning order identical to the serial
    version.

    Falls back to serial silently -- a login node with no spare cores, or a
    pool that cannot start, must still produce the right answer.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    verdict: dict[tuple[int, int], bool] = {}
    todo: list[tuple[int, int]] = []
    for _formula, i, j in pairs:
        # An identical file is identical; no geometry comparison needed.
        if structures[i].content_hash == structures[j].content_hash:
            verdict[(i, j)] = True
        else:
            todo.append((i, j))
    if not todo:
        return verdict

    wanted = sorted({i for pair in todo for i in pair})
    try:
        converted = [AseAtomsAdaptor.get_structure(structures[i].atoms) for i in wanted]
    except Exception:                                          # noqa: BLE001
        for pair in todo:
            verdict[pair] = False
        return verdict

    workers = _worker_count(max(len(wanted), len(todo)))
    parallel = workers > 1 and len(todo) >= _PARALLEL_MIN_PAIRS

    def _map(fn, items):
        """Pool if it is worth it and it works; serial otherwise, same answer."""
        if parallel:
            try:
                from concurrent.futures import ProcessPoolExecutor

                with ProcessPoolExecutor(max_workers=workers) as pool:
                    return list(pool.map(fn, items, chunksize=1))
            except Exception:                                  # noqa: BLE001
                pass
        return [fn(x) for x in items]

    reduced = dict(zip(wanted, _map(_reduce_one, converted)))
    results = _map(_fit_pair, [(reduced[i], reduced[j]) for i, j in todo])

    verdict.update(dict(zip(todo, results)))
    return verdict


def _structure_matcher():
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
    except ImportError:                                 # pragma: no cover
        return None
    return StructureMatcher()


def _same_structure(a: EmittedStructure, b: EmittedStructure, matcher) -> bool:
    if a.content_hash == b.content_hash:
        return True
    if matcher is None:
        return False
    from pymatgen.io.ase import AseAtomsAdaptor

    try:
        return bool(matcher.fit(AseAtomsAdaptor.get_structure(a.atoms),
                                AseAtomsAdaptor.get_structure(b.atoms)))
    except Exception:
        return False


def _sha256(path: Path) -> str:
    """A content hash that ASE will accept as a key_value_pair.

    The `sha256:` prefix is not decoration. A bare 16-character hex digest is
    number-like roughly once in 600 -- every digit, or digits-e-digits -- and ASE
    refuses any string key_value_pair that `float()` or `int()` would parse,
    because such a value would come back from the database as a number:

        ValueError: Value 35720269025179e3 is put in as string but can be
        interpreted as float!

    Measured on a 14,777-seed campaign: 23 seeds produced a rejected digest
    (e.g. `35720269025179e3`, `4516549686969335`), and the source stage died on
    the first of them AFTER `--dry-run` had reported success -- the dry run never
    writes, so it cannot see this. A 92-seed campaign hits it about once in seven
    runs, which is how it survived CeFeB and CePdGe.

    The prefix makes the value unparseable as a number by construction rather
    than by luck, and says what the digest is. It is provenance only: nothing
    queries content_hash back out of the database, and comparisons are within a
    single run, so the format is free to change.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()[:16]
