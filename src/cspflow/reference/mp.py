"""Stage 3 -- Materials Project reference phases, and the trap in them.

**The finding that determines this module's design.** MP's thermo endpoint
returns several `thermo_type` values for one chemical system, and for
`GGA_GGA+U_R2SCAN` -- MP's mixed scheme -- the two energy fields come from
*different functionals*:

    Fe-Sm, measured 2026-08-27 against the live API

    formula    thermo_type              raw   corrected
    SmFe2      GGA_GGA+U            -7.1966     -7.1966
    SmFe2      GGA_GGA+U_R2SCAN    -19.4095     -7.1966   <-- raw is r2SCAN
    SmFe2      r2SCAN              -19.4095    -19.4095
    Sm2Fe17    GGA_GGA+U            -8.0523     -8.0523
    Sm2Fe17    GGA_GGA+U_R2SCAN     -8.0523     -8.0523   <-- no r2SCAN calc

`uncorrected_energy_per_atom` on a `GGA_GGA+U_R2SCAN` row is the raw energy of
whichever calculation that row came from -- r2SCAN where one exists, GGA where
none does. So the natural instruction "use raw energies, because that is what an
MLIP is trained on" produces a reference set mixing GGA and r2SCAN raw energies
**inside one chemical system**, differing by 5-10 eV/atom.

Confirmed across four systems on the same day:

    Fe-Sm     SmFe2  gap  12.21 eV/atom      SmFe5  gap 9.08
    Co-Gd     GdCo5  gap   9.93
    Fe-Ti     TiFe2  gap   5.56              TiFe   gap 5.29
    Fe-Nd     NdFe5  gap   9.02

It is silent because it affects a *minority* of entries -- one or two per system,
exactly those that happen to have an r2SCAN calculation. Everything else looks
right, and the few that are wrong are wrong by enough to destroy the hull.

**So: raw-scale reference data comes from `thermo_type == "GGA_GGA+U"` and
nothing else.** Requesting the raw scale from the mixed scheme is refused.

The API key is read from `$MP_API_KEY` and never written to a cache file, a log
line, or an error message.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..chem import canonical_formula, chemsys as chemsys_of, parse_formula
from .hull import Entry

# MP's own labels.
THERMO_GGA = "GGA_GGA+U"
THERMO_MIXED = "GGA_GGA+U_R2SCAN"
THERMO_R2SCAN = "R2SCAN"

# Deliberately NOT under ~/.cache.  On every convention that defines it --
# XDG's, and the plain English one -- a cache is data that may be deleted at any
# time and regenerated on demand.  Half of what lives here cannot: the MP
# download is regenerable, but our own DFT energies for the same phases cost
# thousands of core-hours and no amount of network access brings them back.  A
# directory named "cache" invites exactly the cleanup that would destroy them.
#
# `$CSPFLOW_REFERENCE` overrides.  `$CSPFLOW_CACHE` is still honoured so an
# existing setup does not silently start writing somewhere else.
REFERENCE_ENV = "CSPFLOW_REFERENCE"
LEGACY_ENV = "CSPFLOW_CACHE"
DEFAULT_REFERENCE = Path.home() / ".local" / "share" / "cspflow" / "reference"


class ReferenceError(Exception):
    pass


def reference_root() -> Path:
    """Where reference data lives -- deliberately OUTSIDE any campaign.

    A `structure_list` campaign and a `chemical_space` campaign that both touch
    Sm-Fe-Ti should pay for that reference set once, not once each. Keyed on
    chemsys + thermo_type + scale, which is the whole of what determines the
    numbers.
    """
    for name in (REFERENCE_ENV, LEGACY_ENV):
        value = os.environ.get(name)
        if value:
            return Path(value)
    return DEFAULT_REFERENCE


# The old name, kept so nothing outside this module breaks on the rename.
# It was never a cache; see the comment on DEFAULT_REFERENCE.
cache_root = reference_root


def assert_scale_matches_thermo_type(thermo_type: str, energy_scale: str) -> None:
    """Refuse the combination that silently mixes functionals.

    See the module docstring. This is a hard error rather than a warning because
    the result is a hull that is wrong by 5-10 eV/atom for a handful of entries
    and right for the rest -- which no amount of looking at the output reveals.
    """
    if energy_scale == "raw" and thermo_type.upper() == THERMO_MIXED.upper():
        raise ReferenceError(
            f"energy_scale='raw' cannot be taken from thermo_type='{THERMO_MIXED}'. "
            f"On that mixed scheme, `uncorrected_energy_per_atom` is the raw energy of "
            f"whichever functional produced the row -- r2SCAN where such a calculation "
            f"exists, GGA where it does not -- so the raw energies within one chemical "
            f"system differ by 5-10 eV/atom. Measured 2026-08-27: SmFe2 differs by "
            f"12.21 eV/atom, GdCo5 by 9.93, TiFe2 by 5.56. "
            f"Use thermo_type='{THERMO_GGA}' for raw-scale work, or "
            f"energy_scale='mp_corrected' if you want the mixed scheme's own numbers."
        )


@dataclass
class ReferenceEntry:
    """One MP phase, with both energy scales kept and neither preferred."""

    mp_id: str
    formula: str
    chemsys: str
    counts: dict[str, int]
    n_atoms: int
    thermo_type: str
    e_raw_per_atom: float | None = None
    e_corrected_per_atom: float | None = None
    e_above_hull_mp: float | None = None
    run_type: str = ""

    def energy_on(self, scale: str) -> float:
        per_atom = (self.e_raw_per_atom if scale == "raw" else self.e_corrected_per_atom)
        if per_atom is None:
            raise ReferenceError(
                f"{self.mp_id} has no {scale} energy recorded; it cannot enter a "
                f"{scale}-scale hull. Refetch, or exclude it explicitly."
            )
        return per_atom * self.n_atoms

    def to_hull_entry(self, scale: str) -> Entry:
        return Entry(
            label=self.mp_id, counts=dict(self.counts), energy=self.energy_on(scale),
            scale="raw" if scale == "raw" else "mp_corrected", source="mp",
            run_type=self.run_type or self.thermo_type,
        )


@dataclass
class FetchResult:
    chemsys: str
    thermo_type: str
    entries: list[ReferenceEntry] = field(default_factory=list)
    snapshot_id: str = ""
    from_cache: bool = False
    fetched_at: str = ""
    warnings: list[str] = field(default_factory=list)

    def hull_entries(self, scale: str) -> list[Entry]:
        return [e.to_hull_entry(scale) for e in self.entries]

    def render(self) -> str:
        lines = [f"{self.chemsys}: {len(self.entries)} {self.thermo_type} entries "
                 f"({'cache' if self.from_cache else 'fetched'}, "
                 f"snapshot {self.snapshot_id[:12]})"]
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


# --------------------------------------------------------------------------


def fetch_chemsys(
    chemsys: str,
    *,
    thermo_type: str = THERMO_GGA,
    energy_scale: str = "raw",
    refresh: bool = False,
    api_key: str | None = None,
    cache: Path | None = None,
) -> FetchResult:
    """Reference phases for one chemical system, cached across campaigns."""
    assert_scale_matches_thermo_type(thermo_type, energy_scale)

    elements = sorted(e for e in chemsys.split("-") if e)
    if not elements:
        raise ReferenceError(f"not a chemical system: {chemsys!r}")
    chemsys = "-".join(elements)

    path = _cache_path(chemsys, thermo_type, cache)
    if path.is_file() and not refresh:
        return _from_cache(path)

    docs = _query(chemsys, thermo_type, api_key)
    entries, warnings = _to_entries(docs, thermo_type)
    result = FetchResult(
        chemsys=chemsys, thermo_type=thermo_type, entries=entries,
        snapshot_id=snapshot_id(entries), from_cache=False,
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"), warnings=warnings,
    )
    _to_cache(path, result)
    return result


def fetch_structures(
    chemsys: str, *, api_key: str | None = None, cache: Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """MP's own DFT-relaxed structures, keyed by material id.

    Needed for Stage 4a, which evaluates the MLIP **at MP's geometry** rather
    than at its own -- see `calibrate/parity.py` for why that separation is not
    optional.

    Cached as CIF text alongside the energies. CIF rather than pymatgen's JSON
    because it round-trips through both pymatgen and ASE, and the cache should
    not require the library that wrote it.
    """
    path = _cache_path(chemsys, "structures", cache).with_suffix(".cif.json")
    if path.is_file() and not refresh:
        return _structures_from_cache(path)

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise ReferenceError(
            "no Materials Project API key. Set MP_API_KEY in the environment; it is "
            "read from there and never written to a cache file, a log or an error."
        )
    import warnings as _warnings

    from mp_api.client import MPRester

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with MPRester(key) as mpr:
            docs = mpr.materials.summary.search(
                chemsys=sub_systems(chemsys),
                fields=["material_id", "structure"],
            )

    out = {str(d.material_id): d.structure for d in docs if d.structure is not None}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps({mp_id: s.to(fmt="cif") for mp_id, s in out.items()}))
    tmp.replace(path)
    return out


def _structures_from_cache(path: Path) -> dict[str, Any]:
    from pymatgen.core import Structure

    payload = json.loads(path.read_text())
    out = {}
    for mp_id, cif in payload.items():
        try:
            out[mp_id] = Structure.from_str(cif, fmt="cif")
        except Exception:                                  # pragma: no cover
            continue
    return out


def snapshot_id(entries: Iterable[ReferenceEntry]) -> str:
    """A hash over exactly the numbers that would change a hull.

    Recorded in provenance (pipeline.md §12.7) so that "re-running reproduces
    the hull" is checkable rather than hoped for, and a `--refresh` that moves
    something can say what moved.
    """
    payload = sorted(
        (e.mp_id, e.thermo_type, round(e.e_raw_per_atom or 0.0, 9),
         round(e.e_corrected_per_atom or 0.0, 9), e.n_atoms)
        for e in entries
    )
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def compare_snapshots(old: FetchResult, new: FetchResult) -> list[str]:
    """What a `--refresh` changed.  Empty means the hull is unmoved."""
    before = {e.mp_id: e for e in old.entries}
    after = {e.mp_id: e for e in new.entries}
    notes = []
    for mp_id in sorted(set(after) - set(before)):
        notes.append(f"+ {mp_id} {after[mp_id].formula} is new")
    for mp_id in sorted(set(before) - set(after)):
        notes.append(f"- {mp_id} {before[mp_id].formula} is gone")
    for mp_id in sorted(set(before) & set(after)):
        a, b = before[mp_id], after[mp_id]
        for scale, x, y in (("raw", a.e_raw_per_atom, b.e_raw_per_atom),
                            ("corrected", a.e_corrected_per_atom, b.e_corrected_per_atom)):
            if x is not None and y is not None and abs(x - y) > 1e-6:
                notes.append(f"~ {mp_id} {a.formula} {scale} {x:.6f} -> {y:.6f} "
                             f"({y - x:+.6f} eV/atom)")
    return notes


# --------------------------------------------------------------------------


def sub_systems(chemsys: str) -> list[str]:
    """Every sub-system of a chemical system, including the elements themselves.

    `chemsys="Fe-Sm"` on MP's modern API matches **exactly** that binary and
    returns no elemental Fe and no elemental Sm. A hull built from what comes
    back has no lower boundary, and every `e_above_hull` would be measured
    against whichever compound happened to be lowest.

    Measured: `Fe-Sm` returns 7 `GGA_GGA+U` entries, all binary compounds, and
    `build_hull` refuses them. The legacy `get_entries_in_chemsys` that
    `prescreen.py` uses walks the sub-systems for you, which is very likely what
    its comment about the modern client "missing many stable GGA phases" is
    really describing.
    """
    from itertools import combinations

    elements = sorted(e for e in chemsys.split("-") if e)
    out = []
    for size in range(1, len(elements) + 1):
        out.extend("-".join(combo) for combo in combinations(elements, size))
    return out


def _query(chemsys: str, thermo_type: str, api_key: str | None) -> list[Any]:
    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise ReferenceError(
            "no Materials Project API key. Set MP_API_KEY in the environment; it is "
            "read from there and never written to a cache file, a log or an error."
        )
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise ReferenceError(f"mp_api is not importable: {exc}") from exc

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with MPRester(key) as mpr:
            # Every sub-system, not just the full one -- see `sub_systems`.
            docs = mpr.materials.thermo.search(
                chemsys=sub_systems(chemsys),
                fields=["material_id", "formula_pretty", "chemsys", "thermo_type",
                        "energy_per_atom", "uncorrected_energy_per_atom",
                        "energy_above_hull", "nsites", "energy_type"],
            )
    # Filtered here rather than in the query so that what MP offered and what we
    # kept are both visible, and the discarded types can be reported.
    return [d for d in docs if str(d.thermo_type).upper() == thermo_type.upper()]


def _to_entries(docs: list[Any], thermo_type: str) -> tuple[list[ReferenceEntry], list[str]]:
    entries, warnings = [], []
    for doc in docs:
        try:
            counts = parse_formula(str(doc.formula_pretty))
        except Exception as exc:
            warnings.append(f"{doc.material_id}: unparseable formula "
                            f"{doc.formula_pretty!r} ({exc})")
            continue
        n_atoms = sum(counts.values())
        entries.append(ReferenceEntry(
            mp_id=str(doc.material_id),
            formula=canonical_formula(counts),
            chemsys=chemsys_of(counts),
            counts=counts,
            n_atoms=n_atoms,
            thermo_type=thermo_type,
            e_raw_per_atom=_float(getattr(doc, "uncorrected_energy_per_atom", None)),
            e_corrected_per_atom=_float(getattr(doc, "energy_per_atom", None)),
            e_above_hull_mp=_float(getattr(doc, "energy_above_hull", None)),
            # `energy_type` is the functional that produced *this document's*
            # energy -- 'GGA' or 'GGA+U'. `thermo_type` is the name of the
            # mixing scheme the document was drawn from, and is the same string
            # for every row in the set.
            #
            # Storing the latter under `run_type` made every entry claim the
            # same functional, which is exactly the condition `assert_one_
            # functional` exists to detect: the guard saw one uniform value and
            # passed a set that mixed GGA and GGA+U. Verified against MP: a
            # Co-Gd query returns thermo_type='GGA_GGA+U' with energy_type='GGA'
            # on every document.
            run_type=str(getattr(doc, "energy_type", "") or "") or thermo_type,
        ))
    if not entries:
        warnings.append(
            f"no {thermo_type} entries returned. A hull cannot be built from an empty "
            f"reference set, and an empty one is not the same as a stable candidate."
        )
    return entries, warnings


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cache_path(chemsys: str, thermo_type: str, cache: Path | None) -> Path:
    root = cache or reference_root()
    safe = thermo_type.replace("+", "p").replace("/", "_")
    return root / f"{chemsys}__{safe}.json"


def _to_cache(path: Path, result: FetchResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chemsys": result.chemsys,
        "thermo_type": result.thermo_type,
        "snapshot_id": result.snapshot_id,
        "fetched_at": result.fetched_at,
        "warnings": result.warnings,
        "entries": [asdict(e) for e in result.entries],
    }
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _from_cache(path: Path) -> FetchResult:
    payload = json.loads(path.read_text())
    entries = [ReferenceEntry(**row) for row in payload["entries"]]
    stored = payload.get("snapshot_id", "")
    recomputed = snapshot_id(entries)
    warnings = list(payload.get("warnings", []))
    if stored and stored != recomputed:
        # The cache file was edited, or written by a different version. Either
        # way its snapshot id no longer identifies its contents, and provenance
        # recorded against it would be a lie.
        warnings.append(
            f"cache {path.name} records snapshot {stored[:12]} but its contents hash "
            f"to {recomputed[:12]}; the file has been modified. Re-fetch with "
            f"--refresh rather than trusting it."
        )
    return FetchResult(
        chemsys=payload["chemsys"], thermo_type=payload["thermo_type"],
        entries=entries, snapshot_id=recomputed, from_cache=True,
        fetched_at=payload.get("fetched_at", ""), warnings=warnings,
    )
