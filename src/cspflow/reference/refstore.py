"""Hull entries read straight out of the reference store folder.

WHY THIS EXISTS
    The store at `$CSPFLOW_STORE` is the living source of truth: folders of
    VASP runs plus a derived `index.csv`.  It keeps being extended.  The older
    path put a SECOND copy of the same numbers under
    `$CSPFLOW_REFERENCE/computed/<recipe_id>/`, and a second copy goes stale --
    measured 2026-09-11, that cache held 2,713 DFT energies while the store
    held 3,408, and it was keyed to a policy the store no longer used.

    So: no export step, no second database, no recipe_id.  Read the folder.
    This is the same rule that makes `index.csv` trustworthy -- it is derived
    by walking the tree, so it cannot disagree with what is on disk.

THREE HULLS, ONE FOLDER
    Every structure in the store carries three energies, and each defines a
    hull on its own scale.  Mixing them is the one thing that must never
    happen (D101), so `source` is explicit and never defaulted silently:

        "mp"    Materials Project's own energy   -- mp-cache/*__GGA_GGApU.json
        "mlip"  MatterSim, as we ran it          -- index.csv e_mlip_relaxed
        "dft"   our VASP recompute               -- index.csv e_static_eV

    "mlip" and "dft" are OURS: same settings, same MLIP, same store.  "mp" is
    on MP's scale and is useful for triage and for comparing against, never
    for mixing into one of the other two.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from .computed import ComputedError
from .hull import Entry

EnergySource = Literal["dft", "mlip", "mp"]
ENERGY_SOURCES: tuple[str, ...] = ("dft", "mlip", "mp")

DEFAULT_STORE = "/projects/mmi/cspflow-shared/store"


class StoreError(ComputedError):
    """A refusal to build a hull from the store.

    Subclasses ComputedError deliberately: the refusal means exactly what it
    meant on the old cache path -- "a vertex is missing and I will not borrow
    one from another scale" -- so every handler that already catches that
    keeps working, including AnalyzeStage's, which turns it into a note rather
    than a crash.
    """


def store_root() -> Path:
    return Path(os.environ.get("CSPFLOW_STORE") or DEFAULT_STORE)


# --------------------------------------------------------------------------
# the three raw readers
# --------------------------------------------------------------------------

#: path -> (signature, value). D144: the readers below were called once or twice
#: PER CHEMICAL SYSTEM, and each call re-read the store from NFS -- `index.csv`
#: (3,806 rows) and every `mp-cache/*.json` (~680 files). A reference stage over
#: 288 systems made 576 such passes: 391,104 file opens, 292 of the stage's 320
#: seconds, with nothing to place. Now each is read once and reused until the
#: files on disk change.
#:
#: The values are shared, so callers must treat them as READ-ONLY. None mutates
#: them today; `scripts/refstore.py` builds new lists from them.
_READ_CACHE: dict[str, tuple[tuple, object]] = {}


def _signature(path: Path) -> tuple:
    """What changes when `path` changes, for the price of one stat or readdir.

    A file: its mtime and size. A directory: its mtime and entry count -- the MP
    cache is written to a temporary name and RENAMED into place (`mp.py`), and a
    rename updates the directory's mtime, so a refreshed file is seen.
    """
    try:
        st = path.stat()
    except OSError:
        return ("missing",)
    if path.is_dir():
        try:
            n = sum(1 for _ in os.scandir(path))
        except OSError:
            n = -1
        return ("dir", st.st_mtime_ns, n)
    return ("file", st.st_mtime_ns, st.st_size)


def _cached(path: Path, load):
    key = str(path)
    sig = _signature(path)
    hit = _READ_CACHE.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]
    value = load()
    _READ_CACHE[key] = (sig, value)
    return value


def load_index(root: Path | None = None) -> list[dict]:
    """`index.csv`, as written by `refstore.py index`.

    Derived from the folders, so it cannot drift from disk -- but it is only
    as fresh as the last `index` run, which is why `stale_minutes` exists.
    """
    p = (root or store_root()) / "index.csv"
    if not p.is_file():
        raise StoreError(
            f"no index.csv at {p}. Build it with:\n"
            f"    python scripts/refstore.py index"
        )
    def read() -> list[dict]:
        with p.open(newline="") as fh:
            return list(csv.DictReader(fh))

    return _cached(p, read)


def stale_minutes(root: Path | None = None) -> float:
    import time
    p = (root or store_root()) / "index.csv"
    return (time.time() - p.stat().st_mtime) / 60 if p.is_file() else float("inf")


def load_mp_cache(root: Path | None = None) -> dict[str, dict]:
    """mp_id -> MP's own entry: element counts, energies, e_above_hull."""
    folder = (root or store_root()) / "mp-cache"

    def read() -> dict[str, dict]:
        out: dict[str, dict] = {}
        for f in folder.glob("*__GGA_GGApU.json"):
            try:
                data = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            for e in data.get("entries", []):
                mid = e.get("mp_id")
                if mid and mid not in out:
                    out[mid] = e
        return out

    return _cached(folder, read)


def load_ignored(root: Path | None = None) -> dict[str, dict]:
    """Phases the store will not wait for -- too far above the hull to matter.

    IGNORED is not FAILED; see <store>/ignored.json.  A hull must still refuse
    when a phase that COULD be a vertex is missing, but refusing over one that
    provably cannot be is how a campaign gets blocked for nothing.
    """
    f = (root or store_root()) / "ignored.json"

    def read() -> dict[str, dict]:
        try:
            return json.loads(f.read_text()).get("ignored", {})
        except (OSError, ValueError):
            return {}

    return _cached(f, read)


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------

def _counts(row: dict, mp: dict | None, root: Path) -> dict[str, int] | None:
    """Element counts for OUR cell.

    Our structures came from MP, so the composition RATIO is MP's; only the
    cell multiplicity can differ.  Scaling MP's counts by that multiple is
    exact and costs nothing -- true for 3,459 of 3,463 rows measured
    2026-09-11.  For the handful where the ratio is not integral the POSCAR is
    read, because a wrong composition puts a point at the wrong place on the
    hull and nothing downstream can tell.
    """
    try:
        ours = int(row["n_atoms"])
    except (KeyError, TypeError, ValueError):
        ours = 0
    if mp and ours:
        theirs = int(mp.get("n_atoms") or 0)
        base = mp.get("counts") or {}
        if theirs and base:
            q = ours / theirs
            if abs(q - round(q)) < 1e-9:
                k = int(round(q))
                return {el: n * k for el, n in base.items()}
    # fall back to the structure itself
    folder = root / "structures" / row["folder"]
    for cand in ("dft_static/POSCAR", "dft_relax/CONTCAR", "dft_relax/POSCAR",
                 "mlip/CONTCAR", "POSCAR.orig"):
        p = folder / cand
        if p.is_file() and p.stat().st_size:
            try:
                from ase.io import read
                atoms = read(str(p))
            except Exception:                                  # noqa: BLE001
                continue
            from collections import Counter
            return dict(Counter(atoms.get_chemical_symbols()))
    return None


# --------------------------------------------------------------------------
# coverage and entries
# --------------------------------------------------------------------------

@dataclass
class Coverage:
    chemsys: str
    source: str
    wanted: list[str] = field(default_factory=list)   # mp_ids in the system
    have: list[str] = field(default_factory=list)     # with an energy
    missing: list[str] = field(default_factory=list)  # without, and could be a vertex
    ignored: list[str] = field(default_factory=list)  # too far above the hull

    @property
    def complete(self) -> bool:
        return not self.missing


def _in_system(chemsys: str, want: set[str]) -> bool:
    els = {e for e in (chemsys or "").split("-") if e}
    return bool(els) and els <= want


def _energy(row: dict, mp: dict | None, source: str) -> float | None:
    """TOTAL energy for this row's cell, on the requested scale, or None."""
    if source == "dft":
        if row.get("ready") != "True":
            return None
        v = row.get("e_static_eV")
        return float(v) if v not in (None, "") else None
    if source == "mlip":
        v = row.get("e_mlip_relaxed")
        return float(v) if v not in (None, "") else None
    if source == "mp":
        if not mp:
            return None
        per = mp.get("e_raw_per_atom")
        n = mp.get("n_atoms")
        return float(per) * int(n) if per is not None and n else None
    raise StoreError(f"unknown energy source {source!r}; choose from {ENERGY_SOURCES}")


# A phase this far above MP's own hull cannot be a vertex, so its absence
# from the store cannot change the hull.  Matches `refstore.py hull`'s default.
ABSENT_MATTERS_BELOW = 0.10


def coverage(chemsys: str, source: EnergySource = "dft",
             root: Path | None = None,
             absent_matters_below: float = ABSENT_MATTERS_BELOW) -> Coverage:
    """What this system needs, what it has, and what is genuinely missing.

    `wanted` is NOT just what index.csv contains.  index.csv says what the
    store HAS; the MP cache says what EXISTS.  A phase that was never added to
    the store at all is invisible to the first and would let the hull build
    silently without it -- which is the exact failure the refusal exists to
    prevent, and it is how this function was first written.

    Absent phases are only counted as missing when MP puts them close enough
    to its own hull to be a plausible vertex.  Without that cut every system
    would refuse: the store deliberately skips phases far above the hull
    (`add` cuts at 0.5 eV/atom), so the MP cache is always the larger list --
    4,101 entries against 3,463 rows, measured 2026-09-11.
    """
    root = root or store_root()
    want = {e for e in chemsys.split("-") if e}
    mpc = load_mp_cache(root)
    ign = load_ignored(root)
    cov = Coverage(chemsys=chemsys, source=source)

    seen: set[str] = set()
    for row in load_index(root):
        if not _in_system(row.get("chemsys", ""), want):
            continue
        mid = row["mp_id"]
        seen.add(mid)
        cov.wanted.append(mid)
        if _energy(row, mpc.get(mid), source) is not None:
            cov.have.append(mid)
        elif mid in ign:
            cov.ignored.append(mid)
        else:
            cov.missing.append(mid)

    # phases MP has for this system that the store never took
    for mid, e in mpc.items():
        if mid in seen or not _in_system(e.get("chemsys", ""), want):
            continue
        cov.wanted.append(mid)
        if source == "mp":
            cov.have.append(mid)            # MP's own number is right here
            continue
        far = e.get("e_above_hull_mp")
        if mid in ign or (far is not None and float(far) > absent_matters_below):
            cov.ignored.append(mid)
        else:
            cov.missing.append(mid)
    return cov


def entries_for(chemsys: str, source: EnergySource = "dft", *,
                allow_partial: bool = False,
                root: Path | None = None) -> list[Entry]:
    """Hull vertices for one system on ONE scale, or a refusal.

    Partial coverage is refused, and that refusal is the point: a hull missing
    a vertex still builds and looks correct.  Phases listed in ignored.json do
    not count as missing -- they provably cannot be vertices.
    """
    root = root or store_root()
    if source not in ENERGY_SOURCES:
        raise StoreError(f"unknown energy source {source!r}; choose from {ENERGY_SOURCES}")
    cov = coverage(chemsys, source, root)
    if not cov.wanted:
        raise StoreError(
            f"{chemsys}: no phases in the store at all. Add them with:\n"
            f"    python scripts/refstore.py add {chemsys} --apply"
        )
    if cov.missing and not allow_partial:
        raise StoreError(
            f"{chemsys}: {len(cov.have)} of {len(cov.wanted)} phases have a "
            f"{source} energy; {len(cov.missing)} do not "
            f"({', '.join(cov.missing[:6])}{' ...' if len(cov.missing) > 6 else ''}). "
            f"Refusing rather than borrowing a vertex from another scale: a hull "
            f"with one borrowed vertex still builds and looks correct, and is "
            f"wrong by the scale offset in D101.\n"
            f"    python scripts/refstore.py submit dft --chemsys {chemsys} --apply"
        )
    mpc = load_mp_cache(root)
    have = set(cov.have)            # built once, not once per index row
    out: list[Entry] = []
    for row in load_index(root):
        if row["mp_id"] not in have:
            continue
        mp = mpc.get(row["mp_id"])
        e = _energy(row, mp, source)
        if e is None:
            continue
        if source == "mp":
            counts = dict(mp.get("counts") or {})
        else:
            counts = _counts(row, mp, root)
        if not counts:
            continue
        out.append(Entry(
            label=row["folder"], counts=counts, energy=e, scale="raw",
            source={"dft": "ours", "mlip": "mlip", "mp": "mp"}[source],
            run_type=(mp or {}).get("run_type", "GGA"),
        ))
    return out
