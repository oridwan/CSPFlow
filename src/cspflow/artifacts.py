"""Composition-keyed ASE databases: the record the campaign can be rebuilt from.

WHY THIS EXISTS
    `campaign.db` is a cache. It has been wrong before: D129 lost 37 finished
    calculations because two drivers shared one job table, and the results sat
    on disk, converged and unreachable, while the database said nothing had
    happened. D131 answered that for DFT by writing `structure.json` beside each
    run. The screen and generate stages had no equivalent -- a relaxed cell went
    to `<batch>/relaxed-task0/<id>.vasp`, geometry only, with the energy that
    belongs to it in a different file.

    So: one ASE database per composition, carrying the cells AND the numbers
    measured for them. Delete `campaign.db` and these still hold the work.

WHY COMPOSITION AND NOT BATCH
    A batch is an accident of scheduling -- whichever structures happened to be
    claimed in one cycle, chunked by 500. Re-run the campaign and the batches
    differ. It is not a thing anyone looks for.

    A composition is what was asked for, it is stable across runs, and it is how
    the question is actually posed: "what did we get for CeFe5?".

WHY THE DRIVER WRITES THESE AND NOT THE WORKER
    One screen chunk spans many compositions, and two chunks can share one, so
    workers writing here would contend on the same SQLite file -- which is the
    exact problem per-task files exist to avoid. The driver is already the only
    process that writes `campaign.db` (see `worker.py`'s docstring), so making
    it the only writer here keeps one rule instead of adding a second.

    The worker still writes its OWN task database, immediately, so a worker that
    dies keeps everything it finished.

INPUTS
    a workdir, a composition/formula, ASE Atoms plus the values measured for them

OUTPUTS
    <workdir>/structures/<formula>/relaxed.db     MLIP-relaxed cells
    <workdir>/structures/<formula>/generated.db   as generated, before relaxation

RUN
    written automatically by the driver when a screen or generate job is
    reconciled; read by `scripts/repair/rebuild_campaign.py`
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

#: What ASE's key_value_pairs will accept. Anything else is dropped rather than
#: crashing a write: losing one annotation is survivable, losing the geometry
#: it was attached to is not.
_SCALAR = (str, int, float, bool)

#: Names ASE owns on a row and refuses as key_value_pairs -- it raises
#: `ValueError: Bad key: energy` rather than storing them. This has now cost
#: time twice in one day: `volume` silently read back as ASE's own cell volume
#: in `report/structures.py`, and `energy`/`formula` hard-failed every write
#: here. A colliding key is prefixed rather than dropped, because the value is
#: usually the one worth keeping and a silent disappearance is how a record
#: starts lying.
_RESERVED = frozenset({
    "energy", "forces", "stress", "magmom", "magmoms", "charge", "charges",
    "dipole", "formula", "volume", "mass", "natoms", "id", "unique_id",
    "ctime", "mtime", "user", "calculator", "pbc", "cell", "positions",
    "numbers", "fmax", "smax", "age", "key_value_pairs", "data",
})


def slug(formula: str) -> str:
    """A directory-safe form of a formula. `Ce2Fe17` stays `Ce2Fe17`."""
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(formula or "unknown"))
    return cleaned.strip("_") or "unknown"


def composition_dir(workdir: Path, formula: str) -> Path:
    return Path(workdir) / "structures" / slug(formula)


def db_path(workdir: Path, formula: str, kind: str = "relaxed") -> Path:
    return composition_dir(workdir, formula) / f"{kind}.db"


def as_float64(atoms: Any) -> Any:
    """A copy of `atoms` whose attached results are float64, or `atoms` itself.

    D145. ASE stores a result array as its raw bytes and reads it back ASSUMING
    float64 (`np.frombuffer(buf, float)`). MatterSim returns float32 forces, so
    every row the screen worker wrote held 4-byte forces read as 8-byte ones:

    * even atom count -- the byte length happens to divide by 8, the row reads,
      and its forces are silently garbage;
    * odd atom count  -- `ValueError: buffer size must be a multiple of element
      size`, which `_read_relaxed` swallowed as "no relaxed cell".

    On RE-magnets-CHGNet that was 4,520 of 14,752 structures that kept their
    UNRELAXED seed geometry for DFT, split exactly by atom-count parity.

    So results are copied through a single-point calculator as float64 before
    ASE sees them. A copy, so the caller's live calculator is untouched.
    """
    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", None)
    if not results:
        return atoms
    import numpy as np
    from ase.calculators.calculator import all_properties
    from ase.calculators.singlepoint import SinglePointCalculator

    clean: dict[str, Any] = {}
    for name, value in results.items():
        if name not in all_properties or value is None:
            continue
        array = np.asarray(value)
        if array.ndim == 0:
            clean[name] = float(array)
        elif np.issubdtype(array.dtype, np.floating):
            clean[name] = array.astype(np.float64)
        else:
            clean[name] = array
    copy = atoms.copy()
    copy.calc = SinglePointCalculator(copy, **clean)
    return copy


#: Every ASE row column except the calculator RESULT arrays. Selecting only these
#: is how a row written before D145, with float32 result blobs, still yields its
#: cell and positions -- which were always float64 and are intact.
GEOMETRY_COLUMNS = [
    "id", "unique_id", "ctime", "mtime", "username", "numbers", "positions",
    "cell", "pbc", "initial_magmoms", "initial_charges", "masses", "tags",
    "momenta", "constraints", "key_value_pairs",
]


def clean_kv(values: dict[str, Any]) -> dict[str, Any]:
    """Only what ASE can store, and never a None.

    ASE raises on a `None` value rather than storing a null, so a single
    unmeasured quantity would otherwise take the whole row down with it.
    """
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None or not isinstance(value, _SCALAR):
            continue
        if isinstance(value, float) and value != value:        # NaN
            continue
        name = str(key)
        if name in _RESERVED:
            name = f"x_{name}"
        out[name] = value
    return out


def record(path: Path, atoms: Any, **kv: Any) -> bool:
    """Write one structure into `path`, replacing any earlier row for its id.

    Replacing rather than appending is what makes a retry idempotent: a
    structure screened twice must leave one row carrying the SECOND answer, not
    two rows disagreeing with each other and no way to tell which is current.

    Returns False rather than raising if the write fails. This database is a
    safety net; a campaign that stopped because its safety net could not be
    written would be a worse campaign than one that carries on without it.
    """
    try:
        from ase.db import connect

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = clean_kv(kv)
        sid = values.get("structure_id")
        # No lock file (D144): one writer per file by construction, and ASE's
        # lock is acquired with timeout=inf, so a writer killed mid-record left
        # a `.lock` that hung the next one forever.
        with connect(str(path), use_lock_file=False) as db:
            if sid is not None:
                stale = [row.id for row in db.select(structure_id=sid,
                                                     columns=["id"], include_data=False)]
                if stale:
                    db.delete(stale)
            db.write(as_float64(atoms), **values)
        return True
    except Exception:                                          # noqa: BLE001
        return False


def read(path: Path) -> Iterator[Any]:
    """Every row in `path`, or nothing if it cannot be opened."""
    try:
        from ase.db import connect

        if not Path(path).is_file():
            return
        # Geometry, key-values and energy only: a file written before D145
        # holds float32 result blobs that raise on odd atom counts, and nothing
        # a rebuild needs lives in the forces.
        with connect(str(path), use_lock_file=False) as db:
            yield from db.select(columns=GEOMETRY_COLUMNS + ["energy", "calculator"])
    except Exception:                                          # noqa: BLE001
        return


def compositions(workdir: Path, kind: str = "relaxed") -> list[Path]:
    """Every composition database of `kind` under `workdir`, sorted."""
    root = Path(workdir) / "structures"
    return sorted(root.glob(f"*/{kind}.db")) if root.is_dir() else []
