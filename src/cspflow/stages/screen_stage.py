"""Stage 2 -- MLIP relaxation.

A submitted stage: chunks of structures go to a GPU array. The shape of it is
worth stating because it is a deliberate departure from how the legacy scripts
work.

**Array workers never write to the database.** They read the structures they
were given (SQLite in WAL mode allows any number of concurrent readers), relax
them, and write their results to a per-task JSON file. The driver reads those
files and does all the writing, one process at a time.

The alternative -- every array task opening the campaign database for writing --
is what the ~48 concurrent DFT jobs of this cluster's CPU cap would produce, and
SQLite serialises writers with a lock. At best that is 48 processes taking turns;
at worst it is `database is locked` after the busy timeout, in a job that has
already spent its GPU minutes. Writing to a file the worker owns outright cannot
contend with anything, and the handoff is a file rename.

It also makes the failure mode benign: a worker that dies leaves no results file,
which reconciliation reports as missing rather than as silent partial data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config.loader import ResolvedConfig
from ..db.store import Store, StructureState
from ..scheduler.base import JobSpec, JobStatus
from .base import StageReport, WorkItem

# How many structures one array task relaxes. The plan's figure (pipeline.md
# sec.4.4) is 500-2,000; the default is the low end because a task that dies
# loses everything it had not yet written, and at ~1 s per structure on GPU a
# 500-structure task is under ten minutes.
DEFAULT_CHUNK = 500



# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------


def _batch_tag(items: list[WorkItem]) -> str:
    """A name for the submission that describes the WHOLE submission.

    It used to be `items[0].key` -- the first CHUNK's id range. One sbatch
    covers an array of chunks, so a job spanning structures 1-200 in four
    chunks went out called `screen-1-50`, naming a fifth of its own work. That
    is the same fault D130 found on the DFT side, where ten hours went into
    reading the directory a job name pointed at while the job ran elsewhere.

    So the span is taken across every chunk, ids are zero-padded (string sort =
    numeric sort, and `ids-0001-0003` cannot be misread as "1 of 3" the way
    `screen-1-3` can), and anything covering more than one chunk says so:

        one chunk, ids 1-3      ->  ids-0001-0003
        four chunks, ids 1-200  ->  ids-0001-0200-x4
    """
    ids = sorted({sid for item in items for sid in item.structure_ids})
    if not ids:
        return "ids-empty"
    span = f"ids-{ids[0]:04d}-{ids[-1]:04d}"
    return span if len(items) == 1 else f"{span}-x{len(items)}"


def _results_path(workdir: Path, tag: str, index: int) -> Path:
    """Where task `index` of `tag` wrote its results.

    Checks the current layout first and the pre-D136 flat one second. A results
    file that cannot be found is a whole chunk of screened structures marked
    failed, so the old location is still READ long after it stopped being
    written.
    """
    current = workdir / "batches" / tag / f"results-task{index}.json"
    if current.is_file():
        return current
    legacy = workdir / f"{tag}.task{index}.json"
    return legacy if legacy.is_file() else current

class ScreenStage:
    name = "screen"
    role = "gpu"
    in_process = False

    def __init__(self, cfg: ResolvedConfig, chunk: int = DEFAULT_CHUNK) -> None:
        self.cfg = cfg
        self.chunk = chunk

    # -- what is ready -----------------------------------------------------

    def pending(self, store: Store) -> int:
        return store.count_structures(state=StructureState.new.value)

    def estimate_tasks(self, store: Store, budget: int) -> int:
        """`budget` is in array tasks; `pending` is in structures."""
        import math

        return min(budget, math.ceil(self.pending(store) / self.chunk))

    def claim(self, store: Store, budget: int) -> list[WorkItem]:
        """Take up to `budget` structures and mark them `screening`.

        Marking on claim is what makes the driver safe to run twice: a second
        cycle, or a second driver, sees `screening` rather than `new` and does
        not take the same structures again. A worker that then dies leaves them
        in `screening`, which `csp status` reports as stuck -- visible, rather
        than quietly re-run forever.
        """
        rows = [r for r in store.structures(state=StructureState.new.value)]
        rows = rows[: budget * self.chunk]
        if not rows:
            return []
        # Seeds supplied with `relax: false` are read here, where the rows are
        # already in hand, rather than reopened in `build`.
        no_relax = {int(r.id) for r in rows
                    if r.key_value_pairs.get("needs_relax") is False}
        ids = [int(r.id) for r in rows]
        items = []
        for start in range(0, len(ids), self.chunk):
            batch = ids[start: start + self.chunk]
            for sid in batch:
                store.set_structure_state(sid, StructureState.screening)
            items.append(WorkItem(
                key=f"screen-{batch[0]}-{batch[-1]}", structure_ids=batch,
                payload={"single_point": sorted(no_relax.intersection(batch))}))
        return items

    # -- the job -----------------------------------------------------------

    def build(self, items: list[WorkItem], workdir: Path) -> JobSpec:
        """One array, one task per chunk, driven by a manifest on disk.

        The manifest is written rather than passed on the command line because a
        2,000-id argument list is both unreadable and, at scale, longer than the
        shell will accept.
        """
        workdir = workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        tag = _batch_tag(items)
        # Every task's results are named after this one tag, so each item has to
        # carry it: by reconcile time an item may arrive alone, and its own key
        # names a file the worker never wrote (D121).
        for item in items:
            item.group_key = tag
        # ONE DIRECTORY PER SUBMISSION (D136), the same shape `runs/` gives DFT.
        # Everything the worker writes is derived from the manifest's parent, so
        # putting the manifest here puts the results, the progress files and the
        # relaxed geometries here too -- and `spec.workdir` below puts the job
        # script and the SLURM log in beside them.
        batch_dir = workdir / "batches" / tag
        batch_dir.mkdir(parents=True, exist_ok=True)
        manifest = batch_dir / "inputs.json"
        manifest.write_text(json.dumps({
            "key": tag,
            "db": str(self.cfg.campaign_db.resolve()),
            "model": self.cfg.campaign.screen.mattersim.model,
            "fmax": self.cfg.campaign.screen.mattersim.fmax,
            "max_steps": self.cfg.campaign.screen.mattersim.max_steps,
            "chunks": [item.structure_ids for item in items],
            # Seeds supplied with `relax: false` get a single point instead.
            # The flag was written onto the structure at source time and read by
            # nothing, so `relax: false` -- documented as "MLIP-relax the seed
            # before DFT" -- relaxed it anyway, moving a geometry the user
            # supplied on purpose. Mode 3's control-group use depends on this.
            "single_point": sorted({sid for item in items
                                    for sid in item.payload.get("single_point", [])}),
        }, indent=2))

        resources = self.cfg.campaign.screen.resources
        return JobSpec(
            name=tag, stage=self.name, workdir=batch_dir,
            command=f"csp screen-worker --manifest {manifest}",
            role=self.role, ntasks=resources.ntasks or 1,
            cpus_per_task=resources.cpus_per_task or 1,
            gpus=resources.gpus or 1, mem=resources.mem or "32G",
            time=resources.time, array_size=len(items),
            env={"CSPFLOW_MANIFEST": str(manifest)},
        )

    # -- folding the answer back -------------------------------------------

    def reconcile(self, store: Store, job_row: Any, status: JobStatus,
                  items: list[WorkItem]) -> None:
        """Read what the workers wrote and record it.  Only the driver writes."""
        workdir = Path(job_row["workdir"])
        for position, item in enumerate(items):
            index = item.task_index if item.task_index is not None else position
            tag = item.group_key or item.key
            results_file = _results_path(workdir, tag, index)
            if not results_file.is_file():
                # The task produced nothing. Its structures are still marked
                # `screening`, which is the correct record: work was claimed and
                # did not come back. They are not silently returned to `new`,
                # because an unbounded retry of a structure that crashes the MLIP
                # is a loop, not a recovery.
                for sid in item.structure_ids:
                    store.set_structure_state(
                        sid, StructureState.failed,
                        fail_reason=f"screen worker produced no results ({status.raw_state})",
                    )
                continue
            self._absorb(store, json.loads(results_file.read_text()))

    def _absorb(self, store: Store, payload: dict) -> None:
        for row in payload.get("results", []):
            sid = int(row["structure_id"])
            if row.get("error"):
                store.set_structure_state(sid, StructureState.failed,
                                          fail_reason=row["error"][:200])
                store.add_filter_event(structure_id=sid, gate="screen:validate",
                                       passed=False, detail=row["error"][:200])
                continue

            store.add_relaxation(
                # `energy`, not `e_total`. The worker's ASE key_value_pairs row
                # must call it `e_total` (ASE reserves `energy` on a row), and
                # that name leaked into this call -- but `relaxation` has an
                # `energy` column and no `e_total`, so every absorb raised
                # TypeError and the whole MLIP screen path died. Fixed
                # 2026-09-15; two tests in test_mlip.py cover it.
                structure_id=sid, engine=row.get("engine", "mattersim"),
                energy=row.get("energy"), e_per_atom=row.get("e_per_atom"),
                converged=bool(row.get("converged")), n_steps=int(row.get("n_steps", 0)),
                volume_before=row.get("volume_before"), volume_after=row.get("volume_after"),
            )
            kv = {"mlip_e_per_atom": row["e_per_atom"],
                  "mlip_converged": bool(row["converged"]),
                  "mlip_steps": int(row.get("n_steps", 0)),
                  # False for a seed the campaign asked not to move: its energy
                  # is a single point at the geometry as supplied, and nothing
                  # downstream should read it as a relaxed one.
                  "mlip_relaxed": bool(row.get("relaxed", True))}
            if row.get("volume_drift") is not None:
                kv["mlip_volume_drift"] = float(row["volume_drift"])

            # Carry the relaxed cell back into the row, so the DFT relax starts
            # from it instead of from the seed as supplied. Only when the MLIP
            # actually moved the structure: a `single_point` seed is one the
            # campaign asked NOT to move, and replacing its geometry would be
            # exactly the thing that setting exists to prevent.
            geometry = row.get("geometry")
            if geometry and row.get("relaxed", True):
                carried = _read_relaxed(geometry, sid)
                if carried is not None:
                    store.replace_geometry(sid, carried)
                    kv["mlip_geometry"] = str(geometry)
                    # And into the composition-keyed record, which is what the
                    # campaign can be rebuilt FROM (D141). The driver writes it,
                    # not the worker: a chunk spans many compositions and two
                    # chunks can share one, so workers here would contend on the
                    # same SQLite file.
                    self._keep(store, sid, carried, row, kv)

            store.set_structure_state(sid, StructureState.screened, **kv)
            store.add_filter_event(
                structure_id=sid, gate="screen:converged",
                passed=bool(row["converged"]),
                value=float(row.get("n_steps", 0)),
                threshold=float(payload.get("max_steps", 0)),
                detail="" if row["converged"] else "stopped at the step limit",
            )

    def run(self, store: Store) -> StageReport:            # pragma: no cover
        raise AssertionError("screen is a submitted stage; the driver calls claim/build")


    def _keep(self, store: Store, sid: int, atoms, row: dict, kv: dict) -> None:
        """File one relaxed cell under its composition. Never fatal.

        This database is the safety net under `campaign.db`. A campaign that
        stopped because its safety net could not be written would be worse than
        one that carries on without it, so every failure here is swallowed --
        the authoritative record of this result is already in the store by the
        time this runs.
        """
        from .. import artifacts

        try:
            structure = next(store.structures(id=sid), None)
            skv = structure.key_value_pairs if structure is not None else {}
            formula = skv.get("reduced_formula") or (
                structure.formula if structure is not None else "") or "unknown"
            artifacts.record(
                artifacts.db_path(self.cfg.work_dir, formula, "relaxed"), atoms,
                structure_id=sid, reduced_formula=formula,
                campaign=self.cfg.campaign.name,
                source_name=skv.get("source_name", ""),
                source_path=skv.get("source_path", ""),
                origin=skv.get("origin", ""),
                e_total=row.get("energy"), e_per_atom=row.get("e_per_atom"),
                converged=bool(row.get("converged")),
                n_steps=int(row.get("n_steps") or 0),
                fmax_final=row.get("fmax"),
                volume_drift=row.get("volume_drift"),
                engine=row.get("engine", ""),
                e_above_hull_mlip=kv.get("e_above_hull_mlip"),
            )
        except Exception:                                      # noqa: BLE001
            pass


def _read_relaxed(path: str, structure_id: int | None = None):
    """The relaxed cell the worker saved, or None if it cannot be read.

    Two shapes, because both exist on disk:

    *   `relaxed-task<N>.db` (D141) -- one ASE database per task, from which the
        row for THIS structure is selected. A database read without an id would
        return whichever row came first, which is a silent way to hand a
        campaign the wrong geometry.
    *   `relaxed-task<N>/<sid>.vasp` -- one file per structure, what the worker
        wrote before. Still read, because a campaign screened last week has
        these and nothing else.

    A task database is read ONCE, whole, and kept (`_task_cells`): reconcile
    asks for its 500 structures one at a time, and opening a 500-row database
    500 times over NFS was a per-structure round trip for no reason (D144).

    Only geometry columns are read (D145). Databases written before that fix
    hold float32 force blobs that ASE decodes as float64 and raises on for an
    odd atom count; the positions beside them were always float64 and intact.

    None is not fatal: the energy is still valid and the campaign proceeds on
    the original geometry, which is what it did before these files existed. A
    missing cell must not lose a screening result.
    """
    from pathlib import Path as _Path

    target = _Path(path)
    try:
        if target.suffix == ".db":
            if structure_id is None or not target.is_file():
                return None
            atoms = _task_cells(target).get(int(structure_id))
            return atoms.copy() if atoms is not None else None

        from ase.io import read as ase_read

        if not target.is_file() or not target.stat().st_size:
            return None
        return ase_read(str(target))
    except Exception:                                          # noqa: BLE001
        return None


#: (path, mtime_ns, size) -> {structure_id: Atoms}. Small on purpose: reconcile
#: walks one task file at a time, so the last few are all that is ever reused.
_TASK_CELLS: "dict[tuple, dict[int, Any]]" = {}
_TASK_CELLS_KEEP = 4


def _task_cells(target: Path) -> "dict[int, Any]":
    """Every cell in one relaxed-task database, keyed by structure id.

    Keyed on mtime and size too, so a task file rewritten by a retry is read
    again rather than served stale.
    """
    from ase.db import connect

    from ..artifacts import GEOMETRY_COLUMNS

    stat = target.stat()
    key = (str(target), stat.st_mtime_ns, stat.st_size)
    cells = _TASK_CELLS.get(key)
    if cells is not None:
        return cells
    cells = {}
    with connect(str(target), use_lock_file=False) as db:
        for row in db.select(columns=GEOMETRY_COLUMNS, include_data=False):
            sid = row.key_value_pairs.get("structure_id")
            if sid is not None:
                cells[int(sid)] = row.toatoms()
    while len(_TASK_CELLS) >= _TASK_CELLS_KEEP:
        _TASK_CELLS.pop(next(iter(_TASK_CELLS)))
    _TASK_CELLS[key] = cells
    return cells
