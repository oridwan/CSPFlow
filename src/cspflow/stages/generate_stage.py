"""Stage 1 -- structure generation.

A submitted stage, and the only one whose unit of work is a *composition* rather
than a structure.  Everything upstream of it is bookkeeping; everything
downstream is arithmetic on structures that exist.  This is where the campaign
starts spending GPU time.

Three choices here are worth stating, because each is a departure from the
legacy script and each was made against a measurement of it.

**Compositions are grouped by their requested count, not processed one at a
time.**  MatterGen loads a checkpoint and initialises CUDA on every invocation.
The legacy wrapper called it once per composition per supercell -- for the
ternary campaign, more than 1,692 loads.  MatterGen's own CLI accepts a *list*
of target compositions, so a group that wants the same number of structures each
costs one load between them.  The catch is that MatterGen divides the total by
the number of compositions with `//`, silently dropping the remainder and
returning nothing at all when the product is smaller than the list; the plan in
`generators/base.py` is constructed so the division is always exact.

**A task owns its own output directory tree, and results come back by
composition, not by position.**  See `generators/mattergen_engine.py`.

**Yield is recorded.**  `n_target` and `n_produced` both live on the composition
row, so `csp status` can say "1,692 compositions asked for 177,120 structures and
got 166,004" instead of reporting a number that only ever means "what arrived".
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config.loader import ResolvedConfig
from ..db.store import Origin, Store, StructureState
from ..generators import estimate_gpu_minutes
from ..scheduler.base import JobSpec, JobStatus
from .base import StageReport, WorkItem

# How many compositions share one MatterGen invocation. They must all want the
# same number of structures, so this is an upper bound rather than a batch size.
# Twenty compositions at the campaign's typical 72-200 structures each is
# 1,400-4,000 structures per task -- long enough to amortise the checkpoint
# load, short enough that a task that dies has not lost a day.
DEFAULT_GROUP = 20

# States a composition can be claimed from.
CLAIMABLE = "new"



def _results_path(workdir: Path, tag: str, index: int) -> Path:
    """Where task `index` wrote its results: the D136 layout, else the old flat
    one. A results file that cannot be found fails a whole batch, so the old
    location is still read long after it stopped being written."""
    current = workdir / "batches" / tag / f"results-task{index}.json"
    if current.is_file():
        return current
    legacy = workdir / f"{tag}.task{index}.json"
    return legacy if legacy.is_file() else current

class GenerateStage:
    name = "generate"
    role = "gpu"
    in_process = False

    def __init__(self, cfg: ResolvedConfig, group: int = DEFAULT_GROUP) -> None:
        self.cfg = cfg
        self.group = max(1, int(group))

    # -- what is ready -----------------------------------------------------

    def pending(self, store: Store) -> int:
        return len(self._claimable(store))

    @staticmethod
    def _claimable(store: Store) -> list:
        """Compositions still to be generated.

        Mode 3 (`structure_list`) writes its compositions straight to
        `generated`, because its structures came from disk and there is nothing
        to generate.  Filtering on state rather than on `source_mode` means that
        distinction is made once, at source time, and cannot drift.
        """
        return [c for c in store.compositions(state=CLAIMABLE) if c.n_target > 0]

    def estimate_tasks(self, store: Store, budget: int) -> int:
        """`budget` is in array tasks; compositions are grouped into them.

        The grouping is by requested count, so this is not `pending / group`:
        four compositions wanting three different counts are three tasks, not
        one.
        """
        import math
        from collections import Counter

        sizes = Counter(c.n_target for c in self._claimable(store))
        return min(budget, sum(math.ceil(n / self.group) for n in sizes.values()))

    def claim(self, store: Store, budget: int) -> list[WorkItem]:
        """Take up to `budget` tasks' worth of compositions, grouped by target.

        Marked `generating` on claim, for the same reason `screen` marks
        `screening`: a second cycle must not take the same work, and a worker
        that dies must leave evidence rather than silently returning its
        compositions to the pool to be tried forever.
        """
        available = self._claimable(store)
        if not available:
            return []

        by_target: dict[int, list] = defaultdict(list)
        for row in available:
            by_target[row.n_target].append(row)

        items: list[WorkItem] = []
        for target in sorted(by_target, reverse=True):
            rows = by_target[target]
            for start in range(0, len(rows), self.group):
                if len(items) >= budget:
                    return items
                chunk = rows[start: start + self.group]
                items.append(WorkItem(
                    key=f"generate-n{target}-{chunk[0].id}-{chunk[-1].id}",
                    composition_ids=[c.id for c in chunk],
                    payload={"n_requested": target,
                             "compositions": [
                                 {"id": c.id, "formula": c.formula, "z": c.z,
                                  "n_atoms": c.n_atoms, "n_requested": c.n_target}
                                 for c in chunk]},
                    est_core_hours=estimate_gpu_minutes(target * len(chunk)) / 60.0,
                ))
        for item in items:
            for cid in item.composition_ids:
                store.set_composition_state(cid, "generating")
        return items

    # -- the job -----------------------------------------------------------

    def build(self, items: list[WorkItem], workdir: Path) -> JobSpec:
        """One array, one task per group, driven by a manifest on disk."""
        workdir = workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        tag = items[0].key
        # Every task's results are named after this one tag, so each item has to
        # carry it: by reconcile time an item may arrive alone, and its own key
        # names a file the worker never wrote (D121).
        for item in items:
            item.group_key = tag
        block = self.cfg.campaign.generate
        # One directory per submission (D136), matching the screen stage. The
        # worker derives every path it writes from the manifest's parent, so
        # this moves the results and the outputs here too.
        batch_dir = workdir / "batches" / tag
        batch_dir.mkdir(parents=True, exist_ok=True)
        manifest = batch_dir / "inputs.json"
        manifest.write_text(json.dumps({
            "key": tag,
            "db": str(self.cfg.campaign_db.resolve()),
            "engine": block.engine,
            "model": block.mattergen.model,
            "mode": block.mattergen.mode,
            "max_batch_size": block.mattergen.max_batch_size,
            "timeout_per_batch": block.mattergen.timeout_per_batch,
            "chunks": [item.payload["compositions"] for item in items],
        }, indent=2))

        resources = block.resources
        return JobSpec(
            name=tag, stage=self.name, workdir=batch_dir,
            command=f"csp generate-worker --manifest {manifest}",
            role=self.role, ntasks=resources.ntasks or 1,
            cpus_per_task=resources.cpus_per_task or 8,
            gpus=resources.gpus or 1, mem=resources.mem or "64G",
            time=resources.time, array_size=len(items),
            env={"CSPFLOW_MANIFEST": str(manifest),
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # -- folding the answer back -------------------------------------------

    def reconcile(self, store: Store, job_row: Any, status: JobStatus,
                  items: list[WorkItem]) -> None:
        workdir = Path(job_row["workdir"])
        for position, item in enumerate(items):
            index = item.task_index if item.task_index is not None else position
            tag = item.group_key or item.key
            results_file = _results_path(workdir, tag, index)
            if not results_file.is_file():
                for cid in item.composition_ids:
                    store.set_composition_state(
                        cid, "failed",
                        f"generate worker produced no results ({status.raw_state})")
                continue
            self._absorb(store, json.loads(results_file.read_text()))

    def _absorb(self, store: Store, payload: dict) -> None:
        import ase.io

        for row in payload.get("results", []):
            cid = int(row["composition_id"])
            if row.get("error") and not row.get("n_produced"):
                store.set_composition_state(cid, "failed", str(row["error"])[:200],
                                            n_produced=0)
                continue

            path = Path(row["output"])
            if not path.is_file():
                store.set_composition_state(
                    cid, "failed",
                    f"results claimed {row.get('n_produced')} structures but "
                    f"{path} is not there", n_produced=0)
                continue

            # The composition-keyed record, beside the campaign database rather
            # than inside it (D141). `campaign.db` is a cache; this is what it
            # can be rebuilt from, and it is keyed by composition because that
            # is what was asked for -- a task number is an accident of
            # scheduling that changes on the next run.
            from .. import artifacts

            kept = artifacts.db_path(self.cfg.work_dir, row["formula"], "generated")

            written = 0
            for atoms in ase.io.read(str(path), index=":"):
                sid = store.add_structure(
                    atoms, origin=Origin.generated, state=StructureState.new,
                    composition_id=cid, reduced_formula=row["formula"],
                    generator=payload.get("engine", "mattergen"),
                )
                written += 1
                # After the store, so the id it is filed under is the real one.
                artifacts.record(
                    kept, atoms,
                    structure_id=int(sid), reduced_formula=row["formula"],
                    campaign=self.cfg.campaign.name,
                    composition_id=cid,
                    generator=payload.get("engine", "mattergen"),
                    model=str(payload.get("model") or ""),
                    origin="generated",
                )

            # `written` is what the database now holds; `n_produced` is what the
            # worker counted. They should agree, and a mismatch means the file
            # changed under us -- which is worth a fail_reason rather than a
            # silent preference for one number over the other.
            reason = ""
            if written != int(row.get("n_produced", written)):
                reason = (f"worker counted {row.get('n_produced')} structures, "
                          f"{written} were read back from {path.name}")
            elif written < int(row["n_requested"]):
                reason = f"short: asked {row['n_requested']}, got {written}"
            store.set_composition_state(cid, "generated", reason, n_produced=written)

    def run(self, store: Store) -> StageReport:            # pragma: no cover
        raise AssertionError("generate is a submitted stage; the driver calls claim/build")
