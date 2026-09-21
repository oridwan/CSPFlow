"""What a stage is.

Every stage is a pure, idempotent, resumable function over the database:
`(rows in state X) -> (rows in state Y)`.  Nothing is passed between stages on
disk or by filename convention, which is what makes restart free -- kill
anything, rerun the same command, and it resumes from the rows.

Stages come in two kinds and the difference is not cosmetic:

*   **In-process** (`source`, `reference`, `calibrate`, `filter`, `analyze`) run
    in the driver itself, in seconds to minutes.  They need no scheduler.
*   **Submitted** (`generate`, `screen`, `dft`) go to the queue.  That is the
    entire scheduler surface -- three stages -- which is what replaces the ~210
    shell wrappers in the legacy campaigns.

The protocol below is deliberately small.  A stage says what work is ready,
takes as much of it as the driver's budget allows, describes the job, and later
reconciles a finished job back into the database.  It never decides how much to
submit, never sleeps, and never talks to the scheduler; the driver owns all
three, so throttling and budget policy exist in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..db.store import Store
from ..scheduler.base import JobSpec, JobStatus


@dataclass
class WorkItem:
    """One unit of work, at whatever granularity the stage uses.

    The unit differs per stage (pipeline.md sec.4.4): a chunk of compositions
    for `generate`, a chunk of structures for `screen`, exactly one structure at
    one recipe step for `dft`.  `key` is a stable identity so that claiming the
    same work twice is detectable rather than merely unlikely.
    """

    key: str
    structure_ids: list[int] = field(default_factory=list)
    composition_ids: list[int] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    est_core_hours: float = 0.0
    # Position in the submitted array, stamped by the driver.  A stage must use
    # THIS to name its results file, never the item's position in the list it is
    # handed: per-task reconciliation (D113) passes one item at a time, so
    # `enumerate` would call every task 0 and read task 0's results for all of
    # them.
    task_index: int | None = None
    # The tag every task in this submission shares -- `build` names the manifest
    # and all the result files after ONE item's key, so the others cannot find
    # their own results by their own key.  Stamped by `build`, before the claim
    # is written.
    group_key: str = ""


@dataclass
class StageReport:
    """What one stage did in one cycle."""

    stage: str
    claimed: int = 0
    submitted: int = 0
    reconciled: int = 0
    pending: int = 0
    note: str = ""

    @property
    def did_something(self) -> bool:
        return bool(self.claimed or self.submitted or self.reconciled)

    def render(self) -> str:
        bits = [f"{self.stage:<10}"]
        if self.pending:
            bits.append(f"pending {self.pending}")
        if self.claimed:
            bits.append(f"claimed {self.claimed}")
        if self.submitted:
            bits.append(f"submitted {self.submitted}")
        if self.reconciled:
            bits.append(f"reconciled {self.reconciled}")
        if self.note:
            bits.append(self.note)
        return "  ".join(bits)


@runtime_checkable
class Stage(Protocol):
    """The whole interface the driver needs from a stage."""

    name: str
    role: str
    in_process: bool

    def pending(self, store: Store) -> int:
        """How much work is waiting.  Must not mutate anything."""

    def claim(self, store: Store, budget: int) -> list[WorkItem]:
        """Take up to `budget` units and mark them claimed, atomically enough
        that a second driver cycle does not take them again."""

    def build(self, items: list[WorkItem], workdir: Path) -> JobSpec:
        """Describe the job for these items.  Submitted stages only."""

    def reconcile(self, store: Store, job_row: Any, status: JobStatus,
                  items: list[WorkItem]) -> None:
        """Fold a finished job back into the database.

        This is where a stage may write physics -- energies, convergence,
        properties.  `status` carries the *process* outcome only, and a job that
        SLURM calls `COMPLETED` may still have produced nothing usable.
        """

    def run(self, store: Store) -> StageReport:
        """Do the work here and now.  In-process stages only."""
