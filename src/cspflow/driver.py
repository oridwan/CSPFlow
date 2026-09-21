"""The driver loop.

`csp run` is not a science job. It is a tiny long-lived process that wakes,
queries the database, submits whatever is ready and under the limits, and sleeps
again (pipeline.md sec.4.4). `csp gen | screen | dft` perform *exactly the same
submission* for one stage -- manual and automatic are the same code, which is
the property that stops the two drifting apart.

Four things live here and nowhere else, so that policy exists in one place:

*   **Reconciliation before submission.** Every cycle polls what is in flight
    and folds it back into the database *first*. Submitting before reconciling
    would let a driver dispatch work whose predecessor had already failed.

*   **Throttling.** Stages describe work; the driver decides how much of it goes
    out, against the live QOS limits (D045).

*   **The core cap.** Phase B is a triage engine, not a throughput engine, so
    something has to stop it taking the whole machine. That something is
    `dft.max_cores` (D142): the cores this campaign's own jobs hold, counting
    queued as well as running, recomputed every cycle from what is actually
    out. The core-hour budget it replaced is retired (D143) -- it gated on a
    projection, and a projection made before any job of a stage had finished
    was the requested walltime, so it halted runs that had not overspent.

*   **Stopping.** A stop file, a cycle cap, and SIGTERM all end the loop
    cleanly, at a cycle boundary, with nothing half-submitted.
"""

from __future__ import annotations

import json
import os
import socket
import signal
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from .config.loader import ResolvedConfig
from .db.store import Store, is_transient_error
from .scheduler.base import JobSpec, JobState, JobStatus, Scheduler, compute_throttle
from .stages.base import Stage, StageReport, WorkItem

# Stage order is the funnel order.  `--through` and `--from` slice this list,
# which is what makes Phase A a barrier and Phase B a stream (pipeline.md 4.3).
STAGE_ORDER = [
    "source", "generate", "screen", "dedup", "reference",
    "filter", "dft", "analyze",
]
# Stages the configuration may legitimately leave out.  `generate` is built
# only when the campaign has a `generate:` block; a `structure_list` campaign
# brings its own structures and has none.  Asking for a slice that contains one
# of these is not an error -- it is simply skipped.  See `Driver._order`.
OPTIONAL_STAGES = frozenset({"generate"})

# `calibrate` used to sit between reference and filter.  It measured MatterSim
# against MP's DFT, because the hull that filters mixed those two scales.  The
# vertices of that hull are now MatterSim as well (reference_stage._vertices),
# so there are no longer two scales to compare and the stage had nothing left
# to measure.  Removed 2026-09-12, D126.


class DriverError(Exception):
    pass


#: Gaps between attempts when the DATABASE (never the science) fails with a
#: transient storage error. Seven tries over these gaps wait out 7 min 50 s.
#:
#: Sized against the real thing rather than a guess: the two aqu-fs10 outages on
#: 2026-09-19 lasted 3 min 43 s and 5 min 15 s, and the driver died 110 seconds
#: into the first one. The longest single gap is deliberately larger than the
#: longest outage, so the last attempt lands well clear of it (D148).
DB_RETRY_BACKOFF: tuple[int, ...] = (5, 15, 30, 60, 120, 240, 480)

_T = TypeVar("_T")


@dataclass
class CycleReport:
    """One pass of the loop."""

    cycle: int
    stages: list[StageReport] = field(default_factory=list)
    reconciled: int = 0
    in_flight: int = 0
    core_hours_spent: float = 0.0
    core_hours_projected: float = 0.0
    orphans: int = 0
    note: str = ""
    in_flight_detail: list[str] = field(default_factory=list)

    @property
    def did_something(self) -> bool:
        return self.reconciled > 0 or any(s.did_something for s in self.stages)

    def render(self) -> str:
        lines = [f"cycle {self.cycle}  in flight {self.in_flight}  "
                 f"spent {self.core_hours_spent:,.0f} core-h"]
        if self.core_hours_projected:
            lines[0] += f"  projected {self.core_hours_projected:,.0f}"
        # What the in-flight jobs are actually DOING. Without this the log was
        # "cycle 11  in flight 1" eleven times over, which says only that the
        # driver is alive -- not which job, not how far along, not where its
        # output is going.
        for detail in self.in_flight_detail:
            lines.append("  " + detail)
        for s in self.stages:
            if s.did_something or s.pending:
                lines.append("  " + s.render())
        if self.orphans:
            lines.append(f"  WARNING: {self.orphans} job row(s) have no scheduler id. "
                         f"A driver was killed while submitting; work may be running "
                         f"untracked. See `csp status` and the stage workdir.")
        if self.note:
            lines.append("  " + self.note)
        return "\n".join(lines)


@dataclass
class DriverOptions:
    interval: int = 300                     # seconds between cycles
    max_cycles: int | None = None           # None = until nothing is left
    stop_file: Path | None = None
    stages: Sequence[str] | None = None     # None = every registered stage
    dry_run: bool = False
    idle_exit_after: int = 0                # consecutive idle cycles -> stop
                                            # even under --watch. 0 = never.


def _claim_path(workdir: Path, job_id: str) -> Path:
    """`<stage>/claims/claim-<id>.json`.

    One claim file per SUBMISSION. That was a handful of files while a
    submission was a whole array; with one job per structure (D135) it is one
    per structure, and a few hundred of them beside the run directories is the
    clutter the `runs` layout exists to remove. They go in their own folder.
    """
    return Path(workdir) / "claims" / f"claim-{job_id}.json"


def _write_claim(workdir: Path, job_id: str, items: Sequence[WorkItem]) -> Path:
    """Record which work went into which submission, next to the job itself."""
    path = _claim_path(workdir, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"key": i.key, "structure_ids": i.structure_ids,
                "composition_ids": i.composition_ids, "payload": i.payload,
                "est_core_hours": i.est_core_hours, "task_index": n,
                "group_key": i.group_key}
               for n, i in enumerate(items)]
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return path


def _read_claim(workdir: Path, job_id: str) -> list[WorkItem] | None:
    """The claim a previous process wrote, or None if there is none."""
    path = _claim_path(workdir, job_id)
    if not path.is_file():
        # Where claims were written before they got their own folder. A claim
        # that cannot be found is a job that cannot be reconciled, so the old
        # location is still read -- it is never written.
        legacy = Path(workdir) / f"claim-{job_id}.json"
        if legacy.is_file():
            path = legacy
    if not path.is_file():
        return None
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return [WorkItem(key=r["key"], structure_ids=r.get("structure_ids", []),
                     composition_ids=r.get("composition_ids", []),
                     payload=r.get("payload", {}),
                     est_core_hours=r.get("est_core_hours", 0.0),
                     task_index=r.get("task_index", n),
                     group_key=r.get("group_key", ""))
            for n, r in enumerate(rows)]


def _estimate_tasks(stage: Stage, store: Store, budget: int) -> int:
    """How many jobs `stage` would submit, without claiming anything.

    Most stages submit one task per pending unit, so the default is the obvious
    one; a stage that batches says so by implementing `estimate_tasks`.
    """
    estimate = getattr(stage, "estimate_tasks", None)
    if estimate is not None:
        return int(estimate(store, budget))
    return min(stage.pending(store), budget)


class Driver:
    """Reconcile, submit, sleep, repeat."""

    def __init__(
        self,
        cfg: ResolvedConfig,
        store: Store,
        scheduler: Scheduler,
        stages: Sequence[Stage],
        options: DriverOptions | None = None,
        *,
        emit: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.scheduler = scheduler
        self.options = options or DriverOptions()
        self.emit = emit or (lambda _msg: None)
        self._sleep = sleep
        self._stop = False
        # Consecutive cycles with nothing pending and nothing in flight.
        # Lives on the instance, not in run(), so `_should_stop` can be
        # called on its own.
        self._idle_cycles = 0
        self._claims: dict[str, list[WorkItem]] = {}

        self.stages = self._order(stages)

    # -- setup -------------------------------------------------------------

    def _order(self, stages: Sequence[Stage]) -> list[Stage]:
        by_name = {s.name: s for s in stages}
        unknown = sorted(set(by_name) - set(STAGE_ORDER))
        if unknown:
            raise DriverError(
                f"stage(s) {unknown} are not in the funnel order {STAGE_ORDER}. "
                f"A stage the driver cannot place is a stage it cannot decide when "
                f"to run."
            )
        wanted = list(self.options.stages) if self.options.stages else STAGE_ORDER
        missing = [w for w in wanted if w not in by_name]

        # A stage can be absent from the registry for two very different
        # reasons, and only one of them is a problem.
        #
        # `generate` is absent whenever the campaign has no `generate:` block --
        # which is every `source.mode: structure_list` campaign, where the
        # structures are supplied rather than made.  That is the configuration
        # working as intended, not a missing implementation, and refusing to run
        # over it meant a bare `csp run` could not start ANY structure_list
        # campaign: the default slice is the whole funnel, `generate` is in it,
        # and the driver stopped on "no implementation registered".  The only
        # way to run one was to know to say `--from screen`.
        #
        # Anything else missing IS a real defect, so it still raises.
        optional = {w for w in missing if w in OPTIONAL_STAGES}
        broken = [w for w in missing if w not in optional]
        if broken:
            raise DriverError(
                f"no implementation registered for stage(s) {broken}. "
                f"Registered: {sorted(by_name)}"
            )
        return [by_name[name] for name in STAGE_ORDER
                if name in wanted and name in by_name]

    # -- the loop ----------------------------------------------------------

    def _retry_db(self, what: str, fn: Callable[[], _T]) -> _T:
        """Run `fn`, waiting out a storage outage rather than dying in one.

        Retries ONLY errors `is_transient_error` recognises. A corrupt database
        is re-raised on the first attempt, because retrying corruption turns a
        clear failure into a slow one.

        Between attempts the connection is dropped and reopened. That is the
        part that matters: after an NFSv4 lease expiry the old file descriptor
        is permanently poisoned, so retrying on the SAME connection fails
        forever however long the wait.
        """
        attempts = len(DB_RETRY_BACKOFF) + 1
        for attempt in range(attempts):
            try:
                return fn()
            except sqlite3.Error as exc:
                if not is_transient_error(exc):
                    raise
                if attempt == attempts - 1:
                    raise DriverError(
                        f"database unreachable through {attempts} attempts over "
                        f"{sum(DB_RETRY_BACKOFF)}s while {what}: {exc}\n"
                        f"  The database is almost certainly INTACT -- this is a "
                        f"storage outage, not corruption.\n"
                        f"  Check the filesystem, then re-run `csp run`."
                    ) from exc
                wait = DB_RETRY_BACKOFF[attempt]
                self.emit(f"  database unavailable while {what} ({exc}); "
                          f"reopening and retrying in {wait}s "
                          f"[{attempt + 1}/{attempts - 1}]")
                self.store.reconnect()
                self._sleep(wait)
        raise AssertionError("unreachable")

    def run(self, *, watch: bool = False) -> list[CycleReport]:
        """Run cycles until there is nothing left, or forever if `watch`.

        SIGTERM and SIGINT set a flag rather than raising, so the loop finishes
        the cycle it is in. A driver killed mid-submission would leave jobs in
        the queue with no rows recording them, which is the one state the
        database cannot recover from by itself.
        """
        reports: list[CycleReport] = []
        self._idle_cycles = 0
        with self._graceful_stop():
            n = 0
            while True:
                n += 1
                # A cycle is re-runnable by construction -- it reconciles from
                # the scheduler and the filesystem before it decides anything --
                # so a cycle lost to a storage blip is retried rather than
                # ending the run (D148).
                report = self._retry_db(f"running cycle {n}",
                                        lambda: self.cycle(n))
                reports.append(report)
                self.emit(report.render())
                self._write_status(report)
                # Only when something moved. It reads every structure, and on a
                # 14,755-structure campaign an idle cycle rewrote an identical
                # table from a full read each time (D144).
                if n == 1 or report.did_something:
                    self._write_structures()

                if self._should_stop(report, n, watch):
                    break
                self._sleep(self.options.interval)
        return reports

    def _should_stop(self, report: CycleReport, n: int, watch: bool) -> bool:
        if self._stop:
            self.emit("stopping: asked to")
            return True
        if self.options.stop_file and self.options.stop_file.exists():
            self.emit(f"stopping: {self.options.stop_file} exists")
            return True
        if self.options.max_cycles is not None and n >= self.options.max_cycles:
            return True
        if self.options.dry_run:
            # A dry run claims nothing, so `pending` is the same next cycle and
            # the loop below would never see the work run out: it slept a full
            # interval and reprinted the identical report, forever, which reads
            # as a hang rather than as a report. One cycle is the whole answer.
            return True
        if not watch and not self._work_remains(report):
            return True

        # Auto-exit under --watch.  A watched driver is a batch job, and a batch
        # job that idles after its slice is finished holds an allocation for
        # days doing nothing -- which is what `--watch` cost us before this:
        # the Phase A driver for CeFeB sat at "cycle 5 ... cycle 6 ..." with
        # every stage at pending 0 and nothing in flight.
        #
        # "Idle" is deliberately strict: nothing left to do in ANY stage of the
        # slice, nothing in flight, and the cycle itself did nothing.  Requiring
        # it `idle_exit_after` times in a row guards the one real race -- a job
        # that has left the queue but whose results have not been reconciled
        # yet, which shows up as a single idle cycle followed by a busy one.
        if self.options.idle_exit_after and watch:
            if self._work_remains(report) or report.did_something:
                self._idle_cycles = 0
            else:
                self._idle_cycles += 1
                if self._idle_cycles >= self.options.idle_exit_after:
                    self.emit(
                        f"stopping: nothing left to do and nothing in flight "
                        f"for {self._idle_cycles} consecutive cycles. The slice "
                        f"is finished -- rerun this driver to pick up new work.")
                    return True
        return False

    def _work_remains(self, report: CycleReport) -> bool:
        return report.in_flight > 0 or any(s.pending for s in report.stages)

    def cycle(self, n: int = 1) -> CycleReport:
        """Reconcile what is in flight, then submit what fits."""
        report = CycleReport(cycle=n)

        report.orphans = self.store.orphan_jobs()
        report.reconciled = self._reconcile()
        self._forget_finished_claims()
        spent, projected, in_flight = self._accounting()
        report.core_hours_spent = spent
        report.core_hours_projected = projected
        report.in_flight = in_flight
        report.in_flight_detail = list(getattr(self, "_in_flight_detail", []))

        for stage in self.stages:
            report.stages.append(self._advance(stage, report))

        # Re-read after acting, so the report describes the end of the cycle
        # rather than its beginning. The pre-cycle numbers above are what the
        # decisions were made on; these are what actually happened.
        spent, projected, in_flight = self._accounting()
        report.core_hours_spent = spent
        report.core_hours_projected = projected
        report.in_flight = in_flight
        report.in_flight_detail = list(getattr(self, "_in_flight_detail", []))
        return report

    # -- reconciliation ----------------------------------------------------

    def _reconcile(self) -> int:
        """Poll this driver's non-terminal jobs and fold the answer back in.

        THIS DRIVER'S. A campaign is routinely run as two drivers -- one over
        `--through reference` and one over `--from dft` -- and both see the same
        job table. Polling all of it meant whichever driver got there first
        marked a job `done`, and if that driver did not own the job's stage,
        `_hand_back` then returned silently: no record written, no structure
        advanced, and the row now in a terminal state that `_reconcile` never
        looks at again. The results sat on disk, finished and unread.

        Measured on CeFeB, where `CeFeB-pre` (--through reference) and
        `CeFeB-dft` (--from filter) overlapped for 13.7 hours:

            relax tasks finishing DURING the overlap : 42, of which 30 lost
            relax tasks finishing after it ended     : 13, of which  0 lost

        Those 30 structures sat in `dft_queued` with a converged relaxation on
        disk and no way back -- `dft_queued` was frozen at 60 for eleven cycles
        while everything around it moved.
        """
        mine = {stage.name for stage in self.stages}
        live = [j for j in self.store.jobs()
                if j["state"] in {"queued", "running", "held"} and j["stage"] in mine]
        if not live:
            return 0

        # One array submission produces MANY job rows sharing one slurm_id, so
        # this is a list per id, not a row per id. Keyed as a dict it silently
        # kept only the last row and left the rest queued forever.
        by_id: dict[str, list] = {}
        for job in live:
            if job["slurm_id"]:
                by_id.setdefault(job["slurm_id"], []).append(job)
        if not by_id:
            return 0

        ids = sorted(by_id)
        statuses = self.scheduler.poll(ids)
        per_task = self._poll_tasks(ids)

        n = 0
        for slurm_id, rows in by_id.items():
            n += self._fold(slurm_id, rows, per_task, statuses.get(slurm_id))
        return n

    def _poll_tasks(self, ids: list[str]) -> dict:
        """Task-level statuses, or nothing if this scheduler has none.

        Optional on the protocol so a scheduler that does not model arrays --
        or an older stub in a test -- keeps working on the whole-submission
        path rather than failing.
        """
        poll_tasks = getattr(self.scheduler, "poll_tasks", None)
        if poll_tasks is None:
            return {}
        return poll_tasks(ids) or {}

    def _fold(self, slurm_id: str, rows: list, per_task: dict,
              aggregate) -> int:
        """One submission's rows, reconciled per task wherever SLURM has one.

        A task that has ended frees its own slot straight away and hands back
        only its own claim, so a long array releases capacity as it drains
        instead of all at once when its slowest task ends (D113). Rows with no
        task-level answer -- a single-job submission, a scheduler that reports
        none, or rows written before the task id was recorded -- fall through
        to the whole-submission path unchanged.
        """
        n = 0
        leftover = []
        for job in rows:
            idx = job["array_task_id"]
            status = per_task.get(f"{slurm_id}_{idx}") if idx is not None else None
            if status is None or status.state is JobState.unknown:
                leftover.append(job)
                continue
            n += self._apply(job, status, status.core_hours, only_tasks=[idx])
        if leftover:
            n += self._fold_whole(slurm_id, leftover, aggregate)
        return n

    def _fold_whole(self, slurm_id: str, rows: list, status) -> int:
        """The pre-task behaviour: one answer for the whole submission."""
        if status is None or status.state is JobState.unknown:
            # Deliberately not touched. A job neither squeue nor sacct
            # remembers has NOT been shown to have succeeded, and writing
            # 'done' here is how an outage becomes a campaign reported
            # complete (D043).
            return 0
        # Core-hours are for the submission as a whole; splitting them
        # across its rows keeps the campaign total honest whether the work
        # went out as one array or as individual jobs.
        share = status.core_hours / len(rows) if rows else 0.0
        n = 0
        # The job's new state and the results folded from it commit together
        # (D144) -- see `_apply`.
        with self.store.transaction():
            for job in rows:
                if status.state.terminal or status.state.value != job["state"]:
                    self.store.update_job(
                        job["id"], state=status.state.value,
                        core_hours=share,
                        exit_reason=status.reason or status.raw_state,
                        remedy=status.remedy().value,
                    )
                    n += 1
            if status.state.terminal:
                # Only the claims of the rows actually handled here -- a sibling
                # already reconciled per task must not be handed back twice.
                idxs = [j["array_task_id"] for j in rows if j["array_task_id"] is not None]
                self._hand_back(rows[0], status, only_tasks=idxs or None)
        return n

    def _apply(self, job, status, share: float, only_tasks=None) -> int:
        """Write one row's outcome, and hand its claim back if it is finished.

        ONE TRANSACTION (D144). The terminal state and every result folded from
        the job commit together or not at all. Before, the job was committed
        `done` first and its 500 structures were then written one commit at a
        time -- an fsync round trip apiece on NFS -- and a driver killed in
        between left a job marked finished whose results were never read, and
        which `_reconcile` would never look at again. Now it is simply
        reconciled again next cycle.
        """
        n = 0
        with self.store.transaction():
            if status.state.terminal or status.state.value != job["state"]:
                self.store.update_job(
                    job["id"], state=status.state.value,
                    core_hours=share,
                    exit_reason=status.reason or status.raw_state,
                    remedy=status.remedy().value,
                )
                n = 1
            if status.state.terminal:
                self._hand_back(job, status, only_tasks=only_tasks)
        return n

    def _hand_back(self, job, status: JobStatus, only_tasks=None) -> None:
        stage = next((s for s in self.stages if s.name == job["stage"]), None)
        if stage is None:
            # Unreachable now that `_reconcile` filters to this driver's stages,
            # and kept loud rather than deleted: reaching it means a job was
            # marked terminal by a driver that cannot read its results, which is
            # silent data loss and took a 30-structure hole to notice.
            self.emit(f"  WARNING: {job['stage']} job {job['slurm_id']} was polled by a "
                      f"driver that does not run that stage; its results are on disk "
                      f"and unread. Reconcile it with a driver whose slice includes "
                      f"{job['stage']}.")
            return
        slurm_id = str(job["slurm_id"])
        # Read, not pop: with per-task reconciliation the same submission is
        # handed back many times, once per task, and popping on the first would
        # leave every later task without its claim.  The entry is dropped when
        # the submission has no live rows left.
        items = self._claims.get(slurm_id)
        if items is None:
            items = _read_claim(Path(job["workdir"]), slurm_id)
        if items is None:
            self.emit(f"  cannot reconcile {job['stage']} job {slurm_id}: no claim "
                      f"record in memory or in {job['workdir']}. Its results are on "
                      f"disk and unread.")
            return
        # Stamp the array position BEFORE any filtering: after it, a stage can
        # no longer recover the index from the list, and reading the wrong
        # task's results file is silent (D121).
        for position, item in enumerate(items):
            if item.task_index is None:
                item.task_index = position
        if only_tasks is not None:
            items = [items[i] for i in only_tasks if 0 <= i < len(items)]
            if not items:
                return
        stage.reconcile(self.store, job, status, items)

    def _forget_finished_claims(self) -> None:
        """Drop in-memory claims for submissions with nothing left in flight.

        `_hand_back` no longer pops, so without this the dict grows for the
        life of the driver -- which, now that the driver is a batch job with a
        multi-day walltime, is long enough to matter.
        """
        if not self._claims:
            return
        live = {str(j["slurm_id"]) for j in self.store.jobs()
                if j["state"] in {"queued", "running", "held"} and j["slurm_id"]}
        for slurm_id in [k for k in self._claims if k not in live]:
            del self._claims[slurm_id]

    # -- status file -------------------------------------------------------

    def _write_structures(self) -> None:
        """`structures.csv`: every structure, wherever it got to, refreshed
        each cycle beside `status.json`.

        `status.json` answers "how many are where"; this answers "what happened
        to THIS one", for all of them at once. Before it, that question needed
        `csp status --why <id>` one id at a time, which is fine for a structure
        you already suspect and useless for finding one you do not.

        Never fatal. A table is a convenience; the database is the record, and
        a campaign that stopped because a CSV could not be written would be a
        worse campaign.
        """
        try:
            from .report.structures import write as write_structures

            write_structures(self.store, self.cfg.work_dir / "structures.csv")
        except Exception as exc:                               # noqa: BLE001
            self.emit(f"  could not write structures.csv: {exc}")

    def _write_status(self, report: CycleReport) -> None:
        """Current progress, as JSON, refreshed every cycle.

        The log is a running narrative and answering "where is this up to?"
        from it means reading backwards for the last cycle that said anything.
        This is the same numbers as one flat object that is always current, so
        a person or a script can read progress with `cat` and never has to open
        the campaign database -- which, being a relative path, is easy to open
        in the wrong place and get an empty answer from.

        Written to a temporary file and renamed, because a reader is likely to
        be a `watch cat` running at the same time and a half-written file would
        show up as a parse error at random.
        """
        path = self.cfg.work_dir / "status.json"
        counts: dict[str, int] = {}
        for state in ("selected", "dft_queued", "dft_done", "failed", "done"):
            try:
                n = len(self.store.structure_ids(state=state))
            except Exception:                                 # pragma: no cover
                continue
            if n:
                counts[state] = n

        payload = {
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "campaign": str(self.cfg.work_dir),
            "cycle": report.cycle,
            "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
            "pid": os.getpid(),
            # The host is not decoration. A WAL is coherent only within one
            # host, so this name is what tells a later reader whether ITS view
            # of the database is the real one or a checkpoint-old copy. On this
            # host the database is authoritative; anywhere else it is not.
            "host": socket.gethostname(),
            "in_flight": report.in_flight,
            "reconciled": report.reconciled,
            "core_hours_spent": round(report.core_hours_spent, 1),
            "core_hours_projected": round(report.core_hours_projected, 1),
            "structures": counts,
            "structures_total": sum(counts.values()),
            "stages": [
                {"stage": st.stage, "pending": st.pending, "claimed": st.claimed,
                 "submitted": st.submitted, "note": st.note}
                for st in report.stages
            ],
        }
        # The whole of `csp status`, not just the driver's own bookkeeping.
        #
        # The driver is the one process that can always read this database --
        # it is on the node that holds it. Everyone else may not be: a WAL is
        # coherent only within one host, so `csp status` from the login node
        # gets either an exception or, worse, a stale answer that looks fine.
        # Writing the summary here is what lets that command fall back to a
        # file instead of failing, so the cost of one extra query per cycle
        # buys a status command that works from anywhere.
        #
        # Wrapped because a status file must never be able to stop a campaign:
        # if the summary query fails, the snapshot simply carries less.
        try:
            payload["summary"] = self.store.summary()
            payload["generation"] = self.store.generation_yield()
        except Exception as exc:                              # noqa: BLE001
            payload["summary_error"] = str(exc)[:200]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n")
            tmp.replace(path)
        except OSError as exc:                                # pragma: no cover
            # Never let a status file stop a campaign.
            self.emit(f"  could not write {path}: {exc}")

    # -- accounting --------------------------------------------------------

    def _accounting(self) -> tuple[float, float, int]:
        """Core-hours spent, core-hours projected, and jobs in flight.

        All three are REPORTING, not policy: nothing here holds a submission
        back. Spend is what finished jobs actually cost; the projection adds an
        estimate for what is still out, so the log says where the run is headed
        as well as where it has been. What throttles the run is `dft.max_cores`
        against `cores`, which is measured rather than projected (D142/D143).
        """
        spent = 0.0
        projected = 0.0
        in_flight = 0
        cores = 0
        groups: dict[tuple, list] = {}
        for job in self.store.jobs():
            if job["state"] in {"queued", "running", "held"}:
                in_flight += 1
                # Cores held by work that is queued OR running: both will occupy
                # the allocation, so both count against `dft.max_cores` (D142).
                try:
                    cores += int(job["cores"] or 0)
                except (IndexError, KeyError, TypeError):
                    pass                      # a row written before the column
                projected += self._estimate(job)
                # One SLURM job can hold many structures -- a `dft` array has a
                # row per structure, all sharing one slurm_id -- so group before
                # describing. Ungrouped, a 7-task array printed seven identical
                # lines and a 20-job cycle printed twenty.
                groups.setdefault((job["stage"], job["slurm_id"]), []).append(job)
            else:
                spent += float(job["core_hours"] or 0.0)
        self._in_flight_detail = self._describe_groups(groups)
        self._cores_in_flight = cores
        return spent, spent + projected, in_flight

    # At most this many in-flight lines per cycle; the rest are counted.
    MAX_DETAIL_LINES = 6

    def _describe_groups(self, groups: dict) -> list[str]:
        lines = [self._describe_job(rows[0], tasks=len(rows))
                 for rows in groups.values()]
        if len(lines) <= self.MAX_DETAIL_LINES:
            return lines
        hidden = len(lines) - self.MAX_DETAIL_LINES
        return lines[:self.MAX_DETAIL_LINES] + [f"... and {hidden} more job(s) in flight"]

    def _describe_job(self, job, tasks: int = 1) -> str:
        """One line saying what an in-flight job is, and how far along it is.

        The driver cannot read another allocation's stdout, so progress has to
        come off shared disk: the screen worker writes a small `.progress.json`
        beside its results file after every structure. When there is no progress
        file -- an older worker, a job still queueing, a stage that does not
        write one -- the line still carries the job id, the stage, and how long
        it has been in flight, which is more than "in flight 1" ever said.
        """
        import json as _json
        from datetime import datetime, timezone

        stage = job["stage"]
        slurm = job["slurm_id"] or "(unsubmitted)"
        bits = [f"{stage:<11} job {slurm}  {job['state']}"]
        if tasks > 1:
            bits.append(f"{tasks} structures")

        started = job["submitted_at"]
        if started:
            try:
                then = datetime.fromisoformat(started)
                if then.tzinfo is None:
                    then = then.replace(tzinfo=timezone.utc)
                mins = (datetime.now(timezone.utc) - then).total_seconds() / 60
                bits.append(f"{mins:.0f} min in flight")
            except ValueError:
                pass

        workdir = Path(job["workdir"] or "")
        # `batches/<tag>/progress-task<N>.json` since D136; `*.progress.json`
        # flat in the workdir before it. Both are globbed, because a driver that
        # loses sight of progress reports a healthy job as silent.
        progress = sorted(workdir.glob("batches/*/progress-task*.json")) \
            + sorted(workdir.glob("*.progress.json")) if workdir.is_dir() else []
        for path in progress:
            try:
                p = _json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            done, total = p.get("done", 0), p.get("total", 0)
            if not total:
                continue
            pct = 100 * done / total
            bits.append(f"{done}/{total} ({pct:.0f}%)  "
                        f"{p.get('converged', 0)} converged, {p.get('failed', 0)} failed")
        if str(workdir) not in {"", "."}:
            bits.append(str(workdir))
        return "  ".join(bits)

    def _estimate(self, job) -> float:
        """What an in-flight job will probably cost.

        Uses the mean of what finished jobs of the same stage actually cost --
        this campaign's own history, not a guess. Falls back to the requested
        walltime x ntasks, which over-estimates by however much of its walltime
        a job does not use. Read the projection with that in mind: before any
        job of a stage has finished it is a ceiling, not a forecast. Nothing
        decides anything on it (D143).
        """
        recorded = float(job["core_hours"] or 0.0)
        if recorded:
            return recorded
        row = self.store.sql.execute(
            "SELECT AVG(core_hours) AS mean FROM job "
            "WHERE stage=? AND state='done' AND core_hours > 0",
            (job["stage"],),
        ).fetchone()
        if row and row["mean"]:
            return float(row["mean"])
        return self._walltime_estimate(job["stage"])

    def _walltime_estimate(self, stage: str) -> float:
        """Core-hours one item of `stage` might cost, before any have finished.

        Asked of the stage where it can answer. `dft` was missing from the table
        this used to be, so it fell through to the machine default and a bare
        24-hour walltime -- 384 core-hours against a recipe asking for 32, a 12x
        over-estimate. That mattered while a budget gated on this number; now it
        only skews a reported projection (D143).
        """
        implementation = next((s for s in self.stages if s.name == stage), None)
        hint = getattr(implementation, "resource_hint", None)
        if hint is not None:
            ntasks, walltime = hint()
        else:
            resources = {
                "generate": (self.cfg.campaign.generate.resources
                             if self.cfg.campaign.generate else None),
                "screen": self.cfg.campaign.screen.resources,
            }.get(stage)
            ntasks = (resources.ntasks if resources and resources.ntasks
                      else self.cfg.machine.defaults.ntasks)
            walltime = resources.time if resources else "24:00:00"
        return _walltime_hours(walltime) * (ntasks or self.cfg.machine.defaults.ntasks)

    # -- submission --------------------------------------------------------

    def _advance(self, stage: Stage, report: CycleReport) -> StageReport:
        pending = stage.pending(self.store)
        out = StageReport(stage=stage.name, pending=pending)
        if not pending:
            return out

        if stage.in_process:
            if self.options.dry_run:
                out.note = "dry-run: not executed"
                return out
            # Not wrapped in one transaction: an in-process stage can run for
            # minutes, and holding the write lock that long starves a second
            # driver on the same campaign (D129). The stages commit in their
            # own units -- one chemical system, one chunk (D144).
            done = stage.run(self.store)
            out.claimed = done.claimed
            out.reconciled = done.reconciled
            out.note = done.note
            # Re-read: `pending` was measured before the stage ran, and the loop
            # decides whether to sleep by asking whether work remains. Left
            # stale, an in-process stage that finished its work in this very
            # cycle still reported it as pending, and the driver slept a full
            # interval before noticing it was done.
            out.pending = stage.pending(self.store)
            return out

        # How much goes out this cycle is the throttle's answer and nothing
        # else's. It was the smaller of the throttle and what a core-hour
        # budget could still afford; the budget is retired (D143) because its
        # estimate of "afford" was the requested walltime until the first job
        # of a stage finished, and that held back runs that had spent nothing.
        throttle = self._throttle(stage, report.in_flight)
        allowed = throttle.in_flight
        if allowed <= 0:
            out.note = f"held: {throttle.binding}"
            return out

        if self.options.dry_run:
            # Claiming marks rows in the database, so a dry run must not do it.
            # The count is derivable without mutating anything -- but `pending`
            # and `allowed` are not in the same units for every stage (screen
            # counts structures and submits chunks; generate counts compositions
            # and submits groups), so the stage is asked rather than assumed.
            out.note = (f"dry-run: would submit {_estimate_tasks(stage, self.store, allowed)} "
                        f"task(s) ({throttle.render()})")
            return out

        # Claimed in one transaction (D144): marking 14,755 structures
        # `screening` one commit at a time took ~55 minutes on NFS before the
        # first job went out. It commits BEFORE anything is submitted, so
        # "rows first, then submit" still holds.
        with self.store.transaction():
            items = stage.claim(self.store, allowed)
        out.claimed = len(items)
        out.pending = stage.pending(self.store)
        if not items:
            return out

        workdir = (self.cfg.work_dir / stage.name).resolve()
        # ONE SUBMISSION PER STRUCTURE, when the stage asks for it (D135).
        #
        # An array is right for screening and generation: those tasks are cheap,
        # near-identical, and batching them is what keeps 200 structures from
        # becoming 200 queue entries. It is wrong for DFT, where each task has
        # its own walltime, its own memory and its own failure -- and an array
        # gives all of them the first task's name and the batch's worst-case
        # allocation.
        #
        # So the stage decides, and the driver submits each group separately.
        # Everything below (rows first, then submit, then stamp) is per group,
        # because a driver that dies between two submissions must leave the
        # first one findable.
        groups = ([[item] for item in items]
                  if getattr(stage, "solo_jobs", False) else [items])
        for group in groups:
            out.submitted += self._submit_group(stage, group, workdir, throttle)
        out.note = throttle.render()
        return out

    def _submit_group(self, stage: Stage, items: list[WorkItem], workdir: Path,
                      throttle) -> int:
        """Write the rows, submit one job, stamp the id back on them."""
        spec = stage.build(items, workdir)
        spec.array_throttle = spec.array_throttle or throttle.concurrent_tasks

        # Rows first, then submit, then stamp the id on the rows.
        #
        # The obvious order -- submit, then record -- has a window in which the
        # job is running and nothing in the database says so, and a driver that
        # dies inside that window leaves work in the queue that no later cycle
        # can find, reconcile, or cancel. Writing the rows first makes the worst
        # case a row in `pending` with no scheduler id: visible in `csp status`,
        # reported by the next cycle, and safe to act on by hand.
        #
        # (Found by killing a driver mid-submission: the structures were marked
        # `screening`, the job script and manifest were on disk, the worker was
        # running, and the job table held nothing at all.)
        # `recipe_step` and `attempt` come from the item's payload where the
        # stage put them. The columns existed and were never written, so every
        # DFT job row read `step='' attempt=0` however far up the ladder it
        # actually was -- which is precisely the history `csp status --why` is
        # for.
        rows = []
        for item in items:
            row = self.store.add_job(
                stage=stage.name,
                structure_id=item.structure_ids[0] if item.structure_ids else None,
                recipe_step=str(item.payload.get("step_name", "")),
                workdir=str(workdir),
                # What this job will actually occupy, from the spec that was
                # submitted -- not from the config, which a retry rung may have
                # overridden. `max_cores` is only as honest as this number.
                cores=int(spec.ntasks or 0) * max(1, int(spec.cpus_per_task or 1)),
            )
            attempt = int(item.payload.get("attempt", 0))
            if attempt:
                self.store.update_job(row, attempt=attempt)
            rows.append(row)
        job_id = self.scheduler.submit(spec)
        # PAST THIS LINE THE JOB IS REAL AND IN THE QUEUE. Everything below is
        # bookkeeping, and losing it is the one state the comment above says the
        # database cannot recover from by itself -- so from here the database is
        # retried rather than allowed to fail (D148). `assert_job_id_is_new` is
        # a read and `update_job` an UPDATE keyed by row id, so both are safe to
        # repeat.
        self._retry_db(
            f"checking job {job_id} is new",
            lambda: self.store.assert_job_id_is_new(str(job_id), stage.name, str(workdir)),
        )
        self._claims[str(job_id)] = items
        # And on disk, because the claim has to outlive this process.
        #
        # `csp run --only generate` then, later, `csp run --only screen` is the
        # documented way to work: submit now, reconcile when the queue gets to
        # it. With the claim only in memory the second process reconciled with
        # an empty item list -- so every stage's loop ran zero times, the job
        # was marked done, and 36 generated structures sat in their extxyz files
        # unread with nothing reporting a problem.
        _write_claim(workdir, str(job_id), items)
        # The array index is the row's position in `items`, because that is the
        # order `stage.build` lays the tasks out in.  Recording it is what lets
        # `_reconcile` ask about one task instead of the whole array (D113); the
        # column existed and was never written.
        #
        # `array_task_id` stays None for a submission that is not an array: the
        # column is what lets `_reconcile` ask about one task of many, and a
        # plain job has no task to distinguish. `Driver._fold` already routes a
        # row with no task id to the whole-submission path.
        def _stamp() -> None:
            for idx, row in enumerate(rows):
                self.store.update_job(row, state="queued", slurm_id=str(job_id),
                                      array_task_id=idx if spec.is_array else None)

        self._retry_db(f"recording job {job_id}", _stamp)
        return len(items)

    def _throttle(self, stage: Stage, already_in_flight: int):
        dft = self.cfg.campaign.dft
        resources = {
            "generate": self.cfg.campaign.generate.resources if self.cfg.campaign.generate else None,
            "screen": self.cfg.campaign.screen.resources,
        }.get(stage.name)
        ntasks = (resources.ntasks if resources and resources.ntasks
                  else self.cfg.machine.defaults.ntasks)
        # DFT resources live per RECIPE STEP, not in the campaign's `dft:`
        # block, so `resources` above is None for this stage and the machine
        # default was used instead. With a recipe at 16 ranks and a default of
        # 64, the QOS cap was reported as cpu=768/64 = 12 concurrent when the
        # real figure was 48 -- a throttle four times tighter than the site's,
        # attributed to the site. The stage is asked, because it is the only
        # thing that can answer.
        hint = getattr(stage, "resource_hint", None)
        if hint is not None:
            try:
                ntasks = int(hint()[0]) or ntasks
            except Exception:                       # noqa: BLE001 - never fatal
                pass
        gpus = resources.gpus if resources and resources.gpus else 0
        try:
            limits = self.scheduler.limits(stage.role)
        except Exception:                               # pragma: no cover - live only
            limits = None
        return compute_throttle(
            requested_in_flight=dft.max_in_flight,
            requested_concurrent=dft.max_concurrent_tasks,
            ntasks=ntasks, gpus_per_job=gpus, limits=limits,
            already_in_flight=already_in_flight,
            max_cores=getattr(dft, "max_cores", None),
            cores_in_flight=getattr(self, "_cores_in_flight", 0),
        )

    # -- stopping ----------------------------------------------------------

    def stop(self) -> None:
        self._stop = True

    class _GracefulStop:
        def __init__(self, driver: "Driver") -> None:
            self.driver = driver
            self.previous: dict[int, object] = {}

        def __enter__(self):
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    self.previous[sig] = signal.signal(sig, self._handler)
                except ValueError:      # not the main thread; tests, mostly
                    pass
            return self

        def _handler(self, *_args) -> None:
            self.driver.stop()

        def __exit__(self, *exc):
            for sig, handler in self.previous.items():
                try:
                    signal.signal(sig, handler)          # type: ignore[arg-type]
                except ValueError:                       # pragma: no cover
                    pass
            return False

    def _graceful_stop(self) -> "Driver._GracefulStop":
        return Driver._GracefulStop(self)


def _walltime_hours(walltime: str) -> float:
    """`'2-00:00:00'` -> 48.0."""
    days, _, clock = walltime.partition("-")
    if not clock:
        days, clock = "0", days
    parts = [float(p) for p in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return int(days) * 24 + parts[-3] + parts[-2] / 60 + parts[-1] / 3600
