"""The driver loop.

Tested against a fake stage and the local scheduler, which is the reason
`LocalScheduler` exists: the driver's interesting behaviour -- resuming,
throttling, holding at a budget, stopping cleanly -- is exactly what is painful
to exercise through a real queue.
"""

from pathlib import Path

import pytest
import yaml

from cspflow.config.loader import load_campaign
from cspflow.db.store import Store
from cspflow.driver import CycleReport, STAGE_ORDER, Driver, DriverError, DriverOptions
from cspflow.scheduler.base import JobSpec, JobState, JobStatus, Limits
from cspflow.stages.base import StageReport, WorkItem

CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: composition_list
    name: list
    composition_list: {{items: [{{formula: FeCo5}}]}}
generate:
  engine: mattergen
  mattergen: {{model: /tmp/model}}
dft:
  max_in_flight: 4
  max_concurrent_tasks: 2
"""


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path))
    return load_campaign(path)


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "c.db", campaign="t") as s:
        yield s


class FakeScheduler:
    """Records what it was asked to do; answers what it is told to answer."""

    def __init__(self, limits: Limits | None = None):
        self.submitted: list[JobSpec] = []
        self.statuses: dict[str, JobStatus] = {}
        self._limits = limits or Limits()
        self._n = 0

    def submit(self, spec: JobSpec) -> str:
        self._n += 1
        self.submitted.append(spec)
        return str(1000 + self._n)

    def poll(self, job_ids):
        return {
            j: self.statuses.get(j, JobStatus(job_id=j, state=JobState.running))
            for j in job_ids
        }

    def cancel(self, job_ids):
        pass

    def limits(self, role):
        return self._limits

    def in_flight(self):
        return 0


class FakeStage:
    """A submitted stage over a fixed pool of structure ids."""

    name = "dft"
    role = "cpu"
    in_process = False

    def __init__(self, work: int = 6):
        self.pool = list(range(1, work + 1))
        self.reconciled: list[tuple] = []
        self.indices: list[list] = []

    def pending(self, store):
        return len(self.pool)

    def claim(self, store, budget):
        taken, self.pool = self.pool[:budget], self.pool[budget:]
        return [WorkItem(key=f"s{i}", structure_ids=[i], est_core_hours=10.0) for i in taken]

    def build(self, items, workdir):
        return JobSpec(name=f"dft-{items[0].key}", stage="dft", workdir=workdir,
                       command="true", array_size=len(items))

    def reconcile(self, store, job_row, status, items):
        self.reconciled.append((job_row["id"], status.state, len(items)))
        self.indices.append([i.task_index for i in items])

    def run(self, store):                                # pragma: no cover
        raise AssertionError("submitted stage must not be run in process")


class FakeSoloStage(FakeStage):
    """A stage that asks for one submission per item, as DftStage does (D135)."""

    solo_jobs = True

    def build(self, items, workdir):
        assert len(items) == 1, "a solo stage must be built one item at a time"
        return JobSpec(name=f"camp-{items[0].key}", stage="dft", workdir=workdir,
                       command="true", array_size=0)


class FakeInProcessStage:
    name = "filter"
    role = "cpu"
    in_process = True

    def __init__(self, work: int = 3):
        self.remaining = work
        self.ran = 0

    def pending(self, store):
        return self.remaining

    def claim(self, store, budget):                      # pragma: no cover
        raise AssertionError("in-process stage must not be claimed against")

    def build(self, items, workdir):                     # pragma: no cover
        raise AssertionError("in-process stage must not be built")

    def reconcile(self, store, job_row, status, items):  # pragma: no cover
        pass

    def run(self, store):
        self.ran += 1
        done, self.remaining = self.remaining, 0
        return StageReport(stage=self.name, claimed=done, reconciled=done)


def driver(cfg, store, scheduler, impls, **opts):
    options = DriverOptions(interval=0, **opts)
    return Driver(cfg, store, scheduler, impls, options, sleep=lambda _s: None)


# --------------------------------------------------------------------------


class TestOrdering:
    def test_stages_run_in_funnel_order(self, cfg, store):
        d = driver(cfg, store, FakeScheduler(),
                   [FakeStage(), FakeInProcessStage()], stages=["dft", "filter"])
        assert [s.name for s in d.stages] == ["filter", "dft"]

    def test_a_stage_outside_the_funnel_is_refused(self, cfg, store):
        class Weird(FakeStage):
            name = "teleport"

        with pytest.raises(DriverError, match="funnel order"):
            driver(cfg, store, FakeScheduler(), [Weird()], stages=["teleport"])

    def test_asking_for_an_unregistered_stage_is_refused(self, cfg, store):
        with pytest.raises(DriverError, match="no implementation"):
            driver(cfg, store, FakeScheduler(), [FakeStage()], stages=["dft", "screen"])

    def test_an_optional_stage_the_config_left_out_is_skipped(self, cfg, store):
        """`generate` is built only when the campaign has a `generate:` block.

        A `source.mode: structure_list` campaign brings its own structures and
        has none -- so a bare `csp run`, whose default slice is the whole funnel,
        used to stop on "no implementation registered for stage(s)
        ['generate']".  Every structure_list campaign was unrunnable unless you
        knew to say `--from screen`.  The stage is skipped now; a stage missing
        for any OTHER reason still raises.
        """
        d = driver(cfg, store, FakeScheduler(),
                   [FakeStage(), FakeInProcessStage()],
                   stages=["generate", "filter", "dft"])
        assert [s.name for s in d.stages] == ["filter", "dft"]

    def test_only_generate_is_optional(self, cfg, store):
        """The skip must not become a way to lose a stage that should be there."""
        from cspflow.driver import OPTIONAL_STAGES

        assert OPTIONAL_STAGES == {"generate"}
        with pytest.raises(DriverError, match="no implementation"):
            driver(cfg, store, FakeScheduler(), [FakeStage()],
                   stages=["dft", "analyze"])

    def test_a_watched_driver_stops_once_the_slice_is_finished(self, cfg, store):
        """`--watch` used to mean forever, so a driver submitted as a batch job
        held its allocation for days after its work was done. The CeFeB Phase A
        driver sat at "cycle 5 ... cycle 6 ..." with every stage at pending 0
        and nothing in flight."""
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=0)],
                   stages=["dft"], idle_exit_after=2)
        reports = d.run(watch=True)
        assert len(reports) == 2          # two idle cycles, then out

    def test_one_idle_cycle_is_not_enough(self, cfg, store):
        """The race this guards: a job has left the queue but its results are
        not reconciled yet, which reads as one idle cycle followed by a busy
        one. Requiring two in a row means a single blip does not end the run."""
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=0)],
                   stages=["dft"], idle_exit_after=2)
        first = d.cycle(1)
        assert not d._should_stop(first, 1, True)      # one idle cycle: keep going

    def test_a_busy_cycle_resets_the_idle_count(self, cfg, store):
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=0)],
                   stages=["dft"], idle_exit_after=2)
        d._idle_cycles = 0
        d._should_stop(d.cycle(1), 1, True)
        assert d._idle_cycles == 1
        busy = CycleReport(cycle=2, in_flight=3)
        assert not d._should_stop(busy, 2, True)
        assert d._idle_cycles == 0

    def test_watch_forever_is_still_available(self, cfg, store):
        """A campaign you intend to keep feeding wants the old behaviour."""
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=0)],
                   stages=["dft"], idle_exit_after=0, max_cycles=3)
        assert len(d.run(watch=True)) == 3       # only max_cycles stopped it

    def test_idle_exit_does_not_fire_without_watch(self, cfg, store):
        """Unwatched runs already stop on their own; the flag must not change
        when, or a `--dry-run` would report a different number of cycles."""
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=0)],
                   stages=["dft"], idle_exit_after=2)
        assert len(d.run(watch=False)) == 1

    def test_the_order_is_the_documented_funnel(self):
        assert STAGE_ORDER[0] == "source" and STAGE_ORDER[-1] == "analyze"
        assert STAGE_ORDER.index("screen") < STAGE_ORDER.index("dft")


class TestSubmission:
    def test_claims_and_submits_up_to_the_throttle(self, cfg, store):
        sched = FakeScheduler()
        stage = FakeStage(work=10)
        d = driver(cfg, store, sched, [stage], stages=["dft"])
        report = d.cycle()
        # max_in_flight is 4 in the fixture
        assert report.stages[0].submitted == 4
        assert len(sched.submitted) == 1
        assert len(stage.pool) == 6

    def test_a_solo_stage_gets_one_submission_per_structure(self, cfg, store):
        """D135. An array is one allocation, one name, one walltime and one
        `--mem` shared by every task in it -- all four wrong for DFT, where each
        structure has its own. The stage decides; the driver obeys."""
        sched = FakeScheduler()
        stage = FakeSoloStage(work=10)
        d = driver(cfg, store, sched, [stage], stages=["dft"])
        report = d.cycle()

        assert report.stages[0].submitted == 4          # max_in_flight in the fixture
        assert len(sched.submitted) == 4, "the batch went out as one array"
        assert all(spec.array_size == 0 for spec in sched.submitted)
        assert [spec.name for spec in sched.submitted] == [
            "camp-s1", "camp-s2", "camp-s3", "camp-s4"]
        # Four separate ids, so a job can be cancelled or reconciled on its own.
        assert len({j["slurm_id"] for j in store.jobs()}) == 4

    def test_a_solo_submission_records_no_array_task_id(self, cfg, store):
        """The column is what lets `_reconcile` ask about one task OF MANY. A
        plain job has no task to distinguish, and a stale 0 there would send it
        down the per-task path asking sacct about `<jobid>_0`, which does not
        exist."""
        d = driver(cfg, store, FakeScheduler(), [FakeSoloStage(work=2)],
                   stages=["dft"])
        d.cycle()
        assert all(j["array_task_id"] is None for j in store.jobs())

    def test_a_batching_stage_is_left_alone(self, cfg, store):
        """Screening and generation batch on purpose: the tasks are cheap and
        near-identical, and one array is what keeps 200 structures from becoming
        200 queue entries."""
        sched = FakeScheduler()
        d = driver(cfg, store, sched, [FakeStage(work=10)], stages=["dft"])
        d.cycle()
        assert len(sched.submitted) == 1
        assert sched.submitted[0].array_size == 4
        assert all(j["array_task_id"] is not None for j in store.jobs())

    def test_job_rows_are_written_for_every_item(self, cfg, store):
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=10)], stages=["dft"])
        d.cycle()
        assert store.count_jobs_by_state("dft") == {"queued": 4}

    def test_the_array_throttle_comes_from_the_qos(self, cfg, store):
        sched = FakeScheduler(Limits(max_submit=2048, max_cpus=32))
        d = driver(cfg, store, sched, [FakeStage(work=10)], stages=["dft"])
        d.cycle()
        # cpu=32 at the machine default ntasks=16 permits 2 concurrent
        assert sched.submitted[0].array_throttle == 2

    def test_nothing_pending_means_nothing_submitted(self, cfg, store):
        sched = FakeScheduler()
        d = driver(cfg, store, sched, [FakeStage(work=0)], stages=["dft"])
        assert d.cycle().stages[0].submitted == 0
        assert not sched.submitted

    def test_dry_run_claims_nothing_and_submits_nothing(self, cfg, store):
        sched = FakeScheduler()
        stage = FakeStage(work=10)
        d = driver(cfg, store, sched, [stage], stages=["dft"], dry_run=True)
        report = d.cycle()
        assert not sched.submitted and len(stage.pool) == 10
        assert "dry-run" in report.stages[0].note

    def test_in_process_stage_runs_here_and_now(self, cfg, store):
        stage = FakeInProcessStage(work=3)
        d = driver(cfg, store, FakeScheduler(), [stage], stages=["filter"])
        report = d.cycle()
        assert stage.ran == 1 and report.stages[0].claimed == 3

    def test_in_flight_work_is_subtracted_from_the_throttle(self, cfg, store):
        sched = FakeScheduler()
        stage = FakeStage(work=10)
        d = driver(cfg, store, sched, [stage], stages=["dft"])
        d.cycle()                       # submits 4, all stay 'queued'
        second = d.cycle()
        # max_in_flight 4 minus 4 already queued leaves nothing
        assert second.stages[0].submitted == 0
        assert "held" in second.stages[0].note


class TestReconciliation:
    def _submitted(self, cfg, store):
        sched = FakeScheduler()
        stage = FakeStage(work=4)
        d = driver(cfg, store, sched, [stage], stages=["dft"])
        d.cycle()
        return d, sched, stage

    def test_a_finished_job_updates_its_row(self, cfg, store):
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.done,
                                        elapsed_seconds=3600, alloc_cpus=16,
                                        raw_state="COMPLETED")
        d.cycle()
        assert store.count_jobs_by_state("dft") == {"done": 4}

    def test_the_stage_is_handed_the_result(self, cfg, store):
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.done,
                                        raw_state="COMPLETED")
        d.cycle()
        assert stage.reconciled and stage.reconciled[0][1] is JobState.done

    def test_an_unknown_job_is_left_alone(self, cfg, store):
        """Neither squeue nor sacct remembers it -- that is not success."""
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.unknown)
        d.cycle()
        assert store.count_jobs_by_state("dft") == {"queued": 4}
        assert not stage.reconciled

    def test_a_timeout_records_its_remedy(self, cfg, store):
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.timeout,
                                        raw_state="TIMEOUT")
        d.cycle()
        assert store.jobs()[0]["remedy"] == "more_walltime"

    def test_command_not_found_is_marked_do_not_retry(self, cfg, store):
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.failed,
                                        exit_code=127, raw_state="FAILED")
        d.cycle()
        assert store.jobs()[0]["remedy"] == "do_not_retry"

    def test_core_hours_are_split_across_an_arrays_rows(self, cfg, store):
        """One submission, four rows: the campaign total must still be right."""
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.done,
                                        elapsed_seconds=7200, alloc_cpus=8,
                                        raw_state="COMPLETED")
        d.cycle()
        rows = store.jobs()
        assert len(rows) == 4
        assert sum(r["core_hours"] for r in rows) == pytest.approx(16.0)

    def test_every_row_of_an_array_is_updated_not_just_one(self, cfg, store):
        """Keyed as a dict on slurm_id, three of four rows stayed queued forever."""
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.done,
                                        raw_state="COMPLETED")
        d.cycle()
        assert {r["state"] for r in store.jobs()} == {"done"}


class TestAccounting:
    """Core-hours are REPORTED here, never enforced (D143).

    They used to gate submission. The gate was removed because it fired on a
    projection: with no finished job of a stage to average, the per-item
    estimate is the requested WALLTIME, so a run that had spent nothing was
    held as though it had overspent.
    """

    def test_the_projection_counts_queued_work_not_only_finished(self, cfg, store):
        """Reporting spend alone would say 0 while thousands of cores are out."""
        sched = FakeScheduler()
        d = driver(cfg, store, sched, [FakeStage(work=10)], stages=["dft"])
        first = d.cycle()
        assert first.core_hours_spent == 0.0
        assert first.core_hours_projected > 0.0

    def test_a_huge_projection_does_not_hold_submission(self, cfg, store):
        """The regression this removal fixes.

        `FakeStage` work is priced at the requested walltime, so one cycle
        projects far more than any small budget would have allowed. Before D143
        that printed "BUDGET REACHED" and submitted nothing; now the throttle is
        the only thing that can hold work back.
        """
        sched = FakeScheduler()
        d = driver(cfg, store, sched, [FakeStage(work=100)], stages=["dft"])
        report = d.cycle()
        assert sched.submitted, "nothing was submitted, so something still gates on cost"
        assert report.core_hours_projected > 0
        assert "budget" not in report.render().lower()
        assert report.stages[0].submitted == 4      # max_in_flight, not a budget

    def test_the_setting_is_gone_from_the_schema(self):
        from cspflow.config.schema import Select

        assert "budget_core_hours" not in Select.model_fields

    def test_an_old_campaign_that_sets_it_still_loads(self):
        """`extra="forbid"` would otherwise refuse every file written before
        today. It is dropped with a warning instead."""
        import warnings

        from cspflow.config.schema import Select

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            block = Select(max_total=7, budget_core_hours=20000)
        assert not hasattr(block, "budget_core_hours")
        assert block.max_total == 7, "the rest of the block must survive"
        assert any("budget_core_hours" in str(w.message) for w in caught)


class TestLoop:
    def test_stops_when_no_work_remains(self, cfg, store):
        """One cycle: the stage runs, reports 0 pending, and the loop concludes."""
        stage = FakeInProcessStage(work=2)
        d = driver(cfg, store, FakeScheduler(), [stage], stages=["filter"])
        reports = d.run(watch=False)
        assert len(reports) == 1 and stage.ran == 1

    def test_max_cycles_caps_the_loop(self, cfg, store):
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=1000)],
                   stages=["dft"], max_cycles=3)
        assert len(d.run(watch=True)) == 3

    def test_a_stop_file_ends_the_loop(self, cfg, store, tmp_path):
        stop = tmp_path / "STOP"
        stop.write_text("")
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=1000)],
                   stages=["dft"], max_cycles=99, stop_file=stop)
        assert len(d.run(watch=True)) == 1

    def test_stop_ends_the_loop_at_a_cycle_boundary(self, cfg, store):
        sched = FakeScheduler()
        stage = FakeStage(work=1000)
        d = driver(cfg, store, sched, [stage], stages=["dft"], max_cycles=99)

        original = stage.claim

        def claim_then_stop(store_, budget):
            d.stop()
            return original(store_, budget)

        stage.claim = claim_then_stop
        reports = d.run(watch=True)
        assert len(reports) == 1
        assert len(sched.submitted) == 1     # the cycle it was in still completed

    def test_reports_render_without_error(self, cfg, store):
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=5)], stages=["dft"])
        text = d.cycle().render()
        assert "cycle 1" in text and "dft" in text


# --------------------------------------------------------------------------
# The source stage, and the registry
# --------------------------------------------------------------------------


class TestSourceStage:
    def test_is_in_process(self, cfg, store):
        from cspflow.stages import SourceStage

        assert SourceStage(cfg).in_process is True

    def test_writes_composition_rows(self, cfg, store):
        from cspflow.stages import SourceStage

        stage = SourceStage(cfg)
        assert stage.pending(store) == 1
        report = stage.run(store)
        assert report.claimed == 1                # FeCo5 at Z=1
        assert store.chemsystems() == ["Co-Fe"]

    def test_stops_being_pending_once_it_has_run(self, cfg, store):
        """Otherwise the driver never concludes the campaign is finished."""
        from cspflow.stages import SourceStage

        stage = SourceStage(cfg)
        stage.run(store)
        assert stage.pending(store) == 0

    def test_the_driver_runs_it_end_to_end(self, cfg, store):
        from cspflow.stages import SourceStage

        d = driver(cfg, store, FakeScheduler(), [SourceStage(cfg)], stages=["source"])
        reports = d.run(watch=False)
        assert store.compositions()
        assert reports[0].stages[0].claimed == 1

    def test_the_loop_does_not_sleep_after_finishing(self, cfg, store):
        """`pending` was measured before the stage ran, so a finished stage
        still looked pending and the driver slept a full interval."""
        from cspflow.stages import SourceStage

        d = driver(cfg, store, FakeScheduler(), [SourceStage(cfg)], stages=["source"])
        report = d.cycle()
        assert report.stages[0].pending == 0

    def test_registry_lists_only_what_is_implemented(self, cfg):
        from cspflow.stages import IMPLEMENTED, PLANNED, build_registry

        names = [s.name for s in build_registry(cfg)]
        assert names == IMPLEMENTED
        assert set(names).isdisjoint(PLANNED)
        assert set(IMPLEMENTED) | set(PLANNED) == set(STAGE_ORDER)


# -- dry runs --------------------------------------------------------------

def test_a_dry_run_stops_after_one_cycle(cfg, store):
    """A dry run claims nothing, so `pending` is identical next cycle.

    Without this the loop never sees the work run out: it sleeps a whole
    interval and reprints the same report, forever, which reads as a hang
    rather than as a report. Found by running the command -- every unit test
    passed `max_cycles`, which is exactly what masked it.
    """
    sched = FakeScheduler()
    stage = FakeStage(work=6)
    d = driver(cfg, store, sched, [stage], stages=["dft"], dry_run=True)
    reports = d.run(watch=False)
    assert len(reports) == 1
    assert sched.submitted == []
    assert stage.pending(store) == 6              # nothing was claimed


def test_a_dry_run_counts_tasks_not_pending_units(cfg, store):
    """`pending` and `budget` are in different units for a batching stage."""

    class Batching(FakeStage):
        name = "screen"

        def estimate_tasks(self, store, budget):
            return min(budget, (len(self.pool) + 499) // 500)

    sched = FakeScheduler()
    d = driver(cfg, store, sched, [Batching(work=1200)], stages=["screen"],
               dry_run=True)
    report = d.cycle(1)
    screen = next(s for s in report.stages if s.stage == "screen")
    assert "pending 1200" in screen.render()
    assert "would submit 3 task(s)" in screen.note


def test_without_estimate_tasks_the_count_is_one_per_pending_unit(cfg, store):
    sched = FakeScheduler()
    d = driver(cfg, store, sched, [FakeStage(work=3)], stages=["dft"], dry_run=True)
    note = d.cycle(1).stages[0].note
    assert "would submit 3 task(s)" in note


# -- submitting is two writes, in the safe order ---------------------------

def test_job_rows_exist_before_the_scheduler_is_called(cfg, store):
    """Submit-then-record has a window where a job is running and nothing says
    so; a driver killed inside it leaves work that no cycle can find.

    Found by killing a driver mid-submission: the structures were marked
    `screening`, the job script and manifest were on disk, the worker was
    running, and the job table held nothing at all.
    """
    seen: dict[str, int] = {}

    class Watching(FakeScheduler):
        def submit(self, spec):
            seen["rows_at_submit"] = len(store.jobs())
            return super().submit(spec)

    d = driver(cfg, store, Watching(), [FakeStage(work=2)], stages=["dft"],
               max_cycles=1)
    d.cycle(1)
    assert seen["rows_at_submit"] == 2
    assert all(j["slurm_id"] for j in store.jobs())


def test_an_orphaned_row_is_reported_every_cycle(cfg, store):
    row = store.add_job(stage="screen", workdir="/w")
    assert store.orphan_jobs() == 1
    d = driver(cfg, store, FakeScheduler(), [FakeStage(work=0)], stages=["dft"],
               max_cycles=1)
    report = d.cycle(1)
    assert report.orphans == 1
    assert "no scheduler id" in report.render()
    store.update_job(row, state="queued", slurm_id="7")
    assert store.orphan_jobs() == 0


# -- claims outlive the process that made them -----------------------------

def test_a_second_driver_can_reconcile_the_first_ones_job(cfg, store, tmp_path):
    """`csp run --only generate` then, later, `csp run --only screen` is the
    documented way to work: submit now, reconcile when the queue gets to it.

    With the claim only in memory the second process reconciled with an empty
    item list, so every stage's loop ran zero times, the job was marked done,
    and the results sat on disk unread with nothing reporting a problem. Found
    on a live submission: 36 generated structures, none ingested.
    """
    sched = FakeScheduler()
    first = driver(cfg, store, sched, [FakeStage(work=2)], stages=["dft"])
    first.cycle(1)
    job_id = sched.submitted and "1001"
    sched.statuses[job_id] = JobStatus(job_id=job_id, state=JobState.done,
                                       raw_state="COMPLETED")

    # A brand-new driver: nothing in memory, everything in the database.
    second_stage = FakeStage(work=0)
    second = driver(cfg, store, sched, [second_stage], stages=["dft"])
    assert second._claims == {}
    second.cycle(2)
    assert second_stage.reconciled, "the second process reconciled with nothing"
    assert second_stage.reconciled[0][2] == 2      # both items came back


def test_a_job_with_no_claim_record_says_so_rather_than_reconciling_empty(cfg, store,
                                                                          tmp_path):
    sched = FakeScheduler()
    d = driver(cfg, store, sched, [FakeStage(work=1)], stages=["dft"])
    d.cycle(1)
    # Remove the claim file, simulating a workdir that was cleaned up.
    for path in Path(cfg.campaign.workdir).rglob("claim-*.json"):
        path.unlink()
    sched.statuses["1001"] = JobStatus(job_id="1001", state=JobState.done,
                                       raw_state="COMPLETED")
    lines = []
    second = Driver(cfg, store, sched, [FakeStage(work=0)],
                    DriverOptions(interval=0, stages=["dft"]),
                    sleep=lambda _s: None, emit=lines.append)
    second.cycle(2)
    assert any("no claim record" in line for line in lines)


def test_the_job_row_records_the_recipe_step_and_attempt(cfg, store):
    """The columns existed and were never written, so every DFT job row read
    `step='' attempt=0` however far up the ladder it actually was -- which is
    precisely the history `csp status --why` exists to show."""

    class Laddered(FakeStage):
        def claim(self, store, budget):
            taken, self.pool = self.pool[:budget], self.pool[budget:]
            return [WorkItem(key=f"s{i}", structure_ids=[i],
                             payload={"step_name": "relax", "attempt": 2})
                    for i in taken]

    d = driver(cfg, store, FakeScheduler(), [Laddered(work=2)], stages=["dft"])
    d.cycle(1)
    rows = store.jobs(stage="dft")
    assert rows and all(r["recipe_step"] == "relax" for r in rows)
    assert all(r["attempt"] == 2 for r in rows)


def test_the_projection_asks_the_stage_for_its_resources(cfg, store):
    """`dft` was missing from the table this used to be, so it fell through to
    the machine default and a bare 24-hour walltime -- 384 core-hours against a
    recipe asking for 32.

    That 12x error used to hold back submissions, because a budget gated on it.
    Now it only skews a reported number, which is why the budget is gone (D143)
    and this still checks the stage is asked.
    """

    class Hinted(FakeStage):
        def resource_hint(self):
            return 16, "02:00:00"

    d = driver(cfg, store, FakeScheduler(), [Hinted(work=1)], stages=["dft"])
    assert d._walltime_estimate("dft") == pytest.approx(32.0)


def test_a_stage_with_no_hint_falls_back_to_the_configured_resources(cfg, store):
    d = driver(cfg, store, FakeScheduler(), [FakeStage(work=1)], stages=["dft"])
    assert d._walltime_estimate("screen") > 0


# --------------------------------------------------------------------------
# Per-task reconciliation (D113)
#
# A 200-task array held 200 in-flight slots until its slowest task ended, so
# 197 finished jobs freed nothing and the queue drained to a handful while
# 1,078 phases waited. Live: t1's array 26816467 sat at 197 COMPLETED, 1
# FAILED, 2 RUNNING while the driver reported "200 already out".


class TaskScheduler(FakeScheduler):
    """A scheduler that reports per-task statuses, as SLURM does for arrays."""

    def __init__(self, limits=None):
        super().__init__(limits)
        self.tasks: dict[str, JobStatus] = {}

    def poll_tasks(self, job_ids):
        return dict(self.tasks)


def _submit_one(cfg, store, sched, stage):
    d = driver(cfg, store, sched, [stage], stages=["dft"], max_cycles=1)
    d.cycle(1)
    return d


class TestPerTaskReconciliation:

    def test_array_task_id_is_stamped_on_every_row(self, cfg, store):
        sched, stage = TaskScheduler(), FakeStage(work=4)
        _submit_one(cfg, store, sched, stage)
        rows = [j for j in store.jobs() if j["slurm_id"]]
        assert [r["array_task_id"] for r in rows] == list(range(len(rows)))

    def test_finished_task_frees_its_own_slot(self, cfg, store):
        """One task done, the rest running: only that row leaves flight."""
        sched, stage = TaskScheduler(), FakeStage(work=4)
        d = _submit_one(cfg, store, sched, stage)
        jid = sched.submitted and "1001"
        sched.tasks = {f"{jid}_0": JobStatus(job_id=f"{jid}_0", state=JobState.done)}
        d._reconcile()
        states = sorted(j["state"] for j in store.jobs())
        assert states.count("done") == 1
        assert states.count("queued") + states.count("running") == len(states) - 1

    def test_only_the_finished_task_is_handed_back(self, cfg, store):
        """The stage must not be handed a sibling's unfinished claim."""
        sched, stage = TaskScheduler(), FakeStage(work=4)
        d = _submit_one(cfg, store, sched, stage)
        sched.tasks = {"1001_2": JobStatus(job_id="1001_2", state=JobState.done)}
        d._reconcile()
        assert len(stage.reconciled) == 1
        assert stage.reconciled[0][2] == 1        # exactly one item, not four

    def test_unknown_task_is_never_written_done(self, cfg, store):
        """D043 still holds per task: silence is not success."""
        sched, stage = TaskScheduler(), FakeStage(work=4)
        d = _submit_one(cfg, store, sched, stage)
        sched.tasks = {"1001_0": JobStatus(job_id="1001_0", state=JobState.unknown)}
        sched.statuses = {"1001": JobStatus(job_id="1001", state=JobState.unknown)}
        d._reconcile()
        assert not any(j["state"] == "done" for j in store.jobs())
        assert stage.reconciled == []

    def test_scheduler_without_poll_tasks_still_works(self, cfg, store):
        """The whole-submission path is unchanged for schedulers with no arrays."""
        sched, stage = FakeScheduler(), FakeStage(work=4)
        d = _submit_one(cfg, store, sched, stage)
        sched.statuses = {"1001": JobStatus(job_id="1001", state=JobState.done)}
        d._reconcile()
        assert all(j["state"] == "done" for j in store.jobs() if j["slurm_id"])
        assert stage.reconciled and stage.reconciled[0][2] == 4   # all four at once

    def test_a_handed_back_item_knows_its_array_position(self, cfg, store):
        """The item must carry its OWN index, not its position in the list.

        Per-task reconciliation hands the stage a one-element list, so a stage
        that called `enumerate` would name every task 0 and read task 0's
        results for all fourteen of them -- which is how a campaign spends the
        GPU hours, writes the structures to disk and records zero (D121).
        """
        sched, stage = TaskScheduler(), FakeStage(work=4)
        d = _submit_one(cfg, store, sched, stage)
        sched.tasks = {"1001_2": JobStatus(job_id="1001_2", state=JobState.done)}
        d._reconcile()
        assert stage.indices == [[2]]

    def test_every_position_survives_a_whole_submission_handback(self, cfg, store):
        sched, stage = FakeScheduler(), FakeStage(work=4)
        d = _submit_one(cfg, store, sched, stage)
        sched.statuses = {"1001": JobStatus(job_id="1001", state=JobState.done)}
        d._reconcile()
        assert stage.indices == [[0, 1, 2, 3]]

    def test_the_index_survives_a_driver_restart(self, cfg, store, tmp_path):
        """A second driver reads the claim from disk; the index must come back."""
        sched, stage = TaskScheduler(), FakeStage(work=4)
        _submit_one(cfg, store, sched, stage)
        sched.tasks = {"1001_3": JobStatus(job_id="1001_3", state=JobState.done)}
        fresh = driver(cfg, store, sched, [stage], stages=["dft"], max_cycles=1)
        fresh._claims.clear()                     # nothing in memory, as after a restart
        fresh._reconcile()
        assert stage.indices == [[3]]

    def test_a_task_is_not_handed_back_twice(self, cfg, store):
        """Per-task first, then the aggregate: the same claim must not repeat."""
        sched, stage = TaskScheduler(), FakeStage(work=4)
        d = _submit_one(cfg, store, sched, stage)
        sched.tasks = {"1001_0": JobStatus(job_id="1001_0", state=JobState.done)}
        d._reconcile()
        # Now the whole array reports done; task 0 is already reconciled.
        sched.tasks = {}
        sched.statuses = {"1001": JobStatus(job_id="1001", state=JobState.done)}
        d._reconcile()
        handed = [n for _, _, n in stage.reconciled]
        assert handed == [1, 3]          # one task, then only its three siblings


class TestStatusFile:

    def test_status_json_is_written_each_cycle(self, cfg, store, tmp_path):
        sched, stage = TaskScheduler(), FakeStage(work=4)
        d = driver(cfg, store, sched, [stage], stages=["dft"], max_cycles=1)
        d.run(watch=False)
        path = Path(cfg.campaign.workdir) / "status.json"
        assert path.is_file()
        import json as _json
        data = _json.loads(path.read_text())
        assert data["cycle"] == 1
        assert "in_flight" in data and "structures" in data
        assert data["stages"][0]["stage"] == "dft"

    def test_status_json_is_valid_after_every_write(self, cfg, store):
        """Written via rename, so a concurrent reader never sees half a file."""
        import json as _json
        sched, stage = TaskScheduler(), FakeStage(work=4)
        d = driver(cfg, store, sched, [stage], stages=["dft"], max_cycles=3)
        d.run(watch=True)
        path = Path(cfg.campaign.workdir) / "status.json"
        assert _json.loads(path.read_text())["cycle"] == 3
        assert not list(Path(cfg.campaign.workdir).glob("*.json.tmp"))


class TestMaintainsTargetInFlight:
    """The point of D113: hold `max_in_flight` jobs out, refilling as they end.

    `max_in_flight` is 4 in the fixture campaign. The driver should submit up
    to 4, and on every later cycle top back up to 4 as tasks finish -- not wait
    for the whole array, and not overshoot.
    """

    def _finish(self, sched, jid, idxs):
        for i in idxs:
            sched.tasks[f"{jid}_{i}"] = JobStatus(job_id=f"{jid}_{i}",
                                                  state=JobState.done)

    def test_tops_up_to_the_cap_as_tasks_finish(self, cfg, store):
        sched, stage = TaskScheduler(), FakeStage(work=12)
        d = driver(cfg, store, sched, [stage], stages=["dft"], max_cycles=1)

        def in_flight():
            return sum(1 for j in store.jobs()
                       if j["state"] in {"queued", "running", "held"})

        d.cycle(1)
        assert in_flight() == 4, "first cycle fills the cap"

        # Two of the four finish. The next cycle must submit exactly two.
        self._finish(sched, "1001", [0, 1])
        d.cycle(2)
        assert in_flight() == 4, "topped back up to the cap, not left at 2"
        assert len(sched.submitted) == 2, "a second submission went out"
        assert sched.submitted[1].array_size == 2, "exactly the free slots"

        # One more finishes; one more goes out.
        self._finish(sched, "1001", [2])
        d.cycle(3)
        assert in_flight() == 4
        assert sched.submitted[2].array_size == 1

    def test_does_not_overshoot_the_cap(self, cfg, store):
        """Nothing finished, so nothing new is submitted."""
        sched, stage = TaskScheduler(), FakeStage(work=12)
        d = driver(cfg, store, sched, [stage], stages=["dft"], max_cycles=1)
        d.cycle(1)
        d.cycle(2)
        d.cycle(3)
        assert len(sched.submitted) == 1
        assert sum(1 for j in store.jobs()
                   if j["state"] in {"queued", "running", "held"}) == 4

    def test_one_straggler_does_not_block_refill(self, cfg, store):
        """The live failure: 3 of 4 done, 1 running, and the driver held.

        Before D113 the whole array stayed in flight until its last task ended,
        so three free slots submitted nothing.
        """
        sched, stage = TaskScheduler(), FakeStage(work=12)
        d = driver(cfg, store, sched, [stage], stages=["dft"], max_cycles=1)
        d.cycle(1)
        self._finish(sched, "1001", [0, 1, 2])        # 3 done, task 3 still running
        d.cycle(2)
        assert len(sched.submitted) == 2, "the straggler must not hold the other three"
        assert sched.submitted[1].array_size == 3


class TestInFlightDetail:
    """`cycle 11  in flight 1` eleven times over says only that the driver is
    alive. The lines below say which job, how many structures, how long, and
    how far along."""

    @staticmethod
    def _row(**kw):
        base = {"stage": "dft", "slurm_id": "999", "state": "running",
                "workdir": "", "submitted_at": None}
        base.update(kw)
        return base

    def test_one_array_job_is_one_line_not_one_per_structure(self, cfg, store):
        """A `dft` array has a row per structure, all sharing one slurm_id. A
        seven-task array printed seven identical lines."""
        d = driver(cfg, store, FakeScheduler(), [FakeStage()], stages=["dft"])
        groups = {("dft", "26931389"):
                  [self._row(slurm_id="26931389") for _ in range(7)]}
        lines = d._describe_groups(groups)

        assert len(lines) == 1
        assert "26931389" in lines[0]
        assert "7 structures" in lines[0]
        assert lines[0].rstrip().endswith("structures")   # no stray "." workdir

    def test_a_single_structure_job_does_not_say_1_structures(self, cfg, store):
        d = driver(cfg, store, FakeScheduler(), [FakeStage()], stages=["dft"])
        lines = d._describe_groups({("dft", "1"): [self._row()]})
        assert "structures" not in lines[0]

    def test_many_jobs_are_capped_and_counted(self, cfg, store):
        """Twenty in-flight jobs must not bury the cycle line."""
        d = driver(cfg, store, FakeScheduler(), [FakeStage()], stages=["dft"])
        groups = {("dft", str(n)): [self._row(slurm_id=str(n))] for n in range(20)}
        lines = d._describe_groups(groups)

        assert len(lines) == d.MAX_DETAIL_LINES + 1
        assert lines[-1] == "... and 14 more job(s) in flight"

    def test_progress_written_by_a_worker_is_read_back(self, cfg, store, tmp_path):
        """The driver cannot see another allocation's stdout, so the worker
        leaves progress on shared disk and the driver reads it."""
        import json

        work = tmp_path / "screen"
        work.mkdir()
        (work / "screen-1-232.progress.json").write_text(json.dumps(
            {"done": 187, "total": 232, "converged": 185, "failed": 0}))

        d = driver(cfg, store, FakeScheduler(), [FakeStage()], stages=["dft"])
        line = d._describe_job(self._row(stage="screen", workdir=str(work)))

        assert "187/232 (81%)" in line
        assert "185 converged, 0 failed" in line


def test_a_job_whose_results_fail_to_fold_is_not_left_marked_done(cfg, store):
    """D144: the job's terminal state and its results commit together.

    Before, the job row was committed `done` and THEN its results were written;
    anything that stopped the fold in between left a finished job no later
    cycle would look at, and its results unread on disk.
    """
    class Breaks(FakeStage):
        def reconcile(self, store, job_row, status, items):
            store.add_filter_event(structure_id=1, gate="partial", passed=True)
            raise RuntimeError("killed mid-fold")

    sched = FakeScheduler()
    stage = Breaks(work=1)
    d = driver(cfg, store, sched, [stage], stages=["dft"])
    d.cycle()
    jid = store.jobs()[0]["slurm_id"]
    sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.done, raw_state="COMPLETED")
    with pytest.raises(RuntimeError):
        d.cycle()
    assert store.jobs()[0]["state"] == "queued", "reconciled again next cycle, not lost"
    assert store.sql.execute("SELECT COUNT(*) FROM filter_event").fetchone()[0] == 0
