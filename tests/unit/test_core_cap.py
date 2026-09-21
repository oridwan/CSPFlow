"""`dft.max_cores` -- capping the CORES a campaign holds, not the job count.

WHY A SECOND CAP
    `max_in_flight` counts JOBS, which is not the quantity anyone is trying to
    protect. On the RE-magnets campaign the recipe asks for 64 ranks, so
    `max_in_flight: 48` was 3,072 cores -- and nothing on this cluster's `cpu`
    role caps that, so the campaign would have taken the machine and left its
    owner unable to run anything else.

    The campaign.yaml comment even said "cpu=768 / ntasks=16 = 48", which was
    arithmetic against the wrong rank count: the recipe is 64, not 16.
"""

from __future__ import annotations

import pytest

from cspflow.scheduler.base import compute_throttle

RANKS = 64


def throttle(**kw):
    kw.setdefault("requested_in_flight", 48)
    kw.setdefault("requested_concurrent", 48)
    kw.setdefault("ntasks", RANKS)
    return compute_throttle(**kw)


class TestTheCap:
    def test_without_a_cap_nothing_bounds_the_cores(self):
        """The state this was written to end: 48 jobs x 64 ranks."""
        t = throttle()
        assert t.in_flight * RANKS == 3072

    def test_the_cap_converts_cores_into_a_job_count(self):
        t = throttle(max_cores=720)
        assert t.in_flight == 11                       # 720 // 64
        assert t.in_flight * RANKS == 704 <= 720

    def test_queued_cores_count_against_it_not_only_running(self):
        """Work already in the queue WILL occupy those cores. A cap that
        ignored it would approve a submission the queue has already spent."""
        t = throttle(max_cores=720, cores_in_flight=672, already_in_flight=11)
        assert t.in_flight == 0

    def test_it_never_pushes_the_total_past_the_cap(self):
        for held in range(0, 721, 32):
            t = throttle(max_cores=720, cores_in_flight=held)
            assert held + t.in_flight * RANKS <= 720, f"overshot with {held} held"

    def test_lowering_the_cap_mid_run_stops_submission_but_cannot_recall_work(self):
        """Worth stating rather than discovering.

        Cores already held belong to jobs already in the queue or on a node.
        Lowering `max_cores` below what is out stops anything further going in;
        it does not cancel what is running, and the throttle does not pretend
        otherwise by reporting negative room.
        """
        t = throttle(max_cores=720, cores_in_flight=1024, already_in_flight=16)
        assert t.in_flight == 0
        assert "1024 held" in t.binding

    def test_a_full_campaign_submits_nothing_rather_than_a_little(self):
        t = throttle(max_cores=720, cores_in_flight=720)
        assert t.in_flight == 0 and t.concurrent_tasks == 0

    def test_a_cap_smaller_than_one_job_admits_nothing(self):
        """Better to submit nothing and say so than to breach the cap by one
        job because a partial job cannot exist."""
        assert throttle(max_cores=32).in_flight == 0

    def test_the_rank_count_is_what_converts_cores_to_jobs(self):
        """So the cap stays right when the recipe changes, rather than needing
        the arithmetic redone by hand."""
        assert throttle(max_cores=720, ntasks=16).in_flight == 45
        assert throttle(max_cores=720, ntasks=64).in_flight == 11


class TestNamingTheLimit:
    def test_the_binding_reason_names_the_core_cap_when_it_binds(self):
        """A message that blames `max_in_flight` while the core cap is what
        stopped submission sends the reader to change the wrong number."""
        t = throttle(max_cores=720, cores_in_flight=480, already_in_flight=8)
        assert "max_cores" in t.binding
        assert "480 held" in t.binding

    def test_the_job_count_is_named_when_it_is_the_lower_limit(self):
        t = throttle(requested_in_flight=4, max_cores=10_000)
        assert "max_cores" not in t.binding

    def test_a_limit_reads_as_a_limit_and_not_as_a_count(self):
        """`render()` sits on a line that already carries real counts, so it
        must not look like one -- it was read as '8 jobs were submitted'."""
        rendered = throttle(max_cores=720).render()
        assert rendered.startswith("may submit")
        assert "may run at once" in rendered


def test_the_job_row_records_what_it_actually_asked_for(tmp_path):
    """`max_cores` is only as honest as the number stored per job, and that
    comes from the submitted spec rather than from config -- a retry rung can
    raise a job's resources after the config was read."""
    from cspflow.db.store import Store

    with Store.create(tmp_path / "c.db", campaign="t") as store:
        a = store.add_job(stage="dft", cores=64)
        b = store.add_job(stage="dft")                  # unknown, older row
        rows = {j["id"]: j for j in store.jobs()}
        assert rows[a]["cores"] == 64
        assert rows[b]["cores"] == 0, "an unrecorded job must count as 0, not crash"


def test_an_existing_database_gains_the_column_without_a_version_bump(tmp_path):
    """The migration had to be additive.

    `Store.open` REFUSES a schema-version mismatch -- "migration is required;
    refusing to guess" -- so bumping the version would have locked a live
    14,755-structure campaign out of its own database over a bookkeeping column.
    """
    import sqlite3

    from cspflow.db.store import Store

    path = tmp_path / "old.db"
    with Store.create(path, campaign="t") as store:
        store.add_job(stage="dft")
    # simulate the pre-column file
    with sqlite3.connect(path) as raw:
        raw.execute("ALTER TABLE job RENAME TO job_old")
        cols = [r[1] for r in raw.execute("PRAGMA table_info(job_old)") if r[1] != "cores"]
        raw.execute(f"CREATE TABLE job AS SELECT {', '.join(cols)} FROM job_old")
        raw.execute("DROP TABLE job_old")
    with sqlite3.connect(path) as raw:
        assert "cores" not in {r[1] for r in raw.execute("PRAGMA table_info(job)")}

    with Store.open(path) as store:                     # must not raise
        assert all(j["cores"] == 0 for j in store.jobs())
