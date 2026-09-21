"""Surviving a storage outage instead of mistaking one for a broken database.

`/scratch` is NFSv4. When the server's lease expires mid-write, SQLite raises
`OperationalError: disk I/O error` -- and because `OperationalError` is a
SUBCLASS of `DatabaseError`, that used to be handled by the code written for a
genuinely corrupt file, which told the user to delete the database (D148).

These tests pin the two halves of the fix: the classifier that tells the two
conditions apart, and the driver's willingness to wait one out.
"""

import sqlite3

import pytest

from cspflow.db.store import Store, is_transient_error
from cspflow.driver import DB_RETRY_BACKOFF, Driver, DriverError, DriverOptions

from .test_driver import CAMPAIGN, FakeScheduler, FakeStage, FakeInProcessStage  # noqa: F401
from cspflow.config.loader import load_campaign


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path))
    return load_campaign(path)


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "c.db", campaign="t") as s:
        yield s


class TestClassifier:
    """The distinction the old handler did not make."""

    @pytest.mark.parametrize("text", [
        "disk I/O error",
        "unable to open database file",
        "database is locked",
    ])
    def test_a_storage_blip_is_transient(self, text):
        assert is_transient_error(sqlite3.OperationalError(text))

    @pytest.mark.parametrize("text", [
        "file is not a database",
        "database disk image is malformed",
    ])
    def test_real_corruption_is_not(self, text):
        """Retrying corruption would turn a clear failure into a slow one."""
        assert not is_transient_error(sqlite3.DatabaseError(text))

    def test_a_non_sqlite_error_is_not_transient(self):
        assert not is_transient_error(ValueError("disk I/O error"))

    def test_the_subclass_relationship_that_caused_the_bug(self):
        """If this ever stops holding, the cli handler can be simplified."""
        assert issubclass(sqlite3.OperationalError, sqlite3.DatabaseError)


class TestReconnect:
    def test_reconnect_reopens_a_usable_connection(self, store):
        store.set_meta("k", "v")
        store.sql.commit()
        store.reconnect()
        assert store.meta("k") == "v"

    def test_reconnect_does_not_commit_uncommitted_work(self, store):
        """`close()` commits; `reconnect()` must not.

        The reason to reconnect is that a write just failed, so flushing
        whatever is pending is the last thing wanted.
        """
        store.set_meta("committed", "yes")
        store.sql.commit()
        store.sql.execute("INSERT INTO campaign_meta(key, value) VALUES ('dirty', '1')")
        store.reconnect()
        assert store.meta("committed") == "yes"
        assert store.meta("dirty") is None

    def test_reconnect_clears_an_open_transaction_depth(self, store):
        store._tx_depth = 3
        store.reconnect()
        assert not store.in_transaction


def _driver(cfg, store, scheduler, impls, sleeps, **opts):
    options = DriverOptions(interval=0, **opts)
    return Driver(cfg, store, scheduler, impls, options,
                  sleep=lambda s: sleeps.append(s))


class TestCycleRetry:
    def test_a_cycle_lost_to_an_outage_is_retried_not_fatal(self, cfg, store):
        """The 2026-09-19 failure: one blip at cycle 32 ended a healthy run."""
        sleeps: list[float] = []
        d = _driver(cfg, store, FakeScheduler(), [FakeInProcessStage(work=2)],
                    sleeps, stages=["filter"])
        real, calls = d.cycle, []

        def flaky(n):
            calls.append(n)
            if len(calls) == 1:
                raise sqlite3.OperationalError("disk I/O error")
            return real(n)

        d.cycle = flaky
        reports = d.run(watch=False)
        assert len(calls) == 2                      # failed once, then succeeded
        assert len(reports) == 1                    # and the run carried on
        assert DB_RETRY_BACKOFF[0] in sleeps        # having waited first

    def test_corruption_is_not_retried(self, cfg, store):
        sleeps: list[float] = []
        d = _driver(cfg, store, FakeScheduler(), [FakeInProcessStage(work=1)],
                    sleeps, stages=["filter"])
        calls = []

        def broken(n):
            calls.append(n)
            raise sqlite3.DatabaseError("file is not a database")

        d.cycle = broken
        with pytest.raises(sqlite3.DatabaseError):
            d.run(watch=False)
        assert len(calls) == 1 and sleeps == []

    def test_an_outage_that_never_ends_says_the_file_is_intact(self, cfg, store):
        """The message must not send the user to delete a healthy database."""
        sleeps: list[float] = []
        d = _driver(cfg, store, FakeScheduler(), [FakeInProcessStage(work=1)],
                    sleeps, stages=["filter"])
        d.cycle = lambda n: (_ for _ in ()).throw(
            sqlite3.OperationalError("disk I/O error"))
        with pytest.raises(DriverError, match="INTACT"):
            d.run(watch=False)
        assert len(sleeps) == len(DB_RETRY_BACKOFF)


class TestSubmissionWindowIsProtected:
    def test_a_blip_after_sbatch_still_records_the_job(self, cfg, store):
        """The one state the database cannot recover from by itself.

        Between `scheduler.submit()` returning and the slurm id being stamped on
        the rows, a job is running that nothing records. Before D148 an outage
        there killed the driver and orphaned the job.
        """
        sleeps: list[float] = []
        sched = FakeScheduler()
        d = _driver(cfg, store, sched, [FakeStage(work=1)], sleeps, stages=["dft"])

        real_update, calls = store.update_job, []

        def flaky_update(row, **kw):
            calls.append(kw)
            if len(calls) == 1:
                raise sqlite3.OperationalError("disk I/O error")
            return real_update(row, **kw)

        store.update_job = flaky_update
        d.cycle(1)

        assert len(sched.submitted) == 1, "the job was submitted exactly once"
        stamped = store.sql.execute(
            "SELECT slurm_id, state FROM job WHERE slurm_id IS NOT NULL").fetchall()
        assert stamped, "the submitted job was recorded despite the outage"
        assert stamped[0]["state"] == "queued"
