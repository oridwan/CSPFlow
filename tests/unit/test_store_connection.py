"""One connection per Store, one commit per transaction, and no ASE lock file (D144).

WHY THIS EXISTS
    A 14,755-structure campaign spent nine hours in Phase A with its GPU work
    finished after two. Measured on a copy of it on nfs4, the time went to the
    database layer, not the science:

    | operation                               | before   | after    |
    |-----------------------------------------|----------|----------|
    | read the screened set                   | 59 s     | 22 s     |
    | one structure update                    | 702 ms   | 2.5 ms   |
    | find one system's candidates (x 288)    | 65 s     | once     |

    Three causes, each pinned by a test below so it cannot drift back:

    * ASE opened a NEW sqlite connection for every row a `select()` returned
      and for every write, because `Store.ase` handed out a fresh connection;
    * ASE's key-value tables have no index on `id`, so every update's
      `DELETE ... WHERE id IN (...)` scanned every key-value row in the file;
    * every write was its own commit, an fsync round trip on NFS.

    And one hang: ASE's lock file is acquired with `timeout=inf`. A process
    killed holding it wedged the campaign until someone deleted the file.
"""

from __future__ import annotations

import signal
import sqlite3

import pytest
from ase.build import bulk

import ase.db.sqlite as ase_sqlite
from cspflow.db.store import Origin, Store, StructureState


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "c.db", campaign="t") as s:
        yield s


@pytest.fixture
def connections(monkeypatch):
    """Count every sqlite connection ASE opens for itself."""
    opened = {"n": 0}
    original = ase_sqlite.SQLite3Database._connect

    def counting(self):
        opened["n"] += 1
        return original(self)

    monkeypatch.setattr(ase_sqlite.SQLite3Database, "_connect", counting)
    return opened


def _seed(store, n=5, state=StructureState.screened):
    return [store.add_structure(bulk("Fe", "bcc", a=2.87), origin=Origin.seed,
                                state=state, mlip_e_per_atom=-1.0 - i)
            for i in range(n)]


class TestOneConnection:
    def test_ase_runs_on_the_stores_own_connection(self, store):
        assert store.ase.connection is store.sql

    def test_no_operation_opens_a_connection_of_its_own(self, store, connections):
        """Before D144: one per write, two per update, one per ROW selected."""
        ids = _seed(store, 5)
        for sid in ids:
            store.update_structure(sid, flag=True)
            store.set_structure_state(sid, StructureState.new)
        store.replace_geometry(ids[0], bulk("Fe", "bcc", a=2.9))
        rows = list(store.structures(state=StructureState.new.value))
        assert len(rows) == 5
        assert store.count_structures(state=StructureState.new.value) == 5
        assert connections["n"] == 0

    def test_a_reopened_store_rebinds_to_its_new_connection(self, tmp_path):
        path = tmp_path / "c.db"
        with Store.create(path, campaign="t") as s:
            _seed(s, 1)
        with Store.open(path) as s:
            assert s.ase.connection is s.sql
            assert s.count_structures() == 1


class TestNoLockFile:
    def test_writing_leaves_no_lock_file(self, store, tmp_path):
        ids = _seed(store, 3)
        store.update_structure(ids[0], flag=True)
        assert not list(tmp_path.glob("*.lock"))

    def test_a_stale_lock_file_does_not_hang_a_write(self, store, tmp_path):
        """What wedged RE-magnets-CHGNet: `kill` left `campaign.db.lock` behind,
        and the next `csp source` waited on it for as long as anyone let it."""
        (tmp_path / "c.db.lock").write_text("")

        def hung(*_):
            raise TimeoutError("a write waited on campaign.db.lock")

        previous = signal.signal(signal.SIGALRM, hung)
        signal.alarm(20)
        try:
            ids = _seed(store, 2)
            store.update_structure(ids[0], flag=True)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        assert store.count_structures() == 2


class TestIdIndices:
    TABLES = ("keys", "text_key_values", "number_key_values", "species")

    def _plans(self, store):
        return {t: " ".join(str(r[-1]) for r in store.sql.execute(
                    f"EXPLAIN QUERY PLAN DELETE FROM {t} WHERE id IN (1)"))
                for t in self.TABLES}

    def test_a_new_campaign_is_indexed(self, store):
        for table, plan in self._plans(store).items():
            assert plan.startswith("SEARCH") and "ix_ase_" in plan, f"{table}: {plan}"

    def test_an_existing_campaign_gains_them_when_opened(self, tmp_path):
        """Additive, like the late columns: no version bump, no lock-out."""
        path = tmp_path / "c.db"
        with Store.create(path, campaign="t") as s:
            _seed(s, 2)
            for name, _table, _col in Store._LATE_INDICES:
                s.sql.execute(f"DROP INDEX {name}")
            s.sql.commit()
            assert any("SCAN" in p for p in self._plans(s).values())
        with Store.open(path) as s:
            for table, plan in self._plans(s).items():
                assert plan.startswith("SEARCH") and "ix_ase_" in plan, f"{table}: {plan}"
            assert s.count_structures() == 2


class TestTransaction:
    def test_nothing_is_visible_to_another_reader_until_it_ends(self, store, tmp_path):
        outside = sqlite3.connect(str(tmp_path / "c.db"))
        with store.transaction():
            _seed(store, 3)
            store.add_filter_event(structure_id=1, gate="g", passed=True)
            assert outside.execute("SELECT COUNT(*) FROM systems").fetchone()[0] == 0
        assert outside.execute("SELECT COUNT(*) FROM systems").fetchone()[0] == 3
        assert outside.execute("SELECT COUNT(*) FROM filter_event").fetchone()[0] == 1

    def test_an_escaping_exception_undoes_structures_and_tables_together(self, store):
        """A job's state and the results folded from it land together or not at all."""
        with pytest.raises(RuntimeError):
            with store.transaction():
                _seed(store, 2)
                store.add_job(stage="screen", structure_id=1, workdir="w")
                raise RuntimeError("driver killed mid-reconcile")
        assert store.count_structures() == 0
        assert store.jobs() == []

    def test_only_the_outermost_block_commits(self, store, tmp_path):
        outside = sqlite3.connect(str(tmp_path / "c.db"))
        with store.transaction():
            with store.transaction():
                _seed(store, 1)
            assert store.in_transaction
            assert outside.execute("SELECT COUNT(*) FROM systems").fetchone()[0] == 0
        assert not store.in_transaction
        assert outside.execute("SELECT COUNT(*) FROM systems").fetchone()[0] == 1

    def test_outside_a_transaction_every_write_still_commits(self, store, tmp_path):
        """The default must stay durable: the driver writes job rows BEFORE it submits."""
        outside = sqlite3.connect(str(tmp_path / "c.db"))
        store.add_job(stage="dft", workdir="w")
        _seed(store, 1)
        assert outside.execute("SELECT COUNT(*) FROM job").fetchone()[0] == 1
        assert outside.execute("SELECT COUNT(*) FROM systems").fetchone()[0] == 1


class TestCounts:
    def test_count_in_state_lacking_a_key(self, store):
        ids = _seed(store, 4)
        store.update_structure(ids[0], dedup_checked=True)
        store.set_structure_state(ids[1], StructureState.failed)
        assert store.count_in_state_lacking(StructureState.screened, "dedup_checked") == 2

    def test_unplaced_candidates_skip_placed_unhullable_and_energyless(self, store):
        ids = _seed(store, 4)
        store.add_structure(bulk("Fe", "bcc", a=2.87), origin=Origin.seed,
                            state=StructureState.screened)          # no MLIP energy
        store.add_hull(structure_id=ids[0], hull_type="mlip", energy_scale="raw",
                       e_above_hull=0.0, ref_set_hash="r")
        store.add_hull(structure_id=ids[1], hull_type="dft", energy_scale="raw",
                       e_above_hull=0.0, ref_set_hash="r")          # DFT does not count
        store.update_structure(ids[2], hull_error="Fe: no reference")
        assert store.unplaced_mlip_candidates() == 2                # ids[1], ids[3]

    def test_hull_error_can_be_cleared(self, store):
        [sid] = _seed(store, 1)
        store.update_structure(sid, hull_error="x")
        store.update_structure(sid, delete_keys=["hull_error"], e_above_hull_mlip=0.1)
        row = store.get_structure(sid)
        assert "hull_error" not in row.key_value_pairs
        assert row.key_value_pairs["e_above_hull_mlip"] == pytest.approx(0.1)


def test_a_second_writer_waits_for_a_transaction_instead_of_failing(tmp_path):
    """Two drivers on one campaign (D129) are routine. Transactions are batches now,
    so the holder keeps the write lock for seconds; the other must wait, not die
    with `database is locked`."""
    import threading
    import time

    path = tmp_path / "c.db"
    Store.create(path, campaign="t").close()
    holding, done = threading.Event(), threading.Event()

    def first_driver():
        with Store.open(path) as a:
            with a.transaction():
                _seed(a, 3)
                holding.set()
                time.sleep(1.5)                  # a reconcile in progress
        done.set()

    worker = threading.Thread(target=first_driver)
    worker.start()
    assert holding.wait(10)
    with Store.open(path) as b:
        started = time.perf_counter()
        b.add_job(stage="dft", workdir="w")      # blocks until the first commits
        waited = time.perf_counter() - started
    worker.join(10)
    assert done.is_set()
    assert waited > 0.5, "the second writer did not actually contend"
    with Store.open(path) as c:
        assert c.count_structures() == 3 and len(c.jobs()) == 1


def test_reading_structures_does_not_query_once_per_row(store):
    """ASE looked up its external-table list for EVERY row of a select: 14,752
    extra queries, 19 of 22 seconds, to read the screened set on nfs4 (D144)."""
    _seed(store, 40)
    seen = []
    store.sql.set_trace_callback(seen.append)
    try:
        rows = list(store.structures(state=StructureState.screened.value))
    finally:
        store.sql.set_trace_callback(None)
    assert len(rows) == 40
    per_row = [q for q in seen if "external_table_name" in q]
    assert len(per_row) == 0, f"{len(per_row)} external-table queries for 40 rows"
    assert len(seen) < 10, f"{len(seen)} statements to read 40 rows"
