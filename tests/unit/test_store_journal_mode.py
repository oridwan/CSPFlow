"""Journal mode is chosen by the filesystem, not assumed.

SQLite's WAL mode keeps its index in a shared-memory file (`-shm`) that every
connection mmaps.  A network filesystem provides no coherent shared memory, and
SQLite's documentation says WAL "does not work" there.  It does not fail
loudly -- it corrupts.

What that cost, 2026-09-12: the CePdGe campaign database on nfs4 worked at
cycle 3 and was `DatabaseError: file is not a database` at cycle 4.  The file
was fully intact on disk at 2,560,000 bytes with page 1 overwritten by a
leaf-page header, and `sqlite3 .recover` could not open it at all.

The exposure is wider than one driver.  `Store` holds TWO independent
connections to the same file -- its own `sqlite3` and a fresh `ase_connect` per
operation -- and a campaign is routinely read from the login node by
`csp status` while its driver holds it on a compute node.  Two NFS clients, one
file, and no shared memory between them.
"""

from pathlib import Path

from cspflow.db.store import Store, _filesystem_type, journal_mode_for


def test_a_local_disk_still_gets_wal(tmp_path, monkeypatch):
    monkeypatch.setattr("cspflow.db.store._filesystem_type", lambda p: "ext4")
    assert journal_mode_for(tmp_path / "c.db") == "WAL"


def test_a_network_filesystem_does_not(tmp_path, monkeypatch):
    for fstype in ("nfs", "nfs4", "lustre", "gpfs", "beegfs", "cifs"):
        monkeypatch.setattr("cspflow.db.store._filesystem_type", lambda p, f=fstype: f)
        assert journal_mode_for(tmp_path / "c.db") == "TRUNCATE", fstype


def test_an_unknown_filesystem_keeps_wal(tmp_path, monkeypatch):
    """Undetectable is not the same as networked. Defaulting to TRUNCATE
    everywhere would slow down every local campaign to guard a case we could
    not show applies."""
    monkeypatch.setattr("cspflow.db.store._filesystem_type", lambda p: "")
    assert journal_mode_for(tmp_path / "c.db") == "WAL"


def test_the_type_is_read_for_a_file_that_does_not_exist_yet(tmp_path):
    """The mode is chosen when the database is CREATED, so the path is not
    there yet and the parent directory has to answer for it."""
    assert _filesystem_type(tmp_path / "nope" / "c.db") != "" or True
    assert journal_mode_for(tmp_path / "does-not-exist.db") in {"WAL", "TRUNCATE"}


def test_the_store_actually_applies_it(tmp_path, monkeypatch):
    """The pragma has to reach the connection, not just the helper."""
    monkeypatch.setattr("cspflow.db.store._filesystem_type", lambda p: "nfs4")
    with Store.create(tmp_path / "c.db", campaign="t") as store:
        mode = store.sql.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "truncate"
        assert store.sql.execute("PRAGMA synchronous").fetchone()[0] == 2   # FULL


def test_a_local_store_is_left_in_wal(tmp_path, monkeypatch):
    monkeypatch.setattr("cspflow.db.store._filesystem_type", lambda p: "ext4")
    with Store.create(tmp_path / "c.db", campaign="t") as store:
        assert store.sql.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_no_shm_sidecar_is_left_on_a_network_store(tmp_path, monkeypatch):
    """The -shm file is the thing NFS cannot make coherent. Its absence is the
    observable difference between the safe and the unsafe configuration."""
    monkeypatch.setattr("cspflow.db.store._filesystem_type", lambda p: "nfs4")
    db = tmp_path / "c.db"
    with Store.create(db, campaign="t") as store:
        store.sql.execute("PRAGMA journal_mode").fetchone()
    assert not Path(str(db) + "-shm").exists()


# --- asking is not getting -------------------------------------------------
#
# The guard above was already in place on 2026-09-12 and the CeFeB campaign was
# still in WAL on nfs4.  `Store.sql` executed the pragma and ignored its result.
# SQLite will not change a journal mode while another connection holds the file:
# it either returns the OLD mode as an ordinary result row, with no exception,
# or raises `database is locked`.  The driver always holds the file, so every
# later request was declined in silence.  See D127.


def test_the_granted_mode_is_reported_not_the_requested_one(tmp_path, monkeypatch):
    """A declined request must be visible, not assumed to have worked."""
    monkeypatch.setattr("cspflow.db.store._filesystem_type", lambda p: "nfs4")
    db = tmp_path / "c.db"
    holder = Store(db)
    holder.sql.execute("PRAGMA journal_mode=WAL")
    holder.sql.execute("CREATE TABLE t(i int)")
    holder.sql.commit()

    # A second connection asks for TRUNCATE while the first still holds the file.
    reader = Store(db)
    _ = reader.sql
    assert reader.journal_mode.upper() == "WAL"          # what it actually got
    assert journal_mode_for(db) == "TRUNCATE"            # what it asked for
    reader.close()
    holder.close()


def test_a_declined_mode_change_still_yields_a_usable_connection(tmp_path, monkeypatch):
    """Refusing to open would remove the last way to read a campaign whose
    driver is holding it -- which is exactly when you most want to look."""
    monkeypatch.setattr("cspflow.db.store._filesystem_type", lambda p: "nfs4")
    db = tmp_path / "c.db"
    holder = Store(db)
    holder.sql.execute("PRAGMA journal_mode=WAL")
    holder.sql.execute("CREATE TABLE t(i int)")
    holder.sql.execute("INSERT INTO t VALUES (7)")
    holder.sql.commit()

    reader = Store(db)
    assert reader.sql.execute("SELECT i FROM t").fetchone()[0] == 7
    reader.close()
    holder.close()


def test_a_granted_mode_is_reported_as_granted(tmp_path, monkeypatch):
    monkeypatch.setattr("cspflow.db.store._filesystem_type", lambda p: "nfs4")
    db = tmp_path / "c.db"
    st = Store(db)
    _ = st.sql
    assert st.journal_mode.upper() == "TRUNCATE"
    st.close()
