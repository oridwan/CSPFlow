"""The only thing that opens a campaign database.

One SQLite file holds both halves of the state model: ASE owns `systems` (one
row per structure), and the tables in schema.sql own campaign state.  Keeping
every read and write behind this class is what lets the integrity rules be
enforced at write time rather than hoped for -- in particular the refusal to mix
energy scales or settings inside one hull, which no analysis path can then
bypass.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.db import connect as ase_connect

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

#: How long a write waits for another process's transaction, in ms (D144).
#: Transactions are now batches rather than single rows, so the holder keeps the
#: lock for seconds, not microseconds -- and a campaign is routinely run as TWO
#: drivers, `-pre` and `-dft`, on one database (D129). At 30 s a second driver
#: waiting on the first's reconcile raised `database is locked` and died; a
#: background loop that waits a minute instead is harmless.
BUSY_TIMEOUT_MS = 300_000
SCHEMA_VERSION = "2"

# ASE key_value_pairs accept only these.  A list, dict or None raises
# ValueError deep inside ASE; we catch it at the boundary with a message that
# says what to do instead.
_ASE_SCALARS = (str, int, float, bool)

# ASE also reserves key *names*: every element symbol plus about forty of its own
# row attributes (`formula`, `energy`, `magmom`, `natoms`, `id`, `user`, `age`,
# `fmax`, ...).  Writing one raises a bare `ValueError: Bad key: formula` from
# four frames inside ASE, which says nothing about why a perfectly ordinary word
# is not allowed -- so the check is hoisted here, where the reason can be given.
try:                                                     # pragma: no cover - ASE layout
    from ase.db.core import reserved_keys as _ASE_RESERVED
except ImportError:                                      # pragma: no cover
    _ASE_RESERVED = frozenset()

# ASE refuses a string key_value_pair that int() or float() would parse, because it
# would come back out of the database as a number. Mirrors ASE's own
# str_represents() check in ase/db/core.py.
_NUMBER_LIKE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$|^(True|False)$")


class StoreError(Exception):
    """Anything wrong with the campaign database."""


class StructureState(str, Enum):
    """Explicit states.  Pending work is found by these, never by a missing key.

    ASE's ``db.select('~key')`` does not return the complement of
    ``db.select('key')``, so "not yet computed" cannot be expressed as an absent
    key.  Every stage writes its state here instead.
    """

    new = "new"
    screening = "screening"
    screened = "screened"
    deduped = "deduped"
    filtered_out = "filtered_out"
    selected = "selected"
    dft_queued = "dft_queued"
    dft_running = "dft_running"
    dft_done = "dft_done"
    failed = "failed"


class Origin(str, Enum):
    generated = "generated"
    mp = "mp"
    seed = "seed"


@dataclass(frozen=True)
class CompositionRow:
    id: int
    formula: str
    chemsys: str
    z: int
    n_atoms: int
    n_target: int
    n_produced: int
    source_mode: str
    source_name: str
    state: str
    fail_reason: str = ""


def _clean_kv(kv: dict[str, Any]) -> dict[str, Any]:
    """Validate key-value pairs destined for ASE, with an actionable message."""
    out: dict[str, Any] = {}
    for key, value in kv.items():
        if value is None:
            raise StoreError(
                f"key {key!r} is None. ASE key_value_pairs cannot hold null; a value "
                f"that is not yet measured must be represented by a `state`, and a "
                f"failure by state='failed' plus a reason."
            )
        if isinstance(value, Enum):
            value = value.value
        if not isinstance(value, _ASE_SCALARS):
            raise StoreError(
                f"key {key!r} is a {type(value).__name__}. ASE key_value_pairs hold only "
                f"str/int/float/bool; put structured values in the `data=` blob instead "
                f"(they round-trip, but are not queryable)."
            )
        if isinstance(value, str) and _NUMBER_LIKE.match(value):
            raise StoreError(
                f"key {key!r} holds {value!r}, which ASE will not store: it refuses "
                f"any string key_value_pair that int() or float() would parse, "
                f"because the value would come back out as a number. This bites "
                f"hash-like values -- a bare 16-character hex digest is number-like "
                f"about once in 600 (all digits, or digits-e-digits) -- so prefix "
                f"them with what they are, e.g. 'sha256:{value}', which cannot parse "
                f"as a number by construction. Caught here so the message names the "
                f"key; ASE's own error names only the value."
            )
        if key in _ASE_RESERVED:
            raise StoreError(
                f"key {key!r} is reserved by ASE (it reserves every element symbol "
                f"plus its own row attributes such as formula/energy/magmom/natoms/"
                f"id/user). Writing it raises `ValueError: Bad key: {key}` from inside "
                f"ASE. Prefix or qualify the name -- e.g. 'reduced_formula', "
                f"'vasp_energy' -- so it cannot collide with a column ASE owns."
            )
        if _looks_like_a_formula(key):
            raise StoreError(
                f"key {key!r} parses as a chemical formula. ASE warns about this rather "
                f"than refusing it, and the consequence is silent: db.select({key!r}) "
                f"returns rows CONTAINING those elements, not rows carrying this key, so "
                f"the query looks like it works and returns the wrong set. Rename the key."
            )
        out[key] = value
    return out


def _looks_like_a_formula(key: str) -> bool:
    try:
        from ase.formula import Formula
    except ImportError:                                  # pragma: no cover
        return False
    try:
        Formula(key, strict=True)
    except (ValueError, KeyError):
        return False
    return True


def _git_sha(repo: Path | None = None) -> str:
    root = repo or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


# Filesystems where SQLite's WAL mode is unsafe.  WAL keeps its index in a
# shared-memory file (`-shm`) that every connection mmaps; a network filesystem
# provides no coherent shared memory, and SQLite's own documentation says WAL
# "does not work" there.  It does not fail loudly -- it corrupts.
#
# Measured here 2026-09-12: the CePdGe campaign database on
# aqu-fs10:/exports/... (nfs4) went from working at cycle 3 to
# `DatabaseError: file is not a database` at cycle 4.  2,560,000 bytes intact on
# disk, page 1 overwritten with a leaf-page header, `sqlite3 .recover` unable to
# open it at all.  The exposure is wider than one driver: `Store` holds TWO
# independent connections to the file (its own `sqlite3` and a fresh
# `ase_connect` per operation), and a campaign is routinely touched from the
# login node by `csp status` while its driver runs on a compute node -- two NFS
# clients, one file, no shared memory between them.
_NETWORK_FS = frozenset({
    "nfs", "nfs4", "cifs", "smb2", "smb3", "lustre", "gpfs", "beegfs",
    "afs", "ceph", "fuse.sshfs", "fuse.glusterfs", "9p",
})


def _filesystem_type(path: Path) -> str:
    """The fstype of the filesystem holding `path` ('' if it cannot be told)."""
    try:
        target = path.resolve()
        if not target.exists():
            target = target.parent
        best, kind = "", ""
        with open("/proc/mounts", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount, fstype = parts[1], parts[2]
                if (str(target) == mount or str(target).startswith(mount.rstrip("/") + "/")) \
                        and len(mount) > len(best):
                    best, kind = mount, fstype
        return kind
    except Exception:                                          # noqa: BLE001
        return ""


def journal_mode_for(path: Path) -> str:
    """WAL on a local disk, a rollback journal on a network one.

    TRUNCATE rather than DELETE because it rewrites one existing file instead of
    creating and unlinking one per transaction, which is markedly less work over
    NFS -- and unlike WAL it uses only POSIX locks, which NFSv4 does provide.
    """
    return "TRUNCATE" if _filesystem_type(path) in _NETWORK_FS else "WAL"


# SQLite primary result codes that mean "the storage blinked", as opposed to
# "the file has rotted". The distinction is the whole point: one is waited out,
# the other is fatal, and until D148 both raised `sqlite3.DatabaseError` and
# were handled as if they were the second.
_SQLITE_BUSY, _SQLITE_IOERR, _SQLITE_CANTOPEN = 5, 10, 14
_SQLITE_CORRUPT, _SQLITE_NOTADB = 11, 26

#: Matched against the message text, because `sqlite3.Error.sqlite_errorcode`
#: only exists on Python 3.11+ and the `cspflow` env runs 3.10. Checked in
#: full-lowercase against the message SQLite itself produces.
_TRANSIENT_TEXT = (
    "disk i/o error",                # SQLITE_IOERR
    "unable to open database file",  # SQLITE_CANTOPEN
    "database is locked",            # SQLITE_BUSY escaping the busy timeout
    "database table is locked",
)
_FATAL_TEXT = (
    "file is not a database",
    "database disk image is malformed",
    "malformed database schema",
)


def is_transient_error(exc: BaseException) -> bool:
    """True when a database error is the storage blinking, not the file rotting.

    `/scratch` is NFSv4 with `local_lock=none`, so every lock lives on the
    server. A `hard` mount makes reads and writes block and retry forever, which
    is why it is easy to assume the driver is safe -- but `hard` says nothing
    about LOCK state. When a lease expires during an outage the client cannot
    reclaim its locks and the next operation returns EIO, which SQLite reports
    as `SQLITE_IOERR`, whose message text is the bare string "disk I/O error".

    Measured 2026-09-19: aqu-fs10 went unresponsive 17:25:10-17:28:53 and the
    RE-magnets-CHGNet driver exited at 17:30:43, at cycle 32 of a healthy run.
    The database was untouched -- `PRAGMA integrity_check` returned `ok` and the
    file's MD5 was unchanged -- but 36 DFT jobs then completed over the next
    three hours with no driver alive to reconcile them.

    Note that `PRAGMA busy_timeout` is NOT protection here despite being set to
    five minutes: SQLite invokes the busy handler only for SQLITE_BUSY on lock
    contention, and never for SQLITE_IOERR.
    """
    if not isinstance(exc, sqlite3.Error):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None:                                   # Python 3.11+
        primary = int(code) & 0xFF
        if primary in (_SQLITE_CORRUPT, _SQLITE_NOTADB):
            return False
        return primary in (_SQLITE_BUSY, _SQLITE_IOERR, _SQLITE_CANTOPEN)
    text = str(exc).strip().lower()
    # Fatal wins over transient: a corrupt file must never be retried into
    # looking like a network problem.
    if any(t in text for t in _FATAL_TEXT):
        return False
    return any(t in text for t in _TRANSIENT_TEXT)


def _no_schema_message(path: Path) -> str:
    """Why an existing database can read back as empty, and what to do.

    Almost always this is not a broken file. It is a WAL database being read
    from a DIFFERENT MACHINE than the one holding it: the `-wal` file carries
    every recent write and its index lives in `-shm`, which is shared memory and
    coherent only within one host. A second NFS client sees the stale main file
    instead -- which, for a campaign whose whole history is still in the WAL,
    has no schema_version in it.

    Seen 2026-09-12: `csp status` on str-c6 against a CeFeB database whose
    driver held it on str-c221. Main file two hours stale, `-wal` 4.2 MB and
    thirty seconds old. The campaign itself was untouched and still running.
    """
    wal = Path(str(path) + "-wal")
    lines = [f"{path} did not read back as a cspflow database (no schema_version)."]
    if wal.is_file() and wal.stat().st_size:
        main_age = path.stat().st_mtime
        wal_age = wal.stat().st_mtime
        lines += [
            f"  A write-ahead log is present and holds {wal.stat().st_size:,} bytes"
            f" of newer data:",
            f"      {path.name}      last written {_when(main_age)}",
            f"      {wal.name}  last written {_when(wal_age)}",
            "  That almost always means this database is open on ANOTHER NODE and you",
            "  are reading it from this one. WAL needs shared memory, which a network",
            "  filesystem cannot provide between hosts, so you are seeing the stale",
            "  main file rather than the live campaign.",
            "  Your data is fine. Read it from the node holding it (`squeue` shows",
            "  which), or read the driver's status.json, or wait for the driver to",
            "  finish and checkpoint.",
        ]
    else:
        lines.append("  The file exists but carries no cspflow schema. If it was created"
                     " by something else, move it aside and re-run the campaign.")
    return "\n".join(lines)


def _when(stamp: float) -> str:
    import time as _time

    return _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(stamp))


def wal_lag(path: Path) -> float | None:
    """Seconds by which a non-empty `-wal` is newer than the main database.

    `None` means there is nothing to worry about: no WAL, an empty one, or a
    main file at least as new.

    This is the check that separates "I can read this database" from "I can read
    the CURRENT database", and only the second one is worth anything. On a
    network filesystem the `-shm` index that makes a WAL readable is coherent
    only within one host, so a second node reads the main file as it stood at
    the last checkpoint. That read SUCCEEDS -- it simply answers with an old
    campaign. A silently stale answer is worse than an error, because nothing
    about it looks wrong: `csp status` prints a clean table of numbers that
    stopped being true hours ago.

    Both failure shapes come from the same cause, so both are detected here:
      * schema_version missing entirely -> `Store.open` raises (see
        `_no_schema_message`), which happens when the campaign is young enough
        that even its schema is still in the WAL;
      * schema_version present but everything after it stale -> no exception at
        all, and this function is the only thing that catches it.
    """
    wal = Path(f"{path}-wal")
    try:
        if not path.is_file() or not wal.is_file() or wal.stat().st_size == 0:
            return None
        lag = wal.stat().st_mtime - path.stat().st_mtime
    except OSError:
        return None
    return lag if lag > 0 else None


def _external_tables_once(db) -> None:
    """Stop ASE querying its external-table list once PER ROW it returns.

    `SQLite3Database._convert_tuple_to_row` calls `_get_external_table_names()`
    for every row of every `select()` -- one extra query per structure, 1.3 ms
    each on nfs4, 19 of the 22 seconds it took to read 14,752 screened
    structures even on one connection (D144).

    cspflow never creates ASE external tables: `_clean_kv` refuses the dict an
    `external_tables=` value would need. So the list is read once and reused. A
    guard, not an assumption -- if a future ASE renames the method, this does
    nothing and reads are merely slower.
    """
    getter = getattr(db, "_get_external_table_names", None)
    if getter is None:
        return
    try:
        names = list(getter())
    except Exception:                                          # noqa: BLE001
        return
    db._get_external_table_names = lambda db_con=None: list(names)


class Store:
    """A campaign database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._sql: sqlite3.Connection | None = None
        self._ase = None
        # Depth of `transaction()` blocks. While > 0, writes accumulate and
        # nothing commits until the outermost block exits (D144).
        self._tx_depth = 0
        # The mode actually in force, filled in when the connection opens. It
        # is not always the mode we asked for -- see `_set_journal_mode`.
        self.journal_mode: str = ""

    # -- lifecycle ---------------------------------------------------------

    @property
    def sql(self) -> sqlite3.Connection:
        if self._sql is None:
            self._sql = sqlite3.connect(str(self.path), timeout=BUSY_TIMEOUT_MS / 1000)
            self._sql.row_factory = sqlite3.Row
            self._sql.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            want = journal_mode_for(self.path)
            self.journal_mode = self._set_journal_mode(want)
            self._sql.execute("PRAGMA foreign_keys=ON")
            # On a network filesystem the rollback journal is only as safe as
            # the flushes behind it, so do not let the OS reorder them.
            if self.journal_mode.upper() != "WAL":
                self._sql.execute("PRAGMA synchronous=FULL")
            # 64 MB of page cache, held for the life of the Store. On NFS every
            # page SQLite has to re-read is a network round trip; with ASE on
            # this one connection (D144) the cache now survives between
            # operations instead of being thrown away with each connection.
            # It also keeps a large transaction from spilling to disk part-way,
            # which would take the EXCLUSIVE lock early and block readers.
            self._sql.execute("PRAGMA cache_size=-65536")
        return self._sql

    def _set_journal_mode(self, want: str) -> str:
        """Ask for a journal mode; report what was actually granted.

        Asking is not getting, and `PRAGMA journal_mode` is the worst kind of
        API for that: the mode is persisted in the database header, and SQLite
        will not change it while any other connection has the file open. What
        happens then is version- and timing-dependent -- it either returns the
        OLD mode as an ordinary result row, with no exception, or it raises
        `database is locked`. Executing the pragma and ignoring both is how the
        CeFeB and CePdGe campaigns stayed in WAL on NFS after the guard that was
        supposed to prevent exactly that had already been added: the driver had
        the file open, so every later connection's request was declined in
        silence.

        So return the granted mode rather than the requested one, and never let
        a declined request raise. A reader that cannot switch the mode can still
        read; refusing to open at all would take away the last way to see a
        campaign whose driver is holding it.
        """
        try:
            row = self._sql.execute(f"PRAGMA journal_mode={want}").fetchone()
            got = (row[0] if row else want) or want
        except sqlite3.OperationalError:
            # Declined by a concurrent holder. Read the mode in force instead.
            try:
                row = self._sql.execute("PRAGMA journal_mode").fetchone()
                got = (row[0] if row else "") or "unknown"
            except sqlite3.OperationalError:
                got = "unknown"
        return str(got)

    @property
    def ase(self):
        """ASE's view of this database, running on this Store's OWN connection.

        D144 replaced D015 ("a fresh ASE connection per operation"). That rule
        protected array workers from a driver's write lock, and D056 has since
        made workers read-only, so it was paying for a hazard that no longer
        exists -- at a cost measured on RE-magnets-CHGNet (14,752 structures,
        nfs4):

        * `select()` opened a NEW sqlite connection for EVERY ROW it returned
          (ASE looks up its external-table names per row, and outside a
          `with db:` block each lookup connects afresh): 14,753 connections
          and 59 s to read the screened set, against 22 s on one connection.
        * every `update()` was two connections, a lock file, and a commit with
          `synchronous=FULL`: 702 ms per structure.
        * two connections in one process deadlock under a rollback journal the
          moment one holds a write while the other wants one -- which is why
          `legacy.py` needed a load-bearing `gc.collect()`.

        So ASE is handed `self.sql`: one process, one connection, one
        transaction scope shared by the SQL tables and the structure rows.

        `use_lock_file=False`: ASE's lock file is a second, file-based lock on
        top of SQLite's own, acquired with `timeout=inf` and doubling back-off.
        A process killed while holding it left `campaign.db.lock` behind and
        every later open waited forever in silence (2026-09-15). SQLite's POSIX
        locks, which NFSv4 provides, are released when the process dies.

        On a shared connection ASE commits by itself only every 5,000
        operations, so every write wrapper below commits through `_commit()`,
        which defers to an enclosing `transaction()`.
        """
        if self._ase is None:
            db = ase_connect(str(self.path), serial=True, use_lock_file=False)
            db.connection = self.sql
            db.change_count = 0
            _external_tables_once(db)
            self._ase = db
        return self._ase

    @contextmanager
    def transaction(self) -> Iterator["Store"]:
        """Commit everything written inside the block once, or none of it.

        Two reasons, and they are separate. SPEED: on NFS a commit is an fsync
        round trip, and per-structure loops of them were most of a nine-hour
        reconcile. ATOMICITY: a finished job's state and the results folded
        from it now land together, so a driver killed half-way leaves the job
        un-reconciled -- and it is reconciled again next cycle -- instead of
        `done` with half its results written.

        Nesting is by depth: only the outermost block commits or rolls back. An
        exception that escapes the outermost block rolls everything back and
        propagates; one caught INSIDE the block does not undo writes made before
        it, exactly as the per-write commits it replaces did not.
        """
        self._tx_depth += 1
        try:
            yield self
        except BaseException:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.sql.rollback()
            raise
        self._tx_depth -= 1
        if self._tx_depth == 0:
            self.sql.commit()

    @property
    def in_transaction(self) -> bool:
        return self._tx_depth > 0

    def _commit(self) -> None:
        """Commit now, unless an enclosing `transaction()` will."""
        if self._tx_depth == 0:
            self.sql.commit()

    def commit(self) -> None:
        """`_commit` for callers outside this module that run raw SQL."""
        self._commit()

    @staticmethod
    def _check_sidecars(path: Path) -> None:
        """Refuse an orphaned WAL, with the fix named.

        The database runs in WAL mode, so it is really three files: `x.db`,
        `x.db-wal` and `x.db-shm`. Deleting only `x.db` -- the obvious way to
        start a campaign over -- leaves the other two, and SQLite then fails
        with a bare

            OperationalError: disk I/O error

        which says nothing whatever about the cause. Hit while testing `csp run`
        after `rm -f campaign.db`, which is exactly what a user would type.
        """
        orphans = [p for p in (Path(f"{path}-wal"), Path(f"{path}-shm")) if p.exists()]
        if orphans and not path.exists():
            names = ", ".join(p.name for p in orphans)
            raise StoreError(
                f"{path} does not exist but its write-ahead log does ({names}). "
                f"SQLite reports this as a bare 'disk I/O error'. The database is "
                f"three files in WAL mode; remove the leftovers too:\n"
                f"    rm -f {path}-wal {path}-shm"
            )

    @classmethod
    def create(cls, path: str | Path, *, campaign: str, config_hash: str = "") -> "Store":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cls._check_sidecars(path)
        store = cls(path)
        store.sql.executescript(SCHEMA_PATH.read_text())
        # Touch the ASE side so both halves exist from the start; otherwise the
        # first structure write creates ASE's tables at an arbitrary later time.
        _ = store.ase.count()
        store._add_missing_indices()
        store.set_meta("schema_version", SCHEMA_VERSION)
        store.set_meta("campaign", campaign)
        store.set_meta("config_hash", config_hash)
        store.sql.commit()
        return store

    #: Columns added after the schema version was first stamped. Each is
    #: additive with a default, so an older cspflow reading the same file is
    #: unaffected -- `SELECT *` simply returns one more column.
    #:
    #: They are applied rather than version-bumped because `open()` REFUSES a
    #: version mismatch ("migration is required; refusing to guess"), and a bump
    #: would lock every existing campaign out of its own database. A 14,755
    #: structure campaign is not worth losing to a bookkeeping column.
    _LATE_COLUMNS = (
        ("job", "cores", "INTEGER NOT NULL DEFAULT 0"),
    )

    def _add_missing_columns(self) -> None:
        """Add any `_LATE_COLUMNS` this file does not have yet. Idempotent."""
        for table, column, decl in self._LATE_COLUMNS:
            try:
                have = {r[1] for r in self.sql.execute(f"PRAGMA table_info({table})")}
                if column not in have:
                    self.sql.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                    self.sql.commit()
            except Exception:                                  # noqa: BLE001
                pass

    #: Indexes ASE's own schema lacks (D144). ASE indexes its key-value tables
    #: by KEY only, but every `update()` begins `DELETE FROM <table> WHERE id
    #: IN (...)` on keys, text_key_values and number_key_values -- and a write
    #: with atoms does species too. Without an index on `id` each is a full
    #: table scan, so one structure update scanned every key-value row in the
    #: campaign, and got slower as the campaign grew: 702 ms on nfs4 at 14,752
    #: structures, 46 ms with these. Additive, like `_LATE_COLUMNS`, and for the
    #: same reason: a version bump would lock existing campaigns out.
    _LATE_INDICES = (
        ("ix_ase_keys_id", "keys", "id"),
        ("ix_ase_text_id", "text_key_values", "id"),
        ("ix_ase_number_id", "number_key_values", "id"),
        ("ix_ase_species_id", "species", "id"),
    )

    def _add_missing_indices(self) -> None:
        """Create any `_LATE_INDICES` that are missing. Idempotent, never fatal.

        Reads `sqlite_master` first, so opening an already-indexed database
        never asks for a write lock. When one IS missing and another process
        holds the file, give up after two seconds rather than thirty: the next
        open tries again, and `csp status` must not stall behind a busy driver
        in order to add an index.
        """
        try:
            have = {r[0] for r in self.sql.execute(
                "SELECT name FROM sqlite_master WHERE type='index'")}
            tables = {r[0] for r in self.sql.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        except Exception:                                      # noqa: BLE001
            return
        missing = [(n, t, c) for n, t, c in self._LATE_INDICES
                   if n not in have and t in tables]
        if not missing:
            return
        try:
            self.sql.execute("PRAGMA busy_timeout=2000")
            for name, table, column in missing:
                self.sql.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}({column})")
            self.sql.commit()
        except Exception:                                      # noqa: BLE001
            try:
                self.sql.rollback()
            except Exception:                                  # noqa: BLE001
                pass
        finally:
            try:
                self.sql.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            except Exception:                                  # noqa: BLE001
                pass

    @classmethod
    def open(cls, path: str | Path) -> "Store":
        path = Path(path)
        cls._check_sidecars(path)
        if not path.is_file():
            raise StoreError(f"no campaign database at {path}. Run `csp init` first.")
        store = cls(path)
        store._add_missing_columns()
        store._add_missing_indices()
        found = store.meta("schema_version")
        if found is None:
            raise StoreError(_no_schema_message(path))
        if found != SCHEMA_VERSION:
            raise StoreError(
                f"{path} has schema version {found}, this cspflow expects "
                f"{SCHEMA_VERSION}. Migration is required; refusing to guess."
            )
        return store

    def close(self) -> None:
        self._ase = None
        if self._sql is not None:
            self._sql.commit()
            self._sql.close()
            self._sql = None

    def reconnect(self) -> None:
        """Drop the connection so the next use opens a fresh one.

        Deliberately NOT `close()`: that commits first, and the whole reason to
        reconnect is that the storage just refused a write. Committing would
        either fail again or, worse, half-apply. Anything uncommitted is
        discarded on purpose -- the caller's contract is that it re-runs the
        work, and a cycle is built to be re-runnable (D148).

        The ASE handle goes too. It rides on this same connection (D144), so a
        stale one would keep the dead file descriptor alive.
        """
        self._ase = None
        if self._sql is not None:
            try:
                self._sql.close()
            except sqlite3.Error:
                # The connection is being thrown away; a failure to close it
                # cleanly is not worth propagating over the original error.
                pass
            self._sql = None
        self._tx_depth = 0
        self.journal_mode = ""

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- meta --------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self.sql.execute(
            "INSERT INTO campaign_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
        self._commit()

    def meta(self, key: str) -> str | None:
        """Read campaign metadata.

        Returns None rather than raising when the table is absent: this is how
        `open` distinguishes a cspflow database from any other SQLite file, and
        a raw OperationalError there would be a worse message than the one
        `open` gives.
        """
        try:
            row = self.sql.execute(
                "SELECT value FROM campaign_meta WHERE key=?", (key,)
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return row["value"] if row else None

    # -- structures (ASE side) --------------------------------------------

    def add_structure(
        self,
        atoms: Atoms,
        *,
        origin: Origin | str,
        state: StructureState | str = StructureState.new,
        data: dict[str, Any] | None = None,
        **kv: Any,
    ) -> int:
        kv["origin"] = origin.value if isinstance(origin, Origin) else origin
        kv["state"] = state.value if isinstance(state, StructureState) else state
        sid = int(self.ase.write(atoms, data=data or {}, **_clean_kv(kv)))
        self._commit()
        return sid

    def update_structure(self, sid: int, *, data: dict[str, Any] | None = None,
                         delete_keys: list[str] | None = None, **kv: Any) -> None:
        extra: dict[str, Any] = {}
        if data is not None:
            extra["data"] = data
        if delete_keys:
            extra["delete_keys"] = list(delete_keys)
        self.ase.update(sid, **extra, **_clean_kv(kv))
        self._commit()

    def replace_geometry(self, sid: int, atoms) -> None:
        """Replace a row's CELL AND POSITIONS, keeping its id and its key-values.

        Used when the MLIP relaxation finishes: the relaxed cell is a far better
        starting point for the DFT relax than the seed as supplied, and until
        2026-09-12 it was discarded -- the worker recorded the energy and threw
        the geometry away, so `dft_stage` step 0 ran on the unrelaxed structure.
        The same defect at step > 0 had already been found and fixed there, with
        volumes measured 176.15 vs 179.03 A^3 before and after.

        The id is preserved because every claim, filter event and hull placement
        refers to it.
        """
        self.ase.update(sid, atoms=atoms)
        self._commit()

    def get_structure(self, sid: int):
        try:
            return self.ase.get(id=sid)
        except KeyError as exc:
            raise StoreError(f"no structure with id {sid}") from exc

    def structures(self, selection: str | None = None, **kwargs: Any) -> Iterator[Any]:
        yield from self.ase.select(selection, **kwargs)

    def count_structures(self, selection: str | None = None, **kwargs: Any) -> int:
        return int(self.ase.count(selection, **kwargs))

    def structure_ids(self, selection: str | None = None, **kwargs: Any) -> list[int]:
        return [int(r.id) for r in self.ase.select(selection, **kwargs)]

    def unplaced_mlip_candidates(self) -> int:
        """Screened structures with an MLIP energy and no MLIP hull placement.

        Plain SQL on ASE's key-value tables rather than `select()`, because this
        is asked on every driver cycle and the answer is a count (D144). A
        structure whose system could not be hulled carries `hull_error` and is
        not counted: counting it made the stage "pending" forever, so it re-ran
        a full placement every cycle and the driver never went idle.
        """
        row = self.sql.execute(
            "SELECT COUNT(*) FROM text_key_values s "
            "WHERE s.key='state' AND s.value=? "
            "  AND EXISTS (SELECT 1 FROM number_key_values e "
            "              WHERE e.id=s.id AND e.key='mlip_e_per_atom') "
            "  AND NOT EXISTS (SELECT 1 FROM hull h "
            "                  WHERE h.structure_id=s.id AND h.hull_type='mlip') "
            "  AND NOT EXISTS (SELECT 1 FROM keys k "
            "                  WHERE k.id=s.id AND k.key='hull_error')",
            (StructureState.screened.value,)).fetchone()
        return int(row[0] or 0)

    def count_in_state_lacking(self, state: StructureState | str, key: str) -> int:
        """Structures in `state` that do not carry `key` -- a count, in SQL.

        For per-cycle "is there work?" questions (D144). Answering them with
        `select()` read and decoded every structure in the state just to count
        the ones missing a flag: 59 s per ask on nfs4 at 14,752 structures,
        asked every cycle.
        """
        value = state.value if isinstance(state, StructureState) else state
        row = self.sql.execute(
            "SELECT COUNT(*) FROM text_key_values s "
            "WHERE s.key='state' AND s.value=? "
            "  AND NOT EXISTS (SELECT 1 FROM keys k WHERE k.id=s.id AND k.key=?)",
            (value, key)).fetchone()
        return int(row[0] or 0)

    def mlip_hull_hashes(self) -> dict[int, set[str]]:
        """structure id -> every reference-set hash it has been placed against."""
        out: dict[int, set[str]] = {}
        for sid, ref in self.sql.execute(
                "SELECT structure_id, ref_set_hash FROM hull WHERE hull_type='mlip'"):
            out.setdefault(int(sid), set()).add(str(ref))
        return out

    def set_structure_state(self, sid: int, state: StructureState | str, **kv: Any) -> None:
        self.update_structure(sid, state=state, **kv)

    # -- compositions ------------------------------------------------------

    def add_composition(
        self,
        *,
        formula: str,
        chemsys: str,
        z: int,
        n_atoms: int,
        n_target: int,
        source_mode: str,
        source_name: str,
        state: str = "new",
    ) -> int:
        cur = self.sql.execute(
            "INSERT INTO composition "
            "(formula, chemsys, z, n_atoms, n_target, source_mode, source_name, state) "
            "VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(formula, z, source_name) DO UPDATE SET "
            "  n_target=excluded.n_target, n_atoms=excluded.n_atoms",
            (formula, chemsys, z, n_atoms, n_target, source_mode, source_name, state),
        )
        self._commit()
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = self.sql.execute(
            "SELECT id FROM composition WHERE formula=? AND z=? AND source_name=?",
            (formula, z, source_name),
        ).fetchone()
        return int(row["id"])

    def compositions(self, *, state: str | None = None, chemsys: str | None = None) -> list[CompositionRow]:
        q = "SELECT * FROM composition WHERE 1=1"
        args: list[Any] = []
        if state is not None:
            q += " AND state=?"
            args.append(state)
        if chemsys is not None:
            q += " AND chemsys=?"
            args.append(chemsys)
        q += " ORDER BY id"
        return [
            CompositionRow(
                id=r["id"], formula=r["formula"], chemsys=r["chemsys"], z=r["z"],
                n_atoms=r["n_atoms"], n_target=r["n_target"],
                n_produced=r["n_produced"], source_mode=r["source_mode"],
                source_name=r["source_name"], state=r["state"],
                fail_reason=r["fail_reason"],
            )
            for r in self.sql.execute(q, args)
        ]

    def set_composition_state(self, cid: int, state: str, fail_reason: str = "",
                              n_produced: int | None = None) -> None:
        """Move a composition to a new state, optionally recording its yield.

        `n_produced` is separate from the state on purpose.  "generated" says
        the generator ran and returned something; it does not say it returned
        what was asked for.  A campaign that quietly gets 60% of its requested
        structures back looks identical, at the state level, to one that gets
        100% -- so the number is stored rather than inferred.
        """
        if n_produced is None:
            self.sql.execute(
                "UPDATE composition SET state=?, fail_reason=? WHERE id=?",
                (state, fail_reason, cid))
        else:
            self.sql.execute(
                "UPDATE composition SET state=?, fail_reason=?, n_produced=? WHERE id=?",
                (state, fail_reason, int(n_produced), cid))
        self._commit()

    def generation_yield(self) -> dict[str, int]:
        """Requested versus produced across every composition that has run.

        Reported by `csp status`, because a shortfall here is invisible further
        down: the funnel narrows anyway, and 40% fewer candidates entering it
        looks exactly like a campaign that was always going to be small.
        """
        row = self.sql.execute(
            "SELECT COUNT(*) AS n, "
            "       COALESCE(SUM(n_target), 0)   AS requested, "
            "       COALESCE(SUM(n_produced), 0) AS produced, "
            "       COALESCE(SUM(CASE WHEN n_produced < n_target THEN 1 ELSE 0 END), 0) AS short "
            "FROM composition WHERE state IN ('generated','done')").fetchone()
        return {"compositions": int(row["n"]), "requested": int(row["requested"]),
                "produced": int(row["produced"]), "short": int(row["short"])}

    def chemsystems(self) -> list[str]:
        return [r["chemsys"] for r in self.sql.execute(
            "SELECT DISTINCT chemsys FROM composition ORDER BY chemsys")]

    # -- provenance --------------------------------------------------------

    def add_provenance(
        self,
        *,
        config_hash: str,
        settings_hash: str = "",
        machine: str = "",
        code_version: str = "",
        resolved_config: dict[str, Any] | None = None,
    ) -> int:
        git_sha = _git_sha()
        blob = json.dumps(resolved_config or {}, sort_keys=True)
        self.sql.execute(
            "INSERT OR IGNORE INTO provenance "
            "(config_hash, settings_hash, git_sha, code_version, machine, resolved_config_json) "
            "VALUES (?,?,?,?,?,?)",
            (config_hash, settings_hash, git_sha, code_version, machine, blob),
        )
        self._commit()
        row = self.sql.execute(
            "SELECT id FROM provenance WHERE config_hash=? AND settings_hash=? AND git_sha=?",
            (config_hash, settings_hash, git_sha),
        ).fetchone()
        return int(row["id"])

    # -- hull --------------------------------------------------------------

    def add_hull(
        self,
        *,
        structure_id: int,
        hull_type: str,
        energy_scale: str,
        e_above_hull: float,
        ref_set_hash: str,
        formation_energy: float | None = None,
        settings_hash: str = "",
    ) -> int:
        """Record a hull placement.

        The (hull_type, energy_scale, ref_set_hash, settings_hash) tuple is part
        of the identity.  `assert_hull_consistent` is what actually enforces
        that two scales never enter one hull; this method stores the labels that
        make the check possible.
        """
        cur = self.sql.execute(
            "INSERT INTO hull (structure_id, hull_type, energy_scale, e_above_hull, "
            "formation_energy, ref_set_hash, settings_hash) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(structure_id, hull_type, energy_scale, ref_set_hash) DO UPDATE SET "
            "  e_above_hull=excluded.e_above_hull, formation_energy=excluded.formation_energy, "
            "  settings_hash=excluded.settings_hash",
            (structure_id, hull_type, energy_scale, e_above_hull, formation_energy,
             ref_set_hash, settings_hash),
        )
        self._commit()
        return int(cur.lastrowid or 0)

    def assert_hull_consistent(self, ref_set_hash: str) -> None:
        """Refuse a hull built from mixed energy scales or mixed settings.

        This is the write-time form of the rule stated in Stages 3b and 3c: two
        energies with different settings_hash may never enter the same hull.
        Enforced here so no analysis path can bypass it.
        """
        rows = self.sql.execute(
            "SELECT DISTINCT energy_scale, settings_hash FROM hull WHERE ref_set_hash=?",
            (ref_set_hash,),
        ).fetchall()
        scales = {r["energy_scale"] for r in rows}
        settings = {r["settings_hash"] for r in rows}
        if len(scales) > 1:
            raise StoreError(
                f"hull {ref_set_hash[:12]} mixes energy scales {sorted(scales)}. "
                f"Raw and MP-corrected energies may not enter the same hull."
            )
        if len(settings) > 1:
            raise StoreError(
                f"hull {ref_set_hash[:12]} mixes settings_hash {sorted(s[:12] for s in settings)}. "
                f"Energies from different INCAR/POTCAR settings may not enter the same hull."
            )

    # -- reference ---------------------------------------------------------

    def add_reference_entry(self, **fields: Any) -> int:
        cols = [
            "mp_id", "chemsys", "thermo_type", "run_type", "formula", "n_atoms",
            "e_dft_raw", "e_dft_corrected", "correction", "e_mlip_static",
            "e_mlip_relaxed", "volume_drift", "rmsd", "structure_id", "snapshot_id",
            "state", "fail_reason",
        ]
        unknown = set(fields) - set(cols)
        if unknown:
            raise StoreError(f"unknown reference_entry field(s): {sorted(unknown)}")
        present = [c for c in cols if c in fields]
        sql = (
            f"INSERT INTO reference_entry ({','.join(present)}) "
            f"VALUES ({','.join('?' * len(present))}) "
            f"ON CONFLICT(mp_id, thermo_type, snapshot_id) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in present if c != "mp_id")
        )
        self.sql.execute(sql, [fields[c] for c in present])
        self._commit()
        row = self.sql.execute(
            "SELECT id FROM reference_entry WHERE mp_id=? AND thermo_type=? AND snapshot_id=?",
            (fields["mp_id"], fields["thermo_type"], fields.get("snapshot_id", "")),
        ).fetchone()
        return int(row["id"])

    def update_reference_entry(self, entry_id: int, **fields: Any) -> None:
        """Amend an existing reference entry in place.

        Distinct from `add_reference_entry`, which is an upsert and therefore
        has to supply every NOT NULL column. That makes it unable to express
        "record the MLIP energy for an entry that already exists": SQLite
        evaluates the INSERT before the ON CONFLICT clause, so the row fails on
        `chemsys` being NULL even though the conflicting row has one.

        Found when Stage 4a tried exactly that.
        """
        if not fields:
            return
        allowed = {
            "run_type", "formula", "n_atoms", "e_dft_raw", "e_dft_corrected",
            "correction", "e_mlip_static", "e_mlip_relaxed", "volume_drift",
            "rmsd", "structure_id", "state", "fail_reason",
        }
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise StoreError(
                f"unknown or immutable reference_entry field(s): {unknown}. "
                f"Identity columns (mp_id, thermo_type, snapshot_id, chemsys) are not "
                f"amendable -- a row whose identity changed is a different row."
            )
        assignments = ", ".join(f"{k}=?" for k in fields)
        self.sql.execute(f"UPDATE reference_entry SET {assignments} WHERE id=?",
                         [*fields.values(), entry_id])
        self._commit()

    def reference_entries(
        self, *, chemsys: str | None = None, include_subsystems: bool = False
    ) -> list[sqlite3.Row]:
        """Reference phases, optionally including every sub-system.

        `include_subsystems` matters more than it looks, and defaults to False
        only because an exact-match query is the less surprising default.

        An elemental Fe entry's `chemsys` is `'Fe'`, not `'Fe-Sm'`. So
        `reference_entries(chemsys='Fe-Sm')` returns the binary compounds and
        **no elemental references at all** -- which is precisely the set the
        hull's elemental guard (D058) refuses, and precisely the trap D061 had
        to fix at the MP API level. It reappears here because the database
        stores each entry under its own chemical system, correctly.

        Anything building a hull wants `include_subsystems=True`.
        """
        q = "SELECT * FROM reference_entry"
        args: list[Any] = []
        if chemsys is not None:
            wanted = self._subsystems(chemsys) if include_subsystems else [chemsys]
            q += f" WHERE chemsys IN ({','.join('?' * len(wanted))})"
            args.extend(wanted)
        return list(self.sql.execute(q + " ORDER BY id", args))

    @staticmethod
    def _subsystems(chemsys: str) -> list[str]:
        from itertools import combinations

        elements = sorted(e for e in chemsys.split("-") if e)
        return ["-".join(c) for n in range(1, len(elements) + 1)
                for c in combinations(elements, n)]

    def assert_single_thermo_type(self) -> str:
        """MP silently mixes functionals; a reference set may contain only one.

        SmFe2 comes back as -7.1966 eV/atom under GGA and -19.4095 under
        r2SCAN, with no field in MP's summary response saying which you got.
        """
        rows = self.sql.execute(
            "SELECT DISTINCT thermo_type FROM reference_entry"
        ).fetchall()
        kinds = sorted(r["thermo_type"] for r in rows)
        if len(kinds) > 1:
            raise StoreError(
                f"reference set mixes functionals {kinds}. MP's summary endpoint "
                f"returns whichever functional it prefers per material with no field "
                f"saying which; a mixed set puts a multi-eV/atom discontinuity into "
                f"the hull. Pin reference.thermo_type and refetch."
            )
        return kinds[0] if kinds else ""

    # -- jobs --------------------------------------------------------------

    def add_job(self, *, stage: str, structure_id: int | None = None,
                recipe_step: str = "", workdir: str = "",
                provenance_id: int | None = None, cores: int = 0) -> int:
        cur = self.sql.execute(
            "INSERT INTO job (structure_id, stage, recipe_step, workdir, "
            "provenance_id, cores) VALUES (?,?,?,?,?,?)",
            (structure_id, stage, recipe_step, workdir, provenance_id, int(cores)),
        )
        self._commit()
        return int(cur.lastrowid or 0)

    def update_job(self, job_id: int, **fields: Any) -> None:
        allowed = {
            "slurm_id", "array_task_id", "state", "attempt", "workdir",
            "core_hours", "exit_reason", "remedy", "submitted_at", "finished_at",
            "cores",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise StoreError(f"unknown job field(s): {sorted(unknown)}")
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        self.sql.execute(f"UPDATE job SET {sets} WHERE id=?", [*fields.values(), job_id])
        self._commit()

    def orphan_jobs(self) -> int:
        """Job rows written but never stamped with a scheduler id.

        The driver writes its rows before submitting so that a process killed
        mid-submission leaves evidence rather than nothing; this is how that
        evidence is read back. A non-zero count means a job may be running that
        no cycle will ever reconcile.
        """
        row = self.sql.execute(
            "SELECT COUNT(*) AS n FROM job WHERE slurm_id='' AND state='pending'"
        ).fetchone()
        return int(row["n"])

    def assert_job_id_is_new(self, slurm_id: str, stage: str, workdir: str) -> None:
        """Refuse a scheduler id that already belongs to some other submission.

        One array submission legitimately writes many job rows under a single
        id -- that is how the funnel's chunked stages work -- so an id is not
        unique by itself.  What must never happen is the *same* id naming two
        different submissions, because reconciliation groups rows by id and
        would then apply one job's outcome to the other's rows.

        Real SLURM ids are globally unique, so this only ever fires for a local
        run whose scheduler restarted its counter.  It is checked here anyway:
        nothing downstream can tell a reused id from an array, and the failure
        it produces is silent.
        """
        row = self.sql.execute(
            "SELECT stage, workdir FROM job WHERE slurm_id=? AND "
            "(stage!=? OR workdir!=?) LIMIT 1", (slurm_id, stage, workdir)).fetchone()
        if row is not None:
            raise StoreError(
                f"scheduler id {slurm_id!r} is already recorded for stage "
                f"{row['stage']!r} in {row['workdir']!r}; this submission is "
                f"stage {stage!r} in {workdir!r}. Two submissions under one id "
                f"would be reconciled against each other."
            )

    def jobs(self, *, state: str | None = None, stage: str | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM job WHERE 1=1", []
        if state is not None:
            q += " AND state=?"
            args.append(state)
        if stage is not None:
            q += " AND stage=?"
            args.append(stage)
        return list(self.sql.execute(q + " ORDER BY id", args))

    def count_jobs_by_state(self, stage: str | None = None) -> dict[str, int]:
        q = "SELECT state, COUNT(*) n FROM job"
        args: list[Any] = []
        if stage is not None:
            q += " WHERE stage=?"
            args.append(stage)
        return {r["state"]: r["n"] for r in self.sql.execute(q + " GROUP BY state", args)}

    # -- filter events, properties, calibration ----------------------------

    def add_filter_event(self, *, structure_id: int, gate: str, passed: bool,
                         value: float | None = None, threshold: float | None = None,
                         detail: str = "") -> None:
        self.sql.execute(
            "INSERT INTO filter_event (structure_id, gate, passed, value, threshold, detail) "
            "VALUES (?,?,?,?,?,?)",
            (structure_id, gate, int(passed), value, threshold, detail),
        )
        self._commit()

    def filter_events(self, structure_id: int) -> list[sqlite3.Row]:
        return list(self.sql.execute(
            "SELECT * FROM filter_event WHERE structure_id=? ORDER BY id", (structure_id,)))

    def add_property(self, *, structure_id: int, key: str, source: str,
                     value: float | None = None, text_value: str = "") -> None:
        self.sql.execute(
            "INSERT INTO property (structure_id, key, value, text_value, source) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(structure_id, key, source) DO UPDATE SET "
            "  value=excluded.value, text_value=excluded.text_value",
            (structure_id, key, value, text_value, source),
        )
        self._commit()

    def properties(self, structure_id: int) -> list[sqlite3.Row]:
        return list(self.sql.execute(
            "SELECT * FROM property WHERE structure_id=? ORDER BY key", (structure_id,)))

    def add_calibration(self, *, kind: str, n_points: int, verdict: str,
                        provenance_id: int | None = None, **metrics: Any) -> int:
        allowed = {"mae_e_per_atom", "mae_e_hull", "spearman", "volume_drift",
                   "alpha", "beta", "detail"}
        unknown = set(metrics) - allowed
        if unknown:
            raise StoreError(f"unknown calibration field(s): {sorted(unknown)}")
        cols = ["kind", "n_points", "verdict", "provenance_id", *metrics]
        vals = [kind, n_points, verdict, provenance_id, *metrics.values()]
        cur = self.sql.execute(
            f"INSERT INTO calibration ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            vals,
        )
        self._commit()
        return int(cur.lastrowid or 0)

    def latest_calibration(self, kind: str) -> sqlite3.Row | None:
        return self.sql.execute(
            "SELECT * FROM calibration WHERE kind=? ORDER BY id DESC LIMIT 1", (kind,)
        ).fetchone()

    # -- relaxations -------------------------------------------------------

    def add_relaxation(self, *, structure_id: int, engine: str, energy: float | None = None,
                       e_per_atom: float | None = None, converged: bool = False,
                       n_steps: int = 0, volume_before: float | None = None,
                       volume_after: float | None = None,
                       provenance_id: int | None = None) -> int:
        cur = self.sql.execute(
            "INSERT INTO relaxation (structure_id, engine, energy, e_per_atom, converged, "
            "n_steps, volume_before, volume_after, provenance_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (structure_id, engine, energy, e_per_atom, int(converged), n_steps,
             volume_before, volume_after, provenance_id),
        )
        self._commit()
        return int(cur.lastrowid or 0)

    # -- summary -----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "campaign": self.meta("campaign"),
            "structures": self.count_structures(),
            "compositions": self.sql.execute("SELECT COUNT(*) n FROM composition").fetchone()["n"],
            "chemsystems": len(self.chemsystems()),
            "reference_entries": self.sql.execute(
                "SELECT COUNT(*) n FROM reference_entry").fetchone()["n"],
            "jobs": self.count_jobs_by_state(),
            "structures_by_state": {
                r["state"]: r["n"] for r in self.sql.execute(
                    "SELECT value AS state, COUNT(*) n FROM text_key_values "
                    "WHERE key='state' GROUP BY value")
            },
            "relaxations": self.relaxation_outcomes(),
            "core_hours": float(self.sql.execute(
                "SELECT COALESCE(SUM(core_hours), 0.0) h FROM job").fetchone()["h"]),
        }

    def relaxation_outcomes(self) -> dict[str, int]:
        """Converged vs. not, per engine, counted in STRUCTURES -- never folded
        into a single "done".

        This is D027 surfaced where a user will actually see it. Over 106 jobs
        in `redo-new-ter-mag`, 100% exited cleanly and only 39% reached required
        accuracy; a status line reading "106 done" describes the process
        faithfully and the science not at all.

        **Counted in structures, not rows (D134).** `relaxation` is append-only
        attempt history: a structure that hits the ionic step limit, is retried
        from CONTCAR and then converges holds TWO rows, one of each kind. The
        original `COUNT(*)` reported both, so CePdGe read

            vasp:relax:converged      157
            vasp:relax:not converged   34   <- not usable as a relaxed geometry

        while `campaign_audit.py` found nothing wrong. Neither was lying. Those
        34 rows were 30 structures (4 had failed twice), of which 6 had SINCE
        converged and were `dft_done`. Only 24 structures actually lacked a
        converged relax, and every one was already re-running.

        A count of failed attempts is not a count of broken structures, and the
        line is read as the second. So a structure counts as converged if it has
        any converged record for that engine -- a later success supersedes an
        earlier failure -- and `:retried` reports how many got there the hard
        way, because those attempts cost core-hours and are worth seeing.
        """
        out: dict[str, int] = {}
        for row in self.sql.execute(
            "SELECT engine, "
            "       COUNT(DISTINCT structure_id) AS seen, "
            "       COUNT(DISTINCT CASE WHEN converged THEN structure_id END) AS ok, "
            "       COUNT(DISTINCT CASE WHEN NOT converged THEN structure_id END) AS failed "
            "FROM relaxation GROUP BY engine ORDER BY engine"
        ):
            engine, seen, ok = row["engine"], row["seen"], row["ok"]
            if ok:
                out[f"{engine}:converged"] = ok
            if seen - ok:
                out[f"{engine}:not converged"] = seen - ok
            # Inclusion-exclusion: in both sets = |ok| + |failed| - |union|.
            retried = ok + row["failed"] - seen
            if retried:
                out[f"{engine}:retried"] = retried
        return out
