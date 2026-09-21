"""`csp status` has to answer from any node, at any time, including mid-run.

The database cannot do that on its own.  It is SQLite on a network filesystem,
and a running driver keeps its recent writes in a `-wal` whose index lives in
`-shm` -- shared memory, coherent within one host and nowhere else.  Read from a
second node it gives one of two wrong answers:

  * an exception, when even the schema is still in the WAL -- this is what
    `csp status` on str-c6 did against the CeFeB campaign held by its driver on
    str-c221, main file 35 minutes stale and the `-wal` 4.2 MB;
  * worse, a clean table of numbers that stopped being true at the last
    checkpoint, when the schema happens to have been written before the driver
    started.  Nothing about that output looks wrong.

The fix is not to make the database readable -- it cannot be, and an existing
campaign is stuck in WAL because the mode is persisted in the file and cannot be
changed while the driver holds it open.  The fix is for `csp status` to prefer
`status.json`, which the driver rewrites every cycle by atomic rename from the
one node that is guaranteed to be able to read the database.
"""

import json
import os
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cspflow.cli import app
from cspflow.db.store import wal_lag

runner = CliRunner()


def output_of(result) -> str:
    """stdout + stderr, whichever the installed click keeps them in."""
    parts = [result.stdout or ""]
    try:
        parts.append(result.stderr or "")
    except (ValueError, AttributeError):
        pass
    return "".join(parts)


CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/a.vasp"]}}
"""


SNAPSHOT = {
    "updated": "2026-09-12T20:00:00+00:00",
    "campaign": "t",
    "cycle": 173,
    "slurm_job": "26930224",
    "in_flight": 20,
    "core_hours_spent": 5111.3,
    "core_hours_projected": 5457.0,
    "structures": {"dft_done": 3, "dft_queued": 60, "selected": 30},
    "structures_total": 93,
    "stages": [{"stage": "dft", "pending": 30, "claimed": 1,
                "submitted": 1, "note": "bound by max_in_flight=20"}],
    "summary": {
        "campaign": "t",
        "structures": 93,
        "compositions": 71,
        "chemsystems": 42,
        "reference_entries": 4716,
        "jobs": {"done": 7, "queued": 19},
        "structures_by_state": {"dft_done": 3, "dft_queued": 60, "selected": 30},
        "relaxations": {"vasp:relax:converged": 6, "vasp:relax:not converged": 4},
        "core_hours": 65.0,
    },
    "generation": {"compositions": 71, "requested": 100, "produced": 80, "short": 3},
}


@pytest.fixture()
def campaign(tmp_path):
    """A campaign whose database is present but whose WAL is newer.

    That mtime relationship is the whole of the cross-node symptom: it is what a
    second NFS client sees while a driver on another host is writing.
    """
    (tmp_path / "campaign.yaml").write_text(CAMPAIGN.format(workdir=tmp_path))
    db = tmp_path / "campaign.db"
    db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)
    wal = tmp_path / "campaign.db-wal"
    wal.write_bytes(b"\x00" * 4096)
    old = time.time() - 3600
    os.utime(db, (old, old))            # main file an hour behind
    return tmp_path


def write_snapshot(dirpath: Path, payload: dict, *, age_s: float = 0.0) -> Path:
    p = dirpath / "status.json"
    p.write_text(json.dumps(payload) + "\n")
    if age_s:
        os.utime(p, (time.time() - age_s, time.time() - age_s))
    return p


# --- wal_lag: the staleness detector --------------------------------------


def test_no_wal_is_not_stale(tmp_path):
    db = tmp_path / "c.db"
    db.write_bytes(b"x")
    assert wal_lag(db) is None


def test_an_empty_wal_is_not_stale(tmp_path):
    """A zero-length WAL is a checkpointed database, not a stale one."""
    db = tmp_path / "c.db"
    db.write_bytes(b"x")
    (tmp_path / "c.db-wal").write_bytes(b"")
    assert wal_lag(db) is None


def test_a_newer_main_file_is_not_stale(tmp_path):
    db = tmp_path / "c.db"
    wal = tmp_path / "c.db-wal"
    wal.write_bytes(b"\x00" * 64)
    db.write_bytes(b"x")
    old = time.time() - 600
    os.utime(wal, (old, old))
    assert wal_lag(db) is None


def test_a_newer_wal_reports_the_lag(tmp_path):
    db = tmp_path / "c.db"
    db.write_bytes(b"x")
    (tmp_path / "c.db-wal").write_bytes(b"\x00" * 64)
    os.utime(db, (time.time() - 1800, time.time() - 1800))
    lag = wal_lag(db)
    assert lag is not None and 1700 < lag < 1900


def test_a_missing_database_is_not_reported_as_stale(tmp_path):
    assert wal_lag(tmp_path / "absent.db") is None


# --- the command ----------------------------------------------------------


def test_a_stale_database_falls_back_to_the_snapshot(campaign):
    """The regression.  Before the fix this exited with StoreError and no data."""
    write_snapshot(campaign, SNAPSHOT)
    r = runner.invoke(app, ["status", "--campaign", str(campaign / "campaign.yaml")])
    out = output_of(r)
    assert r.exit_code == 0, out
    assert "driver snapshot status.json" in out
    assert "cannot be read from here" in out
    assert "structures   93" in out


def test_the_snapshot_renders_the_same_blocks_as_the_database(campaign):
    """A fallback that answered a narrower question would not be a fallback."""
    write_snapshot(campaign, SNAPSHOT)
    out = output_of(runner.invoke(
        app, ["status", "--campaign", str(campaign / "campaign.yaml")]))
    assert "compositions 71  across 42 chemical systems" in out
    assert "generated    80 of 100 requested (80.0%)" in out
    assert "3 composition(s) short" in out
    assert "reference    4716 MP entries" in out
    assert "relaxations" in out
    # D027: a clean exit at the ionic step limit is `done` and NOT relaxed.
    assert "not usable as a relaxed geometry" in out


def test_an_old_snapshot_says_so_rather_than_looking_current(campaign):
    """A snapshot is only current while the driver writing it is alive."""
    write_snapshot(campaign, SNAPSHOT, age_s=7200)
    out = output_of(runner.invoke(
        app, ["status", "--campaign", str(campaign / "campaign.yaml")]))
    assert "WARNING: no cycle in" in out
    assert "squeue" in out


def test_a_snapshot_without_a_summary_still_reports_counts(campaign):
    """Written by a driver older than the summary field.  Print what is there."""
    old = {k: v for k, v in SNAPSHOT.items() if k not in ("summary", "generation")}
    write_snapshot(campaign, old)
    out = output_of(runner.invoke(
        app, ["status", "--campaign", str(campaign / "campaign.yaml")]))
    assert "structures   93" in out
    assert "dft_queued       60" in out


def test_source_db_is_not_silently_redirected(campaign):
    """An explicit --source db must fail loudly rather than answer from a file."""
    write_snapshot(campaign, SNAPSHOT)
    r = runner.invoke(app, ["status", "--campaign", str(campaign / "campaign.yaml"),
                            "--source", "db"])
    assert r.exit_code != 0
    assert "driver snapshot" not in output_of(r)


def test_why_refuses_a_stale_database_instead_of_answering_from_it(campaign):
    """`--why` is a per-structure history; a snapshot carries counts, so there
    is nothing to fall back to.  Saying that beats printing a partial history."""
    write_snapshot(campaign, SNAPSHOT)
    r = runner.invoke(app, ["status", "--campaign", str(campaign / "campaign.yaml"),
                            "--why", "87"])
    out = output_of(r)
    assert r.exit_code != 0
    assert "--why needs the database" in out
    assert "squeue" in out


def test_a_stale_database_with_no_snapshot_names_the_remedy(campaign):
    r = runner.invoke(app, ["status", "--campaign", str(campaign / "campaign.yaml")])
    out = output_of(r)
    assert r.exit_code != 0
    assert "no driver snapshot" in out or "there is no driver snapshot" in out
    assert "--source db" in out


def test_the_reason_given_is_the_cause_not_the_mtime_gap(campaign):
    """The gap is not a measure of severity.  SQLite auto-checkpoints, so it
    collapses to near zero just after one and grows until the next, while the
    amount of campaign a wrong-host reader cannot see stays whatever is in the
    WAL.  Reporting "2s behind" invites the wrong inference -- that the database
    is nearly current and could be used.  The situation is binary."""
    write_snapshot(campaign, SNAPSHOT)
    out = output_of(runner.invoke(
        app, ["status", "--campaign", str(campaign / "campaign.yaml")]))
    assert "WAL mode and open on another node" in out
    assert "cannot be read from here" in out
    assert "4,096-byte write-ahead log" in out   # what is actually unreadable
    assert "behind its write-ahead log" not in out


def test_an_unknown_source_is_rejected(campaign):
    r = runner.invoke(app, ["status", "--campaign", str(campaign / "campaign.yaml"),
                            "--source", "bogus"])
    assert r.exit_code != 0
    assert "auto, db or snapshot" in output_of(r)


# --- a newer WAL is only a problem on the wrong host ------------------------
#
# On the node running the driver, a `-wal` newer than the main database is just
# what WAL mode looks like while work is happening, and reads there are correct:
# SQLite merges the WAL for any connection that can see the `-shm`.  Treating
# that as staleness would divert `csp status` to the snapshot on the one node
# where the database is authoritative, and would make `--why` refuse on the only
# node where it can answer.  The driver records its hostname for exactly this.


def test_the_holding_node_still_uses_its_database(campaign, monkeypatch):
    import socket
    payload = dict(SNAPSHOT, host="str-c221")
    write_snapshot(campaign, payload)
    monkeypatch.setattr(socket, "gethostname", lambda: "str-c221")
    from cspflow.cli import _stale_lag
    assert _stale_lag(campaign / "campaign.db", campaign / "status.json") is None


def test_another_node_does_not(campaign, monkeypatch):
    import socket
    write_snapshot(campaign, dict(SNAPSHOT, host="str-c221"))
    monkeypatch.setattr(socket, "gethostname", lambda: "str-c6")
    from cspflow.cli import _stale_lag
    lag = _stale_lag(campaign / "campaign.db", campaign / "status.json")
    assert lag is not None and lag > 1000


def test_no_snapshot_means_distrust_the_read(campaign):
    """Nothing to discriminate with.  A confident wrong number is the failure
    this path exists to avoid, and --source db is one flag away."""
    from cspflow.cli import _stale_lag
    assert _stale_lag(campaign / "campaign.db", campaign / "status.json") is not None


def test_an_old_driver_snapshot_without_a_host_still_distrusts(campaign):
    write_snapshot(campaign, {k: v for k, v in SNAPSHOT.items() if k != "host"})
    from cspflow.cli import _stale_lag
    assert _stale_lag(campaign / "campaign.db", campaign / "status.json") is not None


def test_why_is_allowed_on_the_holding_node(campaign, monkeypatch):
    """The mirror of `test_why_refuses_a_stale_database`: where the database IS
    authoritative, --why must not be blocked."""
    import socket
    write_snapshot(campaign, dict(SNAPSHOT, host="str-c221"))
    monkeypatch.setattr(socket, "gethostname", lambda: "str-c221")
    r = runner.invoke(app, ["status", "--campaign", str(campaign / "campaign.yaml"),
                            "--why", "87"])
    # It gets past the staleness guard and fails on the fake database instead.
    assert "--why needs the database" not in output_of(r)
