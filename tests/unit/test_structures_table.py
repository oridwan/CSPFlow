"""`structures.csv` -- every structure, wherever it got to.

The table `candidates.csv` is not: candidates lists what came out, this lists
what happened to everything, including the rows that stopped. Two properties
are worth pinning, because both were wrong in the first version:

*   the gate and the reason must come from the SAME record. Taking the last
    failed event for one and the stored reason for the other reads plausibly
    and misattributes the stop;
*   a structure still moving must report neither, rather than a made-up
    "in progress" that sorts alongside real verdicts.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from ase.build import bulk

from cspflow.db.store import Origin, Store, StructureState
from cspflow.report.structures import COLUMNS, structure_rows, write


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "campaign.db", campaign="t") as s:
        yield s


def add(store, state, **kv):
    return store.add_structure(bulk("Fe", "bcc", a=2.87, cubic=True),
                               origin=Origin.generated, state=state, **kv)


def test_every_structure_appears_whatever_state_it_is_in(store):
    """The point of the file: a row that stopped is still a row."""
    add(store, StructureState.new)
    add(store, StructureState.screened)
    add(store, StructureState.filtered_out, filter_reason="too far above the hull")
    add(store, StructureState.dft_done)
    add(store, StructureState.failed, dft_fail_reason="relax: CANCELLED")

    rows = structure_rows(store)
    assert len(rows) == 5, "a structure vanished from the table"
    assert {r["state"] for r in rows} == {
        "new", "screened", "filtered_out", "dft_done", "failed"}


def test_a_structure_still_moving_reports_no_verdict(store):
    """An empty cell is honest. A placeholder would sort with real verdicts."""
    add(store, StructureState.screened)
    [row] = structure_rows(store)
    assert row["stopped_at"] == ""
    assert row["why"] == ""


def test_a_finished_structure_is_not_described_as_stopped(store):
    add(store, StructureState.dft_done)
    [row] = structure_rows(store)
    assert row["stopped_at"] == ""
    assert row["selected"] is True


def test_the_gate_and_the_reason_come_from_the_same_record(store):
    """The defect this pins, found on CePdGe structure 122.

    It had an unconverged relax event AND a later operator exclusion. Reading
    the gate from the events and the reason from `filter_reason` produced

        stopped_at  dft:relax:converged
        why         zero computed moment ... excluded by operator

    which names a gate the structure did not stop at, and sends anyone
    investigating to the wrong place.
    """
    sid = add(store, StructureState.selected)
    store.add_filter_event(structure_id=sid, gate="dft:relax:converged",
                           passed=False, detail="ionic step limit")
    store.set_structure_state(sid, StructureState.filtered_out,
                              filter_reason="excluded by operator")

    [row] = structure_rows(store)
    assert row["why"] == "excluded by operator"
    assert not row["stopped_at"].startswith("dft:"), (
        f"gate {row['stopped_at']!r} contradicts reason {row['why']!r}")


def test_a_real_gate_refusal_keeps_its_own_gate_and_detail(store):
    sid = add(store, StructureState.filtered_out)
    store.add_filter_event(structure_id=sid, gate="filter:e_above_hull",
                           passed=False, value=0.4, threshold=0.1,
                           detail="0.400 > 0.100")
    [row] = structure_rows(store)
    assert row["stopped_at"] == "filter:e_above_hull"
    assert "0.400" in row["why"]


def test_a_stopped_row_never_has_one_without_the_other(store):
    """Half an answer is the failure mode this file exists to remove."""
    a = add(store, StructureState.filtered_out, filter_reason="operator")
    b = add(store, StructureState.failed, dft_fail_reason="relax: TIMEOUT")
    store.add_filter_event(structure_id=b, gate="dft:relax:converged",
                           passed=False, detail="wall clock")
    for row in structure_rows(store):
        if row["state"] in ("filtered_out", "failed"):
            assert row["stopped_at"] and row["why"], row


def test_volume_per_atom_is_derived_not_stored(store):
    """Derived on read so it cannot disagree with the two columns it comes
    from. A stale stored copy is how a table starts lying quietly.

    The invariant is asserted rather than a literal: `volume` comes from ASE's
    own column, so writing one in and expecting it back tests the fixture, not
    the code.
    """
    sid = add(store, StructureState.dft_done)
    # `volume` is one of ASE's own columns, so it appears in key_value_pairs
    # only once the row has been updated -- which is what the analyze stage
    # does. Passing a number in is pointless: ASE returns the real cell volume.
    store.update_structure(sid, spacegroup=229)
    [row] = structure_rows(store)
    assert row["volume"] and row["n_atoms"]
    assert row["volume_per_atom"] == pytest.approx(
        float(row["volume"]) / row["n_atoms"], abs=1e-4)


def test_the_csv_has_a_header_line_for_every_column(store):
    add(store, StructureState.dft_done)
    path = write(store, Path(store.path).parent / "structures.csv")
    text = path.read_text()
    for name, _ in COLUMNS:
        assert f"#   {name}" in text, f"{name} is not explained in the header"
    rows = list(csv.DictReader(l for l in text.splitlines() if not l.startswith("#")))
    assert len(rows) == 1
    assert set(rows[0]) == {n for n, _ in COLUMNS}
