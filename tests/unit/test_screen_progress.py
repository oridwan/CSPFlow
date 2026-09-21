"""The screen job says what it is doing, and keeps the geometry it produced.

Two defects, found 2026-09-12 from a CePdGe driver log that read

    cycle 1  in flight 1 ...
    cycle 11  in flight 1 ...

and nothing else for eleven cycles.

**The worker was silent.** It printed two lines for a 24-minute run -- the
MatterSim checkpoint banner and the output path -- so neither the job's own log
nor the driver's could say how far along it was or whether anything converged.

**The relaxed geometry was thrown away.** The worker recorded energy, fmax and
step count and discarded `result.atoms`, so `dft_stage` step 0 started from the
seed exactly as supplied and the relaxation bought nothing downstream. The
reference store had already fixed this; the campaign path had not. The same
class of bug at recipe step > 0 was caught earlier with volumes measured at
176.15 vs 179.03 A^3.
"""

import json

import pytest
from ase.build import bulk

from cspflow.db.store import Origin, Store, StructureState
from cspflow.worker import run_screen_task


@pytest.fixture
def campaign(tmp_path):
    store = Store.create(tmp_path / "c.db", campaign="t")
    store.add_composition(formula="Fe2", chemsys="Fe", z=1, n_atoms=2, n_target=0,
                          source_mode="structure_list", source_name="s",
                          state="new")
    ids = [store.add_structure(bulk("Fe", "bcc", a=2.87, cubic=True),
                               origin=Origin.seed, state=StructureState.new)
           for _ in range(3)]
    store.close()
    manifest = tmp_path / "screen-1-3.manifest.json"
    manifest.write_text(json.dumps({
        "key": "screen-1-3", "db": str(tmp_path / "c.db"),
        "chunks": [ids], "max_steps": 5, "fmax": 0.05,
    }))
    return manifest, ids


def test_the_relaxed_cell_is_written_to_disk(campaign, tmp_path, capsys):
    """ONE database per task, not one POSCAR per structure (D141).

    It was `relaxed-task0/<sid>.vasp`: geometry only, one file per structure,
    with the energy that belongs to it in a different file. A 500-cell chunk
    left 500 files saying nothing about what was measured. Now the cell and its
    numbers share a row, so this file alone can rebuild the task's record.
    """
    from ase.db import connect

    manifest, ids = campaign
    out = run_screen_task(manifest, task_id=0)
    payload = json.loads(out.read_text())

    # Named for the side of the job it is on, and derived from the manifest's
    # own directory so it follows the manifest into batches/<tag>/ (D136).
    relaxed = tmp_path / "relaxed-task0.db"
    assert relaxed.is_file(), "the relaxed cells went nowhere"
    assert payload["relaxed_db"] == str(relaxed)

    with connect(str(relaxed)) as db:
        rows = {r.key_value_pairs["structure_id"]: r for r in db.select()}
    assert sorted(rows) == sorted(ids), "a structure is missing from the database"
    for sid in ids:
        kv = rows[sid].key_value_pairs
        # the cell AND what was measured for it, in one row
        assert rows[sid].toatoms() is not None
        assert "e_total" in kv and "converged" in kv and "n_steps" in kv
        assert kv["reduced_formula"]

    for row in payload["results"]:
        assert row["geometry"] == str(relaxed)


def test_a_single_point_seed_keeps_its_geometry(campaign, tmp_path):
    """`single_point` means the campaign asked NOT to move this structure.
    Saving a 'relaxed' cell for it would be the thing that setting prevents."""
    manifest, ids = campaign
    spec = json.loads(manifest.read_text())
    spec["single_point"] = ids
    manifest.write_text(json.dumps(spec))

    payload = json.loads(run_screen_task(manifest, task_id=0).read_text())
    assert all(row["geometry"] == "" for row in payload["results"])
    assert all(row["relaxed"] is False for row in payload["results"])


def test_every_structure_reports_a_line(campaign, capsys):
    manifest, ids = campaign
    run_screen_task(manifest, task_id=0)
    out = capsys.readouterr().out

    for n in range(1, len(ids) + 1):
        assert f"[{n}/{len(ids)}]" in out
    assert "converged" in out or "STEP LIMIT" in out
    assert "done:" in out                      # the summary at the end


def test_progress_is_readable_by_the_driver_while_it_runs(campaign, tmp_path,
                                                          monkeypatch):
    """The driver cannot read another allocation's stdout, so progress has to
    land on shared disk."""
    manifest, ids = campaign
    seen = []
    import cspflow.worker as mod

    real = mod._write_progress

    def spy(path, *a, **k):
        real(path, *a, **k)
        if path.is_file():
            seen.append(json.loads(path.read_text()))

    monkeypatch.setattr(mod, "_write_progress", spy)
    run_screen_task(manifest, task_id=0)

    assert len(seen) == len(ids)
    assert [s["done"] for s in seen] == list(range(1, len(ids) + 1))
    assert all(s["total"] == len(ids) for s in seen)
    # Removed on success: a leftover file would report a finished job as running.
    assert not (tmp_path / "screen-1-3.task0.progress.json").exists()


def test_the_store_can_replace_a_geometry_keeping_the_id(tmp_path):
    """Every claim, filter event and hull row refers to the structure id, so
    carrying the relaxed cell in must not mint a new one."""
    with Store.create(tmp_path / "c.db", campaign="t") as store:
        store.add_composition(formula="Fe2", chemsys="Fe", z=1, n_atoms=2,
                              n_target=0, source_mode="structure_list",
                              source_name="s", state="new")
        sid = store.add_structure(bulk("Fe", "bcc", a=2.87, cubic=True),
                                  origin=Origin.seed, state=StructureState.new)
        before = store.get_structure(sid).toatoms().get_volume()

        bigger = bulk("Fe", "bcc", a=3.10, cubic=True)
        store.replace_geometry(sid, bigger)

        after = store.get_structure(sid)
        assert int(after.id) == sid
        assert after.toatoms().get_volume() != pytest.approx(before)
        assert after.toatoms().get_volume() == pytest.approx(bigger.get_volume())
