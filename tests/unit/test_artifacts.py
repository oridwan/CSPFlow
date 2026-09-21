"""Composition-keyed databases, and whether they can really rebuild a campaign.

`campaign.db` is a cache. The claim these files make is that the expensive part
-- the relaxed cells and the numbers measured for them -- survives it. A claim
like that is worth nothing untested, so the last test here deletes the database
and rebuilds from the artifacts alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import bulk

from cspflow import artifacts


@pytest.fixture
def atoms():
    return bulk("Fe", "bcc", a=2.87, cubic=True)


class TestWhereThingsGo:
    def test_the_key_is_the_composition_not_the_batch(self, tmp_path):
        """A batch is an accident of scheduling; a composition is what was
        asked for and is stable across runs."""
        path = artifacts.db_path(tmp_path, "Ce2Fe17", "relaxed")
        assert path == tmp_path / "structures" / "Ce2Fe17" / "relaxed.db"

    def test_a_formula_that_would_break_a_path_is_made_safe(self, tmp_path):
        assert artifacts.slug("Ce3Fe(Ge6Pd)2") == "Ce3Fe_Ge6Pd_2"
        assert artifacts.slug("") == "unknown"

    def test_generated_and_relaxed_are_separate_files(self, tmp_path):
        """The cell as generated and the cell after relaxation are different
        answers to different questions; one must not overwrite the other."""
        a = artifacts.db_path(tmp_path, "CeFe5", "generated")
        b = artifacts.db_path(tmp_path, "CeFe5", "relaxed")
        assert a != b and a.parent == b.parent


class TestWriting:
    def test_a_row_carries_the_cell_and_its_numbers(self, tmp_path, atoms):
        path = artifacts.db_path(tmp_path, "Fe", "relaxed")
        assert artifacts.record(path, atoms, structure_id=7, e_total=-16.9,
                                converged=True, n_steps=23, engine="mattersim")
        [row] = list(artifacts.read(path))
        assert row.key_value_pairs["structure_id"] == 7
        assert row.key_value_pairs["e_total"] == pytest.approx(-16.9)
        assert row.key_value_pairs["converged"] is True
        assert len(row.toatoms()) == len(atoms)

    def test_a_none_does_not_take_the_whole_row_down(self, tmp_path, atoms):
        """ASE raises on a None value rather than storing a null. One
        unmeasured quantity must not cost the geometry it was attached to."""
        path = artifacts.db_path(tmp_path, "Fe", "relaxed")
        assert artifacts.record(path, atoms, structure_id=1, e_total=None,
                                fmax_final=None, converged=False)
        [row] = list(artifacts.read(path))
        assert "e_total" not in row.key_value_pairs
        assert row.key_value_pairs["structure_id"] == 1

    def test_screening_a_structure_twice_leaves_one_current_row(self, tmp_path, atoms):
        """A retry must not leave two rows disagreeing with no way to tell
        which is current."""
        path = artifacts.db_path(tmp_path, "Fe", "relaxed")
        artifacts.record(path, atoms, structure_id=3, e_total=-1.0, n_steps=5)
        artifacts.record(path, atoms, structure_id=3, e_total=-2.0, n_steps=99)
        rows = list(artifacts.read(path))
        assert len(rows) == 1
        assert rows[0].key_value_pairs["e_total"] == pytest.approx(-2.0)

    def test_two_structures_of_one_composition_share_the_file(self, tmp_path, atoms):
        path = artifacts.db_path(tmp_path, "Fe", "relaxed")
        artifacts.record(path, atoms, structure_id=1)
        artifacts.record(path, atoms, structure_id=2)
        assert len(list(artifacts.read(path))) == 2

    def test_an_unwritable_path_is_reported_not_raised(self, tmp_path, atoms):
        """The safety net must never be the thing that stops a campaign."""
        blocked = tmp_path / "file"
        blocked.write_text("not a directory")
        assert artifacts.record(blocked / "x" / "relaxed.db", atoms,
                                structure_id=1) is False

    def test_reading_a_file_that_is_not_there_yields_nothing(self, tmp_path):
        assert list(artifacts.read(tmp_path / "missing.db")) == []


def test_the_campaign_can_be_rebuilt_after_its_database_is_deleted(tmp_path, atoms):
    """THE PROPERTY ALL OF THIS EXISTS FOR.

    D129 lost 37 finished calculations to a database that forgot them while the
    results sat on disk. Write the artifacts, delete the database, rebuild, and
    the structures and their measured values must come back.
    """
    from cspflow.db.store import Origin, Store, StructureState

    workdir = tmp_path / "work"
    db = workdir / "campaign.db"

    with Store.create(db, campaign="t") as store:
        for n, formula in ((1, "Fe"), (2, "Fe"), (3, "CeFe5")):
            store.add_structure(atoms, origin=Origin.generated,
                                state=StructureState.screened)
            artifacts.record(
                artifacts.db_path(workdir, formula, "relaxed"), atoms,
                structure_id=n, reduced_formula=formula, campaign="t",
                e_total=-16.9 - n, converged=True, n_steps=10 + n,
                engine="mattersim")

    db.unlink()                      # the database is gone; the work is not
    assert not db.exists()

    recovered = {}
    for path in artifacts.compositions(workdir, "relaxed"):
        for row in artifacts.read(path):
            recovered[row.key_value_pairs["structure_id"]] = row

    assert sorted(recovered) == [1, 2, 3], "a structure was lost with the database"
    assert recovered[3].key_value_pairs["reduced_formula"] == "CeFe5"
    assert recovered[2].key_value_pairs["e_total"] == pytest.approx(-18.9)
    assert recovered[1].toatoms().get_chemical_formula() == atoms.get_chemical_formula()
    # and they are filed where a person would look for them
    assert (workdir / "structures" / "CeFe5" / "relaxed.db").is_file()


# ---------------------------------------------------------------------------
# D145 -- float32 results. MatterSim returns float32 forces; ASE writes a result
# array's raw bytes and reads them back as float64. Even atom counts read as
# garbage, odd ones raise -- and `_read_relaxed` swallowed the raise, so 4,520
# of 14,752 RE-magnets-CHGNet structures kept their UNRELAXED seed geometry.
# ---------------------------------------------------------------------------


def _float32_forces(n_atoms: int):
    """An odd- or even-sized cell carrying float32 forces, as MatterSim returns."""
    import numpy as np
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms = Atoms("Fe" * n_atoms, positions=[[0.3 * i, 0.1 * i, 0.2 * i] for i in range(n_atoms)],
                  cell=[6, 6, 6], pbc=True)
    forces = (np.arange(3 * n_atoms, dtype=np.float32).reshape(-1, 3) / 10).astype(np.float32)
    atoms.calc = SinglePointCalculator(atoms, energy=-8.25, forces=forces)
    # SinglePointCalculator casts to float64 on the way in, which would hide the
    # bug. A live MatterSimCalculator does not: its results stay float32.
    atoms.calc.results["forces"] = forces
    assert atoms.calc.results["forces"].dtype.name == "float32"
    return atoms, forces


@pytest.mark.parametrize("n_atoms", [3, 4])
def test_float32_results_are_stored_as_float64(tmp_path, n_atoms):
    from ase.db import connect

    atoms, forces = _float32_forces(n_atoms)
    path = tmp_path / "relaxed.db"
    assert artifacts.record(path, atoms, structure_id=7)
    row = next(connect(str(path)).select(structure_id=7))      # full row, forces included
    assert row.forces.dtype.name == "float64"
    assert row.forces == pytest.approx(forces.astype("float64"))
    assert atoms.calc.results["forces"].dtype.name == "float32", \
        "the caller's own calculator must not be changed"


def test_a_row_written_before_the_fix_still_yields_its_geometry(tmp_path):
    """The 4,520: an odd atom count and float32 forces already on disk."""
    from ase.db import connect

    from cspflow.stages.screen_stage import _read_relaxed

    atoms, _ = _float32_forces(5)
    path = tmp_path / "relaxed-task0.db"
    with connect(str(path), use_lock_file=False) as db:        # the pre-D145 write
        db.write(atoms, structure_id=11)
    with pytest.raises(ValueError):
        next(connect(str(path)).select(structure_id=11))       # what ASE does with it

    cell = _read_relaxed(str(path), 11)
    assert cell is not None, "the geometry was silently dropped"
    assert cell.positions == pytest.approx(atoms.positions)
    assert list(cell.numbers) == list(atoms.numbers)
    assert [row.key_value_pairs["structure_id"] for row in artifacts.read(path)] == [11]


def test_a_task_database_is_opened_once_not_per_structure(tmp_path, monkeypatch):
    """Reconcile asks for 500 structures one at a time; that was 500 opens on NFS."""
    import ase.db.sqlite as ase_sqlite

    from cspflow.stages import screen_stage

    path = tmp_path / "relaxed-task0.db"
    for sid in range(1, 21):
        atoms, _ = _float32_forces(2)
        artifacts.record(path, atoms, structure_id=sid)
    screen_stage._TASK_CELLS.clear()

    opened = {"n": 0}
    original = ase_sqlite.SQLite3Database._connect

    def counting(self):
        opened["n"] += 1
        return original(self)

    monkeypatch.setattr(ase_sqlite.SQLite3Database, "_connect", counting)
    cells = [screen_stage._read_relaxed(str(path), sid) for sid in range(1, 21)]
    assert all(c is not None for c in cells)
    assert opened["n"] == 1
