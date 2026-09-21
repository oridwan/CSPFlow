"""MLIP engines, the screen stage, and the array worker.

The validation gates and the plumbing are tested with a fake engine so they run
anywhere. The tests that need real MatterSim are marked and skip cleanly when it
is not importable, so the suite passes in the base environment too.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from ase.build import bulk

from cspflow.config.loader import load_campaign
from cspflow.db.store import Origin, Store, StructureState
from cspflow.mlip.base import BatchStats, RelaxResult, validate_structure
from cspflow.mlip.mattersim_engine import MatterSimEngine, apply_ase_compat_shim

CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/seeds"]}}
screen:
  mlip: mattersim
  mattersim: {{fmax: 0.05, max_steps: 20}}
  resources: {{role: cpu, ntasks: 1, time: "01:00:00"}}
"""


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "seeds").mkdir()
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path))
    return load_campaign(path)


@pytest.fixture
def seeded(tmp_path):
    """A store with four unrelaxed structures in state `new`."""
    store = Store.create(tmp_path / "c.db", campaign="t")
    for symbol, structure in [("Fe", bulk("Fe", "bcc", a=3.10, cubic=True)),
                              ("Co", bulk("Co", "fcc", a=3.70, cubic=True)),
                              ("Ni", bulk("Ni", "fcc", a=3.60, cubic=True)),
                              ("Cu", bulk("Cu", "fcc", a=3.70, cubic=True))]:
        store.add_structure(structure, origin=Origin.seed, state=StructureState.new,
                            reduced_formula=f"{symbol}1")
    yield store
    store.close()


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


class TestValidateStructure:
    def test_a_sensible_cell_passes(self):
        assert validate_structure(bulk("Fe", "bcc", a=2.87, cubic=True)) == ""

    def test_a_collapsed_cell_is_rejected(self):
        atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
        atoms.set_cell([0.5, 0.5, 0.5], scale_atoms=True)
        assert "lattice parameter" in validate_structure(atoms)

    def test_an_enormous_cell_is_rejected(self):
        atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
        atoms.set_cell([80.0, 80.0, 80.0], scale_atoms=True)
        assert "lattice parameter" in validate_structure(atoms)

    def test_the_length_cap_can_be_raised_for_a_known_good_source(self):
        """The 50 A default catches a generated cell with a 200 A axis. Applied
        to a Materials Project phase it also rejects real long-period stacking
        polytypes -- Cu27Se20 (mp-684606) is genuinely c = 65 A -- so the store
        raises it. See `MatterSimEngine(max_lattice=...)`.
        """
        atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
        atoms.set_cell([2.87, 2.87, 65.0], scale_atoms=False)
        assert "lattice parameter" in validate_structure(atoms)
        assert validate_structure(atoms, max_lattice=120.0) == ""

    def test_overlapping_atoms_are_rejected(self):
        """An MLIP does not refuse this; it returns an energy for it."""
        atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
        atoms.positions[1] = atoms.positions[0] + [0.1, 0.0, 0.0]
        assert "A apart" in validate_structure(atoms)

    def test_a_degenerate_angle_is_rejected(self):
        from ase import Atoms

        atoms = Atoms("Fe2", positions=[(0, 0, 0), (1.5, 0, 0)],
                      cell=[[3, 0, 0], [2.99, 0.1, 0], [0, 0, 3]], pbc=True)
        assert "angle" in validate_structure(atoms)

    def test_an_empty_cell_is_rejected(self):
        from ase import Atoms

        assert "empty" in validate_structure(Atoms())

    def test_a_single_atom_needs_no_distance_check(self):
        from ase import Atoms

        assert validate_structure(Atoms("Fe", positions=[(0, 0, 0)],
                                        cell=[3, 3, 3], pbc=True)) == ""


# --------------------------------------------------------------------------
# RelaxResult
# --------------------------------------------------------------------------


class TestRelaxResult:
    def test_ok_and_converged_are_different_questions(self):
        """Stopped at the step limit: it ran, and the geometry is not a minimum."""
        r = RelaxResult(energy=-1.0, e_per_atom=-1.0, converged=False, n_steps=500)
        assert r.ok and not r.converged
        assert r.exit_reason == "step_limit"

    def test_an_error_is_not_ok(self):
        assert not RelaxResult(error="rejected: overlapping atoms").ok

    def test_volume_drift_is_fractional(self):
        r = RelaxResult(energy=-1.0, volume_before=100.0, volume_after=90.0)
        assert r.volume_drift == pytest.approx(-0.1)

    def test_volume_drift_is_none_without_a_before(self):
        assert RelaxResult(energy=-1.0).volume_drift is None

    def test_a_converged_result_has_no_exit_reason(self):
        assert RelaxResult(energy=-1.0, converged=True).exit_reason == ""


class TestBatchStats:
    def test_counts_the_three_outcomes_separately(self):
        stats = BatchStats()
        stats.note(RelaxResult(energy=-1.0, converged=True))
        stats.note(RelaxResult(energy=-1.0, converged=False))
        stats.note(RelaxResult(error="rejected: atoms too close"))
        stats.note(RelaxResult(error="failed: RuntimeError"))
        assert (stats.total, stats.converged, stats.step_limited,
                stats.rejected, stats.failed) == (4, 1, 1, 1, 1)

    def test_render_flags_the_step_limited_ones(self):
        stats = BatchStats()
        stats.note(RelaxResult(energy=-1.0, converged=False))
        assert "not a minimum" in stats.render()


# --------------------------------------------------------------------------
# The compatibility shim
# --------------------------------------------------------------------------


class TestAseCompatShim:
    def test_it_is_additive_and_idempotent(self):
        """Only missing names are filled in; nothing ASE still defines is touched."""
        import ase.constraints
        import ase.filters

        before = getattr(ase.constraints, "FixAtoms", None)
        apply_ase_compat_shim()
        assert apply_ase_compat_shim() == []           # nothing left to add
        assert getattr(ase.constraints, "FixAtoms", None) is before
        assert ase.constraints.Filter is ase.filters.Filter


# --------------------------------------------------------------------------
# The screen stage, with a fake worker
# --------------------------------------------------------------------------


class TestScreenStage:
    def _stage(self, cfg):
        from cspflow.stages import ScreenStage

        return ScreenStage(cfg, chunk=2)

    def test_pending_counts_new_structures(self, cfg, seeded):
        assert self._stage(cfg).pending(seeded) == 4

    def test_claim_marks_them_screening(self, cfg, seeded):
        """So a second cycle, or a second driver, does not take the same work."""
        stage = self._stage(cfg)
        items = stage.claim(seeded, budget=10)
        assert sum(len(i.structure_ids) for i in items) == 4
        assert stage.pending(seeded) == 0
        assert seeded.count_structures(state="screening") == 4

    def test_claim_chunks_the_work(self, cfg, seeded):
        items = self._stage(cfg).claim(seeded, budget=10)
        assert [len(i.structure_ids) for i in items] == [2, 2]

    def test_build_writes_a_manifest_not_a_command_line(self, cfg, seeded, tmp_path):
        """A 2,000-id argument list is unreadable and, at scale, too long."""
        stage = self._stage(cfg)
        items = stage.claim(seeded, budget=10)
        spec = stage.build(items, tmp_path / "screen")
        manifest = json.loads(next((tmp_path / "screen").rglob("inputs.json")).read_text())
        assert manifest["chunks"] == [i.structure_ids for i in items]
        assert "--manifest" in spec.command and spec.array_size == 2

    def test_build_uses_absolute_paths(self, cfg, seeded, tmp_path):
        """The job runs with cwd=workdir, where a relative path resolves to nothing."""
        stage = self._stage(cfg)
        spec = stage.build(stage.claim(seeded, budget=10), tmp_path / "screen")
        assert spec.workdir.is_absolute()
        assert Path(spec.command.split("--manifest ")[1].strip()).is_absolute()

    def test_reconcile_absorbs_a_results_file(self, cfg, seeded, tmp_path):
        stage = self._stage(cfg)
        items = stage.claim(seeded, budget=10)
        workdir = tmp_path / "screen"
        stage.build(items, workdir)

        item = items[0]
        # `build` stamped the batch tag onto every item; the results file goes
        # in that batch's directory (D136), which is where reconcile looks.
        results = workdir / "batches" / item.group_key / "results-task0.json"
        results.parent.mkdir(parents=True, exist_ok=True)
        results.write_text(json.dumps({
            "max_steps": 20,
            "results": [
                {"structure_id": item.structure_ids[0], "energy": -16.9,
                 "e_per_atom": -8.45, "converged": True, "n_steps": 23,
                 "volume_before": 29.8, "volume_after": 22.8, "volume_drift": -0.23,
                 "error": "", "engine": "mattersim"},
                {"structure_id": item.structure_ids[1], "energy": None,
                 "e_per_atom": None, "converged": False, "n_steps": 0,
                 "error": "rejected: atoms 0.100 A apart", "engine": "mattersim"},
            ],
        }))
        job = {"workdir": str(workdir), "id": 1}
        from cspflow.scheduler.base import JobState, JobStatus

        stage.reconcile(seeded, job, JobStatus(job_id="1", state=JobState.done), [item])

        good = seeded.get_structure(item.structure_ids[0])
        assert good.state == "screened" and good.mlip_e_per_atom == pytest.approx(-8.45)
        assert good.mlip_volume_drift == pytest.approx(-0.23)
        bad = seeded.get_structure(item.structure_ids[1])
        assert bad.state == "failed" and "atoms" in bad.fail_reason
        assert seeded.relaxation_outcomes() == {"mattersim:converged": 1}

    def test_a_missing_results_file_fails_the_structures_with_a_reason(
            self, cfg, seeded, tmp_path):
        """A worker that died left no file; that is not a reason to retry forever."""
        stage = self._stage(cfg)
        items = stage.claim(seeded, budget=10)
        workdir = tmp_path / "screen"
        stage.build(items, workdir)
        from cspflow.scheduler.base import JobState, JobStatus

        stage.reconcile(seeded, {"workdir": str(workdir), "id": 1},
                        JobStatus(job_id="1", state=JobState.failed, raw_state="FAILED"),
                        [items[0]])
        row = seeded.get_structure(items[0].structure_ids[0])
        assert row.state == "failed" and "no results" in row.fail_reason


# --------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------


class TestWorker:
    def test_a_missing_manifest_is_an_error_naming_it(self, tmp_path):
        from cspflow.worker import WorkerError, run_screen_task

        with pytest.raises(WorkerError, match="manifest not found"):
            run_screen_task(tmp_path / "nope.json")

    def test_a_task_id_outside_the_manifest_is_an_error(self, tmp_path):
        from cspflow.worker import WorkerError, run_screen_task

        manifest = tmp_path / "m.manifest.json"
        manifest.write_text(json.dumps({"db": str(tmp_path / "c.db"), "chunks": [[1]]}))
        with pytest.raises(WorkerError, match="no chunk"):
            run_screen_task(manifest, task_id=7)

    def test_results_are_written_atomically(self, tmp_path):
        """A task killed mid-write must not leave a truncated file that parses as
        nothing and reads as a corrupt result rather than an absent one."""
        from cspflow.worker import _atomic_write_json

        out = tmp_path / "r.json"
        _atomic_write_json(out, {"results": []})
        assert out.is_file() and not list(tmp_path.glob("*.partial"))


# --------------------------------------------------------------------------
# Real MatterSim
# --------------------------------------------------------------------------

# A module-level `importorskip` here would skip the whole file, including every
# test above that needs no MLIP at all -- and those are most of them.
has_mattersim = pytest.mark.skipif(
    importlib.util.find_spec("mattersim") is None,
    reason="MatterSim not installed in this environment",
)


@has_mattersim
@pytest.mark.slow
class TestMatterSimEngine:
    def test_relaxes_iron_to_the_right_lattice_constant(self):
        engine = MatterSimEngine(fmax=0.01, max_steps=200)
        result = engine.relax(bulk("Fe", "bcc", a=3.10, cubic=True))
        assert result.ok and result.converged
        assert result.atoms.cell.lengths()[0] == pytest.approx(2.837, abs=0.02)
        assert result.e_per_atom == pytest.approx(-8.478, abs=0.01)

    def test_supercells_agree_with_the_primitive_cell(self):
        """The internal consistency check worth having: E/atom must not depend on N."""
        engine = MatterSimEngine(fmax=0.01, max_steps=200)
        small = engine.relax(bulk("Fe", "bcc", a=3.10, cubic=True))
        big = engine.relax(bulk("Fe", "bcc", a=3.10, cubic=True) * (2, 2, 2))
        assert small.e_per_atom == pytest.approx(big.e_per_atom, abs=1e-4)

    def test_the_step_limit_is_honoured_and_recorded(self):
        """The MLIP counterpart of a VASP run that hits NSW."""
        engine = MatterSimEngine(fmax=1e-9, max_steps=3)
        result = engine.relax(bulk("Fe", "bcc", a=3.10, cubic=True))
        assert result.ok and not result.converged
        assert result.n_steps == 3 and result.exit_reason == "step_limit"

    def test_a_nonsense_cell_never_reaches_the_model(self):
        atoms = bulk("Fe", "bcc", a=3.10, cubic=True)
        atoms.positions[1] = atoms.positions[0] + [0.1, 0.0, 0.0]
        result = MatterSimEngine().relax(atoms)
        assert not result.ok and result.error.startswith("rejected:")

    def test_a_single_point_leaves_the_geometry_alone(self):
        engine = MatterSimEngine()
        atoms = bulk("Fe", "bcc", a=2.8369, cubic=True)
        result = engine.single_point(atoms)
        assert result.ok and result.n_steps == 0
        assert result.volume_drift == pytest.approx(0.0)

    def test_batch_stats_summarise_a_run(self):
        engine = MatterSimEngine(fmax=0.05, max_steps=50)
        bad = bulk("Fe", "bcc", a=3.10, cubic=True)
        bad.positions[1] = bad.positions[0] + [0.05, 0.0, 0.0]
        _, stats = engine.relax_with_stats([bulk("Fe", "bcc", a=3.10, cubic=True), bad])
        assert stats.total == 2 and stats.rejected == 1 and stats.converged == 1


@pytest.fixture
def empty_store(tmp_path):
    store = Store.create(tmp_path / "seeds.db", campaign="t")
    yield store
    store.close()


def _screen(cfg):
    from cspflow.stages import ScreenStage

    return ScreenStage(cfg)


class TestSeedsThatMustNotMove:
    """`structure_list.relax: false` -- "MLIP-relax the seed before DFT" -- was
    written onto the structure as `needs_relax` and read by nothing, so a seed
    the user supplied deliberately was relaxed anyway. Mode 3's control-group
    use depends on this: a prototype compared against its own relaxed form is
    not a control if both were relaxed.
    """

    def test_the_claim_marks_which_structures_must_not_move(self, cfg, empty_store):
        moving = empty_store.add_structure(bulk("Fe", "bcc", a=2.87, cubic=True),
                                     origin=Origin.seed, needs_relax=True)
        fixed = empty_store.add_structure(bulk("Fe", "bcc", a=2.90, cubic=True),
                                    origin=Origin.seed, needs_relax=False)
        items = _screen(cfg).claim(empty_store, budget=1)
        flagged = {sid for item in items for sid in item.payload["single_point"]}
        assert flagged == {fixed}
        assert moving not in flagged

    def test_the_manifest_carries_it(self, cfg, empty_store, tmp_path):
        fixed = empty_store.add_structure(bulk("Fe", "bcc", a=2.90, cubic=True),
                                    origin=Origin.seed, needs_relax=False)
        stage = _screen(cfg)
        items = stage.claim(empty_store, budget=1)
        stage.build(items, tmp_path)
        manifest = json.loads(next(tmp_path.rglob("inputs.json")).read_text())
        assert manifest["single_point"] == [fixed]

    def test_a_normal_structure_is_not_flagged(self, cfg, empty_store):
        empty_store.add_structure(bulk("Fe", "bcc", a=2.87, cubic=True),
                            origin=Origin.generated)
        items = _screen(cfg).claim(empty_store, budget=1)
        assert items[0].payload["single_point"] == []

    def test_the_result_records_which_it_was(self, cfg, empty_store):
        """`mlip_relaxed=False` so nothing downstream reads a single-point
        energy as a relaxed one."""
        sid = empty_store.add_structure(bulk("Fe", "bcc", a=2.90, cubic=True),
                                  origin=Origin.seed, needs_relax=False)
        stage = _screen(cfg)
        stage._absorb(empty_store, {"max_steps": 300, "results": [
            {"structure_id": sid, "e_per_atom": -8.0, "converged": True,
             "n_steps": 0, "relaxed": False, "energy": -16.0}]})
        assert empty_store.get_structure(sid).key_value_pairs["mlip_relaxed"] is False


# --------------------------------------------------------------------------
# Admission: a bad SETTING is not a bad structure
# --------------------------------------------------------------------------


class TestAdmission:
    """`_admit` Niggli-reduces before validating.

    Four Materials Project phases in the reference store were silently dropped
    from the MatterSim half because their cells arrive in a badly skewed
    setting: mp-673643 (Ce17O32) has axes of 66/41/10 A at angles 18/19/5
    degrees.  Reduced, it is 8.7/8.7/10.3 A at 105/95/107 -- same atoms, same
    volume, same energy -- and it passes.  Those four blocked the MatterSim hull
    in five chemical systems until 2026-09-12.
    """

    @staticmethod
    def _engine(**kw):
        from cspflow.mlip.mattersim_engine import MatterSimEngine

        return MatterSimEngine(**kw)

    def test_a_skewed_setting_is_reduced_and_admitted(self):
        import numpy as np

        atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
        # A basis change: same lattice, expressed with a near-parallel vector.
        cell = np.array(atoms.cell)
        cell[0] = cell[0] + 12 * cell[2]
        atoms.set_cell(cell, scale_atoms=False)
        assert validate_structure(atoms) != ""          # rejected as it stands

        used, reason = self._engine()._admit(atoms)
        assert reason == ""
        assert used.get_volume() == pytest.approx(atoms.get_volume(), rel=1e-9)
        assert max(used.cell.lengths()) < max(atoms.cell.lengths())

    def test_reduction_preserves_the_atom_count_and_volume(self):
        atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
        used, reason = self._engine()._admit(atoms)
        assert reason == ""
        assert len(used) == len(atoms)
        assert used.get_volume() == pytest.approx(atoms.get_volume(), rel=1e-9)

    def test_a_genuinely_bad_cell_is_still_rejected(self):
        """Reduction must not become a way in for nonsense: overlapping atoms
        survive any change of basis, so the rejection survives too."""
        atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
        atoms.positions[1] = atoms.positions[0] + [0.1, 0.0, 0.0]
        used, reason = self._engine()._admit(atoms)
        assert "A apart" in reason

    def test_the_original_cells_reason_is_reported_not_the_reduced_ones(self):
        """The caller handed us the original; that is the cell to debug."""
        atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
        atoms.set_cell([0.4, 0.4, 0.4], scale_atoms=True)
        _, reason = self._engine()._admit(atoms)
        assert "lattice parameter" in reason
