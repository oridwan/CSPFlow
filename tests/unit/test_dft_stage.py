"""Stage 6 -- the DFT stage and its retry ladder.

Run against synthetic VASP output directories, so the whole state machine --
walking a two-step recipe, retrying with the right remedy, giving up for the
right reason -- is exercised with no cluster and no VASP.
"""

import json
from pathlib import Path

import pytest
import yaml
from ase.build import bulk

from cspflow.config.loader import load_campaign
from cspflow.db.store import Origin, Store, StructureState
from cspflow.scheduler.base import JobState, JobStatus
from cspflow.dft.vasp.inputs import InputError
from cspflow.stages.dft_stage import (ATTEMPT_KEY, DIR_KEY, STEP_KEY,
                                      DftStage)

MACHINES = Path(__file__).resolve().parents[2] / "src" / "cspflow" / "machines"
POTCARS = Path("/projects/mmi/cspflow-shared/potcars/pmg")
has_potcars = pytest.mark.skipif(not POTCARS.is_dir(), reason="POTCAR tree not present")

CAMPAIGN = """\
name: t
machine: orion
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/seeds"]}}
dft:
  recipe: magnets
  layout: {layout}
  magnetism: {{mode: ferrimagnetic_retm}}
"""


def _campaign(tmp_path, layout):
    (tmp_path / "seeds").mkdir(exist_ok=True)
    path = tmp_path / f"campaign-{layout}.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path, layout=layout))
    return load_campaign(path)


@pytest.fixture
def cfg(tmp_path):
    """The default a new campaign gets: one directory per structure, one job
    per structure, every step of it inside that job (D131, D135)."""
    return _campaign(tmp_path, "runs")


@pytest.fixture
def flat_cfg(tmp_path):
    """The older per-STEP layout, pinned explicitly.

    The tests that use it are about the per-step walk -- one job reconciled,
    one step advanced -- which is the path `layout: stages` takes and which
    `_reconcile_combined` replaces. It is still reachable: `layout.resolve()`
    gives it to any campaign whose disk already holds `dft-<id>-<step>/`
    directories, whatever the config says, so it still needs testing.
    """
    return _campaign(tmp_path, "stages")


def step_dir(stage, item, workdir, step_name=None):
    """Where `stage` will look for one step's VASP output.

    Asked of the stage rather than spelled out, because the whole point of the
    layout change is that the answer is no longer `workdir / item.key`.
    """
    sid = item.structure_ids[0]
    if stage.combined:
        return stage.layout.run_root(
            sid, item.payload.get("formula", ""),
            item.payload.get("source_path", "")) / (
                step_name or item.payload.get("step_name", "relax"))
    return Path(workdir) / item.key


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "campaign.db", campaign="t") as s:
        yield s


def add_selected(store, n=1):
    ids = []
    for _ in range(n):
        ids.append(store.add_structure(bulk("Fe", "bcc", a=2.87, cubic=True),
                                       origin=Origin.generated,
                                       state=StructureState.selected))
    return ids


OUTCAR_CONVERGED = """\
 reached required accuracy - stopping structural energy minimisation
 General timing and accounting informations for this job:
                  Total CPU time used (sec):       100.0
"""

OUTCAR_STEP_LIMIT = """\
 General timing and accounting informations for this job:
                  Total CPU time used (sec):       100.0
"""


def fake_job_output(directory: Path, *, converged: bool, energy=-16.9,
                    steps=None, nsw=99):
    # An unconverged run that reached NSW is the ionic-step-limit case; a
    # converged one stops earlier.
    steps = steps if steps is not None else (12 if converged else nsw)
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"   1 F= {energy:.6E} E0= {energy:.6E}  d E =0.0  mag=     4.2"]
    for i in range(2, steps + 1):
        lines.append(f"   {i} F= {energy:.6E} E0= {energy:.6E}  d E =0.0  mag=     4.2")
    (directory / "OSZICAR").write_text("\n".join(lines) + "\n")
    body = OUTCAR_CONVERGED if converged else OUTCAR_STEP_LIMIT
    (directory / "OUTCAR").write_text(f"   NIONS = 2\n" + body)
    # NSW is read from the INCAR, not the OUTCAR -- which is where the parser
    # looks and where a real job directory has it.
    (directory / "INCAR").write_text(f"NSW = {nsw}\nNCORE = 4\n")
    (directory / "VASP_DONE").write_text("done\n")
    return directory


class TestClaim:
    def test_pending_counts_selected_structures(self, cfg, store):
        add_selected(store, 3)
        assert DftStage(cfg).pending(store) == 3

    def test_claiming_marks_them_queued(self, cfg, store):
        add_selected(store, 2)
        stage = DftStage(cfg)
        items = stage.claim(store, budget=10)
        assert len(items) == 2
        assert store.count_structures(state="dft_queued") == 2
        assert stage.pending(store) == 0

    def test_a_claim_carries_its_recipe_step(self, cfg, store):
        add_selected(store, 1)
        item = DftStage(cfg).claim(store, budget=1)[0]
        assert item.payload["step"] == 0
        assert item.payload["step_name"] == "relax"

    def test_the_budget_caps_the_claim(self, cfg, store):
        add_selected(store, 5)
        assert len(DftStage(cfg).claim(store, budget=2)) == 2

    def test_a_finished_structure_is_not_reclaimed(self, cfg, store):
        sid = add_selected(store, 1)[0]
        store.set_structure_state(sid, StructureState.dft_done, **{STEP_KEY: 2})
        assert DftStage(cfg).pending(store) == 0


class TestRecipeWalk:
    def _reconcile(self, flat_cfg, store, sid, workdir, *, converged, step=0,
                   step_name="relax", attempt=0, status=None):
        from cspflow.stages.base import WorkItem

        item = WorkItem(key=f"dft-{sid}-{step_name}", structure_ids=[sid],
                        payload={"step": step, "step_name": step_name,
                                 "attempt": attempt})
        fake_job_output(Path(workdir) / item.key, converged=converged)
        DftStage(flat_cfg).reconcile(
            store, {"workdir": str(workdir), "id": 1},
            status or JobStatus(job_id="1", state=JobState.done, raw_state="COMPLETED"),
            [item],
        )
        return item

    def test_a_converged_relax_advances_to_the_next_step(self, flat_cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._reconcile(flat_cfg, store, sid, tmp_path, converged=True)
        row = store.get_structure(sid)
        assert row.state == "selected" and row.dft_step == 1

    def test_the_last_step_finishes_the_structure(self, flat_cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._reconcile(flat_cfg, store, sid, tmp_path, converged=True,
                        step=1, step_name="static")
        assert store.get_structure(sid).state == "dft_done"

    def test_the_energy_is_recorded(self, flat_cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._reconcile(flat_cfg, store, sid, tmp_path, converged=True)
        assert store.get_structure(sid).vasp_energy is not None
        assert store.relaxation_outcomes() == {"vasp:relax:converged": 1}

    def test_a_gate_event_is_recorded_either_way(self, flat_cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._reconcile(flat_cfg, store, sid, tmp_path, converged=False)
        gates = [e["gate"] for e in store.filter_events(sid)]
        assert "dft:relax:converged" in gates


class TestRetryLadder:
    def _fail(self, flat_cfg, store, sid, workdir, *, status, attempt=0):
        from cspflow.stages.base import WorkItem

        item = WorkItem(key=f"dft-{sid}-relax-{attempt}", structure_ids=[sid],
                        payload={"step": 0, "step_name": "relax", "attempt": attempt})
        fake_job_output(Path(workdir) / item.key, converged=False)
        DftStage(flat_cfg).reconcile(store, {"workdir": str(workdir), "id": 1},
                                status, [item])

    def test_an_ionic_step_limit_is_retried(self, flat_cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._fail(flat_cfg, store, sid, tmp_path,
                   status=JobStatus(job_id="1", state=JobState.done,
                                    raw_state="COMPLETED"))
        row = store.get_structure(sid)
        assert row.state == "selected"          # queued again
        assert row.dft_attempt == 1

    def test_the_remedy_that_was_applied_is_recorded(self, flat_cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._fail(flat_cfg, store, sid, tmp_path,
                   status=JobStatus(job_id="1", state=JobState.done,
                                    raw_state="COMPLETED"))
        assert "NSW" in store.get_structure(sid).dft_last_remedy

    def test_a_timeout_is_retried(self, flat_cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._fail(flat_cfg, store, sid, tmp_path,
                   status=JobStatus(job_id="1", state=JobState.timeout,
                                    raw_state="TIMEOUT"))
        assert store.get_structure(sid).state == "selected"

    def test_command_not_found_is_never_retried(self, flat_cfg, store, tmp_path):
        """Exit 127 hit this account 287 times in 60 days; retrying cannot fix it."""
        sid = add_selected(store, 1)[0]
        self._fail(flat_cfg, store, sid, tmp_path,
                   status=JobStatus(job_id="1", state=JobState.failed,
                                    exit_code=127, raw_state="FAILED"))
        row = store.get_structure(sid)
        assert row.state == "failed" and row.dft_attempt == 1

    def test_the_ladder_runs_out_and_says_why(self, flat_cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._fail(flat_cfg, store, sid, tmp_path, attempt=99,
                   status=JobStatus(job_id="1", state=JobState.done,
                                    raw_state="COMPLETED"))
        row = store.get_structure(sid)
        assert row.state == "failed"
        assert "relax" in row.dft_fail_reason

    def test_a_success_resets_the_attempt_counter(self, flat_cfg, store, tmp_path):
        from cspflow.stages.base import WorkItem

        sid = add_selected(store, 1)[0]
        store.set_structure_state(sid, StructureState.selected, **{ATTEMPT_KEY: 2})
        item = WorkItem(key=f"dft-{sid}-relax", structure_ids=[sid],
                        payload={"step": 0, "step_name": "relax", "attempt": 2})
        fake_job_output(tmp_path / item.key, converged=True)
        DftStage(flat_cfg).reconcile(store, {"workdir": str(tmp_path), "id": 1},
                                JobStatus(job_id="1", state=JobState.done,
                                          raw_state="COMPLETED"), [item])
        assert store.get_structure(sid).dft_attempt == 0


@has_potcars
class TestBuild:
    def test_it_writes_a_complete_input_directory_per_task(self, cfg, store, tmp_path):
        """Inputs are assembled on the login node, so an unresolvable POTCAR
        fails in milliseconds rather than after the queue wait."""
        add_selected(store, 2)
        stage = DftStage(cfg)
        items = stage.claim(store, budget=10)
        for item in items:
            stage.build([item], tmp_path / "dft")
            directory = step_dir(stage, item, tmp_path / "dft")
            for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "inputs.json"):
                assert (directory / name).is_file(), f"{item.key}/{name}"

    def test_one_structure_is_one_job_and_the_job_is_named_after_it(
            self, cfg, store, tmp_path):
        """D135. An array names itself after task 0 and hands every task that
        one name, one walltime and one `--mem`; with one structure per job the
        name IS the directory and there is nothing left to look up."""
        add_selected(store, 1)
        stage = DftStage(cfg)
        item = stage.claim(store, budget=1)[0]
        spec = stage.build([item], tmp_path / "dft")

        assert stage.solo_jobs
        assert spec.array_size == 0, "a lone structure must not go out as an array"
        assert spec.name.endswith(item.key), spec.name
        assert spec.name.startswith("t-"), "the campaign is not in the name"
        # squeue is per USER, so two campaigns numbering from 1 collide without it
        assert "0001" in spec.name and "Fe" in spec.name
        # The script and the log belong with the structure they describe.
        assert spec.workdir == stage.layout.run_root(
            item.structure_ids[0], item.payload.get("formula", ""),
            item.payload.get("source_path", ""))
        # No manifest: there is no index to resolve.
        assert not list((tmp_path / "dft").glob("*.tasks.json"))

    def test_the_task_manifest_lists_every_directory(self, cfg, store, tmp_path):
        add_selected(store, 2)
        stage = DftStage(cfg)
        items = stage.claim(store, budget=10)
        stage.build(items, tmp_path / "dft")
        manifest = json.loads(
            next((tmp_path / "dft").glob("*.tasks.json")).read_text())
        assert len(manifest["dirs"]) == 2

    def test_the_settings_hash_is_carried_on_the_item(self, cfg, store, tmp_path):
        add_selected(store, 1)
        stage = DftStage(cfg)
        items = stage.claim(store, budget=1)
        stage.build(items, tmp_path / "dft")
        assert len(items[0].payload["settings_hash"]) == 64

    def test_the_command_uses_the_machine_profile_binary(self, cfg, store, tmp_path):
        add_selected(store, 1)
        stage = DftStage(cfg)
        spec = stage.build(stage.claim(store, budget=1), tmp_path / "dft")
        assert "vasp_std" in spec.command
        assert "srun" in spec.command

    def test_vasp_done_is_written_as_a_process_marker_only(self, cfg, store, tmp_path):
        """It means VASP exited. 61% of the legacy campaign wrote it unconverged."""
        add_selected(store, 1)
        stage = DftStage(cfg)
        spec = stage.build(stage.claim(store, budget=1), tmp_path / "dft")
        assert "VASP_DONE" in spec.command


class TestPlumbing:
    def test_it_is_a_submitted_stage(self, cfg):
        assert DftStage(cfg).in_process is False

    def test_it_sits_last_before_analyze(self):
        from cspflow.driver import STAGE_ORDER

        assert STAGE_ORDER.index("filter") < STAGE_ORDER.index("dft")
        assert STAGE_ORDER.index("dft") < STAGE_ORDER.index("analyze")

    def test_the_recipe_is_the_campaigns(self, cfg):
        assert DftStage(cfg).recipe.stage_names == ["relax", "static"]


# -- the ladder has to change something ------------------------------------

class TestTheLadderActuallyApplies:
    """A retry that reruns the identical calculation is worse than no retry.

    It costs the same again, fails the same way, and looks like diligence. The
    remedy was recorded on the structure row as `dft_last_remedy` and read by
    nothing: `claim` never put it in the payload, so `_write_inputs` always saw
    an empty override dict.

    Found live: three VASP relaxations hit the ionic step limit, the ladder
    recorded `{"NSW": 200}`, and the INCAR written for the retry said `NSW = 99`.
    """

    def test_the_incar_override_reaches_the_written_incar(self, cfg, store, tmp_path):
        [sid] = add_selected(store, 1)
        store.set_structure_state(sid, StructureState.selected,
                                  **{ATTEMPT_KEY: 1,
                                     "dft_last_remedy": json.dumps(
                                         {"set": {"NSW": 200}, "remedy": ""})})
        stage = DftStage(cfg)
        items = stage.claim(store, budget=1)
        assert items[0].payload["incar_overrides"] == {"NSW": 200}

    def test_the_older_remedy_shape_still_reads(self, cfg, store):
        """It was stored as the bare override dict before the remedy was added."""
        from cspflow.stages.dft_stage import _decode_remedy

        # `resources` joined the shape when the ladder learned to raise --mem;
        # the legacy override still decodes into `set` exactly as before.
        assert _decode_remedy(json.dumps({"NSW": 200})) == {"set": {"NSW": 200},
                                                            "resources": {},
                                                            "remedy": ""}
        assert _decode_remedy(None) == {}
        assert _decode_remedy("not json") == {}

    def test_the_remedy_name_is_carried_too(self, cfg, store):
        [sid] = add_selected(store, 1)
        store.set_structure_state(sid, StructureState.selected,
                                  **{"dft_last_remedy": json.dumps(
                                      {"set": {}, "remedy": "resume_from_contcar"})})
        items = DftStage(cfg).claim(store, budget=1)
        assert items[0].payload["remedy"] == "resume_from_contcar"


class TestArchivingAndResuming:
    """A retry must not erase the evidence of what it is retrying."""

    def test_a_previous_attempt_is_moved_aside(self, tmp_path):
        from cspflow.stages.dft_stage import _archive_previous

        d = tmp_path / "job"
        d.mkdir()
        (d / "OUTCAR").write_text("old outcar")
        (d / "OSZICAR").write_text("old oszicar")
        (d / "INCAR").write_text("NSW = 99")

        archived = _archive_previous(d, attempt=1)
        assert archived == d / "attempt-0"
        assert (archived / "OUTCAR").read_text() == "old outcar"
        assert not (d / "OUTCAR").exists()

    def test_a_first_attempt_archives_nothing(self, tmp_path):
        from cspflow.stages.dft_stage import _archive_previous

        d = tmp_path / "job"
        d.mkdir()
        assert _archive_previous(d, attempt=0) is None

    def test_archiving_twice_does_not_lose_the_first_archive(self, tmp_path):
        from cspflow.stages.dft_stage import _archive_previous

        d = tmp_path / "job"
        d.mkdir()
        (d / "OUTCAR").write_text("first")
        _archive_previous(d, attempt=1)
        (d / "OUTCAR").write_text("second")
        second = _archive_previous(d, attempt=1)
        assert (second / "OUTCAR").read_text() == "first"

    def test_resuming_reads_the_previous_contcar(self, tmp_path):
        import ase.io
        from ase.build import bulk

        from cspflow.stages.dft_stage import _read_contcar

        atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
        path = tmp_path / "CONTCAR"
        ase.io.write(str(path), atoms, format="vasp")
        read = _read_contcar(path)
        assert read is not None and len(read) == len(atoms)

    def test_an_empty_contcar_is_not_a_resume(self, tmp_path):
        from cspflow.stages.dft_stage import _read_contcar

        path = tmp_path / "CONTCAR"
        path.write_text("")
        assert _read_contcar(path) is None
        assert _read_contcar(tmp_path / "absent") is None

    @has_potcars
    def test_a_resume_starts_from_the_relaxed_geometry(self, cfg, store, tmp_path):
        import ase.io

        [sid] = add_selected(store, 1)
        stage = DftStage(cfg)
        workdir = tmp_path / "dft"

        # First attempt: write inputs, then pretend VASP ran and moved the cell.
        first = stage.claim(store, budget=1)
        stage.build(first, workdir)
        directory = step_dir(stage, first[0], workdir)
        (directory / "OUTCAR").write_text("pretend")
        moved = ase.io.read(str(directory / "POSCAR"), format="vasp")
        moved.set_cell(moved.get_cell() * 1.05, scale_atoms=True)
        ase.io.write(str(directory / "CONTCAR"), moved, format="vasp")

        # Second attempt, with the resume remedy recorded.
        store.set_structure_state(
            sid, StructureState.selected,
            **{ATTEMPT_KEY: 1, "dft_last_remedy": json.dumps(
                {"set": {"NSW": 200}, "remedy": "resume_from_contcar"})})
        second = stage.claim(store, budget=1)
        stage.build(second, workdir)

        assert (directory / "attempt-0" / "OUTCAR").is_file()
        assert second[0].payload["started_from"].endswith("attempt-0/CONTCAR")
        written = ase.io.read(str(directory / "POSCAR"), format="vasp")
        assert written.get_volume() == pytest.approx(moved.get_volume(), rel=1e-6)
        assert "NSW = 200" in (directory / "INCAR").read_text()

    @has_potcars
    def test_a_resume_keeps_the_grid_it_was_already_running(self, cfg, store,
                                                            tmp_path):
        """Re-deriving the grid from the relaxed cell is what turned a resume
        into a restart on a different energy surface.

        The grid is floor(mult / length), so a cell that expands past an
        integer boundary during the relaxation gets a COARSER grid on resume.
        Structure 2498 of RE-magnets-CHGNet went 2x2x2 -> 2x2x1 that way, its
        forces climbed from 0.069 to 0.244 eV/A over 185 further ionic steps,
        and it was killed at walltime.

        The bcc Fe fixture at a = 2.87 A sits on an 8x8x8 grid and the 8 -> 7
        boundary is at 3.1416 A, so the 10% expansion below crosses it. That
        crossing is asserted, not assumed: a fixture that stopped crossing
        would leave this test passing while testing nothing.
        """
        import ase.io

        from cspflow.dft.vasp.kpoints import grid_for

        [sid] = add_selected(store, 1)
        stage = DftStage(cfg)
        workdir = tmp_path / "dft"

        first = stage.claim(store, budget=1)
        stage.build(first, workdir)
        directory = step_dir(stage, first[0], workdir)
        before = (directory / "KPOINTS").read_text().splitlines()[3].split()

        (directory / "OUTCAR").write_text("pretend")
        moved = ase.io.read(str(directory / "POSCAR"), format="vasp")
        moved.set_cell(moved.get_cell() * 1.10, scale_atoms=True)
        ase.io.write(str(directory / "CONTCAR"), moved, format="vasp")

        kpoints = stage.recipe.stages[0].kpoints
        would_be = grid_for(list(moved.cell.lengths()), kpoints)
        assert [str(would_be.a), str(would_be.b), str(would_be.c)] != before, (
            "the fixture no longer crosses a grid boundary, so this test would "
            "pass whether or not the grid is carried")

        store.set_structure_state(
            sid, StructureState.selected,
            **{ATTEMPT_KEY: 1, "dft_last_remedy": json.dumps(
                {"set": {"NSW": 200}, "remedy": "resume_from_contcar"})})
        second = stage.claim(store, budget=1)
        stage.build(second, workdir)

        after = (directory / "KPOINTS").read_text().splitlines()[3].split()
        assert after == before, (
            f"resume changed the k-point grid {before} -> {after}")

    def test_the_archived_grid_is_read_from_the_manifest(self, tmp_path):
        from cspflow.stages.dft_stage import _archived_grid

        d = tmp_path / "attempt-0"
        d.mkdir()
        (d / "inputs.json").write_text(json.dumps(
            {"kpoints": {"a": 2, "b": 2, "c": 2,
                         "scheme": "reciprocal_density", "gamma": True}}))
        grid = _archived_grid(d)
        assert (grid.a, grid.b, grid.c) == (2, 2, 2)

    def test_the_archived_grid_falls_back_to_the_kpoints_file(self, tmp_path):
        """A directory adopted from elsewhere has KPOINTS and no manifest."""
        from cspflow.stages.dft_stage import _archived_grid

        d = tmp_path / "attempt-0"
        d.mkdir()
        (d / "KPOINTS").write_text("legacy\n0\nGamma\n5 3 3\n")
        grid = _archived_grid(d)
        assert (grid.a, grid.b, grid.c) == (5, 3, 3)

    def test_no_grid_to_read_is_not_a_guessed_one(self, tmp_path):
        from cspflow.stages.dft_stage import _archived_grid

        d = tmp_path / "attempt-0"
        d.mkdir()
        assert _archived_grid(d) is None


def test_an_already_archived_directory_still_offers_its_previous_attempt(tmp_path):
    """Returning None for a directory whose outputs a previous cycle already
    archived is how a resume quietly turns back into a restart."""
    from cspflow.stages.dft_stage import _archive_previous, _latest_archive

    d = tmp_path / "job"
    (d / "attempt-0").mkdir(parents=True)
    (d / "attempt-0" / "CONTCAR").write_text("relaxed")
    assert _archive_previous(d, attempt=1) == d / "attempt-0"
    assert _latest_archive(d) == d / "attempt-0"


def test_the_latest_archive_wins(tmp_path):
    from cspflow.stages.dft_stage import _latest_archive

    d = tmp_path / "job"
    for n in (0, 1, 2):
        (d / f"attempt-{n}").mkdir(parents=True)
    (d / "attempt-notanumber").mkdir()
    assert _latest_archive(d) == d / "attempt-2"


def test_no_archive_at_all_is_none(tmp_path):
    from cspflow.stages.dft_stage import _latest_archive

    d = tmp_path / "job"
    d.mkdir()
    assert _latest_archive(d) is None


class TestOptionsThatDidNothing:
    """Four documented options were in the schema and the shipped template and
    read by no code at all. Same family as B29: a value recorded and never
    applied is worse than an absent one, because the config file says it works.
    """

    def test_a_static_step_is_not_judged_by_the_force_criterion(self, tmp_path):
        """VASP never prints "reached required accuracy" for IBRION=-1, NSW=0,
        so a static run was always "not converged" and the ladder retried it.

        Found live: a static step reported 200 ionic steps, having inherited
        NSW=200 from the relax step's remedy and been retried once.
        """
        from cspflow.dft.vasp.parse import read_job_directory

        d = tmp_path / "static"
        d.mkdir()
        (d / "INCAR").write_text("IBRION = -1\nNSW = 0\n")
        (d / "OUTCAR").write_text(
            "   NIONS =      2\n"
            " General timing and accounting informations for this job:\n"
            "                         Elapsed time (sec):     10.0\n")
        (d / "OSZICAR").write_text("   1 F= -.20E+02 E0= -.20E+02  d E =0.0  mag=  1.0\n")
        outcome = read_job_directory(d)
        assert outcome.converged
        assert outcome.state == "done"
        assert outcome.exit_reason == ""

    def test_a_relax_step_still_needs_the_force_criterion(self, tmp_path):
        from cspflow.dft.vasp.parse import read_job_directory

        d = tmp_path / "relax"
        d.mkdir()
        (d / "INCAR").write_text("IBRION = 1\nNSW = 99\n")
        (d / "OUTCAR").write_text(
            "   NIONS =      2\n"
            " General timing and accounting informations for this job:\n")
        (d / "OSZICAR").write_text(
            "".join(f"  {i} F= -.20E+02 E0= -.20E+02  d E =0.0  mag=  1.0\n"
                    for i in range(1, 100)))
        outcome = read_job_directory(d)
        assert not outcome.converged
        assert outcome.exit_reason == "ionic_step_limit"

    def test_a_static_that_ran_out_of_scf_steps_is_not_converged(self, tmp_path):
        """D151. A static has no force criterion, so the old rule fell back on
        `outcar.finished` -- "VASP wrote its epilogue", a statement about the
        process. A static whose SCF gave up at NELM writes that epilogue exactly
        like a converged one, so every static in RE-magnets-CHGNet read as
        converged: 2,245 of 2,245 rows, 155 of them wrong.
        """
        from cspflow.dft.vasp.parse import read_job_directory

        d = tmp_path / "static"
        d.mkdir()
        (d / "INCAR").write_text("IBRION = -1\nNSW = 0\nNELM = 200\n")
        (d / "OUTCAR").write_text(
            "   NIONS =      2\n"
            " General timing and accounting informations for this job:\n")
        (d / "OSZICAR").write_text(
            "RMM: 200    -0.20E+02    0.84E-02   -0.79E-04 11025   0.55E-02\n"
            "   1 F= -.20E+02 E0= -.20E+02  d E =0.0  mag=  1.0\n")
        outcome = read_job_directory(d)
        assert not outcome.converged
        assert outcome.exit_reason == "scf_not_converged", (
            "must name the rung that fits -- ALGO/NELM, not more ionic steps")

    def test_a_static_whose_scf_converged_is_still_converged(self, tmp_path):
        """The guard on the guard: the fix must not re-break what it fixed."""
        from cspflow.dft.vasp.parse import read_job_directory

        d = tmp_path / "static"
        d.mkdir()
        (d / "INCAR").write_text("IBRION = -1\nNSW = 0\nNELM = 200\n")
        (d / "OUTCAR").write_text(
            "   NIONS =      2\n"
            " General timing and accounting informations for this job:\n")
        (d / "OSZICAR").write_text(
            "RMM:  7    -0.20E+02    0.84E-06   -0.79E-08 11025   0.55E-06\n"
            "   1 F= -.20E+02 E0= -.20E+02  d E =0.0  mag=  1.0\n")
        outcome = read_job_directory(d)
        assert outcome.converged
        assert outcome.exit_reason == ""

    def test_a_relax_whose_final_scf_gave_up_is_not_converged(self, tmp_path):
        """"reached required accuracy" on forces is not enough if the electronic
        structure underneath it never converged."""
        from cspflow.dft.vasp.parse import read_job_directory

        d = tmp_path / "relax"
        d.mkdir()
        (d / "INCAR").write_text("IBRION = 1\nNSW = 99\nNELM = 200\n")
        (d / "OUTCAR").write_text(
            "   NIONS =      2\n"
            " reached required accuracy - stopping structural energy minimisation\n"
            " General timing and accounting informations for this job:\n")
        (d / "OSZICAR").write_text(
            "RMM: 200    -0.20E+02    0.84E-02   -0.79E-04 11025   0.55E-02\n"
            "  12 F= -.20E+02 E0= -.20E+02  d E =0.0  mag=  1.0\n")
        outcome = read_job_directory(d)
        assert not outcome.converged
        assert outcome.exit_reason == "scf_not_converged"

    def test_advancing_a_step_clears_the_previous_remedy(self, cfg, store, tmp_path):
        """A remedy belongs to the step that failed. Carried forward it rewrites
        the next step's INCAR -- live, a relax retry's NSW=200 landed in the
        static INCAR and a fixed-position calculation ran 200 ionic steps."""
        [sid] = add_selected(store, 1)
        store.set_structure_state(
            sid, StructureState.selected,
            **{"dft_last_remedy": json.dumps({"set": {"NSW": 200}, "remedy": ""})})
        stage = DftStage(cfg)
        outcome = type("O", (), {"energy": -1.0, "e_per_atom": -0.5,
                                 "magnetisation": None})()
        stage._advance(store, sid, step=0, outcome=outcome, directory=tmp_path)
        assert store.get_structure(sid).key_value_pairs["dft_last_remedy"] == ""

    def test_rank_by_orders_the_fresh_candidates(self, cfg, store, tmp_path):
        ids = add_selected(store, 3)
        for sid, hull in zip(ids, (0.30, 0.05, 0.20)):
            store.update_structure(sid, e_above_hull_mlip=hull)
        cfg.campaign.dft.select.rank_by = "e_above_hull_mlip"
        ready = DftStage(cfg)._ready(store)
        assert [int(r.id) for r in ready] == [ids[1], ids[2], ids[0]]

    def test_a_candidate_with_no_ranking_value_sorts_last(self, cfg, store):
        ids = add_selected(store, 2)
        store.update_structure(ids[1], e_above_hull_mlip=0.9)
        cfg.campaign.dft.select.rank_by = "e_above_hull_mlip"
        ready = DftStage(cfg)._ready(store)
        assert [int(r.id) for r in ready] == [ids[1], ids[0]]

    def test_max_total_caps_the_campaign(self, cfg, store):
        add_selected(store, 5)
        cfg.campaign.dft.select.max_total = 2
        assert len(DftStage(cfg)._ready(store)) == 2

    def test_a_structure_already_started_keeps_its_place(self, cfg, store):
        """Abandoning a half-finished relaxation to start a better-ranked one
        from scratch spends more and finishes less."""
        ids = add_selected(store, 3)
        store.update_structure(ids[2], dft_step=1, e_above_hull_mlip=0.9)
        for sid, hull in zip(ids[:2], (0.01, 0.02)):
            store.update_structure(sid, e_above_hull_mlip=hull)
        cfg.campaign.dft.select.rank_by = "e_above_hull_mlip"
        cfg.campaign.dft.select.max_total = 1
        ready = DftStage(cfg)._ready(store)
        assert [int(r.id) for r in ready] == [ids[2]]


class TestWhatTheReportReads:
    """`dft_dir` is the ONLY thing connecting a finished calculation to the
    report, and nothing tested that the two agree.

    `report/structure.py` reads the row's `dft_dir` and looks for a CONTCAR and
    an OUTCAR inside it. The DFT stage writes that key. Neither knows anything
    about the other, and neither would fail loudly if they disagreed: the report
    falls back to the database geometry and adds a note, so a campaign whose
    every card silently showed the UNRELAXED cell would still produce a report
    that looks finished. That is the failure this pins.
    """

    @has_potcars
    def test_the_recorded_directory_is_the_one_the_report_reads(
            self, cfg, store, tmp_path):
        import ase.io

        from cspflow.report.structure import detail

        [sid] = add_selected(store, 1)
        stage = DftStage(cfg)
        workdir = tmp_path / "dft"
        [item] = stage.claim(store, budget=1)
        stage.build([item], workdir)

        # Both steps finish, as a combined job runs them.
        run_root = stage.layout.run_root(sid, item.payload.get("formula", ""),
                                         item.payload.get("source_path", ""))
        for name in ("relax", "static"):
            d = fake_job_output(run_root / name, converged=True)
            ase.io.write(str(d / "CONTCAR"),
                         bulk("Fe", "bcc", a=2.95, cubic=True), format="vasp")

        stage.reconcile(store, {"workdir": str(workdir), "id": 1},
                        JobStatus(job_id="1", state=JobState.done,
                                  raw_state="COMPLETED"), [item])

        recorded = Path(store.get_structure(sid).key_value_pairs[DIR_KEY])
        assert recorded.is_dir(), f"dft_dir points nowhere: {recorded}"
        assert (recorded / "OUTCAR").is_file()
        assert (recorded / "CONTCAR").is_file()
        # The LAST step, because that is the energy that goes on the hull.
        assert recorded.name == "static"

        # And the report actually uses it, rather than falling back silently.
        d = detail(store, sid)
        assert d.geometry_source == "dft-contcar", (
            f"the report fell back to {d.geometry_source!r}; its cards would "
            f"show the unrelaxed cell. Notes: {d.notes}")


class TestTheNextStepStartsFromTheRelaxedGeometry:
    """`static` exists to give a high-accuracy energy AT THE RELAXED GEOMETRY,
    and its energy is what goes onto the DFT hull.

    Started from the structure in the database it runs on the generated cell
    instead and reports a number that looks entirely plausible and is wrong by
    whatever the relaxation was worth. Measured live before the fix: static
    POSCARs at 176.15, 172.92 and 260.70 A^3 against relax CONTCARs at 179.03,
    179.71 and 260.84.
    """

    @has_potcars
    def test_the_static_step_uses_the_relax_contcar(self, cfg, store, tmp_path):
        import ase.io

        [sid] = add_selected(store, 1)
        stage = DftStage(cfg)
        workdir = tmp_path / "dft"

        relax_items = stage.claim(store, budget=1)
        stage.build(relax_items, workdir)
        relax_dir = step_dir(stage, relax_items[0], workdir, "relax")
        relaxed = ase.io.read(str(relax_dir / "POSCAR"), format="vasp")
        relaxed.set_cell(relaxed.get_cell() * 1.04, scale_atoms=True)
        ase.io.write(str(relax_dir / "CONTCAR"), relaxed, format="vasp")

        outcome = type("O", (), {"energy": -1.0, "e_per_atom": -0.5,
                                 "magnetisation": None})()
        stage._advance(store, sid, step=0, outcome=outcome, directory=relax_dir)

        static_items = stage.claim(store, budget=1)
        assert static_items[0].payload["step_name"] == "static"
        stage.build(static_items, workdir)
        written = ase.io.read(
            str(step_dir(stage, static_items[0], workdir, "static") / "POSCAR"),
            format="vasp")
        assert written.get_volume() == pytest.approx(relaxed.get_volume(), rel=1e-6)
        assert static_items[0].payload["started_from"].endswith("CONTCAR")

    @has_potcars
    def test_a_missing_contcar_refuses_rather_than_using_the_generated_cell(
            self, cfg, store, tmp_path):
        """Silently falling back is exactly the failure this guards."""
        [sid] = add_selected(store, 1)
        store.set_structure_state(sid, StructureState.selected,
                                  **{STEP_KEY: 1, DIR_KEY: str(tmp_path / "gone")})
        stage = DftStage(cfg)
        items = stage.claim(store, budget=1)
        with pytest.raises(InputError, match="unrelaxed geometry"):
            stage.build(items, tmp_path / "dft")

    def test_the_first_step_needs_no_previous_geometry(self, cfg, store, tmp_path):
        [sid] = add_selected(store, 1)
        items = DftStage(cfg).claim(store, budget=1)
        assert items[0].payload["step"] == 0
        assert "started_from" not in items[0].payload


def test_the_resource_hint_comes_from_the_recipe(cfg):
    """DFT resources live per recipe step, not in the campaign's `dft:` block,
    so nothing outside this stage can find them."""
    ntasks, walltime = DftStage(cfg).resource_hint()
    assert ntasks > 0
    assert walltime.count(":") == 2


@pytest.mark.parametrize("text,hours", [
    ("02:00:00", 2.0), ("12:30:00", 12.5), ("1-00:00:00", 24.0),
    ("2-06:00:00", 54.0), ("00:45:00", 0.75),
])
def test_walltime_parsing(text, hours):
    from cspflow.stages.dft_stage import _hours

    assert _hours(text) == pytest.approx(hours)


@has_potcars
class TestParallelisation:
    """KPAR and NCORE reach the INCAR the pipeline actually writes.

    `parallel.py` was complete, unit-tested and wired into `resolve_inputs`,
    and still did nothing: `resolve_inputs` only builds a plan when it is told
    `ntasks`, and `_write_inputs` did not pass it.  So every INCAR carried the
    recipe's fixed `NCORE = 8` and no `KPAR` at all -- the exact condition
    parallel.py was written to end.  Nothing failed, because nothing asserted
    on the INCAR the stage wrote; these tests do.
    """

    @staticmethod
    def _incar(cfg, store, tmp_path):
        add_selected(store, 1)
        stage = DftStage(cfg)
        items = stage.claim(store, budget=1)
        spec = stage.build(items, tmp_path / "dft")
        text = (step_dir(stage, items[0], tmp_path / "dft") / "INCAR").read_text()
        tags = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
        return {k.strip(): v.strip() for k, v in tags.items()}, spec

    def test_kpar_is_written(self, cfg, store, tmp_path):
        tags, _ = self._incar(cfg, store, tmp_path)
        assert "KPAR" in tags, "no KPAR: the parallel plan never ran"
        assert int(tags["KPAR"]) >= 1

    def test_ncore_comes_from_the_plan_not_the_recipe(self, cfg, store, tmp_path):
        """The recipe hardcodes NCORE = 8 for every structure; the plan picks
        per structure, from its own irreducible k-point count."""
        tags, spec = self._incar(cfg, store, tmp_path)
        assert int(tags["KPAR"]) * int(tags["NCORE"]) <= spec.ntasks

    def test_kpar_divides_the_rank_count_the_job_is_submitted_with(
            self, cfg, store, tmp_path):
        """The failure this guards is silent in the config and fatal at runtime.

        VASP requires KPAR to divide the rank count. `build` and `_write_inputs`
        each decide `ntasks` independently, so if they ever diverge the INCAR
        asks for a split the allocation cannot provide and VASP aborts at
        startup -- after the full queue wait, before one electronic step.
        """
        tags, spec = self._incar(cfg, store, tmp_path)
        assert spec.ntasks % int(tags["KPAR"]) == 0


# --------------------------------------------------------------------------
# One unbuildable structure must not stop the campaign
# --------------------------------------------------------------------------


def test_incar_error_is_not_an_input_error():
    """The fact that made a single seed fatal.

    `_write_inputs` catches `InputError`. `magmom_for` raises `IncarError`,
    which is not a subclass of it, so nothing between the stage and
    `Driver.run` caught it. On 2026-09-12 one In-bearing CePdGe seed -- the
    magnetism table had no initial moment for In, and `strict: true` refuses to
    guess -- killed the whole Phase B driver on its first cycle.
    """
    from cspflow.dft.vasp.incar import IncarError
    from cspflow.dft.vasp.inputs import InputError

    assert not issubclass(IncarError, InputError)


def test_a_structure_whose_inputs_fail_is_marked_and_the_batch_goes_on(
        cfg, store, tmp_path, monkeypatch):
    """The refusal is right; the blast radius was not."""
    from cspflow.dft.vasp.incar import IncarError

    add_selected(store, n=3)
    stage = DftStage(cfg)
    items = stage.claim(store, budget=10)
    assert len(items) == 3
    bad = items[1]

    real = stage._write_inputs

    def flaky(item, directory):
        if item is bad:
            raise IncarError("no initial moment for ['In']")
        return real(item, directory)

    monkeypatch.setattr(stage, "_write_inputs", flaky)
    stage.build(items, tmp_path / "dft")

    # The good two were still submitted...
    assert len(items) == 2
    assert bad not in items
    # ...and the bad one carries the reason, so `--why` can answer for it.
    row = store.get_structure(bad.structure_ids[0])
    assert row.key_value_pairs["state"] == "failed"
    assert "no initial moment" in row.key_value_pairs["fail_reason"]


def test_a_batch_where_every_structure_fails_still_raises(cfg, store, tmp_path,
                                                         monkeypatch):
    """Silently submitting nothing would look like progress."""
    from cspflow.dft.vasp.incar import IncarError
    from cspflow.dft.vasp.inputs import InputError

    add_selected(store, n=2)
    stage = DftStage(cfg)
    items = stage.claim(store, budget=10)
    monkeypatch.setattr(stage, "_write_inputs",
                        lambda i, d: (_ for _ in ()).throw(IncarError("nope")))
    with pytest.raises(InputError, match="none of the 2"):
        stage.build(items, tmp_path / "dft")


class TestTwoStepsThatDoNotDescribeTheSameCalculation:
    """Both steps can converge perfectly and still not agree.

    The static starts its SCF from the MAGMOM guess again -- no WAVECAR, no
    CHGCAR are kept -- so it re-finds the magnetic solution from scratch and can
    settle in a different one from the relaxation that produced its geometry.
    Nothing downstream noticed: two converged steps, two plausible energies, and
    a structure ranked on the energy of a magnetic state its geometry was never
    optimised for.

    Measured over 2,067 structures of RE-magnets-CHGNet: of the 46 whose energy
    moved more than 60 meV/atom between steps, 41 (89%) had also changed moment,
    against 3% of those that moved under 5 meV/atom.
    """

    def outcome(self, tmp_path, name, e_per_atom, mag, n_atoms=4):
        from cspflow.dft.vasp.parse import JobOutcome
        return JobOutcome(
            path=tmp_path / name, state="done", converged=True, n_ionic_steps=1,
            step_limit=0, energy=e_per_atom * n_atoms, e_per_atom=e_per_atom,
            magnetisation=mag, n_atoms=n_atoms, slurm_id="", core_hours=0.0,
            exit_reason="", potcar_symbols=[])

    def test_a_magnetic_flip_between_the_steps_is_flagged(self, tmp_path):
        """Structure 13836's real numbers: -2.735 -> +4.099 uB, 656 meV/atom."""
        from cspflow.dft.vasp.parse import step_consistency

        check = step_consistency(
            self.outcome(tmp_path, "relax", -7.000, -2.735),
            self.outcome(tmp_path, "static", -6.344, 4.099))
        assert not check.ok
        assert check.energy_shift == pytest.approx(656.0, abs=1.0)
        assert check.magmom_shift == pytest.approx(1.708, abs=0.01)
        assert "different magnetic states" in check.detail

    def test_an_ordinary_relax_to_static_shift_is_not_flagged(self, tmp_path):
        """The median real shift is 0.25 meV/atom. A gate that fires on that
        teaches people to ignore it."""
        from cspflow.dft.vasp.parse import step_consistency

        check = step_consistency(
            self.outcome(tmp_path, "relax", -7.00000, 12.0),
            self.outcome(tmp_path, "static", -7.00025, 12.0))
        assert check.ok
        assert check.detail == ""

    def test_a_big_shift_with_a_steady_moment_says_so(self, tmp_path):
        """5 of the 46 were not magnetic flips. The detail must not claim they
        were -- naming the wrong cause is worse than naming none."""
        from cspflow.dft.vasp.parse import step_consistency

        check = step_consistency(
            self.outcome(tmp_path, "relax", -7.000, 12.0),
            self.outcome(tmp_path, "static", -6.900, 12.0))
        assert not check.ok
        assert "not a magnetic flip" in check.detail

    def test_a_missing_number_is_not_evidence_of_a_problem(self, tmp_path):
        from cspflow.dft.vasp.parse import step_consistency

        blank = self.outcome(tmp_path, "static", -7.0, None)
        blank = blank.__class__(**{**blank.__dict__, "e_per_atom": None})
        check = step_consistency(self.outcome(tmp_path, "relax", -7.0, 1.0), blank)
        assert check.ok and not check.measurable

    @has_potcars
    def test_the_flag_reaches_the_row_and_the_funnel(self, cfg, store, tmp_path):
        """A warning nobody is shown is not a warning."""
        from cspflow.report.candidates import COLUMNS
        from cspflow.stages.dft_stage import DftStage

        [sid] = add_selected(store, 1)
        stage = DftStage(cfg)
        stage._record_consistency(
            store, sid, "static",
            self.outcome(tmp_path, "relax", -7.000, -2.735),
            self.outcome(tmp_path, "static", -6.344, 4.099))

        kv = next(store.structures(id=sid)).key_value_pairs
        assert kv["dft_static_shift_mev"] == pytest.approx(656.0, abs=1.0)
        assert "different magnetic states" in kv["dft_warning"]
        gates = [e["gate"] for e in store.filter_events(sid)]
        assert "dft:static:consistent_with_previous" in gates
        assert any(not e["passed"] for e in store.filter_events(sid))
        assert "dft_warning" in [name for name, _ in COLUMNS], \
            "the column must exist or the CSV drops the warning"
