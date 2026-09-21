"""One job per structure instead of one job per structure per step.

WHY.  Measured on 29 CeFeB relax/static pairs, the queue wait between the two
jobs was 54 min median, 94 mean, 372 at worst -- 45.7 hours of wall-clock idle
across those structures alone, computing nothing.

The bigger reason is structural.  When a task is "one structure's complete
DFT", every array is homogeneous, and a batch can no longer mix a 24-hour relax
with a 12-hour static and hand both the shorter walltime.  That is not a
hypothetical: it happened to 13 relaxes in one CePdGe array (D130).

SAFETY.  CeFeB and CePdGe are mid-flight on the old flat layout.  Three
independent guards keep them there -- the schema default is `stages`, disk
evidence overrides configuration, and `combined` requires `runs` -- because
pointing a running campaign at a different layout orphans its finished work and
re-runs it.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cspflow.config.loader import load_campaign
from cspflow.stages.base import WorkItem
from cspflow.stages.dft_stage import DftStage

REAL = Path("/projects/mmi/Ridwan/cspflow/campaigns/CeFeB/campaign.yaml")
pytestmark = pytest.mark.skipif(not REAL.is_file(), reason="needs a real campaign")


def campaign_at(tmp_path, **dft):
    cfg = yaml.safe_load(REAL.read_text())
    cfg["workdir"] = str(tmp_path)
    block = cfg.setdefault("dft", {})
    # The real campaign may name a campaign-LOCAL recipe (`recipe: recipe.yaml`),
    # which resolves against ITS folder (D132) -- not against the tmp_path copy
    # this fixture writes. Pin it to an absolute path so the copy reads the same
    # file. Without this, switching CeFeB to a local recipe broke eight tests in
    # here that have nothing to do with recipe resolution: a fixture that reads a
    # live production config inherits every edit anyone makes to it.
    recipe = str(block.get("recipe", "magnets"))
    if recipe.endswith((".yaml", ".yml")) and not Path(recipe).is_absolute():
        block["recipe"] = str((REAL.parent / recipe).resolve())
    block.update(dft)
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return DftStage(load_campaign(path))


def item_for(sid=88, step=0, steps=("relax", "static")):
    return WorkItem(
        key="x", structure_ids=[sid],
        payload={"step": step, "step_name": steps[0], "attempt": 0,
                 "steps": list(steps), "formula": "Ce8Fe56B4",
                 "source_path": "/s/02-strain__Ce2Fe14B_strainiso_p0.00pct.vasp"})


# --- the guards ------------------------------------------------------------


def test_the_live_campaigns_keep_their_paths():
    """The property that matters most: nothing here may move CeFeB or CePdGe."""
    for name in ("CeFeB", "CePdGe"):
        p = Path(f"/projects/mmi/Ridwan/cspflow/campaigns/{name}/campaign.yaml")
        if not p.is_file():
            continue
        stage = DftStage(load_campaign(p))
        assert stage.layout.name == "stages"
        assert stage.combined is False
        got = stage._directory_for(
            WorkItem(key="dft-88-relax", structure_ids=[88],
                     payload={"step": 0, "step_name": "relax"}),
            stage.cfg.work_dir / "dft")
        assert got.name == "dft-88-relax", "a live campaign's paths moved"


def test_combined_requires_the_runs_layout(tmp_path):
    """A combined job needs somewhere to put both steps of one structure, and
    the flat layout has no such directory."""
    stage = campaign_at(tmp_path, layout="stages", combined_job=True)
    assert stage.combined is False


def test_disk_evidence_overrides_a_runs_setting(tmp_path):
    (tmp_path / "dft" / "dft-3-relax").mkdir(parents=True)
    (tmp_path / "dft" / "dft-3-relax" / "OUTCAR").write_text("x")
    stage = campaign_at(tmp_path, layout="runs", combined_job=True)
    assert stage.layout.name == "stages"
    assert stage.combined is False


# --- the layout ------------------------------------------------------------


def test_a_structure_gets_one_directory_holding_every_step(tmp_path):
    stage = campaign_at(tmp_path, layout="runs", combined_job=True)
    root = stage._directory_for(item_for(), tmp_path / "dft")
    assert root.parent.name == "runs"
    assert "0088" in root.name and "Ce8Fe56B4" in root.name
    assert "strain" in root.name, "the seed's provenance was dropped"


def test_the_old_layout_is_reproduced_exactly(tmp_path):
    stage = campaign_at(tmp_path, layout="stages", combined_job=False)
    got = stage._directory_for(
        WorkItem(key="dft-88-relax", structure_ids=[88],
                 payload={"step": 0, "step_name": "relax"}), tmp_path / "dft")
    assert got == tmp_path / "dft" / "dft-88-relax"


# --- the script the job will run -------------------------------------------


def test_the_rendered_body_covers_every_remaining_step(tmp_path):
    from cspflow.dft.runscript import render

    body = render(["relax", "static"], launcher="srun", binary="vasp_std",
                  campaign=REAL, ntasks=64)
    assert "---------- relax ----------" in body
    assert "---------- static ----------" in body
    # the static's inputs are prepared in-job, from the relax's CONTCAR
    assert "--prepare-stage" in body and "static" in body


def test_only_the_first_step_is_prepared_ahead_of_time(tmp_path):
    """A later step's POSCAR *is* the previous step's CONTCAR, which will not
    exist until the job has run. Writing it now would mean running the static on
    the unrelaxed cell -- a plausible, wrong hull energy."""
    from cspflow.dft.runscript import render

    body = render(["relax", "static"], launcher="srun", binary="vasp_std",
                  campaign=REAL, ntasks=64)
    before_relax = body.split("---------- relax ----------")[0]
    assert "--prepare-stage" not in before_relax


# --- resources for a job that runs its steps in sequence -------------------
#
# Both of these were found by BUILDING a real job and reading the JobSpec, not
# by reading the code. Neither is visible in the source.


def _step_resources(stage, name):
    """What the loaded recipe asks for at one step, inheritance applied."""
    steps = {s.name: s for s in stage.recipe.stages}
    return {**stage.recipe.stages[0].resources, **steps[name].resources}


def _seconds(walltime: str) -> int:
    """`[D-]HH:MM:SS` as seconds."""
    days, _, rest = str(walltime).partition("-")
    if not rest:
        days, rest = 0, days
    h, m, sec = (int(x) for x in rest.split(":"))
    return int(days) * 86400 + h * 3600 + m * 60 + sec


def test_the_walltime_is_the_sum_not_the_maximum(tmp_path):
    """A combined job runs relax THEN static inside one allocation, so it needs
    relax + static. The max is correct for a heterogeneous ARRAY -- separate
    tasks that each have to fit -- and wrong here in the dangerous direction: the
    first build produced the relax walltime alone, and the job would have died
    partway through the static having already spent a day of compute.

    Derived from the recipe, not hard-coded: these three assertions were written
    as the shipped 64 ranks / 24 h / 12 h, so pointing CeFeB at a campaign-local
    recipe at 16 ranks broke them while the arithmetic stayed right. The property
    is "sum", not "1-12:00:00".
    """
    stage = campaign_at(tmp_path, layout="runs", combined_job=True)
    got = stage._resources_for([item_for(steps=("relax", "static"))])
    want = sum(_seconds(_step_resources(stage, n)["time"]) for n in ("relax", "static"))
    assert _seconds(got["time"]) == want


def test_memory_and_ranks_stay_the_maximum(tmp_path):
    """Walltime is consumed in turn; memory and ranks are held concurrently.
    Summing those would ask for twice the cores the job can use."""
    stage = campaign_at(tmp_path, layout="runs", combined_job=True)
    got = stage._resources_for([item_for(steps=("relax", "static"))])
    want = max(int(_step_resources(stage, n)["ntasks"]) for n in ("relax", "static"))
    assert int(got["ntasks"]) == want
    # The point of the test: summed instead of maxed, it would be double.
    summed = sum(int(_step_resources(stage, n)["ntasks"]) for n in ("relax", "static"))
    assert int(got["ntasks"]) < summed


def test_a_single_step_run_is_not_inflated(tmp_path):
    stage = campaign_at(tmp_path, layout="runs", combined_job=True)
    got = stage._resources_for([item_for(steps=("static",))])
    assert _seconds(got["time"]) == _seconds(_step_resources(stage, "static")["time"])


def test_the_memory_estimate_is_not_skipped_in_a_run_directory(tmp_path):
    """`_mem_for` is handed the structure's ROOT when combined, and the INCAR is
    a level down in the first step's subdirectory. Looking for it at the root
    found nothing, the handler returned the machine default, and every
    per-structure estimate from D128 was silently skipped -- back to one flat
    32G, which is what OOM-killed three statics to begin with.

    A KPAR=8 structure must therefore ask for MORE than the default.
    """
    stage = campaign_at(tmp_path, layout="runs", combined_job=True)
    run = tmp_path / "run"
    (run / "relax").mkdir(parents=True)
    (run / "relax" / "INCAR").write_text(
        "NBANDS = 496\nENCUT = 520\nKPAR = 8\nNCORE = 2\n")
    # a real 68-atom cell
    import shutil
    shutil.copy("/scratch/oridwan/cspflow/CeFeB/dft/dft-18-static/POSCAR",
                run / "relax" / "POSCAR")

    got = stage._mem_for({}, [str(run)], 64)
    assert got.endswith("G")
    assert int(got[:-1]) > 32, f"estimate was skipped; got the default {got}"


class TestCampaignPathIsAbsolute:
    """The combined job's hand-off runs with cwd inside the run directory.

    `csp run` defaults `-c` to the bare string `campaign.yaml`, so the normal
    habit of running from inside the campaign folder stored a relative path that
    the compute node could not resolve. The relax finished and its result was
    stranded.
    """

    def test_a_relative_campaign_path_is_resolved(self, tmp_path, monkeypatch):
        from cspflow.stages.dft_stage import _campaign_file

        campaign = tmp_path / "my-campaign"
        campaign.mkdir()
        (campaign / "campaign.yaml").touch()
        monkeypatch.chdir(campaign)

        cfg = SimpleNamespace(campaign_path=Path("campaign.yaml"),
                              base_dir=campaign)
        got = _campaign_file(cfg)
        assert got.is_absolute()
        assert got == (campaign / "campaign.yaml").resolve()

    def test_a_missing_campaign_path_falls_back_to_the_campaign_folder(self, tmp_path):
        """Not to work_dir: campaign.yaml is never on scratch."""
        from cspflow.stages.dft_stage import _campaign_file

        cfg = SimpleNamespace(campaign_path=None, base_dir=tmp_path)
        got = _campaign_file(cfg)
        assert got == (tmp_path / "campaign.yaml").resolve()
        assert got.is_absolute()
