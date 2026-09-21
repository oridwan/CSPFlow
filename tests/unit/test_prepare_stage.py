"""The step that runs INSIDE a combined job, between two `vasp_std` calls.

WHY IT CANNOT RUN EARLIER.  A `static`'s POSCAR *is* the relax's CONTCAR, and
that file does not exist until the relax has finished.  So a job that runs both
steps has to regenerate the second step's inputs partway through.

WHY IT MATTERS.  `static` exists to give a high-accuracy energy AT THE RELAXED
GEOMETRY, and that energy is what goes on the DFT hull.  Run on the generated
cell instead, it reports a number that looks entirely plausible and is wrong by
whatever the relaxation was worth -- measured on this campaign at 176.15 vs
179.03 A^3, 172.92 vs 179.71, 260.70 vs 260.84.  Nothing about the output says
so.

VALIDATED AGAINST REALITY, not a mock: regenerating the static for CeFeB
structure 88 from its real finished relax reproduces the INCAR, KPOINTS and
POSCAR that cspflow actually ran, byte for byte.  The tests here cover the
guards around that path, which are the parts that fail quietly.
"""

from pathlib import Path

import pytest

from cspflow.worker import WorkerError, prepare_stage

CAMPAIGN = "/projects/mmi/Ridwan/cspflow/campaigns/CeFeB/campaign.yaml"
pytestmark = pytest.mark.skipif(
    not Path(CAMPAIGN).is_file(), reason="needs a real campaign for recipe + POTCARs")


def test_the_first_step_is_refused(tmp_path):
    """`relax` inputs are written when the job is built. Regenerating them here
    would mean silently starting from something other than the seed."""
    with pytest.raises(WorkerError, match="first step"):
        prepare_stage(tmp_path, "relax", CAMPAIGN)


def test_an_unknown_step_names_the_real_ones(tmp_path):
    with pytest.raises(WorkerError, match="not a step"):
        prepare_stage(tmp_path, "bogus", CAMPAIGN)


def test_a_missing_contcar_refuses_rather_than_using_the_seed(tmp_path):
    """The failure that matters. Falling back to the unrelaxed cell here is how
    a plausible, wrong energy reaches the hull."""
    (tmp_path / "relax").mkdir()
    with pytest.raises(WorkerError, match="relaxed geometry"):
        prepare_stage(tmp_path, "static", CAMPAIGN)


def test_an_empty_contcar_is_treated_as_missing(tmp_path):
    """A CONTCAR of zero bytes is the normal residue of a job killed early."""
    (tmp_path / "relax").mkdir()
    (tmp_path / "relax" / "CONTCAR").write_text("")
    with pytest.raises(WorkerError, match="missing or empty"):
        prepare_stage(tmp_path, "static", CAMPAIGN)


def test_an_already_converged_step_is_left_alone(tmp_path, monkeypatch):
    """What makes the combined script safe to re-run: a `static` that failed
    must never cost its `relax`."""
    (tmp_path / "static").mkdir()
    marker = tmp_path / "static" / "OUTCAR"
    marker.write_text("pretend")

    class _Outcome:
        converged = True
        energy = -1.0

    monkeypatch.setattr("cspflow.dft.vasp.parse.read_job_directory",
                        lambda d: _Outcome())
    out = prepare_stage(tmp_path, "static", CAMPAIGN)
    assert out == tmp_path / "static"
    # untouched: no inputs written over a finished run
    assert not (tmp_path / "static" / "INCAR").exists()
    assert marker.read_text() == "pretend"


REAL_RELAX = Path("/scratch/oridwan/cspflow/CeFeB/dft/dft-88-relax")
REAL_STATIC = Path("/scratch/oridwan/cspflow/CeFeB/dft/dft-88-static")


@pytest.mark.skipif(not (REAL_RELAX / "CONTCAR").is_file(),
                    reason="needs the finished CeFeB relax on this machine")
def test_ntasks_comes_from_the_argument_not_the_environment(tmp_path, monkeypatch):
    """Measured while writing this: `SLURM_NTASKS` read **2**, leaked from the
    surrounding allocation, and beat the recipe's 64 -- producing an INCAR with
    KPAR dropped and NCORE halved.

    KPAR must divide the job's OWN rank count or VASP refuses to start before a
    single electronic step, and an inherited value is not this job's allocation.
    `campaign_driver.sbatch` already unsets that variable for the same reason.
    """
    import shutil

    (tmp_path / "relax").mkdir()
    for f in ("CONTCAR", "OUTCAR", "OSZICAR", "INCAR", "KPOINTS", "POSCAR"):
        shutil.copy(REAL_RELAX / f, tmp_path / "relax" / f)

    monkeypatch.setenv("SLURM_NTASKS", "2")          # the leak
    out = prepare_stage(tmp_path, "static", CAMPAIGN, ntasks=64)
    incar = (out / "INCAR").read_text()
    assert "KPAR = 4" in incar, "KPAR was chosen against the leaked rank count"
    assert "NCORE = 4" in incar


@pytest.mark.skipif(not (REAL_STATIC / "INCAR").is_file(),
                    reason="needs the finished CeFeB static on this machine")
def test_it_reproduces_the_static_cspflow_actually_ran(tmp_path):
    """The real validation: regenerate structure 88's static from its own
    finished relax and diff against what the two-job path produced at the time.
    Identical, or the combined job is not a drop-in replacement."""
    import shutil

    (tmp_path / "relax").mkdir()
    for f in ("CONTCAR", "OUTCAR", "OSZICAR", "INCAR", "KPOINTS", "POSCAR"):
        shutil.copy(REAL_RELAX / f, tmp_path / "relax" / f)

    out = prepare_stage(tmp_path, "static", CAMPAIGN, ntasks=64)
    for f in ("INCAR", "KPOINTS", "POSCAR"):
        generated = (out / f).read_text().splitlines()
        actual = (REAL_STATIC / f).read_text().splitlines()
        if f == "INCAR":      # line 1 is a hash comment of the tags themselves
            generated, actual = generated[1:], actual[1:]
        assert generated == actual, f"{f} differs from the static that really ran"
