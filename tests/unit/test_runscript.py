"""The combined job's shell body, exercised by actually running it.

Both bugs this file guards against were found by RUNNING the generated script,
not by reading it, and neither is visible in the source:

  1. Under the sbatch's `set -eo pipefail`, a failing VASP exits the shell
     before the next line. Written the obvious way -- `vasp; rc=$?` -- the
     `FAILED_STEP` marker was never written, so the retry ladder could not tell
     a relax failure from a static one and would pick a remedy that does not fit
     the cause.

  2. The convergence check was a line of python embedded in the shell. It passed
     `sys.argv[1]` (a `str`) to a parser that indexes it with `/`, raised
     TypeError, was read as "not converged", and RE-RAN a finished relaxation --
     precisely what the check exists to prevent.

So these tests run bash. A test that only inspected the rendered text would have
passed against both.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from cspflow.dft.runscript import FAILED_STEP, render

CAMPAIGN = Path("/projects/mmi/Ridwan/cspflow/campaigns/CeFeB/campaign.yaml")
REAL_RELAX = Path("/scratch/oridwan/cspflow/CeFeB/dft/dft-88-relax")


def write_and_run(tmp_path, steps, binary, run_dir):
    """Render the body into a real sbatch-like script and execute it."""
    body = render(steps, launcher="", binary=binary, campaign=CAMPAIGN,
                  ntasks=64, python=sys.executable)
    script = tmp_path / "job.sh"
    script.write_text('#!/bin/bash\nset -eo pipefail\nRUN="$1"\n' + body + "\n")
    proc = subprocess.run(["bash", str(script), str(run_dir)],
                          capture_output=True, text=True)
    return proc


UNCONVERGED_OUTCAR = """\
 NIONS =      2
 General timing and accounting informations for this job:
                  Elapsed time (sec):      100.0
"""
UNCONVERGED_OSZICAR = """\
DAV:   1    -0.100000000000E+02   -0.10E+02   -0.10E-01  100   0.1E+00
   1 F= -.10000000E+02 E0= -.10000000E+02  d E =0.100000E-01
"""
CONVERGED_OUTCAR = """\
 NIONS =      2
 reached required accuracy - stopping structural energy minimisation
 General timing and accounting informations for this job:
                  Elapsed time (sec):      100.0
"""


def fake_vasp(tmp_path, outcar, oszicar):
    """A binary that exits 0 and leaves the given OUTCAR -- what VASP does when
    it reaches NSW without meeting the force criterion."""
    binary = tmp_path / "fake_vasp.sh"
    binary.write_text(
        "#!/bin/bash\n"
        f"cat > OUTCAR <<'EOF'\n{outcar}EOF\n"
        f"cat > OSZICAR <<'EOF'\n{oszicar}EOF\n"
        "printf 'NSW = 99\\nNELM = 200\\nIBRION = 1\\n' > INCAR\n"
        "printf 'x\\n' > CONTCAR\n"
        "exit 0\n")
    binary.chmod(0o755)
    return str(binary)


def test_a_failing_step_records_which_step_it_was(tmp_path):
    """The `set -e` trap. Without the marker the ladder has to guess."""
    run = tmp_path / "run"
    (run / "relax").mkdir(parents=True)
    proc = write_and_run(tmp_path, ["relax", "static"], "false", run)
    assert proc.returncode != 0
    assert (run / FAILED_STEP).read_text().strip() == "relax"


def test_a_successful_step_gets_its_own_done_marker(tmp_path):
    """`true` is not a stand-in for a successful VASP: it exits 0 and writes no
    OUTCAR, which since D151 the step gate correctly refuses to call success."""
    run = tmp_path / "run"
    (run / "relax").mkdir(parents=True)
    binary = fake_vasp(tmp_path, CONVERGED_OUTCAR, UNCONVERGED_OSZICAR)
    write_and_run(tmp_path, ["relax"], binary, run)
    assert (run / "relax" / "VASP_DONE").is_file()
    assert not (run / FAILED_STEP).exists()


def test_no_done_marker_is_written_when_vasp_fails(tmp_path):
    """`VASP_DONE` means the process exited 0 and nothing more -- two placements
    were once made on that marker alone and both were wrong. It must at least
    mean that much."""
    run = tmp_path / "run"
    (run / "relax").mkdir(parents=True)
    write_and_run(tmp_path, ["relax"], "false", run)
    assert not (run / "relax" / "VASP_DONE").exists()


def test_the_directory_records_the_job_that_ran_it(tmp_path):
    run = tmp_path / "run"
    (run / "relax").mkdir(parents=True)
    write_and_run(tmp_path, ["relax"], "true", run)
    assert (run / "SLURM_TASK").is_file()


def test_a_stale_failure_marker_is_cleared_on_entry(tmp_path):
    """A resumed job must not be judged by its previous attempt's marker."""
    run = tmp_path / "run"
    (run / "relax").mkdir(parents=True)
    (run / FAILED_STEP).write_text("static")
    binary = fake_vasp(tmp_path, CONVERGED_OUTCAR, UNCONVERGED_OSZICAR)
    write_and_run(tmp_path, ["relax"], binary, run)
    assert not (run / FAILED_STEP).exists()


def test_an_empty_step_list_is_refused():
    with pytest.raises(ValueError):
        render([], launcher="", binary="x", campaign=CAMPAIGN, ntasks=64)


@pytest.mark.skipif(not (REAL_RELAX / "OUTCAR").is_file(),
                    reason="needs the finished CeFeB relax on this machine")
def test_a_converged_step_is_skipped_not_repeated(tmp_path):
    """The regression: a `static` that fails must never cost its `relax`.

    Uses a REAL converged relax, so the decision is made by the same parser the
    reconciler uses rather than by a marker file.
    """
    import shutil

    run = tmp_path / "run"
    (run / "relax").mkdir(parents=True)
    for f in ("OUTCAR", "OSZICAR", "CONTCAR", "INCAR", "KPOINTS", "POSCAR"):
        shutil.copy(REAL_RELAX / f, run / "relax" / f)

    proc = write_and_run(tmp_path, ["relax", "static"], "false", run)
    assert "already converged, skipping" in proc.stdout
    assert not (run / "relax" / "vasp.out").exists(), "the relax was re-run"
    # It went on to the static, prepared its inputs from CONTCAR, and failed there.
    assert (run / "static" / "INCAR").is_file()
    assert (run / FAILED_STEP).read_text().strip() == "static"


# ---------------------------------------------------------------------------
# D151: the exit code cannot decide whether to continue.
# ---------------------------------------------------------------------------


def test_a_relax_that_exits_zero_without_converging_does_not_run_the_static(tmp_path):
    """The defect this guards against, in one sentence: VASP exits 0 when it
    reaches NSW, so `rc` cannot tell a converged relaxation from an abandoned
    one, and the static used to run on a geometry that was not a minimum."""
    run = tmp_path / "run"
    (run / "relax").mkdir(parents=True)
    (run / "static").mkdir()
    binary = fake_vasp(tmp_path, UNCONVERGED_OUTCAR, UNCONVERGED_OSZICAR)

    proc = write_and_run(tmp_path, ["relax", "static"], binary, run)

    assert proc.returncode == 0, "the job did its work; it is not a crash"
    assert (run / FAILED_STEP).read_text().strip() == "relax"
    assert "did not converge" in proc.stderr
    assert not (run / "static" / "OUTCAR").exists(), "the static ran anyway"
    assert not (run / "static" / "vasp.out").exists(), "the static ran anyway"


def test_a_stale_static_is_superseded_not_reused(tmp_path):
    """A resumed job would otherwise SKIP the static and report an energy taken
    at the previous attempt's unconverged geometry. 23 run directories were in
    exactly that state when this was written."""
    run = tmp_path / "run"
    (run / "relax").mkdir(parents=True)
    (run / "static").mkdir()
    # A finished static left behind by an earlier attempt.
    (run / "static" / "OUTCAR").write_text(CONVERGED_OUTCAR)
    (run / "static" / "OSZICAR").write_text(UNCONVERGED_OSZICAR)
    binary = fake_vasp(tmp_path, UNCONVERGED_OUTCAR, UNCONVERGED_OSZICAR)

    proc = write_and_run(tmp_path, ["relax", "static"], binary, run)

    assert not (run / "static" / "OUTCAR").exists(), \
        "the stale static would be skipped and reported as the answer"
    moved = list((run / "static").glob("superseded-*/OUTCAR"))
    assert moved, "the stale output must be moved aside, not deleted"
    assert moved[0].read_text() == CONVERGED_OUTCAR
    assert "superseded" in proc.stdout


def test_a_converged_relax_still_proceeds(tmp_path):
    """The gate must not stop a healthy run."""
    run = tmp_path / "run"
    (run / "relax").mkdir(parents=True)
    binary = fake_vasp(tmp_path, CONVERGED_OUTCAR, UNCONVERGED_OSZICAR)

    proc = write_and_run(tmp_path, ["relax"], binary, run)

    assert proc.returncode == 0
    assert not (run / FAILED_STEP).exists()
    assert "relax: finished" in proc.stdout


def test_exiting_zero_with_no_outcar_is_not_success(tmp_path):
    """VASP always writes an OUTCAR. A step that exits 0 without one produced
    nothing, and before D151 the job marched on to the next step regardless."""
    run = tmp_path / "run"
    (run / "relax").mkdir(parents=True)
    (run / "static").mkdir()

    proc = write_and_run(tmp_path, ["relax", "static"], "true", run)

    assert proc.returncode == 0
    assert (run / FAILED_STEP).read_text().strip() == "relax"
    assert not (run / "static" / "vasp.out").exists()
