"""The body of a combined DFT job: every step of one structure, in one job.

WHY ONE JOB
    Measured on 29 CeFeB relax/static pairs, the queue wait BETWEEN the two
    jobs was 54 minutes median, 94 mean, 372 at worst -- 45.7 hours of pure
    wall-clock idle across those structures alone.  Nothing computes during it.
    The partitions here allow 30 days, so a combined 36-hour job is nowhere near
    a limit.

    The second reason matters more.  When a task is "one structure's complete
    DFT", every array is homogeneous -- and a batch can no longer mix a 24-hour
    relax with a 12-hour static and hand both the shorter walltime, which is a
    bug that really happened, to 13 relaxes at once (D130).

THE THREE THINGS THIS SCRIPT MUST GET RIGHT
    1. **Be idempotent.**  A step that already holds a converged run is skipped.
       Without this, a static that fails costs its relax as well, and a resumed
       job silently repeats hours of finished work.

    2. **Say WHICH step failed.**  The retry ladder is per-step: `timeout` on a
       relax means resume from CONTCAR, on a static it means ask for more time.
       A combined job that reports only "failed" makes the ladder guess, and a
       remedy that does not fit its cause is how a retry fails the same way
       twice at the same cost.  Hence `FAILED_STEP`.

    3. **Never write a success marker it has not earned.**  `VASP_DONE` means
       "the process exited", nothing more -- two placements were once made on
       that alone and both were wrong.  The marker goes down only after VASP
       returns 0, and the reconciler still reads the OUTCAR to decide anything.

WHAT IT LOOKS LIKE ON DISK AFTERWARDS
    runs/0088-Ce8Fe56B4-strain_.../
        relax/   INCAR KPOINTS POSCAR CONTCAR OUTCAR ... VASP_DONE
        static/  INCAR KPOINTS POSCAR         OUTCAR ... VASP_DONE
        FAILED_STEP        <- only if something failed; holds the step name
        SLURM_TASK         <- jobid_taskid, so the folder names its own job
"""

from __future__ import annotations

from pathlib import Path

#: Written by the script when a step fails; read by the reconciler to pick the
#: ladder rung that fits the cause.
FAILED_STEP = "FAILED_STEP"
#: Written on entry so a directory can name the job that ran it -- the reverse
#: of the manifest, and the lookup nobody had when a job name pointed elsewhere.
SLURM_TASK = "SLURM_TASK"


def render(steps: list[str], *, launcher: str, binary: str, campaign: Path,
           ntasks: int, python: str = "python") -> str:
    """The shell body for one structure's run directory.

    `steps` is the recipe's step names in order; `$RUN` is exported by the
    caller's preamble and is the run directory.
    """
    if not steps:
        raise ValueError("a combined job needs at least one step")

    lines = [
        'cd "$RUN"',
        # The id this directory is being run by, written from inside the job --
        # the one place both facts are known. A solo job (D135) records its
        # plain job id; a task of an array records `jobid_taskid`, because that
        # is what `scancel` and `sacct` will accept for the TASK rather than for
        # the whole array.
        'if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then',
        f'    echo "${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}" > {SLURM_TASK}',
        "else",
        f'    echo "${{SLURM_JOB_ID}}" > {SLURM_TASK}',
        "fi",
        f'rm -f {FAILED_STEP}',
        'echo "run directory: $RUN"',
        "",
        "# A step is skipped only when its OWN output says it converged -- not",
        "# because a marker file exists. VASP_DONE means the process exited.",
        "#",
        "# This calls a tested function rather than embedding python in the",
        "# shell. The embedded version passed a str to a parser that indexes it",
        "# with `/`, raised TypeError, was read as 'not converged', and re-ran a",
        "# finished relaxation -- the one thing the check exists to prevent.",
        "converged () {",
        f'    {python} -m cspflow.worker --is-converged "$1"',
        "}",
        "",
        # A step that did not converge invalidates every step after it: their
        # outputs describe a geometry that is about to change. Moving the OUTCAR
        # is what makes both `converged` above and the driver's reconcile treat
        # the step as not-run; without it, a resumed job SKIPS the later step and
        # reports the stale energy as the structure's answer. Measured: 23 run
        # directories were in exactly that state. Moved, never deleted.
        "supersede () {",
        '    for s in "$@"; do',
        '        [ -f "$s/OUTCAR" ] || continue',
        '        dest="$s/superseded-${SLURM_JOB_ID:-0}"',
        '        mkdir -p "$dest"',
        '        find "$s" -maxdepth 1 -type f -exec mv -t "$dest" {} +',
        '        echo "  superseded $s -> $dest"',
        "    done",
        "}",
        "",
    ]

    for i, step in enumerate(steps):
        lines += [
            f'# ---------- {step} ----------',
            f'if converged "{step}"; then',
            f'    echo "{step}: already converged, skipping"',
            "else",
        ]
        if i > 0:
            # The inputs cannot exist yet: this step runs at the PREVIOUS step's
            # relaxed geometry, and that CONTCAR is minutes old.
            lines += [
                f'    echo "{step}: preparing inputs from {steps[i - 1]}/CONTCAR"',
                f'    if ! {python} -m cspflow.worker --prepare-stage "$RUN" {step} \\',
                f'            --campaign {campaign} --ntasks {ntasks}; then',
                f'        echo "{step}" > {FAILED_STEP}',
                f'        echo "{step}: could not prepare inputs" >&2',
                "        exit 1",
                "    fi",
            ]
        # `if ! cmd; then` and NOT `cmd; rc=$?`.
        #
        # The sbatch preamble runs `set -eo pipefail`, under which a failing
        # command exits the shell IMMEDIATELY -- before the next line. Written
        # the obvious way, `rc=$?` never executes and no FAILED_STEP is written,
        # so the ladder cannot tell a relax failure from a static one and picks
        # a remedy that does not fit the cause. Verified by running the
        # generated script under `set -e` with a failing binary: the marker was
        # absent, exactly as this comment describes. A command inside an `if`
        # condition is exempt from `set -e`, which is why this form works.
        lines += [
            f'    cd "$RUN/{step}"',
            f'    rc=0; {launcher} {binary} > vasp.out 2>&1 || rc=$?',
            '    cd "$RUN"',
            "    if [ $rc -ne 0 ]; then",
            f'        echo "{step}" > {FAILED_STEP}',
            f'        echo "{step}: vasp exited $rc" >&2',
            "        exit $rc",
            "    fi",
            # Only now, and only for this step.
            f'    echo done > "{step}/VASP_DONE"',
        ]
        # The exit code cannot decide whether to continue. VASP exits 0 when it
        # reaches NSW without meeting the force criterion -- a clean exit from a
        # relaxation that did not converge -- and the old script read that as
        # permission to run the next step. The static then ran at a geometry
        # that was not a minimum, and its energy went into the database marked
        # usable. Measured before the fix: 19 structures, 5 of them differing
        # from their own relax energy by more than the 60 meV/atom selection
        # threshold, against 3% in the converged control. D151.
        #
        # `converged` is the same function the skip-check above already trusts.
        # It was written, tested and called -- just never asked this question.
        later = steps[i + 1:]
        lines += [
            f'    if ! converged "{step}"; then',
            f'        echo "{step}" > {FAILED_STEP}',
            f'        echo "{step}: finished but did not converge -- '
            f'not running {", ".join(later) or "any further step"}" >&2',
        ]
        if later:
            lines.append(f'        supersede {" ".join(later)}')
        lines += [
            # Exit 0 on purpose: the job did what it was asked, and the step's
            # own OUTCAR is what the ladder reads to pick a remedy. A non-zero
            # exit would have SLURM report a crash that did not happen.
            "        exit 0",
            "    fi",
            f'    echo "{step}: finished"',
            "fi",
            "",
        ]

    lines.append('echo "all steps done: ' + " ".join(steps) + '"')
    return "\n".join(lines)
