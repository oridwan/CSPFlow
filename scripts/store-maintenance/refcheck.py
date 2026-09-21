#!/usr/bin/env python
"""
refcheck.py -- why are reference structures not finished?

Walks the store and puts every relax and every static run into ONE bucket,
using VASP's own words rather than just the job's exit state.  The distinction
that matters: a run that finished cleanly at the ionic step limit is not a
crash, and a crash is not a walltime kill.  They need different fixes.

BUCKETS
    ok           finished and converged (for a static: finished at all)
    max_steps    finished cleanly but stopped at NSW without reaching EDIFFG
    not_conv     finished, below NSW, still did not reach the force criterion
    zbrent       "ZBRENT: fatal error in bracketing" -- the line minimiser lost
                 its bracket, usually within meV of the minimum.  Restart from
                 CONTCAR, ideally with IBRION=2.
    crash        VASP printed some other fatal error block and stopped
    avx512       "illegal instruction" -- landed on a node with no AVX-512.
                 A machine problem, not a physics one.
    running      no epilogue, but the OUTCAR was written minutes ago -- this
                 job is in flight, not broken
    killed       no epilogue, not running, no error block: walltime or OOM
    staged       inputs written, VASP never even started (no vasp.out).  Almost
                 always "still queued", not a failure.
    no_outcar    VASP ran -- there is a vasp.out -- but produced no OUTCAR
    setup_error  the job died BEFORE VASP -- writing inputs raised.  Invisible
                 in the structure folder (nothing was written), so this one is
                 read from the slurm logs in jobs/.
    not_started  no inputs written yet, and no job has tried

INPUTS
    $CSPFLOW_STORE/structures/<mp-id>-<formula>/{dft_relax,dft_static}/
    (default store: /projects/mmi/cspflow-shared/store)

OUTPUT
    a counts table, stage x bucket, printed to stdout

RUN
    export CSPFLOW_STORE=/projects/mmi/cspflow-shared/store
    python scripts/store-maintenance/refcheck.py                     # the table
    python scripts/store-maintenance/refcheck.py zbrent              # list those folders
    python scripts/store-maintenance/refcheck.py killed --stage static
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cspflow.dft.vasp.parse import read_job_directory, tail_text

STAGES = {"relax": "dft_relax", "static": "dft_static"}
BUCKETS = ["ok", "ignored", "running", "max_steps", "not_conv", "zbrent", "crash",
           "avx512", "killed", "no_outcar", "staged", "setup_error",
           "not_started"]

# Each job log opens with `=== <folder> on <host> ===`, so the logs can be
# attributed back to structures without consulting the scheduler.
LOG_HEADER = re.compile(r"^=== (\S+) on \S+ ===", re.M)

# An OUTCAR touched more recently than this is being written by a live job.
# A heuristic, but the alternative -- matching folders against squeue -- needs
# the task files of every array still in flight, which is a lot of machinery
# for a diagnostic.
RUNNING_SECONDS = 20 * 60

# VASP's fatal blocks are drawn inside a box of `|` characters, but so is its
# ordinary advice ("Mind: For very accurate calculation ..."), which is why
# matching the box alone reported four healthy running jobs as crashes.  Only
# look for an error line once a marker proves the box is a fatal one.
FATAL_MARKERS = ("REFUSE TO CONTINUE", "internal error", "ERROR:", "LAPACK: Routine")
# VASP's message is the boxed line that mentions an error, e.g.
#   |     VERY BAD NEWS! internal error in subroutine POSMAP: symmetry        |
# which does not start with a "WORD:" token, so match on the word instead.
ERROR_LINE = re.compile(r"\|\s{2,}(\S.*?(?:error|ERROR|BAD NEWS).*?)\s{2,}\|")


def classify(stage_dir: Path) -> tuple[str, str]:
    """One run directory -> (bucket, a short detail string)."""
    if not stage_dir.is_dir() or not (stage_dir / "INCAR").is_file():
        return "not_started", ""

    outcar = stage_dir / "OUTCAR"
    if not outcar.is_file():
        vout = stage_dir / "vasp.out"
        if not vout.is_file():
            # Inputs staged and nothing else.  Reporting this as a failure sends
            # you hunting for a bug in 106 structures that are simply still in
            # the queue -- which is exactly what happened.
            return "staged", "inputs written, VASP not started"
        text = tail_text(vout)
        if "illegal instruction" in text:
            return "avx512", "forrtl 168"
        last = text.strip().splitlines()
        return "no_outcar", (last[-1][:60] if last else "vasp.out is empty")

    # read_job_directory knows the rule a hand-rolled check keeps getting wrong:
    # a static step has no force criterion, so "not converged" is normal for it.
    job = read_job_directory(stage_dir)
    if job.state == "done" and job.converged:
        return "ok", ""
    if job.exit_reason == "ionic_step_limit":
        return "max_steps", f"{job.n_ionic_steps} steps"

    text = tail_text(outcar)
    if "ZBRENT" in text:
        return "zbrent", "bracketing"
    if any(marker in text for marker in FATAL_MARKERS):
        m = ERROR_LINE.search(text)
        return "crash", (m.group(1).strip()[:60] if m else "fatal error block")
    if job.state == "timeout":
        # No epilogue and no error.  Either a live job or a killed one; the
        # OUTCAR's age is what separates them.
        age = time.time() - outcar.stat().st_mtime
        if age < RUNNING_SECONDS:
            return "running", f"written {int(age / 60)} min ago"
        return "killed", "no epilogue"
    return "not_conv", job.exit_reason[:44]


def setup_errors(root: Path) -> dict[str, str]:
    """folder name -> the exception that stopped it before VASP ever started.

    A structure whose inputs could not be written leaves NOTHING behind in its
    own folder, so scanning the tree alone reports it as `not_started` and the
    real cause -- one missing entry in the magnetism table, say -- stays buried
    in a slurm log.  That happened to 47 structures at once.
    """
    out: dict[str, str] = {}
    # Oldest first, so a structure that was retried later shows its LATEST log.
    for log in sorted((root / "jobs").glob("*.out"), key=lambda f: f.stat().st_mtime):
        try:
            text = log.read_text(errors="ignore")
        except OSError:
            continue
        m = LOG_HEADER.search(text)
        if not m:
            continue
        folder = m.group(1)
        for line in reversed(text.splitlines()):
            # The last line of a Python traceback: `module.Class: message`.
            if re.match(r"^[\w.]+(Error|Exception): ", line):
                out[folder] = line.split(": ", 1)[1][:70]
                break
        else:
            out.pop(folder, None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bucket", nargs="?", choices=BUCKETS,
                    help="list the folders in this bucket instead of counting")
    ap.add_argument("--stage", choices=list(STAGES), help="restrict to one stage")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    root = Path(os.environ.get("CSPFLOW_STORE") or "/projects/mmi/cspflow-shared/store")
    folders = sorted(p for p in (root / "structures").glob("mp-*") if p.is_dir())
    stages = [a.stage] if a.stage else list(STAGES)

    def one(f: Path) -> tuple[Path, dict[str, tuple[str, str]]]:
        return f, {s: classify(f / STAGES[s]) for s in stages}

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(one, folders))

    # IGNORED overrides every failure bucket, because it is not one.  These
    # structures sit too far above the convex hull to change it, so the store
    # does not wait for them -- see <store>/ignored.json.  Counting them as
    # crashes or max_steps is what made 67 non-problems look like problems.
    ignored = {}
    try:
        doc = json.loads((root / "ignored.json").read_text())
        for mid, rec in doc.get("ignored", {}).items():
            ignored[rec["folder"]] = rec.get("mp_e_above_hull")
    except (OSError, ValueError):
        pass

    # Overlay what the structure folder cannot know.
    setup = setup_errors(root)
    for f, per_stage in results:
        why = setup.get(f.name)
        if why:
            for st, (bucket, _) in per_stage.items():
                if bucket == "not_started":
                    per_stage[st] = ("setup_error", why)
        if f.name in ignored:
            eah = ignored[f.name]
            for st, (bucket, _) in per_stage.items():
                if bucket != "ok":
                    per_stage[st] = ("ignored",
                                     f"MP e_above_hull {eah} -- cannot affect the hull")

    if a.bucket:
        for f, per_stage in results:
            for s, (bucket, detail) in per_stage.items():
                if bucket == a.bucket:
                    print(f"{f.name:<34} {s:<7} {detail}")
        n = sum(1 for _, ps in results for b, _ in ps.values() if b == a.bucket)
        print(f"\n{n} run(s) in bucket {a.bucket!r}")
        return

    counts = {s: Counter() for s in stages}
    details = defaultdict(Counter)
    for _, per_stage in results:
        for s, (bucket, detail) in per_stage.items():
            counts[s][bucket] += 1
            if detail and bucket in ("crash", "no_outcar", "zbrent", "setup_error"):
                details[bucket][detail] += 1

    print(f"store {root}   {len(folders)} structures\n")
    w = {b: max(len(b), 5) + 2 for b in BUCKETS}
    head = f"{'stage':<8}" + "".join(f"{b:>{w[b]}}" for b in BUCKETS)
    print(head)
    print("-" * len(head))
    for s in stages:
        print(f"{s:<8}" + "".join(
            f"{counts[s][b] or '.':>{w[b]}}" for b in BUCKETS))

    # A structure is blocked if EITHER stage is not ok.
    blocked = sum(1 for _, ps in results if any(b != "ok" for b, _ in ps.values()))
    live = sum(1 for _, ps in results if any(b == "running" for b, _ in ps.values()))
    print(f"\n{len(folders) - blocked} structure(s) complete, "
          f"{blocked} not -- of which {live} running right now")

    for bucket, c in details.items():
        if c:
            print(f"\nwhat '{bucket}' actually said:")
            for k, v in c.most_common(8):
                print(f"  {v:>5}  {k}")


if __name__ == "__main__":
    main()
