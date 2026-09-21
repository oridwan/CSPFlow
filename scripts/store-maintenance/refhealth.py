#!/usr/bin/env python
"""refhealth.py -- is each RUNNING reference DFT job actually making progress?

WHY THIS EXISTS
    refcheck.py classifies jobs that have STOPPED.  It cannot tell a job that is
    converging steadily from one that has been alive for 14 hours and moved
    nothing -- both look like "running".  This reads the live OSZICAR/OUTCAR of
    every running array task and answers the only question that matters while
    the clock is burning: will this finish, and if not, why not.

WHAT IT MEASURES, per running task
    stage        which of dft_relax / dft_static is currently being written
    step/NSW     ionic steps done vs the recipe's limit
    fmax         largest force on any atom, against EDIFFG
    dE           energy change on the last ionic step
    scf          electronic steps used by the last ionic step, vs NELM
    age          seconds since OUTCAR was last written -- the liveness test
    proj         ionic steps projected by the SLURM walltime, at the observed rate

VERDICTS
    ok            progressing, and projected to reach NSW inside the walltime
    slow          alive, but will hit the SLURM walltime before NSW
    nsw-limit     will reach NSW without meeting EDIFFG -> unconverged geometry
    scf-hard      last ionic step needed >NELM/2 electronic steps
    stalled       OUTCAR not written for >20 min: suspect a hung rank
    no-outcar     dispatched but nothing written yet (normal for a few minutes)

INPUT   $CSPFLOW_STORE (default /projects/mmi/cspflow-shared/store)
            jobs/*.tasks.jobid   array id -> task list, to map index to folder
            structures/<folder>/dft_{relax,static}/{OSZICAR,OUTCAR,INCAR}
        live `squeue` for the running array indices and their time limits
OUTPUT  a table on stdout; exit 0 always (this is a report, not a gate)

RUN     python scripts/store-maintenance/refhealth.py
        python scripts/store-maintenance/refhealth.py --job 26902766     # one array only
        python scripts/store-maintenance/refhealth.py -v                 # add the INCAR context
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STALL_SECONDS = 20 * 60          # OUTCAR silent this long -> suspect a hang
FORCE_TAIL_BYTES = 3_000_000     # enough for the last few TOTAL-FORCE blocks


def store_root() -> Path:
    return Path(os.environ.get("CSPFLOW_STORE", "/projects/mmi/cspflow-shared/store"))


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""


def running_tasks(only_job: str | None) -> list[dict]:
    """Every RUNNING ref-dft array task, with its elapsed and remaining time."""
    # %F, not %A.  For a running array element %A is that ELEMENT's own job id
    # (26903178), which matches no task file; %F is the array's parent id
    # (26902766), which is what jobs/*.tasks.jobid records.
    out = sh(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", "ref-dft",
              "-t", "R", "-r", "-o", "%F|%A|%K|%M|%L|%N|%P"])
    rows = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 7:
            continue
        arr, elem, idx, elapsed, left, node, part = parts
        if only_job and arr != only_job:
            continue
        rows.append({"job": arr, "elem": elem, "index": idx, "elapsed": elapsed,
                     "left": left, "node": node, "partition": part})
    return rows


def task_lists(root: Path) -> dict[str, Path]:
    """array job id -> the task file it is reading, from jobs/*.tasks.jobid."""
    out = {}
    for jf in (root / "jobs").glob("*.tasks.jobid"):
        try:
            out[jf.read_text().strip()] = jf.with_suffix("")   # drop .jobid
        except OSError:
            pass
    return out


def folder_for(tasks: Path, index: str) -> Path | None:
    try:
        lines = tasks.read_text().splitlines()
        return Path(lines[int(index)].strip())
    except (OSError, ValueError, IndexError):
        return None


def _hms(text: str) -> float:
    """SLURM elapsed/remaining ('1-09:52:38', '13:17:31', '5:59') -> seconds."""
    text = text.strip()
    if not text or text in ("UNLIMITED", "NOT_SET", "INVALID"):
        return 0.0
    days = 0
    if "-" in text:
        d, _, text = text.partition("-")
        days = int(d)
    bits = [float(b) for b in text.split(":")]
    while len(bits) < 3:
        bits.insert(0, 0.0)
    return days * 86400 + bits[0] * 3600 + bits[1] * 60 + bits[2]


def incar_tags(path: Path) -> dict[str, str]:
    tags = {}
    try:
        for line in path.read_text().splitlines():
            key, _, val = line.partition("=")
            if val:
                tags[key.strip().upper()] = val.split("#")[0].strip()
    except OSError:
        pass
    return tags


def oszicar_progress(path: Path) -> dict:
    """Ionic steps, last dE and mag, and electronic steps used per step.

    One pass, because on a long relax OSZICAR reaches a few hundred kB and the
    caller does this for every running task at once.
    """
    steps, scf, mag, de, energy = 0, [], None, None, None
    n = 0
    try:
        with path.open() as fh:
            for line in fh:
                if line.startswith(("RMM:", "DAV:")):
                    n += 1
                elif " F= " in line:
                    steps += 1
                    scf.append(n)
                    n = 0
                    m = re.search(r"F=\s*([-.\d E+]+?)\s+E0=", line)
                    if m:
                        try:
                            energy = float(m.group(1).replace(" ", ""))
                        except ValueError:
                            pass
                    m = re.search(r"d E =\s*([-\d.E+]+)", line)
                    if m:
                        de = float(m.group(1))
                    m = re.search(r"mag=\s*([-\d.]+)", line)
                    if m:
                        mag = float(m.group(1))
    except OSError:
        pass
    return {"steps": steps, "scf": scf, "de": de, "mag": mag, "energy": energy}


def max_force(path: Path) -> float | None:
    """Largest |force| component on any atom, from the last TOTAL-FORCE block.

    VASP only writes the 'FORCES: max atom, RMS' summary at higher NWRITE, so
    the block itself has to be parsed.  Only the tail is read: the OUTCAR of a
    100-atom relax is ~18 MB and the last block is all that is wanted.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > FORCE_TAIL_BYTES:
                fh.seek(size - FORCE_TAIL_BYTES)
            chunk = fh.read().decode("utf-8", "replace")
    except OSError:
        return None

    best = None
    in_block = False
    current = 0.0
    seen = 0
    for line in chunk.splitlines():
        if "TOTAL-FORCE" in line:
            in_block, current, seen = True, 0.0, 0
            continue
        if not in_block:
            continue
        if line.startswith(" ---") or not line.strip():
            continue
        cols = line.split()
        if len(cols) == 6:
            try:
                f = max(abs(float(c)) for c in cols[3:6])
            except ValueError:
                in_block = False
                continue
            current = max(current, f)
            seen += 1
        else:
            if seen:
                best = current
            in_block = False
    if in_block and seen:
        best = current
    return best


def inspect(row: dict, root: Path, tasks: Path) -> dict:
    folder = folder_for(tasks, row["index"])
    rec = dict(row, folder="?", stage="-", verdict="no-task-line")
    if folder is None:
        return rec
    rec["folder"] = folder.name

    # The stage being written now is the one whose OUTCAR is newest.
    stage_dir, newest = None, -1.0
    for name in ("dft_relax", "dft_static"):
        out = folder / name / "OUTCAR"
        if out.exists():
            m = out.stat().st_mtime
            if m > newest:
                stage_dir, newest = folder / name, m
    if stage_dir is None:
        rec["verdict"] = "no-outcar"
        return rec

    rec["stage"] = stage_dir.name.replace("dft_", "")
    rec["age"] = time.time() - newest

    tags = incar_tags(stage_dir / "INCAR")
    nsw = int(float(tags.get("NSW", 0) or 0))
    nelm = int(float(tags.get("NELM", 60) or 60))
    ediffg = float(tags.get("EDIFFG", -0.01) or -0.01)
    rec.update(nsw=nsw, nelm=nelm, ediffg=ediffg,
               ibrion=tags.get("IBRION", "?"), isif=tags.get("ISIF", "?"),
               ispin=tags.get("ISPIN", "?"), ncore=tags.get("NCORE", "?"))

    prog = oszicar_progress(stage_dir / "OSZICAR")
    rec.update(prog)
    rec["fmax"] = max_force(stage_dir / "OUTCAR")
    rec["last_scf"] = prog["scf"][-1] if prog["scf"] else 0

    try:
        rec["natoms"] = sum(
            int(x) for x in (stage_dir / "POSCAR").read_text().splitlines()[6].split())
    except Exception:
        rec["natoms"] = None

    # Projection: at the rate observed so far, how many ionic steps fit in the
    # time SLURM will still allow?  Only the elapsed time of THIS stage counts,
    # which for a relax is the whole job and for a static is unknown -- so the
    # projection is reported for the relax only, where it is meaningful.
    elapsed = _hms(row["elapsed"])
    left = _hms(row["left"])
    rec["proj"] = None
    if prog["steps"] > 1 and elapsed > 0 and rec["stage"] == "relax":
        per_step = elapsed / prog["steps"]
        rec["per_step_min"] = per_step / 60.0
        rec["proj"] = prog["steps"] + int(left / per_step)

    converged = (stage_dir / "OUTCAR").read_bytes()[-4000:].find(
        b"reached required accuracy") >= 0

    # Did the RELAX that fed this stage actually meet EDIFFG?  VASP writes
    # "General timing and accounting" on any NORMAL exit, NSW exhaustion
    # included, so the job script's VASP_DONE marker cannot tell a converged
    # geometry from one that merely ran out of ionic steps.  A static running
    # on such a cell produces a total energy on an unrelaxed structure --
    # mp-1244911-Fe2O3 hit NSW=99 with fmax 0.182 eV/A and went straight on.
    relax_ok = None
    if rec["stage"] == "static":
        rout = folder / "dft_relax" / "OUTCAR"
        try:
            relax_ok = rout.read_bytes()[-4000:].find(
                b"reached required accuracy") >= 0
        except OSError:
            relax_ok = None

    # EDDRMM/ZHEGV failures mean the subspace diagonalisation broke down; with
    # ALGO=Fast this is recoverable only by switching to blocked Davidson.
    eddrmm = 0
    try:
        vo = stage_dir / "vasp.out"
        if vo.exists():
            eddrmm = vo.read_text(errors="replace").count("WARNING in EDDRMM")
    except OSError:
        pass
    rec["eddrmm"] = eddrmm

    # --- verdict, most serious first --------------------------------------
    if rec["age"] > STALL_SECONDS:
        rec["verdict"] = "stalled"
    elif prog.get("energy") is not None and prog["energy"] > 0:
        # A positive total energy for a periodic solid is not a slow
        # convergence, it is a broken one.  Nothing recovers from here.
        rec["verdict"] = "DIVERGED"
    elif prog.get("de") is not None and prog["de"] > 1.0:
        rec["verdict"] = "DIVERGED"
    elif relax_ok is False:
        rec["verdict"] = "static-on-bad-relax"
    elif (rec["stage"] == "relax" and prog["steps"] == 0
          and elapsed > 3600):
        # Six hours inside the FIRST ionic step means the initial SCF is not
        # converging, not that the cell is large.
        rec["verdict"] = "first-scf"
    elif converged:
        rec["verdict"] = "converged"
    elif rec["last_scf"] >= nelm:
        rec["verdict"] = "scf-hard"
    elif nsw and prog["steps"] >= nsw:
        rec["verdict"] = "nsw-limit"
    elif rec["proj"] is not None and nsw and rec["proj"] < nsw:
        rec["verdict"] = "slow"
    elif rec["last_scf"] > max(nelm // 2, 20):
        rec["verdict"] = "scf-hard"
    elif nsw and rec["proj"] is not None and prog["steps"] > 0.9 * nsw:
        rec["verdict"] = "nsw-limit"
    else:
        rec["verdict"] = "ok"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", help="only this array job id")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print IBRION/ISIF/ISPIN/NCORE per task")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    root = store_root()
    rows = running_tasks(a.job)
    if not rows:
        print("no ref-dft tasks are running")
        return 0
    lists = task_lists(root)

    def work(r):
        tl = lists.get(r["job"])
        if tl is None:
            return dict(r, folder="?", stage="-", verdict="no-task-file")
        return inspect(r, root, tl)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        recs = list(pool.map(work, rows))

    recs.sort(key=lambda r: (r["verdict"] != "ok", -_hms(r["elapsed"])))

    hdr = (f"{'task':<12} {'folder':<24} {'nat':>4} {'stg':<6} {'step/NSW':>9} "
           f"{'fmax':>8} {'dE':>10} {'scf':>7} {'mag':>7} {'age':>5} "
           f"{'elapsed':>9} {'left':>9} {'proj':>5}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        fmax = f"{r['fmax']:.4f}" if r.get("fmax") is not None else "-"
        de = f"{r['de']:.2e}" if r.get("de") is not None else "-"
        mag = f"{r['mag']:.1f}" if r.get("mag") is not None else "-"
        step = (f"{r.get('steps', 0)}/{r.get('nsw', 0)}"
                if r.get("nsw") else str(r.get("steps", 0)))
        scf = (f"{r.get('last_scf', 0)}/{r.get('nelm', 0)}"
               if r.get("nelm") else str(r.get("last_scf", 0)))
        age = f"{int(r.get('age', 0) // 60)}m" if r.get("age") is not None else "-"
        print(f"{r['job'][-5:]}_{r['index']:<5} {r['folder'][:24]:<24} "
              f"{r.get('natoms') or '-':>4} {r['stage']:<6} {step:>9} "
              f"{fmax:>8} {de:>10} {scf:>7} {mag:>7} {age:>5} "
              f"{r['elapsed']:>9} {r['left']:>9} "
              f"{r.get('proj') or '-':>5}  {r['verdict']}")

    print()
    counts = {}
    for r in recs:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if a.verbose:
        print(f"\n{'task':<12} {'IBRION':>6} {'ISIF':>5} {'ISPIN':>6} "
              f"{'NCORE':>6} {'EDIFFG':>8} {'min/step':>9}")
        for r in recs:
            ps = r.get("per_step_min")
            print(f"{r['job'][-5:]}_{r['index']:<5} {r.get('ibrion', '?'):>6} "
                  f"{r.get('isif', '?'):>5} {r.get('ispin', '?'):>6} "
                  f"{r.get('ncore', '?'):>6} {r.get('ediffg', 0):>8} "
                  f"{(f'{ps:.1f}' if ps else '-'):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
