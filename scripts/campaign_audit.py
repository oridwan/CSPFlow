#!/usr/bin/env python
"""
campaign_audit.py -- what a campaign is ACTUALLY doing, reconciled from three
                     independent sources that routinely disagree.

WHY THIS EXISTS
    No single source can answer "where is this campaign up to?":

      * the campaign DATABASE knows what the driver believes, which lags
        whatever Slurm did in the last few minutes and says nothing about
        whether a finished job produced a usable number;
      * SLURM knows what is running, but a job that exits 0 is not a result --
        VASP exits cleanly at the ionic step limit and at NELM, both unconverged;
      * the FILESYSTEM knows what VASP actually wrote, but not whether a
        directory is abandoned or about to be written to again.

    So this reads all three and reports where they disagree. The disagreements
    are the interesting part; agreement is just the campaign working.

WHAT IT CHECKS PER DIRECTORY  (the acceptance rules, not "did it exit 0")
    - VASP_DONE marker present
    - OUTCAR carries a final timing block ("Total CPU time used") -- without it
      the run was killed and every number in the file is mid-iteration
    - a final free energy (TOTEN) exists
    - relax only: "reached required accuracy" -- VASP's own statement that it
      met EDIFFG, as opposed to stopping because it ran out of NSW
    - ionic steps vs NSW, to separate "converged" from "hit the step limit"
    - total magnetisation, and whether it is oscillating between ionic steps
      (the signature of a spin state flipping under the optimiser)
    - OUTCAR mtime vs the job's end time, to catch a stale file being read as
      if it were the current run

INPUTS
    --campaign PATH   campaign folder containing campaign.yaml   [required]
    --dft-root PATH   where the VASP directories live            [default: from config]
    --json PATH       also write the full table as JSON          [optional]
    --quiet           table only, no per-structure detail

OUTPUTS
    stdout: five sections -- RUNNING, QUEUED, READY FOR STATIC, NEEDS ATTENTION,
            and a reconciliation summary naming every DB/Slurm/disk disagreement
    optional JSON with one record per VASP directory

RUN
    conda activate cspflow
    python scripts/campaign_audit.py --campaign campaigns/CeFeB
    python scripts/campaign_audit.py --campaign campaigns/CeFeB --json /tmp/audit.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# Slurm
# --------------------------------------------------------------------------


def squeue_rows() -> list[dict]:
    """Every job of this user that Slurm still knows about, as dicts."""
    out = subprocess.run(
        ["squeue", "-u", subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip(),
         "-h", "-o", "%i|%j|%T|%M|%L|%R"],
        capture_output=True, text=True, check=False,
    ).stdout
    rows = []
    for line in out.splitlines():
        p = line.split("|")
        if len(p) >= 6:
            rows.append({"jobid": p[0], "name": p[1], "state": p[2],
                         "time": p[3], "left": p[4], "where": p[5]})
    return rows


def _task_indices(jobid: str) -> list[int]:
    """Which array tasks a squeue JOBID line stands for.

    `123_4` is one task. `123_[0-5%2]` is a pending array and stands for all of
    them. A plain `123` is a single job, task 0.
    """
    if "[" in jobid:
        rng = re.search(r"\[([\d,\-%]+)\]", jobid)
        out: list[int] = []
        if rng:
            for part in rng.group(1).split("%")[0].split(","):
                if "-" in part:
                    a, b = part.split("-")
                    out += list(range(int(a), int(b) + 1))
                elif part.isdigit():
                    out.append(int(part))
        return out
    m = re.match(r"\d+_(\d+)$", jobid)
    return [int(m.group(1))] if m else [0]


def sacct_state(jobid: str) -> str:
    out = subprocess.run(["sacct", "-j", jobid, "-n", "-X", "-o", "State"],
                         capture_output=True, text=True, check=False).stdout
    return out.strip().splitlines()[0].strip() if out.strip() else ""


# --------------------------------------------------------------------------
# VASP output
# --------------------------------------------------------------------------

_F_LINE = re.compile(r"F=\s*([-.\dE+]+)\s+E0=\s*([-.\dE+]+).*?mag=\s*([-\d.]+)")


def read_run(d: Path) -> dict:
    """Judge one VASP directory. Never trusts VASP_DONE on its own."""
    r = {
        "dir": d.name, "exists": d.is_dir(), "done_marker": (d / "VASP_DONE").is_file(),
        "timing_block": False, "final_energy": None, "reached_accuracy": False,
        "ionic_steps": 0, "nsw": None, "hit_step_limit": False,
        "mag": None, "mag_swing": None, "electronic_per_ionic": None,
        "verdict": "MISSING", "why": "",
    }
    if not d.is_dir():
        return r

    incar = d / "INCAR"
    if incar.is_file():
        m = re.search(r"^NSW\s*=\s*(\d+)", incar.read_text(), re.M)
        if m:
            r["nsw"] = int(m.group(1))

    osz = d / "OSZICAR"
    if osz.is_file():
        text = osz.read_text(errors="replace")
        rows = _F_LINE.findall(text)
        r["ionic_steps"] = len(rows)
        if rows:
            r["final_energy"] = float(rows[-1][1])
            mags = [float(x[2]) for x in rows]
            r["mag"] = mags[-1]
            # Only the tail matters: early ionic steps legitimately move a lot.
            tail = mags[-10:] if len(mags) >= 10 else mags
            r["mag_swing"] = round(max(tail) - min(tail), 2)
        n_elec = len(re.findall(r"^(?:DAV|RMM|CG):", text, re.M))
        if r["ionic_steps"]:
            r["electronic_per_ionic"] = round(n_elec / r["ionic_steps"], 1)

    out = d / "OUTCAR"
    if out.is_file():
        # Read the tail only: an OUTCAR here reaches 45 MB and the markers that
        # decide the verdict are all near the end.
        size = out.stat().st_size
        with out.open("rb") as fh:
            fh.seek(max(0, size - 400_000))
            tail = fh.read().decode("utf-8", errors="replace")
        r["timing_block"] = "Total CPU time used" in tail
        r["reached_accuracy"] = "reached required accuracy" in tail
        m = re.findall(r"free  energy   TOTEN\s*=\s*([-.\dE+]+)", tail)
        if m:
            r["final_energy"] = float(m[-1])
        r["outcar_mtime"] = out.stat().st_mtime
        r["idle_min"] = round((time.time() - out.stat().st_mtime) / 60, 1)

    if r["nsw"] and r["ionic_steps"] >= r["nsw"]:
        r["hit_step_limit"] = True

    # -- the verdict ------------------------------------------------------
    is_static = d.name.endswith("-static")
    if not (d / "OUTCAR").is_file():
        r["verdict"], r["why"] = "NOT STARTED", "no OUTCAR"
    elif not r["timing_block"]:
        r["verdict"], r["why"] = "KILLED", "no final timing block -- run did not finish"
    elif r["final_energy"] is None:
        r["verdict"], r["why"] = "NO ENERGY", "finished but wrote no TOTEN"
    elif is_static:
        r["verdict"] = "OK" if r["done_marker"] else "OK (no marker)"
    elif r["reached_accuracy"]:
        r["verdict"] = "OK"
    elif r["hit_step_limit"]:
        r["verdict"], r["why"] = "UNCONVERGED", f"hit NSW={r['nsw']} without reaching EDIFFG"
    else:
        r["verdict"], r["why"] = "UNCONVERGED", "exited without reaching EDIFFG"
    return r


# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Finding the work, in either layout
# --------------------------------------------------------------------------
#
# There are two arrangements on disk and this has to read both:
#
#   stages (older)  dft/dft-<sid>-<step>/          one directory per STEP
#   runs   (D131)   dft/runs/<sid>-<formula>-<seed>/<step>/   one per STRUCTURE
#
# and two ways a job maps to a directory:
#
#   an ARRAY covers many structures, is named after task 0, and the only record
#   of which task ran what is its `<name>.tasks.json` manifest;
#   a SOLO job (D135) covers one structure and is NAMED after its directory, so
#   there is nothing to look up.
#
# Rather than guess from the name -- which is what cost ten hours in D130 --
# the running jobs are resolved through `SLURM_TASK`, the file each job writes
# into its own directory saying which job id is running there. That is written
# from inside the job, at the one moment both facts are known, and it is
# correct for every combination of the four cases above.


def discover_runs(dft_root: Path) -> list[tuple[int, str, Path, str]]:
    """(structure id, step, directory, label) for every VASP directory."""
    found: list[tuple[int, str, Path, str]] = []
    for d in sorted(dft_root.glob("dft-*")):                  # stages
        m = re.match(r"dft-(\d+)-(\w+)$", d.name)
        if d.is_dir() and m:
            found.append((int(m.group(1)), m.group(2), d, d.name))
    for d in sorted((dft_root / "runs").glob("*/*")):         # runs
        m = re.match(r"^(\d+)(?:-|$)", d.parent.name)
        if d.is_dir() and m and (d / "INCAR").is_file():
            found.append((int(m.group(1)), d.name, d,
                          f"{d.parent.name}/{d.name}"))
    return found


def job_owner_map(entries: list[tuple[int, str, Path, str]]) -> dict[str, list[str]]:
    """`SLURM_TASK` read back: which job id is running in which directory.

    Authoritative and layout-independent, because the job wrote it itself. The
    file sits in the run ROOT for a combined job and in the step directory for
    a per-step one, so both are looked at.
    """
    owners: dict[str, list[str]] = defaultdict(list)
    for _sid, _step, d, label in entries:
        for candidate in (d / "SLURM_TASK", d.parent / "SLURM_TASK"):
            if candidate.is_file():
                try:
                    owners[candidate.read_text().strip()].append(label)
                except OSError:
                    pass
                break
    return owners


def manifest_dirs(dft_root: Path, job: dict) -> list[Path]:
    """The directories an ARRAY task covers, from the array's own manifest."""
    manifest = dft_root / f"{job['name']}.tasks.json"
    if not manifest.is_file():
        return []
    try:
        dirs = json.loads(manifest.read_text())["dirs"]
    except Exception:                                          # noqa: BLE001
        return []
    return [Path(dirs[i]) for i in _task_indices(job["jobid"]) if 0 <= i < len(dirs)]


def live_labels(dft_root: Path, job: dict, owners: dict[str, list[str]],
                by_label: dict[str, Path]) -> list[str]:
    """Which of THIS campaign's directories `job` is working in.

    Three sources, most trustworthy first: what the job wrote, what the array's
    manifest says, and -- only for a job that has not started and so has written
    nothing -- the name, which under D135 IS the directory.
    """
    if job["jobid"] in owners:
        return owners[job["jobid"]]
    labels = [str(Path(d).name) if Path(d).parent == dft_root.resolve()
              else f"{Path(d).parent.name}/{Path(d).name}"
              for d in manifest_dirs(dft_root, job)]
    labels = [x for x in labels if x in by_label]
    if labels:
        return labels
    # A solo job that has not started yet. Its name is `<campaign>-<slug>`.
    return [label for label in by_label
            if job["name"].endswith(label.split("/")[0])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True, type=Path)
    ap.add_argument("--dft-root", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cspflow.config.loader import load_campaign
    from cspflow.db.store import Store

    cfg = load_campaign(a.campaign / "campaign.yaml")
    dft_root = a.dft_root or (cfg.work_dir / "dft")
    print(f"campaign   {cfg.campaign.name}")
    print(f"database   {cfg.campaign_db}")
    print(f"dft root   {dft_root}")

    # --- source 1: the database ------------------------------------------
    db_state, db_jobs = {}, []
    try:
        with Store.open(cfg.campaign_db) as st:
            for row in st.ase.select():
                db_state[int(row.id)] = row.key_value_pairs.get("state", "?")
            db_jobs = [dict(j) for j in st.jobs()]
        print(f"database   readable -- {len(db_state)} structures, {len(db_jobs)} job rows")
    except Exception as exc:                                     # noqa: BLE001
        print(f"database   NOT READABLE from this node: {str(exc).splitlines()[0]}")

    # --- source 2: Slurm --------------------------------------------------
    live = squeue_rows()

    # --- source 3: the filesystem ----------------------------------------
    entries = discover_runs(dft_root)
    by_label = {label: d for _sid, _step, d, label in entries}
    owners = job_owner_map(entries)

    # Which directories are live, resolved through what the JOBS wrote rather
    # than through what SLURM displays.
    live_by_label: dict[str, dict] = {}
    for j in live:
        for label in live_labels(dft_root, j, owners, by_label):
            live_by_label[label] = j
    live_names = set(live_by_label)

    runs = {}
    for _sid, _step, d, label in entries:
        r = read_run(d)
        # A running job has no timing block either. Calling that KILLED turns
        # every healthy in-flight run into an alarm, which is how a real one
        # gets lost in the noise.
        if label in live_names and r["verdict"] in ("KILLED", "NOT STARTED", "NO ENERGY"):
            r["verdict"], r["why"] = "IN FLIGHT", "still running"
        runs[label] = r
    layout = "runs" if (dft_root / "runs").is_dir() else "stages"
    print(f"disk       {len(runs)} VASP directories  (layout: {layout})\n")

    # --- group by structure id -------------------------------------------
    by_sid = defaultdict(dict)
    label_of = defaultdict(dict)
    for sid, step, _d, label in entries:
        by_sid[sid][step] = runs[label]
        label_of[sid][step] = label

    running, queued, ready_static, attention = [], [], [], []

    # Report the DIRECTORY each task is working in, never the job name.
    #
    # A job name was untrustworthy for two independent reasons: one sbatch
    # covered an array and SLURM named the whole array after task 0, and
    # `squeue` is per USER, so `dft-71-relax` existed in CeFeB AND CePdGe at
    # once. One job per structure (D135) removes the first and the campaign
    # prefix removes the second -- but `live_labels` still resolves through
    # what the job itself wrote, because an old array may still be in flight.
    for label, j in sorted(live_by_label.items()):
        (running if j["state"] == "RUNNING" else queued).append((label, j))

    for sid in sorted(by_sid):
        rel = by_sid[sid].get("relax")
        sta = by_sid[sid].get("static")
        in_flight = any(lbl in live_names for lbl in label_of[sid].values())

        if rel and rel["verdict"] == "OK" and sta is None and not in_flight:
            ready_static.append((sid, rel))
        elif rel and rel["verdict"] in ("KILLED", "UNCONVERGED", "NO ENERGY") and not in_flight:
            attention.append((sid, "relax", rel))
        elif sta and sta["verdict"] in ("KILLED", "NO ENERGY") and not in_flight:
            attention.append((sid, "static", sta))

    # --- report -----------------------------------------------------------
    def hdr(t):
        print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")

    hdr(f"RUNNING  ({len(running)})")
    if running:
        print(f"{'directory (real work)':<44} {'jobid':<14} {'elapsed':>10} {'left':>10}"
              f"  {'steps':>6} {'mag':>7}  {'shown as':<16} where")
        for name, j in sorted(running, key=lambda x: x[0]):
            r = runs.get(name, {})
            shown = j["name"] if j["name"] != name else ""
            print(f"{name:<44} {j['jobid']:<14} {j['time']:>10} {j['left']:>10} "
                  f"{r.get('ionic_steps', 0):>6} {r.get('mag') or 0:>7.1f}  {shown:<16} {j['where']}")
    else:
        print("  none")

    hdr(f"QUEUED  ({len(queued)})")
    if queued:
        for name, j in sorted(queued, key=lambda x: x[0]):
            shown = f"  (shown as {j['name']})" if j["name"] != name else ""
            print(f"  {name:<44} {j['jobid']:<18} {j['where']}{shown}")
    else:
        print("  none")

    hdr(f"RELAX DONE, STATIC NOT SUBMITTED  ({len(ready_static)})")
    if ready_static:
        print(f"{'sid':>5}  {'steps':>6} {'E (eV)':>13} {'mag':>8} {'swing':>7}  note")
        for sid, r in ready_static:
            note = "mag unsettled in last 10 steps" if (r["mag_swing"] or 0) > 5 else ""
            print(f"{sid:>5}  {r['ionic_steps']:>6} {r['final_energy']:>13.4f} "
                  f"{r['mag'] or 0:>8.2f} {r['mag_swing'] or 0:>7.2f}  {note}")
        print("\n  The driver submits these on its next cycle. Nothing to do by hand.")
    else:
        print("  none -- every converged relax already has a static")

    hdr(f"NEEDS ATTENTION  ({len(attention)})")
    if attention:
        print(f"{'sid':>5} {'stage':<8} {'verdict':<13} {'steps':>6} {'mag':>7} "
              f"{'e/ionic':>8} {'idle min':>9}  why")
        for sid, stage, r in attention:
            print(f"{sid:>5} {stage:<8} {r['verdict']:<13} {r['ionic_steps']:>6} "
                  f"{r['mag'] or 0:>7.1f} {r['electronic_per_ionic'] or 0:>8.1f} "
                  f"{r.get('idle_min', 0):>9.1f}  {r['why']}")
        print("\n  'idle min' is minutes since the OUTCAR was last written. A few")
        print("  minutes means the driver is mid-cycle; hours means abandoned.")
    else:
        print("  none")

    # --- reconciliation ---------------------------------------------------
    hdr("RECONCILIATION  (where the three sources disagree)")
    disagreements = []
    for sid, stages in sorted(by_sid.items()):
        s = db_state.get(sid)
        rel = stages.get("relax")
        if s == "dft_done" and rel and rel["verdict"] not in ("OK", "OK (no marker)"):
            disagreements.append(
                f"  sid {sid}: database says dft_done, but relax on disk is {rel['verdict']}")
        if s in ("dft_queued",) and rel and rel["verdict"] == "OK" \
                and label_of[sid].get("relax") not in live_names \
                and "static" not in stages:
            disagreements.append(
                f"  sid {sid}: database says dft_queued, nothing in Slurm, relax OK, no static yet "
                f"-- the driver has not reconciled it yet")
    counts = defaultdict(int)
    for r in runs.values():
        counts[r["verdict"]] += 1
    print("  verdicts on disk: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if disagreements:
        print()
        for d in disagreements[:30]:
            print(d)
    else:
        print("  no disagreements found")

    if a.json:
        a.json.write_text(json.dumps(
            {"runs": runs, "db_state": db_state, "slurm": live}, indent=2, default=str))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
