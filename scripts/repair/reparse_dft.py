#!/usr/bin/env python
"""
reparse_dft.py -- re-read every DFT run directory and correct what the database
                  says about it, when the PARSER has changed and the runs have
                  not.

WHY THIS EXISTS
    `relaxation` rows are written once, at reconcile time, from whatever the
    parser believed then. Fix a defect in the parser and every row written
    before the fix still carries the old verdict -- the calculations on disk are
    fine, the record of them is not, and nothing re-reads a finished run.

    D151 is the case this was written for. `read_job_directory` judged a static
    step converged if VASP had merely written its epilogue, so every static in
    RE-magnets-CHGNet was recorded converged: 2,245 of 2,245 rows. 155 of the
    runs on disk had actually stopped at NELM with the SCF unconverged. The
    parser now tests electronic convergence; this makes the table agree.

WHAT IT DOES
    For every structure, for every step directory of its run:
      1. re-parses the directory with the CURRENT parser
      2. compares against the newest `relaxation` row for that (structure, engine)
      3. compares each step against the step whose geometry it inherited, and
         flags the pair when they do not describe the same calculation (D152)
      4. reports every disagreement, grouped
      5. with --apply, writes the corrected `converged`/`n_steps` and the flag

    It corrects the NEWEST row per (structure, engine) only. Older rows belong
    to attempts now archived under `attempt-N/`, and there is no reliable
    mapping from a row to the attempt that produced it -- guessing one would
    rewrite history rather than correct it.

    ENERGIES ARE REPORTED, NEVER WRITTEN. A changed energy does not mean the
    parser changed; it means the directory did, and that needs a person rather
    than a flag.

VALIDATION
    Every record is recomputed, not just the ones expected to move, and the run
    that does not change is the evidence. The summary prints unchanged and
    changed counts side by side: a fix that quietly reclassifies rows it was not
    aimed at shows up as an unchanged count that is smaller than it should be.

REFUSES
    while a driver for this campaign is in the queue -- the driver is the only
    writer (D056).

SKIPS
    every structure not in `dft_done` or `failed`. A structure the driver still
    owns is being written as this reads it, so its directory is ahead of its row
    by design; correcting the row to match a half-written OUTCAR would record a
    verdict on a run that has not finished. The first dry run reported 12 energy
    disagreements and one relax flipping the wrong way -- all of them live jobs.

INPUTS
    -c, --campaign PATH   campaign.yaml, or the folder holding it   (required)
    --apply               actually write; without it this is a dry run
    --force               skip the running-driver check (you have checked)

OUTPUTS
    stdout   unchanged/changed tallies, then every disagreement with its cause
    exit 0   on success or a clean dry run; 1 on a refusal

RUN
    conda activate cspflow
    python scripts/repair/reparse_dft.py -c campaigns/RE-magnets-CHGNet
    python scripts/repair/reparse_dft.py -c campaigns/RE-magnets-CHGNet --apply
"""
from __future__ import annotations

import argparse
import getpass
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from cspflow.config.loader import load_campaign
from cspflow.db.store import Store
from cspflow.dft.layout import Layout, resolve as resolve_layout, structure_id_of
from cspflow.dft.recipe import load_recipe
from cspflow.dft.vasp.parse import (ENERGY_SHIFT_MEV_PER_ATOM,
                                    read_job_directory, step_consistency)


def driver_in_queue(name: str) -> list[str]:
    try:
        out = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%A|%j"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.stdout.splitlines():
        jid, _, jname = line.partition("|")
        if jname.strip().startswith(f"{name}-d"):
            found.append(jid.strip())
    return found


def step_dirs(layout: Layout, dft_root: Path, sid: int, steps: list[str]):
    """(step name, directory) for one structure, under either layout."""
    if layout.name == "runs":
        runs = dft_root / "runs"
        if not runs.is_dir():
            return []
        for d in runs.iterdir():
            if d.is_dir() and structure_id_of(d.name) == sid:
                return [(s, d / s) for s in steps if (d / s / "OUTCAR").is_file()]
        return []
    return [(s, dft_root / f"dft-{sid}-{s}")
            for s in steps if (dft_root / f"dft-{sid}-{s}" / "OUTCAR").is_file()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="re-read DFT run directories and correct the database",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--campaign", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    path = Path(args.campaign)
    if path.is_dir():
        path = path / "campaign.yaml"
    cfg = load_campaign(path)

    db = Path(cfg.work_dir) / "campaign.db"
    if not db.exists():
        print(f"no database at {db}", file=sys.stderr)
        return 1

    drivers = driver_in_queue(cfg.campaign.name)
    if drivers and not args.force:
        print(f"refusing: driver {', '.join(drivers)} is in the queue for "
              f"{cfg.campaign.name}. The driver is the only writer (D056) -- "
              f"scancel it, run this, then start it again. --force overrides.",
              file=sys.stderr)
        return 1

    dft_root = Path(cfg.work_dir) / "dft"
    layout_name, note = resolve_layout(cfg.campaign.dft.layout, dft_root)
    layout = Layout(layout_name, dft_root)
    if note:
        print(f"layout: {note}\n")
    steps = [st.name for st in
             load_recipe(cfg.campaign.dft.recipe, cfg.base_dir).stages]

    store = Store.open(db)
    try:
        # newest relaxation row per (structure_id, engine)
        newest: dict[tuple[int, str], dict] = {}
        for r in store.sql.execute(
                "SELECT id, structure_id, engine, energy, e_per_atom, converged, "
                "n_steps FROM relaxation WHERE engine LIKE 'vasp:%' ORDER BY id"):
            newest[(r["structure_id"], r["engine"])] = dict(r)

        unchanged, changed = Counter(), Counter()
        fixes: list[tuple[int, str, str, dict, object]] = []
        energy_drift: list[tuple[int, str, float, float]] = []
        no_row = Counter()

        # A structure the driver still owns is being WRITTEN as this reads it.
        # Its directory is ahead of its row by design, and "correcting" the row
        # to match a half-finished OUTCAR would record a verdict on a run that
        # has not happened yet. The first dry run of this script reported 12
        # energy disagreements and one relax flipping False -> True; every one
        # of them was a live job, not a parser fix.
        in_flight = []
        for sid in sorted({k[0] for k in newest}):
            row = next(store.structures(id=sid), None)
            state = row.key_value_pairs.get("state") if row else None
            if state not in ("dft_done", "failed"):
                in_flight.append((sid, state))
        skip = {sid for sid, _ in in_flight}

        flags: list[tuple[int, str, object]] = []
        for sid in sorted({k[0] for k in newest} - skip):
            dirs = step_dirs(layout, dft_root, sid, steps)

            # D152: does each step agree with the one before it?
            previous = None
            for step, directory in dirs:
                outcome = read_job_directory(directory)
                if previous is not None:
                    check = step_consistency(previous, outcome)
                    if check.measurable and not check.ok:
                        flags.append((sid, step, check))
                previous = outcome

            for step, directory in dirs:
                row = newest.get((sid, f"vasp:{step}"))
                if row is None:
                    no_row[step] += 1
                    continue
                outcome = read_job_directory(directory)
                key = f"{step}: converged {bool(row['converged'])} -> {outcome.converged}"
                if bool(row["converged"]) != bool(outcome.converged):
                    changed[key] += 1
                    fixes.append((sid, step, outcome.exit_reason or "-", row, outcome))
                else:
                    unchanged[key] += 1
                if (row["energy"] is not None and outcome.energy is not None
                        and abs(row["energy"] - outcome.energy) > 1e-6):
                    energy_drift.append((sid, step, row["energy"], outcome.energy))

        if in_flight:
            from collections import Counter as _C
            print(f"SKIPPED: {len(in_flight)} structure(s) the driver still owns "
                  f"-- their directories are being written now. {dict(_C(s for _, s in in_flight))}")
            print(f"  {sorted(sid for sid, _ in in_flight)[:12]}\n")

        print("UNCHANGED  (the evidence that the change was surgical)")
        for k, n in sorted(unchanged.items()):
            print(f"  {n:>6}  {k}")
        print("\nCHANGED")
        if not changed:
            print("        0  nothing to correct")
        for k, n in sorted(changed.items()):
            print(f"  {n:>6}  {k}")
        if no_row:
            print(f"\nrun directories with no matching relaxation row: {dict(no_row)}")

        if fixes:
            by_reason = defaultdict(list)
            for sid, step, reason, _, _ in fixes:
                by_reason[f"{step}: {reason}"].append(sid)
            print("\nwhy each one changed:")
            for reason, sids in sorted(by_reason.items()):
                print(f"  {len(sids):>6}  {reason}")
                print(f"          e.g. structures {sorted(sids)[:6]}")

        if energy_drift:
            print(f"\nENERGY DISAGREEMENTS ({len(energy_drift)}) -- reported, never "
                  f"written. The directory changed, not the parser; this needs a "
                  f"person.")
            for sid, step, was, now in energy_drift[:10]:
                print(f"  structure {sid:>6} {step:<7} recorded {was:.6f}  "
                      f"on disk {now:.6f}")

        if flags:
            print(f"\nSTEPS THAT DISAGREE ({len(flags)}) -- neither step failed, "
                  f"but the geometry and the energy come from different "
                  f"calculations (D152):")
            for sid, step, ch in sorted(flags, key=lambda f: -abs(f[2].energy_shift))[:12]:
                mag = f"{ch.magmom_shift:.2f}" if ch.magmom_shift is not None else "?"
                print(f"  structure {sid:>6} {step:<7} "
                      f"{ch.energy_shift:+9.1f} meV/atom, moment moved {mag} uB/atom")
            if len(flags) > 12:
                print(f"  ... and {len(flags) - 12} more")

        if not args.apply:
            print("\ndry run -- pass --apply to write")
            return 0

        for sid, step, _, row, outcome in fixes:
            store.sql.execute(
                "UPDATE relaxation SET converged=?, n_steps=? WHERE id=?",
                (int(outcome.converged), int(outcome.n_ionic_steps), row["id"]))
        for sid, step, ch in flags:
            store.add_filter_event(
                structure_id=sid, gate=f"dft:{step}:consistent_with_previous",
                passed=False, value=ch.energy_shift,
                threshold=ENERGY_SHIFT_MEV_PER_ATOM, detail=ch.detail[:400])
            kv = {f"dft_{step}_shift_mev": ch.energy_shift,
                  "dft_warning": f"{step}: {ch.detail}"[:200]}
            if ch.magmom_shift is not None:
                kv[f"dft_{step}_magmom_shift"] = ch.magmom_shift
            store.update_structure(sid, **kv)
        store.sql.commit()
        print(f"\ncorrected {len(fixes)} row(s) and flagged {len(flags)} "
              f"step pair(s). Energies untouched.")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
