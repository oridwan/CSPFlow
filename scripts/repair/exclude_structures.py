#!/usr/bin/env python
"""
Cancel a structure's running DFT and take it out of the campaign for good.

WHY THIS IS NEEDED
------------------
`scancel` alone is not an exclusion. The driver reconciles the cancellation,
finds no retry rung matching CANCELLED, and marks the structure `failed` with
`dft_fail_reason: relax: CANCELLED`. That does stop resubmission -- `failed` is
outside the set the DFT stage claims from -- but it records the wrong thing: it
says the calculation broke, when in fact an operator decided the structure was
not worth finishing. Six months later nobody can tell those two apart, and
`requeue.py` will happily resurrect it, because its bulk selectors read JOB
state and knows nothing about intent.

So this does three things in one pass, in an order the live driver cannot undo:

  1. `scancel` the EXACT array task (`26932053_15`), never the whole array.
     One sbatch covers many structures (D130); cancelling `26932053` would kill
     every task in it, including work you are keeping.
  2. Marks the job row `cancelled`. `Driver._reconcile` polls only
     queued/running/held, so a terminal row is never handed to `_retry_or_fail`
     and cannot overwrite the structure state set below. Core-hours already
     recorded on the row are preserved -- the compute was really spent.
  3. Sets the structure `filtered_out` with an explicit `filter_reason`, which
     is the state that means "deliberately not carried forward". The DFT stage
     reads only `selected` and `dft_done`, so it is never claimed again.

SAFETY
------
  * Refuses any structure that is already `dft_done` -- finished work is not
    excluded by this route.
  * Cancels only tasks it can address individually, and says so if a job row
    has no array index.
  * Dry run unless you pass --yes.
  * Reversible: set the structure back to `selected` with `dft_step` 0 and
    resubmit. Nothing on disk is deleted.

INPUTS
------
  -c / --campaign PATH   campaign.yaml, or the folder holding it   (required)
  --sid N                structure id to exclude; repeatable       (required)
  --reason TEXT          why, recorded on the structure            (required)
  --yes                  actually write; without it this is a dry run

OUTPUTS
-------
  One line per structure: id, formula, the task cancelled, the new state.
  Exit 1 if the campaign or database cannot be found, or a sid is unknown.

RUN
---
  conda activate cspflow
  python scripts/repair/exclude_structures.py -c campaigns/CePdGe \
      --sid 122 --sid 127 --reason "zero computed moment; not a magnet target"
  # add --yes to write
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cspflow.config.loader import load_campaign
from cspflow.db.store import Store, StructureState

LIVE = {"running", "queued", "pending", "held"}


def task_token(job) -> str | None:
    """How SLURM addresses this structure's work: '26932053_15' or '26940187'.

    A row with an array index is one task of an array, and only the
    `<jobid>_<index>` form cancels just that task -- the bare id would take
    every sibling structure in the array with it.

    A row with NO index is a job of its own (D135: one structure, one job), so
    the bare id addresses exactly this structure and nothing else.
    """
    slurm = str(job["slurm_id"] or "").strip()
    if not slurm:
        return None
    idx = job["array_task_id"]
    return f"{slurm}_{idx}" if idx is not None else slurm


def main() -> int:
    ap = argparse.ArgumentParser(
        description="cancel and permanently exclude structures from a campaign")
    ap.add_argument("-c", "--campaign", required=True)
    ap.add_argument("--sid", type=int, action="append", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--yes", action="store_true", help="write; otherwise dry run")
    args = ap.parse_args()

    path = Path(args.campaign)
    if path.is_dir():
        path = path / "campaign.yaml"
    if not path.exists():
        print(f"no campaign at {path}", file=sys.stderr)
        return 1
    cfg = load_campaign(path)

    db = Path(cfg.work_dir) / "campaign.db"
    if not db.exists():
        print(f"no database at {db}", file=sys.stderr)
        return 1

    store = Store.open(db)
    try:
        jobs_by_sid: dict[int, list] = {}
        for j in store.jobs(stage="dft"):
            if j["structure_id"] is not None:
                jobs_by_sid.setdefault(int(j["structure_id"]), []).append(j)

        plan, refused = [], []
        for sid in sorted(set(args.sid)):
            row = next(store.structures(id=sid), None)
            if row is None:
                refused.append((sid, "?", "no such structure"))
                continue
            state = str(row.key_value_pairs.get("state", "?"))
            if state == StructureState.dft_done.value:
                refused.append((sid, row.toatoms().get_chemical_formula(),
                                "already dft_done -- finished work is not excluded here"))
                continue
            live = [j for j in jobs_by_sid.get(sid, []) if j["state"] in LIVE]
            tokens = []
            for j in live:
                tok = task_token(j)
                if tok is None:
                    refused.append((sid, row.toatoms().get_chemical_formula(),
                                    f"job {j['id']} has no slurm id -- cannot cancel safely"))
                else:
                    tokens.append((j, tok))
            plan.append((sid, row, state, tokens))

        verb = "excluding" if args.yes else "would exclude"
        print(f"{verb} {len(plan)} structure(s)   ({db})")
        print(f"reason: {args.reason}\n")
        print(f"  {'sid':>4}  {'formula':<20} {'from':<12} {'cancel':<16} -> state")
        for sid, row, state, tokens in plan:
            formula = row.toatoms().get_chemical_formula()
            shown = ", ".join(t for _, t in tokens) or "(nothing running)"
            print(f"  {sid:>4}  {formula:<20} {state:<12} {shown:<16} -> filtered_out")
            if not args.yes:
                continue
            for job, tok in tokens:
                subprocess.run(["scancel", tok], check=False)
                # Terminal row: _reconcile polls only queued/running/held, so the
                # driver can no longer hand this to _retry_or_fail and overwrite
                # the state we are about to set.
                store.update_job(job["id"], state="cancelled",
                                 exit_reason="excluded by operator",
                                 remedy="none")
            store.set_structure_state(sid, StructureState.filtered_out,
                                      filter_reason=args.reason[:200])

        if refused:
            print(f"\n  refused {len(refused)}:")
            for sid, formula, why in refused:
                print(f"    {sid:>4}  {formula:<20} {why}")

        print("\nwrote the exclusions." if args.yes else "\ndry run -- pass --yes to write")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
