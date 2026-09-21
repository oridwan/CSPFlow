#!/usr/bin/env python
"""
requeue.py -- put structures back in the DFT queue.

Replaces requeue_dft.py, retry_failed.py and restart_structure.py, which were
three files doing one operation with three different selectors, three partial
guard implementations, and three reset depths. D150.

WHY THIS IS NOT THE DFT RETRY LADDER
    `dft_stage` already retries what a recipe change can fix: an SCF that needs
    a different ALGO, a relaxation that needs more NSW. It deliberately does NOT
    retry a failure whose cause is environmental, because rerunning the
    identical calculation on the identical broken thing costs the same again and
    looks like diligence. Nor does it retry a cancellation, because that is
    usually deliberate and resubmitting would fight whoever typed scancel.

    Both are the right default, and both leave a gap: when the cause IS fixed --
    a bad node excluded, a module repaired, a quota raised -- or when you
    cancelled on purpose in order to resubmit differently, nothing puts those
    rows back. This does, and only when you say so.

SELECT one of
    --structure ID    exactly these structures (repeatable). Surgical.
    --reason TEXT     every `failed` structure whose dft_fail_reason contains
                      TEXT ("" matches all of them). Bulk, for a cause you have
                      just fixed -- e.g. "no OUTCAR" after constraining the
                      machine profile to AVX-512 hardware, since vasp_std dies
                      instantly on EPYC Rome and never writes an OUTCAR.
    --dead-jobs       every structure whose dft job row is queued/running/held/
                      pending/cancelled/failed while the job itself is gone from
                      squeue. Bulk, for after a mass scancel. --job ID narrows
                      it to particular ones.

DEPTH
    default           put the row back where it left off: state -> selected,
                      and its `dft_last_remedy` is left alone, so a structure
                      mid-ladder resumes from its own CONTCAR as intended.
    --from-seed       discard what is on disk and start over from the seed:
                      also zeroes dft_attempt, CLEARS dft_last_remedy, deletes
                      dft_fail_reason/dft_dir, moves the run directory to
                      <work_dir>/discarded/ with a NOTE, and deletes the job
                      rows.

    Clearing `dft_last_remedy` is the part that cannot be skipped and was
    missing from both predecessors. `DftStage.claim` reads that key
    UNCONDITIONALLY -- it does not consult the attempt counter -- so a row still
    carrying `{"remedy": "resume_from_contcar"}` resumes on its next claim even
    at attempt 0, `_archive_previous` finds the old directory, and the supposedly
    fresh run starts from the attempt you were discarding.

    D149 is the case that forced this. Structure 2498 of RE-magnets-CHGNet ran
    186 ionic steps on a 2x2x1 grid a resume had mis-derived. The grid carry
    added in D149 propagates whatever the previous attempt ACTUALLY ran, so
    resuming from that attempt would faithfully reproduce the bad grid. Only a
    restart from the seed, whose cell derives the grid correctly, gets back to
    2x2x2.

REFUSES
    - while a driver for this campaign is in the queue. The driver is the only
      writer (D056); two writers is how a campaign loses work. scancel it, run
      this, start it again. retry_failed.py had no such guard at all.
    - while any selected structure's own job is still visible in squeue.
      Requeueing running work gives you two jobs writing one directory, and the
      second silently overwrites the first.
    - on a structure that is `filtered_out`. Someone took it out of the campaign
      on purpose (see exclude_structures.py); a bulk requeue selecting on job or
      failure state knows nothing about that intent.

INPUTS
    -c, --campaign PATH   campaign.yaml, or the folder holding it   (required)
    one selector          --structure / --reason / --dead-jobs      (required)
    --job ID              with --dead-jobs: only this slurm id; repeatable
    --from-seed           full reset, discarding on-disk work
    --reason-note TEXT    recorded in the discarded directory name and its NOTE
                          (required with --from-seed)
    --discard-dir PATH    [default: <work_dir>/discarded]
    --apply               actually write; without it this is a dry run
    --force               skip the running-driver check (you have checked)

OUTPUTS
    stdout   what would be / was changed, per structure, and a tally by reason
    exit 0   on success or a clean dry run; 1 on a refusal

RUN
    conda activate cspflow

    # after fixing an environmental cause
    python scripts/requeue.py -c campaigns/CeFeB --reason "no OUTCAR"
    python scripts/requeue.py -c campaigns/CeFeB --reason "no OUTCAR" --apply

    # after a mass scancel
    python scripts/requeue.py -c campaigns/smoke-runs --dead-jobs --apply

    # discard one structure's work and start it from the seed
    python scripts/requeue.py -c campaigns/RE-magnets-CHGNet -s 2498 \
        --from-seed --reason-note "D149 resume ran a mis-derived 2x2x1 grid"
"""
from __future__ import annotations

import argparse
import collections
import getpass
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from cspflow.config.loader import load_campaign
from cspflow.db.store import Store, StructureState
from cspflow.dft.layout import Layout, resolve as resolve_layout

STEP_KEY = "dft_step"
ATTEMPT_KEY = "dft_attempt"
LAST_REMEDY_KEY = "dft_last_remedy"
DIR_KEY = "dft_dir"
FAIL_KEY = "dft_fail_reason"

# Job states that mean "this job should be doing work". A job row in one of
# these whose slurm id is no longer in squeue is a dead job.
LIVE_STATES = {"queued", "running", "held", "pending", "cancelled", "failed"}


def squeue_rows() -> list[tuple[str, str]]:
    """(job id, job name) for everything this user has in the queue."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", getpass.getuser(),
                              "-o", "%A|%j"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.stdout.splitlines():
        if "|" in line:
            jid, _, name = line.partition("|")
            rows.append((jid.strip(), name.strip()))
    return rows


def dirs_for(layout: Layout, sid: int, dft_root: Path) -> list[Path]:
    """Every directory on disk that belongs to THIS structure, and nothing else.

    Not the job row's `workdir`. That is the ARRAY's directory -- for a combined
    job it is `dft/` itself, so resolving it as "the structure's directory"
    would move the whole campaign. A dry run caught exactly that.

    Under `runs`, the structure owns one directory whose name begins with its
    zero-padded id; it is found by parsing the slug rather than rebuilding it,
    so a formula or seed label that has since changed cannot miss it. Under
    `stages` there is no per-structure directory -- only per-step ones -- and
    all of them are returned.
    """
    from cspflow.dft.layout import structure_id_of

    if layout.name == "runs":
        runs = dft_root / "runs"
        found = [d for d in sorted(runs.iterdir())
                 if d.is_dir() and structure_id_of(d.name) == sid] \
            if runs.is_dir() else []
    else:
        found = [d for d in sorted(dft_root.glob(f"dft-{sid}-*")) if d.is_dir()]

    # Belt and braces: nothing at or above the dft root may ever be moved.
    root = dft_root.resolve()
    for d in found:
        rd = d.resolve()
        if rd == root or rd in root.parents:
            raise SystemExit(
                f"refusing: resolved {rd} for structure {sid}, which is the dft "
                f"root or above it. That is a bug in this script, not a state "
                f"to force past.")
    return found


def select(store: Store, args) -> tuple[list[int], list[dict]]:
    """(structure ids, job rows to delete) for whichever selector was given."""
    if args.structure:
        return sorted(set(args.structure)), []

    if args.reason is not None:
        # `dft_fail_reason` must be PRESENT, not merely match. A structure that
        # failed before DFT carries `fail_reason` instead and has no
        # `dft_fail_reason` at all, so an empty --reason substring matched it
        # too and sent it to VASP. Measured on RE-magnets-CHGNet: three rows
        # rejected at screening for atoms 0.177-0.286 A apart would have been
        # requeued into DFT by `retry_failed.py --reason ""`.
        failed, pre_dft = [], []
        for r in store.structures():
            kv = r.key_value_pairs
            if kv.get("state") != StructureState.failed.value:
                continue
            if not kv.get(FAIL_KEY):
                pre_dft.append((int(r.id), str(kv.get("fail_reason") or "?")))
            elif args.reason in str(kv[FAIL_KEY]):
                failed.append(r)
        if pre_dft:
            print(f"skipping {len(pre_dft)} structure(s) that failed BEFORE "
                  f"DFT -- they have no dft_fail_reason, so DFT is not what "
                  f"needs retrying:")
            for sid, why in pre_dft[:10]:
                print(f"  {sid:>5}  {why[:70]}")
            if len(pre_dft) > 10:
                print(f"  ... and {len(pre_dft) - 10} more")
            print()
        hits = failed
        tally = collections.Counter(str(r.key_value_pairs.get(FAIL_KEY, "?"))
                                    for r in hits)
        if tally:
            print(f"matched {len(hits)} failed structure(s) on "
                  f"reason {args.reason!r}:")
            for reason, n in tally.most_common():
                print(f"  {n:4d}  {reason[:74]}")
            print()
        return sorted(int(r.id) for r in hits), []

    rows = [j for j in store.jobs()
            if j["stage"] == "dft" and j["state"] in LIVE_STATES]
    if args.job:
        rows = [j for j in rows if str(j["slurm_id"]) in set(args.job)]
    return sorted({int(j["structure_id"]) for j in rows if j["structure_id"]}), rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="put structures back in the DFT queue",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--campaign", required=True)
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("-s", "--structure", action="append", type=int,
                      help="structure id; repeatable")
    pick.add_argument("--reason", default=None,
                      help="fail-reason substring; '' matches every failed row")
    pick.add_argument("--dead-jobs", action="store_true",
                      help="structures whose dft job is gone from squeue")
    ap.add_argument("--job", action="append", default=[],
                    help="with --dead-jobs: only this slurm id; repeatable")
    ap.add_argument("--from-seed", action="store_true")
    ap.add_argument("--reason-note", default=None)
    ap.add_argument("--discard-dir", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.from_seed and not args.reason_note:
        print("refusing: --from-seed discards work, so --reason-note is "
              "required. It is recorded in the discarded directory's name and "
              "in a NOTE file inside it.", file=sys.stderr)
        return 1

    path = Path(args.campaign)
    if path.is_dir():
        path = path / "campaign.yaml"
    cfg = load_campaign(path)

    db = Path(cfg.work_dir) / "campaign.db"
    if not db.exists():
        print(f"no database at {db}", file=sys.stderr)
        return 1

    queued = squeue_rows()
    name = cfg.campaign.name
    drivers = [j for j, n in queued if n.startswith(f"{name}-d")]
    if drivers and not args.force:
        print(f"refusing: driver {', '.join(drivers)} is in the queue for "
              f"{name}. The driver is the only writer (D056) -- scancel it, run "
              f"this, then start it again. --force overrides.", file=sys.stderr)
        return 1
    live_ids = {j for j, _ in queued}

    # Resolved against the directories, not just the config -- a campaign on the
    # old `stages` layout keeps its work in `dft-<id>-<step>/`, and reading the
    # config alone would look for `runs/` and find nothing to move.
    dft_root = Path(cfg.work_dir) / "dft"
    layout_name, note = resolve_layout(cfg.campaign.dft.layout, dft_root)
    layout = Layout(layout_name, dft_root)
    if note:
        print(f"layout: {note}\n")

    discard = Path(args.discard_dir) if args.discard_dir \
        else Path(cfg.work_dir) / "discarded"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-",
                  (args.reason_note or "").lower()).strip("-")[:48]

    store = Store.open(db)
    try:
        sids, job_rows = select(store, args)

        # Every job row belonging to a selected structure, whichever selector
        # found it -- --from-seed deletes them, and the squeue guard reads them.
        if sids and not job_rows:
            job_rows = [j for j in store.jobs()
                        if j["stage"] == "dft" and j["structure_id"]
                        and int(j["structure_id"]) in set(sids)]

        still = sorted({str(j["slurm_id"]) for j in job_rows
                        if str(j["slurm_id"]) in live_ids})
        if still:
            print(f"refusing: job(s) {', '.join(still)} are still in the queue. "
                  f"scancel them and wait for them to disappear first.",
                  file=sys.stderr)
            return 1

        # A structure someone took out of the campaign on purpose is
        # `filtered_out` with a `filter_reason` (exclude_structures.py). The job
        # and reason selectors know nothing about that intent, so without this a
        # bulk requeue silently resurrects every excluded structure.
        excluded = {}
        for sid in list(sids):
            row = next(store.structures(id=sid), None)
            if row is None:
                print(f"refusing: no structure {sid} in {db}", file=sys.stderr)
                return 1
            if row.key_value_pairs.get("state") == StructureState.filtered_out.value:
                excluded[sid] = str(row.key_value_pairs.get("filter_reason")
                                    or "no reason recorded")
                sids.remove(sid)
        if excluded:
            print(f"skipping {len(excluded)} structure(s) excluded on purpose:")
            for sid, why in excluded.items():
                print(f"  {sid:>5}  {why[:70]}")
            print()
        job_rows = [j for j in job_rows
                    if not j["structure_id"]
                    or int(j["structure_id"]) not in excluded]

        if not sids:
            print("nothing to requeue")
            return 0

        plan = []
        for sid in sids:
            row = next(store.structures(id=sid), None)
            kv = row.key_value_pairs
            jobs = [j for j in job_rows if j["structure_id"]
                    and int(j["structure_id"]) == sid]
            moves = dirs_for(layout, sid, dft_root) if args.from_seed else []
            plan.append((sid, row, kv, jobs, moves))

        verb = "restarting from seed" if args.from_seed else "requeueing"
        print(f"{verb} {len(plan)} structure(s)   ({db})\n")
        show = plan if (args.from_seed or len(plan) <= 20) else plan[:20]
        for sid, row, kv, jobs, moves in show:
            formula = row.toatoms().get_chemical_formula()
            print(f"structure {sid}  {formula}")
            print(f"  state            {kv.get('state', '?')} -> selected")
            print(f"  {STEP_KEY:<16} {kv.get(STEP_KEY, 0)!r} -> "
                  f"{0 if args.from_seed else kv.get(STEP_KEY, 0)!r}")
            if args.from_seed:
                print(f"  {ATTEMPT_KEY:<16} {kv.get(ATTEMPT_KEY, 0)!r} -> 0")
                print(f"  {LAST_REMEDY_KEY:<16} "
                      f"{str(kv.get(LAST_REMEDY_KEY, ''))[:42]!r} -> ''")
                for key in (FAIL_KEY, DIR_KEY):
                    if key in kv:
                        print(f"  {key:<16} deleted  ({str(kv[key])[:42]!r})")
                for d in moves:
                    print(f"  move             {d}")
                    print(f"                -> {discard / f'{d.name}-{stamp}-{slug}'}")
                if not moves:
                    print(f"  move             nothing on disk for this structure")
                print(f"  job rows         {len(jobs)} deleted "
                      f"({', '.join(str(j['slurm_id']) for j in jobs) or 'none'})")
            else:
                print(f"  {LAST_REMEDY_KEY:<16} left alone -- it resumes as the "
                      f"ladder intended")
            print()
        if len(show) < len(plan):
            print(f"  ... and {len(plan) - len(show)} more\n")

        if not args.apply:
            print("dry run -- pass --apply to write")
            return 0

        for sid, row, kv, jobs, moves in plan:
            if args.from_seed:
                for run_dir in moves:
                    target = discard / f"{run_dir.name}-{stamp}-{slug}"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(run_dir), str(target))
                    (target / "NOTE").write_text(
                        f"moved aside by requeue.py --from-seed on "
                        f"{datetime.now().isoformat(timespec='seconds')}\n"
                        f"structure {sid}\n"
                        f"reason: {args.reason_note}\n\n"
                        f"The structure was reset to `selected` at step 0, "
                        f"attempt 0, with dft_last_remedy cleared, so its next "
                        f"run starts from the seed rather than resuming from\n"
                        f"anything in here.\n")
                store.update_structure(
                    sid, state=StructureState.selected,
                    delete_keys=[k for k in (FAIL_KEY, DIR_KEY) if k in kv],
                    **{STEP_KEY: 0, ATTEMPT_KEY: 0, LAST_REMEDY_KEY: ""})
                for job in jobs:
                    store.sql.execute("DELETE FROM job WHERE id=?", (job["id"],))
            else:
                store.set_structure_state(sid, StructureState.selected,
                                          **{STEP_KEY: int(kv.get(STEP_KEY, 0))})
        store.sql.commit()
        print(f"done. Start the driver and it will claim "
              f"{len(plan)} structure(s) on its next cycle.")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
