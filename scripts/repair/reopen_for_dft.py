#!/usr/bin/env python
"""
reopen_for_dft.py -- send every screened structure of a campaign to DFT, magnets
                     first, after it was filtered on hull distance.

WHY THIS EXISTS
    D147. RE-magnets-CHGNet filtered on the MLIP hull at 0.05 eV/atom and
    removed 10,434 of 14,752 screened structures. The decision since then is to
    ignore hull distance and compute all of them. Editing `e_above_hull_max`
    alone does nothing: the filter stage reads only structures still in state
    `screened`, so a structure already marked `filtered_out` is never looked at
    again, whatever the threshold. They have to be put back.

    Second, the order. `select.rank_by: e_above_hull_mlip` runs the most stable
    structures first, and here those are Al-rich non-magnets (the first 64 DFT
    jobs had a median CHGNet J_s(TM) of 0.03 T, none above 1.5 T). The pipeline
    cannot rank on moment itself, so this writes the CHGNet ranking into each
    row where `rank_by` can read it.

WHAT IT DOES
    1. Joins every structure to its CHGNet score through the merged_id in its
       seed filename (inputs/seed_manifest.csv) and writes two keys:
         chgnet_js_tm_tesla   J_s(TM), the transition-metal sublattice
                              polarisation in tesla (reporting)
         chgnet_js_rank       1 = highest J_s(TM). An integer so that
                              `rank_by`, which sorts ASCENDING, runs magnets first.
       Written on every structure, including ones DFT has started, because the
       keys are labels and change nothing about a calculation.
    2. Puts every `filtered_out` structure back to `screened`, deletes its stale
       `filter_reason`, and records the reopening as a filter_event, so the
       history of why it was once removed is kept, not overwritten.
    3. Reads campaign.yaml and says, before anything runs, how many structures
       the NEXT driver would actually pass through filter and select with the
       config as it stands -- so a forgotten config edit shows up here and not
       as a campaign that quietly stops at 150.
    One transaction: all of it lands, or none of it.

    Does NOT touch structures in `failed` (no hull placement; screening failed),
    and does not change any geometry or energy.

REFUSES while a driver for this campaign is in the queue. The driver is the
    only writer (D056); two writers is how a campaign loses work.

INPUTS
    -c, --config PATH    the campaign.yaml
    --manifest PATH      seed manifest  [default: <campaign>/inputs/seed_manifest.csv]
    --apply              actually write; without it this is a dry run
    --force              skip the running-driver check (you have checked yourself)

OUTPUTS
    stdout   counts written, and what the next driver would pass under the
             current config

RUN
    conda activate cspflow
    python scripts/repair/reopen_for_dft.py -c campaigns/RE-magnets-CHGNet/campaign.yaml
    python scripts/repair/reopen_for_dft.py -c campaigns/RE-magnets-CHGNet/campaign.yaml --apply
"""
from __future__ import annotations

import argparse
import csv
import getpass
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REASON = "hull-distance and per-composition caps removed; all screened structures go to DFT (D147)"
RTM = re.compile(r"(RTM-\d+)")


def _driver_in_queue(campaign_dir: Path) -> list[str]:
    """Queue entries named like this campaign's driver (`<folder>-pre`, `-dft`, ...)."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i %j"],
                             capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    prefix = f"{campaign_dir.name}-"
    # DFT jobs are named <campaign>-<id>-<formula>; the driver is <campaign>-<phase>.
    phases = {"pre", "dft", "all", "run"}
    hits = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[1].startswith(prefix):
            tail = parts[1][len(prefix):]
            if tail in phases or len(tail) == 4:
                hits.append(line)
    return hits


def _js(value: str) -> float | None:
    try:
        x = float(value)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from cspflow.config.loader import load_campaign
    from cspflow.db.store import Store

    campaign_dir = a.config.resolve().parent
    cfg = load_campaign(a.config)
    manifest = a.manifest or campaign_dir / "inputs" / "seed_manifest.csv"

    running = _driver_in_queue(campaign_dir)
    if running and a.apply and not a.force:
        print("REFUSING: a driver for this campaign is in the queue:\n  "
              + "\n  ".join(running)
              + "\nThe driver is the only writer (D056). Stop it first:\n"
                f"  touch {campaign_dir}/PAUSED\n"
                "  scancel --signal=TERM --batch <jobid>   # exits at a cycle boundary\n"
                "then run this again once `squeue` no longer lists it.")
        return 2

    # CHGNet score per merged_id, ranked: 1 = highest J_s(TM).
    scores = {r["merged_id"]: _js(r["J_s_TM_tesla"]) for r in csv.DictReader(manifest.open())}
    ranked = sorted((m for m, v in scores.items() if v is not None), key=lambda m: -scores[m])
    rank = {m: i + 1 for i, m in enumerate(ranked)}

    tally: Counter = Counter()
    by_state: Counter = Counter()
    no_hull: list[int] = []

    with Store.open(cfg.campaign_db) as store:
        hull = {int(sid) for (sid,) in store.sql.execute(
            "SELECT structure_id FROM hull WHERE hull_type='mlip'")}
        with store.transaction():
            for row in store.structures():
                sid = int(row.id)
                kv = row.key_value_pairs
                state = kv.get("state")
                by_state[state] += 1

                m = RTM.search(str(kv.get("source_path", "")))
                merged = m.group(1) if m else None
                if merged in rank:
                    tally["scored"] += 1
                    if a.apply:
                        store.update_structure(sid, chgnet_js_tm_tesla=float(scores[merged]),
                                               chgnet_js_rank=int(rank[merged]))
                else:
                    tally["no CHGNet score"] += 1

                if state != "filtered_out":
                    continue
                if sid not in hull:
                    no_hull.append(sid)          # the filter could not pick it up again
                    continue
                tally["reopened"] += 1
                tally[f"was: {kv.get('filter_reason', '?')[:40]}"] += 1
                if a.apply:
                    store.update_structure(sid, state="screened",
                                           delete_keys=["filter_reason"])
                    store.add_filter_event(structure_id=sid, gate="reopen", passed=True,
                                           detail=REASON)
            if not a.apply:
                store.sql.rollback()

    # What the NEXT driver would do with campaign.yaml as it stands now.
    f = cfg.campaign.filter
    sel = cfg.campaign.dft.select
    total = sum(n for s, n in by_state.items() if s != "failed")
    with Store.open(cfg.campaign_db) as store:
        dist = [float(v) for (v,) in store.sql.execute(
            "SELECT e_above_hull FROM hull WHERE hull_type='mlip'")]
    over_hull = sum(1 for d in dist if d > f.e_above_hull_max)

    print(f"campaign   {cfg.campaign.name}")
    print(f"database   {cfg.campaign_db}")
    print(f"manifest   {manifest}  ({len(rank):,} scored seeds)")
    print("\nstates now:  " + "  ".join(f"{s}={n:,}" for s, n in by_state.most_common()))
    print(f"\n  {'CHGNet keys written':<34} {tally['scored']:>7,}")
    if tally["no CHGNet score"]:
        print(f"  {'no CHGNet score (sorts last)':<34} {tally['no CHGNet score']:>7,}")
    print(f"  {'filtered_out -> screened':<34} {tally['reopened']:>7,}")
    for k, v in sorted(tally.items()):
        if k.startswith("was: "):
            print(f"      {k:<30} {v:>7,}")
    if no_hull:
        print(f"  {'filtered_out, no hull placement':<34} {len(no_hull):>7,}   left alone, e.g. {no_hull[:5]}")

    print("\nwith campaign.yaml AS IT STANDS, the next driver would:")
    problems = []
    line = f"  filter.e_above_hull_max = {f.e_above_hull_max:g}"
    if over_hull:
        problems.append(f"{over_hull:,} structures are above it and would be filtered again")
        line += f"   <- {over_hull:,} still above it"
    print(line)
    line = f"  filter.max_per_composition = {f.max_per_composition}"
    if f.max_per_composition < 1000:
        problems.append("filter.max_per_composition would cap repeated formulas")
        line += "   <- caps repeated formulas"
    print(line)
    line = f"  select.max_per_composition = {sel.max_per_composition}"
    if sel.max_per_composition < 1000:
        problems.append("select.max_per_composition would cap repeated formulas")
        line += "   <- caps repeated formulas"
    print(line)
    line = f"  select.max_total = {sel.max_total:,}"
    if sel.max_total < total:
        problems.append(f"select.max_total stops DFT at {sel.max_total:,} of {total:,}")
        line += f"   <- stops at {sel.max_total:,} of {total:,}"
    print(line)
    line = f"  select.rank_by = {sel.rank_by}"
    if sel.rank_by != "chgnet_js_rank":
        problems.append("rank_by is not chgnet_js_rank, so magnets do not run first")
        line += "   <- magnets do NOT run first"
    print(line)

    if problems:
        print("\nEDIT campaign.yaml BEFORE starting a driver:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nconfig is ready: every screened structure passes, magnets first.")

    if not a.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to write it.")
    else:
        print("\nwritten, in one transaction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
