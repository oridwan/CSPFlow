#!/usr/bin/env python
"""
recarry_relaxed_geometry.py -- put each structure's MLIP-relaxed cell back into
                               the campaign, from the task databases on disk.

WHY THIS EXISTS
    D145. The screen worker saved MatterSim's forces as float32; ASE reads a
    result array back as float64. For an ODD atom count the read raised, and
    `_read_relaxed` swallowed it as "no relaxed cell", so the structure kept its
    UNRELAXED seed geometry -- and DFT would have started from it. For an EVEN
    atom count the read succeeded and the stored forces were garbage.

    Measured on RE-magnets-CHGNet: 4,520 of 14,752 structures lost, split
    exactly by atom-count parity; positions intact in every task database.

    The energies were never affected -- they travel in results-task<N>.json --
    so hull placement and filtering are right. Only the geometry handed to DFT
    is wrong, which is why this matters before Phase B and not after.

WHAT IT DOES
    For every relaxed result in <workdir>/screen/batches/*/results-task*.json:
      * reads the cell from its relaxed-task<N>.db, geometry columns only;
      * replaces the structure's geometry in campaign.db (this also clears the
        garbage force columns on the even-count rows);
      * records `mlip_geometry`, and refiles it under structures/<formula>/.
    One transaction: all of it lands, or none of it.

    SKIPS any structure DFT has already started from (dft_queued, dft_running,
    dft_done): changing the geometry under a running calculation would make the
    database describe a cell that was never computed. They are listed.

REFUSES while a driver for this campaign is in the queue. The driver is the
    only writer (D056); two writers is how a campaign loses work.

INPUTS
    -c, --config PATH   the campaign.yaml
    --batches PATH      screen batch root  [default: <workdir>/screen/batches]
    --apply             actually write; without it this is a dry run
    --force             skip the running-driver check (you have checked yourself)

OUTPUTS
    stdout   counts: lost and recovered, refreshed, skipped, unreadable

RUN
    conda activate cspflow
    python scripts/repair/recarry_relaxed_geometry.py -c campaigns/RE-magnets-CHGNet/campaign.yaml
    python scripts/repair/recarry_relaxed_geometry.py -c campaigns/RE-magnets-CHGNet/campaign.yaml --apply
"""
from __future__ import annotations

import argparse
import getpass
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

STARTED_DFT = {"dft_queued", "dft_running", "dft_done"}


def _driver_in_queue(campaign_dir: Path) -> list[str]:
    """Queue entries named like this campaign's driver (`<folder>-pre`, `-dft`, ...)."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i %j"],
                             capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    prefix = f"{campaign_dir.name}-"
    return [line for line in out.splitlines()
            if len(line.split()) > 1 and line.split()[1].startswith(prefix)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True, type=Path)
    ap.add_argument("--batches", type=Path, default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from cspflow.config.loader import load_campaign
    from cspflow.db.store import Store
    from cspflow.stages.screen_stage import ScreenStage, _read_relaxed

    cfg = load_campaign(a.config)
    batches = a.batches or cfg.work_dir / "screen" / "batches"
    results = sorted(batches.glob("*/results-task*.json"))
    if not results:
        print(f"no results-task*.json under {batches}")
        return 1

    running = _driver_in_queue(a.config.resolve().parent)
    if running and a.apply and not a.force:
        print("REFUSING: a driver for this campaign is in the queue:\n  "
              + "\n  ".join(running)
              + "\nThe driver is the only writer (D056). Wait for it to exit, or "
                "cancel it, then run this again.")
        return 2

    tally: Counter = Counter()
    skipped_dft: list[int] = []
    unreadable: list[int] = []
    screen = ScreenStage(cfg)

    with Store.open(cfg.campaign_db) as store:
        with store.transaction():
            for path in results:
                payload = json.loads(path.read_text())
                for row in payload.get("results", []):
                    geometry = row.get("geometry")
                    if row.get("error") or not geometry or not row.get("relaxed", True):
                        continue
                    sid = int(row["structure_id"])
                    try:
                        current = store.get_structure(sid)
                    except Exception:                          # noqa: BLE001
                        tally["not in campaign.db"] += 1
                        continue
                    kv = current.key_value_pairs
                    if kv.get("state") in STARTED_DFT:
                        skipped_dft.append(sid)
                        continue
                    cell = _read_relaxed(str(geometry), sid)
                    if cell is None:
                        unreadable.append(sid)
                        continue
                    was_lost = "mlip_geometry" not in kv
                    tally["lost, recovered" if was_lost else "carried, refreshed"] += 1
                    if not a.apply:
                        continue
                    store.replace_geometry(sid, cell)
                    store.update_structure(sid, mlip_geometry=str(geometry))
                    screen._keep(store, sid, cell, row,
                                 {"e_above_hull_mlip": kv.get("e_above_hull_mlip")})
            if not a.apply:
                # Nothing was written, but say so explicitly rather than rely on it.
                store.sql.rollback()

    print(f"campaign   {cfg.campaign.name}")
    print(f"database   {cfg.campaign_db}")
    print(f"results    {len(results)} task files under {batches}")
    for label in ("lost, recovered", "carried, refreshed", "not in campaign.db"):
        if tally[label]:
            print(f"  {label:<22} {tally[label]:>7,}")
    if skipped_dft:
        print(f"  {'skipped: DFT started':<22} {len(skipped_dft):>7,}   e.g. {skipped_dft[:5]}")
    if unreadable:
        print(f"  {'unreadable cell':<22} {len(unreadable):>7,}   e.g. {unreadable[:5]}")
    if not a.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to write it.")
    else:
        print("\nwritten, in one transaction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
