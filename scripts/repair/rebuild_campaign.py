#!/usr/bin/env python
"""
rebuild_campaign.py -- reconstruct a campaign database from the composition
                       databases beside it.

WHY THIS EXISTS
    `campaign.db` is a cache, and it has been wrong before: on 2026-09-12 two
    drivers shared one job table and 37 finished calculations became unreachable
    -- converged, on disk, and invisible to the campaign that paid for them
    (D129). The answer for DFT was `structure.json` in every run directory
    (D131); the answer for screening and generation is one ASE database per
    composition (D141).

    This is the other half of that promise. If those files can rebuild the
    campaign, the claim is true; if nothing ever does it, the claim is a hope.

WHAT IT RECOVERS, AND WHAT IT CANNOT
    RECOVERS   the structures themselves, their relaxed geometries, and every
               value measured for them: MLIP energy, convergence, step count,
               volume drift, provenance, hull distance where it was known.
               This is the part that cost GPU and CPU time.

    DOES NOT   job history, filter events, or the gate-by-gate narrative behind
               `csp status --why`. Those describe what the DRIVER did, and they
               live only in the database. A rebuilt campaign can be re-screened
               and re-filtered; it cannot be made to remember its own past.

    So this is a recovery tool, not a backup. It gets the compute back.

INPUTS
    --workdir PATH    the campaign workdir, holding structures/<formula>/*.db
    --out PATH        database to write         [default: <workdir>/rebuilt.db]
    --campaign NAME   name to stamp on it       [default: read from the files]
    --apply           actually write; without it this is a dry run

OUTPUTS
    stdout   what was found, per composition, and what would be / was written
    exit 0   always, unless the workdir has no structures/ tree

RUN
    conda activate cspflow
    python scripts/repair/rebuild_campaign.py --workdir /scratch/$USER/cspflow/t3-structure-list
    python scripts/repair/rebuild_campaign.py --workdir ... --apply
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--campaign", default="")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from cspflow import artifacts
    from cspflow.db.store import Origin, Store, StructureState

    root = a.workdir / "structures"
    if not root.is_dir():
        print(f"no composition databases under {root}", file=sys.stderr)
        return 1

    # `relaxed` last, so a structure that was both generated and then relaxed
    # ends up carrying its RELAXED cell rather than the one it started as.
    plan: list[tuple[str, str, Path]] = []
    for kind in ("generated", "relaxed"):
        for path in artifacts.compositions(a.workdir, kind):
            plan.append((path.parent.name, kind, path))
    if not plan:
        print(f"no .db files under {root}")
        return 0

    rows: dict[int, tuple[object, dict]] = {}
    counts: Counter = Counter()
    for formula, kind, path in plan:
        n = 0
        for row in artifacts.read(path):
            kv = dict(row.key_value_pairs)
            sid = kv.get("structure_id")
            if sid is None:
                counts[f"{kind}: no structure_id"] += 1
                continue
            rows[int(sid)] = (row.toatoms(), kv)
            n += 1
        counts[f"{formula}/{kind}"] = n

    for key in sorted(counts):
        print(f"  {counts[key]:>6}  {key}")
    print(f"\n{len(rows)} distinct structure(s) recoverable from {len(plan)} file(s)")

    out = a.out or (a.workdir / "rebuilt.db")
    if not a.apply:
        print(f"\nDRY RUN -- would write {out}\nRe-run with --apply to write it.")
        return 0
    if out.exists():
        print(f"\nrefusing: {out} already exists. Move it aside first.", file=sys.stderr)
        return 1

    name = a.campaign or next(
        (kv.get("campaign", "") for _, kv in rows.values() if kv.get("campaign")), "rebuilt")
    written = 0
    with Store.create(out, campaign=str(name)) as store:
        for sid in sorted(rows):
            atoms, kv = rows[sid]
            state = (StructureState.screened if kv.get("engine")
                     else StructureState.new)
            keep = {k: v for k, v in kv.items()
                    if k not in ("structure_id", "campaign", "origin")}
            store.add_structure(
                atoms,
                origin=Origin.generated if kv.get("origin") == "generated" else Origin.seed,
                state=state, **keep)
            written += 1
    print(f"\nwrote {out}  ({written} structures, campaign {name!r})")
    print("Job history and filter events are NOT recovered -- see the header.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
