#!/usr/bin/env python
"""
ingest_new_seeds.py -- add seed structures to a campaign that has already run.

WHY THIS SCRIPT HAS TO EXIST
    Dropping new files into `inputs/seeds/` and re-running `csp run` does
    NOTHING, silently. The source stage reports how much work it has as:

        def pending(self, store):                 # stages/source_stage.py:40
            return 0 if store.compositions() else 1

    and the driver skips any stage reporting zero (driver.py:790). Once a
    campaign has ingested anything at all, stage 0 never runs again -- not even
    under `csp run --only source`. The new files are simply never read, and the
    run reports success.

    Forcing the stage to run is not the answer either: `write_plan` calls
    `store.add_structure` unconditionally and `add_structure` is a plain INSERT
    with no dedup, so every structure already in the campaign would be inserted
    a SECOND time, re-screened and re-run through DFT.

    So this script does what the source stage would do, and skips anything whose
    `content_hash` is already in the database. Existing rows are never touched:
    their ids, states, hull placements and filter events all survive, which is
    what makes this safe to run on a campaign whose DFT has already finished.

WHAT IT DOES
    1. expands the campaign's own `source:` blocks (same code path as stage 0)
    2. reads every `content_hash` already in the store
    3. inserts only the structures whose hash is absent, at the same state a
       fresh campaign would give them (`new` when `relax: true`)
    4. prints what it added and what it skipped

    After it runs, `csp run` picks the new rows up at `new` and carries them
    through screen -> reference -> filter -> dft exactly like the first batch.

INPUTS
    -c / --campaign   campaign.yaml (default: campaign.yaml in the cwd)
    --dry-run         report what would be added, write nothing
    Requires CSPFLOW_STORE to be set, as `csp run` does.

OUTPUTS
    rows added to the campaign database; a printed summary.
    Exit status 0 on success, 1 if nothing new was found.

RUN
    conda activate cspflow
    cd <campaign folder>
    export CSPFLOW_STORE=/scratch/oridwan/mp-reference
    python /projects/mmi/Ridwan/cspflow/scripts/repair/ingest_new_seeds.py -c campaign.yaml --dry-run
    python /projects/mmi/Ridwan/cspflow/scripts/repair/ingest_new_seeds.py -c campaign.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cspflow.config.loader import load_campaign
from cspflow.db.store import Origin, Store, StructureState
from cspflow.source import expand_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--campaign", default="campaign.yaml")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    path = Path(a.campaign).resolve()
    cfg = load_campaign(str(path))
    base = path.parent

    workdir = Path(str(cfg.campaign.workdir).replace("$USER", __import__("os").environ["USER"]))
    db = workdir / "campaign.db"
    if not db.is_file():
        print(f"no campaign database at {db}", file=sys.stderr)
        return 1
    store = Store(db)

    # Every content_hash already present. This is the only thing that decides
    # whether a file is new, so it must be read before anything is written.
    #
    # The two sides are spelled differently and comparing them raw silently
    # matches NOTHING, which would re-insert the whole campaign: the source
    # plan carries "sha256:bd18b117c6b87fa9" while the database stores
    # "bd18b117c6b87fa9" -- ASE's key-value layer drops the algorithm prefix on
    # write. Normalise both to the digest.
    def digest(value: str | None) -> str | None:
        return value.rsplit(":", 1)[-1] if value else None

    have = {digest(row.key_value_pairs.get("content_hash"))
            for row in store.structures()}
    have.discard(None)
    print(f"database: {db}")
    print(f"already ingested: {len(have)} structures\n")

    plan = expand_all(cfg.campaign, base)

    added, skipped = [], []
    for result in plan.results:
        for s in result.structures:
            if digest(s.content_hash) in have:
                skipped.append(s)
                continue
            added.append(s)

    for s in skipped:
        print(f"  skip (already ingested)  {Path(s.path).name}")
    print()
    for s in added:
        print(f"  ADD  {s.formula:<14} n={s.n_atoms:<4} {Path(s.path).name}")

    if not added:
        print("\nnothing new to add.")
        return 1

    if a.dry_run:
        print(f"\n--dry-run: would add {len(added)} structure(s), wrote nothing.")
        return 0

    for s in added:
        comp_id = store.add_composition(
            formula=s.formula, chemsys=s.chemsys, z=s.z, n_atoms=s.n_atoms,
            n_target=0, source_mode=s.source_mode, source_name=s.source_name,
            state="generated",
        )
        store.add_structure(
            s.atoms,
            origin=Origin.seed,
            state=StructureState.new if s.relax else StructureState.screened,
            composition_id=comp_id,
            reduced_formula=s.formula,
            source_name=s.source_name,
            source_mode=s.source_mode,
            source_path=s.path,
            content_hash=s.content_hash,
            needs_relax=s.relax,
        )

    print(f"\nadded {len(added)} structure(s); {len(skipped)} already present "
          f"and left untouched.")
    print("next:  csp run -c campaign.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
