"""Deduplication of screened structures.

**A deviation from pipeline.md, and the reason for it.** The plan groups dedup
inside Stage 2 (`screen` = MLIP relax + dedup). Operationally it cannot live
there: `screen` is a *submitted* stage whose unit of work is a chunk of a few
hundred structures, and duplicates are global -- two copies of one material can
easily land in different array tasks, or in tasks submitted cycles apart. A
per-chunk dedup would deduplicate within chunks and miss exactly the collisions
that matter.

So it is its own in-process stage, between `screen` and `reference` in the
funnel. That placement is not arbitrary either: it must run before the hull is
built, because fifty copies of one structure on a hull do not change the hull's
shape but do change every count, every "how many candidates survived", and every
per-composition budget downstream.

Generated structures are deduplicated by default -- that is the entire point of
Stage 2. **Seeds are not**, unless asked: a curated input list is a place where
two near-identical entries are usually deliberate, and silently merging them
loses which one survived (pipeline.md §0.3).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..config.loader import ResolvedConfig
from ..db.store import Origin, Store, StructureState
from .base import StageReport

# Set on every structure this stage has compared, survivor or duplicate. The
# state alone cannot say so: a survivor stays `screened`, which is also what an
# unexamined structure is.
CHECKED_KEY = "dedup_checked"


class DedupStage:
    name = "dedup"
    role = "cpu"
    in_process = True

    def __init__(self, cfg: ResolvedConfig, include_seeds: bool = False) -> None:
        self.cfg = cfg
        self.include_seeds = include_seeds

    def pending(self, store: Store) -> int:
        # Counted in SQL, not by reading the pool: this is asked every cycle,
        # and reading 14,752 structures to count unflagged ones took 59 s on
        # nfs4 (D144). Same rule as `_eligible`.
        return store.count_in_state_lacking(StructureState.screened, CHECKED_KEY)

    def claim(self, store, budget):                        # pragma: no cover
        raise AssertionError("dedup is an in-process stage; the driver calls run()")

    def build(self, items, workdir):                       # pragma: no cover
        raise AssertionError("dedup is an in-process stage; nothing is submitted")

    def reconcile(self, store, job_row, status, items):    # pragma: no cover
        pass

    #: Structures flagged per commit. Small enough that the write lock is held
    #: for seconds on NFS, not for the whole pool (D144).
    WRITE_CHUNK = 1000

    def run(self, store: Store) -> StageReport:
        if not self.pending(store):
            return StageReport(stage=self.name, note="nothing to compare")
        # Compare over the whole pool, not only the new arrivals: a structure
        # that duplicates one accepted three cycles ago is still a duplicate.
        # Read ONCE -- `_eligible` and `_pool` each used to read it again.
        rows = self._pool(store)

        matcher = self._matcher()
        if matcher is None:                                # pragma: no cover
            return StageReport(stage=self.name,
                               note="pymatgen unavailable; dedup skipped")

        by_formula: dict[str, list[Any]] = defaultdict(list)
        for row in rows:
            by_formula[row.toatoms().get_chemical_formula()].append(row)

        # Decide everything first, write nothing. SQLite takes the write lock at
        # the first write and keeps it to the commit, so writing inside this
        # loop held it for the whole of the structure matching -- minutes, on a
        # large pool, against a second driver on the same campaign (D144).
        collisions: list[tuple[int, int]] = []
        duplicates_of: list[tuple[int, int]] = []
        groups = 0
        for formula, members in sorted(by_formula.items()):
            if len(members) < 2:
                continue
            for survivor, duplicates in self._group(matcher, members):
                if not duplicates:
                    continue
                groups += 1
                for row in duplicates:
                    if row.get("origin") == Origin.seed.value and not self.include_seeds:
                        # Two seeds that match are a *result* -- a relaxed and an
                        # unrelaxed copy of one prototype, say. Recorded, kept.
                        collisions.append((int(row.id), int(survivor.id)))
                    else:
                        duplicates_of.append((int(row.id), int(survivor.id)))

        with store.transaction():
            for sid, survivor in collisions:
                store.add_filter_event(
                    structure_id=sid, gate="dedup:seed_collision",
                    passed=True, detail=f"matches structure {survivor}; kept",
                )
            for sid, survivor in duplicates_of:
                store.set_structure_state(sid, StructureState.deduped,
                                          duplicate_of=survivor)
                store.add_filter_event(
                    structure_id=sid, gate="dedup",
                    passed=False, detail=f"duplicate of structure {survivor}",
                )

        # Mark everything that was compared, survivors included. A survivor
        # keeps its `screened` state -- `filter` and `reference` read that, and
        # `deduped` means "removed as a duplicate" -- but it must not be
        # compared again. In chunks, one commit each.
        ids = [int(row.id) for row in rows]
        for start in range(0, len(ids), self.WRITE_CHUNK):
            with store.transaction():
                for sid in ids[start:start + self.WRITE_CHUNK]:
                    store.update_structure(sid, **{CHECKED_KEY: True})
        dropped, kept_seeds = len(duplicates_of), len(collisions)

        note = f"{groups} duplicate group(s)"
        if kept_seeds:
            note += f", {kept_seeds} seed collision(s) reported and kept"
        return StageReport(stage=self.name, claimed=dropped, reconciled=len(rows),
                           note=note)

    # -- internals ---------------------------------------------------------

    def _pool(self, store: Store) -> list[Any]:
        """Everything a new arrival has to be compared *against*.

        Survivors included: a structure that arrives in a later cycle and
        duplicates one already accepted is still a duplicate.
        """
        return list(store.structures(state=StructureState.screened.value))

    def _eligible(self, store: Store) -> list[Any]:
        """Structures not yet compared -- the actual work.

        Distinct from `_pool` on purpose. `pending` counted the pool, so a
        campaign whose structures were all unique reported work forever: the
        driver never considered dedup finished (`csp run` without `--watch`
        could not terminate) and the O(n^2) matching ran again over every
        survivor on every cycle.

        Found on the live campaign: 32 unique structures, "reconciled 32,
        0 duplicate groups", every cycle, unchanged.
        """
        return [r for r in self._pool(store)
                if not r.key_value_pairs.get(CHECKED_KEY)]

    def _matcher(self):
        try:
            from pymatgen.analysis.structure_matcher import StructureMatcher
        except ImportError:                                # pragma: no cover
            return None
        cfg = self.cfg.campaign.screen.dedup.matcher
        return StructureMatcher(ltol=cfg.ltol, stol=cfg.stol, angle_tol=cfg.angle_tol)

    def _group(self, matcher, members: list[Any]):
        """Group by structural identity; the lowest MLIP energy survives each group.

        Uses `StructureMatcher.group_structures`, which is materially faster
        than the naive O(N^2) `fit` loop -- it fingerprints first and only
        compares within candidate groups. At a few hundred structures per
        formula the difference is minutes.

        The survivor is the lowest-energy member rather than the first, so which
        structure survives does not depend on the order rows came out of the
        database.
        """
        from pymatgen.io.ase import AseAtomsAdaptor

        index = {}
        structures = []
        for row in members:
            structure = AseAtomsAdaptor.get_structure(row.toatoms())
            index[id(structure)] = row
            structures.append(structure)

        for group in matcher.group_structures(structures):
            rows = [index[id(s)] for s in group]
            rows.sort(key=lambda r: r.key_value_pairs.get("mlip_e_per_atom", float("inf")))
            yield rows[0], rows[1:]
