"""Stage 3 -- fetch the reference phases and place every screened candidate.

In-process: seconds to minutes, no scheduler. The expensive part is the network,
and it is paid once per chemical system across all campaigns because the cache
lives outside any of them.

The stage does two things that are easy to conflate and must not be:

*   It **fetches** MP phases for each chemical system the campaign touches, on
    one `thermo_type` and one energy scale, and stores both energies per entry.
*   It **places** each screened candidate on the hull built from those phases,
    on the MLIP energy scale.

The second is where the subtlety lives, and it used to be wrong.

**What it used to do.**  The candidates carried MatterSim energies and the
vertices came from MP's raw DFT, on the argument that MatterSim is trained on
MPtrj so the two are "close".  Stage 4a existed to measure how close rather than
assume it.  Measured on the store 2026-09-12, that hull misplaces roughly a
third of the candidates in Ce chemistry by more than the 0.06 eV/atom selection
threshold:

    system      placed >0.06 eV/atom from where a one-scale hull puts them
    Ce-Ge-Pd    20 / 64
    Ce-Cu-Ge    21 / 66
    Ce-Ge-Sb    23 / 62
    Fe-Pd        0 / 20     <- no Ce

Ce is the culprit: ours minus MP is +1.17 eV/atom for elemental Ce, a different
4f POTCAR, and MatterSim inherits MP's side of that.

**What it does now.**  The vertices are the store's OWN MatterSim energies, from
the same model and the same settings that produced the candidate energies.  One
scale on both sides, so there is nothing left to calibrate -- which is why
stage 4 is gone (D126).  If the store cannot cover a system, the stage says so
loudly rather than quietly reaching for MP's numbers.

The placement is still recorded as `hull_type='mlip'`, kept separate from any
DFT hull, and never silently compared against one.
"""

from __future__ import annotations

from collections import defaultdict

from ..config.loader import ResolvedConfig
from ..db.store import Store, StructureState
from ..reference.corrections import audit_chemsystems
from ..reference.hull import Entry, HullError, build_hull
from ..reference.mp import ReferenceError, fetch_chemsys
from .base import StageReport, WorkItem


class ReferenceStage:
    name = "reference"
    role = "cpu"
    in_process = True

    def __init__(self, cfg: ResolvedConfig) -> None:
        self.cfg = cfg
        self.last_snapshots: dict[str, str] = {}

    def pending(self, store: Store) -> int:
        """Work for this stage: structures to place, or a reference set to fetch.

        Two triggers, not one. The obvious trigger is screened structures with
        no hull placement. The second is a chemical system that has results in
        it and no reference entries at all -- which is the state an *ingested*
        campaign starts in, because its structures arrive already at
        `dft_done` and never pass through screening. Without it the DFT hull in
        `analyze` has nothing to measure against and refuses, correctly, for a
        reason the user cannot act on.
        """
        rows = store.sql.execute(
            "SELECT COUNT(DISTINCT chemsys) n FROM composition").fetchone()
        if not rows or not rows["n"]:
            return 0

        entries = store.sql.execute(
            "SELECT COUNT(*) n FROM reference_entry").fetchone()["n"]
        if not entries and store.count_structures(state=StructureState.dft_done.value):
            return int(rows["n"])

        # Structures still waiting for a placement -- counted, not inferred.
        # It was `0 if placed >= screened else screened`, comparing two totals:
        # every hull row (DFT ones included) against every screened structure
        # (those with no MLIP energy included). A system that could not be
        # hulled never caught up, so the WHOLE placement re-ran every cycle.
        return store.unplaced_mlip_candidates()

    def claim(self, store, budget):                        # pragma: no cover
        raise AssertionError("reference is an in-process stage; the driver calls run()")

    def build(self, items, workdir):                       # pragma: no cover
        raise AssertionError("reference is an in-process stage; nothing is submitted")

    def reconcile(self, store, job_row, status, items):    # pragma: no cover
        pass

    def run(self, store: Store) -> StageReport:
        reference = self.cfg.campaign.reference
        thermo_type = _thermo_label(reference)
        scale = getattr(reference, "energy_scale", "raw")

        chemsystems = store.chemsystems()
        _, audit = audit_chemsystems(chemsystems)

        fetched, placed, notes = 0, 0, []
        by_chemsys: dict[str, list[Entry]] = {}

        for chemsys in chemsystems:
            try:
                result = fetch_chemsys(chemsys, thermo_type=thermo_type,
                                       energy_scale=scale)
            except ReferenceError as exc:
                notes.append(f"{chemsys}: {exc}")
                continue

            self.last_snapshots[chemsys] = result.snapshot_id
            # One commit per SYSTEM (D144). Per entry was 14,597 fsync round
            # trips on NFS; per run held the write lock for minutes against a
            # second driver on the same campaign.
            with store.transaction():
                for entry in result.entries:
                    store.add_reference_entry(
                        mp_id=entry.mp_id, chemsys=entry.chemsys,
                        thermo_type=entry.thermo_type, run_type=entry.run_type,
                        formula=entry.formula, n_atoms=entry.n_atoms,
                        e_dft_raw=entry.e_raw_per_atom,
                        e_dft_corrected=entry.e_corrected_per_atom,
                        correction=_correction(entry),
                        snapshot_id=result.snapshot_id, state="fetched",
                    )
                    fetched += 1
            by_chemsys[chemsys] = self._vertices(chemsys, result, scale, notes)
            notes.extend(result.warnings)

        placed, place_notes = self._place(store, by_chemsys, scale)
        notes.extend(place_notes)

        note = audit.splitlines()[0] if audit else ""
        if notes:
            note += f"; {len(notes)} note(s)"
        return StageReport(stage=self.name, claimed=fetched, reconciled=placed, note=note)

    def _vertices(self, chemsys: str, result, scale: str,
                  notes: list[str]) -> list[Entry]:
        """Reference vertices for the FILTER hull, on the candidates' own scale.

        The candidates are relaxed by MatterSim, so the vertices must be
        MatterSim too or the hull mixes two scales -- see the module docstring
        for what that costs.  The store carries a MatterSim energy for every
        phase it holds, so this is a folder read, not a network call.

        Falling back to MP is NOT silent.  A mixed hull still filters, and
        filtering has to proceed, but the note says which systems it happened in
        so the verdicts there can be discounted.
        """
        from ..reference.refstore import StoreError, entries_for

        try:
            vertices = entries_for(chemsys, "mlip")
        except StoreError as exc:
            notes.append(
                f"{chemsys}: no MatterSim reference in the store ({exc}); "
                f"falling back to MP's DFT vertices, so this system's hull "
                f"MIXES two scales and its filter verdicts are unreliable")
            return result.hull_entries(scale)
        except Exception as exc:                                   # noqa: BLE001
            notes.append(f"{chemsys}: store read failed ({exc}); using MP vertices")
            return result.hull_entries(scale)

        if not vertices:
            notes.append(f"{chemsys}: store holds no MatterSim energies; using MP vertices")
            return result.hull_entries(scale)
        return vertices

    def _place(self, store: Store, by_chemsys: dict[str, list[Entry]],
               scale: str) -> tuple[int, list[str]]:
        """Put every screened candidate on its own chemical system's hull.

        THE SCREENED SET IS READ ONCE (D144). It was read once PER SYSTEM --
        `_candidates` selected all 14,752 screened structures, rebuilt each as
        an Atoms object to learn its elements, and kept the few that matched,
        288 times over. At 65 s a read on nfs4 that alone was 5.2 hours.

        A SYSTEM IS REBUILT WHOLE, OR NOT AT ALL. The candidates are vertices of
        the hull they are placed on, so one new low-energy structure can lower
        the hull under every structure placed before it: placing only the new
        arrivals would leave the old distances stale and wrong. A system is
        rebuilt when any candidate in it has no placement against the CURRENT
        reference snapshot -- a new arrival, or a reference set that moved --
        and every candidate in it is placed again. A system where nothing moved
        is skipped.

        A system whose hull cannot be built marks its unplaced candidates
        `hull_error`, so `pending` stops counting them. The reason is in the
        note and on the row; they are retried whenever the system is rebuilt.
        """
        pool = self._candidates_by_chemsys(store)
        already = store.mlip_hull_hashes()
        placed, notes = 0, []

        # A system with candidates but no reference set at all -- its fetch
        # failed, or its elements match no composition row -- never reaches the
        # loop below. Unmarked, its candidates stayed "pending" for ever and the
        # whole stage re-ran every cycle. Same treatment as a failed hull.
        with store.transaction():
            for chemsys in sorted(set(pool) - set(by_chemsys)):
                reason = f"{chemsys}: no reference set was fetched for this system"[:200]
                for candidate in pool[chemsys]:
                    if candidate.structure_id is not None and \
                            candidate.structure_id not in already:
                        store.update_structure(candidate.structure_id, hull_error=reason)

        for chemsys, reference in by_chemsys.items():
            candidates = pool.get(chemsys)
            if not candidates:
                continue
            ref_hash = self.last_snapshots.get(chemsys, "")
            if all(ref_hash in already.get(c.structure_id, ()) for c in candidates):
                continue
            # One commit per system: seconds of write lock, not the whole run.
            with store.transaction():
                placed += self._place_system(store, chemsys, reference, candidates,
                                             ref_hash, scale, already, notes)
        return placed, notes

    def _place_system(self, store: Store, chemsys: str, reference: list[Entry],
                      candidates: list[Entry], ref_hash: str, scale: str,
                      already: dict[int, set[str]], notes: list[str]) -> int:
        """Build one system's hull and place every candidate in it."""
        try:
            hull = build_hull([*reference, *candidates])
        except HullError as exc:
            notes.append(f"{chemsys}: {exc}")
            reason = f"{chemsys}: {exc}"[:200]
            for candidate in candidates:
                if candidate.structure_id is not None and \
                        candidate.structure_id not in already:
                    store.update_structure(candidate.structure_id, hull_error=reason)
            return 0

        placed = 0
        for candidate in candidates:
            sid = candidate.structure_id
            if sid is None:                                # pragma: no cover
                continue
            store.add_hull(
                structure_id=sid, hull_type="mlip",
                energy_scale="raw" if scale == "raw" else "mp_corrected",
                e_above_hull=hull.e_above_hull[candidate.label],
                formation_energy=hull.formation_energy.get(candidate.label),
                ref_set_hash=ref_hash,
            )
            store.update_structure(sid, delete_keys=["hull_error"],
                                   e_above_hull_mlip=hull.e_above_hull[candidate.label])
            placed += 1
        notes.extend(hull.warnings)
        return placed

    def _candidates(self, store: Store, chemsys: str) -> list[Entry]:
        """Screened structures in one chemical system, as hull entries."""
        return self._candidates_by_chemsys(store).get(chemsys, [])

    @staticmethod
    def _candidates_by_chemsys(store: Store) -> dict[str, list[Entry]]:
        """Every screened structure with an MLIP energy, grouped by system.

        The MLIP energy enters as `scale='raw'` because the vertices it is
        measured against are now MatterSim too (`_vertices`), so both sides
        carry the same model's zero and no correction applies to either.  The
        placement is labelled `hull_type='mlip'` so it can never be mistaken
        for a DFT hull.

        The system is taken from the atoms themselves, not the composition
        row, and only the columns that decide it are read: the element numbers
        and the key-values. Positions and cell are not needed to know what a
        structure is made of.
        """
        out: dict[str, list[Entry]] = defaultdict(list)
        rows = store.structures(state=StructureState.screened.value,
                                columns=["id", "numbers", "key_value_pairs"],
                                include_data=False)
        for row in rows:
            energy = row.key_value_pairs.get("mlip_e_per_atom")
            if energy is None:
                continue
            counts = _counts(row)
            out["-".join(sorted(counts))].append(Entry(
                label=f"cand-{row.id}", counts=counts,
                energy=float(energy) * sum(counts.values()),
                scale="raw", source="mlip", structure_id=int(row.id),
            ))
        return dict(out)


def _counts(row) -> dict[str, int]:
    """Element counts from the row's atomic numbers -- no Atoms object built."""
    from ase.data import chemical_symbols

    counts: dict[str, int] = {}
    for z in row.numbers:
        symbol = chemical_symbols[int(z)]
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _correction(entry) -> float | None:
    if entry.e_raw_per_atom is None or entry.e_corrected_per_atom is None:
        return None
    return entry.e_corrected_per_atom - entry.e_raw_per_atom


def _thermo_label(reference) -> str:
    """Map the campaign's `thermo_type` enum onto MP's own label."""
    value = getattr(getattr(reference, "thermo_type", None), "value", None)
    return str(value or getattr(reference, "thermo_type", "GGA_GGA+U"))
