"""RETIRED 2026-09-12 (D126).  Not in the funnel; kept for the record.

**Why it is gone.**  4a compared MatterSim against MP's DFT.  That comparison
mattered because the hull that filtered candidates mixed the two: MatterSim
energies for the candidates, MP energies for the vertices.  The vertices are
now MatterSim as well (`reference_stage._vertices`), so both sides of that hull
carry the same model's zero and there is no longer a gap to measure.

Measured on the store before the change, the mixed hull misplaced roughly a
third of candidates in Ce chemistry by more than the 0.06 eV/atom selection
threshold (Ce-Ge-Pd 20/64, Ce-Cu-Ge 21/66, Ce-Ge-Sb 23/62, Fe-Pd 0/20).  The
fix was to remove the mixing, not to keep measuring it.

4b was already off by default (D119).  Everything below is the original text.

---

Stage 4a -- `calibrate:mp`, the free calibration.

Runs on MP phases the moment they are fetched, **before** the MLIP relaxes
anything, and needs nothing from the candidates. So it can run at the very start,
in parallel with generation: if MatterSim is bad at Sm you learn it in minutes
rather than after a GPU-day of MatterGen.

One forward pass per reference structure. For the whole 36-system magnet
campaign that is ~359 structures, seconds of GPU.

**What it does not do, and why.** It does not fit an energy correction. A
per-element correction is the only physically motivated form, and it moves
`e_above_hull` by ~10⁻¹⁵ eV/atom — exact cancellation along the tie-line,
measured on a real 41-phase Sm–Fe–Ti hull (D067). Fitting one would produce a
number that looks like a result and changes nothing.

**What it cannot replace.** Universal MLIPs of this class are trained on
MP-derived trajectories, so MP phases are close to in-distribution and 4a is
substantially a self-consistency check against MP's own numbers. It catches
gross failure, not subtle bias.

**Where 4b went (D119).** 4b compared the MLIP against OUR DFT on a pilot set
of generated candidates, and used to `block` the expensive tier until it
returned. It is now `off` by default, because the reference set carries both
MatterSim and our own DFT for 2,711 phases — so that comparison is available
before a campaign starts, on 68x the sample 4b's default 40 would have given
(MAE 0.0355 eV/atom, Spearman 0.970 on formation energy).

What that does not buy is the out-of-distribution question: those 2,711 are MP
phases, and the structures a campaign cares about are *generated*, often in
prototypes absent from MP. 4b was the only thing that tested the model there,
so `calibrate: {pilot: {on_fail: block}}` is still the right setting for a
chemistry the reference set does not cover.
"""

from __future__ import annotations

from ..calibrate.parity import ParityPoint, build_report, fit_threshold
from ..config.loader import ResolvedConfig
from ..calibrate.parity import ParityPoint
from ..calibrate.pilot import build_pilot_report, stratified_sample
from ..db.store import Store, StructureState
from ..reference.mp import ReferenceError, fetch_structures
from .base import StageReport
from .dedup_stage import CHECKED_KEY

# Marks a structure as a member of the pilot set. Kept on the structure row
# rather than in a side table so that `csp status --why` shows it, and so that
# the DFT stage can let a pilot member through a gate that is holding everything
# else back.
PILOT_KEY = "pilot"


class CalibrateStage:
    name = "calibrate"
    role = "gpu"
    in_process = True

    def __init__(self, cfg: ResolvedConfig, engine=None) -> None:
        self.cfg = cfg
        self._engine = engine
        self.last_report = None
        self.last_pilot = None

    @property
    def engine(self):
        if self._engine is None:
            from ..mlip import for_config

            self._engine = for_config(self.cfg.campaign.screen)
        return self._engine

    def pending(self, store: Store) -> int:
        """Work in either half: 4a's single points, or 4b's pilot set."""
        row = store.sql.execute(
            "SELECT COUNT(*) n FROM reference_entry WHERE e_mlip_static IS NULL"
        ).fetchone()
        outstanding = int(row["n"]) if row else 0
        return outstanding + (1 if self._pilot_has_work(store) else 0)

    def _pilot_has_work(self, store: Store) -> bool:
        """True when 4b could either select a pilot set or judge a finished one."""
        if self.cfg.campaign.calibrate.pilot.on_fail == "off":
            return False
        members = _pilot_members(store)
        if not members:
            # Nothing selected yet, and something to select from.
            return bool(_selectable(store))
        if store.latest_calibration("pilot") is not None:
            return False
        return all(row.key_value_pairs.get("state") == StructureState.dft_done.value
                   for row in members)

    def claim(self, store, budget):                        # pragma: no cover
        raise AssertionError("calibrate is an in-process stage; the driver calls run()")

    def build(self, items, workdir):                       # pragma: no cover
        raise AssertionError("calibrate is an in-process stage; nothing is submitted")

    def reconcile(self, store, job_row, status, items):    # pragma: no cover
        pass

    def run(self, store: Store) -> StageReport:
        """4a then 4b.  They are one stage because they answer one question.

        4a is free and diagnostic; 4b costs pilot DFT and is the gate. Running
        them in one place means a campaign cannot accidentally have one without
        the other, and `csp status` shows both verdicts side by side.
        """
        report = self._run_mp(store)
        pilot_note = self._run_pilot(store)
        if pilot_note:
            report.note = f"{report.note}; 4b {pilot_note}" if report.note else f"4b {pilot_note}"
        report.pending = self.pending(store)
        return report

    # -- 4b --------------------------------------------------------------

    def _run_pilot(self, store: Store) -> str:
        """Select a pilot set, or judge one that has come back."""
        pilot = self.cfg.campaign.calibrate.pilot
        if pilot.on_fail == "off":
            return ""

        members = _pilot_members(store)
        if not members:
            chosen = self._select_pilot(store, pilot.pilot_n)
            if not chosen:
                return ""
            return (f"pilot set of {len(chosen)} selected and sent to DFT "
                    f"(spread across the MLIP hull range, not the top {len(chosen)})")

        if store.latest_calibration("pilot") is not None:
            return ""

        outstanding = [r for r in members
                       if r.key_value_pairs.get("state") != StructureState.dft_done.value]
        if outstanding:
            return f"waiting on {len(outstanding)} of {len(members)} pilot DFT jobs"

        return self._judge_pilot(store, members)

    def _select_pilot(self, store: Store, n: int) -> list[int]:
        """Take `n` screened candidates spread across the MLIP hull range."""
        rows = _selectable(store)
        if not rows:
            return []
        labels = [str(int(r.id)) for r in rows]
        values = [float(r.key_value_pairs.get("e_above_hull_mlip",
                                              r.key_value_pairs.get("mlip_e_per_atom", 0.0)))
                  for r in rows]
        chosen = [int(label) for label in stratified_sample(labels, values, n)]
        for sid in chosen:
            store.set_structure_state(sid, StructureState.selected, **{PILOT_KEY: True})
            store.add_filter_event(structure_id=sid, gate="calibrate:pilot",
                                   passed=True,
                                   detail="selected for the pilot DFT set")
        return chosen

    def _judge_pilot(self, store: Store, members) -> str:
        points = []
        for row in members:
            kv = row.key_value_pairs
            mlip = kv.get("mlip_e_per_atom")
            dft = kv.get("e_per_atom")
            if mlip is None or dft is None:
                continue
            counts: dict[str, int] = {}
            for symbol in row.toatoms().get_chemical_symbols():
                counts[symbol] = counts.get(symbol, 0) + 1
            points.append(ParityPoint(
                label=str(int(row.id)), counts=counts,
                e_mlip_per_atom=float(mlip), e_dft_per_atom=float(dft),
                e_hull_mlip=kv.get("e_above_hull_mlip"),
                e_hull_dft=kv.get("dft_e_above_hull"),
            ))

        if not points:
            return ("pilot DFT finished but no structure has both an MLIP and a "
                    "DFT energy; nothing to compare")

        thresholds = self.cfg.campaign.calibrate.pilot.thresholds
        report = build_pilot_report(
            points,
            mae_max=thresholds.mae_e_per_atom,
            mae_hull_max=thresholds.mae_e_hull,
            spearman_min=thresholds.spearman_min,
            top_n=max(1, len(points) // 4),
        )
        self.last_pilot = report
        store.add_calibration(
            kind="pilot", n_points=report.n, verdict=report.verdict,
            mae_e_per_atom=report.parity.mae_e_per_atom,
            spearman=report.parity.spearman,
            volume_drift=None,
            detail=report.render()[:4000],
        )
        note = (f"{report.verdict.upper()} on {report.n}: "
                f"MAE {report.parity.mae_e_per_atom * 1000:.0f} meV/atom")
        if report.top_n:
            note += f", top-{report.top_n} recovered {report.top_n_overlap}/{report.top_n}"
        return note

    # -- 4a --------------------------------------------------------------

    def _run_mp(self, store: Store) -> StageReport:
        rows = [r for r in store.reference_entries() if r["e_mlip_static"] is None]
        if not rows:
            return StageReport(stage=self.name, note="every reference entry already has one")

        structures = self._structures(store)
        points, evaluated, missing = [], 0, 0

        for row in rows:
            structure = structures.get(row["mp_id"])
            if structure is None:
                missing += 1
                continue
            atoms = _to_atoms(structure)

            # Single point AT MP'S GEOMETRY. Relaxing first would compare
            # E_MLIP(x_MLIP) against E_DFT(x_MP) and fold the geometry error
            # into the energy error irreversibly (D069).
            static = self.engine.single_point(atoms)
            if not static.ok:
                store.update_reference_entry(int(row["id"]), state="failed",
                                             fail_reason=static.error[:200])
                continue

            relaxed = self.engine.relax(atoms) if self.cfg.campaign.reference.relax_with_mlip else None

            store.update_reference_entry(
                int(row["id"]),
                e_mlip_static=static.e_per_atom,
                e_mlip_relaxed=relaxed.e_per_atom if relaxed and relaxed.ok else None,
                volume_drift=relaxed.volume_drift if relaxed and relaxed.ok else None,
                state="static_done" if relaxed is None else "relaxed",
            )
            evaluated += 1

            counts: dict[str, int] = {}
            for symbol in atoms.get_chemical_symbols():
                counts[symbol] = counts.get(symbol, 0) + 1
            points.append(ParityPoint(
                label=row["mp_id"], counts=counts,
                e_mlip_per_atom=static.e_per_atom,
                e_dft_per_atom=row["e_dft_raw"],
                volume_mlip=relaxed.volume_after if relaxed and relaxed.ok else None,
                volume_dft=relaxed.volume_before if relaxed and relaxed.ok else None,
            ))

        if not points:
            return StageReport(stage=self.name, claimed=0,
                               note=f"no usable points ({missing} structures unavailable)")

        thresholds = self.cfg.campaign.calibrate.mp.thresholds
        report = build_report(
            points,
            mae_max=thresholds.mae_e_per_atom,
            # 4a has no DFT hull to compare against -- that is 4b's job -- so the
            # hull bound is left wide here rather than being invented.
            mae_hull_max=float("inf"),
            spearman_min=thresholds.spearman_min,
            volume_drift_max=thresholds.max_volume_drift,
        )
        self.last_report = report

        store.add_calibration(
            kind="mp", n_points=report.n, verdict=report.verdict,
            mae_e_per_atom=report.mae_e_per_atom,
            spearman=report.spearman,
            volume_drift=report.mean_volume_drift,
            detail=report.render()[:4000],
        )

        note = f"{report.verdict.upper()}: MAE {report.mae_e_per_atom * 1000:.0f} meV/atom"
        if report.spearman is not None:
            note += f", Spearman {report.spearman:.3f}"
        if missing:
            note += f"; {missing} structure(s) unavailable"
        return StageReport(stage=self.name, claimed=evaluated,
                           reconciled=report.n, note=note)

    def _structures(self, store: Store) -> dict:
        out: dict = {}
        for chemsys in store.chemsystems():
            try:
                out.update(fetch_structures(chemsys))
            except (ReferenceError, Exception):
                continue
        return out


def _to_atoms(structure):
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(structure)


def _pilot_members(store: Store) -> list:
    """Every structure marked as part of the pilot set."""
    return [row for row in store.structures()
            if row.key_value_pairs.get(PILOT_KEY)]


def _selectable(store: Store) -> list:
    """Deduplicated candidates a pilot set could be drawn from.

    Deduplicated ones only, for two reasons. A pilot containing three settings
    of one structure measures the model once and spends the DFT three times.
    And the selection is meant to *span* the hull range -- which it cannot do
    while dedup is still working, because the range it spans is then whatever
    happened to be processed first.

    So this returns nothing at all while anything is still waiting to be
    deduplicated. The pilot is a one-shot decision that gates the whole
    expensive tier; taking it against a partial set to save one cycle is a poor
    trade.
    """
    rows = [row for row in store.structures(state=StructureState.screened.value)
            if row.key_value_pairs.get("mlip_e_per_atom") is not None]
    if any(not row.key_value_pairs.get(CHECKED_KEY) for row in rows):
        return []
    return rows
