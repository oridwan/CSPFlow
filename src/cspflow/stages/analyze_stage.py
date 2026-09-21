"""Stage 7 -- properties, and the hull on our own energies.

An in-process stage.  Everything it does is arithmetic over finished jobs, so it
runs in the driver in seconds and needs no queue.

Two jobs, and they are separate on purpose.

**Properties** are extracted per structure from its own DFT directory: relaxed
volume, spacegroup at a stated tolerance, the cell magnetisation, the
sublattice split, and -- as a distinct column -- the Hund's-rule reconstruction
of the saturation magnetisation.  See `analysis/hund.py` for why the last one
cannot be the same column as the first.

**The DFT hull** is recomputed for a chemical system whenever a new DFT result
lands in it, so `e_above_hull` is never stale.  This is the one number in the
campaign that is not a property of a structure: it depends on every competing
phase in the same system, including other candidates of ours.  A candidate
placed against a hull that was missing a phase computed an hour later is simply
wrong, and re-placing is cheap.

The hull is built on one energy scale, and `reference.mode` says which:

    recompute     (default) our own recomputed reference phases, read from the
                  shared cache by `recipe_id`.  One scale throughout.  If any
                  phase in a system is missing, that system is REFUSED and
                  reported -- never topped up from MP, because a hull with one
                  borrowed vertex still builds and looks exactly like a correct
                  one.
    mp_energies   MP's own numbers.  Legitimate for a pre-screen or a new
                  chemistry where nothing has been recomputed yet, but it puts
                  two absolute scales on one hull, so it says so on the report.

Mixing our GGA numbers with MP's *corrected* ones is a third thing again, and
`reference/hull.py` refuses it outright whatever the mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..analysis.properties import extract
from ..chem import canonical_formula, chemsys
from ..config.loader import ResolvedConfig
from ..db.store import Store, StructureState
from ..reference.computed import ComputedError
from ..reference.hull import Entry, HullError, build_hull
from .base import StageReport, WorkItem

# Where the DFT stage recorded the directory it wrote.
DIR_KEY = "dft_dir"
# Set once a structure's properties have been extracted, so a second cycle does
# not re-read a 10 MB OUTCAR for every finished structure in the campaign.
DONE_KEY = "analyzed"

# Said once per run, not once per structure -- and now only on
# `reference.mode: mp_energies`, which is the explicit choice to accept the
# mixing this describes.  Until D101 was wired up (see `_reference_entries`)
# `mode` decided nothing and every hull mixed scales regardless of the config.
#
# Measured 2026-08-27 by running four MP structures through this campaign's own
# settings, as a static at MP's own geometry:
#
#     Fe (mp-13)      ours - MP  +0.2068 eV/atom
#     Sm2Fe17         +0.1870      SmFe11Ti  +0.1794      SmFe2  +0.1530
#
# A three-element least squares fits all four to within 1.8 meV/atom with
# Fe +0.2056, Sm +0.0462, Ti +0.0247 eV per atom of that element -- a clean
# per-element offset. It cancels exactly in a hull built on ONE scale (D067) and
# not at all in one built on two: our candidate carries its offset and MP's
# vertices do not, so every e_above_hull is inflated by the candidate's own.
#
# The check that settles it: Sm2Fe17 in this campaign is mp-1426 exactly (R-3m,
# volume within 0.17%, RMS 0.002 A). It was reported at 0.2098 eV/atom above the
# hull. Subtract its measured offset and it is +0.023 -- on the hull, where the
# reference phase must be.
MIXED_SCALE_REFUSAL = (
    "no DFT hull for this system: reference.mode='mp_energies' would put our "
    "DFT energies on MP's vertices, and the two are not the same scale. "
    "Measured on the store 2026-09-12, a per-element correction fitted inside "
    "one chemistry still leaves 42 meV/atom RMS for Ce-Ge-Pd and 103 for "
    "Ce-Fe-B, against a 60 meV/atom selection threshold -- elemental Ce alone "
    "is +1.17 eV/atom ours-minus-MP, a different 4f POTCAR. Rank on the MLIP "
    "hull (whole, one scale), or add this system to $CSPFLOW_STORE and set "
    "reference.mode='recompute'. See DECISIONS.md D101, D126."
)

SCALE_WARNING = (
    "e_above_hull mixes our DFT with MP's on one hull. Measured on four MP "
    "structures through these settings, that is a per-element offset near "
    "+0.2 eV per Fe atom, which inflates every value by the candidate's own "
    "share of it -- see DECISIONS.md D101")


class AnalyzeStage:
    name = "analyze"
    role = "cpu"
    in_process = True

    def __init__(self, cfg: ResolvedConfig, recipe: Any = None) -> None:
        self.cfg = cfg
        self._recipe = recipe
        self._rid: str | None = None

    @property
    def recipe_id(self) -> str:
        """The policy hash of this campaign's DFT settings.

        No longer used to FIND reference energies -- those come from the store
        folder now (`_computed_entries`).  It is kept because it is still the
        honest answer to "were these two runs computed the same way", which
        `csp doctor` reports and the reference-build commands key their cache
        on.  Loaded lazily: nothing on the hull path asks for it.
        """
        if self._rid is None:
            from ..dft.recipe import load_recipe
            from ..reference.computed import recipe_id

            recipe = self._recipe or load_recipe(
                self.cfg.campaign.dft.recipe, self.cfg.base_dir)
            self._rid = recipe_id(self.cfg.campaign.dft, recipe)
        return self._rid

    # -- what is ready -----------------------------------------------------

    def pending(self, store: Store) -> int:
        return len(self._ready(store))

    @staticmethod
    def _ready(store: Store) -> list[Any]:
        return [row for row in store.structures(state=StructureState.dft_done.value)
                if not row.key_value_pairs.get(DONE_KEY)]

    def claim(self, store: Store, budget: int) -> list[WorkItem]:  # pragma: no cover
        raise AssertionError("analyze is an in-process stage; the driver calls run()")

    def build(self, items, workdir):                               # pragma: no cover
        raise AssertionError("analyze is an in-process stage; nothing is submitted")

    def reconcile(self, store, job_row, status, items) -> None:    # pragma: no cover
        pass

    # -- the work ----------------------------------------------------------

    def run(self, store: Store) -> StageReport:
        extracted, problems = self._extract_all(store)
        systems, hull_note = self._place_all(store)

        note = f"{len(systems)} chemical system(s) placed"
        if hull_note:
            note += f"; {hull_note}"
        if problems:
            note += f"; {len(problems)} structure(s) with warnings"
        return StageReport(stage=self.name, claimed=extracted, reconciled=len(systems),
                           note=note, pending=self.pending(store))

    def _extract_all(self, store: Store) -> tuple[int, list[str]]:
        treatment = self._f_treatment()
        done, problems = 0, []
        for row in self._ready(store):
            sid = int(row.id)
            directory = row.key_value_pairs.get(DIR_KEY)
            if not directory or not Path(directory).is_dir():
                store.update_structure(
                    sid, **{DONE_KEY: True,
                            "analyze_note": f"no DFT directory recorded"[:200]})
                problems.append(f"{sid}: no DFT directory")
                continue

            # No `z` argument: it is derived from the cell. The row's own `z`
            # key was read here and written nowhere, so every per-formula-unit
            # moment was divided by 1 -- the cell value under another name.
            props = extract(Path(directory), structure_id=sid,
                            f_treatment=treatment)
            kv = props.as_kv()
            kv[DONE_KEY] = True
            if props.spacegroup_symbol:
                kv["spacegroup_symbol"] = props.spacegroup_symbol
                kv["symprec"] = props.symprec
            if props.warnings:
                kv["analyze_note"] = "; ".join(props.warnings)[:200]
                problems.append(f"{sid}: {props.warnings[0]}")
            # The per-site table goes in `data`, not in key-values: it is a
            # list of dicts, and it is what the report's structure cards draw
            # when the scratch directory has been cleaned away.
            blob = props.as_data()
            if blob:
                store.update_structure(sid, data=blob, **kv)
            else:
                store.update_structure(sid, **kv)

            for key, value in (("m_dft_raw", props.m_dft_raw),
                               ("m_s_reconstructed", props.m_s_reconstructed),
                               ("m_spheres", props.m_spheres),
                               ("volume", props.volume),
                               ("spacegroup", props.spacegroup_number)):
                if value is not None:
                    store.add_property(structure_id=sid, key=key, source="dft",
                                       value=float(value))
            for name, value in props.sublattice.items():
                store.add_property(structure_id=sid, key=f"m_{name}", source="dft",
                                   value=float(value))
            done += 1
        return done, problems

    def _f_treatment(self) -> str:
        """Which 4f treatment the DFT used, and whether to reconstruct at all.

        `rare_earth.reconstruct_ms: false` was in the shipped template and read
        by nothing, so a campaign that asked for the raw number only got the
        Hund's-rule column anyway. Returning a treatment other than `frozen`
        is how `reconstruct` is told not to add anything -- it already refuses
        there, and for the same reason.
        """
        dft = getattr(self.cfg.campaign, "dft", None)
        rare_earth = getattr(dft, "rare_earth", None) if dft else None
        if rare_earth is not None and not getattr(rare_earth, "reconstruct_ms", True):
            return "no_reconstruction"
        treatment = getattr(rare_earth, "f_treatment", None) if rare_earth else None
        return getattr(treatment, "value", treatment) or "frozen"

    # -- the hull ----------------------------------------------------------

    def _place_all(self, store: Store) -> tuple[list[str], str]:
        """Rebuild every chemical system that has a DFT energy in it."""
        ours = self._our_entries(store)
        if not ours:
            return [], ""

        placed: list[str] = []
        notes: list[tuple[str, str]] = []
        on_mp = self.cfg.campaign.reference.mode == "mp_energies"
        for system, entries in sorted(ours.items()):
            if on_mp:
                # Our DFT candidate against MP's vertices is two scales on one
                # hull.  It used to be allowed with a warning; it is refused
                # now, because the warning was not something a reader could act
                # on and the error is larger than the threshold it feeds.
                notes.append((system, MIXED_SCALE_REFUSAL))
                self._mark_absent(store, entries, MIXED_SCALE_REFUSAL)
                continue
            try:
                reference = self._reference_entries(store, system)
            except ComputedError as exc:
                # An incomplete system leaves `dft_e_above_hull` EMPTY rather
                # than filled from MP -- the campaign still ranks on the MLIP
                # hull, which is whole.  The reason is recorded per structure so
                # a blank in the report is explainable instead of mysterious.
                notes.append((system, str(exc)))
                self._mark_absent(store, entries, str(exc))
                continue
            try:
                result = build_hull([*reference, *entries])
            except HullError as exc:
                notes.append((system, str(exc)))
                continue
            for entry in entries:
                if entry.structure_id is None:
                    continue
                store.update_structure(
                    int(entry.structure_id),
                    dft_e_above_hull=float(result.e_above_hull[entry.label]),
                    dft_e_formation=float(result.formation_energy[entry.label]),
                )
                store.add_property(structure_id=int(entry.structure_id),
                                   key="dft_e_above_hull", source="dft",
                                   value=float(result.e_above_hull[entry.label]))
            placed.append(system)

        return placed, _summarise(notes)

    @staticmethod
    def _mark_absent(store: Store, entries, reason: str) -> None:
        """Say WHY a structure has no DFT hull distance, rather than leaving a hole.

        `dft_e_above_hull` stays NULL.  A reader who sees the blank can ask the
        database what happened, and `csp report` prints it in the column instead
        of an empty cell.
        """
        for entry in entries:
            if entry.structure_id is None:
                continue
            store.add_property(structure_id=int(entry.structure_id),
                               key="dft_e_above_hull_absent", source="dft",
                               text_value=reason)

    @staticmethod
    def _our_entries(store: Store) -> dict[str, list[Entry]]:
        """Our own finished DFT results, grouped by chemical system."""
        out: dict[str, list[Entry]] = {}
        for row in store.structures(state=StructureState.dft_done.value):
            energy = row.key_value_pairs.get("vasp_energy")
            if energy is None:
                continue
            counts: dict[str, int] = {}
            for symbol in row.toatoms().get_chemical_symbols():
                counts[symbol] = counts.get(symbol, 0) + 1
            entry = Entry(label=f"ours-{row.id}", counts=counts, energy=float(energy),
                          scale="raw", source="ours", run_type="GGA",
                          structure_id=int(row.id))
            out.setdefault(chemsys(counts), []).append(entry)
        return out

    def _reference_entries(self, store: Store, system: str) -> list[Entry]:
        """The hull's reference vertices, from whichever scale the config names.

        This is where `reference.mode` finally does something (D101).  It had
        defaulted to `recompute` and been wired to nothing, so every DFT hull
        was built from our energies against MP's whatever the config said --
        two absolute scales, and the measured gap between them is +0.15 to
        +0.21 eV/atom, against a 0.06 eV/atom selection threshold.

            recompute     our own recomputed phases, from the shared cache.
                          Refuses if any phase is missing, rather than filling
                          the gap from MP -- see `_computed_entries`.
            mp_energies   MP's own numbers, and `SCALE_WARNING` on the report.
                          An explicit, recorded choice to accept the mixing.

        Stage 3's MLIP pre-screen is a different hull and stays on MP's
        energies, which is right: MatterSim is trained on MPtrj, so MP's scale
        is the one its numbers belong on, and the pre-screen is a wide cut
        rather than the number anything is finally selected on.

        It stays there unconditionally, though -- `reference.prescreen_mode`
        exists in the schema and, like `mode` before D117, is read by nothing.
        That is the same class of fault as D101 and is not fixed here; it is
        recorded so the next person finds a note rather than a surprise.
        """
        if self.cfg.campaign.reference.mode == "recompute":
            return self._computed_entries(system)
        return self._mp_entries(store, system)

    def _computed_entries(self, system: str) -> list[Entry]:
        """Our own recomputed reference phases, read from the store FOLDER.

        The store at `$CSPFLOW_STORE` is the living source of truth and it
        keeps being extended, so the hull is built by reading it -- not from an
        exported `computed/<recipe_id>/` copy.  A second copy of the same
        numbers goes stale: measured 2026-09-11, that cache held 2,713 DFT
        energies while the store held 3,408, and it was keyed to a policy the
        store had already moved past. There is no export step now, and no
        recipe_id.

        `entries_for` still REFUSES on partial coverage, which is the whole
        point: a hull missing one vertex still builds and nothing downstream
        can tell it apart from a complete one. Phases in the store's
        `ignored.json` do not count as missing -- they sit too far above the
        hull to be vertices, so waiting for them blocks a campaign for nothing.

        `reference.energy_source` picks the scale: "dft" (ours, the default),
        "mlip" (MatterSim as we ran it), or "mp" (MP's own numbers, a different
        scale that must never be mixed into either -- see D101).
        """
        from ..reference.refstore import entries_for

        source = getattr(self.cfg.campaign.reference, "energy_source", "dft")
        return entries_for(system, source)

    @staticmethod
    def _mp_entries(store: Store, system: str) -> list[Entry]:
        """MP entries for `system` and every sub-system of it.

        Sub-systems are not optional: a binary query returns the binaries and
        no elemental end members, and a hull without its elemental references is
        not a hull. `reference_entries(include_subsystems=True)` is where that
        expansion already lives.
        """
        entries = []
        for row in store.reference_entries(chemsys=system, include_subsystems=True):
            counts = _counts_from_formula(row["formula"])
            per_atom = row["e_dft_raw"]
            if per_atom is None or not counts:
                continue
            # `e_dft_raw` is stored per atom; `Entry.energy` is the total for
            # `counts`. Scaling by the parsed formula's own atom count rather
            # than by the stored `n_atoms` keeps the two self-consistent even
            # when the stored formula is the reduced one.
            entries.append(Entry(label=f"mp-{row['id']}", counts=counts,
                                 energy=float(per_atom) * sum(counts.values()),
                                 scale="raw", source="mp",
                                 run_type=row["run_type"] or "GGA"))
        return entries


def _summarise(notes: list[tuple[str, str]]) -> str:
    """Collapse one reason repeated across many systems into one line.

    The hull guards explain themselves at length, which is right the first time
    and noise the fourth: four chemical systems missing their elemental
    references produced four identical paragraphs on one cycle line.
    """
    if not notes:
        return ""
    grouped: dict[str, list[str]] = {}
    for system, reason in notes:
        grouped.setdefault(_short(reason), []).append(system)
    out = []
    for reason, systems in grouped.items():
        listed = ", ".join(sorted(systems)[:3])
        if len(systems) > 3:
            listed += f" and {len(systems) - 3} more"
        out.append(f"{len(systems)} system(s) not placed ({listed}): {reason}")
    return "; ".join(out)


def _short(reason: str) -> str:
    """The first sentence of a guard's message; the rest is its explanation."""
    head = reason.split(". ")[0].strip()
    # Drop the element list and the entry count, so systems that differ only in
    # which elements they are missing group into one line. The systems
    # themselves are named by the caller.
    head = head.split(" among ")[0]
    bracket = head.find("[")
    head = head[:bracket].strip() if bracket > 0 else head
    # A message cut before its list can end on a dangling preposition.
    while head.split() and head.split()[-1] in {"for", "in", "of", "among", "at"}:
        head = head.rsplit(" ", 1)[0]
    return head


def _counts_from_formula(formula: str) -> dict[str, int]:
    from ..chem import parse_formula

    try:
        return parse_formula(formula)
    except Exception:                                        # pragma: no cover
        return {}
