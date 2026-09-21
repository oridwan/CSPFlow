# ---------------------------------------------------------------------------
# hull.py -- the MLIP hull for a target chemical space, and the ONE number the
# screen is allowed to threshold on.
#
# WHY THE THRESHOLD IS PARENT-RELATIVE, AND NOT NEGOTIABLE
#   Measured 2026-09-15 on the Ce-Fe-B space built from this store:
#
#       Ce2Fe14B   mp-4459    40.6 meV/atom above the MatterSim hull
#       Ce2Fe17    mp-654     44.7
#       CeFe5      mp-11317   33.3
#
#   Ce2Fe14B is manufactured commercially. Ce2Fe17 is a real phase. MatterSim
#   puts both ~40 meV/atom above its own hull, because it is a non-magnetic
#   potential being asked about magnetic intermetallics and that error does not
#   cancel between a compound and its elemental references.
#
#   So an ABSOLUTE cutoff is meaningless here. "Keep everything under 50
#   meV/atom" would admit the parent by 9 meV and reject every derivative of it;
#   "under 30" would reject the parent itself. A gate that rejects the material
#   the campaign is built on is measuring its own constants, not the chemistry --
#   the same failure as the radius gate that once failed its own source CIF.
#
#   The screenable quantity is therefore
#
#       ddE_hull = E_hull(candidate) - E_hull(parent)
#
#   "how much less stable than the real compound", which is a question the
#   surrogate can answer because the systematic error largely cancels between
#   two structures of the same family on the same hull.
#
# ONE SCALE, ALWAYS
#   The hull is built from the store's OWN MatterSim energies (`e_mlip_relaxed`)
#   and candidates are relaxed with the same engine and model [E2]. Never mix in
#   MP's energies or ours from DFT -- see the store's three-hull rule.
#
# INPUTS   index.csv of the reference store; the target elements
# OUTPUTS  MlipHull: .e_above_hull(structure, energy) and .dd(...)
#
# RUN      python hull.py --elements Ce Fe B --parent <relaxed parent .vasp>
# ---------------------------------------------------------------------------
from __future__ import annotations

import csv
import os
from pathlib import Path

DEFAULT_INDEX = os.environ.get(
    "CSPFLOW_STORE_INDEX", "/scratch/oridwan/mp-reference/index.csv")


class MlipHull:
    """A phase diagram on the store's MatterSim scale, plus the parent offset."""

    def __init__(self, elements, index_csv: str | Path = DEFAULT_INDEX,
                 parent_e_above_hull: float | None = None):
        from pymatgen.core import Composition
        from pymatgen.entries.computed_entries import ComputedEntry
        from pymatgen.analysis.phase_diagram import PhaseDiagram

        self.elements = set(elements)
        self.index_csv = Path(index_csv)
        self.parent_e_above_hull = parent_e_above_hull
        self.entries, skipped = [], 0
        for r in csv.DictReader(open(self.index_csv)):
            if not r.get("e_mlip_relaxed") or not r.get("n_atoms"):
                continue
            els = set((r.get("chemsys") or "").split("-")) - {""}
            if not els or not els <= self.elements:
                continue
            try:
                # e_mlip_relaxed is the TOTAL energy of the STORED cell of
                # n_atoms; the formula column is the REDUCED formula. Pairing
                # them directly is a silent factor of (n_atoms / reduced atoms)
                # and produces a hull whose only vertices are the elements --
                # which is the sanity check below.
                epa = float(r["e_mlip_relaxed"]) / int(r["n_atoms"])
                comp = Composition(r["formula"])
                self.entries.append(
                    ComputedEntry(comp, epa * comp.num_atoms, entry_id=r["mp_id"]))
            except Exception:
                skipped += 1
        if not self.entries:
            raise SystemExit(
                f"no reference entries for {sorted(self.elements)} in {self.index_csv}. "
                f"Extend the store for this chemsys before screening on a hull.")
        self.pd = PhaseDiagram(self.entries)
        self.skipped = skipped
        self.vertices = sorted({e.composition.reduced_formula
                                for e in self.pd.stable_entries})

        # WHAT THIS CHECK IS FOR: pairing the store's REDUCED formula with the
        # whole-CELL energy is a silent factor of (n_atoms / reduced atoms). It
        # produces a plausible-looking hull whose only vertices are the elements.
        #
        # The first version of the check tested exactly that symptom -- "all
        # vertices are elements, therefore the entries are wrong" -- and it was
        # WRONG, in the same way the hull_incomplete cut was wrong: it inferred a
        # construction bug from a physical result. Measured 2026-09-15: Fe-Nd has
        # NO stable binary compound on the MatterSim hull. All 20 Nd-Fe phases lie
        # above the Fe + Nd tie-line, because MatterSim is non-magnetic and
        # destabilises Nd-Fe intermetallics (the same error that puts Nd2Fe14B 57
        # meV/atom up). An elements-only hull is therefore a legitimate answer for
        # this chemistry, and refusing it killed every Nd2Fe17 candidate.
        #
        # So test the CAUSE instead of the symptom: the mismatch makes energy per
        # atom too negative by a whole integer factor, and real values here are
        # -9..-4 eV/atom. A band check catches the bug without any assumption
        # about what nature should have made stable.
        bad = [e for e in self.entries
               if not (-25.0 < e.energy / e.composition.num_atoms < 5.0)]
        if bad:
            ex = bad[0]
            raise SystemExit(
                f"{len(bad)} reference entries have implausible energy per atom "
                f"(e.g. {ex.entry_id} {ex.composition.reduced_formula}: "
                f"{ex.energy / ex.composition.num_atoms:.2f} eV/atom). The entries "
                f"are almost certainly built with mismatched composition/energy. "
                f"Refusing to screen against this hull.")
        # Not an error -- recorded so a digest can say so out loud.
        self.elements_only = all(len(Composition(v).elements) == 1
                                 for v in self.vertices)

    def missing_subsystems(self, elements) -> list[str]:
        """Which sub-chemsys of `elements` the store has NO entry for.

        A hull only closes if every subsystem is populated. Adding one
        substituent adds a whole set of them -- rule [E1].

        IMPORTANT (measured 2026-09-15, against MP): for the RE quaternaries of
        this campaign every subsystem this reports as empty is ALSO empty in
        Materials Project -- mixed Ce+RE borides are absent from nature, not
        from our fetch. So this OVER-REPORTS: judge a hull by whether placement
        succeeds, and use `scripts/loop/verify_hulls.py --check-mp` to tell a
        store gap from a gap in nature.
        """
        import itertools
        have = {tuple(sorted(e.composition.chemical_system.split("-")))
                for e in self.entries}
        els = sorted(set(elements))
        missing = []
        for k in range(1, len(els) + 1):
            for c in itertools.combinations(els, k):
                if tuple(sorted(c)) not in have:
                    missing.append("-".join(sorted(c)))
        return missing

    def missing_elements(self) -> set:
        """Elements that appear in a sub-chemsys this hull has NO entry for.

        A candidate containing one of these is placed against a hull missing
        some of its competitors, so its E_hull can come out too LOW -- and as a
        plausible number, not a crash. NOTE (measured 2026-09-15): for the RE
        quaternaries every such subsystem is also empty in MP, i.e. absent from
        nature rather than from our fetch, so this is a FLAG and never a cut.
        """
        if getattr(self, "_missing_els", None) is None:
            els = sorted(self.elements)
            self._missing_els = set()
            for cs in self.missing_subsystems(els):
                self._missing_els |= set(cs.split("-"))
            self._missing_els -= {e for e in els
                                  if tuple([e]) in {tuple(sorted(x.composition.chemical_system.split("-")))
                                                    for x in self.entries}}
        return self._missing_els

    def e_above_hull(self, structure, energy: float) -> float | None:
        """energy is the TOTAL energy of `structure`, same engine as the store.

        Returns None when the candidate carries an element the hull cannot
        place -- a missing chemsys is a RESULT ("extend the store for this
        system" [E1]), not a crash that loses the other 115 seeds in the round.
        """
        from pymatgen.entries.computed_entries import ComputedEntry
        try:
            return float(self.pd.get_e_above_hull(
                ComputedEntry(structure.composition, energy)))
        except Exception:
            return None

    def dd(self, structure, energy: float) -> float | None:
        """ddE_hull: hull distance RELATIVE TO THE PARENT. The screenable one."""
        if self.parent_e_above_hull is None:
            return None
        return self.e_above_hull(structure, energy) - self.parent_e_above_hull

    def describe(self) -> str:
        n = len(self.entries)
        p = (f", parent sits {self.parent_e_above_hull*1000:.1f} meV/atom above it"
             if self.parent_e_above_hull is not None else "")
        return (f"MLIP hull for {'-'.join(sorted(self.elements))}: {n} store entries"
                f"{p}\n  vertices: {', '.join(self.vertices)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", nargs="+", required=True)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--parent", default=None, help="relaxed parent structure file")
    ap.add_argument("--parent-energy", type=float, default=None,
                    help="its MatterSim TOTAL energy; omit to relax it here")
    a = ap.parse_args()
    h = MlipHull(a.elements, a.index)
    print(h.describe())
    if a.parent:
        from pymatgen.core import Structure
        s = Structure.from_file(a.parent)
        e = a.parent_energy
        if e is None:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
            from cspflow.mlip import MatterSimEngine
            from pymatgen.io.ase import AseAtomsAdaptor
            r = MatterSimEngine().relax(AseAtomsAdaptor.get_atoms(s))
            e, s = r.energy, AseAtomsAdaptor.get_structure(r.atoms)
        d = h.e_above_hull(s, e)
        print(f"\nparent {s.composition.reduced_formula}: "
              f"E_hull = {d*1000:.1f} meV/atom")
        print(f"  -> screen candidates on ddE_hull = E_hull - {d*1000:.1f} meV/atom")
