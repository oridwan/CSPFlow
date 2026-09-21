# ---------------------------------------------------------------------------
# verify_hulls.py -- does a quaternary MLIP hull actually CLOSE?
#
# WHAT THIS CHECKS AND WHY
#   A phase diagram can only place a candidate if the store holds at least one
#   entry for EVERY sub-chemical-system of the candidate's elements. Add one
#   substituent to Ce2Fe14B and you do not add one system, you add eight: the
#   new element alone, its three binaries, its three ternaries, and the
#   quaternary. If even one of those is empty, pymatgen does not warn -- it
#   raises, and cspflow's screen reports a bare "Unable to get decomposition"
#   that looks like a broken candidate rather than a missing reference.
#
#   BUT an empty sub-chemsys is only a STORE gap if MP actually has entries for
#   it. Measured 2026-09-15: all 14 sub-systems this script flags as empty for
#   the seven RE substituents have ZERO entries in Materials Project -- mixed
#   Ce+RE borides are absent from nature, not from our fetch. The hull still
#   closes over them because the elemental vertices span the simplex. So the
#   verdict below is based on PLACEMENT, not on the emptiness list, and the
#   emptiness list is printed as information with --check-mp to resolve it.
#
#   So this script asks three questions per RE substituent:
#     1. which sub-chemsys are still empty (rule [E1]) -- informational;
#     2. what the hull's vertices are, and whether a compound is among them
#        (a hull of only elements means the entries were built wrong);
#     3. can the REAL RE2Fe14B compound in the store be PLACED on it, and
#        where does it sit relative to the Ce2Fe14B parent.
#
#   Question 3 is the one that matters: a hull that closes but misplaces a
#   manufactured magnet is not usable for screening.
#
# INPUTS
#   --index    reference-store index.csv   (default: $CSPFLOW_STORE_INDEX
#              or /scratch/oridwan/mp-reference/index.csv)
#   --res      RE substituents to test     (default: Y La Nd Pr Sm Dy Tb)
#   --base     the parent's elements       (default: Ce Fe B)
#
# OUTPUTS
#   A table on stdout, one row per substituent: entry count, empty subsystems,
#   number of hull vertices, and the placement of the RE 2:14:1 compound.
#   Exit code 1 only if a hull REFUSES to build or cannot place its own
#   2:14:1 compound. An empty sub-chemsys alone is not a failure.
#
# RUN
#   conda activate cspflow
#   python scripts/loop/verify_hulls.py
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hull import MlipHull, DEFAULT_INDEX  # noqa: E402


class _AsStructure:
    """hull.e_above_hull() reads `.composition`; a bare Composition has no
    such attribute, so the method's internal except swallowed it and
    returned None. This shim lets us place a formula with no coordinates."""

    def __init__(self, comp):
        self.composition = comp


def store_rows(index_csv):
    return list(csv.DictReader(open(index_csv)))


def find_2141(rows, re_el):
    """The RE2Fe14B entry in the store, if MP has one. Returns (mp_id, epa)."""
    from pymatgen.core import Composition
    want = Composition(f"{re_el}2Fe14B").reduced_composition
    for r in rows:
        if not r.get("e_mlip_relaxed") or not r.get("n_atoms"):
            continue
        try:
            c = Composition(r["formula"])
        except Exception:
            continue
        if c.reduced_composition == want:
            return r["mp_id"], float(r["e_mlip_relaxed"]) / int(r["n_atoms"])
    return None, None


def mp_gaps(chemsys, have_ids):
    """For each empty sub-chemsys: how many phases MP has, and which of those
    the store is missing. An empty system with 0 MP entries is not a gap."""
    import os

    import yaml
    from mp_api.client import MPRester
    key = os.environ.get("MP_API_KEY")
    if not key:
        d = yaml.safe_load(open(os.environ.get(
            "CSPFLOW_STORE_SETTINGS",
            "/projects/mmi/Ridwan/cspflow-reference/store-settings.yaml")))
        key = (d.get("reference") or {}).get("mp_api_key") or d.get("mp_api_key")
    out = []
    with MPRester(key) as m:
        for cs in chemsys:
            docs = m.materials.summary.search(
                chemsys=cs, fields=["material_id", "formula_pretty"])
            need = [d for d in docs if str(d.material_id) not in have_ids]
            out.append((cs, len(docs), need))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--base", nargs="+", default=["Ce", "Fe", "B"])
    ap.add_argument("--res", nargs="+",
                    default=["Y", "La", "Nd", "Pr", "Sm", "Dy", "Tb"])
    ap.add_argument("--check-mp", action="store_true",
                    help="ask Materials Project whether each empty sub-chemsys "
                         "is a store gap or simply absent from nature")
    a = ap.parse_args()

    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    rows = store_rows(a.index)

    # The parent's own offset, measured on the TERNARY hull it lives on.
    base_hull = MlipHull(a.base, a.index)
    p_id, p_epa = find_2141(rows, "Ce")
    parent_e = base_hull.e_above_hull(
        _AsStructure(Composition("Ce2Fe14B")), p_epa * Composition("Ce2Fe14B").num_atoms) \
        if p_epa is not None else None
    print(f"parent Ce2Fe14B ({p_id}) on the {'-'.join(sorted(a.base))} hull: "
          f"{parent_e*1000:.1f} meV/atom above it\n")

    hdr = (f"{'RE':<4} {'entries':>7} {'empty subsystems':<34} {'vertices':>8} "
           f"{'RE2Fe14B':<13} {'E_hull':>8} {'vs parent':>10}")
    print(hdr)
    print("-" * len(hdr))

    bad, empty = [], set()
    for re_el in a.res:
        els = list(a.base) + [re_el]
        try:
            h = MlipHull(els, a.index, parent_e_above_hull=parent_e)
        except SystemExit as e:
            print(f"{re_el:<4} REFUSED: {e}")
            bad.append(re_el)
            continue
        missing = h.missing_subsystems(els)
        mp_id, epa = find_2141(rows, re_el)
        if epa is None:
            place, eh, dd = "(not in MP)", "", ""
        else:
            comp = Composition(f"{re_el}2Fe14B")
            v = h.e_above_hull(_AsStructure(comp), epa * comp.num_atoms)
            if v is None:
                place, eh, dd = f"{mp_id} UNPLACED", "", ""
                bad.append(re_el)
            else:
                place = mp_id
                eh = f"{v*1000:7.1f}"
                dd = f"{(v - parent_e)*1000:+9.1f}"
        miss = ", ".join(missing) if missing else "-- none --"
        empty.update(missing)
        print(f"{re_el:<4} {len(h.entries):>7} {miss[:34]:<34} "
              f"{len(h.vertices):>8} {place:<13} {eh:>8} {dd:>10}")

    print()
    if empty:
        print(f"{len(empty)} sub-chemsys hold no store entry. These are only a "
              f"problem if MP has phases we failed to fetch:")
        if a.check_mp:
            gaps = mp_gaps(sorted(empty), {r["mp_id"] for r in rows})
            for cs, n_mp, need in gaps:
                tag = ("absent from nature too" if n_mp == 0 else
                       f"STORE GAP: {n_mp} in MP, {len(need)} not fetched")
                print(f"    {cs:<14} {tag}")
                if need:
                    bad.append(cs)
        else:
            print(f"    {', '.join(sorted(empty))}")
            print("    re-run with --check-mp to resolve store gap vs. absent "
                  "from nature")

    print()
    if bad:
        print(f"NOT usable for screening: {sorted(set(bad))}")
        return 1
    print("all hulls build and place their own 2:14:1 compound -> usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
