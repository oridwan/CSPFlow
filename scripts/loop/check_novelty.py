# ---------------------------------------------------------------------------
# check_novelty.py -- is each shortlisted structure already in Materials Project?
#
# WHY BOTH TESTS ARE NEEDED
#   Composition alone is not novelty: a seed can share a formula with an MP entry
#   and be a different polymorph (we carry two Y2Fe17, R-3m and P6_3/mmc, which
#   are distinct compounds with different J_s). Structure alone is not novelty
#   either, because a different composition cannot match anything. So this asks
#   two questions and reports them separately:
#     1. does MP have ANY entry at this reduced composition?
#     2. does MP have one that StructureMatcher calls the SAME structure?
#   The verdict is KNOWN only if both are yes.
#
#   This matters because part of this campaign's shortlist was built BY
#   substituting rare earths into MP prototypes, so some of it is by construction
#   already known -- and those are exactly the entries worth keeping as
#   calibration anchors rather than reporting as discoveries.
#
# INPUTS   --shortlist  CSV with columns file, formula, fingerprint
#          --structures directory of <fingerprint>.vasp (the loop's structures/)
#          --ltol --stol --angle-tol   StructureMatcher tolerances
# OUTPUTS  a table on stdout, and --out writes it as CSV with:
#          mp_same_structure, mp_same_composition, verdict, n_mp_at_composition
# RUN      conda activate cspflow
#          python scripts/loop/check_novelty.py \
#              --shortlist round-02/shortlist_4f_corrected.csv \
#              --structures structures --out round-02/novelty.csv
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import yaml
from mp_api.client import MPRester
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure


def api_key() -> str:
    k = os.environ.get("MP_API_KEY")
    if k:
        return k
    for p in ("/scratch/oridwan/mp-reference/settings.yaml",
              "/projects/mmi/Ridwan/cspflow-reference/store-settings.yaml"):
        if Path(p).is_file():
            d = yaml.safe_load(open(p))
            k = (d.get("reference") or {}).get("mp_api_key") or d.get("mp_api_key")
            if k:
                return k
    raise SystemExit("no MP API key found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--structures", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ltol", type=float, default=0.2)
    ap.add_argument("--stol", type=float, default=0.3)
    ap.add_argument("--angle-tol", type=float, default=5.0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.shortlist)))
    sm = StructureMatcher(ltol=a.ltol, stol=a.stol, angle_tol=a.angle_tol,
                          primitive_cell=True, attempt_supercell=True)
    sdir = Path(a.structures)
    out = []
    with MPRester(api_key()) as m:
        # one query per distinct chemical system, then filter by composition
        systems = {}
        for r in rows:
            cs = "-".join(sorted(Composition(r["formula"]).chemical_system.split("-")))
            systems.setdefault(cs, []).append(r)
        cache = {}
        for cs in systems:
            docs = m.materials.summary.search(
                chemsys=cs, fields=["material_id", "formula_pretty", "structure",
                                    "energy_above_hull", "symmetry"])
            cache[cs] = docs

        print(f"{'formula':<15}{'role':<10}{'MP@comp':>8}{'same struct':>13}  verdict")
        print("-" * 78)
        for r in rows:
            comp = Composition(r["formula"]).reduced_composition
            cs = "-".join(sorted(comp.chemical_system.split("-")))
            same_comp = [d for d in cache[cs]
                         if Composition(d.formula_pretty).reduced_composition == comp]
            st = Structure.from_file(sdir / f"{r['fingerprint']}.vasp")
            hit = None
            for d in same_comp:
                try:
                    if sm.fit(st, d.structure):
                        hit = d
                        break
                except Exception:
                    pass
            verdict = ("KNOWN" if hit else
                       ("new polymorph" if same_comp else "NEW composition"))
            out.append(dict(file=r["file"], formula=r["formula"],
                            role=r.get("role", ""),
                            n_mp_at_composition=len(same_comp),
                            mp_same_structure=(str(hit.material_id) if hit else ""),
                            mp_e_above_hull_meV=(round(1000 * hit.energy_above_hull, 1)
                                                 if hit else ""),
                            verdict=verdict))
            print(f"{r['formula']:<15}{r.get('role',''):<10}{len(same_comp):>8}"
                  f"{(str(hit.material_id) if hit else '-'):>13}  {verdict}")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"\nwrote {a.out}")
    import collections
    print("\n" + str(dict(collections.Counter(o["verdict"] for o in out))))


if __name__ == "__main__":
    main()
