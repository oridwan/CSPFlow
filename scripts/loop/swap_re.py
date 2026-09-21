# ---------------------------------------------------------------------------
# swap_re.py -- whole-sublattice rare-earth substitution on a prototype.
#
# WHAT IT DOES AND WHY IT EXISTS
#   The defect generator in the re-substitution-magnets skill answers "replace
#   ONE of the eight Ce", which is the dilute-defect question. This script
#   answers the other one: "take this prototype and build the WHOLE isostructural
#   RE series", e.g. Ce2Fe17 -> {La,Y,Nd,Pr,Sm,Dy,Tb,Gd}2Fe17. That is a complete
#   species replacement on every RE site, so the space group is preserved exactly
#   [A2] and no symmetry is broken -- the opposite of a defect seed.
#
#   It exists because rounds 2 and 3 of this campaign leave the 2:14:1 basin and
#   walk the Fe-density ladder (1:5 -> 2:14:1 -> 2:17 -> 3:29 -> 1:12). Those are
#   different stoichiometries in different space groups, so they cannot be
#   reached by perturbing one parent; each needs its own real prototype as the
#   starting geometry.
#
#   Cells are Vegard pre-scaled by the cube root of the summed atomic-radius
#   volume ratio, so a large RE does not start inside the contact distance of a
#   cell that was relaxed for a small one. This is the SUBSTITUTION case of the
#   prescale, where it is correct -- unlike the vacancy case, where removing an
#   atom must NOT shrink the cell.
#
# INPUTS
#   --protos      one or more prototype structure files (POSCAR/cif)
#   --res         rare earths to substitute in (default: the 9 with complete
#                 store coverage)
#   --outdir      where the seeds are written
#   --min-dist-frac  contact gate as a fraction of summed radii (default 0.70)
#   --skip-same   do not re-emit the prototype's own RE (default: emit it, so
#                 every family has its own internal reference point)
#
# OUTPUTS
#   <outdir>/<proto>__<RE>.vasp   one POSCAR per (prototype, RE)
#   <outdir>/manifest.csv         prototype, RE, formula, sg_before, sg_after,
#                                 n_atoms, Fe_at_pct, n_Fe_per_A3, min_dist,
#                                 scale, gate
#
# RUN
#   conda activate cspflow
#   python scripts/loop/swap_re.py \
#       --protos inputs/prototypes/*.vasp \
#       --outdir round-02/seeds/binary-re-series
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from pymatgen.core import Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# The rare earths whose RE-Fe and B-RE-Fe hulls are COMPLETE in the store
# (checked 2026-09-15), so every child can be placed without reference work.
DEFAULT_RES = ["Ce", "La", "Y", "Nd", "Pr", "Sm", "Dy", "Tb", "Gd"]
NON_RE = {"Fe", "B", "Co", "Ni", "Ti", "V", "Mo", "W", "Si", "Zr", "Nb", "Cr"}


def radius(el: str) -> float:
    e = Element(el)
    v = getattr(e, "atomic_radius", None) or getattr(e, "atomic_radius_calculated", None)
    return float(v) if v else 1.3


def re_sites(st: Structure) -> list[str]:
    """Which species in this prototype are the rare-earth sublattice."""
    return sorted({str(t.specie) for t in st if str(t.specie) not in NON_RE})


def min_contact(st: Structure) -> tuple[float, float]:
    """Shortest distance, and the worst distance/summed-radii ratio."""
    d_min, frac_min = 1e9, 1e9
    n = len(st)
    for i, j in itertools.combinations(range(n), 2):
        d = st.get_distance(i, j)
        if d < d_min:
            d_min = d
        lim = radius(str(st[i].specie)) + radius(str(st[j].specie))
        if lim and d / lim < frac_min:
            frac_min = d / lim
    return d_min, frac_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protos", nargs="+", required=True)
    ap.add_argument("--res", nargs="+", default=DEFAULT_RES)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-dist-frac", type=float, default=0.70)
    ap.add_argument("--skip-same", action="store_true")
    ap.add_argument("--symprec", type=float, default=0.1)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows, n_ok, n_gated = [], 0, 0

    for pf in a.protos:
        p = Path(pf)
        proto = Structure.from_file(p)
        host = re_sites(proto)
        if not host:
            print(f"!! {p.name}: no RE sublattice found, skipped")
            continue
        sg0 = SpacegroupAnalyzer(proto, symprec=a.symprec).get_space_group_symbol()
        v_old = sum(radius(str(t.specie)) ** 3 for t in proto)

        for re_el in a.res:
            if a.skip_same and host == [re_el]:
                continue
            st = proto.copy()
            # Every RE species in the prototype -> the single target RE. A
            # prototype with two distinct RE species collapses to one, which is
            # intended: this is the isostructural SERIES, not a mixed-RE seed.
            for h in host:
                st.replace_species({h: re_el})
            v_new = sum(radius(str(t.specie)) ** 3 for t in st)
            scale = (v_new / v_old) ** (1.0 / 3.0)
            st.scale_lattice(st.volume * scale ** 3)

            d_min, frac = min_contact(st)
            gate = "" if frac >= a.min_dist_frac else \
                f"[B1] contact {frac:.2f} < {a.min_dist_frac}"
            sg1 = SpacegroupAnalyzer(st, symprec=a.symprec).get_space_group_symbol()
            n_fe = sum(1 for t in st if str(t.specie) == "Fe")
            tag = f"{p.stem}__{re_el}"
            if gate:
                n_gated += 1
            else:
                st.to(filename=str(out / f"{tag}.vasp"), fmt="poscar")
                n_ok += 1
            rows.append(dict(
                seed=tag, prototype=p.stem, RE=re_el,
                formula=st.composition.reduced_formula,
                sg_before=sg0, sg_after=sg1, n_atoms=len(st),
                Fe_at_pct=round(100 * n_fe / len(st), 2),
                n_Fe_per_A3=round(n_fe / st.volume, 5),
                V_per_atom=round(st.volume / len(st), 3),
                min_dist=round(d_min, 3), contact_frac=round(frac, 3),
                scale=round(scale, 4), gate=gate))

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {n_ok} seeds ({n_gated} gated) -> {out}")
    print(f"manifest: {out/'manifest.csv'}")


if __name__ == "__main__":
    main()
