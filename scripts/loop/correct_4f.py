# ---------------------------------------------------------------------------
# correct_4f.py -- add the rare-earth 4f moment back before comparing to a real
# magnet.
#
# WHY THIS IS NEEDED
#   CHGNet predicts essentially ZERO moment on the rare earth: measured on this
#   campaign, the non-Fe share of m_sum is <=0.2% even for Dy, Tb and Sm. So
#   `J_s_max` from the screen is an Fe-SUBLATTICE number. That is fine for
#   ranking Fe-framework changes against each other, and WRONG the moment you
#   compare against a real magnet, because Nd2Fe14B's saturation includes Nd.
#
#   Consequence, which flips conclusions rather than shifting them:
#     LIGHT RE (Ce, Pr, Nd, Sm) couple PARALLEL to Fe   -> Fe-only UNDERestimates
#     HEAVY RE (Gd, Tb, Dy)     couple ANTIPARALLEL     -> Fe-only OVERestimates
#     Y, La have no 4f moment                            -> Fe-only is correct
#   So a Dy candidate that ties YFe12 on the Fe-only scale actually loses badly,
#   and Nd2Fe14B gains ~0.33 T that the screen never saw.
#
# HOW GOOD IS IT
#   EMPIRICAL, calibrated on the RE2Fe14B series against experimental moments
#   (see RE_MOMENT below), so by construction it reproduces those compounds.
#   The Y control lands at -0.14 muB, i.e. the Fe-only pipeline number is good
#   to ~0.5%.
#
#   THE REAL UNCERTAINTY IS THE EXTRAPOLATION, not the fit: RE_MOMENT is
#   calibrated in the 2:14:1 crystal field and applied to 1:12, 2:17 and 3:29,
#   which have different RE site symmetry and so different quenching. The
#   heavy-RE values should carry (Dy came out at -1.02x free ion, i.e. barely
#   quenched); the light-RE and especially the Ce value are the shakiest.
#   TREAT DIFFERENCES UNDER ~5% AS NOT RESOLVED [F1].
#
# INPUTS   --scored  a round's scored.csv (needs formula, m_Fe_sum_muB,
#                    volume_A3, natoms, E_hull_meV)
#          --top     how many distinct formulas to print (default 15)
# OUTPUTS  a table on stdout, ranked by CORRECTED J_s, with the benchmarks
# RUN      python scripts/loop/correct_4f.py --scored round-02/scored.csv
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv

from pymatgen.core import Composition

TESLA_PER_MUB_PER_A3 = 11.6541
# EMPIRICAL per-RE contribution in muB per RE ATOM, not free-ion gJ*J.
#
# Backed out as (experimental moment per f.u. - this pipeline's Fe-only moment
# per f.u.) / 2, over the RE2Fe14B series where experimental data exists. That
# is a far better number than the free-ion value, and for Ce it does not even
# have the same SIGN:
#
#   RE   Fe-only/fu   exp/fu   IMPLIED   free-ion   ratio
#   Y      31.69       31.4     -0.14      0.00      --     <- the control
#   Nd     32.76       37.7     +2.47      3.27     0.76
#   Pr     33.14       37.6     +2.23      3.20     0.70
#   Sm     32.41       33.3     +0.44      0.71     0.62
#   Ce     32.35       29.4     -1.47      2.54    -0.58    <- SIGN FLIP
#   Dy     31.72       11.3    -10.21     10.00    -1.02
#
# Three things this says, all of which change conclusions:
#  1. Y lands at -0.14, i.e. ~0. The Fe-only pipeline number is right to about
#     0.5% of the total, so the correction is the only thing being estimated.
#  2. The LIGHT rare earths are crystal-field quenched to ~0.6-0.76 of free ion.
#     Using free ion overstates Nd2Fe14B, the benchmark, by ~30%.
#  3. Ce CONTRIBUTES NEGATIVELY (-1.47). Ce in RE-Fe intermetallics is
#     valence-fluctuating toward non-magnetic Ce(IV) -- which is exactly WHY
#     Ce2Fe14B is the cheap, weaker magnet and Nd2Fe14B is the strong one.
#     Treating Ce as a normal light RE (+2.54) inflated every Ce candidate and
#     put CeFe12 falsely at the top of the shortlist. See the 4f-treatment trap.
#
# Tb and Gd have no pipeline run here; the heavy-RE pattern (Dy at -1.02x free
# ion) justifies -free_ion for them, flagged as estimated.
RE_MOMENT = {"Y": 0.0, "La": 0.0, "Ce": -1.47, "Nd": +2.47, "Pr": +2.23,
             "Sm": +0.44, "Dy": -10.21, "Tb": -9.00, "Gd": -7.00}
ESTIMATED = {"Tb", "Gd"}          # not calibrated against experiment here
GJ = {k: abs(v) for k, v in RE_MOMENT.items()}
SIGN = {k: (0 if v == 0 else (1 if v > 0 else -1)) for k, v in RE_MOMENT.items()}
# measured this session with the same MatterSim+CHGNet settings as the screen
BENCH = [("Nd2Fe14B", 131.04, 13.720 * 68, 68),
         ("Ce2Fe14B", 129.39, 13.470 * 68, 68)]


def corrected(formula: str, m_fe_only: float, volume: float, natoms: int):
    c = Composition(formula)
    scale = natoms / c.num_atoms          # formula units in the cell
    add, terms = 0.0, []
    for el, n in c.get_el_amt_dict().items():
        if RE_MOMENT.get(el):
            k = n * scale
            add += RE_MOMENT[el] * k
            star = "?" if el in ESTIMATED else ""
            terms.append(f"{el}x{k:.0f} {RE_MOMENT[el]:+.2f}{star}")
    return (TESLA_PER_MUB_PER_A3 * (m_fe_only + add) / volume, add,
            ", ".join(terms) or "no 4f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    def g(r, k):
        try:
            return float(r[k])
        except (TypeError, ValueError, KeyError):
            return None

    out = []
    for name, mfe, vol, na in BENCH:
        js, add, t = corrected(name, mfe, vol, na)
        out.append((js, name, TESLA_PER_MUB_PER_A3 * mfe / vol, add, None, t, True))

    seen = set()
    rows = [r for r in csv.DictReader(open(a.scored)) if g(r, "J_s_max_T")]
    rows.sort(key=lambda r: -(g(r, "J_s_max_T") or 0))
    for r in rows:
        f = r["formula"]
        if f in seen:
            continue
        mfe, vol = g(r, "m_Fe_sum_muB"), g(r, "volume_A3")
        if mfe is None or not vol:
            continue
        seen.add(f)
        js, add, t = corrected(f, mfe, vol, int(float(r["natoms"])))
        out.append((js, f, g(r, "J_s_max_T"), add, g(r, "E_hull_meV"), t, False))
        if len(seen) >= a.top:
            break

    nd = next(x[0] for x in out if x[1] == "Nd2Fe14B")
    out.sort(key=lambda x: -x[0])
    print(f"{'compound':<16}{'Fe-only':>9}{'4f':>8}{'CORRECTED':>11}"
          f"{'vs Nd2Fe14B':>12}{'E_hull':>8}  4f terms")
    print("-" * 88)
    for js, name, feonly, add, eh, t, is_bench in out:
        tag = "  <== BENCHMARK" if is_bench else ""
        d = js - nd
        mark = "" if is_bench else (" *" if d > 0.05 * nd else
                                    (" ~" if d > 0 else ""))
        eh_s = f"{eh:8.1f}" if eh is not None else " " * 8
        print(f"{name:<16}{feonly:9.4f}{add:+8.1f}{js:11.4f}{d:+12.4f}"
              f"{eh_s}  {t}{tag}{mark}")
    print()
    print("* clears Nd2Fe14B by more than the ~5% accuracy of this correction")
    print("~ above Nd2Fe14B but INSIDE that accuracy -- not a resolved difference")


if __name__ == "__main__":
    main()
