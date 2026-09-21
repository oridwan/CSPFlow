# ---------------------------------------------------------------------------
# round.py -- ONE deterministic round of the generate/screen/keep loop.
#
# WHERE THIS SITS
#   The loop is: PROPOSE (an agent) -> GENERATE (toolbox scripts) -> [THIS] ->
#   PROMOTE (DFT) -> LEARN -> propose again. Everything in this file is
#   deterministic and re-runnable: given the same seeds and the same archive it
#   produces the same answer. The agent is NOT in here, on purpose -- a scoring
#   step that an agent can influence is a scoring step nobody can reproduce.
#
# THE STEPS IT OWNS  (3-10 of the loop)
#   3  GATE      geometry [B2][B3][B4] -- free, before any model runs
#   4  DEDUP     StructureMatcher against the WHOLE archive, not just this round
#   5  RELAX     MatterSim  (~24 s per 68 atoms, CPU)
#   6  SCREEN-E  ABSOLUTE E_hull on the MatterSim hull of the candidate's
#   7  SCREEN-M  CHGNet -> J_s_max                  (~3-10 s)
#   8  CLASSIFY  Type I/II/III and the alpha-Fe flag [G2][G12]
#   9  SELECT    PARETO front on (E_hull, J_s_max) + a full J_s ranking
#  10  LEARN     which descriptor actually tracks J_s -- the digest the agent reads
#
# WHY PARETO AND NOT A SINGLE SCORE
#   Scalarising ("0.7*J_s - 0.3*dE") hides the trade-off and hands the search a
#   number it can game: the weights decide the answer before the physics does.
#   The Pareto front on (E_hull, J_s_max) keeps both axes visible, is what the
#   paper plots anyway, and cannot be won by being merely cheap on one axis.
#
# WHY J_s IS CALLED J_s_max
#   CHGNet predicts |m|, magnitudes with no sign, so the sum is the SATURATION
#   value assuming every site aligns -- an upper bound. A ferrimagnetic
#   candidate reads HIGH, which means a search rewarded for J_s drifts towards
#   exactly the structures the surrogate gets wrong. The bound is honest; a
#   point estimate would not be. Ordering stays a DFT question [D3].
#
# INPUTS
#   seeds           files/globs to screen this round
#   --parent        the parent structure (sets ddE_hull's zero, and Type I/II/III)
#   --archive       loop archive directory (created on first use)
#   --round         round number, for the record
#   --elements      the chemical space of the hull (default: the parent's)
#   --keep-ehull    absolute E_hull ceiling in meV/atom (defaults to --max-ehull)
#   --mask-elements moments of these elements are discarded when summing [D1]
#   --no-relax      score the geometry as supplied (for already-relaxed input)
#
# OUTPUTS
#   <archive>/archive.csv              every structure ever seen, appended
#   <archive>/structures/<fp>.vasp     the relaxed geometry
#   <archive>/round-NN/scored.csv      this round, all of it
#   <archive>/round-NN/pareto.csv      the front -- the DFT promotion shortlist
#   <archive>/round-NN/digest.md       what the round LEARNED (the agent reads this)
#
# RUN
#   python round.py 'inputs/seeds/*.vasp' --parent templates/Ce2Fe14B.cif \
#     --archive loop/ --round 1 --mask-elements Ce
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse, csv, glob, hashlib, json, os, sys, time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from pymatgen.core import Structure, Element, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from hull import MlipHull                                   # noqa: E402
from cspflow.mlip import MatterSimEngine, CHGNetEngine      # noqa: E402

TESLA_PER_MUB_PER_A3 = 4.0e-7 * np.pi * 9.2740100783e-24 / 1.0e-30   # 11.6541
HILL_ELS = {"Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb"}
MIN_FRAC, HILL = 0.80, 3.50


def radius(el: str) -> float:
    e = Element(el)
    v = getattr(e, "atomic_radius", None) or getattr(e, "atomic_radius_calculated", None)
    return float(v) if v else 1.3


def geometry_gate(s: Structure) -> str:
    """[B2][B3] on what is actually on disk. Free, so it runs before any model."""
    d = s.distance_matrix.copy()
    np.fill_diagonal(d, 1e9)
    worst = 1e9
    for i in range(len(s)):
        j = int(np.argmin(d[i]))
        lim = MIN_FRAC * (radius(str(s[i].specie)) + radius(str(s[j].specie)))
        worst = min(worst, d[i][j] / lim)
    if worst < 1.0:
        return f"[B2] shortest contact at {worst:.2f}x the 0.80*sum-r limit"
    for el in sorted(HILL_ELS & {str(t.specie) for t in s}):
        idx = [i for i, t in enumerate(s) if str(t.specie) == el]
        if len(idx) > 1:
            dd = min(d[i][j] for i in idx for j in idx if i != j)
            if dd < HILL:
                return f"[B3] {el}-{el} {dd:.3f} A below the {HILL} A Hill gate"
    return ""


def parser_agreement(path: str, s: Structure) -> str:
    """[B4] pymatgen and ASE must agree on what the file says."""
    try:
        from ase.io import read as ase_read
        pm = {k: int(round(v)) for k, v in s.composition.get_el_amt_dict().items()}
        As = {k: int(v) for k, v in Counter(ase_read(path).get_chemical_symbols()).items()}
        return "" if pm == As else f"[B4] parser disagreement pymatgen={pm} ase={As}"
    except Exception as exc:
        return f"[B4] unreadable: {exc}"


def fingerprint(s: Structure) -> str:
    """Stable id: composition + spacegroup + rounded cell. Cheap pre-filter for
    dedup; StructureMatcher is the authority and runs only within a bucket."""
    try:
        sg = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number()
    except Exception:
        sg = 0
    key = (f"{s.composition.reduced_formula}|{sg}|{len(s)}|"
           f"{round(s.volume/len(s), 2)}")
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def site_correspondence(parent: Structure, s: Structure, tol: float):
    pf, cf, M = parent.frac_coords, s.frac_coords, parent.lattice.matrix
    dev_c = np.array([np.linalg.norm(((pf - v) - np.round(pf - v)) @ M, axis=1).min()
                      for v in cf])
    dev_p = np.array([np.linalg.norm(((cf - v) - np.round(cf - v)) @ M, axis=1).min()
                      for v in pf])
    off = int((dev_c > tol).sum())
    unc = int((dev_p > tol).sum())
    return off, unc, 1.0 - unc / len(pf)


def classify(parent: Structure, s: Structure, tol=0.8, min_covered=0.90) -> str:
    """Type I/II/III on the parent's SITE framework, not its species [G12]."""
    try:
        off, unc, covered = site_correspondence(parent, s, tol)
    except Exception:
        return "III"
    if covered < min_covered:
        return "III"
    if off == 0 and unc == 0 and len(s) == len(parent):
        return "I"
    return "II"


def re_free_fraction(s: Structure, mag_el: str, cutoff: float = 4.0) -> float:
    """Fraction of magnetic atoms with NO non-magnetic atom within `cutoff`.

    This is the alpha-Fe discriminator, and it replaces the pure-first-shell
    fraction that round 2 showed to be unusable OUTSIDE the 2:14:1 basin.

    WHY THE OLD ONE FAILED. The pure-shell fraction is a property of the
    PROTOTYPE, not of segregation. Measured 2026-09-15 on clean, ordered,
    perfectly good compounds: YFe5 0.000, Ce2Fe14B 0.143, Ce2Fe17 0.176,
    Y3Fe29 0.241, YFe12 0.333, CeFe5 0.600. Comparing any of those against the
    2:14:1 parent's value flags the denser prototypes for being dense, which is
    the property we are SEARCHING FOR. Round 2 kept 0 of 2 seeds because of it.

    WHY THIS ONE WORKS. alpha-Fe is not "iron with iron neighbours" -- every
    Fe-rich intermetallic has that. It is a REGION with no rare earth in it. So
    measure the distance from each Fe to the nearest non-Fe and count how many
    sit beyond `cutoff`. In every ordered RE-Fe prototype every Fe is within
    3.18-3.32 A of an RE or B, so the clean value is 0.000 for all of them,
    prototype-independent. Calibration at cutoff 4.0 A:

        clean prototypes (1:12, 2:17, 2:14:1)          0.000
        one RE vacancy                                 0.000 - 0.042
        25% of the RE removed                          0.029 - 0.104
        50% of the RE removed                          0.059 - 0.250
        a genuinely RE-free slab (half the cell)       0.179 - 0.417
        bcc alpha-Fe                                   1.000

    A threshold of 0.15 therefore passes ordinary defect chemistry and catches
    real segregation, with no reference to the parent at all.
    """
    non = [i for i, t in enumerate(s) if str(t.specie) != mag_el]
    mag = [i for i, t in enumerate(s) if str(t.specie) == mag_el]
    if not mag:
        return 0.0
    if not non:
        return 1.0           # no non-magnetic atom anywhere: pure alpha-Fe
    far = 0
    for i in mag:
        if min(s.get_distance(i, j) for j in non) > cutoff:
            far += 1
    return far / len(mag)


def alpha_scan(s: Structure, mag_el: str, cutoff=3.0):
    idx = [i for i, t in enumerate(s) if str(t.specie) == mag_el]
    if not idx:
        return 0, 0.0
    pure, purity = 0, []
    for i in idx:
        nb = s.get_neighbors(s[i], cutoff)
        if not nb:
            continue
        nm = sum(1 for n in nb if str(n.specie) == mag_el)
        purity.append(nm / len(nb))
        if nm == len(nb):
            pure += 1
    return pure, (float(np.mean(purity)) if purity else 0.0)


def pareto_front(rows, xkey, ykey):
    """Minimise xkey, maximise ykey. Returns the non-dominated rows."""
    pts = [r for r in rows if r.get(xkey) is not None and r.get(ykey) is not None]
    front = []
    for r in pts:
        if not any((o[xkey] <= r[xkey] and o[ykey] >= r[ykey])
                   and (o[xkey] < r[xkey] or o[ykey] > r[ykey]) for o in pts):
            front.append(r)
    return sorted(front, key=lambda r: r[xkey])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", nargs="+")
    ap.add_argument("--parent", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--elements", nargs="*", default=None)
    ap.add_argument("--magnetic-element", default="Fe")
    ap.add_argument("--mask-elements", nargs="*", default=[])
    # ---- the three filters, and what each is FOR -------------------------
    # They do different jobs and all three are reported, because a filter that
    # removes nothing should be visible as such rather than looking like it
    # earned its place. Measured 2026-09-15 on 19 Ce2Fe14B defect structures:
    # --max-ehull and --min-moment passed 19/19 (the set spans 0.041-0.156
    # eV/atom and its LOWEST moment is 4.2x the floor), while the
    # parent-relative cut took 19 -> 2. In a 2:14:1 perturbation campaign
    # everything is 82-85% Fe, so the absolute floors are a SAFETY NET; they
    # start doing real work only in the pivot batches that leave the basin.
    ap.add_argument("--max-ehull", type=float, default=0.2,
                    help="ABSOLUTE hull-distance ceiling, eV/atom (default 0.2). A "
                         "floor against structures that fell apart, not a selector.")
    ap.add_argument("--min-moment", type=float, default=0.458,
                    help="moment floor in muB per ATOM (default 0.458 = 1/4 of "
                         "Nd2Fe14B's 1.830 muB/atom, mp-5182 on OUR frozen-f scale; "
                         "the experimental 37.7 muB/f.u. would give 0.554). Catches a "
                         "reconstruction that lost its magnetism.")
    # WAS --keep-dd, a PARENT-RELATIVE ceiling. Removed 2026-09-15 on Ridwan's
    # objection, which was correct on three counts:
    #   (a) E_hull is DEFINED per chemical system -- it is the decomposition
    #       energy against that system's own stable phases. Subtracting another
    #       compound's E_hull references nothing physical.
    #   (b) Within ONE system ddE_hull = E_hull - const, so it is a rigid shift
    #       and reorders NOTHING. It never earned its place as "the cut that
    #       shortlists"; it only let me express a tighter number.
    #   (c) Its justification was "the MLIP error cancels against the parent".
    #       Measured against OUR OWN DFT on the RE2Fe14B series, MatterSim sits
    #       +4 to +10 meV/atom for seven of eight REs -- and Ce, the parent, is
    #       the lone outlier at -24.9. Subtracting the parent therefore INJECTS
    #       the one anomaly in the set into every candidate.
    # The ceiling is now absolute, on the MatterSim hull, and defaults to
    # --max-ehull so it is inert unless deliberately tightened.
    ap.add_argument("--keep-ehull", type=float, default=None,
                    help="ABSOLUTE MatterSim E_hull ceiling for the shortlist, in "
                         "meV/atom. Defaults to --max-ehull, i.e. no second cut. "
                         "LOOSE on purpose [G9]: a tight MLIP threshold discards the "
                         "out-of-distribution reconstructions that are the interesting "
                         "output.")
    ap.add_argument("--no-relax", action="store_true")
    ap.add_argument("--fmax", type=float, default=0.05,
                    help="MLIP force criterion, eV/A. 0.05 for SCREENING, not the "
                         "engine's 0.01 production default: measured on Ce2Fe14B, "
                         "0.05 converges in 22 steps / 24 s while 0.01 runs many "
                         "times longer for a hull distance that moves by well under "
                         "the surrogate's own ~40 meV/atom error. Spending minutes "
                         "per structure to polish a number that is then thresholded "
                         "at 50 meV is buying precision the screen cannot use.")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--alpha-max-free", type=float, default=0.15,
                    help="flag when more than this FRACTION of magnetic atoms "
                         "have no non-magnetic atom within --alpha-cutoff. "
                         "Prototype-independent; see re_free_fraction(). 0.15 "
                         "passes a 50%%-vacancy structure and catches a slab.")
    ap.add_argument("--alpha-cutoff", type=float, default=4.0,
                    help="radius in A for the alpha-Fe region test (default 4.0)")
    ap.add_argument("--alpha-margin", type=float, default=0.15,
                    help="how much the fraction of magnetic-element atoms with a "
                         "FULLY magnetic-element first shell must exceed the "
                         "parent's before [G2] fires. 0.15 = 15 percentage points. "
                         "Any-increase was too tight: a single vacancy trips it.")
    ap.add_argument("--start", type=int, default=0,
                    help="SLURM array chunking: skip this many seeds")
    ap.add_argument("--count", type=int, default=0,
                    help="SLURM array chunking: screen at most this many (0 = all). "
                         "Chunked rather than one task per structure because loading "
                         "MatterSim and CHGNet costs ~30 s, which a single-structure "
                         "task would spend its whole life doing.")
    ap.add_argument("--shard", default="",
                    help="suffix for this task's output files, so array tasks writing "
                         "into one round directory do not overwrite each other")
    ap.add_argument("--index", default=os.environ.get(
        "CSPFLOW_STORE_INDEX", "/scratch/oridwan/mp-reference/index.csv"))
    a = ap.parse_args()
    # The shortlist ceiling is absolute and defaults to the floor, so there is
    # exactly ONE hull threshold unless the user deliberately adds a second.
    if a.keep_ehull is None:
        a.keep_ehull = a.max_ehull * 1000.0

    arch = Path(a.archive)
    (arch / "structures").mkdir(parents=True, exist_ok=True)
    rdir = arch / f"round-{a.round:02d}"
    rdir.mkdir(exist_ok=True)
    MAG = a.magnetic_element

    files = [f for pat in a.seeds for f in sorted(glob.glob(pat, recursive=True))]
    if not files:
        sys.exit("no seed files matched")
    if a.count:
        files = files[a.start:a.start + a.count]
        if not files:
            print(f"shard {a.shard or a.start}: nothing to do "
                  f"(start={a.start} beyond the seed list)")
            return
    parent = Structure.from_file(a.parent)
    elements = a.elements or sorted({str(t.specie) for t in parent})

    ms = MatterSimEngine(fmax=a.fmax, max_steps=a.max_steps)
    cg = CHGNetEngine(fmax=a.fmax, max_steps=a.max_steps)

    # ---- the parent sets the zero of every axis ---------------------------
    print(f"round {a.round}: {len(files)} seeds, parent {parent.composition.reduced_formula}")
    pr = ms.relax(AseAtomsAdaptor.get_atoms(parent)) if not a.no_relax else \
        ms.single_point(AseAtomsAdaptor.get_atoms(parent))
    if not pr.ok:
        sys.exit(f"parent failed MatterSim: {pr.error}")
    p_relaxed = AseAtomsAdaptor.get_structure(pr.atoms)
    # The hull must cover every element the ROUND will produce, not just the
    # parent's. A batch that introduces one substituent introduces a whole set
    # of new subsystems [E1], and without them the candidate cannot be placed.
    seed_els = set()
    for f in files:
        try:
            seed_els |= {str(t.specie) for t in Structure.from_file(f)}
        except Exception:
            pass
    space = sorted(set(elements) | seed_els)

    # ONE HULL PER CHEMICAL SYSTEM, not one hull over the union. Added
    # 2026-09-15 for the prototype rounds, which span 9 rare earths.
    #
    # Why this is the CORRECT construction and not just the cheap one: a
    # candidate's decomposition products must satisfy mass balance, so a phase
    # containing an element the candidate does not have can never appear in its
    # decomposition. Nd2Fe17's E_hull is therefore identical whether it is
    # placed on Fe-Nd or on a 10-element hull that also knows about Y -- the
    # extra entries are unreachable. Building the union hull only pays a convex
    # hull in 9 dimensions (qhull cost explodes) and reports every absent
    # mixed-RE subsystem as "missing" when none of them could ever matter.
    #
    # This also makes the code say what E_hull actually MEANS: the decomposition
    # energy in the candidate's OWN chemical system.
    _hull_cache = {}

    def hull_for(els):
        key = tuple(sorted(set(els)))
        h = _hull_cache.get(key)
        if h is None:
            h = MlipHull(list(key), a.index)
            h.parent_e_above_hull = None
            _hull_cache[key] = h
        return h

    hull = hull_for(elements)          # the parent's own system
    missing = hull.missing_subsystems(elements)
    # Reported for the PARENT's system only. Per-candidate completeness is now
    # judged on that candidate's own hull (`chull.missing_elements()`), because
    # each chemical system has its own answer -- a Nd2Fe17 seed is not affected
    # by a gap in the Ce-Y subsystem, and saying otherwise flagged sound
    # candidates for a reason that could not apply to them.
    if missing:
        print(f"\n!! [E1] the store has NO entries for {len(missing)} subsystem(s) of "
              f"the PARENT system {'-'.join(sorted(elements))}:\n   {', '.join(missing)}")
        print("   Candidates are flagged per-system as `hull_incomplete`. An empty")
        print("   subsystem usually has no hull facet either (and for the RE")
        print("   quaternaries is empty in MP too), so E_hull moves little.\n")
    print(f"round spans {len(set(space))} elements: {'-'.join(space)}")
    print(f"  -> one hull per candidate chemical system, built on demand\n")

    p_hull = hull.e_above_hull(p_relaxed, pr.energy)
    if p_hull is None:
        sys.exit("the PARENT could not be placed on the hull -- refusing to screen "
                 "against a hull that cannot hold the reference structure.")
    hull.parent_e_above_hull = p_hull
    print(hull.describe())
    pcg = cg.single_point(pr.atoms)
    p_m = sum(m for m, t in zip(pcg.magmoms or [], p_relaxed)
              if str(t.specie) not in a.mask_elements) if pcg.magmoms else None
    p_js = TESLA_PER_MUB_PER_A3 * p_m / p_relaxed.volume if p_m else None
    p_pure, p_purity = alpha_scan(p_relaxed, MAG)
    p_free = re_free_fraction(p_relaxed, MAG, a.alpha_cutoff)
    p_nmag = sum(1 for t in p_relaxed if str(t.specie) == MAG)
    p_frac = p_pure / p_nmag if p_nmag else 0.0
    print(f"parent: E_hull {p_hull*1000:.1f} meV/atom   J_s_max {p_js:.4f} T   "
          f"alpha baseline {p_pure}/{p_nmag} pure-{MAG} shells "
          f"({p_frac:.2f});  RE-free fraction {p_free:.3f} "
          f"(flag above {a.alpha_max_free} at {a.alpha_cutoff} A)\n")

    # ---- the archive, so a structure is never screened twice ---------------
    seen, arch_csv = {}, arch / "archive.csv"
    if arch_csv.is_file():
        for r in csv.DictReader(open(arch_csv)):
            seen.setdefault(r["formula"], []).append(r)
    matcher = StructureMatcher(primitive_cell=False, attempt_supercell=False)
    archived_structs = defaultdict(list)
    for f in (arch / "structures").glob("*.vasp"):
        try:
            st = Structure.from_file(f)
            archived_structs[st.composition.reduced_formula].append((f.stem, st))
        except Exception:
            pass

    rows, n_gated, n_dup = [], 0, 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        try:
            s = Structure.from_file(f)
        except Exception as exc:
            print(f"  [{i}/{len(files)}] UNREADABLE {f}: {exc}")
            continue
        stem = Path(f).stem
        # 3 GATE
        note = geometry_gate(s) or parser_agreement(f, s)
        if note:
            n_gated += 1
            rows.append({"file": stem, "round": a.round, "status": "gated",
                         "gate": note, "formula": s.composition.reduced_formula})
            continue
        # 4 DEDUP against the whole archive
        dup = next((fp for fp, st in archived_structs[s.composition.reduced_formula]
                    if matcher.fit(s, st)), None)
        if dup:
            n_dup += 1
            rows.append({"file": stem, "round": a.round, "status": "duplicate",
                         "gate": f"matches archived {dup}",
                         "formula": s.composition.reduced_formula})
            continue
        # 5 RELAX
        at = AseAtomsAdaptor.get_atoms(s)
        r = ms.relax(at) if not a.no_relax else ms.single_point(at)
        if not r.ok:
            rows.append({"file": stem, "round": a.round, "status": "mlip_failed",
                         "gate": r.error, "formula": s.composition.reduced_formula})
            continue
        sr = AseAtomsAdaptor.get_structure(r.atoms)
        # 6 SCREEN-E -- on the hull of THIS candidate's chemical system
        cand_els = {str(t.specie) for t in sr}
        try:
            chull = hull_for(cand_els)
        except SystemExit as exc:
            rows.append({"file": stem, "round": a.round, "status": "no_hull",
                         "gate": f"[E1] {exc}",
                         "formula": sr.composition.reduced_formula})
            print(f"  [{i}/{len(files)}] {stem[:44]:<46} NO HULL [E1]", flush=True)
            continue
        eh = chull.e_above_hull(sr, r.energy)
        if eh is None:
            els = "-".join(sorted(cand_els))
            rows.append({"file": stem, "round": a.round, "status": "no_hull",
                         "gate": f"[E1] no hull coverage for {els}; extend the store",
                         "formula": sr.composition.reduced_formula})
            print(f"  [{i}/{len(files)}] {stem[:44]:<46} NO HULL ({els}) [E1]", flush=True)
            continue
        incomplete = sorted(cand_els & set(chull.missing_elements()))
        dd = (eh - p_hull) * 1000.0
        # 7 SCREEN-M
        mres = cg.single_point(r.atoms)
        if mres.magmoms:
            m_sum = sum(m for m, t in zip(mres.magmoms, sr)
                        if str(t.specie) not in a.mask_elements)
            m_mag = sum(m for m, t in zip(mres.magmoms, sr) if str(t.specie) == MAG)
            js = TESLA_PER_MUB_PER_A3 * m_sum / sr.volume
        else:
            m_sum = m_mag = js = None
        # 8 CLASSIFY
        cls = classify(parent, sr)
        pure, purity = alpha_scan(sr, MAG)
        # A MARGIN, not "any increase". The first version flagged whenever the
        # count of pure-Fe first shells exceeded the parent's by even one, which
        # a single Ce vacancy guarantees: take a Ce out and the Fe that had it as
        # a neighbour now sees only Fe. That flagged the INTENDED chemistry as
        # segregation -- 12 of 19 on a set whose worst member was an ordinary
        # antisite. Segregation means a REGION of Fe-only environment, so compare
        # the FRACTION of Fe with a pure shell and require a real shift.
        nmag_t = sum(1 for t in sr if str(t.specie) == MAG)
        frac = pure / nmag_t if nmag_t else 0.0
        re_free = re_free_fraction(sr, MAG, a.alpha_cutoff)
        alpha = int(re_free > a.alpha_max_free)
        nmag = sum(1 for t in sr if str(t.specie) == MAG)
        fp = fingerprint(sr)
        sr.to(filename=str(arch / "structures" / f"{fp}.vasp"), fmt="poscar")
        archived_structs[sr.composition.reduced_formula].append((fp, sr))
        rows.append({
            "file": stem, "round": a.round, "status": "scored", "gate": "",
            "fingerprint": fp, "formula": sr.composition.reduced_formula,
            "natoms": len(sr), "sg": _sg(sr),
            "e_per_atom": round(r.e_per_atom, 6),
            "E_hull_meV": round(eh * 1000, 2), "ddE_hull_meV": round(dd, 2),
            f"n_{MAG}": nmag,
            # x_Fe at.% and n_Fe/V are the TWO RIVAL ANSWERS to the campaign's
            # question -- "more iron" vs "more iron per unit volume". Recording
            # only one of them settles the question by omission. x was missing
            # from the first digest for exactly that reason.
            f"x_{MAG}_atpct": round(100.0 * nmag / len(sr), 4),
            f"n_{MAG}_per_A3": round(nmag / sr.volume, 6),
            "volume_A3": round(sr.volume, 2),
            "vol_per_atom": round(sr.volume / len(sr), 4),
            "m_sum_muB": round(m_sum, 3) if m_sum else None,
            # The SUBLATTICE sum is what calibration compares against DFT. The
            # cell total is not usable for that: the store runs f_treatment:
            # frozen with reconstruct_ms, so DFT's total carries a Hund's-rule
            # 4f term that was assumed rather than computed [D1]. The
            # transition-metal sum is a real number on both sides.
            f"m_{MAG}_sum_muB": round(m_mag, 3) if m_mag else None,
            f"mean_m_{MAG}": round(m_mag / nmag, 4) if (m_mag and nmag) else None,
            "J_s_max_T": round(js, 4) if js else None,
            "d_J_s_T": round(js - p_js, 4) if (js and p_js) else None,
            "topology_class": cls, "alpha_flag": alpha,
            "re_free_frac": round(re_free, 4),
            "hull_incomplete": int(bool(incomplete)),
            "hull_missing_for": "+".join(incomplete),
            # the evidence behind the flag, so it can be audited rather than
            # trusted -- a gate whose numbers are not in the table is a gate
            # nobody can check
            f"{MAG}_pure_shells": pure,
            f"{MAG}_pure_frac": round(frac, 4),
            "mean_shell_purity": round(purity, 4),
            "volume_drift": round(r.volume_drift, 4) if r.volume_drift else None,
        })
        # Print EVERY structure, with its two screen axes. A round of 10 with a
        # progress line every 10 looks hung for six minutes; and the numbers are
        # the thing worth watching go by, not the counter.
        print(f"  [{i}/{len(files)}] {stem[:44]:<46} ddE {dd:+7.1f} meV  "
              f"J_s_max {js:.4f} T  {cls}{' aFe' if alpha else ''}"
              if js is not None else
              f"  [{i}/{len(files)}] {stem[:44]:<46} ddE {dd:+7.1f} meV  (no moments)",
              flush=True)

    scored = [r for r in rows if r["status"] == "scored"]
    if not scored:
        sys.exit(f"nothing scored: {n_gated} gated, {n_dup} duplicates")

    # 9 SELECT -- hard floors, then the parent-relative cut, then Pareto.
    # Counted separately so the digest can say which filter did the work.
    def _mub_per_atom(r):
        return (r["m_sum_muB"] / r["natoms"]) if r.get("m_sum_muB") else None

    passed = []
    # Pre-seeded so a filter that removed NOTHING still appears with a 0. A
    # Counter only lists what it counted, so the two absolute floors -- the ones
    # most likely to be inert on a given chemistry -- silently vanished from the
    # very table written to expose that.
    cut = Counter({f"E_hull >= {a.max_ehull:.2f} eV/atom": 0,
                   f"moment <= {a.min_moment:.3f} muB/atom": 0,
                   f"alpha-{MAG} segregation [G2]": 0,
                   f"E_hull > {a.keep_ehull:.0f} meV/atom (shortlist)": 0})
    for r in scored:
        m = _mub_per_atom(r)
        r["muB_per_atom"] = round(m, 4) if m is not None else None
        r["hull_incomplete"] = int(bool(r.get("hull_incomplete")))
        if r["E_hull_meV"] is None or r["E_hull_meV"] / 1000.0 >= a.max_ehull:
            cut[f"E_hull >= {a.max_ehull:.2f} eV/atom"] += 1; continue
        if m is None or m <= a.min_moment:
            cut[f"moment <= {a.min_moment:.3f} muB/atom"] += 1; continue
        # NOT an exclusion. Corrected 2026-09-15, same day it was added: I read
        # a NEGATIVE ddE_hull as "below the hull" and concluded the hull was
        # broken. It was not -- ddE_hull is PARENT-RELATIVE by construction, so
        # -7.4 meV means "7 meV more stable than Ce2Fe14B". On the full
        # B-Ce-Fe-Y hull that candidate sits at +33.3 meV/atom against the
        # parent's +40.6, which is exactly what it should do: Y2Fe14B is a real
        # compound 6.2 meV above the hull while Ce2Fe14B is 40.7, so swapping Y
        # for Ce genuinely stabilises. The filter discarded 44 sound candidates.
        #
        # The store IS missing phases for those subsystems (MP has 5 Ce-Y, 3
        # Ce-Fe-Y, 5 Ce-La), but every one is itself 0.03-0.11 eV/atom above the
        # hull, so none forms a facet and E_hull barely moves. Worth adding for
        # completeness; not worth throwing candidates away over. A hull that
        # genuinely cannot place a structure still fails loudly as `no_hull`.
        if r["alpha_flag"]:
            cut[f"alpha-{MAG} segregation [G2]"] += 1; continue
        passed.append(r)
    kept = [r for r in passed if r["E_hull_meV"] is not None
            and r["E_hull_meV"] <= a.keep_ehull]
    cut[f"E_hull > {a.keep_ehull:.0f} meV/atom (shortlist)"] = len(passed) - len(kept)
    # Both axes absolute and on the SAME MatterSim hull: minimise E_hull,
    # maximise J_s_max. No parent subtraction anywhere.
    front = pareto_front(kept, "E_hull_meV", "J_s_max_T")
    a.filter_cuts = cut
    a.n_passed_floors = len(passed)

    sfx = f".{a.shard}" if a.shard else ""
    _write(rdir / f"scored{sfx}.csv", rows)
    _write(rdir / f"pareto{sfx}.csv", front)
    _append_archive(arch_csv, scored)
    digest = _digest(a, rows, scored, kept, front, p_hull, p_js, MAG, n_gated, n_dup, t0)
    (rdir / f"digest{sfx}.md").write_text(digest)
    print("\n" + digest)
    print(f"wrote {rdir}/scored{sfx}.csv, pareto{sfx}.csv, digest{sfx}.md")


def _sg(s):
    try:
        return SpacegroupAnalyzer(s, symprec=0.01).get_space_group_number()
    except Exception:
        return None


def _write(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("file", "round"), k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _append_archive(path, rows):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("file", "round"), k))
    new = not path.is_file()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def _corr(xs, ys):
    """Spearman, so a monotone-but-curved relation is not missed."""
    try:
        from scipy.stats import spearmanr
        if len(xs) < 4:
            return None, None
        r, p = spearmanr(xs, ys)
        return (None, None) if np.isnan(r) else (round(float(r), 3), round(float(p), 4))
    except Exception:
        return None, None


def _digest(a, rows, scored, kept, front, p_hull, p_js, MAG, n_gated, n_dup, t0):
    """Step 10 -- what the round LEARNED. This is what the proposer reads."""
    L = [f"# Round {a.round} digest", ""]
    n_nohull = sum(1 for r in rows if r.get("status") == "no_hull")
    L += [f"- seeds in: {len(rows)}   gated: {n_gated}   duplicates: {n_dup}   "
          f"no hull [E1]: {n_nohull}   scored: {len(scored)}",
          f"- kept (E_hull <= {a.keep_ehull:g} meV/atom, no alpha flag): {len(kept)}",
          f"- Pareto front: {len(front)}",
          f"- wall time: {time.time()-t0:.0f}s", ""]
    L += [f"Parent reference: E_hull {p_hull*1000:.1f} meV/atom, "
          f"J_s_max {p_js:.4f} T" if p_js else "", ""]

    L += _digest_tables(scored, kept, front, MAG, a.keep_ehull, rows, a)
    return "\n".join(L)


def _digest_tables(scored, kept, front, MAG, keep_ehull, rows=None, a=None):
    """The evidence tables. Split out of _digest() so `orchestrate.py merge` can
    rebuild them over the UNION of the array shards -- a correlation computed on
    one 40-structure task is not the round's answer."""
    L = []
    rows = rows if rows is not None else scored   # gate counts need the gated rows too
    L += ["## Which descriptor actually tracks J_s [G1]", ""]
    ok = [r for r in scored if r["J_s_max_T"] is not None]
    if len(ok) >= 4:
        js = [r["J_s_max_T"] for r in ok]
        L += ["| descriptor | Spearman rho vs J_s_max | p |", "|---|---|---|"]
        for key, label in ((f"n_{MAG}_per_A3", f"n_{MAG}/V  (density)"),
                           (f"x_{MAG}_atpct", f"x_{MAG} at.%"),
                           ("vol_per_atom", "volume per atom"),
                           (f"mean_m_{MAG}", f"mean m({MAG})")):
            xs = [r.get(key) for r in ok]
            if all(x is not None for x in xs):
                r_, p_ = _corr(xs, js)
                if r_ is not None:
                    L += [f"| {label} | {r_:+.3f} | {p_:.4f} |"]
        # A correlation table is only an answer if the descriptors are actually
        # distinguishable. In a library where every member has nearly the same
        # N_Fe, n_Fe/V is just 1/V wearing a different label, and the two will
        # report equal-and-opposite rho -- which looks like two findings and is
        # one. Say so, and name the experiment that would separate them, rather
        # than letting the next round read it as settled.
        nv = [r.get(f"n_{MAG}_per_A3") for r in ok]
        vv = [r.get("vol_per_atom") for r in ok]
        xv = [r.get(f"x_{MAG}_atpct") for r in ok]
        warn = []
        if all(v is not None for v in nv + vv):
            r_nv, _ = _corr(nv, vv)
            if r_nv is not None and abs(r_nv) > 0.9:
                warn.append(
                    f"- **n_{MAG}/V and volume/atom are collinear here "
                    f"(rho = {r_nv:+.3f})**, so the table cannot say which of them "
                    f"drives J_s -- they are the same variable in this library. "
                    f"To separate them the next round needs EITHER a strain series "
                    f"at FIXED composition (volume moves, N_{MAG} does not) OR a "
                    f"substitution that changes N_{MAG} at nearly fixed volume.")
        if all(v is not None for v in xv + nv):
            r_xn, _ = _corr(xv, nv)
            if r_xn is not None and abs(r_xn) > 0.9:
                warn.append(
                    f"- x_{MAG} at.% and n_{MAG}/V are also collinear "
                    f"(rho = {r_xn:+.3f}): every seed here changes both at once. "
                    f"The count-vs-density question is NOT yet answered.")
        L += [""] + (warn if warn else [
            f"- the descriptors are separable in this library, so the ranking above "
            f"is informative: act on whichever tops it."])
        L += ["", "Read this before proposing the next round: if J_s tracks "
              f"n_{MAG}/V and NOT x_{MAG} at.%, then 'add more {MAG}' is the "
              "wrong axis and the next round should be pushing on volume.", ""]
    else:
        L += ["_too few scored structures to regress_", ""]

    L += ["## Pareto front (the DFT promotion shortlist)", ""]
    if front:
        L += ["| file | E_hull meV | J_s_max T | dJ_s T | type | formula |",
              "|---|---|---|---|---|---|"]
        for r in front[:15]:
            L += [f"| {r['file'][:38]} | {r['E_hull_meV']:.1f} | "
                  f"{r['J_s_max_T']:.4f} | {r.get('d_J_s_T', 0):+.4f} | "
                  f"{r['topology_class']} | {r['formula']} |"]
    else:
        L += ["_empty -- every candidate was either above the E_hull ceiling or "
              "alpha-flagged. Widen --keep-ehull or change route._"]
    L += [""]

    # "List and rank" -- the full ranked table, not only the non-dominated set.
    # The Pareto front answers "which are not beaten on both axes"; this answers
    # "which has the most tesla", which is the campaign's actual question.
    L += ["## Full ranking by J_s_max (MatterSim hull, absolute)", ""]
    rank = sorted(kept, key=lambda r: -(r["J_s_max_T"] or 0))
    if rank:
        L += ["| # | file | J_s_max T | dJ_s T | E_hull meV | muB/atom | sg | formula |",
              "|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rank[:25], 1):
            L += [f"| {i} | {r['file'][:34]} | {r['J_s_max_T']:.4f} | "
                  f"{r.get('d_J_s_T', 0):+.4f} | {r['E_hull_meV']:.1f} | "
                  f"{r.get('muB_per_atom') or 0:.3f} | {r.get('sg', '')} | "
                  f"{r['formula']} |"]
    else:
        L += ["_nothing passed the filters_"]
    L += [""]

    L += ["## What each filter removed", "",
          "| filter | removed |", "|---|---|"]
    cuts = (getattr(a, "filter_cuts", None) or {}) if a is not None else {}
    for k, v in cuts.items():
        L += [f"| {k} | {v} |"]
    if not cuts:
        L += ["| _(per-shard counts; see the shard digests)_ | |"]
    n_cut = sum(cuts.values()) if cuts else 0
    inert = [k for k, v in (cuts or {}).items() if v == 0]
    L += ["",
          "A filter that removed 0 is not doing work on this chemistry -- say so "
          "rather than quoting it as a criterion the candidates 'passed'."]
    if n_cut == 0:
        L += ["**Every filter above removed nothing.** In a single-prototype "
              "perturbation set everything stays 82-85% Fe, so the absolute "
              "floors are a safety net that never engaged; the RANKING is what "
              "selected. Do not report these as criteria the shortlist met."]
    else:
        L += [f"**The filters removed {n_cut} structure(s) this round**, so they "
              f"are doing real work here -- this round left the parent's basin "
              f"far enough for the floors to engage."]
        if inert:
            L += [f"Still inert: {', '.join(inert)}."]
    L += [""]
    L += ["## What bound the round", ""]
    gates = Counter(r["gate"].split("]")[0] + "]" for r in rows
                    if r["status"] == "gated" and "]" in r["gate"])
    L += [f"- gates fired: {dict(gates) if gates else 'none'}",
          f"- alpha-{MAG} flags: {sum(1 for r in scored if r['alpha_flag'])}",
          f"- topology: {dict(Counter(r['topology_class'] for r in scored))}",
          f"- above the E_hull shortlist ceiling: "
          f"{sum(1 for r in scored if (r['E_hull_meV'] or 0) > keep_ehull)}", ""]
    L += ["## Honest limits of these numbers", "",
          "- `J_s_max` is a SATURATION UPPER BOUND from CHGNet magnitudes. It "
          "cannot tell FM from ferrimagnetic, and a ferrimagnet reads HIGH [D3].",
          "- `E_hull` is ABSOLUTE, on the MatterSim hull of the candidate's own "
          "chemical system. Against our own DFT on the RE2Fe14B series MatterSim "
          "runs +4 to +10 meV/atom high (Ce the outlier, -24.9), so it is a "
          "COARSE gate: differences of ~10 meV/atom are inside its noise and must "
          "not be read as a stability ranking [G9]. `ddE_hull_meV` is retained in "
          "scored.csv as a display convenience ONLY and gates nothing.",
          "- No SOC anywhere here, so K1, the easy axis and 'is it a permanent "
          "magnet' are NOT answered [D5][G10]. This round finds candidates worth "
          "DFT, not magnets."]
    return L


if __name__ == "__main__":
    main()
