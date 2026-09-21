"""The convex hull as a picture, built from the same entries the pipeline ranks on.

WHAT THIS MODULE DOES
    `csp report` used to show `e_above_hull` as a number in a column.  That
    number is the output of a `PhaseDiagram` that the pipeline builds, reads one
    float out of, and throws away -- vertices, facets and tie-lines included.
    This module rebuilds the same diagram from the same entries and keeps the
    geometry, so the report can draw it.

    It draws two hulls, never one:

        MLIP hull   MatterSim energies, ours on both sides (candidates and the
                    store's own relaxed reference phases).
        DFT hull    our VASP recompute, ours on both sides.

    They are separate panels with separate captions.  Merging them would be the
    D101 error in visual form: the two scales differ by +0.15 to +0.21 eV/atom
    and the selection threshold is 0.06, so a single plot carrying both would
    show candidates sitting below a hull they are nowhere near.

WHY IT REBUILDS RATHER THAN READS THE `hull` TABLE
    The `hull` table stores one distance per structure and nothing about the
    reference set's shape.  You cannot draw a tie-line from a scalar.  Rebuilding
    costs a second or two per chemical system and guarantees the picture is of
    the same hull the numbers came from -- if the store has changed since the
    campaign ran, the plot and the table disagree visibly, which is the correct
    behaviour.

WHAT IT REFUSES TO DO
    It does not fill a missing vertex.  `entries_for` refuses partial coverage
    and that refusal is passed straight through to the page as a stated reason,
    because a hull with one borrowed vertex still draws, still looks correct,
    and is wrong by the scale offset.

GEOMETRY
    binary      x = fraction of the second element, y = formation energy per
                atom.  The hull is the lower convex envelope; it is drawn as a
                line and the distance of a point above it is visible directly.
    ternary     the composition triangle.  x = (b + c/2), y = c*sqrt(3)/2 in
                fractional coordinates.  Formation energy cannot be a spatial
                axis here, so stable phases are drawn as filled vertices joined
                by the hull's tie-lines and every other point is coloured by how
                far above the hull it sits.
    4+ elements no faithful 2D projection exists, so nothing is drawn.  A strip
                plot of hull distance is shown instead, labelled as such.

INPUTS   a `Store`, and the chemical systems the campaign touches
OUTPUTS  `HullPanel` objects carrying plot-ready coordinates, and inline SVG
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..db.store import Store, StructureState
from ..reference.hull import Entry

# Candidates are coloured on this scale, in eV/atom above the hull.  0.06 is the
# campaign's own selection threshold and is a labelled band rather than a
# gradient stop, so a reader sees the cut rather than infers it.
SELECTION_THRESHOLD = 0.06
BANDS: list[tuple[float, str, str]] = [
    (0.0, "#1a7f37", "on the hull"),
    (SELECTION_THRESHOLD, "#3b82f6", f"within {SELECTION_THRESHOLD:g} eV/atom"),
    (0.2, "#b99c2b", "0.06 - 0.20"),
    (float("inf"), "#9aa0a6", "above 0.20"),
]

REF_COLOUR = "#8a8f98"          # reference phases: present, never the subject
REF_STABLE_COLOUR = "#4b5563"   # a reference phase that is itself a vertex


def band_colour(distance: float | None) -> str:
    if distance is None:
        return REF_COLOUR
    for edge, colour, _ in BANDS:
        if distance <= edge + 1e-12:
            return colour
    return BANDS[-1][1]


@dataclass
class Point:
    """One entry, placed."""

    label: str
    formula: str
    fractions: dict[str, float]       # element -> atomic fraction, sums to 1
    e_form: float                     # eV/atom, formation energy
    e_above_hull: float               # eV/atom
    is_candidate: bool
    structure_id: int | None = None
    stable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "formula": self.formula,
                "e_form": self.e_form, "e_above_hull": self.e_above_hull,
                "candidate": self.is_candidate, "id": self.structure_id,
                "stable": self.stable}


@dataclass
class HullPanel:
    """One hull, on one energy scale, for one chemical system."""

    chemsys: str
    method: str                        # 'mlip' | 'dft'
    elements: list[str] = field(default_factory=list)
    points: list[Point] = field(default_factory=list)
    tielines: list[tuple[str, str]] = field(default_factory=list)
    ok: bool = False
    reason: str = ""                   # why it is not drawable, in full
    n_reference: int = 0
    n_candidates: int = 0

    @property
    def dimension(self) -> int:
        return len(self.elements)

    @property
    def on_hull(self) -> list[Point]:
        return [p for p in self.points if p.is_candidate and p.stable]


METHOD_LABEL = {"mlip": "MatterSim (MLIP)", "dft": "our VASP (DFT)"}
METHOD_NOTE = {
    "mlip": ("Candidates and vertices are both MatterSim, from the same model "
             "and the same settings. One scale on both sides."),
    "dft": ("Candidates and vertices are both our own VASP recompute at the "
            "store's settings. One scale on both sides; MP's own numbers are "
            "never mixed in."),
}


# -- building ---------------------------------------------------------------

def _counts_of(row) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in row.toatoms().get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _candidate_entries(store: Store, method: str) -> dict[str, list[Entry]]:
    """Campaign structures as hull entries, grouped by chemical system.

    The MLIP panel uses every screened structure -- that is the population the
    prescreen actually ranked.  The DFT panel uses only what finished DFT,
    because that is the only population with a VASP energy.
    """
    from ..chem import chemsys as chemsys_of

    if method == "mlip":
        states = (StructureState.screened.value, StructureState.selected.value,
                  StructureState.dft_done.value)
        key, per_atom = "mlip_e_per_atom", True
    else:
        states = (StructureState.dft_done.value,)
        key, per_atom = "vasp_energy", False

    out: dict[str, list[Entry]] = {}
    seen: set[int] = set()
    for state in states:
        for row in store.structures(state=state):
            sid = int(row.id)
            if sid in seen:
                continue
            energy = row.key_value_pairs.get(key)
            if energy is None:
                continue
            seen.add(sid)
            counts = _counts_of(row)
            total = float(energy) * sum(counts.values()) if per_atom else float(energy)
            out.setdefault(chemsys_of(counts), []).append(Entry(
                label=f"cand-{sid}", counts=counts, energy=total, scale="raw",
                source="mlip" if method == "mlip" else "ours",
                run_type="" if method == "mlip" else "GGA",
                structure_id=sid))
    return out


def build_panel(chemsys: str, candidates: Sequence[Entry], method: str) -> HullPanel:
    """One drawable hull, or a panel that says in full why there is not one."""
    from ..reference.refstore import StoreError, entries_for

    panel = HullPanel(chemsys=chemsys, method=method,
                      elements=sorted(e for e in chemsys.split("-") if e),
                      n_candidates=len(candidates))
    try:
        reference = entries_for(chemsys, "mlip" if method == "mlip" else "dft")
    except StoreError as exc:
        panel.reason = str(exc)
        return panel
    except Exception as exc:                                     # noqa: BLE001
        panel.reason = f"reference store unreadable: {type(exc).__name__}: {exc}"
        return panel

    panel.n_reference = len(reference)
    if not reference:
        panel.reason = f"the store holds no {method} energies for {chemsys}"
        return panel

    entries = [*reference, *candidates]
    try:
        # strict_functional=False: store rows carry MP's run_type and campaign
        # candidates carry none, which the strict check reports as an
        # inconsistency it cannot resolve.  The scale check -- the one that
        # matters -- still runs inside build_hull.
        from ..reference.hull import (assert_elemental_references,
                                      assert_one_scale)
        assert_one_scale(entries)
        assert_elemental_references(entries)
        diagram, computed = _phase_diagram(entries)
    except Exception as exc:                                     # noqa: BLE001
        panel.reason = f"{type(exc).__name__}: {exc}"
        return panel

    is_candidate = {e.label: e.structure_id is not None for e in entries}
    formula_of = {e.label: e.formula for e in entries}
    fractions_of = {
        e.label: {el: n / e.n_atoms for el, n in e.counts.items() if n}
        for e in entries
    }

    for computed_entry in computed:
        label = str(computed_entry.entry_id)
        above = float(diagram.get_e_above_hull(computed_entry))
        panel.points.append(Point(
            label=label,
            formula=formula_of.get(label, label),
            fractions=fractions_of.get(label, {}),
            e_form=float(diagram.get_form_energy_per_atom(computed_entry)),
            e_above_hull=above,
            is_candidate=bool(is_candidate.get(label)),
            structure_id=_sid(label),
            stable=above <= 1e-9,
        ))

    panel.tielines = _tielines(diagram)
    panel.ok = True
    return panel


def _sid(label: str) -> int | None:
    if label.startswith("cand-"):
        try:
            return int(label[5:])
        except ValueError:                                       # pragma: no cover
            return None
    return None


def _phase_diagram(entries: Sequence[Entry]):
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    computed = [ComputedEntry(composition=Composition(e.counts), energy=e.energy,
                              entry_id=e.label) for e in entries]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PhaseDiagram(computed), computed


def _tielines(diagram) -> list[tuple[str, str]]:
    """Every edge of every hull facet, as a pair of entry labels.

    Read off `diagram.facets`, which indexes into `diagram.qhull_entries`.
    Edges are deduplicated because two adjacent triangles share one.
    """
    edges: set[tuple[str, str]] = set()
    try:
        qhull = list(diagram.qhull_entries)
        for facet in diagram.facets:
            members = [str(qhull[i].entry_id) for i in facet]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    edges.add(tuple(sorted((members[i], members[j]))))  # type: ignore[arg-type]
    except Exception:                                            # noqa: BLE001
        return []
    return sorted(edges)


def panels(store: Store, *, methods: Sequence[str] = ("mlip", "dft")) -> list[HullPanel]:
    """Every (chemical system, method) hull this campaign can draw."""
    out: list[HullPanel] = []
    for method in methods:
        for chemsys, candidates in sorted(_candidate_entries(store, method).items()):
            out.append(build_panel(chemsys, candidates, method))
    return out


# -- drawing ----------------------------------------------------------------

W, H, PAD = 560, 460, 54
# The binary plot carries numeric tick labels on its y axis, so it needs a
# wider left gutter than the ternary, whose left edge is a triangle corner.
PAD_L = 84


def _ternary_xy(fractions: dict[str, float], elements: list[str]) -> tuple[float, float]:
    """Composition triangle coordinates, in the unit triangle."""
    a, b, c = (fractions.get(e, 0.0) for e in elements)
    total = a + b + c or 1.0
    a, b, c = a / total, b / total, c / total
    return b + c / 2.0, c * math.sqrt(3.0) / 2.0


def svg(panel: HullPanel) -> str:
    """One panel as inline SVG, or an explanatory block when it cannot be drawn."""
    if not panel.ok:
        return _refusal(panel)
    if panel.dimension == 2:
        return _binary_svg(panel)
    if panel.dimension == 3:
        return _ternary_svg(panel)
    return _strip_svg(panel)


def _refusal(panel: HullPanel) -> str:
    return (f"<div class=refusal><b>No {METHOD_LABEL[panel.method]} hull for "
            f"{html.escape(panel.chemsys)}.</b><pre>{html.escape(panel.reason)}</pre>"
            f"<p class=note>Nothing is drawn rather than a hull with a borrowed "
            f"vertex. A hull missing one vertex still draws and still looks "
            f"correct.</p></div>")


def _title(panel: HullPanel) -> str:
    return (f"{html.escape(panel.chemsys)} &middot; {METHOD_LABEL[panel.method]}"
            f" &middot; {panel.n_reference} reference phases, "
            f"{panel.n_candidates} candidates")


def _dot(x: float, y: float, point: Point, radius: float) -> str:
    colour = band_colour(point.e_above_hull) if point.is_candidate else (
        REF_STABLE_COLOUR if point.stable else REF_COLOUR)
    tip = (f"{point.formula}  |  E_hull {point.e_above_hull:+.4f} eV/atom  |  "
           f"E_form {point.e_form:+.4f} eV/atom")
    if point.structure_id is not None:
        tip += f"  |  structure {point.structure_id}"
    stroke = "#111" if point.is_candidate else "none"
    width = 1.1 if point.is_candidate else 0
    extra = (f' data-sid="{point.structure_id}" class="clickable"'
             if point.structure_id else "")
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{colour}" '
            f'stroke="{stroke}" stroke-width="{width}" fill-opacity="'
            f'{0.95 if point.is_candidate else 0.55}"{extra}>'
            f"<title>{html.escape(tip)}</title></circle>")


def _binary_svg(panel: HullPanel) -> str:
    left, right = panel.elements
    xs = [p.fractions.get(right, 0.0) for p in panel.points]
    ys = [p.e_form for p in panel.points]
    lo, hi = min(ys + [0.0]), max(ys + [0.0])
    span = (hi - lo) or 1.0
    lo, hi = lo - 0.06 * span, hi + 0.06 * span

    def px(x: float) -> float:
        return PAD_L + x * (W - PAD_L - PAD)

    def py(y: float) -> float:
        return H - PAD - (y - lo) / (hi - lo) * (H - 2 * PAD)

    parts = [f'<svg viewBox="0 0 {W} {H}" class="hull" role="img">']
    # axes
    parts.append(f'<line x1="{PAD_L}" y1="{py(0.0):.1f}" x2="{W - PAD}" '
                 f'y2="{py(0.0):.1f}" class="axis" />')
    parts.append(f'<line x1="{PAD_L}" y1="{PAD - 10}" x2="{PAD_L}" '
                 f'y2="{H - PAD}" class="axis" />')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{px(frac):.1f}" y="{H - PAD + 18}" '
                     f'class="tick mid">{frac:g}</text>')
    steps = 5
    for i in range(steps + 1):
        value = lo + (hi - lo) * i / steps
        parts.append(f'<text x="{PAD_L - 8}" y="{py(value):.1f}" class="tick end">'
                     f"{value:+.2f}</text>")
    parts.append(f'<text x="{W / 2:.0f}" y="{H - 14}" class="axlab mid">'
                 f"x in {html.escape(left)}<tspan baseline-shift=sub>1-x</tspan>"
                 f"{html.escape(right)}<tspan baseline-shift=sub>x</tspan></text>")
    parts.append(f'<text transform="translate(18,{H / 2:.0f}) rotate(-90)" '
                 f'class="axlab mid">formation energy, eV/atom</text>')

    # the hull itself: lower envelope over the stable points
    hull_pts = sorted(((p.fractions.get(right, 0.0), p.e_form)
                       for p in panel.points if p.stable), key=lambda t: t[0])
    if len(hull_pts) > 1:
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}"
                     for i, (x, y) in enumerate(hull_pts))
        parts.append(f'<path d="{d}" class="tie" />')

    for point in sorted(panel.points, key=lambda p: p.is_candidate):
        parts.append(_dot(px(point.fractions.get(right, 0.0)), py(point.e_form),
                          point, 5.0 if point.is_candidate else 3.4))
    parts.append("</svg>")
    return f"<figure><figcaption>{_title(panel)}</figcaption>{''.join(parts)}</figure>"


def _ternary_svg(panel: HullPanel) -> str:
    a, b, c = panel.elements
    side = min(W - 2 * PAD, (H - 2 * PAD) / (math.sqrt(3) / 2))

    def place(fractions: dict[str, float]) -> tuple[float, float]:
        ux, uy = _ternary_xy(fractions, panel.elements)
        x = (W - side) / 2 + ux * side
        y = H - PAD - uy * side
        return x, y

    corner = {a: place({a: 1.0}), b: place({b: 1.0}), c: place({c: 1.0})}
    parts = [f'<svg viewBox="0 0 {W} {H}" class="hull" role="img">']
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                    (corner[a], corner[b], corner[c]))
    parts.append(f'<polygon points="{poly}" class="frame" />')

    # gridlines every 20 at.%, so a composition can be read off the picture
    for i in range(1, 5):
        t = i / 5.0
        for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
            p1 = place({u: 1 - t, v: t})
            p2 = place({u: 1 - t, w: t})
            parts.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" '
                         f'x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" class="grid" />')

    at = {p.label: place(p.fractions) for p in panel.points}
    for one, two in panel.tielines:
        if one in at and two in at:
            (x1, y1), (x2, y2) = at[one], at[two]
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                         f'y2="{y2:.1f}" class="tie" />')

    for point in sorted(panel.points, key=lambda p: p.is_candidate):
        x, y = at[point.label]
        parts.append(_dot(x, y, point, 5.2 if point.is_candidate else 3.4))

    for element, (x, y) in corner.items():
        dy = 20 if y > H / 2 else -12
        parts.append(f'<text x="{x:.1f}" y="{y + dy:.1f}" class="corner mid">'
                     f"{html.escape(element)}</text>")
    parts.append("</svg>")
    return (f"<figure><figcaption>{_title(panel)}</figcaption>{''.join(parts)}"
            f"<p class=note>Lines are the hull&rsquo;s tie-lines: the two-phase "
            f"equilibria a composition inside that triangle decomposes into. "
            f"Formation energy is not a spatial axis here &mdash; it is in the "
            f"colour and in the tooltip.</p></figure>")


def _strip_svg(panel: HullPanel) -> str:
    """Hull distance on one axis, for systems with no faithful 2D projection."""
    candidates = [p for p in panel.points if p.is_candidate]
    if not candidates:
        return _refusal(panel)
    hi = max(0.3, max(p.e_above_hull for p in candidates))
    height = 190

    def px(v: float) -> float:
        return PAD + min(v, hi) / hi * (W - 2 * PAD)

    parts = [f'<svg viewBox="0 0 {W} {height}" class="hull" role="img">']
    parts.append(f'<line x1="{PAD}" y1="120" x2="{W - PAD}" y2="120" class="axis" />')
    cut = px(SELECTION_THRESHOLD)
    parts.append(f'<line x1="{cut:.1f}" y1="40" x2="{cut:.1f}" y2="130" class="cut" />')
    parts.append(f'<text x="{cut:.1f}" y="32" class="tick mid">'
                 f"{SELECTION_THRESHOLD:g} eV/atom</text>")
    for i in range(6):
        v = hi * i / 5
        parts.append(f'<text x="{px(v):.1f}" y="142" class="tick mid">{v:.2f}</text>')
    for point in candidates:
        parts.append(_dot(px(point.e_above_hull), 120 - 6, point, 5.0))
    parts.append(f'<text x="{W / 2:.0f}" y="172" class="axlab mid">'
                 f"E above hull, eV/atom</text>")
    parts.append("</svg>")
    return (f"<figure><figcaption>{_title(panel)}</figcaption>{''.join(parts)}"
            f"<p class=note>{len(panel.elements)} elements: no 2D composition "
            f"projection is faithful, so the hull is not drawn. This is the "
            f"distance only.</p></figure>")


def legend_html() -> str:
    items = [f'<span class=key><i style="background:{colour}"></i>{html.escape(label)}</span>'
             for _, colour, label in BANDS]
    items.append(f'<span class=key><i style="background:{REF_STABLE_COLOUR}"></i>'
                 f"reference phase, on its own hull</span>")
    items.append(f'<span class=key><i style="background:{REF_COLOUR}"></i>'
                 f"reference phase, above it</span>")
    return "<div class=legend>" + "".join(items) + "</div>"
