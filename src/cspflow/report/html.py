"""One self-contained HTML file, in six sections.

Self-contained is a requirement, not a preference: the report is read from a
laptop over a mounted filesystem or emailed as an attachment, and a page that
fetches a stylesheet is a page that renders as unstyled text half the time. No
external requests, no build step, and every line of JavaScript is inlined.

WHAT IS ON THE PAGE, AND WHY IN THIS ORDER
    1. header       what ran, how much of it, when
    2. funnel       how many structures each gate removed
    3. hulls        the MLIP hull and the DFT hull, drawn, side by side and
                    never merged
    4. magnets      the figures of merit -- M, V, M/V, mu0*M -- ranked
    5. candidates   the full table, sortable, every column named
    6. cards        one per top candidate: the structure in 3D, every ion's
                    moment, and the numbers that came out of that run

    The order is "how many survived" -> "where they sit" -> "how good are they"
    -> "which one" -> "what is it". A reader who stops after section 3 has the
    honest summary; a reader who wants one compound scrolls to its card.

THE DESIGN RULE
    A number is never shown without what it is.  The two magnetisation columns
    are adjacent and differently shaded and the header says which one is
    computed; the two hulls are captioned with the scale each was built on; a
    hull that cannot be built prints the refusal in full instead of drawing
    something.

INPUTS   a `Store`; optionally a JSmol URL for the portal deployment
OUTPUTS  `report.html`
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..db.store import Store
from . import hullplot
from .candidates import COLUMNS, Funnel, candidate_rows, funnel
from .magnetics import ND2FE14B_JS_TESLA, summarise
from .provenance import SeedRecord
from . import provenance as prov
from .structure import StructureDetail, detail, element_table
from .viewer import VIEWER_JS

# Columns holding a modelled rather than a computed number. Shown, but marked.
MODELLED = {"m_s_reconstructed"}
# Columns derived by arithmetic from two computed ones. Not modelled, but not
# straight out of VASP either, so they are marked differently.
DERIVED = {"mu0_m_tesla", "m_emu_per_cc", "m_per_volume", "m_per_formula_unit"}

# How many structures get a detail card by default. Each card embeds a geometry,
# so this is the knob that decides whether the file is 200 kB or 40 MB.
DEFAULT_DETAIL = 25

_CSS = """
:root { --fg:#1a1a1a; --bg:#fff; --line:#d8d8d8; --muted:#666;
        --model:#fff6e5; --derived:#eef4ff; --head:#f4f4f6; --good:#1a7f37;
        --card:#fafafa; --warn:#8a5a00; --warnbg:#fff8e6; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --bg:#16181c; --line:#333; --muted:#9aa0a6;
          --model:#3a2f1a; --derived:#1b2436; --head:#22252b; --good:#4ac26b;
          --card:#1c1f25; --warn:#e0b050; --warnbg:#2c2416; }
}
body { background:var(--bg); color:var(--fg); margin:0; padding:2rem 1.5rem;
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
h1 { font-size:1.4rem; margin:0 0 .2rem; }
h2 { font-size:1.05rem; margin:2.2rem 0 .6rem; font-weight:600;
     border-bottom:1px solid var(--line); padding-bottom:.3rem; }
h3 { font-size:.95rem; margin:1.1rem 0 .4rem; font-weight:600; }
.sub { color:var(--muted); margin:0 0 1.5rem; font-size:.9rem; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th,td { padding:.32rem .5rem; border-bottom:1px solid var(--line);
        text-align:right; white-space:nowrap; }
th { background:var(--head); text-align:right; position:sticky; top:0;
     cursor:pointer; user-select:none; font-weight:600; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align:left; }
td.modelled, th.modelled { background:var(--model); }
td.derived, th.derived { background:var(--derived); }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:6px; }
.note { color:var(--muted); font-size:.85rem; margin:.5rem 0 0; }
.funnel td:first-child { text-align:left; }
.kept { color:var(--good); }
dl { display:grid; grid-template-columns:max-content 1fr; gap:.15rem .9rem;
     font-size:.85rem; color:var(--muted); margin:.4rem 0 0; }
dt { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--fg); }

/* -- hulls -- */
.hulls { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
         gap:1rem; }
figure { margin:0; border:1px solid var(--line); border-radius:8px;
         padding:.7rem .8rem 1rem; background:var(--card); }
figcaption { font-size:.82rem; color:var(--muted); margin-bottom:.3rem;
             font-weight:600; }
svg.hull { width:100%; height:auto; display:block; }
svg.hull .axis { stroke:var(--line); stroke-width:1.2; }
svg.hull .frame { fill:none; stroke:var(--muted); stroke-width:1.3; }
svg.hull .grid  { stroke:var(--line); stroke-width:.6; }
svg.hull .tie   { fill:none; stroke:#2f6f4f; stroke-width:1.5; opacity:.85; }
svg.hull .cut   { stroke:#b04040; stroke-width:1.3; stroke-dasharray:4 3; }
svg.hull text   { fill:var(--muted); font-size:11px; }
svg.hull .mid   { text-anchor:middle; }
svg.hull .end   { text-anchor:end; }
svg.hull .corner{ fill:var(--fg); font-size:14px; font-weight:600; }
svg.hull .axlab { fill:var(--fg); font-size:12px; }
svg.hull circle.clickable { cursor:pointer; }
.legend { display:flex; flex-wrap:wrap; gap:.4rem 1rem; margin:.6rem 0 .9rem;
          font-size:.82rem; color:var(--muted); }
.key i { display:inline-block; width:11px; height:11px; border-radius:50%;
         margin-right:.35rem; vertical-align:-1px; }
.refusal { border:1px solid var(--line); border-radius:8px; padding:.8rem;
           background:var(--card); font-size:.85rem; }
.refusal pre { white-space:pre-wrap; font-size:.78rem; color:var(--muted);
               margin:.4rem 0 0; }

/* -- cards -- */
details.card { border:1px solid var(--line); border-radius:8px; margin:.5rem 0;
               background:var(--card); }
details.card > summary { cursor:pointer; padding:.6rem .8rem; font-size:.92rem;
                         list-style:none; display:flex; flex-wrap:wrap;
                         gap:.2rem 1.1rem; align-items:baseline; }
details.card > summary::-webkit-details-marker { display:none; }
details.card > summary b { font-size:1rem; }
summary .chip { font-size:.78rem; color:var(--muted);
                font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.cardbody { padding:0 .8rem .9rem; display:grid;
            grid-template-columns:minmax(300px,1fr) minmax(280px,1fr); gap:1rem; }
@media (max-width:820px) { .cardbody { grid-template-columns:1fr; } }
.vbar { display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:.5rem; }
.vbar select, .vbar button { font:inherit; font-size:.82rem; padding:.28rem .5rem;
    border:1px solid var(--line); border-radius:5px; background:var(--bg);
    color:var(--fg); cursor:pointer; }
canvas.v-canvas { width:100%; height:400px; display:block; border-radius:6px;
                  background:linear-gradient(180deg,rgba(125,145,175,.10),
                                             rgba(125,145,175,.02)); }
.kv { display:grid; grid-template-columns:max-content 1fr; gap:.12rem .8rem;
      font-size:.85rem; }
.kv span:nth-child(odd) { color:var(--muted); }
.kv span:nth-child(even) { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.sites { max-height:300px; overflow:auto; border:1px solid var(--line);
         border-radius:6px; }
.sites th { font-size:11.5px; }
.sites td { font-size:11.5px; padding:.2rem .45rem; }
tr.re td { background:var(--model); }
p.lede { margin:0 .8rem .6rem; padding:.5rem .7rem; font-size:.88rem;
         border-left:3px solid var(--line); color:var(--fg);
         background:var(--head); border-radius:0 6px 6px 0; }
.warn { background:var(--warnbg); color:var(--warn); border-radius:6px;
        padding:.5rem .7rem; font-size:.84rem; margin:.5rem 0; }
.bar { display:inline-block; height:9px; border-radius:2px; background:#3b82f6;
       vertical-align:1px; }
"""

_JS = """
document.querySelectorAll('th[data-col]').forEach(function (th) {
  th.addEventListener('click', function () {
    var table = th.closest('table'), body = table.tBodies[0];
    var index = Array.prototype.indexOf.call(th.parentNode.children, th);
    var dir = th.dataset.dir === 'asc' ? -1 : 1;
    th.dataset.dir = dir === 1 ? 'asc' : 'desc';
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (a, b) {
      var x = a.cells[index].dataset.v, y = b.cells[index].dataset.v;
      var nx = parseFloat(x), ny = parseFloat(y);
      if (!isNaN(nx) && !isNaN(ny)) { return (nx - ny) * dir; }
      return String(x).localeCompare(String(y)) * dir;
    });
    rows.forEach(function (r) { body.appendChild(r); });
  });
});

// Mount a viewer the first time its card is opened. Mounting all of them up
// front makes the page take seconds to appear for no benefit: a reader opens
// one or two cards.
document.querySelectorAll('details.card').forEach(function (card) {
  function mount() {
    if (!card.open) { return; }
    var host = card.querySelector('.viewer');
    if (host) { window.CSP.mount(host, host.dataset.sid); }
  }
  card.addEventListener('toggle', mount);
  // A card that is already open when the page loads never fires `toggle`, so
  // it would show an empty box. Mount it now instead.
  mount();
});

// A point on a hull, or a row in the table, opens that structure's card.
function openCard(sid) {
  var card = document.getElementById('card-' + sid);
  if (!card) { return; }
  card.open = true;
  card.scrollIntoView({behavior: 'smooth', block: 'center'});
}
document.querySelectorAll('svg.hull circle.clickable').forEach(function (c) {
  c.addEventListener('click', function () { openCard(c.dataset.sid); });
});
document.querySelectorAll('tr[data-sid]').forEach(function (tr) {
  tr.style.cursor = 'pointer';
  tr.addEventListener('click', function () { openCard(tr.dataset.sid); });
});
"""


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "&mdash;"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1000 else f"{value:.1f}"
    return html.escape(str(value))


def _sort_value(value: Any) -> str:
    """What the sort compares. Missing sorts last under either direction."""
    if value is None or value == "":
        return "1e30"
    return str(value)


def _fmt(value: Any, spec: str = ".4f", unit: str = "") -> str:
    if value is None:
        return "&mdash;"
    try:
        text = format(float(value), spec)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return f"{text}{(' ' + unit) if unit else ''}"


# -- sections ---------------------------------------------------------------

def _header(title: str, summary: dict, rows: list[dict]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f"<h1>{html.escape(str(summary.get('campaign') or title))}</h1>"
            f"<p class=sub>{len(rows)} candidates &middot; "
            f"{summary['structures']:,} structures &middot; "
            f"{summary['compositions']:,} compositions &middot; "
            f"{summary['core_hours']:,.0f} core-hours &middot; {stamp}</p>")


def _funnel_section(yield_: dict, counts: Funnel) -> str:
    parts = []
    if yield_["compositions"] and yield_["requested"]:
        pct = 100.0 * yield_["produced"] / max(yield_["requested"], 1)
        parts.append(f"<h2>Generation</h2><p class=note>"
                     f"{yield_['produced']:,} structures produced of "
                     f"{yield_['requested']:,} requested ({pct:.1f}%); "
                     f"{yield_['short']} composition(s) short.</p>")
    parts.append("<h2>Funnel</h2><div class=scroll><table class=funnel><thead><tr>"
                 "<th>gate</th><th>seen</th><th>passed</th><th>rejected</th>"
                 "</tr></thead><tbody>")
    for gate, seen, passed in counts.rows:
        parts.append(f"<tr><td>{html.escape(gate)}</td><td>{seen:,}</td>"
                     f"<td class=kept>{passed:,}</td><td>{seen - passed:,}</td></tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _hull_section(store: Store) -> str:
    """Both hulls, one panel each, with the refusals printed in full."""
    try:
        panels = hullplot.panels(store)
    except Exception as exc:                                     # noqa: BLE001
        return (f"<h2>Convex hulls</h2><div class=refusal><b>Not drawn.</b>"
                f"<pre>{html.escape(f'{type(exc).__name__}: {exc}')}</pre></div>")
    if not panels:
        return ("<h2>Convex hulls</h2><p class=note>No structure carries an "
                "energy yet, so there is nothing to place.</p>")

    parts = ["<h2>Convex hulls</h2>",
             "<p class=note>Two hulls, never one. The MLIP panel is MatterSim "
             "on both sides; the DFT panel is our own VASP on both sides. The "
             "two scales differ by 0.15&ndash;0.21 eV/atom against a "
             "0.06 eV/atom selection threshold, so a single plot carrying both "
             "would put candidates below a hull they are nowhere near. Click a "
             "candidate to open its card.</p>",
             hullplot.legend_html()]
    for method in ("mlip", "dft"):
        chosen = [p for p in panels if p.method == method]
        if not chosen:
            continue
        parts.append(f"<h3>{hullplot.METHOD_LABEL[method]}</h3>")
        parts.append(f"<p class=note>{hullplot.METHOD_NOTE[method]}</p>")
        parts.append("<div class=hulls>")
        for panel in sorted(chosen, key=lambda p: (not p.ok, p.dimension > 3,
                                                   p.chemsys)):
            parts.append(hullplot.svg(panel))
        parts.append("</div>")
    return "".join(parts)


def _magnet_section(rows: list[dict], top: int = 20) -> str:
    """M, V, M/V and mu0*M, ranked, with a bar for the last of them."""
    scored = [r for r in rows if r.get("mu0_m_tesla") is not None]
    if not scored:
        return ("<h2>Magnetisation</h2><p class=note>No structure has both a "
                "cell magnetisation and a relaxed volume yet.</p>")
    scored.sort(key=lambda r: -float(r["mu0_m_tesla"]))
    shown = scored[:top]
    peak = max(float(r["mu0_m_tesla"]) for r in shown) or 1.0

    # Symbols follow the SI convention strictly, because the previous header did
    # not: it wrote "M" for the cell's total moment AND, two columns later, for
    # the magnetisation inside "mu_0 M".  Magnetisation is a moment per volume by
    # definition, so it can never carry units of mu_B/cell.  Here `m` is the
    # total moment, `M = m/V` is the magnetisation, and the last three columns
    # are one quantity in three units -- which the sub-labels now say outright.
    head = ("<tr><th data-col=id>id</th><th data-col=formula>formula</th>"
            "<th data-col=m>m<br><small>moment, &mu;<sub>B</sub>/cell</small></th>"
            "<th data-col=v>V<br><small>&Aring;<sup>3</sup></small></th>"
            "<th data-col=mv>M = m/V<br>"
            "<small>&mu;<sub>B</sub>/&Aring;<sup>3</sup></small></th>"
            "<th data-col=t>J = &mu;<sub>0</sub>M<br><small>T</small></th>"
            "<th data-col=emu>M<br><small>CGS, emu/cm<sup>3</sup></small></th>"
            "<th data-col=frac>vs Nd<sub>2</sub>Fe<sub>14</sub>B</th>"
            "<th data-col=hull>E<sub>hull</sub><br><small>DFT, eV/atom</small></th></tr>")
    body = []
    for row in shown:
        tesla = float(row["mu0_m_tesla"])
        width = max(2.0, 62.0 * tesla / peak)
        frac = tesla / ND2FE14B_JS_TESLA
        body.append(
            f'<tr data-sid="{row["id"]}">'
            f'<td data-v="{row["id"]}">{row["id"]}</td>'
            f'<td data-v="{html.escape(str(row["formula"]))}">{html.escape(str(row["formula"]))}</td>'
            f'<td data-v="{_sort_value(row.get("m_dft_raw"))}">{_fmt(row.get("m_dft_raw"), ".2f")}</td>'
            f'<td data-v="{_sort_value(row.get("volume"))}">{_fmt(row.get("volume"), ".1f")}</td>'
            f'<td data-v="{_sort_value(row.get("m_per_volume"))}">{_fmt(row.get("m_per_volume"), ".4f")}</td>'
            f'<td data-v="{tesla}"><span class=bar style="width:{width:.0f}px"></span> '
            f'{tesla:.3f}</td>'
            f'<td data-v="{_sort_value(row.get("m_emu_per_cc"))}">{_fmt(row.get("m_emu_per_cc"), ".0f")}</td>'
            f'<td data-v="{frac}">{frac * 100:.0f}%</td>'
            f'<td data-v="{_sort_value(row.get("dft_e_above_hull"))}">'
            f'{_fmt(row.get("dft_e_above_hull"), ".4f")}</td></tr>')

    return ("<h2>Magnetisation</h2>"
            f"<p class=note>The {len(shown)} highest of {len(scored)} structures "
            f"with both a moment and a volume, ranked by "
            f"&mu;<sub>0</sub>M &mdash; the saturation polarisation, which is the "
            f"number a permanent magnet is judged on. All of it comes from "
            f"<code>m_dft_raw</code>, the cell magnetisation VASP integrated; "
            f"the Hund&rsquo;s-rule reconstruction is NOT in this table. "
            f"Nd<sub>2</sub>Fe<sub>14</sub>B is {ND2FE14B_JS_TESLA:.2f} T for "
            f"scale.</p>"
            f"<div class=scroll><table><thead>{head}</thead><tbody>"
            f"{''.join(body)}</tbody></table></div>"
            "<p class=note>One magnetisation, three units: "
            "M = 1 &mu;<sub>B</sub>/&Aring;<sup>3</sup> = 9274 emu/cm<sup>3</sup> "
            "(CGS) = 9.274&times;10<sup>6</sup> A/m, and its polarisation "
            "J = &mu;<sub>0</sub>M = 11.654 T. Note the tesla column carries the "
            "&mu;<sub>0</sub> and the emu/cm<sup>3</sup> column does not, so the "
            "two are not the same quantity; the CGS counterpart of J is "
            "4&pi;M = 116,541 G. The percentage is "
            "&mu;<sub>0</sub>M only &mdash; it says nothing about anisotropy or "
            "Curie temperature, neither of which this pipeline computes.</p>")


def _candidate_table(rows: list[dict]) -> str:
    parts = ["<h2>Candidates</h2><div class=scroll><table><thead><tr>"]
    for name, meaning in COLUMNS:
        css = (" class=modelled" if name in MODELLED
               else " class=derived" if name in DERIVED else "")
        parts.append(f'<th{css} data-col="{name}" title="{html.escape(meaning)}">'
                     f"{html.escape(name)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append(f'<tr data-sid="{row.get("id")}">')
        for name, _ in COLUMNS:
            value = row.get(name)
            css = (" class=modelled" if name in MODELLED
                   else " class=derived" if name in DERIVED else "")
            parts.append(f'<td{css} data-v="{html.escape(_sort_value(value))}">'
                         f"{_cell(value)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _site_table(item: StructureDetail) -> str:
    if not item.sites:
        return ("<p class=note>No site-projected moments. "
                "<code>LORBIT = 11</code> writes them; without it the OUTCAR "
                "carries only the cell magnetisation and no sublattice split "
                "is possible.</p>")
    head = ("<tr><th>ion</th><th>el</th><th>s</th><th>p</th><th>d</th>"
            "<th>f</th><th>total</th></tr>")
    body = []
    for site in item.sites:
        css = " class=re" if site.is_rare_earth else ""
        body.append(f"<tr{css}><td>{site.index}</td><td>{html.escape(site.element)}</td>"
                    f"<td>{site.s:.3f}</td><td>{site.p:.3f}</td>"
                    f"<td>{site.d:.3f}</td><td>{site.f:.3f}</td>"
                    f"<td><b>{site.total:.3f}</b></td></tr>")
    return (f"<div class=sites><table><thead>{head}</thead><tbody>"
            f"{''.join(body)}</tbody></table></div>")


def _element_table(item: StructureDetail) -> str:
    if not item.elements:
        return ""
    head = ("<tr><th>element</th><th>n</th><th>sum</th><th>mean</th>"
            "<th>min</th><th>max</th><th>spread</th></tr>")
    body = []
    for e in item.elements:
        css = " class=re" if e["rare_earth"] else ""
        body.append(f"<tr{css}><td>{html.escape(e['element'])}</td><td>{e['n']}</td>"
                    f"<td><b>{e['sum']:+.3f}</b></td><td>{e['mean']:+.3f}</td>"
                    f"<td>{e['min']:+.3f}</td><td>{e['max']:+.3f}</td>"
                    f"<td>{e['spread']:.3f}</td></tr>")
    return (f"<table>{head}<tbody>{''.join(body)}</tbody></table>"
            f"<p class=note>All in &mu;<sub>B</sub>, summed over PAW spheres. "
            f"<b>spread</b> is max &minus; min over the ions of that element: a "
            f"large spread means inequivalent sites carry different moments, "
            f"which is the sublattice structure a single average would hide.</p>")


def _provenance_block(seed: SeedRecord | None) -> str:
    """Where the seed came from, beside what was computed from it.

    Absent for a campaign with no `seed_provenance.csv` -- anything generated
    rather than staged. Nothing is inferred from the formula or parsed out of
    the filename to fill the gap.
    """
    if seed is None:
        return ""
    rows = [
        ("parent", html.escape(seed.parent or "&mdash;")),
        ("what varies", html.escape(seed.what_varies or "&mdash;")),
        ("how it was made", html.escape(seed.method or "&mdash;")),
        ("staging route", html.escape(seed.source_tag or "&mdash;")),
        ("seed file", f"<code>{html.escape(seed.seed_file)}</code>"),
        ("copied from", f"<code>{html.escape(seed.origin_path)}</code>"
                        if seed.origin_path else "&mdash;"),
    ]
    if seed.md5:
        rows.append(("md5 of the seed", f"<code>{html.escape(seed.md5[:12])}&hellip;</code>"))
    body = "".join(f"<span>{k}</span><span>{v}</span>" for k, v in rows)
    return (f"<h3>Provenance</h3><div class=kv>{body}</div>"
            f"<p class=note>From the campaign&rsquo;s "
            f"<code>seed_provenance.csv</code>. This records what the staging "
            f"step was <em>told</em>, not a re-derivation from the structure: "
            f"it says what was intended, and the cell above is what VASP ended "
            f"on. The checksum is of the seed <b>as supplied</b>, so it does "
            f"not match the relaxed cell.</p>")


def _card(item: StructureDetail, row: dict,
          seed: SeedRecord | None = None) -> str:
    mag = item.magnetics or summarise(None, None)
    chips = [f"<span class=chip>id {item.structure_id}</span>",
             f"<span class=chip>{item.n_atoms} atoms</span>"]
    if item.spacegroup:
        chips.append(f"<span class=chip>{html.escape(item.spacegroup)}</span>")
    if row.get("dft_e_above_hull") is not None:
        chips.append(f"<span class=chip>E<sub>hull</sub> "
                     f"{float(row['dft_e_above_hull']):+.4f} eV/atom</span>")
    if mag.mu0_m is not None:
        chips.append(f"<span class=chip>&mu;<sub>0</sub>M {mag.mu0_m:.3f} T</span>")

    warn = ""
    if item.geometry_source != "dft-contcar":
        warn = ("<div class=warn><b>This is not the DFT-relaxed cell.</b> The "
                "geometry drawn is the MLIP-relaxed one from the database, "
                "because the VASP output directory could not be read. Every "
                "number beside it still comes from the VASP run; only the "
                "picture is of the earlier cell.</div>")
    for note in item.notes:
        warn += f"<div class=warn>{html.escape(note)}</div>"

    # The description: what this structure is, and what was changed to make
    # it. First thing in the card, because a formula is not an experiment.
    lede = ""
    if seed is not None:
        lede = f"<p class=lede>{html.escape(seed.sentence())}</p>"

    lattice = item.lattice
    kv = [
        ("geometry shown", {"dft-contcar": "DFT-relaxed CONTCAR",
                            "database-mlip": "MLIP-relaxed (database)"}.get(
                                item.geometry_source, item.geometry_source or "&mdash;")),
        ("a, b, c (&Aring;)", f"{lattice.get('a', 0):.4f}, {lattice.get('b', 0):.4f}, "
                              f"{lattice.get('c', 0):.4f}"),
        ("&alpha;, &beta;, &gamma; (&deg;)",
         f"{lattice.get('alpha', 0):.2f}, {lattice.get('beta', 0):.2f}, "
         f"{lattice.get('gamma', 0):.2f}"),
        ("V", _fmt(item.volume, ".2f", "&Aring;<sup>3</sup>")),
        ("m (cell, computed)", _fmt(item.m_cell, ".3f", "&mu;<sub>B</sub>")),
        ("m (PAW spheres)", _fmt(item.m_spheres, ".3f", "&mu;<sub>B</sub>")),
        ("cell &minus; spheres", _fmt(item.sphere_deficit, ".3f", "&mu;<sub>B</sub>")),
        ("M = m / V", _fmt(mag.m_per_volume, ".4f",
                           "&mu;<sub>B</sub>/&Aring;<sup>3</sup>")),
        ("J = &mu;<sub>0</sub>M", _fmt(mag.mu0_m, ".4f", "T")),
        ("M (CGS)", _fmt(mag.emu_per_cc, ".1f", "emu/cm<sup>3</sup>")),
        ("m per formula unit", _fmt(mag.m_per_formula_unit, ".3f",
                                    "&mu;<sub>B</sub>")),
        ("M<sub>s</sub> reconstructed", _fmt(item.m_s_reconstructed, ".3f",
                                             "&mu;<sub>B</sub> &mdash; MODELLED")),
        ("4f treatment", html.escape(item.f_treatment or "&mdash;")),
    ]
    kv_html = "".join(f"<span>{k}</span><span>{v}</span>" for k, v in kv)

    return (
        f'<details class=card id="card-{item.structure_id}">'
        f"<summary><b>{html.escape(item.formula)}</b>{''.join(chips)}</summary>"
        f"{lede}"
        f"<div class=cardbody>"
        f'<div><div class=viewer data-sid="{item.structure_id}"></div>'
        f"<p class=note>Drag to rotate, scroll to zoom. Bonds are drawn where "
        f"two atoms are closer than 1.25&times; the sum of their covalent "
        f"radii &mdash; a drawing convention, not a calculated bond.</p>"
        f"{warn}</div>"
        f"<div><h3>Numbers</h3><div class=kv>{kv_html}</div>"
        f"{_provenance_block(seed)}"
        f"<h3>By element</h3>{_element_table(item)}"
        f"<h3>Per site</h3>{_site_table(item)}</div>"
        f"</div></details>")


def _cards_section(store: Store, rows: list[dict], n: int,
                   campaign_dir: Path | None = None
                   ) -> tuple[str, dict[int, dict], dict[str, dict]]:
    """The detail cards, plus the geometry payloads the viewer reads."""
    seeds = prov.load(campaign_dir or prov.campaign_dir_of(store.path))
    chosen = rows[:n]
    payloads: dict[int, dict] = {}
    symbols: set[str] = set()
    cards: list[str] = []
    failures: list[str] = []

    for row in chosen:
        sid = int(row["id"])
        try:
            item = detail(store, sid)
        except Exception as exc:                                 # noqa: BLE001
            failures.append(f"{sid}: {type(exc).__name__}: {exc}")
            continue
        payloads[sid] = item.payload()
        symbols.update(item.symbols)
        seed = prov.for_structure(seeds, {"source_path": item.source_path})
        cards.append(_card(item, row, seed))

    header = (f"<h2>Structures</h2><p class=note>The {len(cards)} best "
              f"candidates by hull distance, each with the cell the VASP "
              f"relaxation ended on, where its seed came from, and every "
              f"ion&rsquo;s projected moment. Cards are collapsed so the page "
              f"opens immediately; the 3D view is built when you open one.</p>"
              + _routes_note(seeds))
    if failures:
        header += ("<div class=warn>Could not build a card for: "
                   + html.escape("; ".join(failures)) + "</div>")
    return header + "".join(cards), payloads, element_table(symbols)


def _routes_note(seeds: dict[str, SeedRecord]) -> str:
    """How the campaign's seeds were made, by route, in one table.

    Worth the six lines it takes: a substitution campaign staged from several
    libraries exists in order to ask WHICH route produced the better
    candidates, and that question cannot be asked from a page that does not
    say a candidate came from one.
    """
    rows = prov.summary(seeds)
    if not rows:
        return ("<p class=note>No <code>seed_provenance.csv</code> in this "
                "campaign, so the cards carry no provenance. That file is "
                "written when seeds are staged into <code>inputs/</code>; a "
                "generated campaign has none, and nothing here is inferred "
                "from a formula or a filename to fill the gap.</p>")

    body = "".join(
        f"<tr><td>{html.escape(r['tag'] or '&mdash;')}</td>"
        f"<td>{html.escape(r['method'] or '&mdash;')}</td>"
        f"<td>{html.escape(r['what_varies'] or '&mdash;')}</td>"
        f"<td>{r['n']}</td></tr>" for r in rows)
    return (f"<div class=scroll><table><thead><tr><th>staging route</th>"
            f"<th>how the seeds were made</th><th>what varies</th>"
            f"<th>seeds</th></tr></thead><tbody>{body}</tbody></table></div>"
            f"<p class=note>Every seed this campaign staged, by route &mdash; "
            f"not only the ones with a card. The campaign was run this way to "
            f"ask which route produced the better candidates, so each "
            f"card names the route its structure came from.</p>")


def _glossary() -> str:
    parts = ["<h2>Columns</h2><dl>"]
    for name, meaning in COLUMNS:
        parts.append(f"<dt>{html.escape(name)}</dt><dd>{html.escape(meaning)}</dd>")
    parts.append("</dl>")
    parts.append(
        "<p class=note>Shaded orange columns are modelled, not computed; shaded "
        "blue ones are arithmetic on two computed columns. "
        "<code>m_dft_raw</code> is the cell magnetisation VASP reports; "
        "<code>m_s_reconstructed</code> adds the Hund&rsquo;s-rule 4f "
        "moment back to the transition-metal sublattice, which a "
        "frozen-4f POTCAR leaves out. For a heavy rare earth the two "
        "routinely differ by more than a factor of two and often in "
        "sign. They are never merged.</p>")
    return "".join(parts)


# -- assembly ---------------------------------------------------------------

def render(store: Store, *, title: str = "cspflow", limit: int | None = 2000,
           detail_cards: int = DEFAULT_DETAIL, hulls: bool = True,
           jsmol_url: str | None = None,
           campaign_dir: Path | None = None) -> str:
    rows = candidate_rows(store, limit=limit)
    counts = funnel(store)
    summary = store.summary()
    yield_ = store.generation_yield()

    cards, payloads, elements = ("", {}, {})
    if detail_cards:
        cards, payloads, elements = _cards_section(store, rows, detail_cards,
                                                   campaign_dir)

    parts = [f"<title>{html.escape(title)}</title>",
             f"<style>{_CSS}</style>",
             _header(title, summary, rows),
             _funnel_section(yield_, counts)]
    if hulls:
        parts.append(_hull_section(store))
    parts.append(_magnet_section(rows))
    parts.append(_candidate_table(rows))
    if cards:
        parts.append(cards)
    parts.append(_glossary())

    if jsmol_url:
        parts.append(f'<script src="{html.escape(jsmol_url)}/JSmol.min.js"></script>')
    parts.append("<script>window.CSP = {structures: "
                 + json.dumps({str(k): v for k, v in payloads.items()},
                              separators=(",", ":"))
                 + ", elements: " + json.dumps(elements, separators=(",", ":"))
                 + ", jsmolUrl: " + json.dumps(jsmol_url) + "};</script>")
    parts.append(f"<script>{VIEWER_JS}</script>")
    parts.append(f"<script>{_JS}</script>")
    return "\n".join(parts)


def write(store: Store, path: Path, *, title: str = "cspflow",
          limit: int | None = 2000, detail_cards: int = DEFAULT_DETAIL,
          hulls: bool = True, jsmol_url: str | None = None,
          campaign_dir: Path | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(store, title=title, limit=limit,
                           detail_cards=detail_cards, hulls=hulls,
                           jsmol_url=jsmol_url, campaign_dir=campaign_dir))
    return path
