"""What the reference store knows, per chemical system and per element.

The question this answers is the one that decides where DFT is worth spending:
**in which chemistries does the MLIP reproduce the reference ordering at all?**
On the four adopted families that turned out to be an element-level property --
9 of 11 Gd systems failed while Sm and Tb passed -- so it is asked per element,
not once for the campaign.

Two comparisons, kept apart because they are not the same measurement:

*   ``static``  -- MLIP at the reference geometry.  Isolates energy error from
    geometry error, which have opposite consequences: a uniform energy offset
    largely cancels along a hull tie-line, a volume bias does not.
*   ``relaxed`` -- MLIP at its own minimum.  The like-for-like number, because
    the reference energies are themselves relaxed.

Measured on Fe-Sm, the two disagree in a way that matters: static carries a
-21.8 meV/atom bias that is an artefact of comparing an unrelaxed energy with a
relaxed one, and relaxing removes it (+1.3 meV/atom).  Reporting only the first
would describe the comparison rather than the model.

``against='mp'`` measures the MLIP against Materials Project's own GGA energies.
That is cheap and available immediately, but MatterSim is trained on MPtrj --
which *is* MP -- so it is substantially a self-consistency check and catches
gross failure rather than subtle bias.  ``against='ours'`` uses our own DFT once
the reference build has run, and is the real test.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..calibrate.parity import ParityPoint, ParityReport, build_report
from .computed import ComputedPhase, load_all
from .mp import fetch_chemsys

Against = str            # 'mp' | 'ours'


@dataclass
class SystemRow:
    """One chemical system, measured both ways."""

    chemsys: str
    n_phases: int = 0
    n_scored: int = 0
    n_unconverged: int = 0
    static: ParityReport | None = None
    relaxed: ParityReport | None = None
    volume_drift_mean: float | None = None
    volume_drift_max: float | None = None

    @property
    def verdict(self) -> str:
        """Judged on the relaxed comparison -- the like-for-like one."""
        chosen = self.relaxed or self.static
        return chosen.verdict if chosen else "no data"


@dataclass
class StoreReport:
    recipe_id: str
    against: Against
    systems: list[SystemRow] = field(default_factory=list)
    per_element: dict[str, ParityReport] = field(default_factory=dict)
    n_phases: int = 0
    n_with_mlip: int = 0
    n_with_dft: int = 0

    def render(self) -> str:
        lines = [
            f"reference store report -- MLIP vs {'our DFT' if self.against == 'ours' else 'MP GGA'}",
            f"recipe {self.recipe_id[:16]}",
            f"{self.n_phases:,} phases: {self.n_with_mlip:,} with MLIP energies, "
            f"{self.n_with_dft:,} with our DFT",
            "",
            f"{'element':<8}{'n':>6}{'MAE':>9}{'bias':>9}{'rho':>8}  verdict",
        ]
        for element, rep in sorted(self.per_element.items(),
                                   key=lambda kv: -(kv[1].spearman if kv[1].spearman
                                                    is not None else -9)):
            rho = f"{rep.spearman:.3f}" if rep.spearman is not None else "  --"
            lines.append(f"{element:<8}{rep.n:>6}{rep.mae_e_per_atom * 1000:>8.1f} "
                         f"{rep.bias_e_per_atom * 1000:>+8.1f} {rho:>8}  {rep.verdict}")
            # A verdict with no reason is not actionable, and "warn" on a good
            # Spearman usually means the geometry, not the energy.
            for reason in rep.reasons:
                lines.append(f"{'':<8}{reason}")
        bad = [s for s in self.systems if s.verdict == "fail"]
        lines += ["", f"{len(bad)} of {len(self.systems)} systems fail"]
        if bad:
            lines.append("  " + ", ".join(s.chemsys for s in bad[:20])
                         + (" ..." if len(bad) > 20 else ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------


def _points(phases: Sequence[ComputedPhase], reference: dict[str, float],
            attr: str) -> list[ParityPoint]:
    """ParityPoints for one comparison, per atom on both sides.

    `reference` is keyed by mp_id and holds energy PER ATOM; the stored MLIP
    energies are totals for the cell, so they are divided by the cell's own
    atom count and never by the reduced formula's.
    """
    out = []
    for phase in phases:
        energy = getattr(phase, attr)
        ref = reference.get(phase.mp_id)
        if energy is None or ref is None or not phase.n_atoms:
            continue
        out.append(ParityPoint(
            label=phase.mp_id, counts=dict(phase.counts),
            e_mlip_per_atom=energy / phase.n_atoms,
            e_dft_per_atom=ref,
            volume_mlip=phase.mlip_volume_final,
            volume_dft=phase.mlip_volume_initial,
        ))
    return out


def _reference_energies(chemsys: str, store: dict[str, ComputedPhase],
                        against: Against, thermo_type: str) -> dict[str, float]:
    """Per-atom reference energies for one system, on the requested scale."""
    if against == "ours":
        return {mp_id: p.e_dft_per_atom for mp_id, p in store.items()
                if p.e_dft_per_atom is not None}
    result = fetch_chemsys(chemsys, thermo_type=thermo_type, energy_scale="raw")
    return {e.mp_id: e.e_raw_per_atom for e in result.entries
            if e.e_raw_per_atom is not None}


def build(systems: Iterable[str], *, recipe_id_: str, against: Against = "mp",
          thermo_type: str = "GGA_GGA+U", thresholds: dict[str, float] | None = None,
          ) -> StoreReport:
    store = load_all(recipe_id_)
    limits = {"mae_max": 0.05, "spearman_min": 0.90, "volume_drift_max": 0.05}
    limits.update(thresholds or {})

    report = StoreReport(recipe_id=recipe_id_, against=against)
    report.n_phases = len(store)
    report.n_with_mlip = sum(1 for p in store.values() if p.e_mlip_relaxed is not None
                             or p.e_mlip_static is not None)
    report.n_with_dft = sum(1 for p in store.values() if p.e_dft is not None)

    pooled: dict[str, list[ParityPoint]] = {}
    for chemsys in sorted(set(systems)):
        try:
            reference = _reference_energies(chemsys, store, against, thermo_type)
        except Exception:                                     # noqa: BLE001
            continue
        members = [p for p in store.values() if p.mp_id in reference]
        if not members:
            continue
        row = SystemRow(chemsys=chemsys, n_phases=len(members))
        row.n_unconverged = sum(1 for p in members
                                if p.e_mlip_relaxed is not None and not p.mlip_converged)

        for attr, slot in (("e_mlip_static", "static"), ("e_mlip_relaxed", "relaxed")):
            points = _points(members, reference, attr)
            if not points:
                continue
            setattr(row, slot, build_report(
                points, mae_max=limits["mae_max"],
                spearman_min=limits["spearman_min"],
                volume_drift_max=limits["volume_drift_max"]))
            if slot == "relaxed":
                row.n_scored = len(points)
                # Pool by element for the per-element rollup. A ternary point
                # counts once for each of its elements, which is the question
                # being asked: "is the model reliable where this element is
                # present", not "how many phases are pure it".
                for point in points:
                    for element in point.counts:
                        pooled.setdefault(element, []).append(point)

        drifts = [p.mlip_volume_drift for p in members if p.mlip_volume_drift is not None]
        if drifts:
            row.volume_drift_mean = sum(drifts) / len(drifts)
            row.volume_drift_max = max(drifts, key=abs)
        report.systems.append(row)

    for element, points in pooled.items():
        # Deduplicate: a phase shared by many systems would otherwise be
        # weighted by how many systems happen to contain it.
        unique = {p.label: p for p in points}
        report.per_element[element] = build_report(
            list(unique.values()), mae_max=limits["mae_max"],
            spearman_min=limits["spearman_min"],
            volume_drift_max=limits["volume_drift_max"])
    return report


# --------------------------------------------------------------------------
# One self-contained HTML file -- no external requests, no build step.
# --------------------------------------------------------------------------

_CSS = """
:root { --fg:#1a1a1a; --bg:#fff; --line:#d8d8d8; --muted:#666; --head:#f4f4f6;
        --pass:#1a7f37; --warn:#9a6700; --fail:#b62324; --band:#fafafa; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --fg:#e8e8e8; --bg:#16181c; --line:#333; --muted:#9aa0a6; --head:#22252b;
    --pass:#4ac26b; --warn:#d4a72c; --fail:#f2666b; --band:#1b1e23; }
}
:root[data-theme="dark"] {
  --fg:#e8e8e8; --bg:#16181c; --line:#333; --muted:#9aa0a6; --head:#22252b;
  --pass:#4ac26b; --warn:#d4a72c; --fail:#f2666b; --band:#1b1e23;
}
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); margin:0; padding:2rem 1.5rem;
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
h1 { font-size:1.4rem; margin:0 0 .2rem; }
h2 { font-size:1.05rem; margin:2rem 0 .6rem; font-weight:600; }
.sub { color:var(--muted); margin:0 0 1.2rem; font-size:.9rem; }
.note { color:var(--muted); font-size:.85rem; margin:.5rem 0 0; max-width:62ch; }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:6px; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th,td { padding:.32rem .55rem; border-bottom:1px solid var(--line);
        text-align:right; white-space:nowrap; }
th { background:var(--head); position:sticky; top:0; cursor:pointer;
     user-select:none; font-weight:600; }
th:first-child, td:first-child { text-align:left; }
td:last-child { text-align:left; white-space:normal; color:var(--muted);
                font-size:12px; min-width:22ch; }
tbody tr:nth-child(even) { background:var(--band); }
.pass { color:var(--pass); font-weight:600; }
.warn { color:var(--warn); font-weight:600; }
.fail { color:var(--fail); font-weight:600; }
.group { border-left:2px solid var(--line); }
"""

_SORT = """
document.querySelectorAll('th[data-col]').forEach(function (th) {
  th.addEventListener('click', function () {
    var table = th.closest('table'), body = table.tBodies[0];
    var i = Array.prototype.indexOf.call(th.parentNode.children, th);
    var dir = th.dataset.dir === 'asc' ? -1 : 1;
    th.dataset.dir = dir === 1 ? 'asc' : 'desc';
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (a, b) {
      var x = a.cells[i].dataset.v, y = b.cells[i].dataset.v;
      var nx = parseFloat(x), ny = parseFloat(y);
      if (!isNaN(nx) && !isNaN(ny)) return (nx - ny) * dir;
      return String(x).localeCompare(String(y)) * dir;
    });
    rows.forEach(function (r) { body.appendChild(r); });
  });
});
"""


def _num(value: Any, scale: float = 1.0, digits: int = 1, sign: bool = False) -> str:
    if value is None:
        return "&mdash;"
    fmt = f"{{:{'+' if sign else ''}.{digits}f}}"
    return fmt.format(value * scale)


def _sortable(value: Any) -> str:
    return "1e30" if value is None else str(value)


def _cells(rep: ParityReport | None) -> str:
    if rep is None:
        return "".join(f'<td data-v="1e30">&mdash;</td>' for _ in range(3))
    return (f'<td data-v="{_sortable(rep.mae_e_per_atom)}">'
            f'{_num(rep.mae_e_per_atom, 1000)}</td>'
            f'<td data-v="{_sortable(rep.bias_e_per_atom)}">'
            f'{_num(rep.bias_e_per_atom, 1000, sign=True)}</td>'
            f'<td data-v="{_sortable(rep.spearman)}">{_num(rep.spearman, 1, 3)}</td>')


def render_html(report: StoreReport, *, title: str = "cspflow reference") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    against = "our own DFT" if report.against == "ours" else "Materials Project GGA"
    counts = {v: sum(1 for s in report.systems if s.verdict == v)
              for v in ("pass", "warn", "fail")}

    parts = [
        f"<title>{html.escape(title)}</title>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<style>{_CSS}</style>",
        "<h1>MLIP calibration against the reference set</h1>",
        f"<p class=sub>{report.n_phases:,} phases &middot; "
        f"{report.n_with_mlip:,} with MLIP energies &middot; "
        f"{report.n_with_dft:,} with our DFT &middot; "
        f"measured against {against} &middot; recipe "
        f"{html.escape(report.recipe_id[:16])} &middot; {stamp}</p>",
        f"<p class=sub><span class=pass>{counts['pass']} pass</span> &middot; "
        f"<span class=warn>{counts['warn']} warn</span> &middot; "
        f"<span class=fail>{counts['fail']} fail</span> "
        f"of {len(report.systems)} systems</p>",
    ]

    parts.append("<h2>By element</h2><div class=scroll><table><thead><tr>"
                 "<th data-col=el>element</th><th data-col=n>phases</th>"
                 "<th data-col=mae>MAE meV/at</th><th data-col=bias>bias meV/at</th>"
                 "<th data-col=rho>Spearman</th><th data-col=v>verdict</th>"
                 "<th data-col=why>why</th>"
                 "</tr></thead><tbody>")
    for element, rep in sorted(report.per_element.items(),
                               key=lambda kv: (kv[1].spearman is None,
                                               -(kv[1].spearman or 0))):
        parts.append(
            f'<tr><td data-v="{html.escape(element)}">{html.escape(element)}</td>'
            f'<td data-v="{rep.n}">{rep.n:,}</td>'
            f'<td data-v="{_sortable(rep.mae_e_per_atom)}">{_num(rep.mae_e_per_atom, 1000)}</td>'
            f'<td data-v="{_sortable(rep.bias_e_per_atom)}">'
            f'{_num(rep.bias_e_per_atom, 1000, sign=True)}</td>'
            f'<td data-v="{_sortable(rep.spearman)}">{_num(rep.spearman, 1, 3)}</td>'
            f'<td data-v="{rep.verdict}" class={rep.verdict}>{rep.verdict}</td>'
            f'<td data-v="{html.escape(rep.reasons[0] if rep.reasons else "")}">'
            f'{html.escape("; ".join(rep.reasons)) or "&mdash;"}</td></tr>')
    parts.append("</tbody></table></div>")
    parts.append("<p class=note>A phase counts once for every element it contains, "
                 "and is deduplicated across the systems that share it. The question "
                 "is whether the model is reliable where an element is present, not "
                 "how many phases are pure it.</p>")

    parts.append("<h2>By chemical system</h2><div class=scroll><table><thead><tr>"
                 "<th data-col=sys>system</th><th data-col=n>phases</th>"
                 "<th class=group data-col=sm>MAE</th><th data-col=sb>bias</th>"
                 "<th data-col=sr>&rho;</th>"
                 "<th class=group data-col=rm>MAE</th><th data-col=rb>bias</th>"
                 "<th data-col=rr>&rho;</th>"
                 "<th class=group data-col=vd>vol drift</th>"
                 "<th data-col=unc>unconv</th><th data-col=v>verdict</th>"
                 "</tr></thead><tbody>")
    for row in sorted(report.systems, key=lambda s: (
            {"fail": 0, "warn": 1, "pass": 2}.get(s.verdict, 3), s.chemsys)):
        parts.append(
            f'<tr><td data-v="{html.escape(row.chemsys)}">{html.escape(row.chemsys)}</td>'
            f'<td data-v="{row.n_phases}">{row.n_phases}</td>'
            + _cells(row.static) + _cells(row.relaxed)
            + f'<td data-v="{_sortable(row.volume_drift_max)}">'
              f'{_num(row.volume_drift_max, 100, 2, sign=True)}%</td>'
              f'<td data-v="{row.n_unconverged}">{row.n_unconverged or ""}</td>'
              f'<td data-v="{row.verdict}" class={row.verdict}>{row.verdict}</td></tr>')
    parts.append("</tbody></table></div>")
    parts.append(
        "<p class=note>The first MAE/bias/&rho; group is the <b>single point at the "
        "reference geometry</b>: it separates energy error from geometry error, "
        "which have opposite consequences &mdash; a uniform energy offset largely "
        "cancels along a hull tie-line, a volume bias does not. The second group is "
        "the <b>MLIP relaxed to its own minimum</b>, which is the like-for-like "
        "number because the reference energies are themselves relaxed. The verdict "
        "is judged on the relaxed comparison. They are never merged.</p>")
    if report.against == "mp":
        parts.append(
            "<p class=note><b>Read this as a floor, not a verdict.</b> MatterSim is "
            "trained on MPtrj, which is Materials Project data, so measuring it "
            "against MP is substantially a self-consistency check: it catches gross "
            "failure and not subtle bias. A system that fails here will certainly "
            "fail on generated structures; one that passes has not yet been "
            "tested where the campaign actually operates.</p>")
    parts.append(f"<script>{_SORT}</script>")
    return "\n".join(parts)


def write_html(report: StoreReport, path: Path, *,
               title: str = "cspflow reference") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report, title=title))
    return path
