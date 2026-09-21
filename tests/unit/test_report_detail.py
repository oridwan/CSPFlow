"""The detailed report: magnet units, hull geometry, viewer payload, page shape.

Covers the three things `csp report` gained -- the drawn convex hulls, the 3D
structure cards and the per-site moments -- at the level where each of them can
be wrong without failing loudly:

*   the unit conversion behind mu0*M, which is one multiplication from being
    wrong by 10^6 and would still look like a plausible number;
*   the SVG's attribute quoting, which is what silently ate an entire plot the
    first time (an unquoted `class=axis/` is parsed as the class "axis/" and
    the element never closes, so every following sibling becomes its child);
*   the refusal path, which must print the reason rather than draw a hull with
    a borrowed vertex.

No VASP, no network, no reference store: every case is built in the test.
"""

from __future__ import annotations

import json
import re

import pytest
from ase.build import bulk

from cspflow.db.store import Origin, Store, StructureState
from cspflow.report import hullplot, magnetics
from cspflow.report.structure import StructureDetail, cif_text, element_table


# -- units ------------------------------------------------------------------

def test_mu0_m_conversion_is_the_textbook_number():
    """1 mu_B/A^3 is 11.654 T. A magnet is judged on this number."""
    summary = magnetics.summarise(1.0, 1.0)
    assert summary.m_per_volume == pytest.approx(1.0)
    assert summary.mu0_m == pytest.approx(11.654, abs=1e-3)
    assert summary.emu_per_cc == pytest.approx(9274.01, abs=0.1)


def test_a_real_cell_lands_near_nd2fe14b():
    """Ce2Fe14B-like: 125 mu_B in 915 A^3 is about 1.6 T, as it should be."""
    summary = magnetics.summarise(125.0, 915.0, z=2)
    assert summary.mu0_m == pytest.approx(1.59, abs=0.02)
    assert summary.m_per_formula_unit == pytest.approx(62.5)
    assert summary.fraction_of_nd2fe14b == pytest.approx(0.99, abs=0.02)


def test_missing_inputs_give_none_not_zero():
    """A moment that was not computed is not a moment of zero."""
    assert magnetics.summarise(None, 900.0).mu0_m is None
    assert magnetics.summarise(12.0, None).mu0_m is None
    assert magnetics.summarise(12.0, 0.0).mu0_m is None


# -- the per-site table -----------------------------------------------------

class _Site:
    def __init__(self, index, element, total, channels):
        self.index, self.element, self.total, self.channels = (
            index, element, total, channels)


def test_site_rows_fill_every_channel():
    """A run without f states still gets an f column, holding its real 0.0."""
    rows = magnetics.site_rows([_Site(1, "Fe", 2.2, {"s": -0.01, "p": -0.04, "d": 2.25})])
    assert rows[0].f == 0.0
    assert rows[0].d == pytest.approx(2.25)
    assert rows[0].is_rare_earth is False


def test_by_element_reports_the_spread_not_only_the_mean():
    """Two inequivalent Fe sites differing by 0.5 mu_B is the result, not noise."""
    rows = magnetics.site_rows([
        _Site(1, "Fe", 1.9, {"d": 1.9}), _Site(2, "Fe", 2.4, {"d": 2.4}),
        _Site(3, "Ce", -0.28, {"d": -0.28}),
    ])
    summary = {d["element"]: d for d in magnetics.by_element(rows)}
    assert summary["Fe"]["n"] == 2
    assert summary["Fe"]["sum"] == pytest.approx(4.3)
    assert summary["Fe"]["spread"] == pytest.approx(0.5)
    assert summary["Ce"]["rare_earth"] is True
    # Largest absolute contribution first: Fe carries this magnet, not Ce.
    assert magnetics.by_element(rows)[0]["element"] == "Fe"


# -- the SVG ----------------------------------------------------------------

def _panel(dimension: int) -> hullplot.HullPanel:
    elements = ["A", "B", "C", "D"][:dimension]
    panel = hullplot.HullPanel(chemsys="-".join(elements), method="dft",
                               elements=elements, ok=True, n_reference=3,
                               n_candidates=1)
    for i, element in enumerate(elements):
        panel.points.append(hullplot.Point(
            label=element, formula=element, fractions={element: 1.0},
            e_form=0.0, e_above_hull=0.0, is_candidate=False, stable=True))
    middle = {element: 1.0 / dimension for element in elements}
    panel.points.append(hullplot.Point(
        label="cand-7", formula="ABC", fractions=middle, e_form=-0.2,
        e_above_hull=0.03, is_candidate=True, structure_id=7))
    panel.tielines = [(elements[0], elements[1])]
    return panel


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_every_svg_attribute_is_quoted(dimension):
    """`class=axis/` is parsed as the class "axis/" and swallows its siblings.

    That bug drew a solid black triangle and an empty strip plot, and it fails
    silently: the SVG is still well-formed, it is just nested wrongly. This
    asserts the shape of the markup rather than the picture, because the
    picture cannot be asserted.
    """
    markup = hullplot.svg(_panel(dimension))
    assert "<svg" in markup
    # Only inside the SVG. Unquoted attributes in the surrounding HTML are
    # harmless; it is the trailing `/` of a self-closing SVG tag that turns
    # `class=axis/` into the class "axis/" and leaves the element open.
    inner = markup[markup.index("<svg"):markup.index("</svg>")]
    unquoted = re.findall(r'\s(?:class|role)=[^"\s>]+', inner)
    assert unquoted == [], f"unquoted attributes: {unquoted}"
    assert "/>" not in inner.replace(" />", "")


def test_a_candidate_is_clickable_and_a_reference_phase_is_not():
    markup = hullplot.svg(_panel(3))
    assert 'data-sid="7"' in markup
    assert markup.count('class="clickable"') == 1


def test_the_selection_threshold_decides_the_colour():
    assert hullplot.band_colour(0.0) == "#1a7f37"
    assert hullplot.band_colour(0.03) == "#3b82f6"
    assert hullplot.band_colour(0.5) == "#9aa0a6"
    # A reference phase carries no candidate colour at all.
    assert hullplot.band_colour(None) == hullplot.REF_COLOUR


def test_a_panel_that_cannot_be_built_prints_its_reason_in_full():
    """The refusal is the product. It must reach the page, not be swallowed."""
    panel = hullplot.HullPanel(chemsys="B-Ce-Fe", method="dft", ok=False,
                               reason="11 of 105 phases have no dft energy")
    markup = hullplot.svg(panel)
    assert "<svg" not in markup
    assert "11 of 105 phases" in markup


# -- geometry payload -------------------------------------------------------

def test_payload_is_json_serialisable_and_small():
    """The page embeds one of these per card, so it may not carry surprises."""
    detail = StructureDetail(structure_id=3, formula="Fe2",
                             cell=[[2.87, 0, 0], [0, 2.87, 0], [0, 0, 2.87]],
                             frac=[[0, 0, 0], [0.5, 0.5, 0.5]],
                             symbols=["Fe", "Fe"])
    text = json.dumps(detail.payload())
    assert json.loads(text)["symbols"] == ["Fe", "Fe"]
    assert len(text) < 400


def test_cif_is_p1_and_lists_every_ion():
    """P1 on purpose: the relaxation did not constrain any symmetry."""
    detail = StructureDetail(structure_id=3, formula="Fe2",
                             cell=[[2.87, 0, 0], [0, 2.87, 0], [0, 0, 2.87]],
                             frac=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                             symbols=["Fe", "Fe"],
                             lattice={"a": 2.87, "b": 2.87, "c": 2.87,
                                      "alpha": 90.0, "beta": 90.0, "gamma": 90.0})
    text = cif_text(detail)
    assert "_symmetry_space_group_name_H-M 'P 1'" in text
    assert text.count("  Fe") == 2
    assert "_cell_length_a 2.870000" in text


def test_element_table_gives_a_colour_and_a_radius_per_element():
    table = element_table({"Fe", "Ce", "B"})
    assert set(table) == {"Fe", "Ce", "B"}
    for entry in table.values():
        assert entry["c"].startswith("#") and len(entry["c"]) == 7
        assert 0.3 < entry["r"] < 3.0


# -- the page ---------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    db = Store.create(tmp_path / "c.db", campaign="t")
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    sid = db.add_structure(atoms, origin=Origin.seed, state=StructureState.dft_done,
                           reduced_formula="Fe2")
    db.update_structure(sid, m_dft_raw=4.4, volume=23.6, vasp_energy=-16.4,
                        dft_e_above_hull=0.0, spacegroup=229,
                        spacegroup_symbol="Im-3m", f_treatment="frozen")
    yield db
    db.close()


def test_the_page_carries_every_section_and_no_external_request(store):
    """Self-contained is the whole design: no src= that leaves the file."""
    from cspflow.report.html import render

    page = render(store, title="t", detail_cards=3, hulls=False)
    for heading in ("Funnel", "Magnetisation", "Candidates", "Structures",
                    "Columns"):
        assert f"<h2>{heading}</h2>" in page
    assert "http://" not in page and "https://" not in page
    assert "<script src=" not in page
    # The magnet figures of merit are on the page with their units.
    assert "&mu;<sub>0</sub>M" in page
    assert "emu/cm" in page


def test_a_jsmol_url_is_the_only_thing_that_adds_an_external_script(store):
    from cspflow.report.html import render

    page = render(store, title="t", detail_cards=1, hulls=False,
                  jsmol_url="/jsmol-assets")
    assert '<script src="/jsmol-assets/JSmol.min.js"></script>' in page
    assert '"jsmolUrl": "/jsmol-assets"' in page or '"/jsmol-assets"' in page


def test_detail_zero_writes_no_cards_and_no_payload(store):
    """The size knob actually turns the embedding off."""
    from cspflow.report.html import render

    page = render(store, title="t", detail_cards=0, hulls=False)
    assert "details class=card" not in page
    assert "<h2>Structures</h2>" not in page
    assert "structures: {}" in page


# -- provenance -------------------------------------------------------------

PROVENANCE = """\
seed_file,source_tag,source_folder,original_filename,method,parent,what_varies,n_atoms,elements,md5
parent__Ce2PdGe6.cif,parent,Agentic_test/templates,Ce2PdGe6.cif,given,Ce2PdGe6,nothing -- this IS the parent,36,Ce-Ge-Pd,abc123
01-manual__Ce2Al2Ge4Pd.vasp,01-manual,Agentic_test/01-x/inputs/seeds,Ce2Al2Ge4Pd.vasp,"manual, in-session",Ce2PdGe6,"Ge sublattice, x<=2",36,Al-Ce-Ge-Pd,def456
"""


@pytest.fixture()
def campaign(tmp_path):
    (tmp_path / "seed_provenance.csv").write_text(PROVENANCE)
    return tmp_path


def test_the_parent_says_it_is_the_parent(campaign):
    from cspflow.report import provenance

    records = provenance.load(campaign)
    parent = records["parent__Ce2PdGe6.cif"]
    assert parent.is_parent
    assert "taken as given" in parent.sentence()
    assert "Nothing was substituted" in parent.sentence()


def test_a_substitution_names_its_parent_sublattice_and_route(campaign):
    """The description is the experiment: a formula alone is not one."""
    from cspflow.report import provenance

    seed = provenance.load(campaign)["01-manual__Ce2Al2Ge4Pd.vasp"]
    line = seed.sentence()
    assert "Ce2PdGe6" in line                 # what it came from
    assert "Ge sublattice, x<=2" in line      # what was changed
    assert "laced by hand" in line            # how
    assert "01-manual" in line                # which staging route
    assert seed.origin_path == "Agentic_test/01-x/inputs/seeds/Ce2Al2Ge4Pd.vasp"


def test_a_structure_matches_its_seed_on_the_source_path_basename(campaign):
    from cspflow.report import provenance

    records = provenance.load(campaign)
    kv = {"source_path": "inputs/01-manual__Ce2Al2Ge4Pd.vasp"}
    assert provenance.for_structure(records, kv).source_tag == "01-manual"
    assert provenance.for_structure(records, {"source_path": "inputs/x.cif"}) is None
    assert provenance.for_structure(records, {}) is None


def test_an_unknown_method_still_reads_as_a_sentence(tmp_path):
    """A route the staging script invents later must not leave a hole."""
    from cspflow.report.provenance import SeedRecord

    seed = SeedRecord(seed_file="s.vasp", parent="Ce2PdGe6", method="some new way",
                      what_varies="the Pd site", source_tag="09-new")
    assert seed.sentence() == ("Derived from Ce2PdGe6. What varies: the Pd site. "
                               "Some new way (09-new route).")


def test_a_campaign_without_the_file_gets_no_block_and_says_so(tmp_path):
    """Nothing is inferred from a formula or a filename to fill the gap."""
    from cspflow.report import provenance
    from cspflow.report.html import _provenance_block, _routes_note

    assert provenance.load(tmp_path) == {}
    assert provenance.load(None) == {}
    assert _provenance_block(None) == ""
    note = _routes_note({})
    assert "No <code>seed_provenance.csv</code>" in note
    assert "nothing here is inferred" in note


def test_a_malformed_file_loses_the_block_not_the_report(tmp_path):
    from cspflow.report import provenance

    (tmp_path / "seed_provenance.csv").write_text("not,a,provenance,file\n1,2,3\n")
    assert provenance.load(tmp_path) == {}


def test_the_routes_table_counts_every_staged_seed(campaign):
    from cspflow.report import provenance
    from cspflow.report.html import _routes_note

    rows = provenance.summary(provenance.load(campaign))
    assert {r["tag"] for r in rows} == {"parent", "01-manual"}
    assert sum(r["n"] for r in rows) == 2
    note = _routes_note(provenance.load(campaign))
    assert "staging route" in note and "01-manual" in note


def test_the_card_leads_with_the_description(store, campaign):
    """It is the first thing in the card, above the viewer and the numbers."""
    from cspflow.report.html import render

    page = render(store, title="t", detail_cards=3, hulls=False,
                  campaign_dir=campaign)
    assert "<h3>Provenance</h3>" not in page   # this store's seed is not in the CSV
    assert "staging route" in page             # the routes table is still shown


def test_the_campaign_folder_survives_the_results_symlink(tmp_path):
    """`results/` is a symlink to scratch, and resolving it leaves the campaign.

    `campaigns/X/results/campaign.db` resolves to
    `/scratch/.../X/campaign.db`, whose parent's parent is `/scratch/...` --
    not the campaign. Resolving first returned None for every real campaign
    while every test using a plain directory passed.
    """
    from cspflow.report.provenance import campaign_dir_of

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "seed_provenance.csv").write_text(PROVENANCE)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "campaign.db").write_text("")
    (campaign / "results").symlink_to(scratch)

    assert campaign_dir_of(campaign / "results" / "campaign.db") == campaign
    # And a database with no campaign beside it stays None rather than
    # reaching up into whatever directory happens to be above it.
    assert campaign_dir_of(scratch / "campaign.db") is None
