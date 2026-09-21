"""The 4f-treatment check: is the moment computed, or is it bookkeeping?

`dft.rare_earth.f_treatment: frozen` runs each rare earth on its `_3` POTCAR,
with the 4f electrons in the core.  That is the right choice for ENERGIES -- one
convention across the series, which Materials Project does not have (D109) --
but VASP then never computes a 4f moment.  `reconstruct_ms: true` adds a nominal
spin-only value back when reporting, which is a constant per rare-earth atom.

So a cell whose only magnetic species is the rare earth has NO computed moment,
and its reported moment varies only with composition.  Measured in the reference
store on 443 Ce phases run at exactly these settings:

    Ce with an Fe/Co/Ni/Mn/Cr partner   median 0.001, 90th pct 1.17 uB/atom
    Ce with no magnetic 3d              87% below 0.05 uB/atom, median 0.000

ISPIN=2 was confirmed on 357 of 358 sampled static runs, so those zeros are the
physics and not a missing tag.
"""

from cspflow.config.schema import Campaign
from cspflow.doctor import check_magnetism_is_computed


class FakeCfg:
    def __init__(self, campaign):
        self.campaign = campaign


def cfg(f_treatment="frozen"):
    return FakeCfg(Campaign.model_validate({
        "name": "t",
        "machine": "local",
        "workdir": "/tmp/t",
        "source": [{"mode": "structure_list",
                    "structure_list": {"paths": ["seeds"]}}],
        "dft": {"recipe": "magnets",
                "rare_earth": {"f_treatment": f_treatment,
                               "reconstruct_ms": True}},
    }))


def test_a_rare_earth_only_campaign_is_warned():
    check = check_magnetism_is_computed(cfg(), ["Ce", "Pd", "Ge"])
    assert check.status == "warn"
    assert "Ce" in check.detail
    assert any("ranks them by composition" in r for r in check.rows)


def test_a_campaign_with_a_magnetic_3d_partner_is_fine():
    check = check_magnetism_is_computed(cfg(), ["Ce", "Pd", "Ge", "Fe"])
    assert check.status == "ok"
    assert "Fe" in check.detail


def test_it_still_says_the_4f_part_is_reconstructed():
    """Fine is not the same as complete: the Ce moment is still not computed."""
    check = check_magnetism_is_computed(cfg(), ["Ce", "Fe", "B"])
    assert check.status == "ok"
    assert any("reconstructed, not computed" in r for r in check.rows)


def test_valence_treatment_needs_no_warning():
    check = check_magnetism_is_computed(cfg("valence"), ["Ce", "Pd", "Ge"])
    assert check.status == "ok"
    assert "valence" in check.detail


def test_a_campaign_with_nothing_magnetic_is_not_warned():
    """The warning must not become background noise on a non-magnetic system."""
    check = check_magnetism_is_computed(cfg(), ["Pd", "Ge", "Si"])
    assert check.status == "ok"


def test_it_is_skipped_before_the_elements_are_known():
    check = check_magnetism_is_computed(cfg(), None)
    assert check.status == "skip"
