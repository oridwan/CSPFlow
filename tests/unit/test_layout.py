"""Where DFT work lives, what it is called, and which layout actually applies.

WHY THE NAMES CARRY CHEMISTRY.  `dft-18-relax` says nothing about the structure.
The seed it came from was `02-agentic__Ce2Fe11Co3B_x3_o5-7_Co.vasp` -- which says
what it IS -- and that was discarded at submission.  It also collides: every
campaign numbers its structures from 1, so `dft-69-relax` exists in CeFeB AND in
CePdGe with different contents, and a tool matching on directory name pairs one
campaign's job with another's folder.

WHY BOTH LAYOUTS EXIST.  CeFeB and CePdGe are mid-flight on the flat `stages`
arrangement.  Switching a running campaign would point every path at an empty
directory: nothing done, and 15 GB of finished work re-run.
"""

from pathlib import Path

import pytest

from cspflow.dft.layout import (
    Layout,
    resolve,
    run_slug,
    seed_label,
    slugify,
    structure_id_of,
)

SEED = "/seeds/02-agentic__Ce2Fe11Co3B_x3_o5-7_Co.vasp"


# --- naming ----------------------------------------------------------------


def test_the_slug_carries_id_formula_and_provenance():
    got = run_slug(18, "Ce8Co12Fe44B4", SEED)
    assert got.startswith("0018-")
    assert "Ce8Co12Fe44B4" in got
    assert "Ce2Fe11Co3B_x3_o5-7_Co" in got


def test_the_id_is_zero_padded_so_ls_sorts_numerically():
    """With a bare id, `ls` puts 100 between 10 and 11 and a few-hundred
    structure campaign becomes unreadable exactly where you go to read it."""
    names = sorted(run_slug(i, "X") for i in (2, 10, 100, 7))
    assert [structure_id_of(n) for n in names] == [2, 7, 10, 100]


def test_the_source_mode_survives():
    """Dropping the word before `__` makes `parent__Ce2Fe14B.cif` and a
    substituted Ce2Fe14B indistinguishable -- the one comparison the campaign
    exists to make."""
    assert "parent" in run_slug(94, "Ce8Fe56B4", "/s/parent__Ce2Fe14B.cif")
    assert "strain" in run_slug(87, "Ce8Fe56B4",
                                "/s/02-strain__Ce2Fe14B_strainiso_m6.00pct.vasp")
    assert "agentic" in run_slug(18, "Ce8Co12Fe44B4", SEED)


def test_the_numeric_ordering_prefix_is_dropped():
    """Every seed in a batch shares `02-`, so it separates nothing."""
    assert not seed_label(SEED).startswith("02")


def test_two_campaigns_do_not_collide_on_the_same_id():
    """The real collision: structure 69 exists in both campaigns."""
    a = run_slug(69, "Ce8Fe56B4", "/s/02-strain__Ce2Fe14B.vasp")
    b = run_slug(69, "Ce4Pd4Ge24", "/s/02-agentic__Ce2PdGe6_Cu.vasp")
    assert a != b


def test_a_slug_is_safe_in_a_shell_and_a_glob():
    nasty = "/s/weird name (v2)__A*B?C;rm -rf.vasp"
    got = run_slug(1, "A B", nasty)
    assert not (set(got) & set(" */?;()'\"\\$`|&<>"))


def test_a_missing_seed_still_produces_a_usable_name():
    assert run_slug(7, "Ce8Fe56B4") == "0007-Ce8Fe56B4"
    assert run_slug(7) == "0007"


def test_slugs_are_length_capped():
    assert len(run_slug(1, "X" * 200, "/s/" + "y" * 300 + ".vasp")) <= 96


def test_slugify_never_returns_empty():
    assert slugify("***") and slugify("") and slugify(".vasp")


# --- path policy -----------------------------------------------------------


def test_runs_layout_groups_a_structure_together():
    lay = Layout("runs", Path("/w/dft"))
    rel = lay.stage_dir(18, "relax", "Ce8Co12Fe44B4", SEED)
    sta = lay.stage_dir(18, "static", "Ce8Co12Fe44B4", SEED)
    assert rel.parent == sta.parent, "both steps must live under one structure"
    assert rel.parent.parent.name == "runs"


def test_stages_layout_is_unchanged():
    """The old paths must be reproduced exactly; a running campaign depends on
    them, and 'close enough' means orphaning its finished work."""
    lay = Layout("stages", Path("/w/dft"))
    assert lay.stage_dir(18, "relax") == Path("/w/dft/dft-18-relax")
    assert lay.stage_dir(18, "static") == Path("/w/dft/dft-18-static")


def test_an_unknown_layout_is_refused():
    with pytest.raises(ValueError):
        Layout("whatever", Path("/w/dft"))


# --- which layout actually applies -----------------------------------------


def test_disk_evidence_beats_configuration(tmp_path):
    """The regression this guard exists for: a campaign mid-flight on `stages`
    must not be switched by a config default."""
    (tmp_path / "dft-18-relax").mkdir()
    (tmp_path / "dft-18-relax" / "OUTCAR").write_text("x")
    name, why = resolve("runs", tmp_path)
    assert name == "stages"
    assert "orphans" in why


def test_a_runs_tree_is_likewise_respected(tmp_path):
    (tmp_path / "runs" / "0001-X").mkdir(parents=True)
    name, why = resolve("stages", tmp_path)
    assert name == "runs" and why


def test_a_fresh_campaign_takes_what_it_was_told(tmp_path):
    assert resolve("runs", tmp_path / "nothing-here")[0] == "runs"
    assert resolve("stages", tmp_path / "nothing-here")[0] == "stages"


def test_no_override_is_reported_when_none_happened(tmp_path):
    assert resolve("runs", tmp_path)[1] == ""


def test_the_schema_default_is_one_directory_per_structure():
    """`runs` is the default (D135).

    It was `stages` while CeFeB and CePdGe were mid-flight without a `layout:`
    key, because re-pointing a running campaign orphans its finished work. Those
    campaigns are retired; an old one is still protected by `resolve()`, which
    lets the directories on disk override this -- see
    `test_disk_evidence_overrides_a_runs_setting`.
    """
    from cspflow.config.schema import Dft

    assert Dft().layout == "runs"
