"""POTCAR resolution, identity, and the 4f convention."""

from pathlib import Path

import pytest

from cspflow.config.schema import FTreatment, Machine
from cspflow.dft.vasp.potcar import (
    PotcarError,
    assert_one_f_convention,
    ensure_pmg_layout,
    f_in_valence,
    max_enmax,
    potcar_symbols,
    read_potcar,
    tree_directory,
)

TREE = Path("/projects/mmi/cspflow-shared/potcars/VASP6.4/potpaw_PBE")
has_tree = pytest.mark.skipif(not TREE.is_dir(), reason="local POTCAR tree not present")


# --- symbol selection ------------------------------------------------------


def test_frozen_applies_to_the_whole_series():
    """Including Gd, Eu and Ce -- deliberately diverging from MPRelaxSet, which
    frozen-izes most of the series but leaves those three in valence."""
    syms = potcar_symbols(["Sm", "Gd", "Eu", "Ce", "Tb"], f_treatment=FTreatment.frozen)
    assert syms == {"Sm": "Sm_3", "Gd": "Gd_3", "Eu": "Eu_3", "Ce": "Ce_3", "Tb": "Tb_3"}


def test_valence_gives_bare_symbols():
    syms = potcar_symbols(["Sm", "Gd"], f_treatment=FTreatment.valence)
    assert syms == {"Sm": "Sm", "Gd": "Gd"}


def test_transition_metals_use_the_explicit_table():
    syms = potcar_symbols(["Fe", "Co", "Ti", "Y"])
    assert syms == {"Fe": "Fe_pv", "Co": "Co", "Ti": "Ti_pv", "Y": "Y_sv"}


def test_overrides_always_win():
    syms = potcar_symbols(["Fe", "Gd"], f_treatment=FTreatment.frozen,
                          overrides={"Fe": "Fe", "Gd": "Gd"})
    assert syms == {"Fe": "Fe", "Gd": "Gd"}


def test_lanthanum_has_no_frozen_variant_because_it_has_no_4f():
    """La is 4f0, so VASP ships no `La_3` and the bare symbol IS the frozen
    convention.  Without this a La campaign fails at its first structure with a
    missing-POTCAR error rather than at configuration time."""
    syms = potcar_symbols(["La", "Sm", "Gd"], f_treatment=FTreatment.frozen)
    assert syms == {"La": "La", "Sm": "Sm_3", "Gd": "Gd_3"}


def test_lanthanum_counts_as_frozen_not_as_f_in_valence():
    """The suffix heuristic reads bare `La` as f-in-valence and is wrong: there
    are no f electrons to put there.  ZVAL 11 (5s2 5p6 5d1 6s2) is in the
    frozen-core range; every genuine f-in-valence dataset is 14-20.  Left
    uncorrected, La trips `assert_one_f_convention` against its own correctly
    frozen Sm_3 and Gd_3."""
    assert f_in_valence("La", "La", 11.0) is False


def test_yttrium_is_not_a_rare_earth_here():
    """Y is group 3 with no f electrons, so the 4f machinery must not touch it."""
    assert potcar_symbols(["Y"], f_treatment=FTreatment.frozen)["Y"] == "Y_sv"


# --- f-in-valence detection ------------------------------------------------


def test_suffix_decides_the_convention():
    assert f_in_valence("Gd", "Gd_3", 9.0) is False
    assert f_in_valence("Gd", "Gd", 18.0) is True
    assert f_in_valence("Eu", "Eu_2", 8.0) is False


def test_non_rare_earths_are_never_f_in_valence():
    assert f_in_valence("Fe", "Fe_pv", 14.0) is False


def test_cerium_overlaps_the_ranges_and_must_still_resolve():
    """Bare Ce carries one 4f electron: ZVAL 12 against Ce_3 at 11.  A
    lower-bound ZVAL test would reject a perfectly good POTCAR."""
    assert f_in_valence("Ce", "Ce", 12.0) is True
    assert f_in_valence("Ce", "Ce_3", 11.0) is False


def test_a_file_that_contradicts_its_name_is_caught():
    with pytest.raises(PotcarError, match="does not match its name"):
        f_in_valence("Gd", "Gd_3", 18.0)


# --- against the real trees ------------------------------------------------


@has_tree
@pytest.mark.parametrize(
    "symbol,zval,f_val",
    [("Sm_3", 11.0, False), ("Sm", 16.0, True), ("Gd_3", 9.0, False),
     ("Gd", 18.0, True), ("Ce_3", 11.0, False), ("Ce", 12.0, True),
     ("Fe_pv", 14.0, False)],
)
def test_real_potcar_headers(symbol, zval, f_val):
    element = symbol.split("_")[0]
    info = read_potcar(TREE, symbol, element)
    assert info.zval == zval
    assert info.n_f_valence is f_val
    assert info.titel.startswith("PAW_PBE")
    assert len(info.md5_header_hash) >= 8


@has_tree
def test_missing_symbol_lists_what_is_available():
    with pytest.raises(PotcarError, match="Candidates present"):
        read_potcar(TREE, "Fe_nonsense", "Fe")


@has_tree
def test_mp_own_mixed_convention_is_refused():
    """MP runs Gd2Fe17 with 4f in valence and Tb2Fe17 with it frozen."""
    infos = [read_potcar(TREE, "Gd", "Gd"), read_potcar(TREE, "Tb_3", "Tb")]
    with pytest.raises(PotcarError, match="mixed 4f conventions"):
        assert_one_f_convention(infos)


@has_tree
def test_consistent_series_passes():
    infos = [read_potcar(TREE, s, s.split("_")[0]) for s in ("Gd_3", "Tb_3", "Sm_3")]
    assert_one_f_convention(infos)


@has_tree
def test_max_enmax_is_composition_dependent():
    """The measured reason ENCUT must be explicit: VASP's default would be this
    number, and it changes with what is in the cell."""
    smfeti = [read_potcar(TREE, s, e) for s, e in
              (("Sm_3", "Sm"), ("Fe", "Fe"), ("Ti", "Ti"))]
    with_cu = smfeti + [read_potcar(TREE, "Cu", "Cu")]
    assert max_enmax(smfeti) == pytest.approx(267.882, abs=1e-3)
    assert max_enmax(with_cu) == pytest.approx(295.446, abs=1e-3)
    assert max_enmax(with_cu) > max_enmax(smfeti)


# --- tree layout -----------------------------------------------------------


def _machine(root):
    return Machine.model_validate({
        "potcar_root": str(root),
        "potcar_dirs": {"PBE_64": "POT_PAW_PBE_64"},
        "potcar_trees": {"VASP6.4": "PBE_64"},
    })


def test_tree_directory_resolution(tmp_path):
    _, d = tree_directory(_machine(tmp_path), "VASP6.4")
    assert d == tmp_path / "POT_PAW_PBE_64"


def test_unknown_tree_lists_known(tmp_path):
    with pytest.raises(PotcarError, match="known:"):
        tree_directory(_machine(tmp_path), "VASP9.9")


def test_no_potcar_root_is_an_error():
    with pytest.raises(PotcarError, match="no potcar_root"):
        tree_directory(Machine(), "VASP6.4")


def test_layout_reports_missing_then_creates_it(tmp_path):
    real = tmp_path / "real_tree"
    (real / "Fe").mkdir(parents=True)
    m = _machine(tmp_path / "pmg")

    before = ensure_pmg_layout(m, {"PBE_64": str(real)}, create=False)
    assert any(s == "fail" for s, _ in before)

    after = ensure_pmg_layout(m, {"PBE_64": str(real)}, create=True)
    assert any(s == "fixed" for s, _ in after)
    assert (tmp_path / "pmg" / "POT_PAW_PBE_64").is_symlink()

    again = ensure_pmg_layout(m, {"PBE_64": str(real)}, create=False)
    assert all(s == "ok" for s, _ in again)


def test_layout_flags_a_symlink_pointing_somewhere_else(tmp_path):
    real, other = tmp_path / "real", tmp_path / "other"
    real.mkdir(); other.mkdir()
    root = tmp_path / "pmg"; root.mkdir()
    (root / "POT_PAW_PBE_64").symlink_to(other)
    m = _machine(root)
    out = ensure_pmg_layout(m, {"PBE_64": str(real)}, create=True)
    assert any(s == "fail" and "points at" in msg for s, msg in out)
