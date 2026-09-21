"""M2 -- recipes, INCAR, KPOINTS and job-directory assembly.

Several of these check cspflow against the legacy campaign's own input files,
which is the only independent evidence that the port is faithful. Where they
disagree, the test says which is which and why.
"""

import json
import math
from pathlib import Path

import pytest
import yaml
from ase.build import bulk

from cspflow.config.schema import Dft, Ldau, Machine, Magnetism, RareEarth
from cspflow.dft.recipe import (
    Kpoints,
    Recipe,
    RecipeError,
    RecipeStage,
    build_recipe,
    load_recipe,
    validate_recipe,
)
from cspflow.dft.vasp.incar import (
    FERRI_RETM,
    IncarContext,
    IncarError,
    build_incar,
    ldau_block,
    lmaxmix_for,
    magmom_for,
    nbands_auto,
    render_incar,
)
from cspflow.dft.vasp.kpoints import KpointsError, grid_for

MACHINES = Path(__file__).resolve().parents[2] / "src" / "cspflow" / "machines"
LEGACY = Path("/projects/mmi/shuo/redo-new-ter-mag/VASP_JOBS/Gd1Co10Cr2/Gd1Co10Cr2_s020/Relax")
has_legacy = pytest.mark.skipif(not LEGACY.is_dir(), reason="legacy campaign not present")


@pytest.fixture
def orion() -> Machine:
    return Machine(**yaml.safe_load((MACHINES / "orion.yaml").read_text()))


def minimal_incar(**extra):
    base = {"ENCUT": 520, "ISPIN": 2, "LASPH": ".TRUE.", "NELM": 200, "LORBIT": 11}
    base.update(extra)
    return base


# --------------------------------------------------------------------------
# Recipes
# --------------------------------------------------------------------------


class TestRecipe:
    def test_inherit_copies_it_does_not_defer(self):
        """A resolved stage holds every tag literally, so --dry-run can print it."""
        recipe = build_recipe({"stages": [
            {"name": "relax", "incar": minimal_incar(NSW=99, ISMEAR=1)},
            {"name": "static", "inherit": "relax", "incar": {"NSW": 0, "ISMEAR": -5}},
        ]})
        static = recipe.stage("static")
        assert static.incar["ENCUT"] == 520          # inherited
        assert static.incar["NSW"] == 0              # overridden
        assert static.incar["ISMEAR"] == -5

    def test_inheriting_forwards_is_refused(self):
        """Single-pass resolution, so no cycle is possible."""
        with pytest.raises(RecipeError, match="not defined above it"):
            build_recipe({"stages": [
                {"name": "a", "inherit": "b", "incar": {}},
                {"name": "b", "incar": {}},
            ]})

    def test_resources_and_retry_are_inherited_too(self):
        recipe = build_recipe({"stages": [
            {"name": "relax", "incar": minimal_incar(),
             "resources": {"ntasks": 64}, "retry": [{"when": "timeout"}]},
            {"name": "static", "inherit": "relax", "incar": {}},
        ]})
        static = recipe.stage("static")
        assert static.resources["ntasks"] == 64
        assert [r["when"] for r in static.retry] == ["timeout"]

    def test_a_missing_stage_names_the_ones_that_exist(self):
        recipe = build_recipe({"stages": [{"name": "relax", "incar": minimal_incar()}]})
        with pytest.raises(RecipeError, match=r"\['relax'\]"):
            recipe.stage("soc")

    def test_no_stages_is_an_error(self):
        with pytest.raises(RecipeError, match="no stages"):
            build_recipe({"stages": []})

    def test_an_unknown_recipe_lists_the_shipped_ones(self):
        with pytest.raises(RecipeError, match="Shipped recipes"):
            load_recipe("no-such-recipe")

    def test_a_campaign_local_recipe_resolves_from_any_directory(self, tmp_path, monkeypatch):
        """`recipe: my.yaml` means the campaign's copy, wherever you ran csp from.

        The path is resolved against `base_dir`, and three call sites used to
        omit it. They worked only when the shell happened to be sitting in the
        campaign folder, and raised "no recipe at my.yaml" from anywhere else --
        a failure that depends on the user's cwd, not on the campaign.
        """
        campaign = tmp_path / "campaign"
        campaign.mkdir()
        (campaign / "my.yaml").write_text(yaml.safe_dump(
            {"name": "mine", "stages": [{"name": "relax", "incar": minimal_incar()}]}))

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        assert load_recipe("my.yaml", campaign).stage_names == ["relax"]
        with pytest.raises(RecipeError, match="no recipe at"):
            load_recipe("my.yaml")          # without base_dir it cannot be found


class TestRecipeValidation:
    @pytest.mark.parametrize(
        "tag, fragment",
        [
            ("ENCUT", "CHANGES WITH COMPOSITION"),
            ("ISPIN", "every moment would be zero"),
            ("LASPH", "aspherical"),
            ("NELM", "gives up early"),
            ("LORBIT", "nothing to analyse"),
        ],
    )
    def test_a_missing_tag_is_refused_with_what_vasp_would_do(self, tag, fragment):
        incar = minimal_incar()
        del incar[tag]
        with pytest.raises(RecipeError, match=fragment):
            validate_recipe(build_recipe({"stages": [{"name": "x", "incar": incar}]}))

    def test_lmaxmix_may_be_omitted_because_it_is_computed(self):
        """From the POTCARs -- more reliable than asking the user to keep it in step."""
        validate_recipe(build_recipe({"stages": [{"name": "x", "incar": minimal_incar()}]}))

    def test_an_unknown_tag_warns_but_is_not_dropped(self):
        """"Add any tag you like" has to remain true, so there is no whitelist."""
        warnings = validate_recipe(build_recipe({"stages": [
            {"name": "x", "incar": minimal_incar(MYTAG=1)}]}))
        assert any("MYTAG" in w for w in warnings)

    def test_the_yaml_on_trap_is_caught(self):
        """YAML 1.1 parses a bare `on` as True, so `- on: timeout` loses its key."""
        warnings = validate_recipe(build_recipe({"stages": [
            {"name": "x", "incar": minimal_incar(), "retry": [{True: "timeout"}]}]}))
        assert any("YAML parses a bare `on`" in w for w in warnings)

    def test_the_shipped_magnets_recipe_validates_cleanly(self):
        assert validate_recipe(load_recipe("magnets")) == []

    def test_the_shipped_recipe_has_relax_then_static(self):
        assert load_recipe("magnets").stage_names == ["relax", "static"]


# --------------------------------------------------------------------------
# MAGMOM
# --------------------------------------------------------------------------


class TestMagmom:
    def test_the_retm_convention_is_re_negative_tm_positive(self):
        moments = magmom_for(["Gd", "Co"], Magnetism(), RareEarth())
        assert moments[0] < 0 < moments[1]

    def test_early_3d_metals_are_antiparallel_to_late_3d(self):
        """Ported from the campaign's own table: 'early 3d TM = small negative'."""
        for early in ("Ti", "V", "Cr"):
            assert FERRI_RETM[early] < 0, early
        for late in ("Fe", "Co", "Ni"):
            assert FERRI_RETM[late] > 0, late

    def test_it_reproduces_the_campaigns_own_magmom(self):
        """Gd1Co10Cr2 was run with `MAGMOM = 2*-7.0 4*-2.0 20*2.0`."""
        symbols = ["Gd"] * 2 + ["Cr"] * 4 + ["Co"] * 20
        moments = magmom_for(symbols, Magnetism(), RareEarth())
        assert moments == [-7.0] * 2 + [-2.0] * 4 + [2.0] * 20

    def test_ferro_flips_the_rare_earth_parallel(self):
        """The one-line version of the old-vs-redo campaign difference."""
        ferri = magmom_for(["Gd", "Co"], Magnetism(), RareEarth(magnetic_order="ferri"))
        ferro = magmom_for(["Gd", "Co"], Magnetism(), RareEarth(magnetic_order="ferro"))
        assert ferri[0] < 0 and ferro[0] > 0
        assert abs(ferri[0]) == abs(ferro[0])

    def test_mode_none_writes_nothing(self):
        assert magmom_for(["Fe"], Magnetism(mode="none")) is None

    def test_strict_refuses_an_element_with_no_entry(self):
        """A silent default moment converges to the wrong magnetic state quietly."""
        with pytest.raises(IncarError, match="no initial moment"):
            magmom_for(["Xe"], Magnetism(mode="table", table={"Fe": 2.0}))

    def test_non_strict_falls_back_and_says_what_it_used(self):
        moments = magmom_for(["Xe"], Magnetism(mode="table", table={"Fe": 2.0},
                                               strict=False))
        assert moments == [0.6]

    def test_site_overrides_win(self):
        moments = magmom_for(["Gd", "Co"], Magnetism(site_overrides={"Gd": -3.5}),
                             RareEarth())
        assert moments[0] == -3.5

    def test_a_user_table_replaces_the_preset(self):
        moments = magmom_for(["Fe"], Magnetism(mode="table", table={"Fe": 5.0}))
        assert moments == [5.0]


# --------------------------------------------------------------------------
# NBANDS, LMAXMIX, LDAU
# --------------------------------------------------------------------------


class TestComputedTags:
    def test_nbands_uses_the_ported_formula(self):
        context = IncarContext(symbols=["Fe"] * 4, zvals={"Fe": 8.0})
        # nelect 32; max(16 + max(2,10), 19) = 26 -> ceil to 28 at NCORE 4
        assert nbands_auto(context, ncore=4) == 28

    def test_nbands_rounds_up_to_a_multiple_of_ncore(self):
        context = IncarContext(symbols=["Fe"] * 10, zvals={"Fe": 8.0})
        assert nbands_auto(context, ncore=8) % 8 == 0

    def test_nbands_is_none_without_zvals(self):
        assert nbands_auto(IncarContext(symbols=["Fe"])) is None

    def test_lmaxmix_is_6_with_f_in_valence_and_4_otherwise(self):
        assert lmaxmix_for(True) == 6
        assert lmaxmix_for(False) == 4

    def test_ldau_is_empty_when_disabled(self):
        assert ldau_block(Ldau(), ["Fe"]) == {}

    def test_ldau_refuses_a_missing_u(self):
        """A missing entry would silently become 0 and mix U with non-U results."""
        with pytest.raises(IncarError, match="no U is given"):
            ldau_block(Ldau(enabled=True, u={"Fe": 4.0}), ["Fe", "Co"])

    def test_ldau_arrays_follow_poscar_element_order(self):
        block = ldau_block(Ldau(enabled=True, u={"Fe": 4.0, "Co": 0.0}), ["Co", "Fe"])
        assert block["LDAUU"] == [0.0, 4.0]      # Co first, as given
        assert block["LDAUL"] == [-1, 2]


class TestBuildIncar:
    def _context(self):
        return IncarContext(symbols=["Gd", "Co", "Co"], formula="Co2Gd1",
                            zvals={"Gd": 9.0, "Co": 9.0}, f_in_valence=False)

    def test_system_defaults_to_the_formula(self):
        incar = build_incar(minimal_incar(), self._context())
        assert incar["SYSTEM"] == "Co2Gd1"

    def test_lmaxmix_is_added_from_the_potcars(self):
        assert build_incar(minimal_incar(), self._context())["LMAXMIX"] == 4

    def test_an_explicit_lmaxmix_in_the_recipe_wins(self):
        assert build_incar(minimal_incar(LMAXMIX=6), self._context())["LMAXMIX"] == 6

    def test_magmom_is_skipped_when_ispin_is_1(self):
        incar = build_incar(minimal_incar(ISPIN=1), self._context(),
                            magnetism=Magnetism(), rare_earth=RareEarth())
        assert "MAGMOM" not in incar

    def test_campaign_overrides_win_over_everything(self):
        incar = build_incar(minimal_incar(), self._context(),
                            overrides={"ENCUT": 700, "NEW_TAG": 1})
        assert incar["ENCUT"] == 700 and incar["NEW_TAG"] == 1


class TestRenderIncar:
    def test_python_booleans_become_vasp_booleans(self):
        """A user writing `LASPH: true` in YAML gets a Python bool here."""
        assert "LASPH = .TRUE." in render_incar({"LASPH": True})
        assert "LWAVE = .FALSE." in render_incar({"LWAVE": False})

    def test_magmom_is_run_length_encoded(self):
        text = render_incar({"MAGMOM": [-7.0, -7.0, 2.0, 2.0, 2.0]})
        assert "MAGMOM = 2*-7 3*2" in text

    def test_a_single_run_still_encodes(self):
        assert "MAGMOM = 3*2" in render_incar({"MAGMOM": [2.0, 2.0, 2.0]})

    def test_lists_other_than_magmom_are_space_separated(self):
        assert "LDAUU = 4 0" in render_incar({"LDAUU": [4.0, 0.0]})

    def test_a_comment_is_prefixed(self):
        assert render_incar({"ENCUT": 520}, comment="hello").startswith("! hello")


# --------------------------------------------------------------------------
# KPOINTS
# --------------------------------------------------------------------------


class TestKpoints:
    def test_reciprocal_density_reproduces_the_campaigns_grid(self):
        """Gd1Co10Cr2_s020: a,b,c = 4.6399, 8.1658, 8.2433 A -> `5 3 3`."""
        grid = grid_for([4.6399342629, 8.1658250943, 8.2432836376],
                        Kpoints("reciprocal_density", 64))
        assert (grid.a, grid.b, grid.c) == (5, 3, 3)

    def test_the_grid_is_independent_of_the_atom_count(self):
        """V_recip * V_cell is (2*pi)^3 identically, so n_atoms cancels -- which
        is what makes the sampling comparable across the cell sizes a hull spans."""
        lengths = [4.64, 8.17, 8.24]
        small = grid_for(lengths, Kpoints("reciprocal_density", 64), n_atoms=2)
        large = grid_for(lengths, Kpoints("reciprocal_density", 64), n_atoms=200)
        assert (small.a, small.b, small.c) == (large.a, large.b, large.c)

    def test_a_denser_setting_gives_a_denser_grid(self):
        lengths = [5.0, 5.0, 5.0]
        coarse = grid_for(lengths, Kpoints("reciprocal_density", 64))
        fine = grid_for(lengths, Kpoints("reciprocal_density", 512))
        assert fine.total > coarse.total

    def test_kspacing(self):
        grid = grid_for([4.6399, 8.1658, 8.2433], Kpoints("kspacing", 0.3))
        assert (grid.a, grid.b, grid.c) == (5, 3, 3)

    def test_explicit(self):
        grid = grid_for([5.0, 5.0, 5.0], Kpoints("explicit", [3, 3, 2]))
        assert (grid.a, grid.b, grid.c) == (3, 3, 2)

    def test_explicit_needs_three_numbers(self):
        with pytest.raises(KpointsError, match="three-element"):
            grid_for([5.0] * 3, Kpoints("explicit", [3, 3]))

    def test_an_unknown_scheme_names_the_known_ones(self):
        with pytest.raises(KpointsError, match="known:"):
            grid_for([5.0] * 3, Kpoints("magic", 1))

    def test_the_grid_is_never_zero(self):
        grid = grid_for([500.0, 500.0, 500.0], Kpoints("reciprocal_density", 1))
        assert grid.a >= 1 and grid.b >= 1 and grid.c >= 1

    def test_gamma_centred_by_default(self):
        """Preserves the point symmetry; an even Monkhorst-Pack grid does not."""
        assert "Gamma" in grid_for([5.0] * 3, Kpoints("reciprocal_density", 64)).render()


# --------------------------------------------------------------------------
# Whole job directories
# --------------------------------------------------------------------------


class TestJobDirectory:
    def _resolve(self, orion, atoms, **kw):
        from cspflow.dft.vasp.inputs import resolve_inputs

        return resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                              Dft(recipe="magnets", **kw), orion)

    @pytest.mark.skipif(not (MACHINES / "orion.yaml").is_file(), reason="no profile")
    def test_species_order_is_first_appearance_not_alphabetical(self, orion):
        """POTCAR concatenation and the LDAU arrays both key on this order."""
        from cspflow.dft.vasp.inputs import _species_order

        atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
        atoms.symbols = ["Ni", "Co"]
        assert _species_order(atoms) == ["Ni", "Co"]

    def test_settings_hash_covers_the_potcars(self):
        """Identical INCARs with different pseudopotentials are not comparable."""
        from cspflow.dft.vasp.inputs import ResolvedInputs
        from cspflow.dft.vasp.kpoints import KpointGrid
        from cspflow.dft.vasp.potcar import PotcarInfo

        def make(hash_):
            return ResolvedInputs(
                stage="relax", incar={"ENCUT": 520},
                grid=KpointGrid(3, 3, 3, "explicit"), symbols=["Fe"],
                potcars=[PotcarInfo("Fe", "Fe_pv", Path("/x"), "t", 8.0, 268.0,
                                    hash_, False)],
            )

        assert make("aaa").settings_hash != make("bbb").settings_hash

    def test_settings_hash_is_stable_for_identical_inputs(self):
        from cspflow.dft.vasp.inputs import ResolvedInputs
        from cspflow.dft.vasp.kpoints import KpointGrid

        def make():
            return ResolvedInputs(stage="relax", incar={"ENCUT": 520, "ISPIN": 2},
                                  grid=KpointGrid(3, 3, 3, "explicit"), symbols=["Fe"])

        assert make().settings_hash == make().settings_hash


@has_legacy
class TestAgainstTheLegacyCampaign:
    """The port is faithful where it should be, and different where it means to be."""

    def _legacy_atoms(self):
        from ase.io import read

        return read(str(LEGACY / "POSCAR"))

    def test_kpoints_match_exactly(self, orion):
        from cspflow.dft.vasp.inputs import resolve_inputs

        atoms = self._legacy_atoms()
        resolved = resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                                  Dft(recipe="magnets"), orion)
        legacy = [int(x) for x in (LEGACY / "KPOINTS").read_text().splitlines()[3].split()]
        assert [resolved.grid.a, resolved.grid.b, resolved.grid.c] == legacy

    def test_magmom_matches_exactly(self, orion):
        from cspflow.dft.vasp.inputs import resolve_inputs

        atoms = self._legacy_atoms()
        resolved = resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                                  Dft(recipe="magnets"), orion)
        rendered = render_incar({"MAGMOM": resolved.incar["MAGMOM"]}).strip()
        assert rendered == "MAGMOM = 2*-7 4*-2 20*2"

    def test_potcar_order_matches_the_poscar_species_line(self, orion):
        from cspflow.dft.vasp.inputs import resolve_inputs

        atoms = self._legacy_atoms()
        resolved = resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                                  Dft(recipe="magnets"), orion)
        legacy_species = (LEGACY / "POSCAR").read_text().splitlines()[5].split()
        assert [p.element for p in resolved.potcars] == legacy_species

    def test_what_we_write_passes_our_own_stage_0_gate(self, orion, tmp_path):
        """The POSCAR we emit must be one we would accept as input."""
        from cspflow.dft.vasp.inputs import resolve_inputs, write_inputs
        from cspflow.source.structure_list import read_seed

        atoms = self._legacy_atoms()
        resolved = resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                                  Dft(recipe="magnets"), orion)
        out = write_inputs(resolved, atoms, tmp_path / "job")
        read_atoms, counts = read_seed(out / "POSCAR")
        assert counts == {"Gd": 2, "Cr": 4, "Co": 20}

    def test_the_manifest_records_what_was_written(self, orion, tmp_path):
        from cspflow.dft.vasp.inputs import resolve_inputs, write_inputs

        atoms = self._legacy_atoms()
        resolved = resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                                  Dft(recipe="magnets"), orion)
        out = write_inputs(resolved, atoms, tmp_path / "job")
        manifest = json.loads((out / "inputs.json").read_text())
        assert manifest["settings_hash"] == resolved.settings_hash
        assert {p["element"] for p in manifest["potcars"]} == {"Gd", "Cr", "Co"}
        assert manifest["incar"]["ENCUT"] == 520

    def test_all_four_files_are_written(self, orion, tmp_path):
        from cspflow.dft.vasp.inputs import resolve_inputs, write_inputs

        atoms = self._legacy_atoms()
        resolved = resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                                  Dft(recipe="magnets"), orion)
        out = write_inputs(resolved, atoms, tmp_path / "job")
        for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "inputs.json"):
            assert (out / name).is_file(), name

# --- the ferromagnetic list (D110) ------------------------------------------


def test_ferromagnetic_covers_every_element_in_the_reference_set():
    """`strict` refuses a default, so a gap here is 83% of the store failing.

    The ferrimagnetic table covered 14 of the 27 elements the reference build
    touches; the other 13 -- Si, Al, Ga, Y, B, C, La, Zn, Cu, N, Zr, Nb, Mo --
    raised IncarError at input-writing time for 2,285 of 2,762 phases.
    """
    from cspflow.dft.vasp.incar import FERRO_RETM

    reference_set = set(
        "Al B C Ce Co Cr Cu Dy Fe Ga Gd La Mn Mo N Nb Nd Ni Pr Si Sm Tb "
        "Ti V Y Zn Zr".split()
    )
    assert not reference_set - set(FERRO_RETM)


def test_ferromagnetic_moments_are_all_positive():
    from cspflow.dft.vasp.incar import FERRO_RETM

    assert all(v > 0 for v in FERRO_RETM.values())


def test_frozen_f_rare_earths_get_the_valence_moment_not_the_4f_one():
    """Gd_3 has ZVAL 9 and no f in the valence: 7 muB is unrepresentable.

    The legacy flow asked for it anyway -- MAGMOM -3.0 on Nd sites while using
    Nd_3 -- and the SCF spent steps collapsing it.  0.6 is the 5d polarisation,
    which is what the POTCAR can actually hold.
    """
    from cspflow.config.schema import RARE_EARTHS
    from cspflow.dft.vasp.incar import FERRI_RETM, FERRO_RETM

    # Every rare earth gets the same 1.0, because with 4f frozen they all have
    # the same valence d shell: 5d1.  The element no longer enters the moment.
    assert {FERRO_RETM[e] for e in RARE_EARTHS} == {1.0}

    # The ferrimagnetic table is for f-in-valence and tracks the 4f count
    # instead, so it varies right across the series -- Ce 1 through Gd/Eu 7.
    ferri = {abs(FERRI_RETM[e]) for e in RARE_EARTHS if e in FERRI_RETM}
    assert len(ferri) > 1 and max(ferri) == 7.0


def test_the_ferromagnetic_table_is_materialised_into_the_hashed_config():
    """A mode NAME in recipe_id records where moments came from, not what they

    were -- so editing FERRO_RETM would change every energy while leaving the
    cache key untouched.  The numbers themselves have to be in the hash.
    """
    from cspflow.config.schema import Magnetism
    from cspflow.dft.vasp.incar import FERRO_RETM

    m = Magnetism(mode="ferromagnetic")
    assert m.table == dict(sorted(FERRO_RETM.items()))
    assert m.model_dump()["table"]["Fe"] == 4.0


def test_an_explicit_moment_still_beats_the_list():
    from cspflow.config.schema import Magnetism

    m = Magnetism(mode="ferromagnetic", table={"Fe": 5.0})
    assert m.table["Fe"] == 5.0
    assert m.table["Co"] == 3.0


def test_changing_a_moment_changes_the_recipe_id():
    from cspflow.config.schema import Dft, Magnetism
    from cspflow.dft.recipe import load_recipe
    from cspflow.reference.computed import recipe_id

    recipe = load_recipe("magnets")
    plain = recipe_id(Dft(magnetism=Magnetism(mode="ferromagnetic")), recipe)
    tweaked = recipe_id(
        Dft(magnetism=Magnetism(mode="ferromagnetic", table={"Fe": 5.0})), recipe
    )
    assert plain != tweaked


def test_every_moment_is_the_free_atom_hund_maximum_for_the_valence_d_shell():
    """One rule, so the list is reproducible rather than a table of guesses.

    Unpaired d electrons in the free atom.  Cr and Mo are 3d5/4d5 (the s1
    configuration); Fe 3d6 leaves four unpaired, Co 3d7 three, Ni 3d8 two.
    A filled or absent d shell gets 0.6, not 0.0 -- a site seeded at exactly
    zero is slow to break symmetry.
    """
    from cspflow.dft.vasp.incar import FERRO_RETM

    unpaired_d = {
        "Ti": 2, "V": 3, "Cr": 5, "Mn": 5, "Fe": 4, "Co": 3, "Ni": 2,
        "Zr": 2, "Nb": 4, "Mo": 5, "Hf": 2, "Ta": 3, "W": 4, "Pt": 1,
        "Cu": 0, "Zn": 0, "Ag": 0, "Pd": 0,
    }
    for element, n in unpaired_d.items():
        expected = float(n) if n else 0.6
        assert FERRO_RETM[element] == expected, element


def test_the_rare_earth_moment_matches_what_zval_leaves_in_the_valence():
    """Gd_3 is ZVAL 9 = 5s2 5p6 5d1.  One d electron, so one muB.

    This is the arithmetic that makes 1.0 a derivation rather than a taste:
    subtract the closed shells the POTCAR still carries and what remains is a
    lone 5d electron.  It is also the right order for the induced 5d moment in
    an RE-TM magnet, 0.3-0.5 muB, which the SCF relaxes down to.
    """
    from cspflow.dft.vasp.incar import FERRO_RETM

    # (POTCAR ZVAL, closed-shell electrons it still carries)
    for symbol, zval, closed in (
        ("Gd_3", 9, 8),      # 5s2 5p6
        ("Tb_3", 9, 8),
        ("Dy_3", 9, 8),
        ("Sm_3", 11, 10),    # 5s2 5p6 6s2
        ("Nd_3", 11, 10),
        ("La", 11, 10),
        ("Y_sv", 11, 10),    # 4s2 4p6 5s2
    ):
        element = symbol.split("_")[0]
        assert zval - closed == 1, symbol
        assert FERRO_RETM[element] == 1.0, element


def test_tetrahedron_falls_back_to_smearing_when_the_grid_is_too_coarse(orion):
    """VASP ABORTS on ISMEAR=-5 with fewer than four k-points -- and does it in

    the static step, after the relaxation has been paid for.  35 of the 2,746 MP
    reference phases land there at reciprocal_density 64, 29 of them Gamma-only.
    """
    from ase import Atoms

    from cspflow.config.schema import Dft
    from cspflow.dft.recipe import load_recipe
    from cspflow.dft.vasp.inputs import resolve_inputs

    recipe = load_recipe("magnets")
    static = recipe.stages[1]
    assert str(static.incar["ISMEAR"]).strip() == "-5"

    # A deliberately huge cell, so reciprocal_density 64 collapses to Gamma.
    big = Atoms("Fe", positions=[[0, 0, 0]], cell=[40.0, 40.0, 40.0], pbc=True)
    resolved = resolve_inputs(big, static, Dft(), orion)

    assert resolved.grid.a * resolved.grid.b * resolved.grid.c < 4
    assert str(resolved.incar["ISMEAR"]).strip() == "0"
    assert float(str(resolved.incar["SIGMA"])) > 0
    assert any("tetrahedron" in w.lower() for w in resolved.warnings)


def test_a_dense_enough_grid_keeps_the_tetrahedron_method(orion):
    from ase import Atoms

    from cspflow.config.schema import Dft
    from cspflow.dft.recipe import load_recipe
    from cspflow.dft.vasp.inputs import resolve_inputs

    static = load_recipe("magnets").stages[1]
    small = Atoms("Fe", positions=[[0, 0, 0]], cell=[2.87, 2.87, 2.87], pbc=True)
    resolved = resolve_inputs(small, static, Dft(), orion)

    assert resolved.grid.a * resolved.grid.b * resolved.grid.c >= 4
    assert str(resolved.incar["ISMEAR"]).strip() == "-5"


def test_the_tetrahedron_guard_counts_irreducible_kpoints_not_the_grid_product(orion):
    """A grid product over the limit can still fold below it, and VASP aborts.

    Found live 2026-09-10: `mp-1192814-Ce3Si3Pd102` was written with ISMEAR=-5
    on a 2x2x2 grid -- product 8, comfortably over the minimum of 4 -- and VASP
    stopped in the static step with

        VERY BAD NEWS! internal error in subroutine BZINTS:
        Tetrahedron method fails (number of k-points < 4) 3

    because symmetry folds that mesh to THREE irreducible points. Four
    structures failed this way, all of them waved through by a guard that
    multiplied the grid dimensions instead of asking spglib.
    """
    from ase import Atoms

    from cspflow.config.schema import Dft
    from cspflow.dft.recipe import load_recipe
    from cspflow.dft.vasp.inputs import resolve_inputs
    from cspflow.dft.vasp.parallel import irreducible_kpoints

    static = load_recipe("magnets").stages[1]
    # High symmetry is what makes the two counts disagree: a simple cubic cell
    # on a 2x2x2 mesh folds to far fewer than eight points.
    cubic = Atoms("Fe", positions=[[0, 0, 0]], cell=[6.0, 6.0, 6.0], pbc=True)
    resolved = resolve_inputs(cubic, static, Dft(), orion)

    product = resolved.grid.a * resolved.grid.b * resolved.grid.c
    folded = irreducible_kpoints(
        resolved.atoms.cell[:], resolved.atoms.get_scaled_positions(),
        resolved.atoms.get_atomic_numbers(),
        (resolved.grid.a, resolved.grid.b, resolved.grid.c),
    )
    if folded is None:
        import pytest
        pytest.skip("spglib absent: the guard falls back to the grid product")

    # The point of the test: when they disagree, the DECISION follows the
    # folded count.  Asserting on the product would restate the old bug.
    if folded < 4:
        assert str(resolved.incar["ISMEAR"]).strip() == "0", (
            f"grid product {product} passed the guard but the mesh folds to "
            f"{folded} irreducible points -- VASP would abort here"
        )
        assert any("irreducible" in w for w in resolved.warnings)
    else:
        assert str(resolved.incar["ISMEAR"]).strip() == "-5"


class TestCarryingTheGridAcrossAResume:
    """A resumed relaxation must keep the sampling it was already running.

    The grid is floor(mult / length), so a cell that drifts across an integer
    boundary re-derives a different grid on resume and the run then minimises a
    different energy surface from the one its starting geometry was nearly
    converged on. Numbers below are structure 2498 of RE-magnets-CHGNet.
    """

    def test_the_real_case_that_found_this(self):
        from cspflow.dft.recipe import Kpoints
        from cspflow.dft.vasp.kpoints import carry_grid, grid_for

        started = [8.6786, 8.6786, 12.5437]      # attempt 0 started here
        reached = [8.6222, 8.6225, 12.6264]      # ... and relaxed to here
        density = Kpoints(scheme="reciprocal_density", value=64)

        first = grid_for(started, density)
        derived = grid_for(reached, density)
        assert (first.a, first.b, first.c) == (2, 2, 2)
        assert (derived.a, derived.b, derived.c) == (2, 2, 1), (
            "0.66% of c-axis relaxation crossed the mult/2 = 12.5664 A boundary")

        carried = carry_grid(first, reached, reached)
        assert (carried.a, carried.b, carried.c) == (2, 2, 2)

    def test_a_niggli_axis_permutation_is_followed(self):
        from cspflow.dft.vasp.kpoints import KpointGrid, carry_grid

        previous = KpointGrid(4, 2, 6, scheme="reciprocal_density")
        source = [5.0, 9.0, 3.0]
        canonical = [3.0, 5.0, 9.0]              # c, a, b
        carried = carry_grid(previous, source, canonical)
        assert (carried.a, carried.b, carried.c) == (6, 4, 2)

    def test_a_basis_that_is_not_a_permutation_refuses(self):
        from cspflow.dft.vasp.kpoints import KpointGrid, carry_grid

        previous = KpointGrid(4, 4, 2, scheme="reciprocal_density")
        assert carry_grid(previous, [5.0, 5.0, 9.0], [5.0, 5.0, 7.1]) is None

    def test_resolve_inputs_uses_the_carried_grid_and_says_so(self, orion):
        from ase import Atoms

        from cspflow.config.schema import Dft
        from cspflow.dft.recipe import load_recipe
        from cspflow.dft.vasp.inputs import resolve_inputs
        from cspflow.dft.vasp.kpoints import KpointGrid

        relax = load_recipe("magnets").stages[0]
        cell = Atoms("Fe", positions=[[0, 0, 0]], cell=[2.87, 2.87, 2.87],
                     pbc=True)

        plain = resolve_inputs(cell, relax, Dft(), orion)
        coarser = KpointGrid(plain.grid.a - 1, plain.grid.b, plain.grid.c,
                             scheme="reciprocal_density")

        resumed = resolve_inputs(cell, relax, Dft(), orion, carried_grid=coarser)
        assert (resumed.grid.a, resumed.grid.b, resumed.grid.c) == (
            coarser.a, coarser.b, coarser.c)
        assert any("carried from the previous attempt" in w
                   for w in resumed.warnings)

    def test_a_carried_grid_that_agrees_is_not_worth_a_warning(self, orion):
        from ase import Atoms

        from cspflow.config.schema import Dft
        from cspflow.dft.recipe import load_recipe
        from cspflow.dft.vasp.inputs import resolve_inputs

        relax = load_recipe("magnets").stages[0]
        cell = Atoms("Fe", positions=[[0, 0, 0]], cell=[2.87, 2.87, 2.87],
                     pbc=True)

        plain = resolve_inputs(cell, relax, Dft(), orion)
        resumed = resolve_inputs(cell, relax, Dft(), orion,
                                 carried_grid=plain.grid)
        assert resumed.settings_hash == plain.settings_hash
        assert not any("carried" in w for w in resumed.warnings)
