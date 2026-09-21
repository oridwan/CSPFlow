"""Stage 0 -- the three source modes and the plan they produce.

The most valuable test in this file is `test_matches_the_original_algorithm`:
it reimplements `search_ternary_magnets.py` as it was written and requires the
generalised enumerator to produce exactly the same set.  A generalisation that
quietly changes what a campaign enumerates is worse than no generalisation, and
"exactly the same 2,616 formulas" is the only version of that claim worth
making.
"""

from itertools import combinations
from math import gcd

import pytest

from cspflow.chem import canonical_formula
from cspflow.config.schema import NStructures, Source, SourceDefaults
from cspflow.db.store import Store, StoreError
from cspflow.source import SourceError, expand_all, expand_source, write_plan
from cspflow.source.base import expand_z
from cspflow.source.chemical_space import expand_chemical_space
from cspflow.source.composition_list import expand_composition_list
from cspflow.source.structure_list import SeedError, expand_structure_list, read_seed


def cs(groups, **kwargs):
    """A chemical_space source with the boilerplate filled in."""
    block = {"groups": groups, "max_atoms_formula": kwargs.pop("max_atoms_formula", 12)}
    if "max_rare_earth" in kwargs:
        block["max_rare_earth"] = kwargs.pop("max_rare_earth")
    return Source(mode="chemical_space", name=kwargs.pop("name", "s"),
                  chemical_space=block, defaults=kwargs or {})


# --------------------------------------------------------------------------
# Mode 1
# --------------------------------------------------------------------------


def original_search(rare_earths, tms, max_atoms, tm_rich_ratio):
    """`search_ternary_magnets.py`, reduced to its enumeration.

    Kept faithful on purpose, including the `T > TP` string comparison it used
    to suppress the duplicates its two-groups-from-one-list construction
    creates.
    """
    out = set()
    for R in rare_earths:
        for T in tms:
            for TP in tms:
                if T == TP or T > TP:
                    continue
                for n_r in range(1, max_atoms):
                    for n_t in range(1, max_atoms):
                        for n_tp in range(1, max_atoms):
                            if n_r + n_t + n_tp > max_atoms:
                                continue
                            g = gcd(gcd(n_r, n_t), n_tp)
                            if g != 1:
                                continue
                            total = n_r + n_t + n_tp
                            if (n_t + n_tp) / total < tm_rich_ratio:
                                continue
                            out.add(canonical_formula({R: n_r, T: n_t, TP: n_tp}))
    return out


class TestChemicalSpace:
    def test_matches_the_original_algorithm(self):
        """Exact set equality with the script this generalises."""
        expected = original_search(["Gd", "Y"], ["Fe", "Co", "Ni"], 20, 0.75)
        result = expand_chemical_space(
            cs({"R": {"elements": ["Gd", "Y"], "pick": 1},
                "T": {"elements": ["Fe", "Co", "Ni"], "pick": 2, "min_fraction": 0.75}},
               max_atoms_formula=20, max_atoms=20)
        )
        assert {c.formula for c in result.compositions} == expected
        assert len(expected) == 2616

    def test_pick_two_is_combinations_not_permutations(self):
        """`pick: 2` cannot emit a group's elements in both orders."""
        result = expand_chemical_space(
            cs({"T": {"elements": ["Fe", "Co"], "pick": 2}}, max_atoms_formula=3)
        )
        assert sorted(c.formula for c in result.compositions) == ["Co1Fe1", "Co1Fe2", "Co2Fe1"]

    def test_pick_list_widens_the_space(self):
        one = expand_chemical_space(cs({"T": {"elements": ["Fe", "Co", "Ni"], "pick": 1}}))
        both = expand_chemical_space(cs({"T": {"elements": ["Fe", "Co", "Ni"], "pick": [1, 2]}}))
        assert len(both.compositions) > len(one.compositions)

    def test_binary_and_ternary_use_one_code_path(self):
        binary = expand_chemical_space(
            cs({"A": {"elements": ["Sm"], "pick": 1}, "B": {"elements": ["Fe"], "pick": 1}})
        )
        ternary = expand_chemical_space(
            cs({"A": {"elements": ["Sm"], "pick": 1}, "B": {"elements": ["Fe"], "pick": 1},
                "C": {"elements": ["Ti"], "pick": 1}})
        )
        assert binary.compositions and ternary.compositions
        assert all(len(c.counts) == 2 for c in binary.compositions)
        assert all(len(c.counts) == 3 for c in ternary.compositions)

    def test_only_reduced_formulas(self):
        """`FeCo5` and `Fe2Co10` must not both enter as separate work."""
        result = expand_chemical_space(
            cs({"A": {"elements": ["Fe"], "pick": 1}, "B": {"elements": ["Co"], "pick": 1}},
               max_atoms_formula=12)
        )
        for c in result.compositions:
            assert gcd(*c.counts.values()) == 1

    def test_min_fraction_applies_to_its_own_group(self):
        result = expand_chemical_space(
            cs({"R": {"elements": ["Sm"], "pick": 1},
                "T": {"elements": ["Fe"], "pick": 1, "min_fraction": 0.8}},
               max_atoms_formula=12)
        )
        for c in result.compositions:
            assert c.counts["Fe"] / sum(c.counts.values()) >= 0.8

    def test_max_fraction_applies_to_its_own_group(self):
        result = expand_chemical_space(
            cs({"R": {"elements": ["Sm"], "pick": 1, "max_fraction": 0.2},
                "T": {"elements": ["Fe"], "pick": 1}},
               max_atoms_formula=12)
        )
        assert result.compositions
        for c in result.compositions:
            assert c.counts["Sm"] / sum(c.counts.values()) <= 0.2

    def test_rejections_are_counted_with_a_reason(self):
        result = expand_chemical_space(
            cs({"R": {"elements": ["Sm"], "pick": 1},
                "T": {"elements": ["Fe"], "pick": 1, "min_fraction": 0.8}},
               max_atoms_formula=12)
        )
        assert result.rejected.total > 0
        assert any("min_fraction" in reason for reason in result.rejected.counts)

    def test_max_rare_earth_guards_the_assembled_system(self):
        """The guard is on the system, not on `pick`: nothing forces RE into one group."""
        mixed = {"M": {"elements": ["Sm", "Tb", "Fe"], "pick": [1, 2]}}
        limited = expand_chemical_space(cs(mixed, max_rare_earth=1, max_atoms_formula=8))
        unlimited = expand_chemical_space(cs(mixed, max_rare_earth=None, max_atoms_formula=8))

        assert all(not ({"Sm", "Tb"} <= set(c.counts)) for c in limited.compositions)
        assert any({"Sm", "Tb"} <= set(c.counts) for c in unlimited.compositions)
        assert any("rare earth" in r for r in limited.rejected.counts)

    def test_max_rare_earth_zero_excludes_them_entirely(self):
        result = expand_chemical_space(
            cs({"M": {"elements": ["Sm", "Fe"], "pick": [1, 2]}}, max_rare_earth=0)
        )
        assert all("Sm" not in c.counts for c in result.compositions)

    def test_element_in_two_groups_is_rejected_not_doubled(self):
        result = expand_chemical_space(
            cs({"A": {"elements": ["Fe", "Co"], "pick": 1},
                "B": {"elements": ["Fe", "Gd"], "pick": 1}}, max_atoms_formula=6)
        )
        assert any("more than one group" in r for r in result.rejected.counts)
        for c in result.compositions:
            assert len(c.counts) == 2

    def test_empty_space_warns_rather_than_returning_silently(self):
        result = expand_chemical_space(
            cs({"R": {"elements": ["Sm"], "pick": 1},
                "T": {"elements": ["Fe"], "pick": 1, "min_fraction": 0.99}},
               max_atoms_formula=8)
        )
        assert not result.compositions
        assert any("no compositions" in w for w in result.warnings)

    def test_unreachable_rare_earth_guard_warns(self):
        result = expand_chemical_space(
            cs({"T": {"elements": ["Fe", "Co"], "pick": 1}}, max_rare_earth=1)
        )
        assert any("no group contains" in w for w in result.warnings)

    def test_enters_the_funnel_at_generate(self):
        assert expand_chemical_space(cs({"A": {"elements": ["Fe"], "pick": 1}})).entry_stage == "generate"


# --------------------------------------------------------------------------
# Z expansion
# --------------------------------------------------------------------------


class TestExpandZ:
    def test_one_row_per_admissible_z(self):
        rows = expand_z({"Fe": 1, "Co": 5},
                        defaults=SourceDefaults(z={"min": 1, "max": 3}, max_atoms=40),
                        source_name="s", source_mode="composition_list")
        assert [r.z for r in rows] == [1, 2, 3]
        assert [r.n_atoms for r in rows] == [6, 12, 18]

    def test_cells_over_max_atoms_are_rejected_with_a_reason(self):
        from cspflow.source.base import RejectionLog

        log = RejectionLog()
        rows = expand_z({"Fe": 1, "Co": 5},
                        defaults=SourceDefaults(z={"min": 1, "max": 3}, max_atoms=13),
                        source_name="s", source_mode="composition_list", reject=log)
        assert [r.z for r in rows] == [1, 2]
        assert any("max_atoms" in r for r in log.counts)

    def test_per_atom_budget_scales_with_the_cell(self):
        rows = expand_z({"Fe": 1, "Co": 5},
                        defaults=SourceDefaults(z={"min": 1, "max": 2}, max_atoms=40,
                                                n_structures=NStructures(structures_per_atom=2.0)),
                        source_name="s", source_mode="composition_list")
        assert [r.n_target for r in rows] == [12, 24]

    def test_scope_total_shares_one_budget_across_z(self):
        per_z = expand_z({"Fe": 1, "Co": 5},
                         defaults=SourceDefaults(z={"min": 1, "max": 2}, max_atoms=40,
                                                 n_structures=NStructures(mode="fixed", count=100)),
                         source_name="s", source_mode="composition_list")
        total = expand_z({"Fe": 1, "Co": 5},
                         defaults=SourceDefaults(z={"min": 1, "max": 2}, max_atoms=40,
                                                 n_structures=NStructures(mode="fixed", count=100),
                                                 n_structures_scope="total"),
                         source_name="s", source_mode="composition_list")
        assert sum(r.n_target for r in per_z) == 200
        assert sum(r.n_target for r in total) == 100

    def test_scope_total_never_leaves_a_z_with_nothing(self):
        rows = expand_z({"Fe": 1, "Co": 5},
                        defaults=SourceDefaults(z={"min": 1, "max": 3}, max_atoms=40,
                                                n_structures=NStructures(mode="fixed", count=2),
                                                n_structures_scope="total"),
                        source_name="s", source_mode="composition_list")
        assert len(rows) == 3 and all(r.n_target >= 1 for r in rows)


# --------------------------------------------------------------------------
# Mode 2
# --------------------------------------------------------------------------


def cl(items=None, **kwargs):
    block = {"items": items or []}
    block.update({k: v for k, v in kwargs.items() if k in {"from_file"}})
    return Source(mode="composition_list", name=kwargs.get("name", "s"),
                  composition_list=block,
                  defaults=kwargs.get("defaults", {"max_atoms": 40}))


class TestCompositionList:
    def test_reduces_and_canonicalises(self):
        result = expand_composition_list(cl([{"formula": "SmFe11Ti"}]))
        assert [c.formula for c in result.compositions] == ["Fe11Sm1Ti1"]

    def test_non_reduced_formula_becomes_that_many_formula_units(self):
        """`Fe2Co10` is a 12-atom cell, not a silently shrunk 6-atom one."""
        result = expand_composition_list(cl([{"formula": "Fe2Co10"}]))
        assert [(c.formula, c.z, c.n_atoms) for c in result.compositions] == [("Co5Fe1", 2, 12)]
        assert any("not reduced" in w for w in result.warnings)

    def test_non_reduced_formula_with_explicit_z_is_refused_not_guessed(self):
        with pytest.raises(SourceError, match="ambiguous"):
            expand_composition_list(cl([{"formula": "Fe2Co10", "z": [1, 2]}]))

    def test_same_formula_same_z_is_deduplicated_and_reported(self):
        result = expand_composition_list(cl([{"formula": "FeCo5"}, {"formula": "Co5Fe1"}]))
        assert len(result.compositions) == 1
        assert any("duplicates" in w for w in result.warnings)

    def test_same_formula_different_z_is_kept_as_separate_work(self):
        result = expand_composition_list(cl([{"formula": "FeCo5"}, {"formula": "Fe2Co10"}]))
        assert sorted((c.formula, c.z) for c in result.compositions) == [("Co5Fe1", 1), ("Co5Fe1", 2)]
        assert any("all reduce to" in w for w in result.warnings)

    def test_repeated_identical_item_says_so_plainly(self):
        result = expand_composition_list(cl([{"formula": "FeCo5"}, {"formula": "FeCo5"}]))
        assert any("appears twice" in w for w in result.warnings)

    def test_per_item_overrides_win(self):
        result = expand_composition_list(cl([
            {"formula": "FeCo5"},
            {"formula": "SmFe11Ti", "n_structures": {"mode": "fixed", "count": 300}},
        ]))
        by_formula = {c.formula: c.n_target for c in result.compositions}
        assert by_formula["Fe11Sm1Ti1"] == 300
        assert by_formula["Co5Fe1"] != 300

    def test_unparseable_formula_stops_the_campaign(self):
        """Unlike ingest, a curated list is the user's own writing: raise, do not skip."""
        with pytest.raises(SourceError, match="not an element"):
            expand_composition_list(cl([{"formula": "Xx2O3"}]))

    def test_from_file_with_header_and_comments(self, tmp_path):
        csv = tmp_path / "comps.csv"
        csv.write_text(
            "# my sweep\nformula,z_min,z_max,n_structures\n"
            "SmFe11Ti,1,2,150\nGd2Co17,,,\nNd2Fe14B\n"
        )
        result = expand_composition_list(
            Source(mode="composition_list", name="s",
                   composition_list={"from_file": str(csv)},
                   defaults={"max_atoms": 40})
        )
        rows = {(c.formula, c.z): c.n_target for c in result.compositions}
        assert rows[("Fe11Sm1Ti1", 1)] == 150 and rows[("Fe11Sm1Ti1", 2)] == 150
        assert ("Co17Gd2", 1) in rows and ("B1Fe14Nd2", 1) in rows

    def test_from_file_missing_is_an_error_naming_the_path(self, tmp_path):
        missing = tmp_path / "nope.csv"
        with pytest.raises(SourceError, match=str(missing)):
            expand_composition_list(
                Source(mode="composition_list", name="s",
                       composition_list={"from_file": str(missing)})
            )

    def test_from_file_bad_cell_names_the_line_and_column(self, tmp_path):
        csv = tmp_path / "comps.csv"
        csv.write_text("SmFe11Ti,one,2,150\n")
        with pytest.raises(SourceError, match="z_min"):
            expand_composition_list(
                Source(mode="composition_list", name="s",
                       composition_list={"from_file": str(csv)})
            )

    def test_every_z_over_max_atoms_warns_rather_than_vanishing(self):
        result = expand_composition_list(cl([{"formula": "Gd2Co17"}],
                                            defaults={"max_atoms": 5}))
        assert not result.compositions
        assert any("no rows" in w for w in result.warnings)


# --------------------------------------------------------------------------
# Mode 3
# --------------------------------------------------------------------------


GOOD_POSCAR = """FeCo test
1.0
2.8500000000 0.0000000000 0.0000000000
0.0000000000 2.8500000000 0.0000000000
0.0000000000 0.0000000000 2.8500000000
Fe Co
1 1
Direct
0.0000000000 0.0000000000 0.0000000000
0.5000000000 0.5000000000 0.5000000000
"""

VASP4_POSCAR = "\n".join(
    line for i, line in enumerate(GOOD_POSCAR.splitlines()) if i != 5
) + "\n"

DISORDERED_CIF = """data_test
_cell_length_a 3.0
_cell_length_b 3.0
_cell_length_c 3.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Ti1 Ti 0.0 0.0 0.0 0.5
Fe1 Fe 0.0 0.0 0.0 0.5
Fe2 Fe 0.5 0.5 0.5 1.0
"""

pymatgen = pytest.importorskip("pymatgen.core", reason="mode 3 needs pymatgen")


@pytest.fixture
def seeds(tmp_path):
    (tmp_path / "good.vasp").write_text(GOOD_POSCAR)
    (tmp_path / "vasp4.vasp").write_text(VASP4_POSCAR)
    (tmp_path / "disordered.cif").write_text(DISORDERED_CIF)
    (tmp_path / "README.md").write_text("not a structure")
    return tmp_path


def sl(paths, **kwargs):
    block = {"paths": [str(p) for p in paths]}
    block.update(kwargs)
    return Source(mode="structure_list", name="seeds", structure_list=block)


class TestSeedGates:
    def test_good_poscar_is_read(self, seeds):
        _, counts = read_seed(seeds / "good.vasp")
        assert counts == {"Fe": 1, "Co": 1}

    def test_vasp4_poscar_is_refused_before_any_parser_invents_elements(self, seeds):
        """pymatgen turns this file into H1 He1 with only a warning."""
        with pytest.raises(SeedError, match="VASP-4"):
            read_seed(seeds / "vasp4.vasp")

    def test_partial_occupancy_is_refused_at_ingest(self, seeds):
        """Otherwise it dies at POSCAR generation, one stage after the GPU time."""
        with pytest.raises(SeedError, match="partial occupanc"):
            read_seed(seeds / "disordered.cif")

    def test_refusal_names_the_remedy(self, seeds):
        with pytest.raises(SeedError, match="OrderDisorderedStructureTransformation"):
            read_seed(seeds / "disordered.cif")


class TestStructureList:
    def test_derives_composition_from_the_file(self, seeds):
        result = expand_structure_list(sl([seeds / "good.vasp"]))
        assert len(result.structures) == 1
        s = result.structures[0]
        assert (s.formula, s.chemsys, s.n_atoms) == ("Co1Fe1", "Co-Fe", 2)

    def test_enters_the_funnel_at_screen(self, seeds):
        assert expand_structure_list(sl([seeds / "good.vasp"])).entry_stage == "screen"

    def test_emits_no_compositions_to_generate(self, seeds):
        assert expand_structure_list(sl([seeds / "good.vasp"])).compositions == []

    def test_records_path_and_content_hash(self, seeds):
        s = expand_structure_list(sl([seeds / "good.vasp"])).structures[0]
        assert s.path.endswith("good.vasp")
        assert s.content_hash.startswith("sha256:")
        assert len(s.content_hash) == len("sha256:") + 16

    def test_content_hash_can_never_look_like_a_number(self, seeds):
        """ASE refuses a string key_value_pair that int()/float() would parse, so a
        bare hex digest breaks the source stage about once in 600 seeds -- measured:
        23 of 14,777 on the RE-magnets campaign. The prefix rules it out by
        construction."""
        import re
        number_like = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
        s = expand_structure_list(sl([seeds / "good.vasp"])).structures[0]
        assert not number_like.match(s.content_hash)

    def test_a_directory_ignores_non_structure_files(self, seeds):
        """A README next to your seeds is not an error."""
        (seeds / "vasp4.vasp").unlink()
        (seeds / "disordered.cif").unlink()
        result = expand_structure_list(sl([seeds]))
        assert len(result.structures) == 1

    def test_globs_are_expanded(self, seeds):
        (seeds / "vasp4.vasp").unlink()
        (seeds / "second.vasp").write_text(GOOD_POSCAR.replace("2.85", "2.90"))
        result = expand_structure_list(sl([seeds / "*.vasp"]))
        assert len(result.structures) == 2

    def test_a_bad_file_in_a_glob_stops_the_campaign(self, seeds):
        with pytest.raises(SourceError, match="VASP-4"):
            expand_structure_list(sl([seeds / "*.vasp"]))

    def test_no_matches_is_an_error_naming_the_pattern(self, tmp_path):
        with pytest.raises(SourceError, match="matched no files"):
            expand_structure_list(sl([tmp_path / "*.cif"]))

    def test_max_atoms_applies_to_seeds_too(self, seeds):
        result = expand_structure_list(sl([seeds / "good.vasp"], max_atoms=1))
        assert not result.structures
        assert any("max_atoms" in r for r in result.rejected.counts)

    def test_duplicates_are_reported_and_both_kept_by_default(self, seeds):
        (seeds / "copy.vasp").write_text(GOOD_POSCAR)
        result = expand_structure_list(sl([seeds / "good.vasp", seeds / "copy.vasp"]))
        assert len(result.structures) == 2
        assert any("BOTH kept" in w for w in result.warnings)

    def test_dedup_drop_removes_them(self, seeds):
        (seeds / "copy.vasp").write_text(GOOD_POSCAR)
        result = expand_structure_list(
            sl([seeds / "good.vasp", seeds / "copy.vasp"], dedup="drop")
        )
        assert len(result.structures) == 1
        assert any("dropped" in w for w in result.warnings)

    def test_dedup_off_skips_the_comparison_entirely(self, seeds):
        """`StructureMatcher.fit` is ~320 ms on a small five-species cell and the
        pairs grow as k-squared inside a formula group. 232 CePdGe seeds cost
        237 s to compare against 6.8 s to parse and validate every file. When
        the overlap is deliberate and recorded elsewhere, there is nothing to
        learn from paying that."""
        (seeds / "copy.vasp").write_text(GOOD_POSCAR)
        result = expand_structure_list(
            sl([seeds / "good.vasp", seeds / "copy.vasp"], dedup="off")
        )
        assert len(result.structures) == 2
        assert not any("BOTH kept" in w or "dropped" in w for w in result.warnings)

    def test_parallel_and_serial_comparison_agree(self, seeds, monkeypatch):
        """The fits run in a process pool. They are pure and independent, so the
        pool must not change a single verdict or the order they are reported in
        -- the drop bookkeeping stays sequential precisely so it cannot."""
        import cspflow.source.structure_list as mod

        (seeds / "copy.vasp").write_text(GOOD_POSCAR)
        (seeds / "copy2.vasp").write_text(GOOD_POSCAR)
        paths = [seeds / "good.vasp", seeds / "copy.vasp", seeds / "copy2.vasp"]

        monkeypatch.setattr(mod, "_PARALLEL_MIN_PAIRS", 0)        # force the pool
        par = expand_structure_list(sl(paths, dedup="drop"))
        monkeypatch.setattr(mod, "_PARALLEL_MIN_PAIRS", 10 ** 9)  # force serial
        ser = expand_structure_list(sl(paths, dedup="drop"))

        assert par.warnings == ser.warnings
        assert [x.path for x in par.structures] == [x.path for x in ser.structures]

    def test_a_structure_is_converted_once_not_once_per_pair(self, seeds, monkeypatch):
        """The comparison used to convert BOTH sides of every pair, so a seed in
        a group of 14 was converted 13 times over -- 1,478 conversions for 232
        CePdGe seeds where 232 would do."""
        import cspflow.source.structure_list as mod
        from pymatgen.io.ase import AseAtomsAdaptor

        paths = [seeds / "good.vasp"]
        for n in range(4):
            p = seeds / f"dup{n}.vasp"
            # Same structure, different bytes, so the content-hash shortcut does
            # not fire and every pair reaches StructureMatcher.
            p.write_text(GOOD_POSCAR.replace("1.0", "1.000", 1))
            paths.append(p)

        calls = []
        real = AseAtomsAdaptor.get_structure
        monkeypatch.setattr(AseAtomsAdaptor, "get_structure",
                            staticmethod(lambda a, **k: (calls.append(1), real(a, **k))[1]))
        monkeypatch.setattr(mod, "_PARALLEL_MIN_PAIRS", 10 ** 9)
        result = expand_structure_list(sl(paths))

        # One conversion per structure that entered a comparison, never per pair.
        assert calls and len(calls) <= len(result.structures)

    def test_relax_false_warns_about_the_geometry_it_will_report(self, seeds):
        result = expand_structure_list(sl([seeds / "good.vasp"], relax=False))
        assert any("relax: false" in w for w in result.warnings)


# --------------------------------------------------------------------------
# Pooling and committing
# --------------------------------------------------------------------------


class TestPlan:
    def test_duplicate_source_names_are_refused(self):
        with pytest.raises(SourceError, match="distinct"):
            expand_all([cs({"A": {"elements": ["Fe"], "pick": 1}}, name="x"),
                        cs({"A": {"elements": ["Co"], "pick": 1}}, name="x")])

    def test_cross_source_collisions_are_reported_not_merged(self):
        """A seed reappearing as a generated candidate is a result, not a redundancy."""
        plan = expand_all([
            Source(mode="composition_list", name="a",
                   composition_list={"items": [{"formula": "FeCo5"}]},
                   defaults={"max_atoms": 40}),
            Source(mode="composition_list", name="b",
                   composition_list={"items": [{"formula": "Co5Fe1"}]},
                   defaults={"max_atoms": 40}),
        ])
        assert len(plan.compositions) == 2
        assert plan.collisions and "Co5Fe1" in plan.collisions[0]

    def test_render_estimates_gpu_time(self):
        plan = expand_all([cs({"A": {"elements": ["Fe"], "pick": 1},
                               "B": {"elements": ["Co"], "pick": 1}}, max_atoms_formula=6)])
        text = plan.render(gpu_seconds_per_structure=3.0)
        assert "implied GPU time" in text and "structures wanted" in text

    def test_bare_mapping_source_is_a_one_element_list(self):
        from cspflow.config.schema import Campaign

        campaign = Campaign(
            name="c", machine="orion", workdir="/tmp/x",
            source={"mode": "composition_list", "name": "only",
                    "composition_list": {"items": [{"formula": "FeCo5"}]}},
            generate={"engine": "mattergen", "mattergen": {"model": "/tmp/model"}},
        )
        plan = expand_all(campaign)
        assert [c.formula for c in plan.compositions] == ["Co5Fe1"]


class TestWritePlan:
    def test_composition_rows_land(self, tmp_path):
        plan = expand_all([cs({"A": {"elements": ["Sm"], "pick": 1},
                               "B": {"elements": ["Fe"], "pick": 1, "min_fraction": 0.8}},
                              max_atoms_formula=12, max_atoms=24)])
        with Store.create(tmp_path / "c.db", campaign="t") as store:
            stats = write_plan(plan, store)
            assert stats.compositions == len(plan.compositions)
            assert store.chemsystems() == ["Fe-Sm"]

    def test_seed_structures_land_with_origin_seed(self, tmp_path, seeds):
        plan = expand_all([sl([seeds / "good.vasp"])])
        with Store.create(tmp_path / "c.db", campaign="t") as store:
            write_plan(plan, store)
            row = next(store.structures())
            assert row.origin == "seed"
            assert row.reduced_formula == "Co1Fe1"
            assert row.source_path.endswith("good.vasp")

    def test_relax_false_seeds_start_already_screened(self, tmp_path, seeds):
        plan = expand_all([sl([seeds / "good.vasp"], relax=False)])
        with Store.create(tmp_path / "c.db", campaign="t") as store:
            write_plan(plan, store)
            assert next(store.structures()).state == "screened"

    def test_relax_true_seeds_start_new(self, tmp_path, seeds):
        plan = expand_all([sl([seeds / "good.vasp"], relax=True)])
        with Store.create(tmp_path / "c.db", campaign="t") as store:
            write_plan(plan, store)
            assert next(store.structures()).state == "new"

    def test_writing_twice_is_idempotent(self, tmp_path):
        plan = expand_all([cs({"A": {"elements": ["Sm"], "pick": 1},
                               "B": {"elements": ["Fe"], "pick": 1}}, max_atoms_formula=8,
                              max_atoms=16)])
        with Store.create(tmp_path / "c.db", campaign="t") as store:
            write_plan(plan, store)
            first = len(store.compositions())
            write_plan(plan, store)
            assert len(store.compositions()) == first


class TestReservedKeys:
    """ASE reserves key names; the failure it produces otherwise says nothing."""

    def _atoms(self):
        from ase import Atoms

        return Atoms("Fe", positions=[(0, 0, 0)], cell=[3, 3, 3], pbc=True)

    @pytest.mark.parametrize("key", ["formula", "energy", "magmom", "natoms", "Co"])
    def test_reserved_key_names_are_refused_with_the_reason(self, tmp_path, key):
        with Store.create(tmp_path / "c.db", campaign="t") as store:
            with pytest.raises(StoreError, match="reserved by ASE"):
                store.add_structure(self._atoms(), origin="seed", **{key: 1})

    def test_formula_like_key_is_refused_because_select_would_lie(self, tmp_path):
        with Store.create(tmp_path / "c.db", campaign="t") as store:
            with pytest.raises(StoreError, match="parses as a chemical formula"):
                store.add_structure(self._atoms(), origin="seed", H2O=1)
