"""Adopting a tree of finished VASP runs into the computed reference cache.

The tests are written against the failure modes that made this module
necessary, not against its happy path: a half-finished run, a directory from a
different policy, a destroyed geometry, and a composition that disagrees with
the energy beside it. Each of those produces a plausible number and a hull that
builds, which is why none of them can be left to a later check.
"""

import json

import pytest

from cspflow.dft.recipe import load_recipe
from cspflow.reference import adopt as A
from cspflow.reference.computed import ComputedPhase, read_record, write_record

# A minimal but realistic static INCAR: every INVARIANT tag the recipe states,
# plus the two PHYSICS tags and a couple of robustness knobs a retry would move.
STATIC_INCAR = """\
SYSTEM = mp-1
ENCUT = 520
PREC = Accurate
SIGMA = 0.05
LASPH = .TRUE.
LREAL = .FALSE.
LMAXMIX = 4
ISMEAR = -5
ISPIN = 2
IBRION = -1
NSW = 0
EDIFF = 1e-06
ALGO = Fast
SYMPREC = 1e-05
"""

OUTCAR = """\
   NIONS =      2
  free  energy   TOTEN  =       -16.000000 eV
  energy  without entropy=      -16.100000  energy(sigma->0) =      -16.000000
 reached required accuracy - stopping structural energy minimisation
 General timing and accounting informations for this job:
                         Elapsed time (sec):     100.0
"""

OSZICAR = "   1 F= -.16000000E+02 E0= -.16000000E+02  d E =0.0  mag=     0.0000\n"

POSCAR = """\
Fe2
1.0
   3.0 0.0 0.0
   0.0 3.0 0.0
   0.0 0.0 3.0
Fe
2
Direct
0.00 0.00 0.00
0.50 0.50 0.50
"""


@pytest.fixture
def recipe():
    return load_recipe("magnets")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """One phase directory shaped like `final/mp-*/`, plus an isolated cache."""
    monkeypatch.setenv("CSPFLOW_REFERENCE", str(tmp_path / "cache"))
    root = tmp_path / "final"
    make_phase(root, "mp-1")
    return root


def make_phase(root, mp_id, *, counts=None, incar=STATIC_INCAR, outcar=OUTCAR,
               poscar=POSCAR, relax_poscar=None, mattersim=True):
    d = root / mp_id
    for step in ("relax", "static"):
        (d / step).mkdir(parents=True, exist_ok=True)
        (d / step / "INCAR").write_text(incar)
        (d / step / "OUTCAR").write_text(outcar)
        (d / step / "OSZICAR").write_text(OSZICAR)
        (d / step / "POSCAR").write_text(poscar)
        (d / step / "CONTCAR").write_text(poscar)
    if relax_poscar is not None:
        (d / "relax" / "POSCAR").write_text(relax_poscar)
    if mattersim:
        (d / "mattersim.json").write_text(json.dumps({
            "mp_id": mp_id, "formula": "Fe1", "chemsys": "Fe",
            "counts": counts or {"Fe": 2}, "n_atoms": sum((counts or {"Fe": 2}).values()),
            "e_mlip_relaxed": -15.5, "e_mlip_static": -15.4,
            "mlip_model": "MatterSim-v1.0.0-5M.pth", "mlip_converged": True,
            "recipe_id": "an-older-id", "state": "pending",
        }))
    return d


def run(root, recipe, **kw):
    return A.adopt(root, rid="r" * 64, recipe=recipe, **kw)


# -- the happy path, and what it must carry --------------------------------

def test_a_finished_phase_is_adopted_with_the_static_energy(tree, recipe):
    report = run(tree, recipe)
    assert report.adopted == 1
    phase = read_record("r" * 64, "mp-1")
    assert phase.e_dft == pytest.approx(-16.0)
    assert phase.state == "done"
    assert phase.dft_converged is True


def test_the_energy_comes_from_static_and_not_from_relax(tree, recipe):
    """`relax` finds the geometry; `static` is computed AT it. Taking the relax
    number is a ~10 meV/atom error that looks entirely plausible."""
    (tree / "mp-1" / "relax" / "OUTCAR").write_text(
        OUTCAR.replace("-16.000000", "-15.000000"))
    run(tree, recipe)
    assert read_record("r" * 64, "mp-1").e_dft == pytest.approx(-16.0)


def test_the_mlip_half_is_carried_across(tree, recipe):
    """Re-running MatterSim to rediscover numbers on disk would cost hours and
    could not improve them."""
    run(tree, recipe)
    phase = read_record("r" * 64, "mp-1")
    assert phase.e_mlip_relaxed == pytest.approx(-15.5)
    assert phase.mlip_model == "MatterSim-v1.0.0-5M.pth"


def test_the_record_is_rekeyed_to_the_recipe_being_adopted_into(tree, recipe):
    """mattersim.json carries whichever recipe_id the MLIP pass used. The DFT
    energy decides which cache the record belongs in, and that is this one."""
    run(tree, recipe)
    assert read_record("r" * 64, "mp-1").recipe_id == "r" * 64


def test_the_physics_tags_actually_used_are_recorded(tree, recipe):
    """`recipe_id` says what the policy was; this says what the phase got."""
    run(tree, recipe)
    assert read_record("r" * 64, "mp-1").incar_physics == {"ISMEAR": "-5", "ISPIN": "2"}


def test_a_gaussian_smeared_phase_records_that_it_differs(tree, recipe):
    """83 of 2,713 real phases ran ISMEAR 0 after the tetrahedron method failed.
    ISMEAR enters the energy through the smearing entropy, so it is not a
    robustness knob and must not vanish into the average."""
    (tree / "mp-1" / "static" / "INCAR").write_text(
        STATIC_INCAR.replace("ISMEAR = -5", "ISMEAR = 0"))
    run(tree, recipe)
    assert read_record("r" * 64, "mp-1").incar_physics["ISMEAR"] == "0"


def test_where_the_record_came_from_is_recorded(tree, recipe):
    run(tree, recipe)
    assert "mp-1" in read_record("r" * 64, "mp-1").adopted_from


# -- what must be refused --------------------------------------------------

def test_a_static_that_never_finished_is_refused(tree, recipe):
    """A killed static still leaves an OSZICAR holding its last electronic
    step, so "has an energy" is true for a run that never finished -- the same
    trap as the 84 tier rows that are `failed` and carry a relax energy.
    Reaching VASP's own epilogue is the only thing that settles it."""
    (tree / "mp-1" / "static" / "OUTCAR").write_text("   NIONS =      2\n")
    report = run(tree, recipe)
    assert report.adopted == 0
    why = report.refused[0][1]
    assert "did not finish" in why
    # and it says the tempting energy is there, so nobody re-derives this
    assert "-16.0 is present but the run was cut short" in why


def test_a_directory_from_a_different_policy_is_refused(tree, recipe):
    """A different ENCUT is not a cheaper version of the same answer."""
    (tree / "mp-1" / "static" / "INCAR").write_text(
        STATIC_INCAR.replace("ENCUT = 520", "ENCUT = 400"))
    report = run(tree, recipe)
    assert report.adopted == 0
    assert "different policy" in report.refused[0][1]
    assert "ENCUT" in report.refused[0][1]


def test_a_destroyed_geometry_is_refused_on_its_energy(tree, recipe):
    """VASP_FAILURES failure 12: a runaway cell still produces a total energy,
    and that energy still builds a hull. +226 eV/atom is not a phase."""
    # The energy is read from OSZICAR's `E0=`, not from OUTCAR's TOTEN -- so a
    # test that edited only the OUTCAR would pass while proving nothing.
    (tree / "mp-1" / "static" / "OSZICAR").write_text(
        "   1 F= 0.45200000E+03 E0= 0.45200000E+03  d E =0.0  mag=     0.0000\n")
    report = run(tree, recipe)
    assert report.adopted == 0
    assert "outside the physical range" in report.refused[0][1]


def test_an_energy_that_disagrees_with_its_composition_is_refused(tree, recipe):
    """A total energy paired with the wrong composition is a hull vertex wrong
    by exactly that ratio, and the hull still builds."""
    make_phase(tree, "mp-2", counts={"Fe": 4})     # OUTCAR says NIONS = 2
    report = run(tree, recipe)
    assert ("mp-2", ) == tuple(m for m, _ in report.refused)
    assert "disagree" in dict(report.refused)["mp-2"]


def test_a_refused_phase_is_not_written_at_all(tree, recipe):
    """Not written as `failed` either: `coverage` must see it as missing, and a
    stale `done` record from an earlier run must not survive a later refusal."""
    (tree / "mp-1" / "static" / "INCAR").write_text(
        STATIC_INCAR.replace("PREC = Accurate", "PREC = Normal"))
    run(tree, recipe)
    assert read_record("r" * 64, "mp-1") is None


# -- what must be kept, but flagged ----------------------------------------

def test_a_vacuum_box_is_kept_and_flagged_not_refused(tree, recipe):
    """Six real phases are isolated atoms or clusters in vacuum, up to 613
    A^3/atom. Refusing them would leave `coverage` incomplete and block every
    hull in their chemistry -- to remove points that are always high in energy
    and so can never define a lower envelope."""
    big = POSCAR.replace("   3.0 0.0 0.0", "  30.0 0.0 0.0")
    make_phase(tree, "mp-3", poscar=big)
    report = run(tree, recipe)
    assert report.adopted == 2
    assert not report.refused
    assert any(m == "mp-3" and "vacuum box" in why for m, why in report.suspect)
    assert read_record("r" * 64, "mp-3").e_dft == pytest.approx(-16.0)


def test_a_large_relaxation_is_flagged_not_refused(tree, recipe):
    """mp-1207665 came in at 380 A^3/atom of vacuum and relaxed correctly to
    19.8. The big move is the relaxation working, not failing."""
    small = POSCAR.replace("   3.0 0.0 0.0", "   1.2 0.0 0.0")
    make_phase(tree, "mp-4", relax_poscar=small)
    report = run(tree, recipe)
    assert report.adopted == 2
    assert any(m == "mp-4" and "volume moved" in why for m, why in report.suspect)


# -- the tree as a whole ---------------------------------------------------

def test_phases_in_EXCLUDED_json_are_skipped(tree, recipe):
    make_phase(tree, "mp-9")
    (tree / "EXCLUDED.json").write_text(json.dumps(
        {"mp-9": {"decision": "permanently excluded, at the user's instruction"}}))
    report = run(tree, recipe)
    assert report.excluded == 1
    assert read_record("r" * 64, "mp-9") is None


def test_a_dry_run_writes_nothing(tree, recipe):
    report = run(tree, recipe, dry_run=True)
    assert report.adopted == 1
    assert read_record("r" * 64, "mp-1") is None


def test_a_phase_already_cached_is_left_alone(tree, recipe):
    write_record(ComputedPhase(mp_id="mp-1", formula="Fe1", chemsys="Fe",
                               counts={"Fe": 2}, n_atoms=2, recipe_id="r" * 64,
                               e_dft=-99.0, state="done"))
    report = run(tree, recipe)
    assert report.skipped_existing == 1
    assert read_record("r" * 64, "mp-1").e_dft == pytest.approx(-99.0)


def test_refresh_re_reads_a_cached_phase(tree, recipe):
    write_record(ComputedPhase(mp_id="mp-1", formula="Fe1", chemsys="Fe",
                               counts={"Fe": 2}, n_atoms=2, recipe_id="r" * 64,
                               e_dft=-99.0, state="done"))
    run(tree, recipe, refresh=True)
    assert read_record("r" * 64, "mp-1").e_dft == pytest.approx(-16.0)


def test_provenance_written_by_the_mlip_pass_survives_adoption(tree, recipe):
    """`mlip_carried_from` records that the MLIP half came from a different
    recipe_id. It was being written into the cache while not existing as a
    dataclass field, so every read silently dropped it."""
    write_record(ComputedPhase(mp_id="mp-1", formula="Fe1", chemsys="Fe",
                               counts={"Fe": 2}, n_atoms=2, recipe_id="r" * 64,
                               e_mlip_relaxed=-15.5,
                               mlip_carried_from="an-older-id", state="pending"))
    run(tree, recipe)
    phase = read_record("r" * 64, "mp-1")
    assert phase.mlip_carried_from == "an-older-id"
    assert phase.e_dft == pytest.approx(-16.0)


# -- the policy check's own rules ------------------------------------------

def test_a_tag_the_recipe_leaves_to_resolution_is_judged_by_the_tree(tree, recipe):
    """LMAXMIX is computed from whether f sits in the valence, so the recipe
    states no value. The only available standard is the rest of the tree."""
    expected = A.expected_tags(recipe, tree)
    assert expected["LMAXMIX"] == ("4", "all 1 directories")
    assert expected["ENCUT"][1] == "the recipe"


def test_a_tag_the_tree_disagrees_on_is_not_checked_at_all(tree, recipe):
    """A split tree cannot vouch for itself, and refusing everything would be
    an accusation the evidence does not support."""
    make_phase(tree, "mp-5", incar=STATIC_INCAR.replace("LMAXMIX = 4", "LMAXMIX = 6"))
    assert "LMAXMIX" not in A.expected_tags(recipe, tree)
    assert run(tree, recipe).adopted == 2


def test_numbers_spelled_differently_are_one_value(tree, recipe):
    """`1e-05`, `1.0E-05` and ` 1e-5 ` are the same SIGMA. A string compare
    would refuse a directory over its formatting."""
    (tree / "mp-1" / "static" / "INCAR").write_text(
        STATIC_INCAR.replace("SIGMA = 0.05", "SIGMA = 5.0E-02"))
    assert run(tree, recipe).adopted == 1


def test_every_tag_in_a_real_incar_is_classified(tree, recipe):
    """An unclassified tag is reported rather than waved through, so this list
    staying complete is what keeps that signal meaningful."""
    from cspflow.dft.vasp.parse import read_incar

    known = set(A.INVARIANT) | set(A.PHYSICS) | set(A.ROBUSTNESS) | set(A.IGNORED)
    assert not set(read_incar(tree / "mp-1" / "static" / "INCAR")) - known
