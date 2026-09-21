"""Stage 7 as a driver stage: extraction, the DFT hull, and idempotence."""

import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.build import bulk

from cspflow.config.loader import load_campaign
from cspflow.db.store import Origin, Store, StructureState
from cspflow.stages.analyze_stage import DIR_KEY, DONE_KEY, AnalyzeStage

CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/seeds"]}}
reference:
  mode: {mode}
dft:
  recipe: magnets
  magnetism: {{mode: ferrimagnetic_retm}}
"""

OUTCAR = """\
   NIONS =      3
  free  energy   TOTEN  =       -20.000000 eV
  energy  without entropy=      -20.100000  energy(sigma->0) =      -20.000000
  number of electron     30.0000000 magnetization       4.0000000

 magnetization (x)

# of ion       s       p       d       tot
------------------------------------------
    1       -0.007  -0.066  -0.175  -0.100
    2       -0.004  -0.034   1.310   1.500
    3       -0.004  -0.034   1.310   1.500
--------------------------------------------------
tot         -0.015  -0.134   2.445   2.900

 reached required accuracy - stopping structural energy minimisation
 General timing and accounting informations for this job:
                         Elapsed time (sec):     100.0
"""

OSZICAR = "   1 F= -.20000000E+02 E0= -.20000000E+02  d E =0.0  mag=     4.0000\n"

CONTCAR = """\
Gd1 Co2
1.0
   4.0 0.0 0.0
   0.0 4.0 0.0
   0.0 0.0 4.0
Gd Co
1 2
Direct
0.00 0.00 0.00
0.50 0.50 0.00
0.00 0.50 0.50
"""


def _campaign(tmp_path, mode):
    (tmp_path / "seeds").mkdir(exist_ok=True)
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path, mode=mode))
    return load_campaign(path)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """The default campaign, on `reference.mode: mp_energies`.

    The extraction tests do not touch a hull, and the hull tests below that use
    this fixture are specifically about the MP scale -- so the mode is named
    here rather than inherited, which is what makes those tests say what they
    are testing.  The empty cache keeps `recompute` from reaching the network
    if a test ever changes mode underneath it.
    """
    monkeypatch.setenv("CSPFLOW_REFERENCE", str(tmp_path / "cache"))
    return _campaign(tmp_path, "mp_energies")


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "c.db", campaign="t") as s:
        yield s


def make_job(tmp_path, name="dft-1-relax", nsw=100):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "OUTCAR").write_text(OUTCAR)
    (d / "OSZICAR").write_text(OSZICAR)
    (d / "CONTCAR").write_text(CONTCAR)
    (d / "INCAR").write_text(f"NSW = {nsw}\nLORBIT = 11\n")
    return d


def add_done(store, tmp_path, energy=-20.0, symbols="GdCo2", **kv):
    atoms = Atoms(symbols, positions=[(0, 0, 0), (2, 2, 0), (0, 2, 2)],
                  cell=[4, 4, 4], pbc=True)
    return store.add_structure(atoms, origin=Origin.generated,
                               state=StructureState.dft_done,
                               vasp_energy=energy, **kv)


# -- extraction ------------------------------------------------------------

def test_properties_are_extracted_and_stored(cfg, store, tmp_path):
    job = make_job(tmp_path)
    sid = add_done(store, tmp_path, **{DIR_KEY: str(job), "z": 1})
    report = AnalyzeStage(cfg).run(store)

    assert report.claimed == 1
    row = store.get_structure(sid)
    kv = row.key_value_pairs
    assert kv["m_dft_raw"] == pytest.approx(4.0)
    assert kv["volume"] == pytest.approx(64.0)
    assert kv["spacegroup"] > 0
    assert kv[DONE_KEY] is True


def test_the_two_moments_are_stored_as_separate_properties(cfg, store, tmp_path):
    """`m_dft_raw` is computed; `m_s_reconstructed` is a model on top of it."""
    job = make_job(tmp_path)
    sid = add_done(store, tmp_path, **{DIR_KEY: str(job), "z": 1})
    AnalyzeStage(cfg).run(store)

    props = {p["key"]: p["value"] for p in store.properties(sid)}
    assert props["m_dft_raw"] == pytest.approx(4.0)
    # TM sublattice 3.0, one Gd at -7.0: 3.0 - 7.0 = -4.0
    assert props["m_s_reconstructed"] == pytest.approx(-4.0)
    assert props["m_dft_raw"] != props["m_s_reconstructed"]


def test_the_sublattice_split_is_stored(cfg, store, tmp_path):
    job = make_job(tmp_path)
    sid = add_done(store, tmp_path, **{DIR_KEY: str(job), "z": 1})
    AnalyzeStage(cfg).run(store)
    props = {p["key"]: p["value"] for p in store.properties(sid)}
    assert props["m_rare_earth"] == pytest.approx(-0.1)
    assert props["m_transition_metal"] == pytest.approx(3.0)


def test_a_second_run_does_not_re_read_the_outcar(cfg, store, tmp_path):
    job = make_job(tmp_path)
    add_done(store, tmp_path, **{DIR_KEY: str(job), "z": 1})
    stage = AnalyzeStage(cfg)
    assert stage.run(store).claimed == 1
    assert stage.pending(store) == 0
    assert stage.run(store).claimed == 0


def test_a_structure_with_no_recorded_directory_is_marked_and_noted(cfg, store,
                                                                    tmp_path):
    sid = add_done(store, tmp_path)
    AnalyzeStage(cfg).run(store)
    kv = store.get_structure(sid).key_value_pairs
    assert kv[DONE_KEY] is True
    assert "no DFT directory" in kv["analyze_note"]


def test_an_unconverged_job_is_analysed_but_flagged(cfg, store, tmp_path):
    job = make_job(tmp_path)
    (job / "OUTCAR").write_text(OUTCAR.replace(
        "reached required accuracy - stopping structural energy minimisation", ""))
    sid = add_done(store, tmp_path, **{DIR_KEY: str(job), "z": 1})
    AnalyzeStage(cfg).run(store)
    kv = store.get_structure(sid).key_value_pairs
    assert kv["volume"] == pytest.approx(64.0)
    assert "not converged" in kv["analyze_note"]


# -- the DFT hull ----------------------------------------------------------

def add_reference(store, formula, e_per_atom, chemsys):
    """`e_dft_raw` is per atom, as the reference stage stores it."""
    from cspflow.chem import parse_formula

    counts = parse_formula(formula)
    return store.add_reference_entry(
        mp_id=f"mp-{formula}", formula=formula, chemsys=chemsys,
        e_dft_raw=e_per_atom, e_dft_corrected=e_per_atom, run_type="GGA",
        thermo_type="GGA_GGA+U", n_atoms=sum(counts.values()), state="fetched")


def fill_store(root):
    """Our own energies for both Co-Gd elemental phases, in the store."""
    for mp_id, counts, e in CO_GD_PHASES:
        add_store_phase(root, mp_id, counts, e * sum(counts.values()))


def test_the_hull_is_built_on_our_energies(recompute_cfg, store, tmp_path,
                                           store_root):
    fill_store(store_root)
    job = make_job(tmp_path)
    sid = add_done(store, tmp_path, energy=-20.0, **{DIR_KEY: str(job), "z": 1})

    AnalyzeStage(recompute_cfg).run(store)
    kv = store.get_structure(sid).key_value_pairs
    # GdCo2: -20.0 total; references -3 and -5 per atom -> formation energy
    # (-20 - (-3) - 2*(-5)) / 3 = -2.333 eV/atom, and it is the only ternary
    # point, so it is on the hull.
    assert kv["dft_e_formation"] == pytest.approx(-2.3333, abs=1e-3)
    assert kv["dft_e_above_hull"] == pytest.approx(0.0, abs=1e-9)


def test_a_system_with_no_elemental_reference_is_reported_not_guessed(
        recompute_cfg, store, tmp_path, store_root):
    job = make_job(tmp_path)
    add_done(store, tmp_path, **{DIR_KEY: str(job), "z": 1})
    report = AnalyzeStage(recompute_cfg).run(store)
    assert "Co-Gd" in report.note


def test_a_better_candidate_pushes_the_other_off_the_hull(recompute_cfg, store,
                                                          tmp_path, store_root):
    fill_store(store_root)
    job = make_job(tmp_path)
    high = add_done(store, tmp_path, energy=-15.0, **{DIR_KEY: str(job), "z": 1})
    low = add_done(store, tmp_path, energy=-20.0, **{DIR_KEY: str(job), "z": 1})

    AnalyzeStage(recompute_cfg).run(store)
    assert store.get_structure(low).key_value_pairs["dft_e_above_hull"] == pytest.approx(0.0, abs=1e-9)
    assert store.get_structure(high).key_value_pairs["dft_e_above_hull"] > 1.0


def test_the_hull_is_recomputed_when_a_new_result_lands(recompute_cfg, store,
                                                       tmp_path, store_root):
    """`e_above_hull` is not a property of one structure; it moves when a
    competing phase appears. Re-placing every cycle is what keeps it honest."""
    fill_store(store_root)
    job = make_job(tmp_path)
    first = add_done(store, tmp_path, energy=-15.0, **{DIR_KEY: str(job), "z": 1})
    stage = AnalyzeStage(recompute_cfg)
    stage.run(store)
    assert store.get_structure(first).key_value_pairs["dft_e_above_hull"] == pytest.approx(0.0, abs=1e-9)

    add_done(store, tmp_path, energy=-20.0, **{DIR_KEY: str(job), "z": 1})
    stage.run(store)
    assert store.get_structure(first).key_value_pairs["dft_e_above_hull"] > 1.0


# -- registry --------------------------------------------------------------

def test_every_funnel_stage_now_has_an_implementation():
    from cspflow.driver import STAGE_ORDER
    from cspflow.stages import IMPLEMENTED, PLANNED

    assert PLANNED == {}
    assert set(IMPLEMENTED) == set(STAGE_ORDER)
    assert IMPLEMENTED == STAGE_ORDER


def test_mp_energies_mode_refuses_the_dft_hull_rather_than_mixing(cfg, store,
                                                                  tmp_path):
    """`mode: mp_energies` puts our DFT on MP's vertices. That is two absolute
    scales on one hull, and it used to be allowed with a warning.

    It is refused now because the error is larger than the threshold it feeds.
    Measured on the store over 3,345 phases carrying both numbers, elemental Ce
    is +1.17 eV/atom ours-minus-MP (a different 4f POTCAR), and a per-element
    correction fitted INSIDE one chemistry still leaves 42 meV/atom RMS for
    Ce-Ge-Pd and 103 for Ce-Fe-B -- against a 60 meV/atom selection threshold.
    """
    job = make_job(tmp_path)
    sid = add_done(store, tmp_path, energy=-20.0, **{DIR_KEY: str(job), "z": 1})
    add_reference(store, "Gd1", -3.0, "Gd")
    add_reference(store, "Co1", -5.0, "Co")

    note = AnalyzeStage(cfg).run(store).note
    assert "not the same scale" in note
    # No number is written -- the column is EMPTY, not wrong.
    assert store.get_structure(sid).key_value_pairs.get("dft_e_above_hull") is None


def test_the_reason_for_an_empty_dft_hull_is_recorded_on_the_structure(
        cfg, store, tmp_path):
    """A blank in the report has to be explainable, or it reads as a bug."""
    job = make_job(tmp_path)
    sid = add_done(store, tmp_path, energy=-20.0, **{DIR_KEY: str(job), "z": 1})
    add_reference(store, "Gd1", -3.0, "Gd")
    AnalyzeStage(cfg).run(store)

    row = store.sql.execute(
        "SELECT text_value FROM property WHERE structure_id=? AND key=?",
        (sid, "dft_e_above_hull_absent")).fetchone()
    assert row is not None
    assert "not the same scale" in row["text_value"]


# -- the hull on our own energies (D101) -----------------------------------
#
# `reference.mode` decided nothing until it was wired into
# `AnalyzeStage._reference_entries`: whatever the config said, every hull was
# built from our energies against MP's.  These tests are the ones that would
# have caught that, so they assert on the two things the mode now controls --
# which scale the vertices come from, and what happens when ours are missing.

def mp_cache(tmp_path, chemsys, phases):
    """An MP thermo download, on disk, so `coverage` needs no network.

    `coverage` asks MP what phases EXIST and the computed cache what we have
    energies for; a test that stubbed only the second would never exercise the
    gap between them, which is the whole subject here.
    """
    from cspflow.reference.mp import THERMO_GGA, ReferenceEntry, snapshot_id

    rows = [{
        "mp_id": mp_id, "formula": "".join(f"{e}{c[e]}" for e in sorted(c)),
        "chemsys": chemsys, "counts": c, "n_atoms": sum(c.values()),
        "thermo_type": THERMO_GGA, "e_raw_per_atom": raw,
        "e_corrected_per_atom": raw, "e_above_hull_mp": 0.0,
        "run_type": THERMO_GGA,
    } for mp_id, c, raw in phases]

    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    (cache / f"{chemsys}__{THERMO_GGA.replace('+', 'p')}.json").write_text(json.dumps({
        "chemsys": chemsys, "thermo_type": THERMO_GGA,
        "snapshot_id": snapshot_id([ReferenceEntry(**r) for r in rows]),
        "fetched_at": "2026-08-27T00:00:00", "warnings": [], "entries": rows,
    }))
    return cache


STORE_COLUMNS = ["mp_id", "formula", "chemsys", "n_atoms", "folder",
                 "mlip_state", "e_mlip_relaxed", "relax_state", "relax_converged",
                 "relax_core_hours", "static_state", "e_static_eV",
                 "e_per_atom_eV", "magnetization_uB", "ready", "fail_reason"]


def add_store_phase(root, mp_id, counts, total_energy):
    """One of our own recomputed reference phases, in the STORE folder.

    Not a `computed/<recipe_id>/` record any more.  The store is the living
    source of truth and the hull is read out of it, so there is no export step
    and no recipe_id on the hull path -- see reference/refstore.py.
    """
    import csv as _csv

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    index = root / "index.csv"
    rows = []
    if index.is_file():
        with index.open(newline="") as fh:
            rows = list(_csv.DictReader(fh))
    n = sum(counts.values())
    rows = [r for r in rows if r["mp_id"] != mp_id]
    rows.append({
        "mp_id": mp_id,
        "formula": "".join(f"{e}{counts[e]}" for e in sorted(counts)),
        "chemsys": "-".join(sorted(counts)), "n_atoms": str(n),
        "folder": f"{mp_id}-x", "e_static_eV": str(total_energy),
        "e_per_atom_eV": str(total_energy / n), "e_mlip_relaxed": str(total_energy),
        "static_state": "done", "relax_converged": "True", "ready": "True",
    })
    with index.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=STORE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in STORE_COLUMNS})


# The two elemental end members of the Co-Gd hull these tests build on.
CO_GD_PHASES = [("mp-co", {"Co": 1}, -5.0), ("mp-gd", {"Gd": 1}, -3.0)]


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    """An empty reference STORE, with MP's phase list in its mp-cache.

    `entries_for` learns which phases a system is supposed to contain from
    index.csv; the mp-cache supplies element counts and MP's own energies.
    Individual tests fill our energies in with `add_store_phase`.
    """
    import json as _json

    root = tmp_path / "store"
    (root / "mp-cache").mkdir(parents=True)
    (root / "mp-cache" / "Co-Gd__GGA_GGApU.json").write_text(_json.dumps({
        "chemsys": "Co-Gd",
        "entries": [
            {"mp_id": mp_id, "chemsys": "-".join(sorted(counts)),
             "counts": dict(counts), "n_atoms": sum(counts.values()),
             "e_raw_per_atom": e / sum(counts.values()),
             "e_corrected_per_atom": e / sum(counts.values()), "run_type": "GGA"}
            for mp_id, counts, e in CO_GD_PHASES],
    }))
    monkeypatch.setenv("CSPFLOW_STORE", str(root))
    return root


@pytest.fixture
def recompute_cfg(tmp_path, monkeypatch, store_root):
    """A campaign on `reference.mode: recompute`, with MP's phase list cached
    but NOTHING recomputed yet.  Individual tests fill the store in."""
    monkeypatch.setenv("CSPFLOW_REFERENCE", str(tmp_path / "cache"))
    mp_cache(tmp_path, "Co-Gd", CO_GD_PHASES)
    return _campaign(tmp_path, "recompute")


def test_recompute_mode_builds_the_hull_on_our_own_reference_energies(
        recompute_cfg, store, tmp_path, store_root):
    """The point of D101: every vertex on one scale, ours.

    Co at -5 and Gd at -3 eV/atom are OUR numbers here, not MP's -- the MP
    cache carries the same values only so the two are distinguishable by which
    code path reads them, not by arithmetic.
    """
    stage = AnalyzeStage(recompute_cfg)
    for mp_id, counts, e in CO_GD_PHASES:
        add_store_phase(store_root, mp_id, counts, e)

    job = make_job(tmp_path)
    sid = add_done(store, tmp_path, energy=-20.0, **{DIR_KEY: str(job), "z": 1})
    note = stage.run(store).note

    assert "1 chemical system(s) placed" in note
    kv = store.get_structure(sid).key_value_pairs
    # GdCo2 at -20.0 total against -3 + 2*(-5) = -13: 7 eV below the tie-line
    # over three atoms, so it is a hull vertex.
    assert kv["dft_e_above_hull"] == pytest.approx(0.0, abs=1e-9)
    assert kv["dft_e_formation"] == pytest.approx(-7.0 / 3, abs=1e-9)


def test_recompute_mode_never_warns_about_mixed_scales(recompute_cfg, store, tmp_path, store_root):
    """There is nothing to warn about: MP supplied no energy to this hull."""
    stage = AnalyzeStage(recompute_cfg)
    for mp_id, counts, e in CO_GD_PHASES:
        add_store_phase(store_root, mp_id, counts, e)
    add_reference(store, "Gd1", -3.0, "Gd")      # present, and must be ignored
    add_reference(store, "Co1", -5.0, "Co")

    job = make_job(tmp_path)
    add_done(store, tmp_path, energy=-20.0, **{DIR_KEY: str(job), "z": 1})
    assert "mixes our DFT" not in AnalyzeStage(recompute_cfg).run(store).note


def test_a_missing_reference_phase_refuses_rather_than_borrowing_from_mp(
        recompute_cfg, store, tmp_path, store_root):
    """The failure this whole mode exists to prevent.

    One recomputed phase absent, MP's own value sitting right there in the
    store: topping up from it would build a hull that looks exactly like a
    correct one and is wrong by the offset D101 measured.  So the system is
    refused and named, and no `dft_e_above_hull` is written at all.
    """
    stage = AnalyzeStage(recompute_cfg)
    add_store_phase(store_root, "mp-co", {"Co": 1}, -5.0)   # Gd left missing
    add_reference(store, "Gd1", -3.0, "Gd")                   # the tempting fallback
    add_reference(store, "Co1", -5.0, "Co")

    job = make_job(tmp_path)
    sid = add_done(store, tmp_path, energy=-20.0, **{DIR_KEY: str(job), "z": 1})
    note = stage.run(store).note

    assert "0 chemical system(s) placed" in note
    assert "mp-gd" in note
    assert "dft_e_above_hull" not in store.get_structure(sid).key_value_pairs


def test_the_refusal_survives_a_failed_phase_as_well_as_an_absent_one(
        recompute_cfg, store, tmp_path, store_root):
    """A phase that was attempted and diverged is not a phase we have.

    `export` writes these as `state='failed'` precisely so they are not
    mistaken for "not attempted yet"; `coverage` must count them as unusable
    either way, or a diverged run silently becomes a hull vertex with no energy.
    """
    from cspflow.reference.computed import ComputedPhase, write_record

    stage = AnalyzeStage(recompute_cfg)
    add_store_phase(store_root, "mp-co", {"Co": 1}, -5.0)
    write_record(ComputedPhase(mp_id="mp-gd", formula="Gd1", chemsys="Gd",
                               counts={"Gd": 1}, n_atoms=1,
                               recipe_id=stage.recipe_id, e_dft=None,
                               state="failed", fail_reason="ionic_step_limit"))

    job = make_job(tmp_path)
    add_done(store, tmp_path, energy=-20.0, **{DIR_KEY: str(job), "z": 1})
    assert "0 chemical system(s) placed" in stage.run(store).note


def test_a_recomputed_reference_does_not_collide_with_the_candidates_functional(
        recompute_cfg, store, tmp_path):
    """A regression guard on `ComputedPhase.to_entry`.

    It used to set `run_type='ours'`, which is a category error -- `run_type`
    names a functional, and `source` already carries the provenance. Against
    candidates that declare `run_type='GGA'`, `hull.assert_one_functional` sees
    two functionals and refuses every hull. Nothing caught it because
    `entries_for` had no caller until the mode was wired up.
    """
    from cspflow.reference.computed import ComputedPhase

    entry = ComputedPhase(mp_id="mp-co", formula="Co1", chemsys="Co",
                          counts={"Co": 1}, n_atoms=1, recipe_id="x",
                          e_dft=-5.0, state="done").to_entry()
    assert entry.run_type == ""
    assert entry.source != "mp"          # never mixable with an MP energy
    assert entry.source != "ours"        # but distinct from a candidate of ours
