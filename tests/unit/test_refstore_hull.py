"""Hull entries read from the store folder, on each of its three scales.

The store is the living source of truth and it keeps being extended, so the
hull is built by reading it rather than from an exported copy keyed on
`recipe_id`. A second copy of the same numbers goes stale: measured
2026-09-11, that cache held 2,713 DFT energies while the store held 3,408.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cspflow.reference.refstore import (ENERGY_SOURCES, StoreError, coverage,
                                        entries_for)

COLUMNS = ["mp_id", "formula", "chemsys", "n_atoms", "folder", "mlip_state",
           "e_mlip_relaxed", "relax_state", "relax_converged", "relax_core_hours",
           "static_state", "e_static_eV", "e_per_atom_eV", "magnetization_uB",
           "ready", "fail_reason"]


def _store(tmp_path: Path, rows, mp_entries, ignored=None) -> Path:
    root = tmp_path / "store"
    (root / "mp-cache").mkdir(parents=True)
    with (root / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    (root / "mp-cache" / "A-B__GGA_GGApU.json").write_text(
        json.dumps({"chemsys": "A-B", "entries": mp_entries}))
    if ignored is not None:
        (root / "ignored.json").write_text(json.dumps({"ignored": ignored}))
    return root


def _rows():
    # two elemental end members and one binary, so a hull actually exists
    return [
        dict(mp_id="mp-1", formula="Fe", chemsys="Fe", n_atoms="2", folder="mp-1-Fe",
             e_static_eV="-16.0", e_per_atom_eV="-8.0", e_mlip_relaxed="-15.6",
             ready="True", static_state="done", relax_converged="True"),
        dict(mp_id="mp-2", formula="B", chemsys="B", n_atoms="2", folder="mp-2-B",
             e_static_eV="-12.0", e_per_atom_eV="-6.0", e_mlip_relaxed="-11.8",
             ready="True", static_state="done", relax_converged="True"),
        dict(mp_id="mp-3", formula="FeB", chemsys="B-Fe", n_atoms="4", folder="mp-3-FeB",
             e_static_eV="-30.0", e_per_atom_eV="-7.5", e_mlip_relaxed="-29.4",
             ready="True", static_state="done", relax_converged="True"),
    ]


def _mp():
    return [
        {"mp_id": "mp-1", "chemsys": "Fe", "counts": {"Fe": 1}, "n_atoms": 1,
         "e_raw_per_atom": -8.2, "e_corrected_per_atom": -8.2, "run_type": "GGA"},
        {"mp_id": "mp-2", "chemsys": "B", "counts": {"B": 1}, "n_atoms": 1,
         "e_raw_per_atom": -6.2, "e_corrected_per_atom": -6.2, "run_type": "GGA"},
        {"mp_id": "mp-3", "chemsys": "B-Fe", "counts": {"Fe": 1, "B": 1}, "n_atoms": 2,
         "e_raw_per_atom": -7.7, "e_corrected_per_atom": -7.7, "run_type": "GGA"},
    ]


@pytest.mark.parametrize("source", ENERGY_SOURCES)
def test_every_energy_source_builds_a_hull_from_the_same_folder(tmp_path, source):
    root = _store(tmp_path, _rows(), _mp())
    entries = entries_for("B-Fe", source, root=root)
    assert len(entries) == 3
    assert {e.source for e in entries} == {
        {"dft": "ours", "mlip": "mlip", "mp": "mp"}[source]}


def test_the_three_scales_are_different_numbers(tmp_path):
    """Not a tautology: it is what makes mixing them a real error (D101)."""
    root = _store(tmp_path, _rows(), _mp())
    per_atom = {}
    for source in ENERGY_SOURCES:
        e = next(x for x in entries_for("B-Fe", source, root=root)
                 if x.label.endswith("-Fe"))
        per_atom[source] = round(e.e_per_atom, 4)
    assert per_atom["dft"] == -8.0
    assert per_atom["mp"] == -8.2
    assert len(set(per_atom.values())) == 3, per_atom


def test_our_counts_are_scaled_to_our_cell_not_mp_s(tmp_path):
    """MP's counts describe MP's cell; a wrong count puts the point in the
    wrong place on the hull and nothing downstream can tell."""
    root = _store(tmp_path, _rows(), _mp())
    ours = next(e for e in entries_for("B-Fe", "dft", root=root)
                if e.label == "mp-3-FeB")
    assert ours.counts == {"Fe": 2, "B": 2}      # our cell is 4 atoms, MP's is 2
    assert ours.n_atoms == 4
    theirs = next(e for e in entries_for("B-Fe", "mp", root=root)
                  if e.label == "mp-3-FeB")
    assert theirs.counts == {"Fe": 1, "B": 1}    # MP's own cell


def test_a_missing_energy_is_refused_not_filled_from_another_scale(tmp_path):
    rows = _rows()
    rows[2]["ready"] = "False"
    rows[2]["e_static_eV"] = ""
    root = _store(tmp_path, rows, _mp())
    cov = coverage("B-Fe", "dft", root=root)
    assert cov.missing == ["mp-3"] and not cov.complete
    with pytest.raises(StoreError) as exc:
        entries_for("B-Fe", "dft", root=root)
    assert "mp-3" in str(exc.value)
    # ... and the same structure still has an MLIP energy, so that hull builds
    assert len(entries_for("B-Fe", "mlip", root=root)) == 3


def test_an_ignored_phase_does_not_block_the_hull(tmp_path):
    """Ignored is not failed: a phase far above the hull cannot be a vertex,
    so refusing over it blocks a campaign for nothing."""
    rows = _rows()
    rows[2]["ready"] = "False"
    rows[2]["e_static_eV"] = ""
    root = _store(tmp_path, rows, _mp(),
                  ignored={"mp-3": {"folder": "mp-3-FeB", "mp_e_above_hull": 0.9}})
    cov = coverage("B-Fe", "dft", root=root)
    assert cov.missing == [] and cov.ignored == ["mp-3"]
    assert len(entries_for("B-Fe", "dft", root=root)) == 2


def test_an_unknown_energy_source_is_refused(tmp_path):
    root = _store(tmp_path, _rows(), _mp())
    with pytest.raises(StoreError):
        entries_for("B-Fe", "nonsense", root=root)      # type: ignore[arg-type]


def test_a_system_with_no_phases_says_so(tmp_path):
    root = _store(tmp_path, _rows(), _mp())
    with pytest.raises(StoreError) as exc:
        entries_for("Cu-Zn", "dft", root=root)
    assert "no phases in the store" in str(exc.value)


def test_store_reads_are_reused_until_the_files_change(tmp_path):
    """D144: `entries_for` re-read index.csv and ~680 mp-cache files for EVERY
    chemical system -- 391,104 opens and 292 s for a run with nothing to place.
    Reuse is only safe if a change on disk is still seen."""
    import json
    import os

    from cspflow.reference import refstore

    refstore._READ_CACHE.clear()
    (tmp_path / "mp-cache").mkdir()
    index = tmp_path / "index.csv"
    index.write_text("mp_id,chemsys,ready\nmp-1,Fe,True\n")

    first = refstore.load_index(tmp_path)
    assert refstore.load_index(tmp_path) is first, "read again with nothing changed"

    index.write_text("mp_id,chemsys,ready\nmp-1,Fe,True\nmp-2,Co,False\n")
    assert [r["mp_id"] for r in refstore.load_index(tmp_path)] == ["mp-1", "mp-2"]

    assert refstore.load_mp_cache(tmp_path) == {}
    tmp = tmp_path / "mp-cache" / "Fe__GGA_GGApU.json.tmp"
    tmp.write_text(json.dumps({"entries": [{"mp_id": "mp-13", "chemsys": "Fe"}]}))
    os.replace(tmp, tmp_path / "mp-cache" / "Fe__GGA_GGApU.json")     # as mp.py writes
    assert set(refstore.load_mp_cache(tmp_path)) == {"mp-13"}
