"""Stage 3 -- MP reference fetching, its cache, and the trap inside it.

Everything but the last class runs offline against a hand-written cache file.
The network tests are marked and skip without a key, but they are the ones that
matter most: they check our hull against MP's own `energy_above_hull`, which is
the only independent verification of the whole reference path available.
"""

import json
import os
from pathlib import Path

import pytest

from cspflow.reference.hull import build_hull
from cspflow.reference.mp import (
    THERMO_GGA,
    THERMO_MIXED,
    FetchResult,
    ReferenceEntry,
    ReferenceError,
    assert_scale_matches_thermo_type,
    reference_root,
    compare_snapshots,
    fetch_chemsys,
    snapshot_id,
    sub_systems,
)


def entry(mp_id, counts, raw, corrected=None, **kw):
    n = sum(counts.values())
    return ReferenceEntry(
        mp_id=mp_id, formula="".join(f"{e}{counts[e]}" for e in sorted(counts)),
        chemsys="-".join(sorted(counts)), counts=counts, n_atoms=n,
        thermo_type=kw.pop("thermo_type", THERMO_GGA),
        e_raw_per_atom=raw,
        e_corrected_per_atom=corrected if corrected is not None else raw,
        **kw,
    )


@pytest.fixture
def cached(tmp_path):
    """A cache file for Fe-Sm, written by hand."""
    entries = [entry("mp-13", {"Fe": 1}, 0.0), entry("mp-69", {"Sm": 1}, 0.0),
               entry("mp-1729", {"Fe": 2, "Sm": 1}, -7.1966)]
    result = FetchResult(chemsys="Fe-Sm", thermo_type=THERMO_GGA, entries=entries,
                         snapshot_id=snapshot_id(entries), fetched_at="2026-08-27T00:00:00")
    path = tmp_path / f"Fe-Sm__{THERMO_GGA.replace('+', 'p')}.json"
    path.write_text(json.dumps({
        "chemsys": "Fe-Sm", "thermo_type": THERMO_GGA,
        "snapshot_id": result.snapshot_id, "fetched_at": result.fetched_at,
        "warnings": [], "entries": [e.__dict__ for e in entries],
    }, indent=2))
    return tmp_path


# --------------------------------------------------------------------------
# The trap
# --------------------------------------------------------------------------


class TestThermoTypeGuard:
    def test_raw_from_the_mixed_scheme_is_refused(self):
        """On GGA_GGA+U_R2SCAN the 'uncorrected' field is not a GGA energy."""
        with pytest.raises(ReferenceError, match="cannot be taken from"):
            assert_scale_matches_thermo_type(THERMO_MIXED, "raw")

    def test_the_error_carries_the_measured_size_of_the_problem(self):
        with pytest.raises(ReferenceError, match="12.21"):
            assert_scale_matches_thermo_type(THERMO_MIXED, "raw")

    def test_raw_from_pure_gga_is_fine(self):
        assert_scale_matches_thermo_type(THERMO_GGA, "raw")

    def test_corrected_from_the_mixed_scheme_is_fine(self):
        """The mixed scheme's own corrected numbers ARE self-consistent."""
        assert_scale_matches_thermo_type(THERMO_MIXED, "mp_corrected")

    def test_fetch_refuses_before_it_would_hit_the_network(self, tmp_path):
        with pytest.raises(ReferenceError, match="cannot be taken from"):
            fetch_chemsys("Fe-Sm", thermo_type=THERMO_MIXED, energy_scale="raw",
                          cache=tmp_path, api_key="unused")


# --------------------------------------------------------------------------
# Sub-systems
# --------------------------------------------------------------------------


class TestSubSystems:
    def test_a_binary_expands_to_its_elements(self):
        assert sub_systems("Fe-Sm") == ["Fe", "Sm", "Fe-Sm"]

    def test_a_ternary_expands_to_all_seven(self):
        assert sub_systems("Co-Fe-Sm") == [
            "Co", "Fe", "Sm", "Co-Fe", "Co-Sm", "Fe-Sm", "Co-Fe-Sm"]

    def test_an_element_is_its_own_only_sub_system(self):
        assert sub_systems("Fe") == ["Fe"]

    def test_order_does_not_matter(self):
        assert sub_systems("Sm-Fe") == sub_systems("Fe-Sm")

    def test_this_is_why_the_elemental_guard_would_otherwise_fire(self):
        """MP's chemsys='Fe-Sm' matches exactly that binary: no elemental Fe or Sm."""
        assert "Fe" in sub_systems("Fe-Sm") and "Sm" in sub_systems("Fe-Sm")


# --------------------------------------------------------------------------
# Entries
# --------------------------------------------------------------------------


class TestReferenceEntry:
    def test_energy_on_scale_returns_a_total_not_per_atom(self):
        e = entry("mp-1", {"Fe": 2, "Sm": 1}, -7.0)
        assert e.energy_on("raw") == pytest.approx(-21.0)

    def test_both_scales_are_kept_and_neither_preferred(self):
        e = entry("mp-1", {"Fe": 1}, raw=-8.0, corrected=-8.5)
        assert e.energy_on("raw") == -8.0
        assert e.energy_on("mp_corrected") == -8.5

    def test_a_missing_energy_is_an_error_not_a_zero(self):
        e = entry("mp-1", {"Fe": 1}, raw=None, corrected=-8.0)
        with pytest.raises(ReferenceError, match="no raw energy"):
            e.energy_on("raw")

    def test_hull_entry_carries_the_scale(self):
        assert entry("mp-1", {"Fe": 1}, -8.0).to_hull_entry("raw").scale == "raw"
        assert entry("mp-1", {"Fe": 1}, -8.0).to_hull_entry(
            "mp_corrected").scale == "mp_corrected"


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


class TestCache:
    def test_a_cached_system_needs_no_key(self, cached):
        result = fetch_chemsys("Fe-Sm", cache=cached)
        assert result.from_cache and len(result.entries) == 3

    def test_the_cache_lives_outside_any_campaign(self, monkeypatch, tmp_path):
        """Two campaigns touching Sm-Fe-Ti pay for its reference hull once."""
        monkeypatch.setenv("CSPFLOW_REFERENCE", str(tmp_path / "shared"))
        assert reference_root() == tmp_path / "shared"

    def test_no_key_and_no_cache_is_a_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MP_API_KEY", raising=False)
        with pytest.raises(ReferenceError, match="no Materials Project API key"):
            fetch_chemsys("Fe-Sm", cache=tmp_path)

    def test_the_error_does_not_invite_writing_the_key_down(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MP_API_KEY", raising=False)
        with pytest.raises(ReferenceError, match="never written to a cache file"):
            fetch_chemsys("Fe-Sm", cache=tmp_path)

    def test_a_tampered_cache_is_reported_not_trusted(self, cached):
        """Its snapshot id no longer identifies its contents, so provenance would lie."""
        path = next(cached.glob("*.json"))
        payload = json.loads(path.read_text())
        payload["entries"][0]["e_raw_per_atom"] = -99.0
        path.write_text(json.dumps(payload))
        result = fetch_chemsys("Fe-Sm", cache=cached)
        assert any("has been modified" in w for w in result.warnings)

    def test_the_recomputed_snapshot_is_the_one_reported(self, cached):
        path = next(cached.glob("*.json"))
        payload = json.loads(path.read_text())
        stored = payload["snapshot_id"]
        payload["entries"][0]["e_raw_per_atom"] = -99.0
        path.write_text(json.dumps(payload))
        assert fetch_chemsys("Fe-Sm", cache=cached).snapshot_id != stored


class TestSnapshots:
    def test_the_same_entries_hash_the_same(self):
        a = [entry("mp-1", {"Fe": 1}, -8.0), entry("mp-2", {"Sm": 1}, -4.0)]
        b = [entry("mp-2", {"Sm": 1}, -4.0), entry("mp-1", {"Fe": 1}, -8.0)]
        assert snapshot_id(a) == snapshot_id(b)          # order must not matter

    def test_a_changed_energy_changes_the_snapshot(self):
        a = [entry("mp-1", {"Fe": 1}, -8.0)]
        b = [entry("mp-1", {"Fe": 1}, -8.001)]
        assert snapshot_id(a) != snapshot_id(b)

    def test_compare_reports_what_moved(self):
        old = FetchResult("Fe-Sm", THERMO_GGA, [entry("mp-1", {"Fe": 1}, -8.0),
                                                entry("mp-9", {"Sm": 1}, -4.0)])
        new = FetchResult("Fe-Sm", THERMO_GGA, [entry("mp-1", {"Fe": 1}, -8.1),
                                                entry("mp-2", {"Fe": 2}, -7.0)])
        notes = " ".join(compare_snapshots(old, new))
        assert "mp-2" in notes and "is new" in notes
        assert "mp-9" in notes and "is gone" in notes
        assert "mp-1" in notes and "-0.100000" in notes

    def test_an_unchanged_refetch_reports_nothing(self):
        entries = [entry("mp-1", {"Fe": 1}, -8.0)]
        old = FetchResult("Fe-Sm", THERMO_GGA, entries)
        new = FetchResult("Fe-Sm", THERMO_GGA, list(entries))
        assert compare_snapshots(old, new) == []


# --------------------------------------------------------------------------
# Against the live API
# --------------------------------------------------------------------------

def mp_api_usable() -> bool:
    """Whether `mp_api` can actually be used here, not merely imported.

    In the base environment `mp_api` imports fine and then fails inside pydantic
    with `Expected sequence to have exactly 1 generic parameter` -- a version
    mismatch between mp_api's emmet models and that env's pydantic. Skipping on
    `find_spec` alone would let these tests fail there for a reason that has
    nothing to do with cspflow.
    """
    if not os.environ.get("MP_API_KEY"):
        return False
    try:
        import warnings

        from mp_api.client import MPRester

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with MPRester(os.environ["MP_API_KEY"]) as mpr:
                mpr.materials.thermo.search(
                    chemsys=["Fe"], fields=["material_id"],
                    num_chunks=1, chunk_size=1,
                )
        return True
    except Exception:
        return False


needs_mp = pytest.mark.skipif(
    not mp_api_usable(), reason="MP_API_KEY unset or mp_api unusable in this env"
)


@needs_mp
@pytest.mark.network
class TestAgainstMaterialsProject:
    def test_our_hull_reproduces_mp_s_own_e_above_hull(self, tmp_path):
        """The only independent check of the whole reference path there is.

        Measured 2026-08-27 across Fe-Sm, Co-Gd, Fe-Ti and Co-Fe-Sm -- 123
        entries including a ternary -- agreement was exact to machine precision
        on values spanning 0 to 0.62 eV/atom.
        """
        result = fetch_chemsys("Fe-Sm", cache=tmp_path)
        hull = build_hull(result.hull_entries("raw"))
        for e in result.entries:
            if e.e_above_hull_mp is not None:
                assert hull.e_above_hull[e.mp_id] == pytest.approx(
                    e.e_above_hull_mp, abs=1e-6
                ), f"{e.mp_id} {e.formula}"

    def test_a_fetch_includes_the_elemental_references(self, tmp_path):
        result = fetch_chemsys("Fe-Sm", cache=tmp_path)
        elemental = {e.formula for e in result.entries if len(e.counts) == 1}
        assert {"Fe1", "Sm1"} <= elemental

    def test_only_the_requested_thermo_type_comes_back(self, tmp_path):
        result = fetch_chemsys("Fe-Sm", cache=tmp_path)
        assert {e.thermo_type for e in result.entries} == {THERMO_GGA}

    def test_the_second_call_is_served_from_cache(self, tmp_path):
        first = fetch_chemsys("Fe-Sm", cache=tmp_path)
        second = fetch_chemsys("Fe-Sm", cache=tmp_path)
        assert not first.from_cache and second.from_cache
        assert first.snapshot_id == second.snapshot_id

    def test_corrections_are_zero_for_an_intermetallic(self, tmp_path):
        """Stage 3b's claim, checked against the live data rather than asserted."""
        result = fetch_chemsys("Fe-Sm", cache=tmp_path)
        for e in result.entries:
            assert e.e_raw_per_atom == pytest.approx(e.e_corrected_per_atom, abs=1e-9)
