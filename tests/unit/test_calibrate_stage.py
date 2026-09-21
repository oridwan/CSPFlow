"""Stage 4a as a driver stage.

Offline: the MLIP is a fake with a controllable error, and MP structures come
from a monkeypatched fetch. That lets the interesting cases -- a faithful model,
a biased one, a model that is bad at exactly one element -- be constructed
rather than hoped for.
"""

import pytest
from ase.build import bulk

from cspflow.config.loader import load_campaign
from cspflow.db.store import Store
from cspflow.mlip.base import RelaxResult
from cspflow.stages import CalibrateStage

pytest.importorskip("pymatgen.io.ase", reason="needs pymatgen")

CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/seeds"]}}
reference:
  relax_with_mlip: {relax}
calibrate:
  mp:
    thresholds:
      mae_e_per_atom: 0.05
      spearman_min: 0.90
      max_volume_drift: 0.05
"""


def make_cfg(tmp_path, relax="true"):
    (tmp_path / "seeds").mkdir(exist_ok=True)
    path = tmp_path / f"campaign-{relax}.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path, relax=relax))
    return load_campaign(path)


class FakeEngine:
    """An MLIP whose error is whatever the test says it is."""

    def __init__(self, offset=0.0, per_element=None, volume_factor=1.0, jitter=None):
        self.offset = offset
        self.per_element = per_element or {}
        self.volume_factor = volume_factor
        self.jitter = jitter or {}
        self.truth: dict[str, float] = {}
        self.seen: list[str] = []

    def _energy(self, atoms):
        formula = atoms.get_chemical_formula()
        base = self.truth.get(formula, -8.0)
        shift = self.offset + self.jitter.get(formula, 0.0)
        symbols = atoms.get_chemical_symbols()
        for element, delta in self.per_element.items():
            shift += delta * symbols.count(element) / len(symbols)
        return base + shift

    def single_point(self, atoms):
        self.seen.append(atoms.get_chemical_formula())
        e = self._energy(atoms)
        return RelaxResult(atoms=atoms, energy=e * len(atoms), e_per_atom=e,
                           converged=True, engine="fake",
                           volume_before=atoms.cell.volume,
                           volume_after=atoms.cell.volume)

    def relax(self, atoms):
        e = self._energy(atoms)
        v = atoms.cell.volume
        return RelaxResult(atoms=atoms, energy=e * len(atoms), e_per_atom=e,
                           converged=True, n_steps=5, engine="fake",
                           volume_before=v, volume_after=v * self.volume_factor)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """A store with reference entries, and a fetch that returns their structures."""
    from pymatgen.io.ase import AseAtomsAdaptor

    store = Store.create(tmp_path / "c.db", campaign="t")
    store.add_composition(formula="Fe1", chemsys="Fe", z=1, n_atoms=1, n_target=0,
                          source_mode="s", source_name="s", state="generated")

    cells = {
        "mp-1": bulk("Fe", "bcc", a=2.87, cubic=True),
        "mp-2": bulk("Ni", "fcc", a=3.52, cubic=True),
        "mp-3": bulk("Co", "fcc", a=3.54, cubic=True),
        "mp-4": bulk("Cu", "fcc", a=3.61, cubic=True),
        "mp-5": bulk("Al", "fcc", a=4.05, cubic=True),
    }
    truth = {"Fe2": -8.4, "Ni4": -5.7, "Co4": -7.1, "Cu4": -3.7, "Al4": -3.6}
    for i, (mp_id, atoms) in enumerate(cells.items()):
        store.add_reference_entry(
            mp_id=mp_id, chemsys=atoms.get_chemical_symbols()[0],
            thermo_type="GGA_GGA+U", formula=atoms.get_chemical_formula(),
            n_atoms=len(atoms), e_dft_raw=truth[atoms.get_chemical_formula()],
            e_dft_corrected=truth[atoms.get_chemical_formula()],
            snapshot_id="snap", state="fetched",
        )

    structures = {k: AseAtomsAdaptor.get_structure(v) for k, v in cells.items()}
    monkeypatch.setattr("cspflow.stages.calibrate_stage.fetch_structures",
                        lambda chemsys, **kw: structures)
    yield store, truth
    store.close()


class TestCalibrateStage:
    def test_it_is_in_process_and_free(self, tmp_path, prepared):
        assert CalibrateStage(make_cfg(tmp_path)).in_process is True

    def test_pending_counts_entries_without_an_mlip_energy(self, tmp_path, prepared):
        store, _ = prepared
        assert CalibrateStage(make_cfg(tmp_path)).pending(store) == 5

    def test_a_faithful_model_passes(self, tmp_path, prepared):
        store, truth = prepared
        engine = FakeEngine(); engine.truth = truth
        stage = CalibrateStage(make_cfg(tmp_path), engine=engine)
        stage.run(store)
        assert stage.last_report.verdict == "pass"
        assert stage.last_report.mae_e_per_atom == pytest.approx(0.0, abs=1e-9)

    def test_the_single_point_is_taken_at_mp_s_geometry(self, tmp_path, prepared):
        """Not at the MLIP's own -- that would fold geometry error into the MAE."""
        store, truth = prepared
        engine = FakeEngine(volume_factor=1.3); engine.truth = truth
        stage = CalibrateStage(make_cfg(tmp_path), engine=engine)
        stage.run(store)
        # The energy metric is unaffected by the 30% volume drift.
        assert stage.last_report.mae_e_per_atom == pytest.approx(0.0, abs=1e-9)
        # ...which is reported separately, and warned about.
        assert stage.last_report.max_volume_drift == pytest.approx(0.3, abs=1e-6)
        assert stage.last_report.verdict == "warn"

    def test_a_uniform_offset_shows_as_bias(self, tmp_path, prepared):
        store, truth = prepared
        engine = FakeEngine(offset=0.2); engine.truth = truth
        stage = CalibrateStage(make_cfg(tmp_path), engine=engine)
        stage.run(store)
        assert stage.last_report.bias_e_per_atom == pytest.approx(0.2, abs=1e-9)
        assert stage.last_report.spearman == pytest.approx(1.0)

    def test_one_bad_element_is_visible(self, tmp_path, prepared):
        """"MatterSim is bad at Sm" is the question this stage exists to answer."""
        store, truth = prepared
        engine = FakeEngine(per_element={"Cu": 1.5}); engine.truth = truth
        stage = CalibrateStage(make_cfg(tmp_path), engine=engine)
        stage.run(store)
        worst = max(stage.last_report.per_element_mae.items(), key=lambda kv: kv[1])
        assert worst[0] == "Cu"

    def test_energies_are_written_back_to_the_reference_entries(self, tmp_path, prepared):
        store, truth = prepared
        engine = FakeEngine(); engine.truth = truth
        CalibrateStage(make_cfg(tmp_path), engine=engine).run(store)
        rows = store.reference_entries()
        assert all(r["e_mlip_static"] is not None for r in rows)
        assert all(r["state"] in {"static_done", "relaxed"} for r in rows)

    def test_relaxed_energy_is_stored_separately_from_the_single_point(
            self, tmp_path, prepared):
        store, truth = prepared
        engine = FakeEngine(volume_factor=1.1); engine.truth = truth
        CalibrateStage(make_cfg(tmp_path), engine=engine).run(store)
        row = store.reference_entries()[0]
        assert row["e_mlip_static"] is not None
        assert row["e_mlip_relaxed"] is not None
        assert row["volume_drift"] == pytest.approx(0.1, abs=1e-6)

    def test_relax_can_be_turned_off(self, tmp_path, prepared):
        store, truth = prepared
        engine = FakeEngine(); engine.truth = truth
        CalibrateStage(make_cfg(tmp_path, relax="false"), engine=engine).run(store)
        row = store.reference_entries()[0]
        assert row["e_mlip_static"] is not None and row["e_mlip_relaxed"] is None
        assert row["state"] == "static_done"

    def test_the_verdict_is_recorded_for_the_gate_to_read(self, tmp_path, prepared):
        store, truth = prepared
        engine = FakeEngine(); engine.truth = truth
        CalibrateStage(make_cfg(tmp_path), engine=engine).run(store)
        row = store.latest_calibration("mp")
        assert row is not None and row["verdict"] == "pass" and row["n_points"] == 5

    def test_pending_falls_to_zero(self, tmp_path, prepared):
        store, truth = prepared
        engine = FakeEngine(); engine.truth = truth
        stage = CalibrateStage(make_cfg(tmp_path), engine=engine)
        stage.run(store)
        assert stage.pending(store) == 0

    def test_a_second_run_finds_nothing_to_do(self, tmp_path, prepared):
        store, truth = prepared
        engine = FakeEngine(); engine.truth = truth
        stage = CalibrateStage(make_cfg(tmp_path), engine=engine)
        stage.run(store)
        assert "already has one" in stage.run(store).note

    def test_it_needs_nothing_from_the_candidates(self, tmp_path, prepared):
        """Which is why it can run at the very start, in parallel with generation."""
        store, truth = prepared
        assert store.count_structures() == 0
        engine = FakeEngine(); engine.truth = truth
        stage = CalibrateStage(make_cfg(tmp_path), engine=engine)
        assert stage.run(store).reconciled == 5


# -- the 4b default (D119) -------------------------------------------------

class TestPilotIsOffByDefault:
    """4b used to `block` the expensive tier until a pilot DFT set came back.

    It is now off by default, because the reference set carries both MatterSim
    and our own DFT for 2,711 phases -- so the MLIP-against-OUR-DFT comparison
    4b existed to make is available before a campaign starts, on 68x the sample
    4b's default 40 would have given. These tests pin that default: it is a
    decision about how much a campaign is trusted to spend unsupervised, and a
    docstring cannot hold it.
    """

    @staticmethod
    def _cfg(tmp_path, extra=""):
        from cspflow.config.loader import load_campaign

        (tmp_path / "seeds").mkdir(exist_ok=True)
        path = tmp_path / "campaign.yaml"
        path.write_text(
            "name: t\nmachine: local\nworkdir: %s\n"
            "source:\n  - mode: structure_list\n    name: seeds\n"
            "    structure_list: {paths: [\"%s/seeds\"]}\n%s"
            % (tmp_path, tmp_path, extra))
        return load_campaign(path)

    def test_the_default_is_off(self, tmp_path):
        cfg = self._cfg(tmp_path)
        assert cfg.campaign.calibrate.pilot.on_fail.value == "off"

    def test_4a_still_warns_by_default(self, tmp_path):
        """4a is free and stays on: it catches gross failure for no DFT."""
        cfg = self._cfg(tmp_path)
        assert cfg.campaign.calibrate.mp.on_fail.value == "warn"

    def test_dft_is_not_gated_by_default(self, tmp_path):
        """The point of the change: a campaign reaches DFT without first
        spending 40 jobs to earn permission."""
        from cspflow.db.store import Store
        from cspflow.stages.dft_stage import DftStage

        cfg = self._cfg(tmp_path)
        with Store.create(tmp_path / "c.db", campaign="t") as store:
            assert DftStage(cfg)._pilot_gate(store) == ""

    def test_no_pilot_work_is_claimed_by_default(self, tmp_path):
        """Off at the other end too: 4b must not select a pilot set that
        nothing will ever judge."""
        from cspflow.db.store import Store
        from cspflow.stages.calibrate_stage import CalibrateStage

        cfg = self._cfg(tmp_path)
        with Store.create(tmp_path / "c.db", campaign="t") as store:
            assert CalibrateStage(cfg)._pilot_has_work(store) is False

    def test_it_can_still_be_turned_back_on(self, tmp_path):
        """Turning it on stays the honest move for a chemistry the reference
        set does not cover -- so the gate must still work when asked for.

        Screened candidates are needed to see the difference: with nothing to
        select from, 4b has no work whatever the policy, and the DFT gate stays
        open on purpose. The barrier exists to stop spending on an unchecked
        model, not to stop a campaign that has nothing to check it with.
        """
        from ase.build import bulk

        from cspflow.db.store import Origin, Store, StructureState
        from cspflow.stages.calibrate_stage import CalibrateStage
        from cspflow.stages.dedup_stage import CHECKED_KEY
        from cspflow.stages.dft_stage import DftStage

        cfg = self._cfg(tmp_path, "calibrate:\n  pilot: {on_fail: block}\n")
        assert cfg.campaign.calibrate.pilot.on_fail.value == "block"
        with Store.create(tmp_path / "c.db", campaign="t") as store:
            # Deduplicated: `_selectable` returns nothing while anything is
            # still waiting, so an undeduplicated row would make this test pass
            # for the wrong reason.
            store.add_structure(bulk("Fe", "bcc", a=2.87, cubic=True),
                                origin=Origin.generated,
                                state=StructureState.screened,
                                mlip_e_per_atom=-8.0, mlip_converged=True,
                                **{CHECKED_KEY: True})
            assert CalibrateStage(cfg)._pilot_has_work(store) is True
            # and the expensive tier is held until 4b reports
            assert DftStage(cfg)._pilot_gate(store) == ""   # nothing selected yet

    def test_with_the_default_a_selected_pilot_never_appears(self, tmp_path):
        """The whole 4b machinery stays dormant, not merely ungated: an
        unselected pilot cannot hold anything up later."""
        from ase.build import bulk

        from cspflow.db.store import Origin, Store, StructureState
        from cspflow.stages.calibrate_stage import CalibrateStage
        from cspflow.stages.dedup_stage import CHECKED_KEY

        cfg = self._cfg(tmp_path)
        with Store.create(tmp_path / "c.db", campaign="t") as store:
            store.add_structure(bulk("Fe", "bcc", a=2.87, cubic=True),
                                origin=Origin.generated,
                                state=StructureState.screened,
                                mlip_e_per_atom=-8.0, mlip_converged=True,
                                **{CHECKED_KEY: True})
            assert CalibrateStage(cfg)._pilot_has_work(store) is False
            assert CalibrateStage(cfg)._run_pilot(store) == ""
