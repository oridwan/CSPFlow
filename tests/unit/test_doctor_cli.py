"""Preflight checks and the command line."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cspflow import doctor as D
from cspflow.cli import app
from cspflow.config.loader import load_campaign
from cspflow.config.schema import Machine

runner = CliRunner()


def output_of(result) -> str:
    """stdout + stderr, whichever the installed click keeps them in.

    click < 8.2 merged stderr into `result.output`; 8.2+ keeps them separate.
    The suite has to pass in both the base env (click 8.1) and the cspflow env
    (click 8.5), so tests never assume which stream an error landed in.
    """
    parts = [result.stdout or ""]
    try:
        parts.append(result.stderr or "")
    except (ValueError, AttributeError):
        pass
    return "".join(parts)

CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/a.vasp"]}}
"""


@pytest.fixture()
def campaign_file(tmp_path):
    p = tmp_path / "campaign.yaml"
    p.write_text(CAMPAIGN.format(workdir=tmp_path))
    return p


# --- report rendering ------------------------------------------------------


def test_report_fails_if_any_check_fails():
    r = D.Report([D.Check("a", "ok"), D.Check("b", "fail", "boom")])
    assert r.failed
    assert "FAIL" in r.render() and "boom" in r.render()


def test_report_passes_when_clean():
    r = D.Report([D.Check("a", "ok"), D.Check("b", "warn")])
    assert not r.failed
    assert "All checks passed" not in r.render()  # a warning is still reported


def test_all_clear_says_so():
    assert "All checks passed" in D.Report([D.Check("a", "ok")]).render()


# --- individual checks -----------------------------------------------------


def test_vasp_binary_missing_is_a_failure():
    m = Machine.model_validate({"codes": {"vasp_std": "/nope/vasp_std"}})
    assert D.check_vasp(m).status == "fail"


def test_no_vasp_configured_is_skipped():
    assert D.check_vasp(Machine()).status == "skip"


def test_vasp_binary_present(tmp_path):
    exe = tmp_path / "vasp_std"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    m = Machine.model_validate({"codes": {"vasp_std": str(exe)}})
    assert D.check_vasp(m).status == "ok"


def test_binary_without_exec_bit_is_a_failure(tmp_path):
    exe = tmp_path / "vasp_std"
    exe.write_text("x")
    exe.chmod(0o644)
    m = Machine.model_validate({"codes": {"vasp_std": str(exe)}})
    assert D.check_vasp(m).status == "fail"


def test_local_scheduler_needs_no_queue():
    assert D.check_scheduler(Machine(scheduler="local")).status == "ok"


def test_unwritable_workdir_is_a_failure(campaign_file, tmp_path):
    cfg = load_campaign(campaign_file, sets=["workdir=/nonexistent-root/x/y"])
    assert D.check_paths(cfg).status == "fail"


def test_writable_workdir_passes(campaign_file):
    assert D.check_paths(load_campaign(campaign_file)).status == "ok"


def test_potcar_layout_skipped_when_profile_defines_none():
    assert D.check_potcar_layout(Machine()).status == "skip"


TREE = Path("/projects/mmi/cspflow-shared/potcars/pmg")
has_tree = pytest.mark.skipif(not TREE.is_dir(), reason="POTCAR symlink tree not created")


@has_tree
def test_4f_check_is_not_a_vacuous_pass(campaign_file):
    """One rare earth cannot disagree with itself, and zero certainly cannot.
    Reporting 'ok' there would be worse than reporting nothing."""
    cfg = load_campaign(campaign_file, machine="orion")
    by_name = {c.name.split(" (")[0]: c for c in D.check_potcars(cfg, ["Fe"])}
    assert by_name["4f convention"].status == "skip"


@has_tree
def test_4f_check_passes_on_a_consistent_series(campaign_file):
    cfg = load_campaign(campaign_file, machine="orion")
    by_name = {c.name.split(" (")[0]: c for c in D.check_potcars(cfg, ["Sm", "Gd", "Fe"])}
    assert by_name["4f convention"].status == "ok"
    assert by_name["POTCAR resolution"].status == "ok"


@has_tree
def test_encut_is_read_from_the_recipe_not_only_from_the_overrides(campaign_file):
    """Removing MPRelaxSet removed the thing that was silently supplying
    ENCUT=520, so doctor has to surface it -- but the shipped recipe sets it,
    and reading only `incar_overrides` made this warn for every correctly
    configured campaign. A guard that always fires teaches the user to ignore
    the one warning that matters.
    """
    cfg = load_campaign(campaign_file, machine="orion")
    by_name = {c.name: c for c in D.check_potcars(cfg, ["Sm", "Fe", "Ti"])}
    assert by_name["ENCUT"].status == "ok"
    assert "520" in by_name["ENCUT"].detail

    cfg2 = load_campaign(campaign_file, machine="orion",
                         sets=["dft.incar_overrides.ENCUT=700"])
    by_name2 = {c.name: c for c in D.check_potcars(cfg2, ["Sm", "Fe", "Ti"])}
    assert by_name2["ENCUT"].status == "ok"
    assert "700" in by_name2["ENCUT"].detail


def test_an_encut_below_the_potcars_still_warns(campaign_file):
    """The case the check exists for: a cutoff lower than max(ENMAX) means the
    basis is not converged for this composition set."""
    cfg = load_campaign(campaign_file, machine="orion",
                        sets=["dft.incar_overrides.ENCUT=100"])
    by_name = {c.name: c for c in D.check_potcars(cfg, ["Sm", "Fe", "Ti"])}
    assert by_name["ENCUT"].status == "warn"
    assert "below max(ENMAX)" in by_name["ENCUT"].detail


@has_tree
def test_missing_potcar_root_is_a_clean_failure(campaign_file):
    cfg = load_campaign(campaign_file, machine="local")
    checks = D.check_potcars(cfg, ["Fe"])
    assert checks[0].status == "fail" and "potcar_root" in checks[0].detail


# --- CLI -------------------------------------------------------------------


def test_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0 and "cspflow" in output_of(res)


def test_init_writes_a_valid_campaign(tmp_path):
    out = tmp_path / "c.yaml"
    res = runner.invoke(app, ["init", "1", "demo", "-o", str(out), "-m", "local"])
    assert res.exit_code == 0 and out.is_file()
    cfg = load_campaign(out, sets=[f"workdir={tmp_path}"])
    assert cfg.campaign.name == "demo"


def test_init_writes_the_knobs_alongside(tmp_path):
    """Every setting is live, with its alternatives beside it on the same line."""
    out = tmp_path / "c.yaml"
    res = runner.invoke(app, ["init", "1", "demo", "-o", str(out), "-m", "local"])
    assert res.exit_code == 0
    text = out.read_text()
    assert "\n  max_per_composition: 5" in text      # live, even though it is the default
    assert "# recompute | mp_energies" in text       # the alternatives, beside it
    assert "# REFERENCE" in text                     # findable by funnel stage
    load_campaign(out, sets=[f"workdir={tmp_path}"])


def test_init_minimal_drops_the_annotations(tmp_path):
    out = tmp_path / "c.yaml"
    res = runner.invoke(app, ["init", "1", "demo", "-o", str(out), "-m", "local", "--minimal"])
    assert res.exit_code == 0
    assert "# REFERENCE" not in out.read_text()
    assert "|" not in out.read_text(), "trailing option comments are stripped too"
    load_campaign(out, sets=[f"workdir={tmp_path}"])


def test_init_refuses_to_clobber(tmp_path):
    out = tmp_path / "c.yaml"
    out.write_text("existing")
    res = runner.invoke(app, ["init", "1", "demo", "-o", str(out), "-m", "local"])
    assert res.exit_code == 1 and out.read_text() == "existing"
    ok = runner.invoke(app, ["init", "1", "demo", "-o", str(out), "-m", "local", "--force"])
    assert ok.exit_code == 0


def test_config_show(campaign_file):
    res = runner.invoke(app, ["config", "show", "-c", str(campaign_file)])
    assert res.exit_code == 0
    out = output_of(res)
    assert "config_hash" in out and "name: t" in out


def test_config_show_origins(campaign_file):
    res = runner.invoke(app, ["config", "show", "-c", str(campaign_file), "--origins"])
    assert res.exit_code == 0 and str(campaign_file) in output_of(res)


def test_config_show_json(campaign_file):
    import json

    res = runner.invoke(app, ["config", "show", "-c", str(campaign_file), "--json"])
    assert res.exit_code == 0
    out = output_of(res)
    payload = json.loads(out[: out.rindex("}") + 1])
    assert payload["name"] == "t"


def test_config_defaults_lists_schema_defaults():
    res = runner.invoke(app, ["config", "defaults"])
    assert res.exit_code == 0 and "e_above_hull_max" in output_of(res)


def test_set_flows_through_the_cli(campaign_file):
    res = runner.invoke(app, ["config", "show", "-c", str(campaign_file),
                              "-s", "filter.e_above_hull_max=0.33"])
    assert res.exit_code == 0 and "0.33" in output_of(res)


def test_bad_config_exits_nonzero_with_a_message(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: t\nmachine: local\nworkdir: /tmp\nsource: []\n")
    res = runner.invoke(app, ["config", "show", "-c", str(bad)])
    assert res.exit_code == 1


def test_status_without_a_database_is_a_clean_error(campaign_file):
    res = runner.invoke(app, ["status", "-c", str(campaign_file)])
    assert res.exit_code == 1
    assert "no campaign database" in output_of(res)


def test_doctor_exit_code_reflects_failure(campaign_file):
    """doctor must be usable as a gate in a submission script."""
    res = runner.invoke(app, ["doctor", "-c", str(campaign_file),
                              "-s", "workdir=/nonexistent-root/x"])
    assert res.exit_code == 1


# --------------------------------------------------------------------------
# `csp source`
# --------------------------------------------------------------------------


SWEEP_CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: chemical_space
    name: sweep
    chemical_space:
      groups:
        R: {{elements: [Sm], pick: 1}}
        T: {{elements: [Fe, Co], pick: 1, min_fraction: 0.8}}
      max_atoms_formula: 12
    defaults:
      max_atoms: 24
generate:
  engine: mattergen
  mattergen: {{model: /tmp/model}}
"""


@pytest.fixture
def sweep_campaign(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(SWEEP_CAMPAIGN.format(workdir=tmp_path))
    return path


class TestSourceCommand:
    def test_dry_run_writes_nothing(self, sweep_campaign, tmp_path):
        result = runner.invoke(app, ["source", "-c", str(sweep_campaign), "--dry-run"])
        assert result.exit_code == 0, output_of(result)
        assert "nothing written" in output_of(result)
        assert not (tmp_path / "campaign.db").exists()

    def test_dry_run_reports_the_work_before_it_is_done(self, sweep_campaign):
        result = runner.invoke(
            app, ["source", "-c", str(sweep_campaign), "--dry-run", "--gpu-seconds", "3"]
        )
        text = output_of(result)
        assert "compositions" in text
        assert "chemical systems" in text
        assert "implied GPU time" in text

    def test_commit_writes_rows(self, sweep_campaign, tmp_path):
        db = tmp_path / "out.db"
        result = runner.invoke(app, ["source", "-c", str(sweep_campaign), "--db", str(db)])
        assert result.exit_code == 0, output_of(result)
        assert db.is_file()
        from cspflow.db.store import Store

        with Store.open(db) as store:
            assert store.compositions()
            assert store.chemsystems() == ["Co-Sm", "Fe-Sm"]

    def test_limit_caps_the_commit_not_the_report(self, sweep_campaign, tmp_path):
        db = tmp_path / "out.db"
        result = runner.invoke(
            app, ["source", "-c", str(sweep_campaign), "--db", str(db), "--limit", "3"]
        )
        assert result.exit_code == 0, output_of(result)
        from cspflow.db.store import Store

        with Store.open(db) as store:
            assert len(store.compositions()) == 3
        assert "wrote 3 composition rows" in output_of(result)

    def test_commit_records_provenance(self, sweep_campaign, tmp_path):
        db = tmp_path / "out.db"
        runner.invoke(app, ["source", "-c", str(sweep_campaign), "--db", str(db)])
        from cspflow.db.store import Store

        with Store.open(db) as store:
            rows = store.sql.execute("SELECT config_hash FROM provenance").fetchall()
        assert rows and rows[0]["config_hash"]

    def test_bad_source_exits_non_zero_with_the_reason(self, tmp_path):
        path = tmp_path / "campaign.yaml"
        path.write_text(
            "name: t\nmachine: local\nworkdir: %s\n"
            "source:\n  - mode: composition_list\n    name: l\n"
            "    composition_list: {items: [{formula: Xx2O3}]}\n"
            "generate: {engine: mattergen, mattergen: {model: /tmp/m}}\n" % tmp_path
        )
        result = runner.invoke(app, ["source", "-c", str(path), "--dry-run"])
        assert result.exit_code == 1
        assert "not an element" in output_of(result)
