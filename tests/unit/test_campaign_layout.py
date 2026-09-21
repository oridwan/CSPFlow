"""A campaign is ONE folder.

Before 2026-09-12 a campaign occupied three configured locations:

    campaigns/CeFeB/              840 KB   campaign.yaml, inputs/, notes/, logs
    /scratch/.../cspflow/CeFeB/   6.2 GB   campaign.db, screen/, dft/
    .../cspflow_archive/CeFeB     absent   configured, never created

A `results` symlink bridged the first two, and the split still leaked: the
database sat on scratch while the config sat elsewhere, and the third location
existed only in the template.

`workdir` now defaults to the relative path `results`, which resolves against
the campaign folder -- so everything a campaign produces lives inside it. An
absolute `workdir` still works, for output that genuinely belongs elsewhere.
"""

from pathlib import Path

import pytest
import yaml

from cspflow.config.loader import load_campaign
from cspflow.templates import campaign_yaml


def write(tmp_path, **overrides):
    doc = {
        "name": "t",
        "machine": "local",
        "source": [{"mode": "structure_list", "structure_list": {"paths": ["inputs"]}}],
        "dft": {"recipe": "magnets"},
    }
    doc.update(overrides)
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(doc))
    return load_campaign(path)


def test_a_relative_workdir_lands_inside_the_campaign_folder(tmp_path):
    cfg = write(tmp_path, workdir="results")
    assert cfg.work_dir == (tmp_path / "results").resolve()
    assert cfg.campaign_db == (tmp_path / "results" / "campaign.db").resolve()


def test_it_does_not_resolve_against_the_current_directory(tmp_path, monkeypatch):
    """The whole point: `csp status` from elsewhere must find the SAME campaign
    as `csp run`, not an empty one beside the shell's cwd."""
    cfg = write(tmp_path, workdir="results")
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert cfg.work_dir == (tmp_path / "results").resolve()


def test_an_absolute_workdir_is_left_alone(tmp_path):
    """Output that genuinely belongs on another filesystem still can."""
    other = tmp_path / "scratch" / "t"
    cfg = write(tmp_path, workdir=str(other))
    assert cfg.work_dir == other
    assert cfg.campaign_db == other / "campaign.db"


def test_the_shipped_template_uses_one_place(tmp_path):
    doc = yaml.safe_load(campaign_yaml(1, name="demo"))
    assert doc["workdir"] == "results"
    # The third location is gone from the template entirely.
    assert "archive" not in doc


def test_the_results_symlink_is_not_made_when_it_would_point_at_itself(tmp_path):
    """With the default layout `results/` IS the workdir, so linking it to
    itself is nonsense -- and a symlink loop is worse than no symlink."""
    from cspflow.cli import _link_results

    cfg = write(tmp_path, workdir="results")
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    _link_results(tmp_path / "campaign.yaml", cfg.work_dir)

    results = tmp_path / "results"
    assert results.is_dir()
    assert not results.is_symlink()


def test_an_absolute_workdir_still_gets_the_symlink(tmp_path):
    from cspflow.cli import _link_results

    other = tmp_path / "elsewhere"
    other.mkdir()
    cfg = write(tmp_path, workdir=str(other))
    _link_results(tmp_path / "campaign.yaml", cfg.work_dir)

    link = tmp_path / "results"
    assert link.is_symlink() and link.resolve() == other.resolve()
