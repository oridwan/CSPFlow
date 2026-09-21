"""The campaign folder: what `csp init` writes, and how it resolves.

A campaign is a folder -- campaign.yaml plus editable copies of the machine
profile and the DFT recipe -- made from one of three example campaigns by
`csp init <TYPE> <NAME>`. The things worth testing are that the folder is
complete for each type, that init and the example cannot drift, and that a
relative path inside it means "beside the campaign file" from anywhere you
might run the command.
"""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from cspflow.cli import _find_campaign, app
from cspflow.config.loader import ConfigError, load_campaign, resolve_machine_path
from cspflow.dft.recipe import load_recipe
from cspflow.templates import EXAMPLES_DIR, campaign_yaml

runner = CliRunner()

WORKSPACE = {"campaign.yaml", "machine.yaml", "recipe.yaml",
             "README.md", "inputs/README.md"}


def _init(tmp_path: Path, *args: str, kind: str = "1"):
    result = runner.invoke(app, ["init", kind, "demo", "-d", str(tmp_path / "demo"),
                                 "-m", "local", "--no-recompute-reference", *args])
    assert result.exit_code == 0, result.output
    return tmp_path / "demo"


# --- what init writes ------------------------------------------------------

def test_init_writes_a_folder_with_every_knob_in_it(tmp_path):
    root = _init(tmp_path)
    written = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert written == WORKSPACE


def test_the_copies_are_real_copies_not_stubs(tmp_path):
    """The point of copying is that the knobs are *there* to be edited."""
    root = _init(tmp_path)
    machine = yaml.safe_load((root / "machine.yaml").read_text())
    recipe = yaml.safe_load((root / "recipe.yaml").read_text())
    assert machine["scheduler"] == "local"
    assert recipe["stages"], "a recipe with no stages is not editable"


def test_minimal_writes_only_the_campaign_file(tmp_path):
    root = _init(tmp_path, "--minimal")
    written = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert written == {"campaign.yaml"}
    assert yaml.safe_load((root / "campaign.yaml").read_text())["machine"] == "local"


def test_existing_files_are_not_clobbered(tmp_path):
    root = _init(tmp_path)
    (root / "campaign.yaml").write_text("name: mine\n")
    result = runner.invoke(app, ["init", "1", "demo", "-d", str(root), "-m", "local"])
    assert result.exit_code == 1
    assert (root / "campaign.yaml").read_text() == "name: mine\n"


# --- the three types ------------------------------------------------------

def test_a_new_campaign_carries_no_example_chemistry(tmp_path):
    """D153: a new user must not run the example's Sm-Fe chemistry by accident.
    No demo inputs are copied, and the yaml's elements / formulas are blank."""
    one = yaml.safe_load(campaign_yaml("1", name="demo"))
    groups = one["source"][0]["chemical_space"]["groups"]
    assert groups and all(g["elements"] == [] for g in groups.values())

    two = _init(tmp_path / "a", kind="2")
    assert yaml.safe_load((two / "campaign.yaml").read_text())[
        "source"][0]["composition_list"]["items"] == []
    csv = (two / "inputs" / "compositions.csv").read_text()
    data = [l for l in csv.splitlines() if l.strip() and not l.startswith("#")]
    assert data == ["formula,z_min,z_max,n_structures"], "header only, no formulas"

    three = _init(tmp_path / "b", kind="3")
    assert (three / "inputs" / "seeds").is_dir()
    assert list((three / "inputs" / "seeds").iterdir()) == []


@pytest.mark.parametrize("kind,says", [
    ("1", "have no elements yet"),
    ("2", "produced no items"),
    ("3", "matched no files"),
])
def test_an_unfilled_campaign_loads_but_refuses_at_source(tmp_path, kind, says):
    """It must LOAD, so `csp doctor` can check the cluster before the chemistry
    is filled in -- and then `csp source` must say what is missing."""
    from cspflow.source import SourceError, expand_all

    root = _init(tmp_path, kind=kind)
    cfg = load_campaign(root / "campaign.yaml")
    with pytest.raises(SourceError, match=says):
        expand_all(cfg.campaign, cfg.base_dir)


@pytest.mark.parametrize("kind,mode", [("1", "chemical_space"), ("2", "composition_list"),
                                       ("3", "structure_list"), ("seeds", "structure_list")])
def test_each_type_scaffolds_its_own_source_mode(tmp_path, kind, mode):
    root = _init(tmp_path, kind=kind)
    cfg = load_campaign(root / "campaign.yaml")
    assert cfg.campaign.name == "demo"
    assert [s.mode.value for s in cfg.campaign.source] == [mode]
    assert (cfg.campaign.generate is None) == (mode == "structure_list")


def test_init_is_the_example_with_only_four_lines_changed(tmp_path):
    """The example IS the template. Apart from its chemistry (blanked, tested
    above), anything but name, machine, recipe and the reference answer
    differing means the two have started to drift."""
    for kind, folder in [("1", "1-chemical-space"), ("2", "2-composition-list"),
                         ("3", "3-structure-list")]:
        made = yaml.safe_load(campaign_yaml(kind, name="demo", reference_mode="mp_energies",
                                            keep_example_chemistry=True))
        example = yaml.safe_load((EXAMPLES_DIR / folder / "campaign.yaml").read_text())
        assert made.pop("name") == "demo"
        example.pop("name")
        assert made["reference"].pop("mode") == "mp_energies"
        example["reference"].pop("mode")
        assert made == example, folder


@pytest.mark.parametrize("kind", ["1", "2", "3"])
def test_minimal_is_the_annotated_file_with_the_comments_removed(kind):
    """Same keys, both parse. `--minimal` is a view, not a second template."""
    full = yaml.safe_load(campaign_yaml(kind, name="demo"))
    terse = campaign_yaml(kind, name="demo", minimal=True)
    assert full == yaml.safe_load(terse)
    body = [line for line in terse.splitlines() if not line.startswith("#")]
    assert not any("#" in line for line in body), "a trailing comment survived"


def test_one_argument_is_a_name_and_the_type_is_asked_for(tmp_path):
    """`csp init my-campaign`, the old form: no terminal to ask on -> the menu."""
    result = runner.invoke(app, ["init", "demo", "-d", str(tmp_path / "demo")])
    assert result.exit_code == 1
    for line in ("chemical_space", "composition_list", "structure_list", "csp init <1|2|3> demo"):
        assert line in result.output
    assert not (tmp_path / "demo").exists(), "nothing is written before the type is known"


@pytest.mark.parametrize("argv,expect", [
    (["init"], "csp init <1|2|3> <name>"),
    (["init", "2"], "csp init 2 <name>"),
    (["init", "7", "demo"], "unknown campaign type '7'"),
])
def test_an_incomplete_init_prints_the_menu(argv, expect):
    result = runner.invoke(app, argv)
    assert result.exit_code == 1
    assert expect in result.output and "structure_list" in result.output


# --- resolution ------------------------------------------------------------

def test_machine_and_recipe_resolve_beside_the_campaign_file(tmp_path, monkeypatch):
    root = _init(tmp_path)
    monkeypatch.chdir(tmp_path)             # deliberately *not* in the folder
    cfg = load_campaign(root / "campaign.yaml")
    assert cfg.machine_path == root / "machine.yaml"
    assert cfg.base_dir == root
    assert load_recipe(cfg.campaign.dft.recipe, cfg.base_dir).stages


def test_a_relative_machine_path_reports_where_it_looked(tmp_path):
    with pytest.raises(ConfigError) as exc:
        resolve_machine_path("nowhere.yaml", tmp_path)
    assert str(tmp_path) in str(exc.value)


def test_a_shipped_name_still_wins_over_the_folder(tmp_path):
    """`machine: local` must keep meaning the shipped profile."""
    assert resolve_machine_path("local", tmp_path).name == "local.yaml"


# --- finding the campaign from anywhere ------------------------------------

def test_commands_walk_up_to_the_campaign(tmp_path, monkeypatch):
    root = _init(tmp_path)
    deep = root / "inputs" / "seeds"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert _find_campaign(Path("campaign.yaml")) == root / "campaign.yaml"


def test_a_named_file_is_never_hunted_for(tmp_path, monkeypatch):
    """If the user names a file, a missing one is an error, not a search."""
    _init(tmp_path)
    monkeypatch.chdir(tmp_path / "demo" / "inputs")
    assert _find_campaign(Path("other.yaml")) == Path("other.yaml")
