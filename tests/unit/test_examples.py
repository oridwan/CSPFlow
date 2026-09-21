"""The shipped examples must load, expand, and mean what they claim.

An example that stops matching its own README is worse than no example: it is
read as documentation and believed. These tests run the same code path
`csp source --dry-run` runs, so a schema change that invalidates an example
fails here rather than in a user's first campaign.
"""

import os
import re
from pathlib import Path

import pytest

from cspflow.config.loader import load_campaign
from cspflow.source import expand_all

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load(name: str):
    """Load an example the way the CLI does, with $USER guaranteed."""
    path = EXAMPLES / name / "campaign.yaml"
    env = dict(os.environ)
    env.setdefault("USER", "tester")
    cfg = load_campaign(path, env=env)
    return cfg, expand_all(cfg.campaign, cfg.base_dir)


def test_all_three_modes_have_an_example():
    assert {p.name for p in EXAMPLES.iterdir() if p.is_dir()} == {
        "1-chemical-space", "2-composition-list", "3-structure-list"}


@pytest.mark.parametrize("name", ["1-chemical-space", "2-composition-list",
                                  "3-structure-list"])
def test_example_loads_and_expands(name):
    cfg, plan = _load(name)
    assert cfg.campaign.name.startswith("example-")
    assert plan.render()          # the same summary --dry-run prints


def test_chemical_space_example_sweeps_what_it_says():
    """The header claims 36 systems and ~165k structures. Hold it to that."""
    _, plan = _load("1-chemical-space")
    assert len(plan.compositions) == 3384
    assert len(plan.chemsystems()) == 36
    assert plan.n_target_total == 165_456


def test_composition_list_example_reads_both_the_items_and_the_csv():
    """34 = 4 inline + 13 CSV rows, expanded over Z, with one collision."""
    _, plan = _load("2-composition-list")
    assert len(plan.compositions) == 34
    formulas = {c.formula for c in plan.compositions}   # canonical, reduced
    assert "Fe29Sm3Ti2" in formulas          # Sm3Fe29Ti2, inline only
    assert "Fe17Sm2" in formulas             # Sm2Fe17, CSV only
    assert {"Fe1", "Sm1", "Ti1"} <= formulas, "the CSV's hull anchors"
    warnings = [w for r in plan.results for w in r.warnings]
    assert any("SmFe11Ti" in w and "appears twice" in w for w in warnings), \
        "the example deliberately overrides one CSV row from the items block"


def test_structure_list_example_reads_the_real_seeds():
    """Five POSCARs, parsed by both readers, entering the funnel at screen."""
    cfg, plan = _load("3-structure-list")
    assert cfg.campaign.generate is None, "a seeds-only campaign has no generate block"
    assert len(plan.structures) == 5
    assert plan.compositions == []
    assert all(r.entry_stage == "screen" for r in plan.results)
    seeds = EXAMPLES / "3-structure-list" / "inputs" / "seeds"
    assert len(list(seeds.glob("*.vasp"))) == 5


def test_the_biggest_seed_clears_the_max_atoms_gate():
    """Sm2Fe17 is 57 atoms; an example whose own seed is refused is a trap."""
    cfg, plan = _load("3-structure-list")
    cap = cfg.campaign.source[0].structure_list.max_atoms
    assert cap >= 57
    assert max(s.n_atoms for s in plan.structures) == 57
    assert len(plan.structures) == 5


# --- the examples are the `csp init` templates ------------------------------

EXAMPLE_NAMES = ["1-chemical-space", "2-composition-list", "3-structure-list"]


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_example_machine_and_recipe_are_what_init_writes(name):
    """The copies are generated, not hand-edited, so they cannot drift from the
    shipped profile and recipe that `csp init` copies."""
    import yaml
    from cspflow import templates
    from cspflow.config.loader import resolve_machine_path
    from cspflow.dft.recipe import RECIPE_DIR

    folder = EXAMPLES / name
    campaign = yaml.safe_load((folder / "campaign.yaml").read_text())
    assert campaign["machine"] == "machine.yaml"
    assert campaign["dft"]["recipe"] == "recipe.yaml"
    assert (folder / "machine.yaml").read_text() == templates.machine_copy(
        resolve_machine_path("orion"), name=campaign["name"])
    assert (folder / "recipe.yaml").read_text() == templates.recipe_copy(
        RECIPE_DIR / "magnets.yaml", name=campaign["name"])


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_no_knob_wired_to_nothing(name):
    """D137, D146: a setting no code reads looks exactly like one that matters.
    These were in the examples and read by nothing (calibrate left the funnel in
    D126; archive was never written; analyze.* has no consumer)."""
    import yaml

    doc = yaml.safe_load((EXAMPLES / name / "campaign.yaml").read_text())
    assert not {"calibrate", "archive", "analyze"} & set(doc)
    assert set(doc["reference"]) == {"thermo_type", "energy_scale", "mode", "energy_source"}
    assert "e_above_hull_max_source" not in doc["filter"]
    assert "max_per_composition" not in doc["dft"]["select"], "read only under filter:"
    if name == "3-structure-list":
        assert "generate" not in doc and "defaults" not in doc["source"][0], \
            "z and structure counts do not apply to seeds"


# `key: value   # a | b | c` -- a pick-one list of bare values beside a setting.
_OPTIONS = re.compile(
    r"^(?P<head>[ \t]*(?:- )?(?P<key>\w+): )(?P<value>\S+)(?P<gap>\s{2,})"
    r"# (?P<opts>[^\s|]+(?: \| [^\s|]+)+)(?=\s|$)(?P<tail>.*)$", re.M)


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_every_listed_alternative_is_accepted_by_the_schema(name):
    """`magnetic_order: ferri  # ferri | ferro | antiferro` was in an example, and
    `antiferro` is rejected. A listed option must be one the schema takes.

    Swapping one value can break a CROSS-check (fixed needs `count:`); that is
    allowed. What is not allowed is an error on the setting itself."""
    import yaml
    from pydantic import ValidationError
    from cspflow.config.schema import Campaign

    text = (EXAMPLES / name / "campaign.yaml").read_text()
    matches = list(_OPTIONS.finditer(text))
    assert len(matches) >= 15, "the option-comment format changed; this test went blind"
    for m in matches:
        for alt in m.group("opts").split(" | "):
            line = f"{m.group('head')}{alt}{m.group('gap')}# {m.group('opts')}{m.group('tail')}"
            doc = yaml.safe_load(text[:m.start()] + line + text[m.end():])
            try:
                Campaign.model_validate(doc)
            except ValidationError as exc:
                own = [e for e in exc.errors() if e["loc"] and e["loc"][-1] == m.group("key")]
                assert not own, f"{name}: {m.group('key')}: {alt} is listed but rejected: {own}"
