"""docs/06-settings.md claims a default for every key. This checks they are true.

WHY THIS EXISTS
    A reference doc is worse than no reference doc when it is wrong: it is
    consulted precisely when someone does not already know the answer, so a
    stale default is believed rather than caught. And nothing about editing
    `schema.py` reminds you that a table in `docs/` quoted the old value.

    `filter.e_above_hull_max` is the example that matters. It is documented as
    "the single most consequential knob in the file"; if the doc says 0.10 and
    the schema says something else, a campaign gets sized on a number the user
    never chose.

WHAT IT DOES NOT DO
    It does not parse the markdown. Parsing prose to check prose invites a test
    that fails on formatting and passes on being wrong. Instead the claims are
    listed here explicitly: changing a default means changing this list, and
    changing this list is the reminder to change the doc.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cspflow.config import schema as S

DOC = Path(__file__).resolve().parents[2] / "docs" / "06-settings.md"

#: (model, field, what docs/06-settings.md tells the reader it defaults to)
CLAIMED = [
    (S.Campaign, "archive", None),
    (S.Source, "name", "default"),
    (S.SourceDefaults, "max_atoms", 40),
    (S.SourceDefaults, "n_structures_scope", "per_z"),
    (S.ZRange, "min", 1),
    (S.ZRange, "max", 1),
    (S.NStructures, "structures_per_atom", 2.0),
    (S.NStructures, "count", None),
    (S.ChemicalSpace, "max_atoms_formula", 20),
    (S.ChemicalSpace, "max_rare_earth", 1),
    (S.StructureList, "relax", True),
    (S.StructureList, "dedup", "warn"),
    (S.Generate, "engine", "mattergen"),
    (S.MatterGen, "mode", "csp"),
    (S.MatterGen, "max_batch_size", 100),
    (S.MatterGen, "timeout_per_batch", 1800),
    (S.Resources, "role", "cpu"),
    (S.Resources, "time", "24:00:00"),
    (S.Screen, "mlip", "mattersim"),
    (S.MatterSim, "model", "MatterSim-v1.0.0-5M.pth"),
    (S.MatterSim, "fmax", 0.01),
    (S.MatterSim, "max_steps", 500),
    (S.StructureMatcherCfg, "ltol", 0.2),
    (S.StructureMatcherCfg, "stol", 0.2),
    (S.StructureMatcherCfg, "angle_tol", 5.0),
    (S.Reference, "energy_scale", "raw"),
    (S.Reference, "mode", "recompute"),
    (S.Reference, "energy_source", "dft"),
    (S.Reference, "prescreen_mode", "mp_energies"),
    (S.Reference, "prescreen_hull_max", 0.20),
    (S.Reference, "snapshot", True),
    (S.Reference, "snapshot_id", "auto"),
    (S.Reference, "relax_with_mlip", True),
    (S.Filter, "e_above_hull_max", 0.10),
    (S.Filter, "e_above_hull_max_source", "calibrated"),
    (S.Filter, "max_per_composition", 5),
    (S.SpacegroupFilter, "min_number", 1),
    (S.Dft, "recipe", "magnets"),
    (S.Dft, "layout", "runs"),
    (S.Dft, "combined_job", True),
    (S.Dft, "nbands", "auto"),
    (S.Dft, "max_in_flight", 200),
    (S.Dft, "max_concurrent_tasks", 48),
    (S.PotcarCfg, "tree", "VASP6.4"),
    (S.PotcarCfg, "functional", "PBE_64"),
    (S.RareEarth, "magnetic_order", "ferri"),
    (S.RareEarth, "reconstruct_ms", True),
    (S.Magnetism, "mode", "ferrimagnetic_retm"),
    (S.Magnetism, "strict", True),
    (S.Ldau, "enabled", False),
    (S.Ldau, "ldau_type", 2),
    (S.Select, "rank_by", "e_above_hull_mlip"),
    (S.Select, "max_per_composition", 3),
    (S.Select, "max_total", 1500),
    (S.Dft, "max_cores", None),
    (S.Analyze, "report", "html"),
    (S.CalibrateMP, "on_fail", "warn"),
    (S.CalibratePilot, "on_fail", "off"),
]


@pytest.mark.parametrize("model,field,claimed", CLAIMED,
                         ids=[f"{m.__name__}.{f}" for m, f, _ in CLAIMED])
def test_the_documented_default_is_the_real_one(model, field, claimed):
    actual = model.model_fields[field].default
    actual = getattr(actual, "value", actual)          # unwrap Enum members
    assert actual == claimed, (
        f"docs/06-settings.md says {model.__name__}.{field} defaults to "
        f"{claimed!r}, but the schema says {actual!r}. Update the doc AND the "
        f"CLAIMED list above."
    )


def test_the_doc_covers_every_block_of_the_campaign_model():
    """A new top-level block must not be added without documenting it."""
    text = DOC.read_text()
    for name in S.Campaign.model_fields:
        assert re.search(rf"[`#] ?`?{re.escape(name)}\b", text), (
            f"`{name}` is a top-level key in campaign.yaml and does not appear "
            f"in docs/06-settings.md"
        )


def test_calibrate_is_not_described_as_a_running_stage():
    """It was removed from the funnel; `--through calibrate` is an error now.

    The docs told users to run that command for weeks after it stopped working.
    """
    from cspflow.driver import STAGE_ORDER

    assert "calibrate" not in STAGE_ORDER
    for doc in ("07-stages.md", "12-cli.md", "02-quickstart.md", "04-workflows.md"):
        text = (DOC.parent / doc).read_text()
        assert "--through calibrate" not in text or "error" in text.lower(), (
            f"docs/{doc} still tells the reader to run `--through calibrate` "
            f"without saying it fails"
        )


def test_batch_size_is_gone_but_an_old_campaign_still_loads():
    """`screen.mattersim.batch_size` was retired (D137).

    Two halves, and the second is the one that matters. `Base` sets
    `extra="forbid"`, so simply deleting the field would make every existing
    campaign.yaml that sets it fail to load — refusing to open a campaign over a
    setting that never did anything. It is dropped with a warning instead.
    """
    import warnings

    from cspflow.config.schema import MatterSim

    assert "batch_size" not in MatterSim.model_fields, (
        "batch_size is back; MLIP relaxation is still one structure at a time"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        block = MatterSim(model="x", fmax=0.05, max_steps=200, batch_size=16)
    assert not hasattr(block, "batch_size")
    assert block.max_steps == 200, "the rest of the block must survive"
    assert any("batch_size" in str(w.message) for w in caught), (
        "an ignored setting must say so"
    )


@pytest.mark.parametrize("key", ["batch_size", "budget_core_hours"])
def test_no_shipped_campaign_still_sets_a_retired_key(key):
    """The warning is for other people's files, not for ours.

    Every campaign under `campaigns/` and `examples/` is either live or a
    worked example someone will copy, so a retired key sitting in one teaches
    the setting to the next person who reads it.
    """
    import re

    import os

    root = DOC.parent.parent
    bare = re.compile(rf"^[ \t]*{re.escape(key)}:", re.M)
    offenders = []
    for folder in ("campaigns", "examples"):
        # os.walk, not rglob: Python 3.10's rglob FOLLOWS directory symlinks,
        # and a campaign's `results -> /scratch/...` link sent this test through
        # a live campaign's 5,000 composition folders over NFS (25 s). Seed
        # folders are skipped for the same reason: 14,777 structure files.
        for here, dirs, files in os.walk(root / folder, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {"inputs", "results", "seeds"}
                       and not os.path.islink(os.path.join(here, d))]
            if "campaign.yaml" in files:
                path = Path(here) / "campaign.yaml"
                if bare.search(path.read_text()):
                    offenders.append(str(path.relative_to(root)))
    assert not offenders, f"still setting a retired key: {offenders}"


def test_the_budget_is_gone_but_the_core_cap_is_documented():
    """The replacement has to be findable, or removing the budget just leaves a
    hole where the throttle used to be explained (D143)."""
    text = DOC.read_text()
    assert "budget_core_hours" not in text or "retired" in text.lower(), (
        "docs/06-settings.md still documents budget_core_hours as a live setting"
    )
    assert "max_cores" in text, "docs/06-settings.md does not document dft.max_cores"
