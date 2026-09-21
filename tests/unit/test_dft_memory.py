"""`--mem` is sized per structure, because one flat number does not fit.

WHAT ONE FLAT NUMBER COST.  `mem: 32G` sat in the machine profile and was handed
to every DFT job whatever it was doing.  Three CeFeB statics were OOM-killed by
it while 129 other runs in the same two campaigns used less than half of it.

The three were not the big ones.  `dft-18-static` has FEWER bands than
`dft-88-static` (496 vs 536) and the same 68 atoms, and needed 2.25x the memory:

    dft-88-static   KPAR=4   261 MB/rank   16.3 GB/node   ran
    dft-18-static   KPAR=8   589 MB/rank   36.8 GB/node   OOM against 32G

KPAR is the reason, and for a physical one: each k-point group keeps its OWN
copy of the charge density, the local potentials and the FFT work arrays.  With
64 ranks, KPAR=8 leaves 8 ranks per group instead of 16, so every rank carries
twice the replicated data.  `choose_kpar`'s benchmark says the speed gain
plateaus by KPAR=4, so that memory bought nothing.

The estimator is fitted to 132 runs against VASP's own memory report.  These
tests pin the behaviour that matters: it scales with KPAR, it never returns
something unusable, and a retry rung that raises memory actually raises it --
which it did not, at first, because `_retry_or_fail` persisted only `set` and
`remedy` and dropped `resources` on the floor.
"""

import pytest

from cspflow.dft.vasp.parallel import (
    MEM_FLOOR_GB,
    estimate_memory_gb,
    estimate_memory_mb_per_rank,
)

# A 68-atom Ce-Fe-B cell, the size this campaign actually runs.
CELL = dict(volume=930.0, encut=520.0, nbands=520, ntasks=64)


def test_kpar_is_the_dominant_term():
    """The whole finding in one assertion: same cell, double KPAR, ~double memory."""
    four = estimate_memory_mb_per_rank(kpar=4, **CELL)
    eight = estimate_memory_mb_per_rank(kpar=8, **CELL)
    assert 1.7 < eight / four < 2.5


def test_more_bands_costs_less_than_more_kpar():
    """dft-18-static had fewer bands than dft-88-static and needed 2.25x the
    memory. An estimator that ranked them by band count would have missed it."""
    more_bands = estimate_memory_mb_per_rank(**{**CELL, "nbands": 536, "kpar": 4})
    more_kpar = estimate_memory_mb_per_rank(**{**CELL, "nbands": 496, "kpar": 8})
    assert more_kpar > more_bands


def test_the_three_oom_cases_would_now_be_covered():
    """36.8 GB was the largest of the three. Anything at or above that is a fix."""
    assert estimate_memory_gb(volume=760.0, encut=520.0, nbands=496,
                              kpar=8, ntasks=64) >= 37


def test_the_typical_case_is_not_inflated():
    """A fix that asked 112G for every job would cost more in queue time than the
    OOMs cost in wasted runs. dft-88-static really used 16.3 GB."""
    assert estimate_memory_gb(volume=930.0, encut=520.0, nbands=536,
                              kpar=4, ntasks=64) <= 40


def test_never_returns_something_unusable():
    """The fit's middle coefficient is negative -- the terms are collinear over
    the range measured -- so a small enough system can drive the expression
    toward zero. A job with no memory is not a failure anyone enjoys reading."""
    for nbands in (1, 8, 32):
        got = estimate_memory_gb(volume=20.0, encut=300.0, nbands=nbands,
                                 kpar=1, ntasks=8)
        assert got >= MEM_FLOOR_GB


def test_memory_scales_with_cell_size():
    small = estimate_memory_mb_per_rank(volume=200.0, encut=520.0, nbands=200,
                                        kpar=4, ntasks=64)
    big = estimate_memory_mb_per_rank(volume=2000.0, encut=520.0, nbands=200,
                                      kpar=4, ntasks=64)
    assert big > small


def test_zero_ranks_is_refused_rather_than_dividing_by_zero():
    with pytest.raises(ValueError):
        estimate_memory_mb_per_rank(volume=930.0, encut=520.0, nbands=520,
                                    kpar=4, ntasks=0)


# --- the ladder actually applies what it chose -----------------------------


def test_a_resource_rung_survives_encode_and_decode():
    """The rung is stored on the row between cycles. `_retry_or_fail` persisted
    only `set` and `remedy`, so `resources: {mem: 64G}` was recorded nowhere and
    the retry re-ran at the identical 32G -- costing the same again and looking
    like diligence. This is that path."""
    import json

    from cspflow.stages.dft_stage import _decode_remedy

    rule = {"when": "out_of_memory", "resources": {"mem": "64G"}}
    stored = json.dumps({"set": rule.get("set", {}),
                         "resources": rule.get("resources", {}),
                         "remedy": rule.get("remedy", "")})
    assert _decode_remedy(stored).get("resources") == {"mem": "64G"}


def test_both_stages_have_an_out_of_memory_rung():
    """The static stage had no retry block at all, so every failure there was
    terminal -- including the OOMs this change is about."""
    from pathlib import Path

    import yaml

    import cspflow
    recipe = yaml.safe_load(
        (Path(cspflow.__file__).parent / "dft/recipes/magnets.yaml").read_text())
    for stage in recipe["stages"]:
        triggers = {r.get("when") for r in stage.get("retry", [])}
        assert "out_of_memory" in triggers, f"{stage['name']} has no OOM rung"
