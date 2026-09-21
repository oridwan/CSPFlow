"""An array's allocation must suit EVERY task in it, not just the first.

`claim()` fills a batch from whatever is ready, so a `static` and a `relax`
routinely go out in one array -- 17 such arrays existed across CeFeB and CePdGe
when this was found.  `build()` read its resources from `items[0]`, and because
a finished relax immediately becomes a ready static, `items[0]` is very often
the static:

    dft-78-static.tasks.json -> [dft-78-static, dft-69-relax]
    #SBATCH --time=12:00:00                 <- the STATIC's walltime
    recipe: relax 24:00:00, static 12:00:00

`dft-69-relax` was running against a 12-hour cap on a recipe that gives a relax
24, and one CePdGe array had THIRTEEN relaxes under that cap.  Nothing about it
is loud: they run until the wall and come back TIMEOUT, having burned the full
12 hours first.

An array cannot give its tasks different walltimes.  Over-serving the short
tasks costs some queue priority; under-serving the long ones costs the run.
"""

from types import SimpleNamespace

import pytest

from cspflow.stages.dft_stage import _larger, _megabytes, _seconds


class _Item:
    def __init__(self, step, resources=None):
        self.payload = {"step": step, "resources": resources or {}}


class _Stage:
    def __init__(self, resources, name=""):
        self.resources = resources
        self.name = name


class _Recipe:
    def __init__(self, stages):
        self.stages = stages


def _stage_with(recipe_stages, combined=False):
    """A DftStage with only what `_resources_for` touches.

    `combined` is set explicitly rather than left to a default: these tests are
    about the ARRAY case, where several tasks run side by side and the
    allocation must fit the largest. A combined job consumes its steps in turn
    and sums the walltime instead -- a different question, tested in
    `test_combined_job.py`.
    """
    from pathlib import Path as _Path

    from cspflow.dft.layout import Layout
    from cspflow.stages.dft_stage import DftStage

    s = DftStage.__new__(DftStage)
    names = ["relax", "static", "extra"]
    s.recipe = _Recipe([_Stage(r, names[i] if i < len(names) else f"s{i}")
                        for i, r in enumerate(recipe_stages)])
    s._layout = Layout("runs" if combined else "stages", _Path("/w/dft"))
    s.cfg = SimpleNamespace(
        campaign=SimpleNamespace(dft=SimpleNamespace(
            layout=s._layout.name, combined_job=combined)),
        machine=SimpleNamespace(defaults=SimpleNamespace(ntasks=64, mem="32G")),
        work_dir=_Path("/w"),
    )
    return s


RELAX = {"role": "cpu", "ntasks": 64, "time": "24:00:00"}
STATIC = {"role": "cpu", "ntasks": 64, "time": "12:00:00"}


def test_a_mixed_batch_gets_the_longer_walltime():
    """The regression, in the exact order it occurred: the static first."""
    st = _stage_with([RELAX, STATIC])
    got = st._resources_for([_Item(1), _Item(0)])     # static, then relax
    assert got["time"] == "24:00:00"


def test_order_does_not_change_the_answer():
    st = _stage_with([RELAX, STATIC])
    a = st._resources_for([_Item(0), _Item(1)])
    b = st._resources_for([_Item(1), _Item(0)])
    assert a["time"] == b["time"] == "24:00:00"


def test_a_single_step_batch_is_unchanged():
    """A static-only array must still ask for 12 h, not be inflated to 24."""
    st = _stage_with([RELAX, STATIC])
    assert st._resources_for([_Item(1), _Item(1)])["time"] == "12:00:00"


def test_a_retry_rung_can_still_raise_beyond_the_recipe():
    st = _stage_with([RELAX, STATIC])
    got = st._resources_for([_Item(1), _Item(1, {"mem": "64G"})])
    assert got["mem"] == "64G"


def test_a_retry_rung_never_lowers_a_resource():
    """`resources: {mem: 32G}` on one task must not shrink a 64G sibling."""
    st = _stage_with([{**RELAX, "mem": "64G"}, STATIC])
    got = st._resources_for([_Item(0), _Item(0, {"mem": "32G"})])
    assert got["mem"] == "64G"


# --- the comparators, because string ordering is the trap ------------------


def test_walltime_is_compared_as_time_not_text():
    """'8:00:00' > '12:00:00' lexically. Text comparison picks the SMALLER."""
    assert _larger("time", "8:00:00", "12:00:00") == "12:00:00"
    assert _larger("time", "12:00:00", "8:00:00") == "12:00:00"


def test_day_prefixed_walltime():
    assert _seconds("1-00:00:00") == 86400
    assert _larger("time", "1-00:00:00", "24:00:00") == "1-00:00:00"


def test_memory_is_compared_by_size_not_text():
    assert _larger("mem", "512M", "1G") == "1G"
    assert _larger("mem", "32G", "64G") == "64G"
    assert _megabytes("1T") == pytest.approx(1024 * 1024)


def test_a_missing_side_is_simply_the_other():
    assert _larger("time", None, "12:00:00") == "12:00:00"
    assert _larger("time", "12:00:00", None) == "12:00:00"


def test_an_unparseable_value_does_not_raise():
    assert _larger("time", "nonsense", "12:00:00") in ("nonsense", "12:00:00")
    assert _megabytes("junk") == 0.0


def test_array_takes_the_max_and_a_combined_job_takes_the_sum():
    """The same two steps, two different right answers.

    In an ARRAY the tasks run side by side, so the allocation has to fit the
    longest one: 24 h. In a COMBINED job they run one after another inside a
    single allocation, so it needs both: 36 h.

    Carrying the array rule into the combined path produced `24:00:00` for a
    job that runs 24 h of relax followed by 12 h of static. It would have died
    partway through the static having already spent a day of compute, and
    nothing in the source says which rule applies -- it was found by building a
    real job and reading the JobSpec.
    """
    as_array = _stage_with([RELAX, STATIC], combined=False)
    as_one_job = _stage_with([RELAX, STATIC], combined=True)

    array_items = [_Item(0), _Item(1)]
    assert as_array._resources_for(array_items)["time"] == "24:00:00"

    combined_item = [_Item(0)]
    combined_item[0].payload["steps"] = ["relax", "static"]
    assert as_one_job._resources_for(combined_item)["time"] == "1-12:00:00"
