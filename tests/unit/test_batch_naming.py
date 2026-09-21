"""A submission's name must not be mistakable for a directory.

The name used to be `items[0].key` -- the first structure in the batch, chosen
by nothing in particular.  One sbatch covers an array, so that name then stood
for work it had nothing to do with:

    26931538_1   dft-78-static   RUNNING        # really running dft-69-relax

Ten hours went into reading the directory the name pointed at, which had
finished hours earlier, because a name in `squeue` looks like an answer.

So: a single-task submission keeps its directory name, where the name IS the
answer.  Anything larger is named for what it is -- `dft-x5-relax-4a1f` -- and
the `x<N>` says "array, go read the manifest" rather than inviting a guess.
"""

from cspflow.stages.dft_stage import _batch_tag


class _Item:
    def __init__(self, key, step_name):
        self.key = key
        self.payload = {"step_name": step_name}


def relaxes(*ids):
    return [_Item(f"dft-{i}-relax", "relax") for i in ids]


def test_a_single_task_keeps_its_directory_name():
    """Here the name is the answer, and hiding it would help nobody."""
    assert _batch_tag(relaxes(87)) == "dft-87-relax"


def test_an_array_is_never_named_after_one_of_its_members():
    """The regression. No member's directory name may be the array's name."""
    items = [_Item("dft-78-static", "static"), _Item("dft-69-relax", "relax")]
    tag = _batch_tag(items)
    assert tag not in {i.key for i in items}
    assert tag.startswith("dft-x2-")


def test_the_name_announces_how_many_tasks():
    assert _batch_tag(relaxes(6, 8, 9, 15, 17)).startswith("dft-x5-")


def test_a_uniform_batch_names_its_stage():
    assert "-relax-" in _batch_tag(relaxes(6, 8, 9))


def test_a_mixed_batch_names_its_composition():
    """`1r+1s` is the thing worth knowing: this array is not one stage, so its
    walltime and memory had to cover both (D130)."""
    items = [_Item("dft-78-static", "static"), _Item("dft-69-relax", "relax")]
    assert "1r+1s" in _batch_tag(items)


def test_two_batches_of_the_same_shape_get_different_names():
    """The name is also the manifest filename, and the manifest is the only
    record of what a task ran. Two batches colliding would overwrite the answer."""
    a = _batch_tag(relaxes(1, 2, 3))
    b = _batch_tag(relaxes(4, 5, 6))
    assert a != b
    assert a.startswith("dft-x3-relax-") and b.startswith("dft-x3-relax-")


def test_the_name_is_stable_for_the_same_batch():
    assert _batch_tag(relaxes(1, 2, 3)) == _batch_tag(relaxes(1, 2, 3))


def test_order_within_a_batch_does_not_change_the_name():
    assert _batch_tag(relaxes(1, 2, 3)) == _batch_tag(relaxes(3, 1, 2))


def test_the_name_is_safe_as_a_filename():
    for items in (relaxes(87), relaxes(1, 2, 3),
                  [_Item("dft-1-static", "static"), _Item("dft-2-relax", "relax")]):
        tag = _batch_tag(items)
        assert "/" not in tag and " " not in tag and tag
