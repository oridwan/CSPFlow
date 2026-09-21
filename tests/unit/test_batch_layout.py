"""Screen and generate write one directory per submission, with honest names.

TWO FAULTS THIS PINS

*   `tag = items[0].key` named an array after its FIRST CHUNK. An array
    covering structures 1-200 in four chunks went out called `screen-1-50`,
    naming a fifth of its own work. That is the same fault D130 found on the
    DFT side, where ten hours went into reading the directory a job name
    pointed at while the job ran somewhere else.
*   `screen-1-3` reads as "screen 1 of 3" at least as easily as "ids 1 to 3".

Both stages share the worker, and the worker derives every path it writes from
the manifest's parent -- so the manifest moving is what moves the results, the
progress files and the relaxed geometries with it.
"""

from __future__ import annotations

from cspflow.stages.base import WorkItem
from cspflow.stages.screen_stage import _batch_tag, _results_path


def chunk(lo, hi):
    return WorkItem(key=f"screen-{lo}-{hi}", structure_ids=list(range(lo, hi + 1)))


class TestTheName:
    def test_a_single_chunk_is_named_for_its_own_ids(self):
        """One chunk, so the name IS the answer and hides nothing."""
        assert _batch_tag([chunk(1, 3)]) == "ids-0001-0003"

    def test_an_array_is_named_for_every_chunk_not_the_first(self):
        """The bug: this used to be `screen-1-50`, a name for one of four."""
        items = [chunk(1, 50), chunk(51, 100), chunk(101, 150), chunk(151, 200)]
        tag = _batch_tag(items)
        assert tag == "ids-0001-0200-x4"
        assert "0200" in tag, "the span stops short of the work it covers"

    def test_the_x_marks_an_array_so_nobody_reads_it_as_one_range(self):
        assert "-x" not in _batch_tag([chunk(1, 3)])
        assert "-x2" in _batch_tag([chunk(1, 3), chunk(4, 6)])

    def test_ids_are_zero_padded_so_string_sort_is_numeric_sort(self):
        names = sorted(_batch_tag([chunk(a, a)]) for a in (2, 10, 1, 100))
        assert names == ["ids-0001-0001", "ids-0002-0002",
                         "ids-0010-0010", "ids-0100-0100"]

    def test_an_empty_batch_does_not_crash(self):
        assert _batch_tag([]) == "ids-empty"


class TestFindingTheResults:
    def test_the_current_layout_is_preferred(self, tmp_path):
        d = tmp_path / "batches" / "ids-0001-0003"
        d.mkdir(parents=True)
        (d / "results-task0.json").write_text("{}")
        assert _results_path(tmp_path, "ids-0001-0003", 0) == d / "results-task0.json"

    def test_the_old_flat_layout_is_still_read(self, tmp_path):
        """A results file that cannot be found fails a whole chunk of screened
        structures, so the pre-D136 location is still read long after it
        stopped being written."""
        legacy = tmp_path / "screen-1-3.task0.json"
        legacy.write_text("{}")
        assert _results_path(tmp_path, "screen-1-3", 0) == legacy

    def test_a_missing_file_resolves_to_the_current_layout(self, tmp_path):
        """So the error names where it SHOULD be, not where it used to be."""
        got = _results_path(tmp_path, "ids-0001-0003", 0)
        assert got.parent.name == "ids-0001-0003"
        assert not got.exists()
