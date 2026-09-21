"""A driver reconciles its OWN stages' jobs, and no others.

WHAT THIS COST.  A campaign is routinely run as two drivers over one database --
one `--through reference`, one `--from dft`.  `_reconcile` polled every job in
the table regardless of stage, so whichever driver reached a finished job first
marked it `done`.  When that was the driver that does NOT run the job's stage,
`_hand_back` looked for the stage in its own list, did not find it, and returned
with no message at all.  The job was then in a terminal state, and `_reconcile`
only ever polls queued/running/held -- so nothing looked at it again.

Measured on CeFeB, where `CeFeB-pre` and `CeFeB-dft` overlapped for 13.7 hours:

    relax tasks finishing DURING the overlap : 42, of which 30 lost
    relax tasks finishing after it ended     : 13, of which  0 lost

Thirty structures held a converged VASP relaxation on disk, had no relaxation
record, and sat in `dft_queued` at step 0 with no job in a pollable state.
`dft_queued` read 60 for eleven consecutive cycles while every other count moved
-- the only visible symptom, and an easy one to read as "still busy".
"""

from types import SimpleNamespace

import pytest


class _Stage:
    def __init__(self, name):
        self.name = name
        self.reconciled = []

    def reconcile(self, store, job_row, status, items):
        self.reconciled.append(job_row["stage"])


class _Store:
    def __init__(self, rows):
        self._rows = rows
        self.updated = []

    def jobs(self, **_):
        return self._rows

    def update_job(self, job_id, **kw):
        self.updated.append((job_id, kw))


JOBS = [
    {"id": 1, "stage": "dft", "state": "running", "slurm_id": "100",
     "array_task_id": 0, "workdir": "/tmp", "structure_id": 7},
    {"id": 2, "stage": "screen", "state": "running", "slurm_id": "101",
     "array_task_id": 0, "workdir": "/tmp", "structure_id": 8},
]


def _driver_with(stage_names, monkeypatch):
    """A Driver with `self.stages` set and everything else stubbed out."""
    from cspflow.driver import Driver

    drv = Driver.__new__(Driver)
    drv.stages = [_Stage(n) for n in stage_names]
    drv.store = _Store(list(JOBS))
    drv._claims = {}
    drv.emit = lambda *a, **k: None
    drv.scheduler = SimpleNamespace(poll=lambda ids: {}, poll_tasks=lambda ids: {})
    return drv


def test_a_phase_a_driver_does_not_poll_dft_jobs(monkeypatch):
    """The regression. `--through reference` has no `dft` stage, so it must not
    touch a dft job -- marking it terminal is what made the results unreadable."""
    drv = _driver_with(["source", "generate", "screen", "reference"], monkeypatch)
    polled = {}

    def fake_poll(ids):
        polled["ids"] = list(ids)
        return {}

    drv.scheduler = SimpleNamespace(poll=fake_poll, poll_tasks=lambda ids: {})
    drv._reconcile()
    assert polled.get("ids") == ["101"], "a dft job was polled by a driver without that stage"


def test_a_phase_b_driver_polls_its_own(monkeypatch):
    drv = _driver_with(["filter", "dft", "analyze"], monkeypatch)
    polled = {}
    drv.scheduler = SimpleNamespace(
        poll=lambda ids: polled.setdefault("ids", list(ids)) and {} or {},
        poll_tasks=lambda ids: {})
    drv._reconcile()
    assert polled.get("ids") == ["100"]


def test_nothing_is_polled_when_no_job_belongs_to_this_driver(monkeypatch):
    drv = _driver_with(["analyze"], monkeypatch)
    called = []
    drv.scheduler = SimpleNamespace(poll=lambda ids: called.append(ids) or {},
                                    poll_tasks=lambda ids: {})
    assert drv._reconcile() == 0
    assert not called, "polled the scheduler with nothing to ask about"


def test_the_unreachable_case_is_loud_rather_than_silent(monkeypatch):
    """Kept as a guard, not deleted. Reaching it means a job was marked terminal
    by a driver that cannot read its results -- silent data loss, and it took a
    thirty-structure hole to notice the first time."""
    drv = _driver_with(["analyze"], monkeypatch)
    said = []
    drv.emit = lambda msg: said.append(msg)
    drv._hand_back(JOBS[0], SimpleNamespace(state=None), only_tasks=None)
    assert said and "unread" in said[0]
    assert "dft" in said[0]
