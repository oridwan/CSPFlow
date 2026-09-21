"""Run jobs as local subprocesses.

Two uses, and the second is the reason it is worth the file.  It makes
`machine: local` a real target, so the pipeline can be exercised on a laptop
with a tiny composition list.  And it lets the driver loop be tested end to end
without a scheduler, which matters because the driver's interesting behaviour --
resuming, throttling, retrying -- is exactly the behaviour that is painful to
test through SLURM.

The semantics deliberately match SLURM's where it costs nothing: a job gets an
id, its state is polled rather than returned, and a failed process becomes
`JobState.failed` with its exit code rather than an exception.  What is *not*
matched is queueing -- jobs run immediately and one at a time -- so the throttle
is honoured by the driver, not by this class.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from itertools import count
from pathlib import Path

from ..config.schema import Machine
from .base import JobSpec, JobState, JobStatus, Limits


class LocalScheduler:
    """Runs each submission in a subprocess, synchronously."""

    def __init__(self, machine: Machine | None = None, *, dry_run: bool = False) -> None:
        self.machine = machine
        self.dry_run = dry_run
        self._ids = count(1)
        # Job ids must be unique across *processes*, not merely within one.
        # A counter alone restarts at 1 for every `csp run`, so the second
        # invocation hands out `local-1` again -- and the job table then holds
        # two unrelated jobs under one id, which reconciliation matches
        # together. SLURM ids are globally unique and nothing else in the
        # system knows that this one was not.
        self._run = uuid.uuid4().hex[:8]
        self._statuses: dict[str, JobStatus] = {}

    def submit(self, spec: JobSpec) -> str:
        job_id = f"local-{self._run}-{next(self._ids)}"
        # Absolute: the job runs with cwd=workdir, so a path relative to the
        # campaign directory resolves to nothing once we are inside it.
        spec.workdir = spec.workdir.resolve()
        spec.workdir.mkdir(parents=True, exist_ok=True)
        script = spec.workdir / f"{spec.name}.sh"
        script.write_text(f"#!/bin/bash\nset -eo pipefail\n\n{spec.command}\n")
        script.chmod(0o755)

        if self.dry_run:
            self._statuses[job_id] = JobStatus(job_id=job_id, state=JobState.queued)
            return job_id

        # An array becomes a loop, with the same variable SLURM would set, so a
        # stage script needs no knowledge of which scheduler it is under.
        tasks = range(spec.array_size) if spec.is_array else [None]
        started = time.monotonic()
        state, exit_code = JobState.done, 0
        log = (spec.workdir / f"{spec.name}.log").open("w")
        try:
            for task in tasks:
                env = {**os.environ, **{k: str(v) for k, v in spec.env.items()}}
                if task is not None:
                    env["SLURM_ARRAY_TASK_ID"] = str(task)
                result = subprocess.run(
                    ["/bin/bash", str(script)], cwd=spec.workdir, env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                )
                if result.returncode != 0:
                    state, exit_code = JobState.failed, result.returncode
                    break
        finally:
            log.close()

        self._statuses[job_id] = JobStatus(
            job_id=job_id, state=state, exit_code=exit_code,
            elapsed_seconds=time.monotonic() - started, alloc_cpus=spec.ntasks,
            raw_state="COMPLETED" if state is JobState.done else "FAILED",
        )
        return job_id

    def poll(self, job_ids: list[str]) -> dict[str, JobStatus]:
        return {
            jid: self._statuses.get(jid, JobStatus(job_id=jid, state=JobState.unknown))
            for jid in job_ids
        }

    def poll_tasks(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """The local scheduler runs work inline, so there are no array tasks.

        Returning nothing is not "unknown": it says this scheduler reports no
        task-level detail, and the driver falls back to whole-submission
        reconciliation, which is exactly right here.
        """
        return {}

    def cancel(self, job_ids: list[str]) -> None:
        for jid in job_ids:
            prior = self._statuses.get(jid)
            if prior is None or not prior.state.terminal:
                self._statuses[jid] = JobStatus(job_id=jid, state=JobState.cancelled)

    def limits(self, role: str) -> Limits:
        """No queue, so the only real limit is the machine's core count."""
        return Limits(max_cpus=os.cpu_count(), source="local")

    def in_flight(self) -> int:
        return 0
