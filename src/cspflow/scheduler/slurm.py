"""SLURM.

Everything site-specific comes from the machine profile; nothing in this file
names a partition, an account or a module.  What it does encode is how SLURM
actually behaves, which is where the sharp edges are:

*   **`squeue` and `sacct` answer different questions.**  `squeue` knows about
    live jobs and forgets a job the moment it finishes.  `sacct` knows about
    finished jobs but only within the accounting retention window.  A job in
    neither is **unknown**, not done -- and treating unknown as done is how a
    driver loop marks a whole campaign complete after an outage.

*   **SLURM's state strings are not the enum you expect.**  Measured on this
    account: 287 of the last 10,888 jobs came back as `CANCELLED by 3883`, with
    the cancelling UID inside the state field.  A `state == "CANCELLED"`
    comparison misses every one of them.

*   **Array tasks have compound IDs** (`26493163_336`), and `sacct` will report
    both the array job and its tasks unless asked not to (`-X`).

*   **`sbatch` failing must not create a job record.**  A submission that did
    not happen, recorded as queued, is a row the driver waits on forever.
"""

from __future__ import annotations

from dataclasses import replace

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

from ..config.schema import Machine
from .base import JobSpec, JobState, JobStatus, Limits, chunk_array

# `sbatch` prints exactly this on success.
_SUBMITTED = re.compile(r"Submitted batch job (\d+)")

# SLURM state -> our vocabulary.  Matching is on the FIRST WORD because several
# real states carry a trailing clause: `CANCELLED by 3883` is the common one and
# accounts for 2.6% of this account's job history.
_STATES = {
    "PENDING": JobState.queued,
    "CONFIGURING": JobState.queued,
    "RUNNING": JobState.running,
    "COMPLETING": JobState.running,
    "SUSPENDED": JobState.running,
    "COMPLETED": JobState.done,
    "FAILED": JobState.failed,
    "NODE_FAIL": JobState.failed,
    "BOOT_FAIL": JobState.failed,
    "OUT_OF_MEMORY": JobState.failed,
    "DEADLINE": JobState.timeout,
    "TIMEOUT": JobState.timeout,
    "CANCELLED": JobState.cancelled,
    "PREEMPTED": JobState.cancelled,
    "REVOKED": JobState.cancelled,
    "SPECIAL_EXIT": JobState.failed,
    "REQUEUED": JobState.queued,
    "RESIZING": JobState.running,
    "STOPPED": JobState.held,
}


class SchedulerError(Exception):
    """A scheduler command that failed, with what was run and what came back."""


def normalise_state(raw: str) -> JobState:
    """`'CANCELLED by 3883'` -> `JobState.cancelled`.

    The trailing clause is why this is a function and not a dict lookup.
    """
    if not raw:
        return JobState.unknown
    return _STATES.get(raw.strip().split()[0].upper(), JobState.unknown)


def parse_exit_code(field: str) -> tuple[int | None, int | None]:
    """`'0:53'` -> `(0, 53)`.  SLURM writes `exit:signal`, not a bare integer."""
    if not field or ":" not in field:
        try:
            return int(field), None
        except (TypeError, ValueError):
            return None, None
    exit_s, _, signal_s = field.partition(":")
    try:
        return int(exit_s), int(signal_s)
    except ValueError:
        return None, None


def parse_elapsed(field: str) -> float:
    """`'3-00:00:09'` or `'00:17:21'` -> seconds."""
    if not field:
        return 0.0
    days, _, clock = field.partition("-")
    if not clock:
        days, clock = "0", days
    parts = clock.split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return 0.0
    while len(values) < 3:
        values.insert(0, 0.0)
    hours, minutes, seconds = values[-3:]
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_tres(field: str, key: str) -> int | None:
    """Pull `cpu=768` or `gres/gpu=12` out of a `MaxTRESPU` string."""
    for item in (field or "").split(","):
        name, _, value = item.partition("=")
        if name.strip() == key:
            try:
                return int(value)
            except ValueError:
                return None
    return None


class SlurmScheduler:
    """Submit, poll and cancel through `sbatch` / `squeue` / `sacct`."""

    def __init__(self, machine: Machine, *, dry_run: bool = False,
                 user: str | None = None) -> None:
        self.machine = machine
        self.dry_run = dry_run
        self.user = user or os.environ.get("USER", "")
        self._limits_cache: dict[str, Limits] = {}

    # -- submission --------------------------------------------------------

    def render_script(self, spec: JobSpec) -> str:
        """The sbatch script, in full.

        Written to disk next to the job rather than piped to `sbatch` so that a
        failed job can be rerun by hand exactly as the driver ran it.  That is
        the difference between "I can reproduce this failure" and "the driver
        did something to it".
        """
        m = self.machine
        partition = spec.partition or m.partition_for(spec.role).name
        profile = m.partition_for(spec.role) if spec.role in m.partitions else None

        lines = ["#!/bin/bash", f"#SBATCH --job-name={spec.name}",
                 f"#SBATCH --partition={partition}",
                 f"#SBATCH --nodes={m.defaults.nodes}",
                 f"#SBATCH --ntasks={spec.ntasks}",
                 f"#SBATCH --cpus-per-task={spec.cpus_per_task}",
                 f"#SBATCH --mem={spec.mem}",
                 f"#SBATCH --time={spec.time}",
                 f"#SBATCH --chdir={spec.workdir}",
                 f"#SBATCH --output={spec.workdir}/slurm-%A_%a.out"
                 if spec.is_array else f"#SBATCH --output={spec.workdir}/slurm-%j.out"]

        if spec.gpus:
            lines.append(f"#SBATCH --gres=gpu:{spec.gpus}")
        if spec.is_array:
            throttle = f"%{spec.array_throttle}" if spec.array_throttle else ""
            lines.append(f"#SBATCH --array=0-{spec.array_size - 1}{throttle}")
        for key, value in (("account", spec.account or (profile.account if profile else None)),
                           ("qos", spec.qos or (profile.qos if profile else None)),
                           ("constraint", spec.constraint or (profile.constraint if profile else None)),
                           ("exclude", spec.exclude or (profile.exclude if profile else None))):
            if value:
                lines.append(f"#SBATCH --{key}={value}")
        for dep in spec.depends_on:
            lines.append(f"#SBATCH --dependency=afterok:{dep}")

        lines.append("")
        # `set -e` but NOT `set -u`: `conda activate` sources third-party
        # activate.d hooks, and one on this machine (julia_activate.sh) reads an
        # unset variable and aborts the whole job under `set -u`.  See D026.
        lines.append("set -eo pipefail")
        lines.append("")
        # VASP compiled with Intel Fortran puts large automatic arrays on the
        # STACK, and this site's soft limit is 8 MB against an unlimited hard
        # limit -- so raising it needs no privilege and is what every VASP
        # install guide asks for.  Without it the run dies as
        #     forrtl: severe (174): SIGSEGV, segmentation fault occurred
        # inside EDDAV, which names neither the stack nor the limit.
        #
        # Measured on t3 2026-09-02: 9 of 9 genuine failures in array 26811406
        # crashed this way, 7 of them at exactly `DAV: 5`, on 9 different nodes
        # across Orion, Apus and Nebula -- with MaxRSS 200-360 MB against the
        # 180 G requested.  That gap is the tell: it is not memory exhaustion,
        # so asking for more `--mem` would have fixed nothing.  ISPIN=2 doubles
        # the wavefunction arrays and brings the overflow forward, which is why
        # the large non-magnetic cells (C240, Si232) hit it first.  See D115.
        lines.append("ulimit -s unlimited || true")
        lines.append("")
        # Submitting from INSIDE a SLURM allocation -- an Open OnDemand shell, an
        # salloc, a job that launches other jobs -- leaks that allocation's
        # SLURM_* variables through sbatch into this script, and from here into
        # srun.  Two of them are actively harmful and neither announces itself:
        #
        #   SLURM_EXPORT_ENV=NONE      srun hands the task an EMPTY environment.
        #                              No PATH ("execve(): bash: No such file or
        #                              directory"), no LD_LIBRARY_PATH (VASP dies
        #                              with "error while loading shared
        #                              libraries: libmkl_intel_lp64.so.2").
        #   SLURM_NTASKS_PER_NODE      a task shape from the OTHER job, which
        #   SLURM_TASKS_PER_NODE       contradicts this job's own #SBATCH lines
        #                              and breaks MPI init.
        #
        # SLURM sets neither of those for a batch job that did not ask for them,
        # so clearing them here restores the default rather than overriding
        # anything this script chose.  Measured on Orion 2026-09-01: without
        # this, 545 of 545 DFT jobs failed in the first seconds on every node of
        # three partitions.
        lines.append("# inherited from the submitting allocation; see D111")
        lines.append("export SLURM_EXPORT_ENV=ALL")
        # NARROW on purpose.  SLURM sets SLURM_NTASKS and SLURM_TASKS_PER_NODE
        # correctly for this batch job from its own #SBATCH --ntasks, and
        # clearing those makes srun fall back to a SINGLE rank -- a 16-core job
        # that quietly runs on one core and still converges.  Measured: the wide
        # version printed "running 1 mpi-ranks" for an --ntasks=16 job.
        #
        # SLURM_NTASKS_PER_NODE is different: it is set only when
        # --ntasks-per-node is requested, which this generator never does, so any
        # value present has leaked in and contradicts the job's own shape.
        lines.append("unset SLURM_NTASKS_PER_NODE")
        lines.append("")

        for module in (spec.modules or m.modules.get(spec.role, [])):
            lines.append(f"module load {module}")
        for key, value in {**m.env, **spec.env}.items():
            text = str(value)
            if "$" in text:
                # A PATH-shaped variable has to be able to name the one it is
                # extending -- LD_LIBRARY_PATH is set by prepending, not by
                # replacing, and clobbering it breaks the conda python that runs
                # two lines further down.  shlex.quote would emit single quotes
                # and the reference would land in the environment literally.
                #
                # Double quotes still protect spaces and globs; what they let
                # through is exactly the expansion that was asked for by writing
                # a `$` in a config file.
                lines.append(f'export {key}="{text}"')
            else:
                lines.append(f"export {key}={shlex.quote(text)}")
        env_name = spec.conda_env or m.conda.get(spec.role, "")
        if env_name:
            lines.append('eval "$(conda shell.bash hook)"')
            lines.append(f"conda activate {env_name}")

        lines.append("")
        lines.append(spec.command)
        lines.append("")
        return "\n".join(lines)

    def submit(self, spec: JobSpec) -> str:
        """Write the script, run `sbatch`, return the job id.

        A non-zero `sbatch` raises rather than returning a fake id.  A
        submission that did not happen, recorded as queued, is a row the driver
        waits on forever.
        """
        # Absolute, for the same reason as LocalScheduler: `#SBATCH --chdir`
        # puts the job inside workdir, where a path relative to the campaign
        # directory no longer resolves.
        spec.workdir = spec.workdir.resolve()
        spec.workdir.mkdir(parents=True, exist_ok=True)
        script = spec.workdir / f"{spec.name}.sbatch"
        script.write_text(self.render_script(spec))
        script.chmod(0o755)

        if self.dry_run:
            return f"dry-run:{spec.name}"

        result = self._run(["sbatch", "--parsable", str(script)])
        text = result.stdout.strip()
        # `--parsable` prints `jobid[;cluster]`; without it, a sentence.
        job_id = text.split(";")[0].strip()
        if not job_id.isdigit():
            match = _SUBMITTED.search(text)
            if not match:
                raise SchedulerError(
                    f"sbatch did not return a job id for {script}.\n"
                    f"stdout: {text!r}\nstderr: {result.stderr.strip()!r}"
                )
            job_id = match.group(1)
        return job_id

    def submit_array(self, spec: JobSpec, total_tasks: int) -> list[str]:
        """Submit `total_tasks` as one array, or as several if the site caps it."""
        cap = self.limits(spec.role).max_array_size
        ids = []
        for offset, (start, end) in enumerate(chunk_array(total_tasks, cap)):
            chunk = JobSpec(**{**spec.__dict__})
            chunk.array_size = end - start + 1
            chunk.name = spec.name if offset == 0 else f"{spec.name}-{offset}"
            chunk.env = {**spec.env, "CSPFLOW_ARRAY_OFFSET": start}
            ids.append(self.submit(chunk))
        return ids

    # -- polling -----------------------------------------------------------

    def poll(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """Live state from `squeue`, finished state from `sacct`, merged.

        Neither alone is sufficient: `squeue` forgets a job the moment it ends,
        `sacct` only reaches back through the accounting retention window.  A
        job neither knows about stays `unknown` -- never `done` -- because
        assuming success for a job nobody remembers is how an outage turns into
        a campaign reported complete.
        """
        if not job_ids:
            return {}
        statuses: dict[str, JobStatus] = {
            jid: JobStatus(job_id=jid, state=JobState.unknown) for jid in job_ids
        }
        observed = self._observe(job_ids)

        # Answer for exactly the ids that were asked about.
        #
        # An array submission returns a base id from `sbatch` -- `26759039` --
        # and both `squeue` and `sacct` report only its *tasks*, `26759039_0`,
        # `26759039_1`, ... So a lookup on the base id finds nothing, the job
        # comes back `unknown`, and `unknown` is deliberately never written as
        # done. The array is therefore polled forever and never reconciled.
        #
        # Neither the local nor the fake scheduler shows this: both echo back
        # whatever id they were handed. It appears only against a real queue,
        # and it appeared on the first live array submission from this
        # repository -- 36 generated structures, job COMPLETED, nothing ingested.
        for jid in job_ids:
            if jid in observed:
                statuses[jid] = observed[jid]
                continue
            tasks = [s for tid, s in observed.items() if tid.startswith(f"{jid}_")]
            if tasks:
                statuses[jid] = aggregate_array(jid, tasks)
        return statuses

    def _observe(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """Every id the scheduler will talk about, base ids and array tasks alike.

        `squeue` is authoritative for what is still live and `sacct` for what
        has ended, so a terminal answer from `sacct` overrides a stale live one
        but never the other way round.
        """
        observed: dict[str, JobStatus] = {}
        observed.update(self._squeue(job_ids))
        for jid, status in self._sacct(job_ids).items():
            if jid not in observed or observed[jid].state is JobState.unknown:
                observed[jid] = status
            elif not observed[jid].state.terminal and status.state.terminal:
                observed[jid] = status
        return observed

    def poll_tasks(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """Per-task statuses, keyed `<base>_<task>` as SLURM reports them.

        `poll` deliberately collapses an array to one status, because a stage
        handed a half-finished array would mark unfinished work done.  But that
        also means a 200-task array holds 200 in-flight slots until its slowest
        task ends, so capacity is released in one lump at the end and the queue
        drains to nothing while work waits (D113).  Reconciling per task frees
        each slot as its own task finishes; the caller is responsible for
        handing back only that task's claim.
        """
        if not job_ids:
            return {}
        return self._observe(job_ids)

    @staticmethod
    def _aggregate(job_id: str, tasks: list[JobStatus]) -> JobStatus:  # pragma: no cover
        return aggregate_array(job_id, tasks)

    def _squeue(self, job_ids: list[str]) -> dict[str, JobStatus]:
        result = self._run(
            ["squeue", "-h", "-o", "%i|%T|%r|%M|%C", "--jobs", ",".join(job_ids)],
            check=False,
        )
        out: dict[str, JobStatus] = {}
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 5:
                continue
            jid, raw, reason, elapsed, cpus = (p.strip() for p in parts[:5])
            out[jid] = JobStatus(
                job_id=jid, state=normalise_state(raw), reason=reason,
                elapsed_seconds=parse_elapsed(elapsed),
                alloc_cpus=int(cpus) if cpus.isdigit() else 0, raw_state=raw,
            )
        return out

    def _sacct(self, job_ids: list[str]) -> dict[str, JobStatus]:
        # -X: allocation rows only.  Without it every job also reports its
        # `.batch` and `.extern` steps, which have their own states.
        result = self._run(
            ["sacct", "-X", "-n", "-P", "-o",
             "JobID,State,ExitCode,Elapsed,AllocCPUS,Reason",
             "--jobs", ",".join(job_ids)],
            check=False,
        )
        out: dict[str, JobStatus] = {}
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 6:
                continue
            jid, raw, code, elapsed, cpus, reason = (p.strip() for p in parts[:6])
            exit_code, signal = parse_exit_code(code)
            out[jid] = JobStatus(
                job_id=jid, state=normalise_state(raw), exit_code=exit_code,
                signal=signal, reason=reason, elapsed_seconds=parse_elapsed(elapsed),
                alloc_cpus=int(cpus) if cpus.isdigit() else 0, raw_state=raw,
            )
        self._promote_step_causes(out)
        return out

    # Step states that say something the allocation row does not.  Only causes
    # that map to a retry rung are worth a second query.
    _STEP_CAUSES = ("OUT_OF_MEMORY",)

    def _promote_step_causes(self, out: dict[str, JobStatus]) -> None:
        """Recover the real cause of death from a job's STEPS.

        `-X` above asks for allocation rows only, so that `.batch` and `.extern`
        do not masquerade as the job.  The cost is that it also hides the step
        that actually died, and SLURM reports the cause THERE, not on the
        allocation:

            26930826_0         dft-18-static   FAILED          1:0
            26930826_0.batch   batch           FAILED          1:0
            26930826_0.0       vasp_std        OUT_OF_MEMORY   0:125

        The driver matched retry rules against the allocation's raw state, saw
        `FAILED`, and never fired the `out_of_memory` rung the magnets recipe has
        carried all along. Three CeFeB statics dead-ended that way on 2026-09-12
        -- sids 18, 50 and 52, all killed after five seconds on three different
        nodes, all with a `resources: {mem: 64G}` remedy sitting unused.

        One extra call, and only when something has already failed.
        """
        failed = [jid for jid, st in out.items()
                  if st.state is JobState.failed and
                  not st.raw_state.startswith(self._STEP_CAUSES)]
        if not failed:
            return

        result = self._run(
            ["sacct", "-n", "-P", "-o", "JobID,State",
             "--jobs", ",".join(failed)],
            check=False,
        )
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 2:
                continue
            step_id, raw = parts[0].strip(), parts[1].strip()
            if "." not in step_id or not raw.startswith(self._STEP_CAUSES):
                continue
            parent = step_id.rsplit(".", 1)[0]
            status = out.get(parent)
            if status is not None:
                # The state stays `failed`; only the REASON is corrected, which
                # is what `_rule_for` matches on.
                out[parent] = replace(status, raw_state=raw)

    def cancel(self, job_ids: list[str]) -> None:
        if job_ids and not self.dry_run:
            self._run(["scancel", *job_ids], check=False)

    def in_flight(self) -> int:
        """How many of this user's jobs are queued or running right now."""
        result = self._run(["squeue", "-h", "-u", self.user, "-o", "%i"], check=False)
        return len([line for line in result.stdout.splitlines() if line.strip()])

    # -- limits ------------------------------------------------------------

    def limits(self, role: str) -> Limits:
        """Read the live QOS and cluster limits, not the machine file.

        The machine profile records them as a starting point, but they are the
        site's to change and a stale copy silently produces submissions that get
        rejected -- or worse, accepted right up to a cap that then blocks
        everything else the account is doing.
        """
        if role in self._limits_cache:
            return self._limits_cache[role]

        qos = self._qos_for(role)
        limits = Limits(max_array_size=self._max_array_size(), source="live")
        if qos:
            row = self._run(
                ["sacctmgr", "-n", "-P", "show", "qos", f"name={qos}",
                 "format=MaxSubmitJobsPU,MaxJobsPU,MaxTRESPU"],
                check=False,
            ).stdout.strip()
            if row:
                submit, jobs, tres = (row.split("|") + ["", "", ""])[:3]
                limits = Limits(
                    max_submit=int(submit) if submit.isdigit() else None,
                    max_jobs=int(jobs) if jobs.isdigit() else None,
                    max_cpus=parse_tres(tres, "cpu"),
                    max_gpus=parse_tres(tres, "gres/gpu"),
                    max_array_size=limits.max_array_size,
                    source=f"sacctmgr qos={qos}",
                )
        self._limits_cache[role] = limits
        return limits

    def _qos_for(self, role: str) -> str | None:
        """Ask SLURM which QOS the partition uses; never guess from its name.

        Guessing looks like it works: on this cluster the `Orion` partition does
        use the `orion` QOS, so a lowercase-the-name heuristic passes its first
        test. It then silently fails on `GPU`, whose QOS is `str_gpu` -- and the
        failure is invisible, because `sacctmgr` simply returns nothing for a
        QOS that does not exist and the limits come back all-`None`, which reads
        as "no limits" rather than "we did not find them".

        Measured here: Orion->orion, GPU->str_gpu, Nebula->nebula,
        Nebula_GPU->nebula_gpu, Apus and Hydrus->N/A (no partition QOS).

        Where a role names several partitions, the first is used. That is the
        conservative choice: it means a comma-list of a limited and an unlimited
        partition throttles to the limited one.
        """
        profile = self.machine.partitions.get(role)
        if profile is None:
            return None
        if profile.qos:
            return profile.qos

        first = profile.name.split(",")[0].strip()
        if not first:
            return None
        out = self._run(["scontrol", "show", "partition", first], check=False).stdout
        match = re.search(r"\bQoS=(\S+)", out)
        if not match or match.group(1) in {"N/A", "(null)"}:
            return None
        return match.group(1)

    def _max_array_size(self) -> int | None:
        out = self._run(["scontrol", "show", "config"], check=False).stdout
        match = re.search(r"MaxArraySize\s*=\s*(\d+)", out)
        return int(match.group(1)) if match else None

    # -- process plumbing --------------------------------------------------

    def _run(self, argv: list[str], *, check: bool = True, timeout: int = 60):
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as exc:
            raise SchedulerError(
                f"{argv[0]} is not on PATH. This machine profile says "
                f"scheduler: slurm -- run `csp doctor` to check."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SchedulerError(f"{' '.join(argv)} timed out after {timeout}s") from exc
        if check and result.returncode != 0:
            raise SchedulerError(
                f"{' '.join(argv)} exited {result.returncode}\n"
                f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
            )
        return result


def aggregate_array(job_id: str, tasks: Sequence[JobStatus]) -> JobStatus:
    """One status for a whole array, from its tasks.

    The array is finished only when every task is. A single running task keeps
    the array running, because reconciling it now would hand a stage a
    half-written set of results and mark the rest of them done.

    Among terminal outcomes the worst wins, in the order failed > timeout >
    cancelled > done: an array where one task ran out of walltime is not a
    successful array, and the retry ladder needs the reason that will actually
    lead somewhere.

    Elapsed time is the maximum (the array's wall clock) and CPUs the sum
    (its cost), which is what makes the campaign's core-hour total honest.
    """
    if not tasks:                                            # pragma: no cover
        return JobStatus(job_id=job_id, state=JobState.unknown)

    live = [t for t in tasks if not t.state.terminal]
    if live:
        worst = min(live, key=lambda t: _LIVE_ORDER.index(t.state)
                    if t.state in _LIVE_ORDER else len(_LIVE_ORDER))
        return JobStatus(
            job_id=job_id, state=worst.state, reason=worst.reason,
            raw_state=worst.raw_state,
            elapsed_seconds=max(t.elapsed_seconds for t in tasks),
            alloc_cpus=sum(t.alloc_cpus for t in tasks),
        )

    worst = min(tasks, key=lambda t: _TERMINAL_ORDER.index(t.state)
                if t.state in _TERMINAL_ORDER else len(_TERMINAL_ORDER))
    return JobStatus(
        job_id=job_id, state=worst.state,
        exit_code=worst.exit_code, signal=worst.signal, reason=worst.reason,
        raw_state=worst.raw_state,
        elapsed_seconds=max(t.elapsed_seconds for t in tasks),
        alloc_cpus=sum(t.alloc_cpus for t in tasks),
    )


# Which live state speaks for an array that has not finished: a running task
# means the array is running, a queued one that it is still queued.
_LIVE_ORDER = [JobState.running, JobState.queued, JobState.pending, JobState.held]
# Which terminal outcome speaks for a finished array.
_TERMINAL_ORDER = [JobState.failed, JobState.timeout, JobState.cancelled, JobState.done]
