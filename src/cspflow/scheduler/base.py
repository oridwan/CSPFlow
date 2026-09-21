"""What a scheduler is, independent of SLURM.

Only three stages are ever submitted -- `generate`, `screen` and `dft` -- which
is the whole scheduler surface (pipeline.md sec.4.4).  Everything else runs
in-process in seconds.  Keeping that surface this small is what replaces the
~210 shell wrappers in the legacy campaigns.

Two ideas here do real work:

`JobState` separates **what the scheduler did** from **what the science did**.
A VASP job that exits cleanly having failed to relax is `COMPLETED` to SLURM,
and 61% of the jobs in `redo-new-ter-mag` were exactly that.  The scheduler
layer only ever reports the first kind; `relaxation.converged` carries the
second, and nothing in this module is allowed to conflate them.

`Throttle` computes what can actually be in flight, which is never what the
config asks for on its own.  It is the smaller of the user's number, the QOS
submission cap, and how many jobs the CPU allocation cap physically permits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol


class JobState(str, Enum):
    """Scheduler-level outcome.  Deliberately NOT a physics outcome.

    Mapped from the scheduler's own vocabulary by `normalise_state`, whose job
    is to make the messy real strings (`CANCELLED by 3883`, `COMPLETED`,
    `OUT_OF_MEMORY`) land on this closed set.
    """

    pending = "pending"
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"
    timeout = "timeout"
    held = "held"
    unknown = "unknown"

    @property
    def terminal(self) -> bool:
        return self in {
            JobState.done, JobState.failed, JobState.cancelled, JobState.timeout
        }


# Remedies the retry ladder can apply, keyed by what actually went wrong.
# Grounded in 60 days of this account's own job history (10,888 jobs):
#
#     COMPLETED        9,300   85.4%
#     FAILED           1,131   10.4%
#     CANCELLED          287    2.6%
#     TIMEOUT            143    1.3%
#     OUT_OF_MEMORY       27    0.25%
#
# so roughly one job in eight needed some remedy, and the remedy differs by
# cause: more walltime fixes a TIMEOUT and does nothing at all for exit 127.
class Remedy(str, Enum):
    none = "none"
    more_walltime = "more_walltime"
    more_memory = "more_memory"
    fewer_tasks = "fewer_tasks"
    do_not_retry = "do_not_retry"


@dataclass(frozen=True)
class JobStatus:
    """One job as the scheduler currently sees it."""

    job_id: str
    state: JobState
    exit_code: int | None = None
    signal: int | None = None
    reason: str = ""
    elapsed_seconds: float = 0.0
    alloc_cpus: int = 0
    raw_state: str = ""

    @property
    def core_hours(self) -> float:
        return self.elapsed_seconds * self.alloc_cpus / 3600.0

    def remedy(self) -> Remedy:
        """What a retry should change, if anything.

        Exit 127 is the case worth calling out: it is "command not found", and
        this account hit it **287 times in 60 days** -- a missing module, an
        unloaded environment, a path that moved.  Retrying it unchanged burns a
        submission slot to get the identical failure, three times over.  It is
        `csp doctor`'s business, not the retry ladder's.
        """
        if self.state is JobState.timeout:
            return Remedy.more_walltime
        if self.state is JobState.failed:
            if self.raw_state.startswith("OUT_OF_MEMORY") or self.exit_code == 125:
                return Remedy.more_memory
            if self.exit_code == 127:
                return Remedy.do_not_retry
        return Remedy.none


@dataclass
class JobSpec:
    """One submission: a single job, or one array covering many tasks."""

    name: str
    stage: str
    workdir: Path
    command: str
    role: str = "cpu"
    ntasks: int = 16
    cpus_per_task: int = 1
    gpus: int = 0
    mem: str = "32G"
    time: str = "24:00:00"
    array_size: int = 0                 # 0 = not an array
    array_throttle: int = 0             # the %N in --array=0-99%N
    modules: list[str] = field(default_factory=list)
    conda_env: str = ""
    env: dict[str, str | int] = field(default_factory=dict)
    account: str | None = None
    qos: str | None = None
    partition: str = ""
    constraint: str | None = None
    exclude: str | None = None
    depends_on: list[str] = field(default_factory=list)

    @property
    def is_array(self) -> bool:
        return self.array_size > 0


@dataclass(frozen=True)
class Limits:
    """What the site will actually let this account do, read live."""

    max_submit: int | None = None       # QOS MaxSubmitJobsPU
    max_jobs: int | None = None         # QOS MaxJobsPU
    max_cpus: int | None = None         # QOS MaxTRESPU cpu=
    max_gpus: int | None = None         # QOS MaxTRESPU gres/gpu=
    max_array_size: int | None = None   # scontrol MaxArraySize
    source: str = ""                    # where these numbers came from


@dataclass(frozen=True)
class Throttle:
    """How much work may be in flight, and which limit decided it."""

    in_flight: int
    concurrent_tasks: int
    binding: str

    def render(self) -> str:
        """These are LIMITS, and the wording has to say so.

        This string is appended to a stage line that already carries real
        counts, and it used to read

            generate  claimed 1  submitted 1  8 submitted, 8 running at once ...

        where "submitted" appears twice meaning two different things: one job
        actually went out, and the campaign MAY have up to eight. It was read as
        "eight GPU jobs were launched for three structures", which is the
        reasonable reading of that sentence and not what happened.

        `may` is doing the work here. A number in a log line is assumed to be a
        measurement unless something marks it otherwise.
        """
        return (f"may submit {self.in_flight} more, {self.concurrent_tasks} "
                f"may run at once (limit: {self.binding})")


def compute_throttle(
    *,
    requested_in_flight: int,
    requested_concurrent: int,
    ntasks: int,
    gpus_per_job: int = 0,
    limits: Limits | None = None,
    already_in_flight: int = 0,
    max_cores: int | None = None,
    cores_in_flight: int = 0,
) -> Throttle:
    """Reconcile what the config asks for with what the QOS permits.

    The number that matters on this cluster is not the submission cap.  With
    `MaxTRESPU cpu=768` and `ntasks=16`, only **48 VASP jobs can run at once**
    however many are queued -- so the `%N` array throttle, not `max_in_flight`,
    is the real ceiling.  Submitting past it is not faster; it just makes
    `squeue` unreadable and risks tripping `MaxSubmitJobsPU`, which
    `redo-new-ter-mag` did at 2,362 individual submissions against a 2,048 cap.

    `already_in_flight` is subtracted from the submission budget so a driver
    that wakes up with work still queued does not resubmit into the cap.
    """
    limits = limits or Limits()
    binding = "config"

    # A CORE cap, counting this campaign's own queued AND running jobs (D142).
    #
    # `max_in_flight` counts JOBS, which is not the quantity anyone is trying to
    # protect. With this recipe's 64 ranks, `max_in_flight: 48` is 3,072 cores --
    # and no QOS on this cluster's `cpu` role caps it, so the campaign would take
    # everything and leave the user unable to run anything else.
    #
    # Queued counts, not just running: work already in the queue WILL occupy
    # those cores, and a cap that ignores it approves a submission the queue has
    # already spent.
    cores_ceiling: int | None = None
    cores_reason = ""
    if max_cores is not None and ntasks > 0:
        room = max(0, int(max_cores) - int(cores_in_flight))
        cores_ceiling = room // ntasks
        cores_reason = (f"max_cores={max_cores}, {cores_in_flight} held "
                        f"-> room for {cores_ceiling} more at {ntasks} ranks")

    # `max_in_flight` means "this many out at once", so work already queued
    # comes out of it whether or not the site publishes a cap. Subtracting only
    # under a QOS limit meant an unlimited site resubmitted its whole budget
    # every cycle.
    in_flight = max(0, requested_in_flight - already_in_flight)
    if in_flight < requested_in_flight:
        binding = f"config max_in_flight={requested_in_flight}, {already_in_flight} already out"
    if limits.max_submit is not None:
        room = max(0, limits.max_submit - already_in_flight)
        if room < in_flight:
            in_flight, binding = room, f"QOS max_submit={limits.max_submit}"

    concurrent = requested_concurrent
    if limits.max_cpus is not None and ntasks > 0:
        cap = limits.max_cpus // ntasks
        if cap < concurrent:
            concurrent = cap
            binding = f"QOS cpu={limits.max_cpus} at ntasks={ntasks}"
    if gpus_per_job and limits.max_gpus is not None:
        cap = limits.max_gpus // gpus_per_job
        if cap < concurrent:
            concurrent = cap
            binding = f"QOS gpu={limits.max_gpus} at {gpus_per_job} per job"

    if cores_ceiling is not None:
        # Whichever limit is lower is the one to name. Reporting the job count
        # while the core cap is what actually stopped submission sends the
        # reader to change a number that is not binding.
        if cores_ceiling < in_flight:
            binding = cores_reason
        in_flight = min(in_flight, cores_ceiling)
        concurrent = min(concurrent, cores_ceiling)

    # Running more at once than are submitted is meaningless.
    concurrent = min(concurrent, in_flight) if in_flight else concurrent
    return Throttle(in_flight=max(0, in_flight), concurrent_tasks=max(0, concurrent),
                    binding=binding)


def chunk_array(total: int, max_array_size: int | None) -> list[tuple[int, int]]:
    """Split `total` tasks into arrays the scheduler will accept.

    `MaxArraySize` is 10,000 here, which covers an entire campaign's DFT in one
    array -- but a site with a smaller cap, or a campaign that outgrows it, must
    not fail at submission time with a message about array indices.
    """
    if total <= 0:
        return []
    size = max_array_size or total
    size = max(1, size)
    return [(start, min(start + size, total) - 1) for start in range(0, total, size)]


class Scheduler(Protocol):
    """The whole interface the driver loop needs."""

    def submit(self, spec: JobSpec) -> str: ...
    def poll(self, job_ids: list[str]) -> dict[str, JobStatus]: ...
    def poll_tasks(self, job_ids: list[str]) -> dict[str, JobStatus]: ...
    def cancel(self, job_ids: list[str]) -> None: ...
    def limits(self, role: str) -> Limits: ...
    def in_flight(self) -> int: ...
