"""Stage 6 -- DFT.

The unit of work is **one structure at one recipe step**, submitted as one task
of a job array. That granularity is deliberate: one structure's failure is one
array task, recorded with a reason, rather than a batch that has to be
disentangled afterwards.

A structure walks the recipe's steps in order (`relax` then `static` in the
shipped recipe), and its position is a number on the row rather than a
directory-naming convention. That is what makes the walk resumable: a driver
that restarts reads the number, not the filesystem.

**The retry ladder is matched to causes, not applied uniformly.** From the
scheduler layer (D046) a job carries both a process outcome and a remedy, and
from the VASP parser it carries a physics outcome. The two are kept apart the
whole way through:

    TIMEOUT              -> resume from CONTCAR, more walltime
    ionic_step_limit     -> resume from CONTCAR, raise NSW
    scf_not_converged    -> ALGO = Normal, raise NELM
    exit code 127        -> do NOT retry; it is a missing binary, and retrying
                            burns three submission slots to reproduce it

Every attempt records what it changed, so `csp status --why` can replay not just
whether a structure succeeded but what was done to make it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from ase.io import read
from collections import Counter
from pathlib import Path
from typing import Any

from ..config.loader import ResolvedConfig
from ..db.store import Store, StructureState
from ..dft.recipe import Recipe, load_recipe
from ..dft.vasp.incar import IncarError
from ..dft.vasp.inputs import InputError, resolve_inputs, write_inputs
from ..dft.layout import (Layout, resolve as resolve_layout, run_slug, slugify,
                          share_potcar, write_structure_json)
from ..dft.runscript import FAILED_STEP as RUNSCRIPT_FAILED_STEP
from ..dft.runscript import render as render_runscript
from ..dft.vasp.parallel import estimate_memory_gb
from ..dft.vasp.parse import (ENERGY_SHIFT_MEV_PER_ATOM, read_job_directory,
                              step_consistency)
from ..scheduler.base import JobSpec, JobStatus, Remedy
from .base import StageReport, WorkItem

# Where a structure is in the recipe, stored on the row rather than inferred
# from the filesystem so that a restarted driver reads it rather than guessing.
STEP_KEY = "dft_step"
DIR_KEY = "dft_dir"
LAST_REMEDY_KEY = "dft_last_remedy"
# Set by calibrate:pilot on the structures it chose. Mirrored here rather than
# imported to keep the two stages from importing each other.

#: SLURM accepts far longer, but `squeue` truncates and a name nobody can read
#: in the tool they read it in is not a name. A run slug plus a campaign fits.
_MAX_JOB_NAME = 80
PILOT_KEY = "pilot"
ATTEMPT_KEY = "dft_attempt"


class DftStage:
    name = "dft"
    role = "cpu"
    in_process = False

    _layout: Layout | None = None
    _layout_note: str = ""

    def __init__(self, cfg: ResolvedConfig, recipe: Recipe | None = None) -> None:
        self.cfg = cfg
        self.recipe = recipe or load_recipe(cfg.campaign.dft.recipe, cfg.base_dir)

    # -- what is ready -----------------------------------------------------

    def pending(self, store: Store) -> int:
        return len(self._ready(store))

    def _ready(self, store: Store) -> list[Any]:
        """Structures selected for DFT that are not finished and not in flight.

        Held back by the pilot gate unless they *are* the pilot. Stage 4b is the
        barrier between the cheap tier and the expensive one, and it cannot
        answer without some DFT of its own -- so its own members always pass,
        and everything else waits for its verdict.

        Ordered by `select.rank_by` and capped by `select.max_total`. Both were
        in the schema and the shipped template and read by nothing, so a
        campaign asking for at most 1,500 DFT jobs ranked by hull distance got
        every candidate in database order.
        """
        gate = self._pilot_gate(store)
        out = []
        for state in (StructureState.selected.value, StructureState.dft_done.value):
            for row in store.structures(state=state):
                if gate and not row.key_value_pairs.get(PILOT_KEY):
                    continue
                step = int(row.key_value_pairs.get(STEP_KEY, 0))
                if step < len(self.recipe.stages):
                    out.append(row)
        return self._rank_and_cap(store, out)

    def _rank_and_cap(self, store: Store, rows: list[Any]) -> list[Any]:
        """Order by `select.rank_by`, then apply `select.max_total`.

        A structure already part-way through the recipe keeps its place at the
        front: abandoning a half-finished relaxation to start a better-ranked
        one from scratch spends more and finishes less.

        `max_total` is a ceiling on the campaign, not a per-cycle throttle, so
        the count includes every structure that has already entered DFT --
        finished, failed or in progress. Capping the ready list alone would let
        a finished structure make room for a new one and the campaign would
        never stop.
        """
        select = getattr(self.cfg.campaign.dft, "select", None)
        if select is None:
            return rows

        key = getattr(select, "rank_by", "") or ""
        started = [r for r in rows if int(r.key_value_pairs.get(STEP_KEY, 0)) > 0]
        fresh = [r for r in rows if int(r.key_value_pairs.get(STEP_KEY, 0)) == 0]

        if key:
            def rank(row):
                value = row.key_value_pairs.get(key)
                # A structure with no value for the ranking key sorts last, not
                # first: an absent hull distance is not a good one.
                return (value is None, float(value) if value is not None else 0.0)

            fresh.sort(key=rank)

        cap = getattr(select, "max_total", None)
        if not cap:
            return started + fresh

        spent = _entered_dft(store)
        room = max(int(cap) - spent, 0)
        return (started + fresh)[:max(room, len(started))]

    def resource_hint(self) -> tuple[int, str]:
        """(ntasks, walltime) for one task, from the recipe's own resources.

        The driver's budget estimate has to come from here: the DFT resources
        live per recipe step, not in the campaign's `dft:` block, so nothing
        outside this stage can find them.  The longest step is used, since the
        estimate is a ceiling.
        """
        ntasks, hours, walltime = 0, 0.0, "24:00:00"
        for stage in self.recipe.stages:
            resources = {**self.recipe.stages[0].resources, **stage.resources}
            ntasks = max(ntasks, int(resources.get("ntasks", 0) or 0))
            time = str(resources.get("time", "24:00:00"))
            parsed = _hours(time)
            if parsed > hours:
                hours, walltime = parsed, time
        return ntasks or self.cfg.machine.defaults.ntasks, walltime

    def _pilot_gate(self, store: Store) -> str:
        """Why the expensive tier is held, or "" if it is open.

        `on_fail: block` (the default) means: no non-pilot DFT until 4b has
        returned a verdict, and none at all if that verdict is FAIL. `warn`
        reports and proceeds; `off` disables the gate entirely.

        A campaign with no `calibrate:` block, or one whose pilot found nothing
        to select from, is not gated -- the barrier exists to stop spending on a
        model that has not been checked, not to stop a campaign that has nothing
        to check it with.
        """
        calibrate = getattr(self.cfg.campaign, "calibrate", None)
        pilot = getattr(calibrate, "pilot", None) if calibrate else None
        policy = getattr(getattr(pilot, "on_fail", None), "value",
                         getattr(pilot, "on_fail", "off"))
        if policy != "block":
            return ""

        latest = store.latest_calibration("pilot")
        if latest is None:
            has_pilot = any(r.key_value_pairs.get(PILOT_KEY) for r in store.structures())
            return ("waiting for the pilot calibration (4b)" if has_pilot
                    else "")
        if latest["verdict"] == "fail":
            return f"pilot calibration FAILED: {latest['detail'][:120]}"
        return ""

    @property
    def layout(self) -> Layout:
        """Which on-disk arrangement this campaign uses.

        Resolved against the directories, not just the config: a campaign that
        already has `dft-<id>-<step>/` directories is mid-flight on the old
        layout, and honouring a `runs` setting would point every path somewhere
        empty -- reporting nothing done and resubmitting finished work.
        """
        if self._layout is None:
            root = self.cfg.work_dir / "dft"
            name, note = resolve_layout(self.cfg.campaign.dft.layout, root)
            if note:
                self._layout_note = note
            self._layout = Layout(name, root)
        return self._layout

    @property
    def combined(self) -> bool:
        """One job per structure, rather than one per structure per step.

        Only possible under `runs`: a combined job needs somewhere to put both
        steps of one structure, and the flat layout has no such directory.
        """
        return bool(getattr(self.cfg.campaign.dft, "combined_job", False)
                    and self.layout.name == "runs")

    @property
    def solo_jobs(self) -> bool:
        """One sbatch per structure, instead of one array over many (D135).

        An array is one allocation with one name, one walltime and one `--mem`
        shared by every task in it, and a side file saying which task is which.
        Every one of those is a thing that can be wrong about a structure it was
        never chosen for:

        * SLURM fixes the array's name at submission from task 0, so `squeue`
          showed `dft-78-static` for a task running `dft-69-relax` (D130);
        * one walltime served every task, and thirteen 24-hour relaxes ran under
          a static's 12-hour cap and hit the wall (D130);
        * one `--mem` had to fit the largest structure, so every task paid for
          the worst one (D128);
        * the array held its in-flight slots until its SLOWEST task drained.

        With one structure per job the name IS the directory, the walltime is
        this structure's, the memory is this structure's, and a finished job
        frees its slot when it finishes. None of the machinery above is guarded;
        it is deleted.

        The old flat layout keeps arrays, because its directories are per STEP
        and a campaign already running on them must not have its paths change
        underneath it.
        """
        return self.layout.name == "runs"

    def _job_name(self, item: WorkItem) -> str:
        """What `squeue` will show for one structure's job.

        `item.key` is already the run directory's name under `runs`:
        `0195-Ce2PdGe6-agentic_x3_o5-7_Cu`. The campaign is prefixed because
        `squeue` is per USER, not per campaign, and every campaign numbers its
        structures from 1 -- `0069-...` exists in two campaigns at once, which
        is how a job gets matched to the wrong folder (D130, cause 2).
        """
        return f"{slugify(self.cfg.campaign.name)}-{item.key}"[:_MAX_JOB_NAME]

    def _identity(self, store: Store, sid: int) -> tuple[str, str]:
        """(formula, seed path) for naming this structure's directory."""
        try:
            row = store.get_structure(sid)
        except Exception:                                      # noqa: BLE001
            return "", ""
        return str(row.formula or ""), str(row.key_value_pairs.get("source_path") or "")

    def claim(self, store: Store, budget: int) -> list[WorkItem]:
        items = []
        for row in self._ready(store)[:budget]:
            kv = row.key_value_pairs
            step = int(kv.get(STEP_KEY, 0))
            stage = self.recipe.stages[step]
            store.set_structure_state(int(row.id), StructureState.dft_queued,
                                      **{STEP_KEY: step})
            # The ladder's decision, carried forward. Without this the remedy was
            # recorded on the row and then never read: `set: {NSW: 200}` was
            # stored as `dft_last_remedy` and the rerun was written with the
            # original NSW. Every retry repeated the identical calculation and
            # failed the identical way, which is worse than not retrying at all
            # -- it costs the same again and looks like diligence.
            remedy = _decode_remedy(kv.get(LAST_REMEDY_KEY))
            payload = {"step": step, "step_name": stage.name,
                       "attempt": int(kv.get(ATTEMPT_KEY, 0)),
                       "incar_overrides": remedy.get("set", {}),
                       "resources": remedy.get("resources", {}),
                       "remedy": remedy.get("remedy", "")}
            if self.combined:
                # One item = one structure's REMAINING steps, so the array is
                # homogeneous: every task is "finish this structure". A batch can
                # then no longer mix a 24-hour relax with a 12-hour static and
                # hand both the shorter walltime (D130), because there is no
                # second kind of task to mix with.
                formula, seed = self._identity(store, int(row.id))
                payload["steps"] = [st.name for st in self.recipe.stages[step:]]
                payload["formula"] = formula
                payload["source_path"] = seed
                key = run_slug(int(row.id), formula, seed)
            else:
                key = f"dft-{row.id}-{stage.name}"
            items.append(WorkItem(key=key, structure_ids=[int(row.id)],
                                  payload=payload))
        return items

    # -- the job -----------------------------------------------------------

    def build(self, items: list[WorkItem], workdir: Path) -> JobSpec:
        """Write every task's input directory, then one array over them.

        Inputs are written now rather than inside the job so that a failure to
        assemble them -- an unresolvable POTCAR, a mixed 4f convention -- is
        caught here, on the login node, in milliseconds, instead of on a compute
        node after the queue wait.
        """
        workdir = workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        # One structure whose inputs cannot be assembled must not stop the
        # campaign.  It used to: `IncarError` is not a subclass of `InputError`,
        # so the per-item handler in `_write_inputs` did not catch it, nothing
        # between here and `Driver.run` did either, and the driver died with a
        # traceback. Seen 2026-09-12 -- a single In-bearing seed, for which the
        # magnetism table had no initial moment, took down the whole CePdGe
        # Phase B driver on its first cycle.
        #
        # The refusal itself is right and stays right (a wrong initial moment
        # does not announce itself). What changes is the blast radius: that
        # structure is marked failed with the reason on it, and the rest of the
        # batch goes to the queue.
        directories, usable, rejected = [], [], []
        for item in items:
            directory = self._directory_for(item, workdir)
            try:
                self._write_inputs(item, directory)
                if self.combined:
                    self._record_structure(item, directory)
            except (InputError, IncarError) as exc:
                rejected.append((item, str(exc)))
                continue
            directories.append(str(directory))
            usable.append(item)

        if rejected:
            self._mark_unbuildable(rejected)
        if not usable:
            raise InputError(
                f"none of the {len(items)} structure(s) in this batch could have "
                f"their inputs written; first reason: {rejected[0][1]}")
        items[:] = usable

        # ONE STRUCTURE, ONE JOB (D135). With a single task there is no index to
        # resolve, so there is no manifest and no lookup: the job's name is its
        # directory, and its script and log are written INTO that directory by
        # `--chdir`. An array still gets the manifest, because an array still has
        # the problem the manifest was invented for.
        solo = self.solo_jobs and len(items) == 1
        var = "RUN" if self.combined else "DIRS"
        if solo:
            tag = self._job_name(items[0])
            spec_workdir = Path(directories[0])
            prologue = f"{var}={shlex.quote(directories[0])}"
        else:
            tag = _batch_tag(items)
            manifest = workdir / f"{tag}.tasks.json"
            manifest.write_text(json.dumps({"dirs": directories}, indent=2))
            spec_workdir = workdir
            prologue = (
                f'{var}=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))'
                f"['dirs'][int(sys.argv[2])])\" {manifest} ${{SLURM_ARRAY_TASK_ID:-0}})")

        step = items[0].payload["step"]
        resources = self._resources_for(items)
        # A retry rung may raise a resource. One array shares one allocation, so
        # every override in the batch is applied and the batch runs at whatever
        # the most demanding task asked for -- under-serving one task to save a
        # little on the others is how a retry fails the same way twice.
        for _item in items:
            resources.update(_item.payload.get("resources") or {})
        launcher = self.cfg.machine.codes.mpi_launcher
        binary = self.cfg.machine.codes.vasp_std or "vasp_std"

        if self.combined:
            # Every step of one structure, in this one task. The body skips a
            # step that already converged, records WHICH step failed, and
            # regenerates each later step's inputs from the previous step's
            # CONTCAR -- which cannot be done before the job runs, because that
            # CONTCAR does not exist yet.
            steps = list(items[0].payload.get("steps")
                         or [st.name for st in self.recipe.stages])
            command = "\n".join([
                prologue,
                render_runscript(
                    steps, launcher=launcher, binary=binary,
                    campaign=_campaign_file(self.cfg),
                    ntasks=int(resources.get("ntasks",
                                             self.cfg.machine.defaults.ntasks)),
                ),
            ])
        else:
            command = "\n".join([
            prologue,
            'cd "$DIRS"',
            # Both directions of the job <-> directory mapping, recorded at the
            # one moment both are known. Without them the only record is the
            # manifest, and a name like `dft-78-static` covering a task that runs
            # `dft-69-relax` is not something anyone guesses.
            'echo "task ${SLURM_ARRAY_TASK_ID:-0} of ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}'
            ' -> $DIRS"',
            'echo "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"'
            ' > SLURM_TASK',
            f"{launcher} {binary} > vasp.out 2>&1",
            "echo done > VASP_DONE",   # a marker that the process exited, nothing more
            ])

        return JobSpec(
            name=tag, stage=self.name, workdir=spec_workdir, command=command,
            role=self.role,
            # From the merged resources, not from items[0]'s step: the INCAR's
            # KPAR is chosen against a rank count, and VASP refuses to start if
            # KPAR does not divide the ranks the job actually gets.
            ntasks=int(resources.get("ntasks", self.cfg.machine.defaults.ntasks)),
            cpus_per_task=int(resources.get("cpus_per_task", 1)),
            mem=self._mem_for(resources, directories,
                              int(resources.get("ntasks",
                                                self.cfg.machine.defaults.ntasks))),
            time=str(resources.get("time", "24:00:00")),
            # 0 means "not an array". A one-task array still carries an array's
            # baggage -- an `_0` task suffix, a `%N` throttle, `%A_%a` logs --
            # for a job that has nothing to throttle against.
            array_size=0 if solo else len(items),
        )

    def _mem_for(self, resources: dict, directories: list[str], ntasks: int) -> str:
        """`--mem` sized to the largest structure in this array.

        A recipe that states `mem:` outright is obeyed and nothing is estimated;
        that is the escape hatch for a structure known to be unusual.

        Otherwise the request is computed per structure and the MAXIMUM taken,
        because one array shares one `--mem` and every task must fit under it.

        Reading the INCAR and POSCAR that were just written, rather than
        recomputing NBANDS and KPAR here, keeps this honest: the estimate is
        made from the values the job will actually run with, so it cannot drift
        away from them the way a parallel derivation would.

        Never fatal. A structure whose files will not parse falls back to the
        machine default, because a memory estimate is an optimisation and a
        campaign that stops over one is worse than one that over-requests.
        """
        stated = resources.get("mem")
        if stated:
            return str(stated)

        default = str(self.cfg.machine.defaults.mem)
        best = 0
        for directory in directories:
            d = Path(directory)
            # Combined: `directory` is the structure's ROOT and the inputs are a
            # level down, in the first step's subdirectory. Looking for the INCAR
            # here found nothing, the handler returned the machine default, and
            # every per-structure memory estimate from D128 was silently skipped
            # -- back to one flat 32G for everything, which is what OOM-killed
            # three statics in the first place.
            if not (d / "INCAR").is_file():
                inner = sorted(x for x in d.glob("*/INCAR"))
                if inner:
                    d = inner[0].parent
            try:
                incar = (d / "INCAR").read_text()
                nbands = int(re.search(r"^NBANDS\s*=\s*(\d+)", incar, re.M).group(1))
                encut = float(re.search(r"^ENCUT\s*=\s*([\d.]+)", incar, re.M).group(1))
                kpar_m = re.search(r"^KPAR\s*=\s*(\d+)", incar, re.M)
                kpar = int(kpar_m.group(1)) if kpar_m else 1
                volume = read(str(d / "POSCAR")).get_volume()
            except Exception:                                  # noqa: BLE001
                return default
            best = max(best, estimate_memory_gb(volume, encut, nbands, kpar, ntasks))
        return f"{best}G" if best else default

    def _record_structure(self, item: WorkItem, run_root: Path) -> None:
        """The structure's own record, and one shared POTCAR instead of many.

        The record exists because a database is not the only place an answer
        should live: 37 finished calculations were once unreachable because the
        DB forgot them and nothing on disk said otherwise (D129).

        Neither is allowed to fail the build. A provenance file and a saved
        90 MB are both worth less than the calculation they sit beside.
        """
        sid = item.structure_ids[0]
        step_name = item.payload.get("step_name") or self.recipe.stages[0].name
        try:
            share_potcar(run_root / step_name, self.layout.potcar_store())
        except Exception:                                      # noqa: BLE001
            pass
        try:
            write_structure_json(run_root / "structure.json", {
                "structure_id": sid,
                "formula": item.payload.get("formula", ""),
                "source_path": item.payload.get("source_path", ""),
                "recipe": self.recipe.name,
                "steps": list(item.payload.get("steps") or []),
                "attempt": int(item.payload.get("attempt", 0)),
                "campaign": self.cfg.campaign.name,
            })
        except Exception:                                      # noqa: BLE001
            pass

    def _directory_for(self, item: WorkItem, workdir: Path) -> Path:
        """Where this item's files go.

        Combined: the STRUCTURE's directory, holding every step as a
        subdirectory. Otherwise: the step's own flat directory, exactly as
        before -- a running campaign depends on those paths.
        """
        sid = item.structure_ids[0]
        if self.combined:
            return self.layout.run_root(sid, item.payload.get("formula", ""),
                                        item.payload.get("source_path", ""))
        return workdir / item.key

    def _resources_for(self, items: list[WorkItem]) -> dict:
        """One allocation that is sufficient for EVERY task in the array.

        A batch is not guaranteed to be all one recipe step. `claim()` fills it
        from whatever is ready, so a `static` and a `relax` routinely go out
        together -- 17 such arrays existed across CeFeB and CePdGe when this was
        found. Taking the resources from `items[0]` then gives the whole array
        the first task's allocation, and because a finished relax becomes a
        ready static, `items[0]` is very often the static:

            dft-78-static.tasks.json -> [dft-78-static, dft-69-relax]
            #SBATCH --time=12:00:00          <- the STATIC's walltime
            recipe: relax 24:00:00, static 12:00:00

        `dft-69-relax` was running on a 12-hour cap against a recipe that gives a
        relax 24, and one CePdGe array had THIRTEEN relaxes under that cap. They
        do not fail loudly; they hit the wall and come back as TIMEOUT, having
        spent the full 12 hours.

        So take the MAXIMUM over every step present, not the first. An array
        cannot give its tasks different walltimes, and over-serving the short
        ones costs queue priority while under-serving the long ones costs the
        whole run.
        """
        steps = sorted({int(i.payload.get("step", 0)) for i in items}) or [0]
        if self.combined:
            # A combined task runs every REMAINING step, not just the one its
            # payload is currently pointing at.
            names = [st.name for st in self.recipe.stages]
            covered = set()
            for i in items:
                for name in (i.payload.get("steps") or []):
                    if name in names:
                        covered.add(names.index(name))
            steps = sorted(covered or set(steps))
        # Stage 0 is the DEFAULT each stage overrides, not a floor. Treating it
        # as a floor would give a static-only array the relax's 24 hours, which
        # is the same class of mistake in the other direction: every static
        # queued for twice as long as it needs.
        merged: dict = {}
        seconds = 0
        for st in steps:
            per_step = {**self.recipe.stages[0].resources,
                        **self.recipe.stages[st].resources}
            for key, value in per_step.items():
                merged[key] = _larger(key, merged.get(key), value)
            seconds += _seconds(per_step.get("time", 0))

        if self.combined and len(steps) > 1:
            # SUM, not max, and only for walltime.
            #
            # A combined job runs its steps one after another inside one
            # allocation, so it needs relax + static, not whichever is longer.
            # The max is right for a heterogeneous ARRAY -- separate tasks, each
            # needing to fit -- and wrong here in the dangerous direction: the
            # recipe's 24 h relax and 12 h static came out as 24 h, and the job
            # would have died partway through the static having already spent a
            # day of compute. Memory and rank count stay MAX: they are held
            # concurrently, not consumed in turn.
            merged["time"] = _format_walltime(seconds)
        # A retry rung may raise a resource. One array shares one allocation, so
        # every override in the batch is applied and the batch runs at whatever
        # the most demanding task asked for -- under-serving one task to save a
        # little on the others is how a retry fails the same way twice.
        for item in items:
            for key, value in (item.payload.get("resources") or {}).items():
                merged[key] = _larger(key, merged.get(key), value)
        return merged

    def _ntasks_for(self, step: int) -> int:
        """How many MPI ranks this step's jobs will be given.

        Read in TWO places -- here for the `JobSpec`, and again when the INCAR
        is written, so `KPAR` can be chosen against the rank count the job will
        really have.  Hence one method rather than the expression inline twice:
        VASP requires `KPAR` to divide the rank count exactly, so a `KPAR`
        picked for 32 ranks on a job submitted with 64 is rejected at startup,
        before a single electronic step -- and the two values drifting apart
        would be invisible in the config, which is where anyone would look.
        """
        resources = {**self.recipe.stages[0].resources,
                     **self.recipe.stages[step].resources}
        return int(resources.get("ntasks", self.cfg.machine.defaults.ntasks))

    def _ntasks_for_items(self, items: list[WorkItem]) -> int:
        """The rank count the ARRAY will really get.

        KPAR must divide the rank count exactly, and the INCAR is written per
        task while the allocation is shared, so both sides have to read the same
        number or VASP refuses to start.
        """
        return int(self._resources_for(items).get(
            "ntasks", self.cfg.machine.defaults.ntasks))

    def _write_inputs(self, item: WorkItem, directory: Path) -> None:
        from ..db.store import Store as _Store

        # Combined: `directory` is the STRUCTURE's root and only the FIRST
        # remaining step's inputs are written here. Every later step is prepared
        # inside the job, from the previous step's CONTCAR -- it cannot be done
        # now, because that CONTCAR will not exist until the job has run.
        if self.combined:
            step_name = item.payload.get("step_name") or self.recipe.stages[0].name
            directory.mkdir(parents=True, exist_ok=True)
            directory = directory / step_name

        # Whatever a previous attempt left here is moved aside before anything
        # is written, so a retry does not erase the evidence of what it is
        # retrying. `csp status --why` names the remedy; the archive is where
        # you look to see whether it was the right one.
        archived = _archive_previous(directory, int(item.payload.get("attempt", 0)))

        store = _Store.open(self.cfg.campaign_db)
        try:
            row = store.get_structure(item.structure_ids[0])
            atoms = row.toatoms()
            previous_dir = row.key_value_pairs.get(DIR_KEY)
        finally:
            store.close()

        # Where this step starts from, in order of preference.
        #
        # 1. A retry that asked to resume picks up its own previous attempt.
        # 2. Otherwise a step after the first starts from the *previous step's*
        #    CONTCAR. This is the one that matters most: `static` exists to give
        #    a high-accuracy energy AT THE RELAXED GEOMETRY, and its energy is
        #    what goes onto the DFT hull. Started from the structure in the
        #    database it runs on the generated cell instead, and reports a
        #    number that looks entirely plausible and is wrong by whatever the
        #    relaxation was worth. Measured live before the fix: 176.15 vs
        #    179.03 A^3, 172.92 vs 179.71, 260.70 vs 260.84.
        # 3. Otherwise the structure as generated.
        source = None
        carried_grid = None
        if item.payload.get("remedy") == "resume_from_contcar" and archived:
            source = archived / "CONTCAR"
            # The grid comes with the geometry. Re-deriving it from the cell
            # the previous attempt reached is what turns a resume into a
            # restart on a different energy surface -- see kpoints.carry_grid.
            carried_grid = _archived_grid(archived)
        elif int(item.payload.get("step", 0)) > 0 and previous_dir:
            source = Path(previous_dir) / "CONTCAR"

        if source is not None:
            carried = _read_contcar(source)
            if carried is not None:
                atoms = carried
                item.payload["started_from"] = str(source)
            elif int(item.payload.get("step", 0)) > 0:
                raise InputError(
                    f"structure {item.structure_ids[0]} is at recipe step "
                    f"{item.payload['step']} but {source} is missing or empty. "
                    f"Running this step on the unrelaxed geometry would produce "
                    f"a plausible energy at the wrong structure.")

        step = int(item.payload["step"])
        stage = self.recipe.stages[step]
        overrides = item.payload.get("incar_overrides") or {}
        if overrides:
            stage = stage.with_overrides(overrides)

        try:
            # `ntasks` is what turns KPAR/NCORE on. Without it `resolve_inputs`
            # leaves both at whatever the recipe hardcodes, and the recipe
            # cannot know: the right split depends on this structure's own
            # irreducible k-point count, which ranged 1 to 232 across the
            # reference set at one fixed NCORE = 8.
            resolved = resolve_inputs(atoms, stage, self.cfg.campaign.dft,
                                      self.cfg.machine,
                                      ntasks=self._ntasks_for(step),
                                      carried_grid=carried_grid)
        except InputError as exc:
            raise InputError(
                f"structure {item.structure_ids[0]} at step '{stage.name}': {exc}"
            ) from exc
        write_inputs(resolved, atoms, directory)
        item.payload["settings_hash"] = resolved.settings_hash

    def _mark_unbuildable(self, rejected: list[tuple[Any, str]]) -> None:
        """Fail the structures whose inputs could not be written, with the reason.

        Recorded on the structure rather than only logged, so `csp status --why`
        can answer "what happened to 118?" after the driver has moved on, and so
        the next cycle does not claim them again and fail the same way forever.
        """
        from ..db.store import Store as _Store

        store = _Store.open(self.cfg.campaign_db)
        try:
            for item, reason in rejected:
                sid = int(item.structure_ids[0])
                store.set_structure_state(sid, StructureState.failed,
                                          fail_reason=reason[:200])
                store.add_filter_event(structure_id=sid, gate="dft:write_inputs",
                                       passed=False, detail=reason[:200])
        finally:
            store.close()

    # -- folding the answer back -------------------------------------------

    def reconcile(self, store: Store, job_row: Any, status: JobStatus,
                  items: list[WorkItem]) -> None:
        workdir = Path(job_row["workdir"])
        for item in items:
            sid = item.structure_ids[0]
            if self.combined:
                self._reconcile_combined(store, sid, item, status)
                continue
            directory = workdir / item.key
            outcome = read_job_directory(directory)
            step = int(item.payload.get("step", 0))
            step_name = item.payload.get("step_name", "")

            store.add_filter_event(
                structure_id=sid, gate=f"dft:{step_name}:converged",
                passed=bool(outcome.converged),
                value=float(outcome.n_ionic_steps),
                threshold=float(outcome.step_limit) if outcome.step_limit else None,
                detail=outcome.exit_reason,
            )
            if outcome.energy is not None:
                store.add_relaxation(
                    structure_id=sid, engine=f"vasp:{step_name}",
                    energy=outcome.energy, e_per_atom=outcome.e_per_atom,
                    converged=outcome.converged, n_steps=outcome.n_ionic_steps,
                )

            # Under this layout the steps are separate jobs, so the previous
            # one's outcome is not in hand -- it is re-read from the directory
            # `_advance` recorded when it finished.
            if step > 0:
                previous_dir = store.get_structure(sid).key_value_pairs.get(DIR_KEY)
                if previous_dir and (Path(previous_dir) / "OUTCAR").is_file():
                    self._record_consistency(
                        store, sid, step_name,
                        read_job_directory(Path(previous_dir)), outcome)

            if outcome.converged:
                self._advance(store, sid, step, outcome, directory)
            else:
                self._retry_or_fail(store, sid, step, step_name, outcome, status, item)

    def _reconcile_combined(self, store: Store, sid: int, item: WorkItem,
                            status: JobStatus) -> None:
        """Fold back a job that ran EVERY remaining step of one structure.

        Each step is read from its own directory and recorded separately --
        `csp status` must keep distinguishing `vasp:relax` from `vasp:static`,
        and one combined verdict would erase that.

        `FAILED_STEP`, written by the job itself, says which step stopped it.
        The retry ladder is per-step -- `timeout` on a relax means resume from
        CONTCAR, on a static it means ask for more time -- so without that file
        the ladder picks a remedy for the wrong stage. When it is absent, the
        first step that did not converge is used instead, which is the same
        answer by a slower route.
        """
        run_root = self.layout.run_root(sid, item.payload.get("formula", ""),
                                        item.payload.get("source_path", ""))
        steps = list(item.payload.get("steps")
                     or [st.name for st in self.recipe.stages])
        names = [st.name for st in self.recipe.stages]

        marker = run_root / RUNSCRIPT_FAILED_STEP
        failed_name = marker.read_text().strip() if marker.is_file() else ""

        last_outcome = None
        previous_outcome = None
        first_bad: tuple[int, str, Any] | None = None
        for name in steps:
            directory = run_root / name
            if not (directory / "OUTCAR").is_file():
                if first_bad is None:
                    first_bad = (names.index(name), name, read_job_directory(directory))
                break
            outcome = read_job_directory(directory)
            if previous_outcome is not None:
                self._record_consistency(store, sid, name, previous_outcome, outcome)
            previous_outcome = outcome
            last_outcome = outcome
            store.add_filter_event(
                structure_id=sid, gate=f"dft:{name}:converged",
                passed=bool(outcome.converged),
                value=float(outcome.n_ionic_steps),
                threshold=float(outcome.step_limit) if outcome.step_limit else None,
                detail=outcome.exit_reason,
            )
            if outcome.energy is not None:
                store.add_relaxation(
                    structure_id=sid, engine=f"vasp:{name}",
                    energy=outcome.energy, e_per_atom=outcome.e_per_atom,
                    converged=outcome.converged, n_steps=outcome.n_ionic_steps,
                )
            if not outcome.converged and first_bad is None:
                first_bad = (names.index(name), name, outcome)

        if failed_name and failed_name in names:
            idx = names.index(failed_name)
            bad = (idx, failed_name, read_job_directory(run_root / failed_name))
        else:
            bad = first_bad

        if bad is None and last_outcome is not None and last_outcome.converged:
            self._advance(store, sid, len(names) - 1, last_outcome,
                          run_root / steps[-1])
            return
        if bad is None:                       # nothing ran at all
            bad = (names.index(steps[0]), steps[0], read_job_directory(run_root / steps[0]))
        idx, name, outcome = bad
        item.payload["step"] = idx
        item.payload["step_name"] = name
        self._retry_or_fail(store, sid, idx, name, outcome, status, item)

    def _record_consistency(self, store: Store, sid: int, step_name: str,
                            previous, current) -> None:
        """Do this step and the one whose geometry it inherited agree?

        Both steps can converge perfectly and still not describe the same
        calculation. The static starts its SCF from the MAGMOM guess again --
        no WAVECAR, no CHGCAR are kept -- so it re-finds the magnetic solution
        from scratch and can settle in a different one from the relaxation that
        produced its geometry. Nothing downstream would notice: two converged
        steps, two plausible energies, and a structure ranked on the energy of a
        magnetic state its geometry was never optimised for.

        Measured over 2,067 structures of RE-magnets-CHGNet: of the 46 whose
        energy moved more than 60 meV/atom between the steps, 41 (89%) had also
        changed moment, against 3% of those that moved less than 5 meV/atom.
        Structure 13836 went -2.7 -> +4.1 uB and its energy moved 656 meV/atom.

        Recorded, never fatal. The number may still be the one you want; what it
        must not be is silently indistinguishable from a clean result.
        """
        check = step_consistency(previous, current)
        if not check.measurable:
            return
        store.add_filter_event(
            structure_id=sid, gate=f"dft:{step_name}:consistent_with_previous",
            passed=check.ok, value=check.energy_shift,
            threshold=ENERGY_SHIFT_MEV_PER_ATOM, detail=check.detail[:400],
        )
        kv: dict[str, Any] = {f"dft_{step_name}_shift_mev": check.energy_shift}
        if check.magmom_shift is not None:
            kv[f"dft_{step_name}_magmom_shift"] = check.magmom_shift
        if not check.ok:
            # A key, not only an event: `csp status --why` and the candidate
            # table read the row, and a warning nobody is shown is not a warning.
            kv["dft_warning"] = f"{step_name}: {check.detail}"[:200]
        store.update_structure(sid, **kv)

    def _advance(self, store: Store, sid: int, step: int, outcome,
                 directory: Path) -> None:
        """This step succeeded.  Move to the next, or finish."""
        # Where the outputs are, recorded on the row rather than left to be
        # rebuilt from a filename convention later. `analyze` reads it: the
        # alternative is a second place that knows how job directories are
        # named, and two such places drift.
        # The remedy belongs to the step that failed. Carried into the next
        # step it silently rewrites that step's INCAR -- live, a `relax` retry's
        # `NSW: 200` landed in the `static` INCAR, so a fixed-position
        # calculation ran two hundred identical ionic steps.
        kv: dict[str, Any] = {STEP_KEY: step + 1, ATTEMPT_KEY: 0,
                              LAST_REMEDY_KEY: "",
                              DIR_KEY: str(directory.resolve())}
        if outcome.energy is not None:
            kv["vasp_energy"] = outcome.energy
        if outcome.e_per_atom is not None:
            kv["e_per_atom"] = outcome.e_per_atom
        if outcome.magnetisation is not None:
            kv["magnetisation"] = outcome.magnetisation

        finished = step + 1 >= len(self.recipe.stages)
        store.set_structure_state(
            sid, StructureState.dft_done if finished else StructureState.selected, **kv
        )

    def _retry_or_fail(self, store: Store, sid: int, step: int, step_name: str,
                       outcome, status: JobStatus, item: WorkItem) -> None:
        """Apply the ladder, or stop and say why."""
        attempt = int(item.payload.get("attempt", 0)) + 1
        max_attempts = _max_attempts(self.recipe.stages[step])

        rule = self._rule_for(step, outcome, status)
        if rule is None or attempt >= max_attempts:
            reason = outcome.exit_reason or status.raw_state or "unknown"
            store.set_structure_state(
                sid, StructureState.failed,
                dft_fail_reason=f"{step_name}: {reason}"[:200],
                **{ATTEMPT_KEY: attempt},
            )
            return

        store.set_structure_state(
            sid, StructureState.selected,
            **{STEP_KEY: step, ATTEMPT_KEY: attempt,
               LAST_REMEDY_KEY: json.dumps({"set": rule.get("set", {}),
                                            "resources": rule.get("resources", {}),
                                            "remedy": rule.get("remedy", "")})[:200]},
        )

    def _rule_for(self, step: int, outcome, status: JobStatus) -> dict | None:
        """Match a failure to a remedy.  A remedy has to fit its cause.

        Exit 127 is never retried: it is `command not found`, this account hit it
        287 times in 60 days, and retrying it unchanged burns a submission slot
        to get the identical failure.
        """
        if status.remedy() is Remedy.do_not_retry:
            return None

        triggers = {outcome.exit_reason}
        if status.raw_state.startswith("TIMEOUT"):
            triggers.add("timeout")
        if status.raw_state.startswith("OUT_OF_MEMORY"):
            triggers.add("out_of_memory")
        if outcome.unconverged_but_finished:
            # VASP exited cleanly without reaching the criterion. Which criterion
            # it missed decides the remedy, and `exit_reason` already carries it
            # -- `ionic_step_limit` wants a resume with a higher NSW, an SCF
            # failure wants a different algorithm. They are not interchangeable.
            triggers.add(outcome.exit_reason or "scf_not_converged")
            if outcome.step_limit and outcome.n_ionic_steps >= outcome.step_limit:
                triggers.add("ionic_step_limit")
            elif outcome.exit_reason.startswith("finished without"):
                # The parser's phrasing for "exited cleanly, never reached the
                # force criterion, and did not hit NSW either" -- which is an SCF
                # problem rather than an ionic one.
                triggers.add("scf_not_converged")

        for rule in self.recipe.stages[step].retry:
            if rule.get("when") in triggers:
                return rule
        return None

    def run(self, store: Store) -> StageReport:            # pragma: no cover
        raise AssertionError("dft is a submitted stage; the driver calls claim/build")


def _max_attempts(stage) -> int:
    """One attempt per ladder rung, plus the original."""
    return len(stage.retry) + 1


def _decode_remedy(raw) -> dict:
    """The ladder rung a previous cycle chose.

    Returns `{set: {...}, resources: {...}, remedy: str}`. All three parts are
    carried: a rung that raises `--mem` is as much a change as one that raises
    NSW, and dropping it here would leave the retry re-running the identical job
    at the identical memory -- which costs the same again and looks like
    diligence.

    Tolerates the two older shapes: the INCAR overrides stored alone, and the
    `{set, remedy}` pair written before resources were carried.
    """
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    if "set" in value or "remedy" in value or "resources" in value:
        return {"set": value.get("set") or {},
                "resources": value.get("resources") or {},
                "remedy": value.get("remedy") or ""}
    return {"set": value, "resources": {}, "remedy": ""}


def _archive_previous(directory: Path, attempt: int) -> Path | None:
    """Move a finished run's files into `attempt-N/`, and say where they went.

    Returns the most recent archive whether or not this call created it: a
    directory whose outputs were archived by an earlier cycle still has a
    previous attempt to resume from, and returning None there is how the resume
    quietly turned back into a restart.
    """
    import shutil

    directory = Path(directory)
    if (directory / "OUTCAR").is_file():
        target = directory / f"attempt-{max(attempt - 1, 0)}"
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            for entry in sorted(directory.iterdir()):
                if entry.is_dir() or entry.name.startswith("attempt-"):
                    continue
                shutil.move(str(entry), str(target / entry.name))
        return target
    return _latest_archive(directory)


def _latest_archive(directory: Path) -> Path | None:
    """The highest-numbered `attempt-N/` in `directory`, or None."""
    archives = []
    for entry in Path(directory).glob("attempt-*"):
        if not entry.is_dir():
            continue
        suffix = entry.name.split("-", 1)[1]
        if suffix.isdigit():
            archives.append((int(suffix), entry))
    return max(archives)[1] if archives else None


def _archived_grid(archived: Path):
    """The k-point grid a previous attempt actually ran with, or None.

    `inputs.json` first, because that is what the attempt resolved; `KPOINTS`
    second, because a directory adopted from elsewhere has one and no manifest.
    Returning None means the resume derives its grid as before -- a silent
    change of sampling is the failure this exists to prevent, so it is better
    to have no answer than a guessed one.
    """
    from ..dft.vasp.kpoints import KpointGrid

    archived = Path(archived)
    manifest = archived / "inputs.json"
    if manifest.is_file():
        try:
            kpoints = json.loads(manifest.read_text())["kpoints"]
            return KpointGrid(a=int(kpoints["a"]), b=int(kpoints["b"]),
                              c=int(kpoints["c"]),
                              scheme=str(kpoints.get("scheme", "explicit")),
                              gamma=bool(kpoints.get("gamma", True)))
        except Exception:
            pass

    kpoints_file = archived / "KPOINTS"
    if kpoints_file.is_file():
        try:
            lines = kpoints_file.read_text().splitlines()
            # comment / 0 / Gamma|Monkhorst-Pack / a b c
            if len(lines) >= 4 and int(lines[1].split()[0]) == 0:
                a, b, c = (int(v) for v in lines[3].split()[:3])
                gamma = lines[2].strip().lower().startswith("g")
                return KpointGrid(a=a, b=b, c=c, scheme="explicit", gamma=gamma,
                                  comment=lines[0].strip())
        except Exception:
            pass
    return None


def _read_contcar(path: Path):
    """The relaxed geometry a previous attempt reached, or None.

    A CONTCAR that is absent or empty is the normal outcome of a job that died
    before its first ionic step, and resuming from nothing is not a resume --
    the caller falls back to the structure in the database.
    """
    import ase.io

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return ase.io.read(str(path), format="vasp")
    except Exception:
        return None


def _entered_dft(store: Store) -> int:
    """How many structures the campaign has already committed to DFT.

    Everything that reached `dft_done`, everything past step 0, and everything
    that failed with a DFT reason. A structure that failed still spent its
    core-hours, so it counts against the ceiling.
    """
    seen = set()
    for state in (StructureState.dft_done.value, StructureState.selected.value,
                  StructureState.dft_queued.value, StructureState.dft_running.value,
                  StructureState.failed.value):
        for row in store.structures(state=state):
            kv = row.key_value_pairs
            if int(kv.get(STEP_KEY, 0)) > 0 or state == StructureState.dft_done.value \
                    or kv.get("dft_fail_reason"):
                seen.add(int(row.id))
    return len(seen)


def _hours(walltime: str) -> float:
    """`[D-]HH:MM:SS` as hours."""
    days, _, rest = walltime.partition("-")
    if not rest:
        days, rest = "0", walltime
    parts = [float(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.append(0.0)
    return float(days) * 24 + parts[0] + parts[1] / 60 + parts[2] / 3600


def _campaign_file(cfg) -> Path:
    """The campaign.yaml an ABSOLUTE path, for a script that runs elsewhere.

    `cfg.campaign_path` is whatever the user typed. `csp run` defaults `-c` to
    the bare string `campaign.yaml`, so running from inside the campaign folder
    -- the normal thing to do -- stores a relative path that is correct only for
    that shell.

    A combined job then fails at the hand-off. The script `cd`s into the run
    directory before preparing the static from `relax/CONTCAR`, so by the time
    the worker resolves `campaign.yaml` the working directory is
    `.../dft/runs/0002-Ce2Pd2Ge2Sb2-b_Ce2Ge2Pd2Sb2` and the file is not there:

        relax: finished
        static: preparing inputs from relax/CONTCAR
        ConfigError: campaign file not found: campaign.yaml
        static: could not prepare inputs

    The relax is complete and correct, and its result is unreachable until the
    structure is resubmitted. Absolute here, always.

    The old fallback was `work_dir / "campaign.yaml"`, which never existed --
    `campaign.yaml` lives in the campaign folder, and `work_dir` is scratch. It
    was also unreachable, because it was guarded by `hasattr`, and
    `campaign_path` is a declared field that is always present and merely
    sometimes None. So a None went straight through and rendered as the literal
    string `None`.
    """
    configured = getattr(cfg, "campaign_path", None)
    if configured:
        return Path(configured).resolve()
    return (cfg.base_dir / "campaign.yaml").resolve()


def _larger(key: str, a, b):
    """The more generous of two resource values for `key`.

    Walltime and memory are strings with units, so "8:00:00" vs "12:00:00" and
    "32G" vs "64G" cannot be compared as text -- "8:00:00" > "12:00:00"
    lexically, which would pick the SMALLER one and is exactly the mistake this
    function exists to prevent. Anything not recognised falls back to the newer
    value, which is the previous behaviour.
    """
    if a is None:
        return b
    if b is None:
        return a
    if key == "time":
        return a if _seconds(a) >= _seconds(b) else b
    if key == "mem":
        return a if _megabytes(a) >= _megabytes(b) else b
    if key in ("ntasks", "cpus_per_task", "nodes"):
        try:
            return max(int(a), int(b))
        except (TypeError, ValueError):
            return b
    return b


def _format_walltime(seconds: int) -> str:
    """Seconds back to `D-HH:MM:SS`, the form SLURM accepts."""
    seconds = max(0, int(seconds))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, sec = divmod(rest, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _seconds(value) -> int:
    """`D-HH:MM:SS`, `HH:MM:SS`, `MM:SS` or a bare minute count, as seconds."""
    text = str(value).strip()
    days = 0
    if "-" in text:
        d, _, text = text.partition("-")
        try:
            days = int(d or 0)
        except ValueError:
            return -1
    try:
        parts = [int(p or 0) for p in text.split(":")] if text else [0]
    except ValueError:
        # Not a walltime. Sorting unparseable input ahead of a real value would
        # silently cap a job; rank it lowest so the real value always wins.
        return -1
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + sec


def _megabytes(value) -> float:
    """`32G`, `512M`, `1T` or a bare number (SLURM reads bare as MB)."""
    text = str(value).strip().upper().rstrip("B")
    unit = text[-1] if text and text[-1] in "KMGT" else ""
    number = text[:-1] if unit else text
    try:
        n = float(number)
    except ValueError:
        return 0.0
    return n * {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024, "": 1.0}[unit]


def _batch_tag(items: list[WorkItem]) -> str:
    """A name for the submission that cannot be mistaken for a directory.

    It used to be `items[0].key` -- the first structure in the batch, chosen by
    nothing in particular. One sbatch covers an array, so that name then stood
    for work it had nothing to do with:

        array named 'dft-78-static', task 1 ran dft-69-relax

    and ten hours went into reading a directory that had finished, because the
    name in `squeue` looked like an answer.

    A single-task submission keeps the directory name: there the name IS the
    answer, and nothing is gained by hiding it. Anything larger is named for
    what it is -- `dft-x5-relax-4a1f`, `dft-x2-1r+1s-77bc` -- so the `x<N>`
    announces an array and sends the reader to the manifest instead of guessing.
    The 4-char digest keeps two batches with the same shape from overwriting
    each other's `.tasks.json`, which is the file that answers the question.
    """
    if len(items) == 1:
        return items[0].key
    steps = Counter(str(i.payload.get("step_name") or "?") for i in items)
    if len(steps) == 1:
        shape = next(iter(steps))
    else:
        shape = "+".join(f"{n}{name[0]}" for name, n in sorted(steps.items()))
    digest = hashlib.sha1("|".join(sorted(i.key for i in items)).encode()).hexdigest()[:4]
    return f"dft-x{len(items)}-{shape}-{digest}"
