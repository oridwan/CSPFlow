"""Preflight checks.

`csp doctor` exists so that everything which can be known before a job is
submitted is known before a job is submitted.  Its hard failures are the ones
that would otherwise surface as a wrong number rather than an error: an
unresolvable POTCAR, a functional label that misdescribes what is on disk, two
4f conventions in one campaign.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .config.loader import ResolvedConfig
from .config.schema import RARE_EARTHS, Machine
from .db.store import Store, _filesystem_type, journal_mode_for, wal_lag
from .dft.vasp import potcar as pc

Status = Literal["ok", "warn", "fail", "fixed", "skip"]

_MARK = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "fixed": "FIX ", "skip": "--  "}


@dataclass
class Check:
    name: str
    status: Status
    detail: str = ""
    rows: list[str] = field(default_factory=list)

    def render(self) -> str:
        head = f"[{_MARK[self.status]}] {self.name}"
        if self.detail:
            head += f": {self.detail}"
        return "\n".join([head, *(f"        {r}" for r in self.rows)])


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, *checks: Check) -> None:
        self.checks.extend(checks)

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    def render(self) -> str:
        body = "\n".join(c.render() for c in self.checks)
        n_fail = sum(c.status == "fail" for c in self.checks)
        n_warn = sum(c.status == "warn" for c in self.checks)
        verdict = (
            f"\n{n_fail} failure(s), {n_warn} warning(s)."
            if n_fail or n_warn
            else "\nAll checks passed."
        )
        return body + "\n" + verdict


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------


def check_potcar_layout(machine: Machine, *, fix: bool = False) -> Check:
    """The symlink tree that makes a functional label mean what it says.

    Without it, `functional: PBE_64` fails and `PBE_54` succeeds by flat-layout
    fallback -- so the recorded provenance says PBE_54 for what are in fact
    VASP 6.4 potentials.
    """
    sources = {
        "PBE_64": "/projects/mmi/cspflow-shared/potcars/VASP6.4/potpaw_PBE",
        "PBE_52": "/projects/mmi/cspflow-shared/potcars/VASP5.2/potpaw_PBE",
    }
    sources = {k: v for k, v in sources.items() if k in machine.potcar_dirs}
    if not sources:
        return Check("POTCAR layout", "skip", "machine profile defines no potcar_dirs")
    actions = pc.ensure_pmg_layout(machine, sources, create=fix)
    worst: Status = "ok"
    for status, _ in actions:
        if status == "fail":
            worst = "fail"
        elif status in ("warn", "fixed") and worst == "ok":
            worst = status  # type: ignore[assignment]
    detail = {
        "ok": "labels match what is on disk",
        "fixed": "symlink tree created",
        "warn": "see below",
        "fail": "run `csp doctor --fix` to create the symlink tree",
    }[worst]
    return Check("POTCAR layout", worst, detail,
                 [f"{_MARK[s]} {m}" for s, m in actions])


def check_potcars(cfg: ResolvedConfig, elements: list[str]) -> list[Check]:
    """Resolve, hash and sanity-check every POTCAR the campaign needs."""
    dft = cfg.campaign.dft
    try:
        infos, errors = pc.resolve_all(
            elements, cfg.machine, tree=dft.potcar.tree,
            f_treatment=dft.rare_earth.f_treatment,
            overrides=dft.potcar.overrides,
        )
    except pc.PotcarError as exc:
        return [Check("POTCAR resolution", "fail", str(exc))]

    rows = [f"{'element':<8} {'symbol':<8} {'TITEL':<26} {'ZVAL':>6} {'ENMAX':>8}  hash"]
    for i in infos:
        rows.append(
            f"{i.element:<8} {i.symbol:<8} {i.titel:<26} {i.zval:6.1f} {i.enmax:8.3f}  {i.short_hash}"
        )
    for err in errors:
        rows.append(f"FAIL {err}")

    checks = [
        Check(
            f"POTCAR resolution ({dft.potcar.tree}, f_treatment={dft.rare_earth.f_treatment.value})",
            "fail" if errors else "ok",
            f"{len(errors)} unresolved" if errors else f"{len(infos)} resolved",
            rows,
        )
    ]

    # A vacuous pass is worse than no answer: if nothing resolved, or there is
    # only one rare earth, there is no convention to be consistent about.
    rare = [i for i in infos if i.element in RARE_EARTHS]
    if not infos:
        checks.append(Check("4f convention", "skip", "no POTCARs resolved"))
    elif len(rare) < 2:
        checks.append(Check("4f convention", "skip",
                            f"{len(rare)} rare earth(s) in this campaign -- nothing to compare"))
    else:
        try:
            pc.assert_one_f_convention(infos)
            checks.append(Check(
                "4f convention", "ok",
                f"one convention across {len(rare)} rare earths: "
                + ", ".join(f"{i.element}({i.symbol}, ZVAL {i.zval:g})" for i in rare)))
        except pc.PotcarError as exc:
            checks.append(Check("4f convention", "fail", str(exc)))

    if infos:
        default_encut = pc.max_enmax(infos)
        encut = _campaign_encut(cfg)
        rows = [
            f"VASP would default ENCUT to max(ENMAX) = {default_encut:.3f} eV for THIS "
            f"composition set",
            "that default moves with composition, so a hull built on it compares "
            "incomparable numbers",
        ]
        if encut is None:
            checks.append(Check("ENCUT", "warn",
                                "no ENCUT in the resolved recipe -- it must be explicit", rows))
        elif encut < default_encut:
            checks.append(Check("ENCUT", "warn",
                                f"ENCUT={encut:g} eV is below max(ENMAX)={default_encut:.1f} eV",
                                rows))
        else:
            checks.append(Check("ENCUT", "ok", f"explicit at {encut:g} eV "
                                               f"(>= max ENMAX {default_encut:.1f} eV)"))
    return checks


def _campaign_encut(cfg: ResolvedConfig) -> float | None:
    """ENCUT as the campaign will actually write it.

    It lives in the *recipe*, which is where the shipped `magnets.yaml` sets it
    to 520 eV; `incar_overrides` only wins where a campaign overrides it.
    Reading the overrides alone made this check warn "no ENCUT in the resolved
    recipe" for every correctly configured campaign -- which is the worst
    outcome for a guard, because it teaches the user to ignore the one warning
    that exists to stop a composition-dependent cutoff.
    """
    override = cfg.campaign.dft.incar_overrides.get("ENCUT")
    if override is not None:
        return float(override)

    try:
        from .dft.recipe import load_recipe

        recipe = load_recipe(cfg.campaign.dft.recipe, cfg.base_dir)
    except Exception:
        return None
    # The lowest ENCUT any step would use: a static step at a lower cutoff than
    # the relax that fed it is the case worth warning about.
    values = [float(stage.incar["ENCUT"]) for stage in recipe.stages
              if stage.incar.get("ENCUT") is not None]
    return min(values) if values else None


def available_modules() -> set[str] | None:
    """Every modulefile this cluster offers, or None if modules are unavailable.

    `module` is a shell function, so it is invoked through a login shell.
    `-t` gives one name per line; directory headers end in ':' and a default is
    marked with a '(default)' suffix, both of which are stripped.
    """
    # Environment Modules writes the terse listing to **stderr** and exits 0,
    # so both streams are read. Reading stdout alone gets an empty string and a
    # clean exit code, which is indistinguishable from "this cluster has no
    # modules" -- and would have skipped the check that exists to catch exactly
    # the failure that motivated it.
    try:
        proc = subprocess.run(["bash", "-lc", "module -t avail"],
                              capture_output=True, text=True, timeout=30.0)
    except (OSError, subprocess.SubprocessError):
        return None
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    names = set()
    for line in out.splitlines():
        line = line.strip()
        if not line or line.endswith(":"):
            continue
        names.add(line.replace("(default)", "").strip())
    return names or None


def check_modules(machine: Machine) -> Check:
    """Every module the profile loads must exist on this cluster.

    `module load` of a missing modulefile writes an error to stderr and still
    exits 0.  Under the `set -e` every generated script uses, the job therefore
    dies at whatever command needed it, with a message about that command rather
    than about the module.

    Found the hard way: `cuda/11.8` was in this profile because every script
    under /projects/mmi/shuo loads it, and the cluster now offers only 12.4,
    12.8 and 13.2.  The first live GPU submission failed in zero seconds with
    `Unable to locate a modulefile for 'cuda/11.8'` and nothing else.
    """
    wanted = {role: list(mods) for role, mods in machine.modules.items() if mods}
    if not wanted:
        return Check("modules", "ok", "the profile loads no modules")

    available = available_modules()
    if available is None:
        return Check("modules", "skip", "`module` is not usable from here")

    rows, worst = [], "ok"
    for role, mods in sorted(wanted.items()):
        for name in mods:
            base = name.split("/")[0]
            if name in available:
                rows.append(f"{role:<4} {name:<24} present")
            elif any(m.split("/")[0] == base for m in available):
                alternatives = sorted(m for m in available if m.split("/")[0] == base)
                rows.append(f"{role:<4} {name:<24} MISSING -- this cluster has "
                            f"{', '.join(alternatives[:4])}")
                worst = "fail"
            else:
                rows.append(f"{role:<4} {name:<24} MISSING -- no {base} module at all")
                worst = "fail"
    return Check("modules", worst,
                 "a missing module still exits 0, so the job dies later and elsewhere",
                 rows)


def check_env_paths(machine: Machine) -> Check:
    """Every profile env value that looks like a path must exist.

    One of them is load-bearing and fails unreadably: without
    `I_MPI_PMI_LIBRARY`, Intel MPI under `srun --mpi=pmi2` aborts in
    `PMPI_Init` with a wall of `mpi/pmi2: request not begin with 'cmd='`, which
    names neither the variable nor the library. Four VASP jobs died in three
    seconds each on the first live submission.
    """
    paths = {k: str(v) for k, v in machine.env.items() if str(v).startswith("/")}
    if not paths:
        return Check("environment", "ok", "no path-valued environment settings")

    rows, worst = [], "ok"
    for key, value in sorted(paths.items()):
        if Path(value).exists():
            rows.append(f"{key:<20} {value}")
        else:
            rows.append(f"{key:<20} {value}   MISSING")
            worst = "fail"
    return Check("environment", worst,
                 "a missing library here fails inside MPI, not at the export", rows)


def check_vasp(machine: Machine) -> Check:
    path = machine.codes.vasp_std
    if not path:
        return Check("VASP binary", "skip", "no vasp_std in machine profile")
    resolved = shutil.which(path) or (path if Path(path).is_file() else None)
    if resolved is None:
        return Check("VASP binary", "fail", f"{path} not found")
    if not os.access(resolved, os.X_OK):
        return Check("VASP binary", "fail", f"{resolved} is not executable")
    return Check("VASP binary", "ok", resolved)


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def check_scheduler(machine: Machine) -> Check:
    if machine.scheduler == "local":
        return Check("scheduler", "ok", "local (no queue)")
    if shutil.which("sinfo") is None:
        return Check("scheduler", "warn", "slurm configured but sinfo not on PATH")

    out = _run(["sinfo", "-h", "-o", "%P|%a|%l|%D"])
    if out is None:
        return Check("scheduler", "warn", "sinfo failed")
    live = {}
    for line in out.strip().splitlines():
        name, avail, walltime, nodes = (line.split("|") + ["", "", ""])[:4]
        live[name.rstrip("*")] = (avail, walltime, nodes)

    rows, worst = [], "ok"
    for role, part in machine.partitions.items():
        wanted = [p for p in part.name.split(",") if p]
        if not wanted:
            rows.append(f"{role:<10} (site default)")
            continue
        missing = [p for p in wanted if p not in live]
        if missing:
            rows.append(f"{role:<10} {part.name}  MISSING: {missing}")
            worst = "fail"
        else:
            info = ", ".join(
                f"{p}[{live[p][0]}, {live[p][2]} nodes, max {live[p][1]}]" for p in wanted
            )
            rows.append(f"{role:<10} {info}")
    return Check("partitions", worst, f"{len(live)} partitions visible", rows)


def check_qos_limits() -> Check:
    """Live submit limits, so the driver throttles against reality."""
    user = os.environ.get("USER", "")
    out = _run(["sacctmgr", "-nP", "show", "assoc", f"user={user}",
                "format=Account,Partition,QOS,MaxSubmitJobs,MaxJobs,GrpTRES"])
    if out is None:
        return Check("QOS limits", "warn",
                     "sacctmgr unavailable; the driver will fall back to profile limits")
    rows = [line for line in out.strip().splitlines() if line.strip()]
    if not rows:
        return Check("QOS limits", "warn", f"no associations reported for user {user!r}")
    return Check("QOS limits", "ok", f"{len(rows)} association(s)",
                 ["Account|Partition|QOS|MaxSubmit|MaxJobs|GrpTRES", *rows[:12]])


def check_throttle(cfg: ResolvedConfig) -> Check:
    """Compare the configured submission limits against the live QOS.

    The plan asks for this explicitly (pipeline.md sec.4.5), and the reason is
    a real event: `redo-new-ter-mag` submitted **2,362 individual jobs** against
    a `MaxSubmitJobsPU` of 2,048. Submitting past the cap does not queue the
    excess -- it is refused, and by then the driver believes it has dispatched
    work it has not.

    The subtler number is the concurrency one. `MaxTRESPU cpu=768` at
    `ntasks=16` means **48 VASP jobs can run at once**, no matter how many are
    queued. A `max_concurrent_tasks` above that is not faster; it just makes the
    `%N` array throttle a lie.
    """
    from .scheduler import compute_throttle, for_machine

    machine = cfg.machine
    if machine.scheduler != "slurm":
        return Check("throttle", "skip", f"scheduler is {machine.scheduler!r}")

    scheduler = for_machine(machine)
    rows, worst = [], "ok"

    # Only `dft` has configured throttles; the GPU stages are bound by the
    # gres cap and are reported so the ceiling is visible, not because the user
    # chose a number that could be wrong.
    dft_cfg = cfg.campaign.dft

    # The RECIPE's rank count, not the machine default. DFT resources live per
    # recipe step, so the machine default (16 here) was reported while the
    # recipe asks for 64 -- and every core number derived from it was wrong by
    # 4x. Same fault as D135 fixed in the driver; this copy was missed.
    ntasks = machine.defaults.ntasks
    try:
        from .stages.dft_stage import DftStage

        ntasks = int(DftStage(cfg).resource_hint()[0]) or ntasks
    except Exception:                                   # noqa: BLE001
        pass

    plans = [
        ("dft", "cpu", dft_cfg.max_in_flight, dft_cfg.max_concurrent_tasks,
         ntasks, 0),
    ]
    for stage, res in (("generate", cfg.campaign.generate.resources if cfg.campaign.generate else None),
                       ("screen", cfg.campaign.screen.resources)):
        if res is None:
            continue
        plans.append((stage, res.role, dft_cfg.max_in_flight, dft_cfg.max_in_flight,
                      res.ntasks or 1, res.gpus or 0))

    for stage, role, want_flight, want_concurrent, ntasks, gpus in plans:
        try:
            limits = scheduler.limits(role)
        except Exception as exc:                        # pragma: no cover - live only
            rows.append(f"{stage:<8} could not read limits for role {role!r}: {exc}")
            worst = "warn"
            continue

        throttle = compute_throttle(
            requested_in_flight=want_flight, requested_concurrent=want_concurrent,
            ntasks=ntasks, gpus_per_job=gpus, limits=limits,
            # Only the CPU/DFT plan is core-capped; the GPU stages are bound by
            # the gres cap. Reporting a cap that does not apply would be as
            # misleading as omitting one that does.
            max_cores=dft_cfg.max_cores if stage == "dft" else None,
        )
        clamped = (throttle.in_flight < want_flight
                   or throttle.concurrent_tasks < want_concurrent)
        cores = f"  = {throttle.in_flight * ntasks} cores" if stage == "dft" else ""
        rows.append(
            f"{stage:<8} role={role:<4} asked {want_flight}/{want_concurrent} "
            f"at {ntasks} ranks  -> {throttle.render()}{cores}"
        )
        if limits.source == "live":
            rows.append(f"         (no QOS found for role {role!r}; limits unknown)")
            worst = "warn" if worst == "ok" else worst
        elif clamped:
            worst = "warn" if worst == "ok" else worst

    return Check("throttle", worst,
                 "configured limits vs. what the QOS actually permits", rows)


def check_optional_deps(cfg: ResolvedConfig) -> Check:
    """Engines are only needed if the campaign actually uses them."""
    wanted: dict[str, str] = {}
    if cfg.campaign.needs_generation and cfg.campaign.generate:
        wanted[cfg.campaign.generate.engine] = "generation"
    wanted[cfg.campaign.screen.mlip] = "screening"
    wanted["pymatgen"] = "structure handling"

    rows, worst = [], "ok"
    for mod, why in sorted(wanted.items()):
        found = importlib.util.find_spec(mod) is not None
        rows.append(f"{mod:<12} {'present' if found else 'MISSING':<8} ({why})")
        if not found:
            worst = "warn"
    return Check("engines", worst,
                 "missing engines block only the stages that use them", rows)


def check_generator(cfg: ResolvedConfig, *, fix: bool = False) -> Check:
    """The generation checkpoint, checked from the login node.

    Everything here is answerable without a GPU except the GPU itself, so the
    device check is downgraded to a note: `csp doctor` normally runs on a login
    node, where CUDA is legitimately absent. The array task repeats the same
    preflight where it matters and refuses there.

    The checks that do matter here are the ones nobody would think to make: that
    the path is the run directory rather than the .ckpt file, and that Hydra
    recorded `config_name: csp`. An unconditional checkpoint accepts
    `--target_compositions`, ignores it, and returns structures of some other
    chemistry -- which costs the whole job before anything notices.
    """
    if not cfg.campaign.needs_generation or cfg.campaign.generate is None:
        return Check("generator", "skip", "no source mode generates structures")

    rows_fixed: list[str] = []
    try:
        from .generators import for_config
        engine = for_config(cfg.campaign.generate)
    except Exception as exc:                                     # pragma: no cover
        return Check("generator", "fail", f"could not build the generator: {exc}")

    if fix:
        repair = getattr(engine, "repair_installation", None)
        if repair is not None:
            for done in repair():
                rows_fixed.append(done)

    rows = [f"engine {cfg.campaign.generate.engine}", f"model  {engine.model}"]
    rows.extend(rows_fixed)
    conf = getattr(engine, "sampling_conf", None)
    if conf is not None:
        directory, vendored = conf()
        rows.append(f"config {directory}" + ("  (vendored -- the installed "
                                             "mattergen ships none)" if vendored else ""))
    device = engine.device()
    rows.append(f"device {device}" + ("  (login node -- the job checks again)"
                                      if device == "cpu" else ""))

    problems = [p for p in engine.preflight() if "CUDA" not in p]
    worst = "ok"
    for problem in problems:
        rows.append(problem)
        worst = ("fail" if ("no checkpoints/" in problem or "not a directory" in problem
                            or "is missing" in problem) else "warn")
    return Check("generator", worst, "checkpoint layout and CSP training", rows)


def check_paths(cfg: ResolvedConfig) -> Check:
    rows, worst = [], "ok"
    workdir = cfg.work_dir
    parent = workdir if workdir.exists() else workdir.parent
    if not parent.exists():
        rows.append(f"workdir {workdir} -- parent {parent} does not exist")
        worst = "fail"
    elif not os.access(parent, os.W_OK):
        rows.append(f"workdir {workdir} -- {parent} is not writable")
        worst = "fail"
    else:
        rows.append(f"workdir {workdir} (writable)")
    if cfg.campaign.archive:
        arch = Path(cfg.campaign.archive)
        ap = arch if arch.exists() else arch.parent
        ok = ap.exists() and os.access(ap, os.W_OK)
        rows.append(f"archive {arch} ({'writable' if ok else 'NOT writable'})")
        if not ok:
            worst = "warn" if worst == "ok" else worst
    return Check("paths", worst, "", rows)


# --------------------------------------------------------------------------


def check_reference_recipe(cfg: ResolvedConfig) -> Check:
    """Is this campaign's DFT policy the same one the store's energies used?

    This is still the one rule that cannot be relaxed (D101): every energy on
    one hull must come from identical settings. Ours minus MP is +0.15 to
    +0.21 eV/atom while the selection threshold is 0.06, so a hull mixing two
    scales still builds, still looks reasonable, and ranks every compound
    wrongly.

    What changed on 2026-09-11 is the CONSEQUENCE of a mismatch, so the old
    message here was misleading. Reference energies are no longer looked up in
    a `computed/<recipe_id>/` cache -- they are read from the store folder
    (`reference/refstore.py`), so a mismatched campaign no longer reads an
    EMPTY reference set. It reads a FULL one computed under different settings,
    which is worse: a plausible hull on a mixed scale.

    The authority is the store's OWN settings.yaml -- the file that actually
    computed the energies -- not a copy beside the reference cache. Those two
    had drifted: the copy was two hours stale and missing five magnetism-table
    entries, which is what made this check fire on a campaign that had lifted
    the store's policy verbatim.
    """
    import os

    from .cli import store_settings_path

    # the store's own settings.yaml first: it is what computed the energies.
    # `store_root()` falls back to the shared default store, so a user who leaves
    # CSPFLOW_STORE unset -- as the install guide says to -- is compared against
    # the store they will actually read, not a stale copy (D154).
    from .reference.refstore import store_root

    candidates = [store_root() / "settings.yaml"]
    fallback = store_settings_path()
    if fallback is not None:
        candidates.append(fallback)
    store = next((c for c in candidates if c and Path(c).is_file()), None)
    if store is None:
        return Check("reference recipe", "skip",
                     "no store settings.yaml found ($CSPFLOW_STORE unset?)")
    try:
        from .cli import _dft_and_recipe
        from .dft.recipe import load_recipe
        from .reference.computed import recipe_id

        mine = recipe_id(cfg.campaign.dft, load_recipe(cfg.campaign.dft.recipe, cfg.base_dir))
        s_dft, s_recipe, _ = _dft_and_recipe(Path(store), None)
        theirs = recipe_id(s_dft, s_recipe)
    except Exception as exc:                                  # noqa: BLE001
        return Check("reference recipe", "skip", f"could not compare: {exc}")

    if mine == theirs:
        return Check("reference recipe", "ok",
                     f"matches the store ({mine[:16]}) -- the hull is on one scale",
                     [f"store  {store}"])
    return Check(
        "reference recipe", "warn",
        f"campaign {mine[:16]} != store {theirs[:16]}: candidate and reference "
        f"energies would land on DIFFERENT scales",
        [f"store   {store}",
         "effect  the hull still builds and looks correct, and ranks wrongly (D101)",
         "fix     copy the store's `dft:` block into campaign.yaml verbatim"])




# Elements whose 3d shell carries a moment our DFT actually computes.  The 4f
# moment is NOT in the calculation when `f_treatment: frozen` -- the rare earth
# runs on its `_3` POTCAR with the f electrons in the core -- so a cell whose
# only magnetic species is a rare earth has no computed moment at all.
MAGNETIC_3D = {"Fe", "Co", "Ni", "Mn", "Cr"}


def check_magnetism_is_computed(cfg: ResolvedConfig,
                                elements: list[str] | None) -> Check:
    """If the moment is the target property, is it being computed or assumed?

    `rare_earth.f_treatment: frozen` puts the 4f electrons in the POTCAR core.
    That is the right choice for ENERGIES -- it gives one convention across the
    series, which MP does not have (D109) -- but it means VASP never computes a
    4f moment.  `reconstruct_ms: true` adds a nominal spin-only value back at
    reporting time, which is a bookkeeping constant per rare-earth atom, not a
    measurement.

    So in a cell whose only magnetic species is the rare earth, the reported
    moment is entirely reconstructed and varies only with how many rare-earth
    atoms the cell holds.  Ranking such candidates by moment ranks them by
    composition.

    Measured in the store on 443 Ce phases run at exactly these settings:

        Ce with an Fe/Co/Ni/Mn/Cr partner   median 0.001, 90th pct 1.17 uB/atom
        Ce with no magnetic 3d              87% below 0.05 uB/atom

    The second row is the trap.  (ISPIN=2 was confirmed on 357 of 358 sampled
    static runs, so those zeros are the physics, not a missing tag.)
    """
    rare_earth = getattr(cfg.campaign.dft, "rare_earth", None)
    treatment = getattr(getattr(rare_earth, "f_treatment", None), "value", None)
    if treatment != "frozen":
        return Check("magnetism", "ok",
                     f"f_treatment={treatment or 'unset'} -- the 4f moment is in "
                     f"the calculation")

    if not elements:
        return Check("magnetism", "skip",
                     "no elements yet -- run `csp source` first, or pass --elements")

    present = set(elements)
    computed = sorted(present & MAGNETIC_3D)
    rares = sorted(e for e in present if e in RARE_EARTHS)
    if computed:
        return Check(
            "magnetism", "ok",
            f"f_treatment=frozen; the computed moment lives on {', '.join(computed)}",
            rows=[f"the {', '.join(rares)} 4f moment is reconstructed, not computed "
                  f"-- report m_dft_raw and m_s_reconstructed separately"] if rares else [])

    if not rares:
        return Check("magnetism", "ok", "no magnetic species in this campaign")

    return Check(
        "magnetism", "warn",
        f"f_treatment=frozen and the only magnetic species is {', '.join(rares)}",
        rows=[
            "VASP computes NO 4f moment here: the f electrons are in the POTCAR",
            "core. Every reported moment is reconstruct_ms bookkeeping, a",
            f"constant per {rares[0]} atom -- so ranking these candidates by",
            "moment ranks them by composition, not by physics.",
            "Measured on 246 store phases at these settings: 87% come out below",
            "0.05 uB/atom, median exactly 0.000.",
            "If the moment is the target property, set",
            "  dft.rare_earth.f_treatment: valence   (+ LDAU on the 4f)",
            "which is a DIFFERENT DFT policy: its own store, its own hull.",
        ])

def check_database_readability(cfg: ResolvedConfig) -> Check:
    """Can this node read this campaign's database, and is what it reads current?

    Both halves matter and only the first is obvious. A campaign database on a
    network filesystem must not be in WAL mode: the `-shm` index that makes a
    WAL readable is coherent only within one host, so a second node reads the
    main file as of the last checkpoint. That read succeeds. It simply answers
    with an old campaign, which is the failure mode you cannot see.

    An existing campaign cannot be converted while its driver holds the file --
    SQLite declines the mode change, and before `_set_journal_mode` it declined
    it in silence. So this reports rather than fixes, and names the one moment
    the conversion is possible.
    """
    db = cfg.campaign_db
    if not db.is_file():
        return Check("campaign database", "skip", f"none yet at {db}")

    fs = _filesystem_type(db)
    want = journal_mode_for(db)
    rows, worst = [], "ok"

    store = Store(db)
    try:
        _ = store.sql
        got = store.journal_mode or "unknown"
    except Exception as exc:                                   # noqa: BLE001
        return Check("campaign database", "fail", f"{db} will not open here: {exc}")
    finally:
        try:
            store.close()
        except Exception:                                      # noqa: BLE001
            pass

    if got.upper() != want.upper():
        worst = "fail"
        rows.append(
            f"journal mode is {got.upper()} on {fs or 'this filesystem'}, which needs "
            f"{want}. A WAL on a network filesystem is not readable from another "
            f"node and is how this database gets corrupted."
        )
        rows.append(
            "SQLite will not change the mode while another connection holds the "
            "file, so convert it when no driver is running:"
        )
        rows.append(f"    sqlite3 {db} 'PRAGMA journal_mode=TRUNCATE;'")
    else:
        rows.append(f"journal mode {got.upper()} on {fs or 'unknown fs'}")

    lag = wal_lag(db)
    if lag is not None:
        worst = "fail" if worst == "ok" else worst
        rows.append(
            f"this node's copy is {lag / 60:.0f} min behind its write-ahead log, so "
            f"reads here are stale. `csp status` uses status.json instead."
        )

    return Check("campaign database", worst, "\n".join(rows))


def run(cfg: ResolvedConfig, *, elements: list[str] | None = None, fix: bool = False) -> Report:
    report = Report()
    report.add(Check("config", "ok",
                     f"{cfg.campaign.name}  hash {cfg.short_hash}  machine {cfg.machine_path.name}"))
    report.add(check_paths(cfg))
    report.add(check_database_readability(cfg))
    report.add(check_potcar_layout(cfg.machine, fix=fix))
    if elements:
        report.add(*check_potcars(cfg, elements))
    else:
        report.add(Check("POTCAR resolution", "skip",
                         "no elements yet -- run `csp source` first, or pass --elements"))
    report.add(check_modules(cfg.machine))
    report.add(check_env_paths(cfg.machine))
    report.add(check_vasp(cfg.machine))
    report.add(check_scheduler(cfg.machine))
    report.add(check_qos_limits())
    report.add(check_throttle(cfg))
    report.add(check_optional_deps(cfg))
    report.add(check_generator(cfg, fix=fix))
    report.add(check_reference_recipe(cfg))
    report.add(check_magnetism_is_computed(cfg, elements))
    return report
