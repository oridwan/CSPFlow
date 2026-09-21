"""The `csp` command line."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml

from . import __version__, doctor as doctor_mod
from .config.loader import ConfigError, load_campaign, resolve_machine_path
from .config.schema import Campaign
from .db.store import Store, StoreError, is_transient_error, wal_lag
from .ingest import IngestError, ingest_campaign
from .legacy import REGISTRY as LEGACY_REGISTRY, LegacyError, adopt as adopt_legacy
from .driver import STAGE_ORDER, Driver, DriverError, DriverOptions
from .scheduler import for_machine
from .source import SourceError, expand_all, write_plan
from .stages import IMPLEMENTED, PLANNED, build_registry
from .worker import WorkerError, run_generate_task, run_screen_task
from . import templates

app = typer.Typer(
    name="csp",
    help="High-throughput crystal structure prediction and first-principles discovery.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")
reference_app = typer.Typer(
    help="Build and inspect the shared MP reference cache.", no_args_is_help=True)
app.add_typer(reference_app, name="reference")

DEFAULT_CAMPAIGN = "campaign.yaml"

CampaignOpt = Annotated[
    Path, typer.Option("--campaign", "-c", help="campaign YAML file")
]
SetOpt = Annotated[
    Optional[list[str]], typer.Option("--set", "-s", help="override, e.g. -s filter.e_above_hull_max=0.2")
]


def _die(message: str) -> None:
    typer.secho(str(message), fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _find_campaign(path: Path) -> Path:
    """Locate the campaign file, walking up from the working directory.

    A campaign is a folder, so every command should work from inside it or from
    any folder beneath it -- the way git works anywhere in a checkout. Only the
    unqualified default name is searched for; if the user named a file, that is
    the file, and a missing one is an error rather than a hunt.
    """
    if path.is_file() or path.is_absolute() or str(path) != DEFAULT_CAMPAIGN:
        return path
    here = Path.cwd()
    for folder in here.parents:
        candidate = folder / DEFAULT_CAMPAIGN
        if candidate.is_file():
            typer.secho(f"# campaign: {candidate}", fg=typer.colors.BLUE, err=True)
            return candidate
    return path


def _load(campaign: Path, sets: list[str] | None, machine: str | None = None):
    try:
        return load_campaign(_find_campaign(campaign), sets=sets, machine=machine)
    except ConfigError as exc:
        _die(str(exc))


def _link_results(campaign_file: Path, workdir: Path) -> None:
    """Put a `results` symlink beside campaign.yaml pointing at the workdir.

    Needed only when `workdir` is an ABSOLUTE path somewhere else. The default
    is `workdir: results`, which already lives inside the campaign folder, and
    a symlink to a directory that is the link's own target is nonsense -- so
    this does nothing in the normal case.
    """
    link = _find_campaign(campaign_file).resolve().parent / "results"
    try:
        if workdir.resolve() == link.resolve():
            return                      # the link would point at itself
    except OSError:
        pass
    if link.exists() or link.is_symlink():
        return
    try:
        link.symlink_to(workdir, target_is_directory=True)
    except OSError:
        pass          # a read-only or exotic filesystem is not a reason to stop


def _db_path(cfg) -> Path:
    return cfg.campaign_db


# --------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(f"cspflow {__version__}")


_REFERENCE_QUESTION = """
Recompute the Materials Project reference phases with THIS campaign's own DFT
settings?

  yes  the hull is built from $CSPFLOW_STORE at your settings, so candidates and
       reference sit on ONE energy scale and e_above_hull means what it says.
       Costs DFT for any chemical system the store does not already hold.

  no   nothing is recomputed. Candidates are filtered on the MatterSim hull
       (whole, one scale, free) and their DFT energies are reported raw.
       dft_e_above_hull stays EMPTY where the store does not cover the system.

Either way our DFT is never placed on MP's hull: measured over 3,345 store
phases, elemental Ce alone is +1.17 eV/atom ours-minus-MP, and a per-element
correction still leaves 42-103 meV/atom against a 60 meV/atom threshold."""


def _ask_reference_mode(answer: Optional[bool]) -> str:
    """Turn the one init question into a `reference.mode` value.

    Asked interactively when the flag is absent and there is a terminal to ask
    on; non-interactive callers (tests, scripts, CI) get `recompute`, which is
    the safe answer because it is the only one that yields a hull on a single
    scale.
    """
    if answer is None:
        if sys.stdin.isatty():
            typer.echo(_REFERENCE_QUESTION)
            answer = typer.confirm("\nRecompute the reference phases?", default=True)
        else:
            answer = True
    return "recompute" if answer else "mp_energies"


@app.command(no_args_is_help=False)
def init(
    kind: Annotated[Optional[str], typer.Argument(
        metavar="TYPE", show_default=False,
        help="1 chemical_space | 2 composition_list | 3 structure_list")] = None,
    name: Annotated[Optional[str], typer.Argument(
        metavar="NAME", show_default=False,
        help="campaign name; also the folder it is created in")] = None,
    directory: Annotated[Optional[Path], typer.Option("--dir", "-d", help="where to create it (default: ./<name>)")] = None,
    machine: Annotated[str, typer.Option("--machine", "-m", help="shipped profile to copy: orion, generic_slurm, local")] = "orion",
    recipe: Annotated[str, typer.Option("--recipe", help="shipped DFT recipe to copy")] = "magnets",
    here: Annotated[bool, typer.Option("--here", help="use the current folder instead of creating one")] = False,
    minimal: Annotated[bool, typer.Option("--minimal", help="campaign.yaml without comments, referring to the shipped profile and recipe")] = False,
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="write just the campaign file, at this path")] = None,
    force: Annotated[bool, typer.Option("--force", help="overwrite existing files")] = False,
    recompute_reference: Annotated[Optional[bool], typer.Option(
        "--recompute-reference/--no-recompute-reference",
        help="recompute the MP reference phases with THIS campaign's DFT settings "
             "(default: ask)")] = None,
) -> None:
    """Create a campaign folder from one of the three example campaigns.

    \b
      TYPE  source mode        the question it answers
      1     chemical_space     search a region of the periodic table
      2     composition_list   I know which formulas I want
      3     structure_list     I already have the structures

    Example: csp init 1 my-sweep. The folder gets campaign.yaml (every setting
    of the example for that type, with its alternatives beside it, but none of
    its chemistry), editable copies of machine.yaml and recipe.yaml, and an
    empty inputs/. Fill in your elements, formulas or structures; `csp source`
    says what is still missing. TYPE also accepts the mode name.
    """
    from .dft.recipe import RECIPE_DIR

    ctype, name = _resolve_campaign_type(kind, name)
    ref_mode = _ask_reference_mode(recompute_reference)

    try:
        templates.example_dir(ctype)
    except FileNotFoundError as exc:
        _die(str(exc))
    starters = templates.starter_inputs(ctype)

    if out is not None:                     # single-file mode
        if out.exists() and not force:
            _die(f"{out} already exists (use --force to overwrite)")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(templates.campaign_yaml(ctype, name=name, machine=machine,
                                               recipe=recipe, minimal=minimal,
                                               reference_mode=ref_mode))
        typer.echo(f"wrote {out}  ({ctype.label})")
        if ctype.number in (2, 3):
            typer.echo(f"It reads inputs/ beside itself; see examples/{ctype.folder}/inputs "
                       f"for the format.")
        typer.echo("Next: edit it, then run `csp doctor`.")
        return

    root = Path.cwd() if here else (directory or Path(name))
    written: list[Path] = []

    def _write(rel: str, data: str | bytes) -> None:
        path = root / rel
        if path.exists() and not force:
            _die(f"{path} already exists (use --force to overwrite)")
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data)
        written.append(path)

    if minimal:
        _write(DEFAULT_CAMPAIGN, templates.campaign_yaml(
            ctype, name=name, machine=machine, recipe=recipe, minimal=True,
            reference_mode=ref_mode))
    else:
        try:
            machine_src = resolve_machine_path(machine)
        except ConfigError as exc:
            _die(str(exc))
        recipe_src = RECIPE_DIR / f"{recipe}.yaml"
        if not recipe_src.is_file():
            shipped = sorted(f.stem for f in RECIPE_DIR.glob("*.yaml"))
            _die(f"unknown recipe {recipe!r}; shipped: {shipped}")

        _write(DEFAULT_CAMPAIGN, templates.campaign_yaml(
            ctype, name=name, reference_mode=ref_mode))
        _write("machine.yaml", templates.machine_copy(machine_src, name=name))
        _write("recipe.yaml", templates.recipe_copy(recipe_src, name=name))
        _write("README.md", templates.workspace_readme(ctype, name=name))
        _write("inputs/README.md", templates.inputs_readme(ctype))
    for rel, text in starters.items():
        _write(rel, text)
    if ctype.number == 3:
        (root / "inputs" / "seeds").mkdir(parents=True, exist_ok=True)

    typer.secho(f"\ncampaign {name} ({ctype.label}) in {root}/", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  \"{ctype.question}\" -- you give {ctype.what_you_give}\n")
    for path in written:
        rel = str(path.relative_to(root))
        typer.echo(f"  {rel:<24} {_BLURB.get(rel, '')}")
    if ctype.number == 3:
        typer.echo(f"  {'inputs/seeds/':<24} empty -- put your POSCAR/CIF files here")
    typer.echo("\nEdit first:")
    for line in ctype.edit_first:
        typer.echo(f"  {line}")
    typer.echo("\nNext:")
    if not here:
        typer.echo(f"  cd {root}")
    typer.echo("  csp doctor                # check the machine before submitting anything")
    typer.echo("  csp source --dry-run      # what would be searched")


def _resolve_campaign_type(kind: Optional[str], name: Optional[str]):
    """Work out (type, name) from `csp init TYPE NAME`, or ask.

    `csp init my-campaign` -- the form every older note uses -- is still
    accepted: a single argument that is not a type is taken as the name, and the
    type is asked for on a terminal. With no terminal to ask on (scripts, CI)
    it stops and prints the menu, rather than guessing a type: the three
    scaffold different funnels, and a wrong guess is a campaign that runs.
    """
    ctype = templates.parse_type(kind)
    if kind is not None and ctype is None:
        if name is not None:
            _menu_exit(f"unknown campaign type {kind!r}.", f"csp init <1|2|3> {name}")
        kind, name = None, kind               # `csp init my-campaign`
    if name is None:
        if ctype is not None:
            _menu_exit("a campaign needs a name.", f"csp init {ctype.number} <name>")
        _menu_exit("which kind of campaign, and what is it called?", "csp init <1|2|3> <name>")
    if ctype is None:
        if not sys.stdin.isatty():
            _menu_exit(f"which kind of campaign is {name!r}?", f"csp init <1|2|3> {name}")
        typer.echo(templates.type_menu() + "\n")
        import click
        choice = typer.prompt("Campaign type", type=click.Choice(["1", "2", "3"]))
        ctype = templates.parse_type(choice)
    return ctype, name


def _menu_exit(message: str, usage: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    typer.echo("\n" + templates.type_menu() + f"\n\nusage: {usage}", err=True)
    raise typer.Exit(code=1)


_BLURB = {
    DEFAULT_CAMPAIGN: "what to search, and how hard",
    "machine.yaml": "partitions, walltime, modules, VASP, POTCARs",
    "recipe.yaml": "the DFT ladder: INCAR, k-points, resources",
    "inputs/README.md": "what goes in inputs/ for this type",
    "inputs/compositions.csv": "empty -- your formulas, one per line",
    "README.md": "what to edit, what to run",
}


@config_app.command("show")
def config_show(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    origins: Annotated[bool, typer.Option("--origins", help="show which layer supplied each value")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="emit JSON")] = False,
) -> None:
    """Print the fully resolved configuration."""
    cfg = _load(campaign, set_)
    payload = cfg.campaign.model_dump(mode="json")
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
    typer.echo(f"# machine:     {cfg.machine_path}")
    typer.echo(f"# config_hash: {cfg.config_hash}")
    if origins:
        typer.echo("\n# where each value came from:")
        for path in sorted(cfg.origins):
            typer.echo(f"#   {path:<48} {cfg.origins[path]}")


@config_app.command("defaults")
def config_defaults() -> None:
    """Print the schema's default values.

    Derived from the pydantic models rather than a checked-in file, so the
    defaults shown here cannot drift from the defaults actually applied.
    """
    skeleton = {
        "name": "<required>",
        "machine": "<required>",
        "workdir": "<required>",
        "source": "<required: a list of source entries>",
    }
    for field, info in Campaign.model_fields.items():
        if field in skeleton:
            continue
        if info.default_factory is not None:
            value = info.default_factory()
            skeleton[field] = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        else:
            skeleton[field] = info.default
    typer.echo(yaml.safe_dump(skeleton, sort_keys=False, default_flow_style=False))


@app.command()
def doctor(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    machine: Annotated[Optional[str], typer.Option("--machine", "-m")] = None,
    elements: Annotated[Optional[str], typer.Option("--elements", help="comma-separated, e.g. Sm,Fe,Ti")] = None,
    fix: Annotated[bool, typer.Option("--fix", help="create the POTCAR symlink layout")] = False,
) -> None:
    """Check everything that can be known before a job is submitted.

    Exits non-zero on any hard failure, so it can gate a submission script.
    """
    cfg = _load(campaign, set_, machine)

    els: list[str] = []
    if elements:
        els = [e.strip() for e in elements.split(",") if e.strip()]
    else:
        db = _db_path(cfg)
        if db.is_file():
            try:
                with Store.open(db) as store:
                    els = sorted({e for cs in store.chemsystems() for e in cs.split("-")})
            except StoreError:
                els = []

    report = doctor_mod.run(cfg, elements=els, fix=fix)
    typer.echo(report.render())
    if report.failed:
        raise typer.Exit(code=1)


@app.command()
def ingest(
    root: Annotated[Path, typer.Argument(help="existing campaign directory to import")],
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="only this many formula directories")] = None,
    source_name: Annotated[str, typer.Option("--source-name")] = "ingested",
    db: Annotated[Optional[Path], typer.Option("--db", help="write here instead of the campaign workdir")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Import an existing campaign directory into a cspflow database.

    `--limit` caps the number of formula directories, which is what makes this
    quick to sanity-check: a handful exercises every code path that all of them
    would.
    """
    cfg = _load(campaign, set_)
    target = db or _db_path(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    _link_results(Path(campaign), cfg.work_dir)

    store = Store.open(target) if target.is_file() else Store.create(
        target, campaign=cfg.campaign.name, config_hash=cfg.config_hash
    )
    seen = 0

    def progress(formula: str) -> None:
        nonlocal seen
        seen += 1
        if not quiet:
            typer.echo(f"  [{seen}] {formula}", err=True)

    with store:
        try:
            stats = ingest_campaign(root, store, limit=limit, source_name=source_name,
                                    progress=None if quiet else progress)
        except IngestError as exc:
            _die(str(exc))
    typer.echo(stats.render())
    typer.echo(f"\nwrote {target}")


@app.command()
def adopt(
    flow: Annotated[str, typer.Argument(help=f"which legacy campaign: {', '.join(LEGACY_REGISTRY)}, or 'all'")],
    dest: Annotated[Path, typer.Option("--dest", help="where the campaign directories go")] = Path("/scratch/$USER/cspflow_results"),
    staging: Annotated[Optional[Path], typer.Option("--staging", help="build here first, then move (SQLite on NFS commits ~15x slower)")] = Path("/tmp"),
    machine: Annotated[str, typer.Option("--machine", help="machine profile to record in campaign.yaml")] = "orion",
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="only this many structures, for a quick check")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Adopt a finished legacy campaign into cspflow's own layout.

    Reads all six of a legacy flow's result artefacts -- not just its VASP
    directories, which is what `csp ingest` does -- and writes the campaign
    cspflow would have written had it run the work itself.  Nothing is
    recomputed and nothing in the source directory is touched.

    The VASP outputs are 549 GB and stay where they are; `dft_dir` and
    `job.workdir` point at them, and `campaign_meta['legacy.root']` records the
    prefix so a move needs one UPDATE rather than a re-adoption.
    """
    import os

    dest = Path(os.path.expandvars(str(dest)))
    names = list(LEGACY_REGISTRY) if flow == "all" else [flow]
    unknown = [n for n in names if n not in LEGACY_REGISTRY]
    if unknown:
        _die(f"unknown legacy campaign(s) {unknown}. Known: {', '.join(LEGACY_REGISTRY)}")

    def say(message: str) -> None:
        if not quiet:
            typer.echo(message, err=True)

    for name in names:
        say(f"{name}  <-  {LEGACY_REGISTRY[name].root}")
        try:
            stats = adopt_legacy(LEGACY_REGISTRY[name], dest, staging=staging,
                          limit=limit, machine=machine,
                          progress=None if quiet else say)
        except LegacyError as exc:
            _die(str(exc))
        typer.echo(stats.render())
        typer.echo(f"\nwrote {dest / name}\n")


@app.command()
def source(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="print the plan, write nothing")] = False,
    db: Annotated[Optional[Path], typer.Option("--db", help="write here instead of the campaign workdir")] = None,
    gpu_seconds: Annotated[float, typer.Option("--gpu-seconds", help="seconds per generated structure, for the time estimate")] = 0.0,
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="commit only the first N composition rows")] = None,
) -> None:
    """Stage 0 -- expand `source:` into composition and seed rows.

    Enumeration happens entirely in memory and the plan is printed before
    anything is written, so `--dry-run` runs the identical code path and the
    estimate you approve is produced by the code that then does the work.

    `--limit` truncates the committed rows, which is what makes a large chemical
    space quick to sanity-check: the full enumeration is still reported, only the
    commit is capped.
    """
    cfg = _load(campaign, set_)
    base = cfg.base_dir

    try:
        plan = expand_all(cfg.campaign, base)
    except SourceError as exc:
        _die(str(exc))

    typer.echo(plan.render(gpu_seconds_per_structure=gpu_seconds or None))

    if dry_run:
        typer.echo("\n--dry-run: nothing written")
        return

    if limit is not None:
        for result in plan.results:
            result.compositions = result.compositions[:limit]

    target = db or _db_path(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    _link_results(Path(campaign), cfg.work_dir)
    store = Store.open(target) if target.is_file() else Store.create(
        target, campaign=cfg.campaign.name, config_hash=cfg.config_hash
    )
    with store:
        store.add_provenance(
            config_hash=cfg.config_hash,
            machine=str(cfg.machine_path),
            resolved_config=cfg.campaign.model_dump(mode="json"),
        )
        stats = write_plan(plan, store)
    typer.echo("")
    typer.echo(stats.render())
    typer.echo(f"wrote {target}")


@app.command()
def run(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    through: Annotated[Optional[str], typer.Option("--through", help="run stages up to and including this one")] = None,
    from_: Annotated[Optional[str], typer.Option("--from", help="run stages from this one on")] = None,
    only: Annotated[Optional[str], typer.Option("--only", help="a single stage")] = None,
    watch: Annotated[bool, typer.Option("--watch", help="keep cycling instead of stopping when idle")] = False,
    interval: Annotated[int, typer.Option("--interval", help="seconds between cycles")] = 300,
    max_cycles: Annotated[Optional[int], typer.Option("--max-cycles")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="report what would be submitted, claim nothing")] = False,
    stop_when_done: Annotated[bool, typer.Option(
        "--stop-when-done/--watch-forever",
        help="under --watch, exit once the slice is finished "
             "(nothing pending, nothing in flight). On by default.")] = True,
    idle_cycles: Annotated[int, typer.Option(
        "--idle-cycles", min=1,
        help="consecutive idle cycles required before --stop-when-done fires")] = 2,
    db: Annotated[Optional[Path], typer.Option("--db")] = None,
) -> None:
    """The driver loop: reconcile what is in flight, submit what fits, repeat.

    `--through reference` is Phase A -- cheap, run to completion for every
    composition. `--from filter --watch` is Phase B -- expensive, streamed under
    `dft.max_cores`. The two are the same loop over a different stage slice.

    `--watch` keeps cycling so that finished jobs are collected rather than left
    on disk. It used to keep cycling FOREVER, which meant a driver submitted as
    a batch job held its allocation for days after its slice was finished.
    `--stop-when-done` (the default) exits once nothing is pending and nothing
    is in flight for `--idle-cycles` cycles running. `--watch-forever` restores
    the old behaviour for a campaign you intend to keep feeding.
    """
    cfg = _load(campaign, set_)
    base = cfg.base_dir
    target = db or _db_path(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    _link_results(Path(campaign), cfg.work_dir)

    wanted = _stage_slice(through, from_, only)
    runnable = [name for name in wanted if name in IMPLEMENTED]
    skipped = [name for name in wanted if name not in IMPLEMENTED]
    if skipped:
        typer.secho(
            "not yet implemented, skipped: "
            + ", ".join(f"{n} ({PLANNED.get(n, '?')})" for n in skipped),
            fg=typer.colors.YELLOW, err=True,
        )
    if not runnable:
        _die("none of the requested stages are implemented yet")

    store = Store.open(target) if target.is_file() else Store.create(
        target, campaign=cfg.campaign.name, config_hash=cfg.config_hash
    )
    options = DriverOptions(interval=interval, max_cycles=max_cycles,
                            stages=runnable, dry_run=dry_run,
                            idle_exit_after=idle_cycles if stop_when_done else 0)
    with store:
        try:
            driver = Driver(cfg, store, for_machine(cfg.machine, dry_run=dry_run),
                            build_registry(cfg, base), options,
                            emit=_emit_now)
            driver.run(watch=watch)
        except sqlite3.DatabaseError as exc:
            # TWO conditions reach here and they want OPPOSITE responses.
            #
            # `sqlite3.OperationalError` is a SUBCLASS of `sqlite3.DatabaseError`,
            # so this one handler used to catch both a genuinely corrupt file
            # (`file is not a database`, from the WAL-on-NFS bug -- see
            # store.journal_mode_for) and a transient `disk I/O error` from the
            # storage blinking. It gave the corruption remedy for both. On
            # 2026-09-19 that told the user to move aside an intact database
            # holding the bookkeeping for 1,005 finished DFT jobs (D148).
            if is_transient_error(exc):
                _die(f"campaign database temporarily unreachable: {exc}\n"
                     f"  file    {target}\n"
                     f"  The driver already retried this and the storage did not\n"
                     f"  come back in time. This is NOT corruption: the file is\n"
                     f"  intact and must NOT be moved aside.\n"
                     f"  Confirm with:  sqlite3 {target} 'PRAGMA integrity_check'\n"
                     f"  Then fix the filesystem and re-run `csp run` -- the next\n"
                     f"  cycle reconciles whatever finished in the meantime.")
            _die(f"campaign database unreadable: {exc}\n"
                 f"  file    {target}\n"
                 f"  Check whether it is really damaged before doing anything:\n"
                 f"      sqlite3 {target} 'PRAGMA integrity_check'\n"
                 f"  Only the STRUCTURE rows are derived. `csp source` rebuilds\n"
                 f"  those from {base / 'inputs'},\n"
                 f"  but job, relaxation, hull, reference_entry and property rows\n"
                 f"  come from finished DFT and CANNOT be rebuilt from inputs.\n"
                 f"  Moving this file aside discards them. Run\n"
                 f"  scripts/campaign_audit.py first to see what is at stake.\n"
                 f"  If it sits on a network filesystem, never touch it from the\n"
                 f"  login node while a driver holds it on a compute node.")
        except (DriverError, SourceError) as exc:
            _die(str(exc))
    typer.echo(f"\ndatabase {target}")


def _stage_slice(through: str | None, from_: str | None, only: str | None) -> list[str]:
    """Turn --through/--from/--only into a contiguous slice of the funnel."""
    for name in (through, from_, only):
        if name is not None and name not in STAGE_ORDER:
            _die(f"unknown stage {name!r}; the funnel is {STAGE_ORDER}")
    if only:
        return [only]
    start = STAGE_ORDER.index(from_) if from_ else 0
    stop = STAGE_ORDER.index(through) + 1 if through else len(STAGE_ORDER)
    if stop <= start:
        _die(f"--from {from_} comes after --through {through}; that selects nothing")
    return STAGE_ORDER[start:stop]


@app.command()
def recipe(
    name: Annotated[Optional[str], typer.Argument(help="shipped recipe name or a path; default: this campaign's")] = None,
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
) -> None:
    """Print a recipe fully resolved -- every tag literal, nothing deferred.

    With no argument this prints the recipe the campaign in this folder would
    actually use, which is the question being asked most of the time. This is
    what `inherit:` copying buys: there is no value here whose meaning requires
    knowing pymatgen to predict.
    """
    from .dft.recipe import RecipeError, load_recipe, validate_recipe
    from .dft.vasp.incar import render_incar

    base: Path | None = None
    if name is None:
        found = _find_campaign(campaign)
        if found.is_file():
            cfg = _load(found, None)
            name, base = cfg.campaign.dft.recipe, cfg.base_dir
        else:
            name = "magnets"

    try:
        loaded = load_recipe(name, base)
        warnings = validate_recipe(loaded)
    except RecipeError as exc:
        _die(str(exc))

    typer.echo(f"recipe {loaded.name}  ({loaded.source})")
    for stage in loaded.stages:
        typer.echo(f"\n=== {stage.name} ===")
        typer.echo(render_incar(stage.incar).rstrip())
        typer.echo(f"kpoints   {stage.kpoints.as_dict()}")
        typer.echo(f"resources {stage.resources}")
        if stage.retry:
            typer.echo(f"retry     {[r.get('when') for r in stage.retry]}")
    typer.echo("\n# MAGMOM, NBANDS, LMAXMIX, SYSTEM and LDAU* are computed per")
    typer.echo("# structure and written into the emitted INCAR.")
    for w in warnings:
        typer.secho(f"warning: {w}", fg=typer.colors.YELLOW, err=True)


@app.command("screen-worker", hidden=True)
def screen_worker(
    manifest: Annotated[Path, typer.Option("--manifest", help="written by the screen stage")],
    task_id: Annotated[Optional[int], typer.Option("--task-id", help="defaults to $SLURM_ARRAY_TASK_ID")] = None,
) -> None:
    """Relax one chunk of a screen manifest.  Run by array tasks, not by hand.

    Hidden because it is an implementation detail of the screen stage: it takes
    a manifest the stage wrote and writes a results file the driver reads. It is
    exposed as a command only because that is how a SLURM array task invokes it.
    """
    try:
        out = run_screen_task(manifest, task_id)
    except WorkerError as exc:
        _die(str(exc))
    typer.echo(str(out))


@app.command()
def report(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="directory for report.html and candidates.csv")] = None,
    limit: Annotated[int, typer.Option("--limit", help="rows in the table; 0 for all")] = 2000,
    detail: Annotated[int, typer.Option("--detail", help="structures with a 3D card and per-site moments; 0 for none")] = 25,
    no_hull: Annotated[bool, typer.Option("--no-hull", help="skip the hull plots (they rebuild the phase diagrams)")] = False,
    jsmol_url: Annotated[Optional[str], typer.Option("--jsmol-url", help="serve the 3D view with a JSmol at this URL instead of the built-in viewer")] = None,
) -> None:
    """Write the candidate table and a self-contained HTML dashboard.

    This is the direct answer to "seeing results is difficult": one file that
    opens in a browser with no server, no build step and no network, plus the
    same table as CSV for anything downstream.

    The page carries, in order: the funnel, both convex hulls drawn (the MLIP
    one and the DFT one, never merged), the magnetisation figures of merit
    (M, V, M/V and mu0*M in tesla), the full candidate table, and one card per
    top candidate holding the relaxed cell in 3D and every ion's moment.

    `--detail` is the size knob: each card embeds a geometry, so 25 cards is a
    file of a few hundred kB and 500 would be tens of MB. `--jsmol-url` swaps
    the built-in viewer for JSmol, for the web portal where JSmol is served;
    the default viewer needs no network and is what keeps the file emailable.
    """
    from .report.candidates import candidate_rows, funnel, write_csv
    from .report.html import write as write_html
    from .report.structures import write as write_structures

    cfg = _load(campaign, set_)
    db = _db_path(cfg)
    if not db.is_file():
        _die(f"no campaign database at {db}. Run `csp init` and then a stage.")

    directory = out or (cfg.base_dir / "report")
    with Store.open(db) as store:
        rows = candidate_rows(store, limit=limit or None)
        csv_path = write_csv(rows, directory / "candidates.csv")
        html_path = write_html(store, directory / "report.html",
                               title=cfg.campaign.name, limit=limit or None,
                               detail_cards=detail, hulls=not no_hull,
                               jsmol_url=jsmol_url,
                               campaign_dir=cfg.base_dir)
        counts = funnel(store)
        # Every structure, not only the candidates. `candidates.csv` answers
        # "what came out"; this answers "what happened to everything", which is
        # the question asked about the rows that are NOT in the other file.
        structures_path = write_structures(store, directory / "structures.csv")

    typer.echo(counts.render())
    typer.echo("")
    typer.echo(f"{len(rows)} candidate(s)")
    typer.echo(f"table  {csv_path}")
    typer.echo(f"all    {structures_path}")
    typer.echo(f"report {html_path}")
    if not rows:
        typer.secho("no candidates yet -- nothing has reached dft_done",
                    fg=typer.colors.YELLOW, err=True)


@app.command("generate-worker", hidden=True)
def generate_worker(
    manifest: Annotated[Path, typer.Option("--manifest", help="written by the generate stage")],
    task_id: Annotated[Optional[int], typer.Option("--task-id", help="defaults to $SLURM_ARRAY_TASK_ID")] = None,
) -> None:
    """Generate structures for one chunk of a generate manifest.

    Hidden for the same reason as `screen-worker`: it exists so that a SLURM
    array task has something to invoke, and takes all of its instructions from
    the manifest the stage wrote.
    """
    try:
        out = run_generate_task(manifest, task_id)
    except WorkerError as exc:
        _die(str(exc))
    typer.echo(str(out))


@app.command()
def status(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    why: Annotated[Optional[int], typer.Option("--why", help="full life history of one structure id")] = None,
    source: Annotated[str, typer.Option(
        "--source", help="where to read progress from: auto, db, or snapshot")] = "auto",
) -> None:
    """Show campaign progress, from any node, while the campaign is running.

    The database cannot be relied on for this. It is SQLite on a network
    filesystem, and a running driver keeps its recent writes in a `-wal` whose
    index lives in shared memory -- coherent within one host and nowhere else.
    Read from a second node it gives one of two wrong answers: an exception, or
    a table of numbers that quietly stopped being true at the last checkpoint.

    So `auto` reads the database only when it is BOTH readable and current, and
    otherwise reads `status.json`, which the driver rewrites every cycle by
    atomic rename. That file is written by the one process guaranteed to be on
    the right node, so it is the more trustworthy source whenever the two
    disagree -- not a degraded fallback.
    """
    if source not in ("auto", "db", "snapshot"):
        _die(f"--source must be auto, db or snapshot (got {source!r})")

    cfg = _load(campaign, set_)
    db = _db_path(cfg)
    snap = db.parent / "status.json"

    if source == "snapshot":
        if not snap.is_file():
            _die(f"no driver snapshot at {snap}. It appears once the driver "
                 f"completes a cycle; use --source db to read the database.")
        _status_from_snapshot(snap, db, note="")
        return

    lag = _stale_lag(db, snap)

    # `--why` is a per-structure history. A snapshot carries counts, so there is
    # nothing to fall back TO -- say that plainly rather than printing a summary
    # the user did not ask for.
    if why is not None:
        if lag is not None and source == "auto":
            _die(f"--why needs the database. {_unreadable_here(db).capitalize()}, "
                 f"so any history it returns would be incomplete.\n"
                 f"  Run it on the node holding the campaign (`squeue` shows which), "
                 f"or pass --source db to read the stale copy anyway.")
        try:
            with Store.open(db) as store:
                _print_history(store, why)
        except StoreError as exc:
            _die(str(exc))
        return

    if source == "auto" and lag is not None:
        if snap.is_file():
            _status_from_snapshot(
                snap, db,
                note=_unreadable_here(db) + " -- it was not used")
            return
        _die(f"{_unreadable_here(db)}, and there is no driver snapshot beside it "
             f"to read instead.\n"
             f"  Read it on the node holding the campaign (`squeue` shows which), "
             f"or pass --source db to read the stale copy anyway.")

    if not db.is_file():
        if snap.is_file() and source == "auto":
            _status_from_snapshot(snap, db, note=f"no database at {db}")
            return
        _die(f"no campaign database at {db}. Run `csp init` and then a stage.")

    try:
        store_cm = Store.open(db)
    except StoreError as exc:
        # The database is unreadable from here. That is expected on a second
        # node and is not a reason to fail if the driver left a snapshot.
        if source == "auto" and snap.is_file():
            _status_from_snapshot(snap, db, note="the database did not read back here")
            return
        _die(str(exc))
        return

    with store_cm as store:
        s = store.summary()
        gen = store.generation_yield()
        typer.echo(f"campaign     {s['campaign']}")
        typer.echo(f"database     {db}")
        if lag is not None:
            typer.echo(f"             WARNING: {_ago(lag)} behind its write-ahead log")
        _render_summary(s, gen)


def _stale_lag(db: Path, snap: Path) -> float | None:
    """How far this node's view of `db` is behind, or None if it is current.

    A `-wal` newer than the main database is NOT by itself a problem. On the
    host running the driver that is simply what WAL mode looks like while work
    is happening, and reads there are perfectly correct -- SQLite merges the WAL
    for any connection that can see the `-shm`. The problem is being on a
    different host, where that shared memory does not reach and the main file is
    all you get.

    So the mtime gap is the symptom and the hostname is the discriminator. The
    driver records the host it runs on in `status.json` precisely because it is
    the one fact that decides whether a database read here means anything; when
    it matches, the database wins even with a large WAL behind it.

    Without a snapshot to compare against there is nothing to discriminate with,
    and the honest default is to distrust the read: a wrong number presented
    confidently is the failure this whole path exists to avoid, and `--source
    db` is one flag away for anyone who knows better.
    """
    if not db.is_file():
        return None
    lag = wal_lag(db)
    if lag is None:
        return None
    try:
        data = json.loads(snap.read_text())
        if data.get("host") and data["host"] == socket.gethostname():
            return None                     # this IS the node holding it
    except (OSError, ValueError):
        pass
    return lag


def _wal_bytes(db: Path) -> int:
    """Size of the write-ahead log, which is what a wrong-host reader misses.

    Deliberately used in preference to the mtime gap when explaining a fallback.
    The gap is not a measure of severity: SQLite auto-checkpoints roughly every
    1000 pages, so it collapses to near zero just after a checkpoint and grows
    until the next one, while the amount of campaign you cannot see from another
    host stays whatever is in the WAL. Reporting "2s behind" invites exactly the
    wrong inference -- that the database is nearly current and could be used --
    when the real situation is binary: this host can read the WAL or it cannot.
    """
    try:
        return Path(f"{db}-wal").stat().st_size
    except OSError:
        return 0


def _unreadable_here(db: Path) -> str:
    """Why this node must not trust `db`, said in terms of the cause."""
    size = _wal_bytes(db)
    held = f"its {size:,}-byte write-ahead log" if size else "its write-ahead log"
    return (f"{db.name} is in WAL mode and open on another node, so {held} "
            f"cannot be read from here")


def _ago(seconds: float) -> str:
    """A duration as a person would say it."""
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _status_from_snapshot(snap: Path, db: Path, *, note: str) -> None:
    """Render `status.json`, and be explicit that that is what it is.

    Every line says where the number came from and how old it is. A status
    command that silently swapped its source would be the same defect as the
    stale database it exists to avoid.
    """
    import json as _json

    try:
        data = _json.loads(snap.read_text())
    except (OSError, ValueError) as exc:
        _die(f"could not read the driver snapshot {snap}: {exc}")
        return

    age = time.time() - snap.stat().st_mtime
    typer.echo(f"campaign     {data.get('campaign', db.parent)}")
    typer.echo(f"source       driver snapshot {snap.name}, written {_ago(age)} ago"
               f" (cycle {data.get('cycle', '?')}, slurm {data.get('slurm_job') or '-'})")
    if note:
        typer.echo(f"             {note}")
    # A snapshot only stays current while the driver that writes it is alive.
    # Past a few cycles' silence the far likelier reading is that the driver
    # stopped, and a confident-looking table would be the wrong impression.
    if age > 3600:
        typer.echo(f"             WARNING: no cycle in {_ago(age)}"
                   f" -- check the driver is still running (squeue)")

    summary = data.get("summary")
    if summary:
        _render_summary(summary, data.get("generation") or {})
    else:
        # An older driver wrote this file before the summary was added to it.
        # Print what it does carry rather than nothing.
        if data.get("summary_error"):
            typer.echo(f"             summary unavailable: {data['summary_error']}")
        typer.echo(f"structures   {data.get('structures_total', '?')}")
        for state, n in sorted((data.get("structures") or {}).items()):
            typer.echo(f"    {state:<16} {n}")

    typer.echo("driver")
    typer.echo(f"    {'in flight':<16} {data.get('in_flight', '?')}")
    typer.echo(f"    {'core-hours':<16} {data.get('core_hours_spent', 0):,.0f} spent, "
               f"{data.get('core_hours_projected', 0):,.0f} projected")
    for st in data.get("stages") or []:
        if st.get("pending") or st.get("claimed") or st.get("submitted"):
            typer.echo(f"    {st['stage']:<16} {st.get('pending', 0)} pending, "
                       f"{st.get('claimed', 0)} claimed, {st.get('submitted', 0)} submitted")
            if st.get("note"):
                typer.echo(f"    {'':<16}   {st['note']}")


def _emit_now(msg: str) -> None:
    """Print a driver line and flush it at once.

    A driver runs as a batch job with stdout going to a file, where Python
    buffers in 8 KB blocks. A cycle line is ~300 bytes, so a slow cycle's
    report sat in the buffer for hours: the RE-magnets-CHGNet log showed
    nothing for nine hours while the database was being written the whole
    time, and only unbuffered warnings came through.
    """
    typer.echo(msg)
    sys.stdout.flush()


def _render_summary(s: dict, gen: dict) -> None:
    """The progress block, identical whichever source supplied the numbers."""
    typer.echo(f"compositions {s.get('compositions', '?')}  across "
               f"{s.get('chemsystems', '?')} chemical systems")
    # Generation yield is reported before the structure counts because a
    # shortfall here is invisible below it: the funnel narrows anyway, and
    # 40% fewer candidates entering it looks exactly like a smaller campaign.
    #
    # Only when something was actually asked for: a seeded campaign has
    # compositions in state `generated` with nothing requested, and
    # "0 of 0 requested (0.0%)" reads as a failure rather than as silence.
    if gen.get("requested"):
        pct = 100.0 * gen["produced"] / gen["requested"]
        line = (f"generated    {gen['produced']:,} of {gen['requested']:,} "
                f"requested ({pct:.1f}%)")
        if gen.get("short"):
            line += f"   <- {gen['short']} composition(s) short"
        typer.echo(line)
    typer.echo(f"structures   {s.get('structures', '?')}")
    for state, n in sorted((s.get("structures_by_state") or {}).items()):
        typer.echo(f"    {state:<16} {n}")
    typer.echo(f"reference    {s.get('reference_entries', 0)} MP entries")
    if s.get("jobs"):
        typer.echo("jobs")
        for state, n in sorted(s["jobs"].items()):
            typer.echo(f"    {state:<16} {n}")
        typer.echo(f"    {'core-hours':<16} {s.get('core_hours', 0):,.0f}")
    if s.get("relaxations"):
        # Reported separately from job state on purpose: a VASP run that
        # exits cleanly at the ionic step limit is `done` and NOT relaxed.
        # 61% of redo-new-ter-mag was exactly that.
        typer.echo("relaxations")
        for key, n in sorted(s["relaxations"].items()):
            flag = ""
            if "not converged" in key:
                flag = "   <- no converged relaxation, not usable as a relaxed geometry"
            elif key.endswith(":retried"):
                flag = "   <- failed once, converged on a later attempt (already counted above)"
            typer.echo(f"    {key:<24} {n}{flag}")


def _print_history(store: Store, sid: int) -> None:
    try:
        row = store.get_structure(sid)
    except StoreError as exc:
        _die(str(exc))
    typer.echo(f"structure {sid}: {row.formula}")
    for key, value in sorted(row.key_value_pairs.items()):
        typer.echo(f"    {key:<20} {value}")
    events = store.filter_events(sid)
    if events:
        typer.echo("  gates")
        for e in events:
            verdict = "pass" if e["passed"] else "FAIL"
            typer.echo(f"    {e['gate']:<20} {verdict:<5} value={e['value']} threshold={e['threshold']}")
    props = store.properties(sid)
    if props:
        typer.echo("  properties")
        for p in props:
            typer.echo(f"    {p['key']:<20} {p['value']}  ({p['source']})")
    jobs = [j for j in store.jobs() if j["structure_id"] == sid]
    if jobs:
        typer.echo("  jobs")
        for j in jobs:
            typer.echo(
                f"    {j['stage']}/{j['recipe_step'] or '-':<8} {j['state']:<9} "
                f"attempt {j['attempt']}  {j['exit_reason']}"
            )


# --------------------------------------------------------------------------
# reference -- the shared MP cache, and our own numbers in it
# --------------------------------------------------------------------------

SystemsArg = Annotated[
    Optional[list[str]],
    typer.Argument(help="chemical systems, e.g. Fe-Sm Fe-Sm-Ti; default: --from-results"),
]
FromResults = Annotated[
    Optional[Path],
    typer.Option("--from-results", help="results root whose family.json manifests name the systems"),
]
SystemsFile = Annotated[
    Optional[Path], typer.Option("--systems-file", help="one chemical system per line")
]


def _systems(explicit, from_results, systems_file) -> list[str]:
    """Where the system list comes from, in priority order."""
    out: list[str] = list(explicit or [])
    if systems_file:
        out += [line.strip() for line in Path(systems_file).read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    if from_results:
        for manifest in sorted(Path(from_results).glob("*/family.json")):
            for system in json.loads(manifest.read_text()).get("systems", []):
                if system.get("chemsys"):
                    out.append(system["chemsys"])
    if not out:
        # Fall back to the store's own scope, so `csp reference report` with no
        # arguments reports on what is actually there. `systems.txt` is what the
        # store was built for; the downloaded files are what it currently holds,
        # and the two differ while a fetch is only partly done.
        out = _systems_in_store()
    if not out:
        _die("no chemical systems given, and the reference store is empty. Pass them "
             "as arguments, --systems-file, or --from-results pointing at a results "
             "root -- or run `csp reference prefetch` first.")
    return sorted(set(out))


def _systems_in_store() -> list[str]:
    from .reference.mp import THERMO_GGA, reference_root

    root = reference_root()
    listed = root / "systems.txt"
    if listed.is_file():
        return [line.strip() for line in listed.read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    suffix = f"__{THERMO_GGA.replace('+', 'p')}.json"
    return [p.name[: -len(suffix)] for p in root.glob(f"*{suffix}")]


# The one file that defines the shared reference store's DFT policy.  It lives
# WITH the store rather than inside any campaign, because it outlives all of
# them and every campaign that reads the store has to match it.
STORE_SETTINGS_NAME = "store-settings.yaml"


def store_settings_path() -> Path | None:
    """The store's own settings file, or None if there isn't one.

    `$CSPFLOW_STORE_SETTINGS` wins if set; otherwise `store-settings.yaml` next
    to the reference cache.
    """
    override = os.environ.get("CSPFLOW_STORE_SETTINGS")
    if override:
        p = Path(override)
        return p if p.is_file() else None
    try:
        from .reference.mp import reference_root
        p = reference_root() / STORE_SETTINGS_NAME
    except Exception:                                         # noqa: BLE001
        return None
    return p if p.is_file() else None


def _dft_and_recipe(campaign: Path, set_: list[str] | None):
    """The DFT policy a reference build must match, and where it came from.

    Resolution order, and the reason for it: an energy is only comparable to the
    others in the store if it was computed under the same policy, so the DEFAULT
    has to be the store's own policy, not a guess.  Falling back to `Dft()` --
    which is what this did -- silently computes a different recipe id and starts
    a fresh, empty store beside the real one.  That had already happened once
    here before it was noticed: two directories, 2,762 records each, one of them
    holding no energies at all.

      1. `-c FILE`, when the file exists           (deliberate override)
      2. the store's own `store-settings.yaml`     (the normal case)
      3. built-in defaults, with a warning         (no store configured yet)
    """
    from .dft.recipe import load_recipe

    if Path(campaign).is_file():
        cfg = _load(campaign, set_)
        return cfg.campaign.dft, load_recipe(cfg.campaign.dft.recipe, cfg.base_dir), str(campaign)

    store = store_settings_path()
    if store is not None:
        cfg = _load(store, set_)
        return cfg.campaign.dft, load_recipe(cfg.campaign.dft.recipe, cfg.base_dir), f"{store} (store default)"

    from .config.schema import Dft

    typer.echo(
        f"warning: no {STORE_SETTINGS_NAME} found next to the reference cache, "
        f"falling back to built-in defaults. Any energy computed now lands under "
        f"a recipe id that the existing store does not share.", err=True)
    dft = Dft()
    return dft, load_recipe(dft.recipe), "built-in defaults"


MinAtoms = Annotated[int, typer.Option("--min-atoms", help="skip cells smaller than this; for splitting a build into size tiers")]
MaxAtoms = Annotated[int, typer.Option("--max-atoms", help="skip cells larger than this")]
MaxEHull = Annotated[float, typer.Option("--max-e-above-hull",
    help="skip phases this far above MP's own hull (eV/atom); 0 disables the cut")]


@reference_app.command("prefetch")
def reference_prefetch(
    chemsys: SystemsArg = None,
    from_results: FromResults = None,
    systems_file: SystemsFile = None,
    thermo_type: Annotated[str, typer.Option("--thermo-type")] = "GGA_GGA+U",
    no_structures: Annotated[bool, typer.Option("--no-structures", help="thermo entries only")] = False,
    refresh: Annotated[bool, typer.Option("--refresh", help="refetch even if cached")] = False,
) -> None:
    """Download MP thermo entries and geometries into the shared cache.

    Paid once per chemical system across every campaign, because the cache lives
    outside all of them.  Nothing else populates it: Stage 3 and Stage 4a both
    need a live campaign database, so without this the first campaign in a new
    system pays the network cost and no later one can run offline.
    """
    from .reference.mp import ReferenceError, fetch_chemsys, fetch_structures, reference_root

    systems = _systems(chemsys, from_results, systems_file)
    typer.echo(f"reference store  {reference_root()}")
    typer.echo(f"systems {len(systems)}  thermo_type {thermo_type}")
    entries = structures = failed = 0
    for n, system in enumerate(systems, 1):
        line = f"[{n:4d}/{len(systems)}] {system:14s}"
        try:
            result = fetch_chemsys(system, thermo_type=thermo_type,
                                   energy_scale="raw", refresh=refresh)
        except (ReferenceError, Exception) as exc:            # noqa: BLE001
            failed += 1
            typer.echo(f"{line} FAILED: {exc}", err=True)
            continue
        entries += len(result.entries)
        line += f" {len(result.entries):4d} entries {'(cache)' if result.from_cache else '(fetched)'}"
        if not no_structures:
            try:
                got = fetch_structures(system, refresh=refresh)
                structures += len(got)
                line += f"  {len(got):4d} structures"
            except (ReferenceError, Exception) as exc:        # noqa: BLE001
                failed += 1
                line += f"  FAILED structures: {exc}"
        typer.echo(line)
    typer.echo(f"\n{len(systems)} systems, {entries} thermo entries, "
               f"{structures} structures, {failed} failure(s)")


@reference_app.command("plan")
def reference_plan(
    chemsys: SystemsArg = None,
    from_results: FromResults = None,
    systems_file: SystemsFile = None,
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    min_atoms: MinAtoms = 0,
    max_atoms: MaxAtoms = 0,
    max_e_above_hull: MaxEHull = 0.5,
) -> None:
    """What recomputing the reference phases at our own settings would cost.

    Deduplicated by material id: one calculation per phase serves every hull the
    phase appears in, so this is the real number rather than the sum over
    systems.
    """
    from .reference.build import plan as build_plan

    systems = _systems(chemsys, from_results, systems_file)
    dft, recipe, where = _dft_and_recipe(campaign, set_)
    typer.echo(f"dft settings from {where}")
    try:
        typer.echo(build_plan(
            systems, dft=dft, recipe=recipe,
            min_atoms=min_atoms or None, max_atoms=max_atoms or None,
            max_e_above_hull=max_e_above_hull or None).render())
    except Exception as exc:                                  # noqa: BLE001
        _die(str(exc))


@reference_app.command("build")
def reference_build(
    dest: Annotated[Path, typer.Argument(help="where the reference campaign goes")],
    chemsys: SystemsArg = None,
    from_results: FromResults = None,
    systems_file: SystemsFile = None,
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    machine: Annotated[str, typer.Option("--machine")] = "orion",
    min_atoms: MinAtoms = 0,
    max_atoms: MaxAtoms = 0,
    max_e_above_hull: MaxEHull = 0.5,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="print the plan, write nothing")] = False,
) -> None:
    """Write a campaign that recomputes the MP reference phases at our settings.

    Writes and seeds only.  Nothing is submitted: run the printed `csp run`
    command when you want the jobs to start.
    """
    from .reference.build import seed, write_campaign

    systems = _systems(chemsys, from_results, systems_file)
    dft, recipe, where = _dft_and_recipe(campaign, set_)
    typer.echo(f"dft settings from {where}")
    cuts = dict(min_atoms=min_atoms or None, max_atoms=max_atoms or None,
                max_e_above_hull=max_e_above_hull or None)
    if dry_run:
        from .reference.build import plan as build_plan
        typer.echo(build_plan(systems, dft=dft, recipe=recipe, **cuts).render())
        return

    dest = Path(dest)
    try:
        dest, the_plan = write_campaign(dest, systems=systems, dft=dft, recipe=recipe,
                                        machine=machine, **cuts)
    except Exception as exc:                                  # noqa: BLE001
        _die(str(exc))
    typer.echo(the_plan.render())

    cfg = _load(dest / "campaign.yaml", None)
    target = _db_path(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    store = Store.open(target) if target.is_file() else Store.create(
        target, campaign=cfg.campaign.name, config_hash=cfg.config_hash)
    with store:
        added = seed(store, dest, plan_=the_plan)
    typer.echo(f"\nseeded {added} structure(s) into {target}")
    missing = dest / "missing.txt"
    if missing.is_file():
        typer.echo(f"note: {missing} lists phases with no MP geometry")
    typer.echo(f"\nnothing has been submitted. To start:\n"
               f"  csp run -c {dest / 'campaign.yaml'} --from dft --watch\n"
               f"then export the results into the shared cache:\n"
               f"  csp reference export -c {dest / 'campaign.yaml'}")


@reference_app.command("export")
def reference_export(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
) -> None:
    """Copy a finished reference campaign's energies into the shared cache."""
    from .reference.build import export
    from .reference.computed import recipe_id

    cfg = _load(campaign, set_)
    dft, recipe, _ = _dft_and_recipe(campaign, set_)
    rid = recipe_id(dft, recipe)
    store = Store.open(_db_path(cfg))
    with store:
        counts = export(store, recipe_id_=rid)
    typer.echo(f"recipe {rid[:16]}: {counts['done']} done, {counts['failed']} failed, "
               f"{counts['skipped']} skipped (no mp_id)")


@reference_app.command("adopt")
def reference_adopt(
    root: Annotated[Path, typer.Argument(
        help="a tree of mp-*/ phase directories, each with relax/ and static/")],
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    refresh: Annotated[bool, typer.Option("--refresh", help="re-read phases already cached")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="report only, write nothing")] = False,
) -> None:
    """Read finished VASP directories into the shared cache.

    The counterpart to `export`, which reads a campaign database.  Use this when
    the reference set is a curated tree of runs rather than one clean campaign --
    which is what a set that took eight repair rounds actually looks like.

    Every phase is checked before it is taken: the INCAR against the recipe's own
    policy, the cell volume and energy per atom against the structure that was
    submitted, and the atom count against the composition on record.  Phases that
    fail are refused and named, never written as `done`.

    Pass `-c` the campaign whose `dft:` block defines the policy -- without it
    the built-in defaults are used, which is almost certainly a different
    recipe_id and a cache directory with nothing in it.
    """
    from .reference.adopt import adopt
    from .reference.computed import recipe_id

    dft, recipe, where = _dft_and_recipe(campaign, set_)
    rid = recipe_id(dft, recipe)
    typer.echo(f"dft settings from {where}")
    typer.echo(f"recipe {rid[:16]}")
    if where == "built-in defaults":
        typer.echo("warning: no campaign file was read, so this is the DEFAULT policy. "
                   "If the tree was computed under a campaign, pass -c or the records "
                   "land under a recipe_id nothing will look for.", err=True)

    report = adopt(Path(root), rid=rid, recipe=recipe, refresh=refresh, dry_run=dry_run)
    typer.echo(report.render())
    if dry_run:
        typer.echo("\n--dry-run: nothing was written")
    else:
        typer.echo(f"\ncheck a system with:\n  csp reference status <chemsys> -c {campaign}")


@reference_app.command("exclude")
def reference_exclude(
    chemsys: SystemsArg = None,
    from_results: FromResults = None,
    systems_file: SystemsFile = None,
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    write: Annotated[bool, typer.Option("--write", help="record the exclusions; without it, report only")] = False,
    show: Annotated[bool, typer.Option("--show", help="list what is already excluded")] = False,
) -> None:
    """Mark phases that cannot change any hull, so coverage stops waiting on them.

    Without this, "we decided this phase cannot matter" and "nobody has run this
    phase yet" are the same state to `csp reference status`, and the first one
    blocks a hull forever.

    Applies the rule from VASP_FAILURES.md Part 4: a structure only matters to a
    hull if it is the sole structure at its composition, or lower than what else
    sits there. A phase that is the ONLY one at its composition is refused --
    dropping it deletes a vertex the hull needs -- and reported as MUST COMPUTE.

    Reports by default. Nothing is recorded until you pass --write.
    """
    from .reference.computed import (load_excluded, propose_exclusions,
                                     recipe_id, write_excluded)

    dft, recipe, where = _dft_and_recipe(campaign, set_)
    rid = recipe_id(dft, recipe)
    typer.echo(f"recipe {rid[:16]} (from {where})")

    if show:
        current = load_excluded(rid)
        if not current:
            typer.echo("nothing is excluded at this recipe")
            return
        typer.echo(f"{len(current)} phase(s) excluded:")
        for mp_id, why in sorted(current.items()):
            typer.echo(f"  {mp_id:16s} {why[:110]}")
        return

    systems = _systems(chemsys, from_results, systems_file)
    proposal = propose_exclusions(systems, rid)
    typer.echo(proposal.render())

    if proposal.keep:
        typer.echo("\nThe MUST COMPUTE phases above are the only structure at "
                   "their composition. They are not excluded whatever --write "
                   "says; compute them or those systems stay incomplete.")
    if not write:
        typer.echo("\nreport only -- pass --write to record these exclusions")
        return
    if not proposal.droppable:
        typer.echo("\nnothing to record")
        return
    path = write_excluded(rid, proposal.droppable)
    typer.echo(f"\nrecorded {len(proposal.droppable)} exclusion(s) in {path}")
    typer.echo("check with:\n  csp reference status <chemsys> -c " + str(campaign))


@reference_app.command("mlip")
def reference_mlip(
    chemsys: SystemsArg = None,
    from_results: FromResults = None,
    systems_file: SystemsFile = None,
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    no_relax: Annotated[bool, typer.Option(
        "--no-relax",
        help="single points only -- parity, but no energy comparable with e_dft")] = False,
    refresh: Annotated[bool, typer.Option("--refresh", help="recompute phases already stored")] = False,
) -> None:
    """Relax every cached MP geometry with the MLIP and store both energies.

    Two numbers per phase, because they answer different questions.  The single
    point at MP's own geometry is the parity number: it separates energy error
    from geometry error, and those have opposite consequences -- a uniform
    energy offset largely cancels along a hull tie-line, a volume bias does not.
    The relaxed energy is the hull number, because `e_dft` is also a relaxed
    energy and only a relaxed MLIP energy is on the same footing.

    Costs GPU-minutes and no scheduler, so it runs before any DFT.  What it
    measures -- whether the MLIP reproduces MP's own ordering in a chemistry --
    is an element-level property, and it is the cheapest thing that can tell you
    a chemical system is not worth spending DFT on.
    """
    from .mlip import for_config
    from .reference.build import mlip_pass
    from .reference.computed import recipe_id

    systems = _systems(chemsys, from_results, systems_file)
    dft, recipe, where = _dft_and_recipe(campaign, set_)
    rid = recipe_id(dft, recipe)
    if Path(campaign).is_file():
        screen = _load(campaign, set_).campaign.screen
    else:
        from .config.schema import Screen

        screen = Screen()
    typer.echo(f"recipe {rid[:16]} (dft settings from {where}); mlip {screen.mlip}")
    counts = mlip_pass(systems, recipe_id_=rid, engine=for_config(screen),
                       relax=not no_relax, refresh=refresh,
                       progress=lambda msg: typer.echo(f"  {msg}", err=True))
    typer.echo(f"{counts['static']} single point(s), {counts['relaxed']} relaxation(s)"
               + (f", {counts['unconverged']} stopped short of the force criterion"
                  if counts.get("unconverged") else "")
               + f", {counts['failed']} failure(s)")


@reference_app.command("report")
def reference_report(
    chemsys: SystemsArg = None,
    from_results: FromResults = None,
    systems_file: SystemsFile = None,
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    against: Annotated[str, typer.Option(
        "--against", help="'mp' = Materials Project GGA, 'ours' = our own DFT")] = "mp",
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="write an HTML report here")] = None,
) -> None:
    """How well the MLIP reproduces the reference set, per system and per element.

    Both comparisons are reported side by side: the single point at the
    reference geometry, which separates energy error from geometry error, and
    the MLIP relaxed to its own minimum, which is the like-for-like number.
    """
    from .reference.computed import recipe_id
    from .reference.report import build, write_html

    systems = _systems(chemsys, from_results, systems_file)
    dft, recipe, where = _dft_and_recipe(campaign, set_)
    rid = recipe_id(dft, recipe)
    if against not in ("mp", "ours"):
        _die("--against must be 'mp' or 'ours'")
    report = build(systems, recipe_id_=rid, against=against)
    if not report.systems:
        _die(f"nothing to report: no phase in these {len(systems)} system(s) has an "
             f"MLIP energy at recipe {rid[:12]}. Run `csp reference mlip` first.")
    typer.echo(report.render())
    if out:
        typer.echo(f"\nwrote {write_html(report, out)}")


@reference_app.command("status")
def reference_status(
    chemsys: SystemsArg = None,
    from_results: FromResults = None,
    systems_file: SystemsFile = None,
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    incomplete: Annotated[bool, typer.Option("--incomplete", help="only systems not ready")] = False,
) -> None:
    """Which systems can build a hull on our own energies, and which cannot."""
    from .reference.computed import coverage, recipe_id

    systems = _systems(chemsys, from_results, systems_file)
    dft, recipe, where = _dft_and_recipe(campaign, set_)
    rid = recipe_id(dft, recipe)
    typer.echo(f"recipe {rid[:16]} (from {where})")
    ready = 0
    for system in systems:
        try:
            cov = coverage(system, rid)
        except Exception as exc:                              # noqa: BLE001
            typer.echo(f"  {system:14s} ERROR {exc}")
            continue
        if cov.complete:
            ready += 1
            if incomplete:
                continue
        typer.echo(f"  {'OK ' if cov.complete else '   '}{cov.render()}")
    typer.echo(f"\n{ready}/{len(systems)} system(s) ready for a hull on our own energies")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        typer.secho("interrupted", fg=typer.colors.YELLOW, err=True)
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
