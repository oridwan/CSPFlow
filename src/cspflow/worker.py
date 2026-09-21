"""What an array task runs.

The worker is deliberately the dumbest process in the system. It reads a
manifest, relaxes the structures named in its chunk, writes a JSON file, and
exits. It does not open the campaign database for writing, does not decide what
work to do, and does not retry.

Everything about that is on purpose (see `stages/screen_stage.py`): the driver
is the only writer, so at 48 concurrent tasks nothing contends for SQLite's
write lock; and a worker that dies leaves no results file, which reconciliation
reports as work that did not come back rather than as silent partial data.

The results file is written to a temporary name and renamed, so a task killed
mid-write cannot leave a half-parsed JSON file behind that looks like a result.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ase.io import read as ase_read


class WorkerError(Exception):
    pass


def _log(message: str) -> None:
    """One line to the job's stdout, flushed.

    The screen job used to print exactly two lines for a 24-minute run -- the
    MatterSim checkpoint banner and the output path -- so a driver reporting
    "in flight 1" for eleven cycles was the only signal that anything was
    happening, and it carried no information.  Flushed because SLURM buffers
    stdout to a file and an unflushed progress line is not progress.
    """
    print(message, flush=True)


def _progress_line(n: int, total: int, sid: int, atoms, result, started: float) -> str:
    """What one structure did, in one line, with the numbers worth seeing."""
    formula = atoms.get_chemical_formula()
    if result.error:
        return f"[{n}/{total}] id={sid} {formula:<16} FAILED {result.error[:70]}"

    rate = (time.time() - started) / max(n, 1)
    eta = rate * (total - n)
    verdict = "converged" if result.converged else "STEP LIMIT"
    energy = "" if result.energy is None else f"E={result.energy:12.4f} eV"
    fmax = "" if result.fmax is None else f"fmax={result.fmax:.4f}"
    drift = ("" if result.volume_drift is None
             else f"dV={result.volume_drift * 100:+.1f}%")
    return (f"[{n}/{total}] id={sid} {formula:<16} {energy} {fmax} "
            f"{result.n_steps:4d} steps {drift:<8} {verdict}  eta {eta / 60:.0f} min")


def _write_progress(path: Path, task_id: int, done: int, total: int,
                    started: float, results: list) -> None:
    """A tiny file the DRIVER can read to say what a running job is doing.

    The driver cannot see the job's stdout -- that belongs to another SLURM
    allocation -- so progress has to land somewhere shared.  Written whole each
    time rather than appended: it is a few hundred bytes, and a partial line in
    an appended file is worse than a slightly stale whole one.  Failure to write
    it is ignored, because this file is a convenience and the results file is
    the contract.
    """
    try:
        path.write_text(json.dumps({
            "task_id": task_id,
            "done": done,
            "total": total,
            "converged": sum(1 for r in results if r["converged"]),
            "failed": sum(1 for r in results if r["error"]),
            "elapsed_s": round(time.time() - started, 1),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }))
    except OSError:
        pass


def run_screen_task(manifest_path: str | Path, task_id: int | None = None) -> Path:
    """Relax one chunk of a screen manifest and write its results file."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise WorkerError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    chunks = manifest["chunks"]
    if not 0 <= task_id < len(chunks):
        raise WorkerError(
            f"array task {task_id} has no chunk in {manifest_path} "
            f"({len(chunks)} chunk(s)). Was --array sized to match the manifest?"
        )
    ids = chunks[task_id]

    from .db.store import Store
    from .mlip import MatterSimEngine

    # Read-only use of the database.  Both journal modes we use (WAL on a local
    # disk, TRUNCATE on a network one) let any number of readers in at once, so
    # every task in the array can do this together.
    store = Store.open(manifest["db"])
    try:
        structures = [(sid, store.get_structure(sid).toatoms()) for sid in ids]
    finally:
        store.close()

    engine = MatterSimEngine(
        model=manifest.get("model", "MatterSim-v1.0.0-5M.pth"),
        fmax=float(manifest.get("fmax", 0.01)),
        max_steps=int(manifest.get("max_steps", 500)),
    )

    # Structures the campaign asked not to move. A single point still gives the
    # energy the hull needs; what it does not do is change a geometry the user
    # supplied deliberately.
    single_point = set(manifest.get("single_point") or [])

    key = manifest.get("key") or manifest_path.name.split(".manifest")[0]
    # Where the relaxed cells go.  They used to go nowhere: the worker recorded
    # energy, fmax and step count and threw the GEOMETRY away, so the whole
    # point of the relaxation was lost and `dft_stage` step 0 started from the
    # seed as supplied.  The reference store already had this fixed; the
    # campaign path did not.
    # D136: names that say which is input and which is output. Everything is
    # derived from the manifest's own directory, so when the stage puts the
    # manifest in `batches/<tag>/` the results, the progress file and the
    # relaxed cells follow it there without the worker knowing anything about
    # the layout.
    # ONE DATABASE PER TASK (D141), not one POSCAR per structure.
    #
    # It was `relaxed-task<N>/<sid>.vasp` -- geometry only, with the energy that
    # belongs to it in a different file, and one file per structure. A 500-cell
    # chunk therefore left 500 files that say nothing about what was measured.
    #
    # One ASE database instead: the relaxed cell and every number measured for
    # it in one row, so this file alone can rebuild the campaign's record of
    # this task. Written PER TASK because a task is one process -- the only
    # granularity at which there is exactly one writer and no locking.
    relaxed_db = manifest_path.parent / f"relaxed-task{task_id}.db"
    progress_path = manifest_path.parent / f"progress-task{task_id}.json"

    results: list[dict[str, Any]] = []
    total = len(structures)
    started = time.time()
    for n, (sid, atoms) in enumerate(structures, start=1):
        result = (engine.single_point(atoms) if sid in single_point
                  else engine.relax(atoms))

        # Save the relaxed cell AND its numbers before anything else can go
        # wrong with them -- written now, inside the loop, so a worker killed at
        # structure 300 of 500 keeps the 299 it finished.
        geometry = ""
        if result.atoms is not None and sid not in single_point:
            from . import artifacts

            if artifacts.record(
                relaxed_db, result.atoms,
                structure_id=sid,
                reduced_formula=result.atoms.get_chemical_formula(
                    mode="metal", empirical=True),
                # NOT `energy`, `formula`, `fmax` or `volume`: ASE owns those
                # names on a row and refuses them as key_value_pairs.
                e_total=result.energy, e_per_atom=result.e_per_atom,
                converged=bool(result.converged), n_steps=int(result.n_steps or 0),
                fmax_final=result.fmax, volume_before=result.volume_before,
                volume_after=result.volume_after, volume_drift=result.volume_drift,
                engine=result.engine or "", model=str(manifest.get("model") or ""),
                task_id=int(task_id), relaxed=True,
            ):
                geometry = str(relaxed_db)
            else:
                _log(f"  WARNING: could not save the relaxed cell for {sid}")

        results.append({
            "structure_id": sid,
            "energy": result.energy,
            "e_per_atom": result.e_per_atom,
            "converged": result.converged,
            "n_steps": result.n_steps,
            "fmax": result.fmax,
            "volume_before": result.volume_before,
            "volume_after": result.volume_after,
            "volume_drift": result.volume_drift,
            "error": result.error,
            "engine": result.engine,
            "relaxed": sid not in single_point,
            "geometry": geometry,
        })

        _log(_progress_line(n, total, sid, atoms, result, started))
        _write_progress(progress_path, task_id, n, total, started, results)

    out = manifest_path.parent / f"results-task{task_id}.json"
    payload = {
        "task_id": task_id,
        "max_steps": manifest.get("max_steps"),
        "device": engine.device,
        "relaxed_db": str(relaxed_db) if relaxed_db.exists() else "",
        "results": results,
    }
    _atomic_write_json(out, payload)
    ok = sum(1 for r in results if r["converged"])
    failed = sum(1 for r in results if r["error"])
    _log(f"done: {ok}/{total} converged, {total - ok - failed} hit the step limit, "
         f"{failed} failed, in {time.time() - started:.0f} s")
    if relaxed_db.exists():
        _log(f"relaxed cells: {relaxed_db}")
    progress_path.unlink(missing_ok=True)
    return out


def run_generate_task(manifest_path: str | Path, task_id: int | None = None) -> Path:
    """Generate structures for one chunk of compositions and write its results.

    The chunk is a group of compositions that all want the same number of
    structures, so the whole chunk is one MatterGen call (or two, when the count
    exceeds the batch cap).  The checkpoint is therefore loaded once for the
    task rather than once per composition.

    Preflight runs *before* the model is touched.  The check that matters most
    is CUDA: MatterGen's `get_device()` returns CPU when no GPU is visible and
    logs nothing, so a task that landed on the wrong partition would sample at
    roughly a thousandth of the intended rate and be killed by its walltime,
    leaving a TIMEOUT with no cause anywhere in the logs.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise WorkerError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    chunks = manifest["chunks"]
    if not 0 <= task_id < len(chunks):
        raise WorkerError(
            f"array task {task_id} has no chunk in {manifest_path} "
            f"({len(chunks)} chunk(s)). Was --array sized to match the manifest?")
    chunk = chunks[task_id]

    from .chem import parse_formula
    from .generators import GenerationRequest, MatterGenEngine

    engine = MatterGenEngine(
        model=manifest["model"], mode=manifest.get("mode", "csp"),
        max_batch_size=int(manifest.get("max_batch_size", 100)),
        timeout_per_batch=int(manifest.get("timeout_per_batch", 1800)),
    )
    problems = engine.preflight()
    if problems:
        raise WorkerError("generation preflight failed:\n  - " + "\n  - ".join(problems))

    requests = []
    for entry in chunk:
        per_fu = parse_formula(entry["formula"])
        counts = {element: count * int(entry["z"]) for element, count in per_fu.items()}
        if sum(counts.values()) != int(entry["n_atoms"]):
            raise WorkerError(
                f"{entry['formula']} Z={entry['z']} is {sum(counts.values())} atoms "
                f"but the composition row says {entry['n_atoms']}")
        requests.append(GenerationRequest(
            composition_id=int(entry["id"]), formula=entry["formula"],
            counts=counts, n_requested=int(entry["n_requested"])))

    key = manifest.get("key") or manifest_path.name.split(".manifest")[0]
    task_dir = manifest_path.parent / f"output-task{task_id}"
    outcomes = engine.generate_many(requests, task_dir)

    out = manifest_path.parent / f"results-task{task_id}.json"
    _atomic_write_json(out, {
        "task_id": task_id,
        "engine": engine.name,
        "model": engine.model,
        "results": [o.as_dict() for o in outcomes],
    })
    return out


def prepare_stage(run_dir: str | Path, step_name: str,
                  campaign: str | Path, ntasks: int | None = None) -> Path:
    """Write one step's VASP inputs from the PREVIOUS step's relaxed cell.

    This runs INSIDE the batch job, between two `vasp_std` invocations, because
    it cannot run before: a `static`'s POSCAR *is* the relax's CONTCAR, and that
    file does not exist until the relax has finished.

    Getting this wrong is not loud. `static` exists to give a high-accuracy
    energy AT THE RELAXED GEOMETRY, and that energy is what goes on the DFT
    hull. Started from the structure in the database it runs on the generated
    cell instead and reports a number that looks entirely plausible and is wrong
    by whatever the relaxation was worth -- measured on this campaign at 176.15
    vs 179.03 A^3, 172.92 vs 179.71, 260.70 vs 260.84.

    The k-grid is regenerated too, not copied. It is chosen from the cell, and
    the cell has just changed; a relaxed cell can legitimately want FEWER points
    than the one it started from. (Compare grids sorted when checking: Niggli
    reduction permutes axes, so [3,3,6] and [6,3,3] are the same grid.)

    Idempotent: a step that already holds a converged run is left untouched, so
    re-running the job after a `static` failure never re-does the relax.
    """
    from .config.loader import load_campaign
    from .dft.recipe import load_recipe
    from .dft.vasp.inputs import resolve_inputs, write_inputs
    from .dft.vasp.parse import read_job_directory

    run_dir = Path(run_dir).resolve()
    target = run_dir / step_name
    cfg = load_campaign(Path(campaign))
    # base_dir, or a campaign-local `recipe: recipe.yaml` is looked for in the
    # run directory the job has chdir'd into, and is not there (D132).
    recipe = load_recipe(cfg.campaign.dft.recipe, cfg.base_dir)

    names = [st.name for st in recipe.stages]
    if step_name not in names:
        raise WorkerError(f"{step_name!r} is not a step of recipe {recipe.name!r}; "
                          f"it has {names}")
    index = names.index(step_name)
    if index == 0:
        raise WorkerError(f"{step_name!r} is the first step; its inputs are written "
                          f"when the job is built, not from a previous step")

    # Already done? Say so and change nothing. This is what makes the combined
    # script safe to re-run: a static that failed must not cost its relax.
    if (target / "OUTCAR").is_file():
        outcome = read_job_directory(target)
        if outcome.converged and outcome.energy is not None:
            _log(f"{step_name}: already converged in {target}, leaving it alone")
            return target

    previous = run_dir / names[index - 1]
    contcar = previous / "CONTCAR"
    if not contcar.is_file() or contcar.stat().st_size == 0:
        raise WorkerError(
            f"cannot prepare {step_name}: {contcar} is missing or empty, so the "
            f"relaxed geometry it must run at does not exist. Running it on the "
            f"unrelaxed cell would produce a plausible and wrong energy."
        )
    atoms = ase_read(str(contcar))

    stage = recipe.stages[index]
    resources = {**recipe.stages[0].resources, **stage.resources}
    # `ntasks` is PASSED IN, never read from the environment.
    #
    # KPAR must divide the rank count exactly or VASP refuses to start, so this
    # number has to be the one in the job's own `#SBATCH --ntasks` -- which the
    # builder knows and writes onto this command line.
    #
    # `SLURM_NTASKS` looks like the same thing and is not. Measured while
    # testing this function: it read **2**, leaked from the surrounding
    # allocation, and silently beat the recipe's 64 -- producing an INCAR with
    # KPAR dropped and NCORE halved. `campaign_driver.sbatch` already unsets
    # that variable for this exact reason ("a leaked SLURM_EXPORT_ENV=NONE from
    # a parent is what broke the first wave"). An inherited value is not this
    # job's allocation, and there is no way to tell the two apart from inside.
    if ntasks is None:
        ntasks = int(resources.get("ntasks") or 0) or None
    resolved = resolve_inputs(atoms, stage, cfg.campaign.dft, cfg.machine, ntasks=ntasks)
    write_inputs(resolved, atoms, target)
    _log(f"{step_name}: inputs written to {target} from {contcar}")
    return target


def step_is_converged(directory: str | Path) -> bool:
    """Did the run in `directory` reach its convergence criterion?

    The combined job asks this before every step, and skips the step when the
    answer is yes. That is what stops a failed `static` from costing its `relax`
    all over again.

    A directory that cannot be read answers **False**. Being wrong in that
    direction repeats work; being wrong in the other direction skips a step that
    never converged and reports whatever half-finished numbers are lying in it.
    Only one of those is recoverable.

    It exists as a function, rather than a line of python embedded in the shell
    script, because it is a decision worth testing. The embedded version passed
    `sys.argv[1]` -- a `str` -- to a parser that indexes it with `/`, raised
    TypeError, was read as "not converged", and re-ran a finished relaxation.
    """
    from .dft.vasp.parse import read_job_directory

    directory = Path(directory)
    if not (directory / "OUTCAR").is_file():
        return False
    try:
        outcome = read_job_directory(directory)
    except Exception as exc:                                   # noqa: BLE001
        _log(f"cannot judge {directory}: {exc}; treating it as not converged")
        return False
    return bool(outcome.converged and outcome.energy is not None)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write to a temporary name, then rename.

    A task killed part-way through writing would otherwise leave a truncated
    file that parses as nothing and reads, to the driver, as a corrupt result
    rather than an absent one. `rename` within a directory is atomic.
    """
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:      # pragma: no cover - entry point
    import argparse

    parser = argparse.ArgumentParser(prog="csp-worker")
    parser.add_argument("--manifest")
    parser.add_argument("--task-id", type=int, default=None)
    # Used by the combined DFT job between its two `vasp_std` calls.
    parser.add_argument("--prepare-stage", nargs=2, metavar=("RUN_DIR", "STEP"))
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--ntasks", type=int, default=None,
                        help="the job's own #SBATCH --ntasks; KPAR must divide it")
    parser.add_argument("--is-converged", metavar="DIR",
                        help="exit 0 if DIR holds a converged run, 1 otherwise")
    args = parser.parse_args(argv)

    if args.is_converged:
        return 0 if step_is_converged(args.is_converged) else 1

    if args.prepare_stage:
        if not args.campaign:
            parser.error("--prepare-stage also needs --campaign")
        print(prepare_stage(args.prepare_stage[0], args.prepare_stage[1],
                            args.campaign, args.ntasks))
        return 0

    if not args.manifest:
        parser.error("one of --manifest or --prepare-stage is required")
    out = run_screen_task(args.manifest, args.task_id)
    print(out)
    return 0


if __name__ == "__main__":                            # pragma: no cover
    sys.exit(main())
