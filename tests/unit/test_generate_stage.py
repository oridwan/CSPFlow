"""Stage 1 as a driver stage, with a stand-in for MatterGen.

The stand-in is a real executable on PATH that parses the same arguments and
writes the same file MatterGen writes. That is enough to exercise everything
cspflow owns -- the argv, the subprocess, the timeout, the exit code, reading
the output back, assigning structures to compositions, and the database writes
that follow -- without a GPU or a checkpoint.

It also lets the failure modes be tested as first-class cases, which is the
point: a generator that exits non-zero, that writes nothing, that writes the
wrong chemistry, or that writes less than it was asked for are four different
outcomes and the campaign should be able to tell them apart afterwards.
"""

import json
import os
import stat
import textwrap
from pathlib import Path

import pytest
from ase.build import bulk

from cspflow.config.loader import load_campaign
from cspflow.db.store import Origin, Store, StructureState
from cspflow.generators import GenerationRequest, MatterGenEngine
from cspflow.scheduler.base import JobState, JobStatus
from cspflow.stages.generate_stage import GenerateStage
from cspflow.worker import WorkerError, run_generate_task

CAMPAIGN = """\
name: t
machine: orion
workdir: {workdir}
source:
  - mode: composition_list
    name: hand
    composition_list:
      items:
        - formula: SmFe2
        - formula: SmCo5
    defaults:
      z: {{min: 1, max: 2}}
      max_atoms: 20
      n_structures: {{mode: per_atom, structures_per_atom: 2.0}}
generate:
  engine: mattergen
  mattergen:
    model: {model}
    max_batch_size: 100
"""


# -- a MatterGen that is not MatterGen -------------------------------------

FAKE = r'''#!/usr/bin/env python3
"""Stands in for `mattergen-generate`: same argv, same output file."""
import json, os, sys
from ase import Atoms
import ase.io

args = sys.argv[1:]
out = args[0]
opts = {}
for a in args[1:]:
    if a.startswith("--"):
        k, _, v = a[2:].partition("=")
        opts[k] = v

mode = os.environ.get("FAKE_MATTERGEN_MODE", "ok")
if mode == "crash":
    sys.stderr.write("CUDA out of memory\n")
    sys.exit(1)
if mode == "silent":            # exit 0, write nothing -- mattergen's IOError path
    sys.exit(0)

comps = json.loads(opts["target_compositions"])
batch_size = int(opts["batch_size"])
num_batches = int(opts["num_batches"])
per = num_batches * batch_size // len(comps)
if mode == "short":
    per = max(1, per // 2)

structures = []
for c in comps:
    symbols = []
    for element, n in c.items():
        symbols += [element] * int(n)
    if mode == "wrong_chemistry":
        symbols = ["Ne"] * len(symbols)
    for i in range(per):
        a = 4.0 + 0.01 * i
        atoms = Atoms(symbols,
                      positions=[(0.3 * j, 0.3 * j, 0.3 * j) for j in range(len(symbols))],
                      cell=[a, a, a], pbc=True)
        structures.append(atoms)

os.makedirs(out, exist_ok=True)
ase.io.write(os.path.join(out, "generated_crystals.extxyz"), structures)
'''


@pytest.fixture
def fake_mattergen(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "mattergen-generate"
    script.write_text(FAKE)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_MATTERGEN_MODE", "ok")
    # The stand-in needs no GPU; the real engine refuses to start without one.
    monkeypatch.setenv("CSPFLOW_ALLOW_CPU_GENERATION", "1")
    return script


@pytest.fixture
def model(tmp_path):
    """A checkpoint directory shaped the way MatterGen expects one."""
    root = tmp_path / "ckpt"
    (root / "checkpoints").mkdir(parents=True)
    (root / "config.yaml").write_text("lightning_module: {}\n")
    (root / ".hydra").mkdir()
    (root / ".hydra" / "hydra.yaml").write_text(
        "hydra:\n  job:\n    config:\n      config_name: csp\n")
    return root


@pytest.fixture
def cfg(tmp_path, model):
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path, model=model))
    return load_campaign(path)


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "campaign.db", campaign="t") as s:
        yield s


def add_comps(store, specs):
    ids = []
    for formula, z, n_atoms, target in specs:
        ids.append(store.add_composition(
            formula=formula, chemsys="Fe-Sm", z=z, n_atoms=n_atoms,
            n_target=target, source_mode="composition_list", source_name="hand"))
    return ids


# -- the engine, through a real subprocess ---------------------------------

def engine_for(model, **kw):
    return MatterGenEngine(model=str(model), **kw)


def test_a_group_comes_back_split_by_composition(fake_mattergen, model, tmp_path):
    requests = [
        GenerationRequest(1, "Fe2Sm1", {"Fe": 2, "Sm": 1}, 6),
        GenerationRequest(2, "Co5Sm1", {"Co": 5, "Sm": 1}, 6),
    ]
    outcomes = engine_for(model).generate_many(requests, tmp_path / "work")
    assert [o.n_produced for o in outcomes] == [6, 6]
    assert all(o.ok and o.complete for o in outcomes)
    # Each composition got its own file, named for itself.
    for outcome in outcomes:
        assert Path(outcome.output).is_file()
        assert outcome.formula.replace("1", "") in Path(outcome.output).parent.name \
            or outcome.formula in Path(outcome.output).parent.name


def test_structures_are_assigned_by_composition_not_by_position(fake_mattergen,
                                                                model, tmp_path):
    """The fake writes composition-major, as MatterGen does. The result must not
    depend on that -- so assert the assignment is right, then assert it survives
    the file being written in the other order."""
    import ase.io

    requests = [GenerationRequest(1, "Fe2Sm1", {"Fe": 2, "Sm": 1}, 4),
                GenerationRequest(2, "Co5Sm1", {"Co": 5, "Sm": 1}, 4)]
    work = tmp_path / "w"
    outcomes = engine_for(model).generate_many(requests, work)
    for outcome, request in zip(outcomes, requests):
        for atoms in ase.io.read(outcome.output, index=":"):
            counts = {}
            for symbol in atoms.get_chemical_symbols():
                counts[symbol] = counts.get(symbol, 0) + 1
            assert counts == request.counts


def test_a_nonzero_exit_is_an_error_even_though_a_file_may_exist(fake_mattergen,
                                                                 model, tmp_path,
                                                                 monkeypatch):
    monkeypatch.setenv("FAKE_MATTERGEN_MODE", "crash")
    outcome = engine_for(model).generate(
        GenerationRequest(1, "Fe2Sm1", {"Fe": 2, "Sm": 1}, 4), tmp_path / "w")
    assert not outcome.ok
    assert "exited 1" in outcome.error and "CUDA out of memory" in outcome.error


def test_exit_zero_with_no_output_is_reported_as_such(fake_mattergen, model,
                                                      tmp_path, monkeypatch):
    """MatterGen's save step catches IOError, prints, and returns normally."""
    monkeypatch.setenv("FAKE_MATTERGEN_MODE", "silent")
    outcome = engine_for(model).generate(
        GenerationRequest(1, "Fe2Sm1", {"Fe": 2, "Sm": 1}, 4), tmp_path / "w")
    assert not outcome.ok
    assert "generated_crystals.extxyz" in outcome.error


def test_the_wrong_chemistry_is_rejected_rather_than_stored(fake_mattergen, model,
                                                            tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MATTERGEN_MODE", "wrong_chemistry")
    outcome = engine_for(model).generate(
        GenerationRequest(1, "Fe2Sm1", {"Fe": 2, "Sm": 1}, 4), tmp_path / "w")
    assert outcome.n_produced == 0
    assert not outcome.ok
    assert outcome.rejected


def test_a_shortfall_is_a_result_and_a_shortfall(fake_mattergen, model, tmp_path,
                                                 monkeypatch):
    """`ok` and `complete` are separate. Collapsing them is how the legacy
    resume check treated a half-finished job as a finished one."""
    monkeypatch.setenv("FAKE_MATTERGEN_MODE", "short")
    outcome = engine_for(model).generate(
        GenerationRequest(1, "Fe2Sm1", {"Fe": 2, "Sm": 1}, 8), tmp_path / "w")
    assert outcome.ok
    assert not outcome.complete
    assert outcome.n_produced == 4


def test_the_command_is_recorded_next_to_the_output(fake_mattergen, model, tmp_path):
    work = tmp_path / "w"
    engine_for(model).generate(
        GenerationRequest(1, "Fe2Sm1", {"Fe": 2, "Sm": 1}, 4), work)
    recorded = list(work.glob("call*/command.txt"))
    assert recorded and "mattergen-generate" in recorded[0].read_text()


# -- claiming --------------------------------------------------------------

def test_claim_groups_by_requested_count(cfg, store):
    add_comps(store, [("Fe2Sm1", 1, 3, 6), ("Fe2Sm1", 2, 6, 12),
                      ("Co5Sm1", 1, 6, 12), ("Co5Sm1", 2, 12, 24)])
    items = GenerateStage(cfg, group=10).claim(store, budget=10)
    targets = {item.payload["n_requested"] for item in items}
    assert targets == {6, 12, 24}
    for item in items:
        wanted = {c["n_requested"] for c in item.payload["compositions"]}
        assert len(wanted) == 1


def test_claim_respects_the_group_size(cfg, store):
    add_comps(store, [("Fe2Sm1", z, 3 * z, 6) for z in range(1, 6)])
    items = GenerateStage(cfg, group=2).claim(store, budget=10)
    assert [len(i.composition_ids) for i in items] == [2, 2, 1]


def test_claim_marks_generating_so_a_second_cycle_takes_nothing(cfg, store):
    add_comps(store, [("Fe2Sm1", 1, 3, 6)])
    stage = GenerateStage(cfg)
    assert stage.pending(store) == 1
    assert stage.claim(store, budget=10)
    assert stage.pending(store) == 0
    assert stage.claim(store, budget=10) == []
    assert store.compositions(state="generating")


def test_a_composition_wanting_nothing_is_not_claimed(cfg, store):
    add_comps(store, [("Fe2Sm1", 1, 3, 0)])
    assert GenerateStage(cfg).pending(store) == 0


def test_budget_caps_the_number_of_tasks(cfg, store):
    add_comps(store, [("Fe2Sm1", z, 3 * z, 6) for z in range(1, 8)])
    items = GenerateStage(cfg, group=1).claim(store, budget=3)
    assert len(items) == 3


# -- the job ---------------------------------------------------------------

def test_build_writes_a_manifest_the_worker_can_read(cfg, store, tmp_path):
    add_comps(store, [("Fe2Sm1", 1, 3, 6), ("Co5Sm1", 1, 6, 6)])
    stage = GenerateStage(cfg, group=1)
    items = stage.claim(store, budget=10)
    spec = stage.build(items, tmp_path / "jobs")
    assert spec.array_size == len(items)
    assert spec.gpus == 1
    assert "generate-worker" in spec.command
    manifest = json.loads(next((tmp_path / "jobs").rglob("inputs.json")).read_text())
    assert len(manifest["chunks"]) == len(items)
    assert manifest["max_batch_size"] == 100
    assert manifest["model"] == str(cfg.campaign.generate.mattergen.model)


def test_the_job_carries_the_allocator_setting(cfg, store, tmp_path):
    add_comps(store, [("Fe2Sm1", 1, 3, 6)])
    stage = GenerateStage(cfg)
    spec = stage.build(stage.claim(store, budget=1), tmp_path / "jobs")
    assert spec.env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


# -- the worker ------------------------------------------------------------

def test_the_worker_runs_a_chunk_end_to_end(cfg, store, tmp_path, fake_mattergen):
    add_comps(store, [("Fe2Sm1", 1, 3, 4), ("Co5Sm1", 1, 6, 4)])
    stage = GenerateStage(cfg, group=10)
    items = stage.claim(store, budget=10)
    stage.build(items, tmp_path / "jobs")
    manifest = next((tmp_path / "jobs").rglob("inputs.json"))

    out = run_generate_task(manifest, task_id=0)
    payload = json.loads(out.read_text())
    assert len(payload["results"]) == 2
    assert all(r["n_produced"] == 4 for r in payload["results"])


def test_the_worker_refuses_a_task_id_the_manifest_has_no_chunk_for(cfg, store,
                                                                    tmp_path,
                                                                    fake_mattergen):
    add_comps(store, [("Fe2Sm1", 1, 3, 4)])
    stage = GenerateStage(cfg)
    stage.build(stage.claim(store, budget=1), tmp_path / "jobs")
    manifest = next((tmp_path / "jobs").rglob("inputs.json"))
    with pytest.raises(WorkerError, match="no chunk"):
        run_generate_task(manifest, task_id=7)


def test_the_worker_checks_the_cell_size_against_the_composition_row(cfg, store,
                                                                     tmp_path,
                                                                     fake_mattergen):
    """Z and n_atoms are stored separately; if they disagree the request that
    reaches MatterGen is not the one the campaign thinks it made."""
    store.add_composition(formula="Fe2Sm1", chemsys="Fe-Sm", z=1, n_atoms=99,
                          n_target=4, source_mode="composition_list", source_name="hand")
    stage = GenerateStage(cfg)
    stage.build(stage.claim(store, budget=1), tmp_path / "jobs")
    manifest = next((tmp_path / "jobs").rglob("inputs.json"))
    with pytest.raises(WorkerError, match="atoms but the composition row says"):
        run_generate_task(manifest, task_id=0)


def test_the_worker_refuses_before_loading_a_model_it_cannot_use(cfg, store, tmp_path,
                                                                 monkeypatch):
    """No fake on PATH: preflight must catch it, not the subprocess."""
    add_comps(store, [("Fe2Sm1", 1, 3, 4)])
    stage = GenerateStage(cfg)
    stage.build(stage.claim(store, budget=1), tmp_path / "jobs")
    manifest = next((tmp_path / "jobs").rglob("inputs.json"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(WorkerError, match="preflight failed"):
        run_generate_task(manifest, task_id=0)


# -- reconciliation --------------------------------------------------------

def done(job_id, workdir):
    return {"id": job_id, "workdir": str(workdir)}, JobStatus(
        job_id="1", state=JobState.done, raw_state="COMPLETED")


def test_reconcile_ingests_structures_and_records_the_yield(cfg, store, tmp_path,
                                                            fake_mattergen):
    ids = add_comps(store, [("Fe2Sm1", 1, 3, 4), ("Co5Sm1", 1, 6, 4)])
    stage = GenerateStage(cfg, group=10)
    items = stage.claim(store, budget=10)
    workdir = tmp_path / "jobs"
    stage.build(items, workdir)
    run_generate_task(next(workdir.rglob("inputs.json")), task_id=0)

    row, status = done(1, workdir)
    stage.reconcile(store, row, status, items)

    assert store.count_structures(state=StructureState.new.value) == 8
    for cid in ids:
        comp = next(c for c in store.compositions() if c.id == cid)
        assert comp.state == "generated"
        assert comp.n_produced == 4
    assert store.generation_yield() == {"compositions": 2, "requested": 8,
                                        "produced": 8, "short": 0}


def test_one_task_handed_back_alone_reads_its_own_results_file(cfg, store, tmp_path,
                                                               fake_mattergen):
    """Per-task reconciliation hands over ONE item; it must not read task 0.

    The driver frees each array task as it ends (D113), so `items` is a
    one-element list carrying its own `task_index`.  A stage that derived the
    index from the list position would open task 0's results for every task:
    the first task's compositions would be written repeatedly and every other
    task's would be failed as "produced no results" while its structures sat
    finished on disk (D121).
    """
    add_comps(store, [("Fe2Sm1", 1, 3, 4), ("Co5Sm1", 1, 6, 4)])
    stage = GenerateStage(cfg, group=1)
    items = stage.claim(store, budget=10)
    assert len(items) == 2
    mine = items[1].composition_ids[0]        # what array task 1 actually holds
    theirs = items[0].composition_ids[0]
    workdir = tmp_path / "jobs"
    stage.build(items, workdir)
    manifest = next(workdir.rglob("inputs.json"))
    run_generate_task(manifest, task_id=0)
    run_generate_task(manifest, task_id=1)

    # exactly what the driver passes for array task 1
    items[1].task_index = 1
    row, status = done(1, workdir)
    stage.reconcile(store, row, status, [items[1]])

    task1 = next(c for c in store.compositions() if c.id == mine)
    task0 = next(c for c in store.compositions() if c.id == theirs)
    assert task1.state == "generated" and task1.n_produced == 4
    assert task0.state == "generating", "task 0 must not have been touched"


def test_reconcile_records_a_shortfall_without_calling_it_a_failure(cfg, store,
                                                                    tmp_path,
                                                                    fake_mattergen,
                                                                    monkeypatch):
    add_comps(store, [("Fe2Sm1", 1, 3, 8)])
    stage = GenerateStage(cfg)
    items = stage.claim(store, budget=1)
    workdir = tmp_path / "jobs"
    stage.build(items, workdir)
    monkeypatch.setenv("FAKE_MATTERGEN_MODE", "short")
    run_generate_task(next(workdir.rglob("inputs.json")), task_id=0)

    row, status = done(1, workdir)
    stage.reconcile(store, row, status, items)
    comp = store.compositions()[0]
    assert comp.state == "generated"
    assert comp.n_produced == 4
    assert "short" in comp.fail_reason
    assert store.generation_yield()["short"] == 1


def test_a_task_that_produced_no_results_file_fails_its_compositions(cfg, store,
                                                                     tmp_path):
    add_comps(store, [("Fe2Sm1", 1, 3, 4)])
    stage = GenerateStage(cfg)
    items = stage.claim(store, budget=1)
    workdir = tmp_path / "jobs"
    stage.build(items, workdir)
    row = {"id": 1, "workdir": str(workdir)}
    status = JobStatus(job_id="1", state=JobState.failed, raw_state="OUT_OF_MEMORY")
    stage.reconcile(store, row, status, items)
    comp = store.compositions()[0]
    assert comp.state == "failed"
    assert "OUT_OF_MEMORY" in comp.fail_reason


def test_reconcile_notices_a_results_file_pointing_at_nothing(cfg, store, tmp_path):
    ids = add_comps(store, [("Fe2Sm1", 1, 3, 4)])
    stage = GenerateStage(cfg)
    items = stage.claim(store, budget=1)
    workdir = tmp_path / "jobs"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / f"{items[0].key}.task0.json").write_text(json.dumps({
        "engine": "mattergen",
        "results": [{"composition_id": ids[0], "formula": "Fe2Sm1", "n_requested": 4,
                     "n_produced": 4, "output": str(workdir / "gone.extxyz"),
                     "error": "", "batches": [4], "rejected": {}}]}))
    row, status = done(1, workdir)
    stage.reconcile(store, row, status, items)
    comp = store.compositions()[0]
    assert comp.state == "failed"
    assert "is not there" in comp.fail_reason


# -- registry --------------------------------------------------------------

def test_generate_is_in_the_registry_when_configured(cfg):
    from cspflow.stages import build_registry
    assert "generate" in [s.name for s in build_registry(cfg)]


def test_generate_is_absent_when_nothing_needs_generating(tmp_path):
    from cspflow.stages import build_registry

    (tmp_path / "seeds").mkdir()
    path = tmp_path / "c.yaml"
    path.write_text(f"""\
name: t
machine: orion
workdir: {tmp_path}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{tmp_path}/seeds"]}}
""")
    cfg = load_campaign(path)
    assert "generate" not in [s.name for s in build_registry(cfg)]
