# 01 — Installation

> **Guide 01 of 13** · [Guide home](README.md) · Next: [02 — Quick start](02-quickstart.md)

## What you need

| | |
|---|---|
| **Python** | 3.10 or newer |
| **A cluster** | SLURM, or `scheduler: local` for a workstation |
| **VASP** | Your own licensed build, plus the POTCAR files. cspflow never ships either. |
| **A Materials Project API key** | Free from [materialsproject.org](https://next-gen.materialsproject.org/api) — reference energies for the convex hull come from there |
| **A GPU** | For structure generation and MLIP screening. Not needed for the DFT half. |

You can install and explore cspflow — write a campaign, enumerate it, inspect
the resolved config and the DFT recipe — with none of the above except Python.
Everything up to `csp source --dry-run` runs on a laptop.

On the MMI cluster, the shared `orion` profile already points at the group VASP
binary, POTCAR tree, and shared reference store. New users should use that
profile before creating custom paths. On another cluster, you will configure
those locations in [Guide 08: Running on your cluster](08-machines.md).

## Install

```bash
pip install cspflow            # the pipeline: config, driver, scheduler, DFT
pip install "cspflow[full]"    # adds pymatgen — needed for real work
```

`[full]` is not really optional for a campaign that runs: pymatgen provides
structure matching (dedup), spacegroup determination, convex-hull construction,
k-point meshes and POTCAR handling. The base install exists so that the config
layer, the CLI and the tests stay usable without a scientific stack.

From a checkout:

```bash
git clone <url> cspflow && cd cspflow
pip install -e ".[full,dev]"
```

## The machine-learning stack

Generation (MatterGen) and screening (MatterSim) are **not** installed by pip,
because they pin each other and pin torch. Trying to add them to an existing
environment usually breaks it. Use the bundled script, which builds one conda
environment holding all three:

```bash
./scripts/build_env.sh cspflow      # python 3.10, torch 2.2.1+cu118
conda activate cspflow
csp version
```

It writes `scripts/env.lock.txt` so the environment can be rebuilt exactly. The
script's header explains every pin — the short version is that mattergen
requires `numpy<2` and mattersim 1.2+ requires `numpy>=2`, so the newest of each
cannot coexist and the working combination is not obvious.

One environment serves every stage. If your stages need different environments,
`machine.yaml` maps each role to one (`conda: {cpu: ..., gpu: ...}`).

## Environment variables

On the MMI cluster you need only two, and should **not** set `CSPFLOW_STORE`:

```bash
export MP_API_KEY=...                      # required to build a hull from MP
export CSPFLOW_REFERENCE=/projects/mmi/cspflow-shared/reference
# CSPFLOW_STORE: leave unset. The default is the group's shared store,
# /projects/mmi/cspflow-shared/store. Pointing it at /scratch/$USER/... gives
# you an EMPTY store, and every hull is then refused for missing phases.
```

Check it once: `python scripts/refstore.py status` should report a `total` in
the thousands. A total near zero means you are pointed at an empty store.

| variable | default | what it decides |
|---|---|---|
| `MP_API_KEY` | *(required)* | Your Materials Project key. |
| `CSPFLOW_REFERENCE` | `~/.local/share/cspflow/reference` | The frozen MP snapshot and the recomputed-energy cache (`$CSPFLOW_REFERENCE/mp`, `/computed`). |
| `CSPFLOW_STORE` | a built-in path | **The reference store: one folder of recomputed phases plus `index.csv`.** This is the living source of truth for the hull. |
| `CSPFLOW_STORE_SETTINGS` | `store-settings.yaml` beside the reference cache | The store's own `dft:` block, which `csp doctor` compares your campaign against. |
| `CSPFLOW_STRUCTURE_DIR` | `$CSPFLOW_REFERENCE/structures` | Where per-structure folders are written. Set it to an **empty string** to stop writing them. |
| `CSPFLOW_ALLOW_CPU_GENERATION` | unset | Lets MatterGen run without a GPU. For debugging only — it is unusably slow. |
| `PMG_VASP_PSP_DIR` | — | Optional; `machine.yaml` can name the POTCAR tree instead. |

The MP key is read from the environment and never written to a config file, a
cache or the database. Keep it out of `campaign.yaml`.

**`CSPFLOW_STORE` is the one to get right.** It is where `e_above_hull` comes
from, and pointing it somewhere else builds a *different* hull from the same
campaign. `csp doctor` prints which store answered and whether your campaign's
DFT settings match it; grepping a store listing does not tell you the same thing.

Where these are set in practice: `scripts/campaign_driver.sbatch` exports all of
them, so a driver submitted through it gets a consistent environment whatever
your shell had. If you run `csp` by hand, export them yourself — a campaign that
built its hull under one `CSPFLOW_STORE` and is inspected under another will
disagree with itself.

> **Any `$VAR` you write in a config file must be defined**, or the campaign
> stops with an error naming it. This is deliberate: an empty expansion turns
> `$SCRATCH/work` into `/work`.

> **If you ever move these directories**, update the exports in
> `scripts/campaign_driver.sbatch` at the same time, and remember that a
> campaign already running holds absolute paths — in its database, in its job
> scripts, and in every `#SBATCH --chdir` already queued. Moving a live
> campaign's `workdir` breaks the jobs in flight.

## POTCARs

pymatgen expects the POTCAR tree to use particular directory names, which the
distribution tarballs do not. `csp doctor --fix` builds the expected symlink
layout beside your files:

```bash
csp doctor --fix
```

Without it, a `functional: PBE_64` label can silently match a flat-layout
directory, and the recorded label then misdescribes what was actually used.
Nothing is copied and nothing is modified — only symlinks are added.

## Verify

```bash
csp version
csp doctor            # exits non-zero on any hard failure
```

For a first check, success means:

- `csp version` prints a version instead of “command not found”;
- `csp doctor` can load the machine profile and campaign configuration;
- every line marked as a hard failure has been fixed before job submission.

Warnings may describe optional features, but do not ignore a warning about the
reference store or DFT settings: those determine whether hull energies are
comparable.

`doctor` resolves a POTCAR for every element in your campaign, reads your live
SLURM limits, checks that each configured module exists, verifies the VASP
binary, and reports what it cannot confirm. Run it before your first submission
and after any cluster change; it is safe to put at the top of a submit script.

## Next

* [Quick start →](02-quickstart.md) — a working campaign in ten minutes
* [A-to-Z workflows](04-workflows.md) — a full walkthrough per source mode
* [Every setting in `campaign.yaml`](06-settings.md)
