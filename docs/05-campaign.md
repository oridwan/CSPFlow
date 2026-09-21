# 05 — `campaign.yaml` explained

> **Guide 05 of 13** · Previous: [04 — A-to-Z workflows](04-workflows.md) · [Guide home](README.md) · Next: [06 — Every campaign setting](06-settings.md)

One file describes the whole search. This page explains that file by purpose:
what enters the pipeline, which candidates survive, and how much expensive work
may run. For an exact key-by-key lookup, use
[Guide 06: Every campaign setting](06-settings.md).

You rarely write it from scratch: `csp init <type> <name>` copies the
[example campaign](../examples/) for that type (1 `chemical_space`,
2 `composition_list`, 3 `structure_list`). **Every setting that matters is live**,
written out even where it equals the default, and the alternatives sit in a
comment on the same line — `# a | b | c` to pick one, `# several: [a, b]` where a
comma-separated list or several `key: value` pairs are allowed. Settings no code
reads are left out on purpose (D137, D146). `csp init --minimal` writes the same
file with the comments removed.

## Start with these decisions

A new user does not need to understand every block before the first test. Make
these decisions first and leave the remaining generated defaults alone:

| Decision | Where to set it | Beginner check |
|---|---|---|
| What is this run called? | `name` | Use a short, unique name; it appears in job names and reports. |
| Where can large files live? | `workdir` | Use scratch, not the Git checkout or your home directory. |
| What do I provide? | `source` | Pick one mode in [Guide 03](03-sources.md). |
| How many candidates may reach DFT? | `filter` and `dft.select` | Keep the first test small. |
| How many cores may be held? | `dft.max_cores` | Set an explicit cap before Phase B. |

Then run:

```bash
csp source --dry-run          # shows the size; writes and submits nothing
csp doctor                    # checks the machine, store, VASP, and POTCARs
csp config show --origins     # shows the final value and source of every key
csp recipe                    # shows the exact DFT steps that will run
```

If these four commands describe the campaign you intended, continue with the
[A-to-Z workflow](04-workflows.md). If not, fix the configuration before submitting
anything; none of these checks spends compute.

```bash
csp config defaults           # the schema's defaults, in full
csp config show --origins     # your resolved values, and which file set each
```

## The shape of the file

```yaml
name: my-campaign             # required
machine: machine.yaml         # required — a path, or a shipped profile name
workdir: /scratch/$USER/...   # required — database and job directories
archive: /projects/.../arch   # or null to keep nothing

source:      [...]            # required — see 03-sources.md
generate:    {...}            # omit entirely for a structure_list campaign
screen:      {...}
reference:   {...}
filter:      {...}
dft:         {...}
analyze:     {...}
```

Every block except `name`, `machine`, `workdir` and `source` may be omitted
entirely; the defaults below apply.

**Unknown keys are rejected**, not ignored. A typo stops the campaign with the
key named, rather than silently doing something else.

## Where values come from

Five layers, later winning over earlier:

```
shipped defaults  <  machine profile's `campaign:` block  <  template
                  <  campaign.yaml  <  --set on the command line
```

`csp config show --origins` names the layer behind every resolved value, so
"where did this number come from" never requires reasoning about merge order.

### `$VAR` expansion

`$VAR` and `${VAR}` are expanded in any string, **after** all layers merge.

> **An undefined variable is a hard error**, not an empty string. Expanding
> `$SCRATCH` to `""` would turn `$SCRATCH/work` into `/work`, and that is
> exactly the class of quiet mistake this pipeline exists to remove.

Only YAML a layer actually supplied is expanded — schema defaults are not. This
has one surprising consequence worth knowing: the default MP cache is
`$CSPFLOW_REFERENCE/mp`, but **writing that literally in your campaign file** turns
an optional environment variable into a required one. Leave it out unless you
have exported `CSPFLOW_REFERENCE`.

### `--set`

```bash
csp run --set filter.e_above_hull_max=0.05
csp run --set dft.select.max_total=200 --set analyze.report=none
```

Dotted path into mappings; the value is parsed as YAML, so `[GGA]`, `true`,
`0.05` and `null` all mean what they look like. Lists are replaced whole —
there is no list-index syntax.

---

## `source` — what to search

Required, and the one block with real structure to it. See
**[Choosing your input](03-sources.md)**.

---

## `generate` — candidate structures

Omit this block entirely for a `structure_list` campaign; its absence is what
tells the driver Stage 1 has nothing to do.

```yaml
generate:
  engine: mattergen
  mattergen:
    model: /path/to/checkpoint_dir
    mode: csp
    max_batch_size: 100
    timeout_per_batch: 1800
  resources: {role: gpu, gpus: 1, time: "24:00:00"}
```

| key | default | meaning |
|---|---|---|
| `engine` | `mattergen` | the only engine implemented |
| `mattergen.model` | required | checkpoint directory |
| `mattergen.mode` | `csp` | `csp` conditions on composition (what you want); `unconditional` ignores it |
| `mattergen.max_batch_size` | 100 | structures per forward pass — lower it if the GPU runs out of memory |
| `mattergen.timeout_per_batch` | 1800 | seconds before a batch is abandoned |
| `resources` | `role: gpu, gpus: 1` | see [Resources](#resources) |

Generation refuses to run on CPU unless `CSPFLOW_ALLOW_CPU_GENERATION` is set.
Sampling on CPU is roughly 1000x slower, so a real run that took that path would
be killed by its walltime having produced nothing. The escape hatch exists for
smoke tests; which device was actually used is recorded in the results file.

## `screen` — MLIP relaxation

```yaml
screen:
  mlip: mattersim
  mattersim:
    model: MatterSim-v1.0.0-5M.pth
    fmax: 0.01
    max_steps: 500
    batch_size: 32
  dedup:
    matcher: {ltol: 0.2, stol: 0.2, angle_tol: 5.0}
  resources: {role: gpu, gpus: 1, time: "24:00:00"}
```

| key | default | meaning |
|---|---|---|
| `mlip` | `mattersim` | `mattersim`, `mace` or `uma` |
| `mattersim.model` | `MatterSim-v1.0.0-5M.pth` | checkpoint |
| `mattersim.fmax` | 0.01 | eV/Å force convergence |
| `mattersim.max_steps` | 500 | a cell still moving at this point is **kept and marked**, not thrown away |
| `mattersim.batch_size` | 32 | structures per batch |
| `dedup.matcher.ltol` | 0.2 | pymatgen `StructureMatcher` fractional length tolerance |
| `dedup.matcher.stol` | 0.2 | site displacement tolerance |
| `dedup.matcher.angle_tol` | 5.0 | degrees |

Note `max_steps` belongs to `mattersim`, not to `screen` directly.

## `reference` — the hull the candidates are measured against

```yaml
reference:
  functionals: [GGA]
  thermo_type: GGA_GGA+U
  energy_scale: raw
  mode: recompute
  prescreen_mode: mp_energies
  prescreen_hull_max: 0.20
  snapshot: true
  snapshot_id: auto
  relax_with_mlip: true
```

| key | default | meaning |
|---|---|---|
| `functionals` | `[GGA]` | which Materials Project functionals may enter the reference set |
| `thermo_type` | `GGA_GGA+U` | **pinned.** MP mixes functionals silently; a reference set containing more than one is refused rather than averaged |
| `energy_scale` | `raw` | `raw`, or `mp_corrected` to apply MP's anion corrections |
| `mode` | `recompute` | `mp_energies` takes MP's numbers as they are; `recompute` runs the reference phases through **your** DFT so both sides of the hull share one absolute scale |
| `prescreen_mode` | `mp_energies` | what Phase A's cheap hull uses |
| `prescreen_hull_max` | 0.20 | eV/atom, deliberately wide for Phase A |
| `snapshot` | `true` | freeze the MP query so the hull cannot move under you mid-campaign |
| `snapshot_id` | `auto` | `auto` stamps a new one with the date and MP release; a fixed id reproduces an old run exactly |
| `cache` | `$CSPFLOW_REFERENCE/mp` | where the frozen snapshot lives (see the `$VAR` note above) |
| `recompute_cache` | `$CSPFLOW_REFERENCE/computed` | recomputed reference energies |
| `relax_with_mlip` | `true` | relax reference phases with the same MLIP, so candidate and reference are treated alike |

> **`mode: recompute` is wired, and it refuses rather than borrows.** The DFT
> hull is built from our own recomputed MP phases at this campaign's
> `recipe_id`. A system missing one recomputed vertex is refused and named, and
> no `dft_e_above_hull` is written for it. The scale error this prevents was
> measured at ~0.19 eV/atom in Fe-Sm-Ti against a 0.06 eV/atom threshold.
>
> **The catch is `recipe_id`.** The store is keyed on your DFT policy, so a
> campaign whose policy differs by one field reads an *empty* reference set and
> every system is refused. `csp init` writes the shared store's own magnetic
> policy (`magnetic_order: ferro`, `magnetism.mode: ferromagnetic` plus its table),
> so a fresh campaign matches out of the box (D154). Change any DFT field and it
> no longer does — `csp doctor` checks this explicitly; see
> [the reference set](10-reference-set.md).
>
> Phase A gates on the MLIP hull and is unaffected.

## `calibrate` — parsed, but no longer a stage

**Calibration was removed from the funnel.** The block still parses, so an older
`campaign.yaml` keeps loading, but nothing reads it during a run and
`csp run --through calibrate` is an **error**, not a slower run.

```yaml
calibrate:
  mp:    {on_fail: warn, thresholds: {...}}
  pilot: {on_fail: off,  thresholds: {...}}   # `off` now, was `block`
```

You can delete the whole block.

Why it went: calibration existed to measure and correct the offset between the
cheap MLIP energies and real DFT. That offset was an artefact of a filter hull
built from two different energy scales. Once the hull's vertices became the
store's own MLIP energies, there was nothing left for a fitted alpha/beta to
correct — the measurement is in [the reference set](10-reference-set.md).

`filter.e_above_hull_max_source: calibrated` therefore has no fit to use. It
does not fail: it falls back to the literal value **and says so** in the log,
rather than silently continuing to call itself calibrated.

## `filter` — who is worth a VASP job

```yaml
filter:
  e_above_hull_max: 0.10
  e_above_hull_max_source: calibrated
  max_per_composition: 5
  spacegroup: {min_number: 1}
```

| key | default | meaning |
|---|---|---|
| `e_above_hull_max` | 0.10 | eV/atom. **The most consequential number in the file** — it sets Phase B's size |
| `e_above_hull_max_source` | `calibrated` | `literal` uses the number as written. `calibrated` *would* shift the cut by a fitted MLIP-vs-DFT offset — but with calibration retired there is no fit, so it falls back to literal and logs the reason. **Prefer writing `literal`:** it is what actually happens. |
| `max_per_composition` | 5 | keeps the field broad; without it one prolific composition fills the queue |
| `spacegroup.min_number` | 1 | 1 keeps everything. 3 excludes P1 and P-1, which are usually unrelaxed noise |

## `dft` — the expensive half

INCAR tags, k-points and per-step resources live in
[`recipe.yaml`](09-recipes.md). This block is everything *about* the DFT that
depends on the campaign rather than on the ladder.

```yaml
dft:
  recipe: recipe.yaml
  potcar: {tree: VASP6.4, functional: PBE_64, overrides: {}}
  rare_earth: {f_treatment: frozen, magnetic_order: ferro, reconstruct_ms: true}
  magnetism: {mode: ferromagnetic, strict: true}   # + the store's table; see the examples
  ldau: {enabled: false, ldau_type: 2, u: {}, j: {}}
  nbands: auto
  incar_overrides: {}
  max_in_flight: 200
  max_concurrent_tasks: 48
  max_cores: 3200               # cores held at once, queued AND running
  layout: runs                  # runs | stages -- see below
  combined_job: true            # every step of a structure in ONE job
  select:
    rank_by: e_above_hull_mlip
    max_per_composition: 3
    max_total: 1500
```

| key | default | meaning |
|---|---|---|
| `recipe` | `magnets` | a shipped recipe name, or a path (`recipe.yaml` beside the campaign) |
| `potcar.tree` | `VASP6.4` | `VASP6.4` or `VASP5.2` |
| `potcar.functional` | `PBE_64` | pymatgen functional label |
| `potcar.overrides` | `{}` | element → POTCAR symbol, e.g. `{Sm: Sm_3}` |
| `rare_earth.f_treatment` | `frozen` | `frozen` puts 4f in the core (tractable, one convention per campaign); `valence` needs LDA+U |
| `rare_earth.magnetic_order` | `ferri` | `ferri`, `ferro` or `none` |
| `rare_earth.reconstruct_ms` | `true` | add the frozen 4f moment back when reporting |
| `magnetism.mode` | `ferrimagnetic_retm` | RE moment antiparallel to the TM sublattice. Also `none`, `pymatgen`, `table` |
| `magnetism.strict` | `true` | fail rather than let any site take a default MAGMOM — a silent default is a wrong answer |
| `magnetism.table` | `{}` | element → initial moment, for `mode: table` |
| `magnetism.site_overrides` | `{}` | per-site moments |
| `ldau.enabled` | `false` | true only with `f_treatment: valence` |
| `ldau.u` / `ldau.j` | `{}` | element → eV, e.g. `{Sm: 6.0}` |
| `nbands` | `auto` | or an integer |
| `incar_overrides` | `{}` | applied on top of **every** recipe step, e.g. `{NCORE: 8, LREAL: Auto}` |
| `max_in_flight` | 200 | jobs this campaign has submitted at once (QOS-aware) |
| `max_cores` | `null` | cores this campaign's DFT jobs may hold at once, counting **queued as well as running**, this campaign only. The cap worth setting: the two rows above count *jobs*, and 200 jobs at a recipe's 64 ranks is 12,800 cores |
| `max_concurrent_tasks` | 48 | how many may **run** at once. It can never exceed `max_in_flight` — the throttle is `min(concurrent, in_flight)` — so raising this alone does nothing. It was the `--array %N` cap; with one job per structure there is no array, and the site's QOS enforces it directly |
| `layout` | `runs` | `runs`: one directory per structure. `stages`: the older flat one directory per structure *per step*. A campaign whose `dft/` already holds `dft-<id>-<step>/` keeps `stages` whatever this says — the directories override the config, and the reason is printed |
| `combined_job` | `true` | every step of a structure in one job; needs `layout: runs` |
| `select.rank_by` | `e_above_hull_mlip` | the order Phase B works through the candidates |
| `select.max_per_composition` | 3 | |
| `select.max_total` | 1500 | a **lifetime** ceiling on the campaign, not a per-cycle throttle. A campaign that has reached it claims nothing further, which looks exactly like a driver that has stopped |

### `layout` and `combined_job`

**`runs`** gives each structure one directory holding its seed, its script, its
log and one subdirectory per step. **`stages`** is the older flat arrangement —
one directory per structure *per step*, with the scripts, manifests, claims and
logs beside them, which put 428 entries at one level for a 94-structure
campaign. See [11-results.md](11-results.md#inside-dft).

`runs` is the default, and `csp init` also writes it explicitly — a
`campaign.yaml` is read by people as well as by code, and this is the setting
that decides where everything ends up.

An older campaign is not disturbed by that default, because the directories
override it: if `dft/` already holds `dft-<id>-<step>/`, `stages` is used
whatever the config says, and the reason is printed. Configuration is an
intention; the directories are a fact. Switching a live campaign would point
every path at an empty directory — reporting nothing done, and re-running
calculations that are sitting complete on disk.

**`combined_job`** runs every step of a structure in one job instead of one job
per step. Two reasons:

* the queue wait between a relax finishing and its static starting was **54 min
  median, 94 mean, 372 at worst** over 29 measured pairs — 45.7 hours of idle
  across those structures alone, computing nothing;
* the job becomes the same unit as the structure. That is what lets one
  structure go out as **one sbatch** with its own name, its own walltime and its
  own memory, instead of sharing an array's allocation with structures that
  needed different ones — which had put 13 relaxes under a static's 12-hour cap
  in a single array.

The job asks for the **sum** of its steps' walltimes (a 24 h relax plus a 12 h
static is a 36 h job) and the **maximum** of their memory and rank counts, since
those are held at once rather than consumed in turn. It skips any step whose own
OUTCAR says it already converged, so a failed static never costs its relax, and
it records which step failed so the retry ladder picks a remedy that fits.

Set `combined_job: false` to keep one job per step. It has no effect under
`layout: stages`, which has nowhere to put a structure's two steps together.

## `analyze` — properties and the report

```yaml
analyze:
  properties: [m_dft_raw, m_s_reconstructed, volume, spacegroup]
  report: html
```

| key | default | meaning |
|---|---|---|
| `properties` | as above | what is extracted from each finished calculation |
| `report` | `html` | `html` or `none`. `csp report` writes `report/` |

---

## Resources

Every stage that submits jobs takes a `resources:` block:

```yaml
resources:
  role: gpu           # a partition role defined in machine.yaml
  ntasks: 64
  cpus_per_task: 1
  gpus: 1
  mem: 64G
  time: "24:00:00"
```

`role` is the indirection that keeps a campaign portable: it names a role
(`cpu`, `gpu`, `bigmem`, …) that [`machine.yaml`](08-machines.md) maps to real
partition names. Anything you leave out falls back to that machine's defaults.

## Top-level keys

| key | required | meaning |
|---|---|---|
| `name` | yes | campaign name, used in job names and the report title |
| `machine` | yes | path to a machine profile, or a shipped name (`orion`, `generic_slurm`, `local`). A relative path resolves against the campaign folder |
| `workdir` | yes | where `campaign.db` and all job directories go. Put it on scratch — it gets large |
| `archive` | no | where finished output is archived; `null` keeps nothing |


---

## See also

* [Every setting, as one table](06-settings.md) — the same file in reference form,
  with every default spelled out
* [A-to-Z workflows](04-workflows.md) — what to actually type, per source mode
* [The DFT recipe](09-recipes.md) — `recipe.yaml`, which `dft.recipe` points at
* [Running on your cluster](08-machines.md) — `machine.yaml`
