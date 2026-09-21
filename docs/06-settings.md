# 06 — Every setting in `campaign.yaml`

> **Guide 06 of 13** · Previous: [05 — `campaign.yaml` explained](05-campaign.md) · [Guide home](README.md) · Next: [07 — Pipeline stages](07-stages.md)

One page, in file order, with what each key does, what it defaults to, and —
where it matters — what goes wrong if you get it wrong.

**Only four keys are genuinely required:** `name`, `machine`, `workdir`, and
`source`. Every block below them has defaults, so the shortest valid campaign
file is about six lines. `csp init <type> <name>` writes the long version
instead — the example campaign for that type, every live setting written out with
its alternatives beside it — because a setting you cannot see is a setting you
cannot change. This page also lists keys the examples leave out: some only apply
to another type, and some are accepted but read by nothing (D146).

Conventions used here:

* **default** is what you get if you omit the key entirely;
* *(required)* means the campaign will not load without it;
* eV/atom is the unit for every energy threshold.

To see what your file actually resolved to, including defaults you never wrote:

```bash
csp config show          # the merged, resolved configuration
csp doctor               # that, plus whether the machine can actually run it
```

---

## Top level

```yaml
name: my-campaign
machine: orion
workdir: /scratch/$USER/cspflow/my-campaign
archive: null
```

| key | default | what it does |
|---|---|---|
| `name` | *(required)* | The campaign's name. It prefixes every DFT job name, so it is what you will read in `squeue`. |
| `machine` | *(required)* | A shipped profile (`orion`, `generic_slurm`, `local`) **or a path** to your own `machine.yaml`. `csp init` writes you a local copy. |
| `workdir` | *(required)* | Where the database, the job directories and all VASP output live. `$USER` is expanded. **Everything a run creates is under here** — deleting it is a clean reset. |
| `archive` | `null` | A second location the database and report are mirrored to at every stage boundary. `null` keeps nothing. Set it if `workdir` is on a scratch filesystem that gets purged: a purge then costs compute but never provenance. |

> **The campaign folder is input; `workdir` is output.** Nothing writes back into
> the folder holding `campaign.yaml` except `csp report` (into `report/`) and the
> driver's own log. That separation is what makes "start over" a one-line
> `rm -rf` of the workdir.

---

## `source` — where structures come from

`source` is a **list**, so one campaign can draw from several places. Each entry
picks one `mode` and fills in the matching block.

```yaml
source:
  - mode: structure_list      # chemical_space | composition_list | structure_list
    name: seeds               # required once there is more than one source
    structure_list: {...}
    defaults: {...}
```

| key | default | what it does |
|---|---|---|
| `mode` | *(required)* | Which of the three below. |
| `name` | `default` | Labels the rows this source produced. Required once a campaign has more than one source, so you can tell them apart later. |
| `defaults` | see below | Applied to every row this source makes, unless the row overrides it. |

### `source.defaults`

| key | default | what it does |
|---|---|---|
| `z.min` / `z.max` | `1` / `1` | How many formula units to build. `z: [1, 2]` is shorthand for `{min: 1, max: 2}`. |
| `max_atoms` | `40` | Hard cap on `z × atoms-per-formula-unit`. A row that would exceed it is dropped, not silently shrunk. |
| `n_structures.mode` | `per_atom` | `per_atom` scales the count with cell size; `fixed` does not. |
| `n_structures.structures_per_atom` | `2.0` | `per_atom` only. A 20-atom cell → 40 structures. |
| `n_structures.count` | `null` | `fixed` only. Structures per row, regardless of size. |
| `n_structures_scope` | `per_z` | Whether the count above is per z value or shared across all of them. |

> **Use `fixed` for a test campaign.** `per_atom` makes the generation cost
> depend on cell size, which is exactly what you cannot predict in advance.

#### `z` and `max_atoms` overlap — know which one is doing the work

Cell size is `z × atoms-per-formula-unit`, so for a **single** formula the two
say the same thing and `max_atoms` is redundant.

They separate as soon as the formulas **vary in size**, which is the whole point
of a sweep. `z` is a per-formula *request*; `max_atoms` is the only *cross-formula*
cap, because atoms-per-formula-unit is different for each:

| formula | atoms/f.u. | z=1 | z=2 |
|---|--:|--:|--:|
| CeFe | 2 | 2 | 4 |
| CeFe5 | 6 | 6 | 12 |
| Ce2Fe17 | 19 | 19 | 38 |

With `z: [1, 2]` and `max_atoms: 20` you keep both CeFe cells, both CeFe5 cells,
and only `z=1` of Ce2Fe17. No `z` range alone expresses that. A `z` the cap
rejects is dropped **with a reason**, never silently clipped — "I asked for Z up
to 4" does not quietly become "I got Z up to 2".

**So: pin `z` and you do not need `max_atoms`; sweep formulas of different sizes
and you do.** Setting a cap that cannot be reached is worse than setting none —
it reads like a safeguard while being decoration. For `mode: chemical_space`,
check `max_atoms_formula` first: it caps the reduced formula, so with `z: 1` it
already fixes the cell size on its own.

Note that `structure_list.max_atoms` is a different thing despite the name: it
is a **read-time refusal** on the seed files, and it is worth setting even when
today's seeds are small, because it fires the day someone drops a 200-atom CIF
into `inputs/`.

### `mode: chemical_space` — you name element groups, it derives the systems

```yaml
chemical_space:
  groups:
    A: {elements: [Ce, Sm], pick: 1}
    B: {elements: [Fe, Co], pick: 1, min_fraction: 0.75}
  max_atoms_formula: 20
  max_rare_earth: 1
```

| key | default | what it does |
|---|---|---|
| `groups.<X>.elements` | *(required)* | The pool to draw from. |
| `groups.<X>.pick` | *(required)* | How many to take. An int, or a list of arities — `[1, 2]` means "one or two". |
| `groups.<X>.min_fraction` | `null` | Minimum share of the atoms in the reduced formula that this group must hold. |
| `groups.<X>.max_fraction` | `null` | The other side of the same constraint. |
| `max_atoms_formula` | `20` | Cap on the **reduced** formula (Sm2Fe17 is 19). |
| `max_rare_earth` | `1` | Max rare-earth *species* in the assembled system. Deliberately separate from `pick`: nothing otherwise stops two groups both contributing a rare earth. |

> **`max_atoms_formula` is the expensive knob.** It controls how many
> stoichiometries get enumerated, and the count grows faster than it looks. For
> a single Ce-Fe system: a cap of 2 gives 1 composition, 3 gives 3, 4 gives 5,
> 6 gives 11, 8 gives **21**. Multiply by `n_structures` to get the number of
> structures you are asking the generator for.

### `mode: composition_list` — you name the formulas

```yaml
composition_list:
  items:
    - formula: SmFe11Ti
      z: [1, 1]
      max_atoms: 26
      n_structures: {mode: fixed, count: 150}
  from_file: inputs/compositions.csv
```

| key | default | what it does |
|---|---|---|
| `items[].formula` | *(required)* | The reduced formula. |
| `items[].z` | from `defaults` | Formula-unit range for this item only. |
| `items[].max_atoms` | from `defaults` | Cap for this item only. |
| `items[].n_structures` | from `defaults` | Count for this item only — spend more on the phase you care most about. |
| `from_file` | `null` | A CSV of `formula[,z_min,z_max,n_structures]`, merged with `items`. Use it when the list is long enough that it belongs in its own file. |

### `mode: structure_list` — you supply the structures

```yaml
structure_list:
  paths: [inputs/seeds]
  relax: true
  dedup: warn
  max_atoms: 60
```

| key | default | what it does |
|---|---|---|
| `paths` | *(required)* | POSCAR/CIF files, globs, or folders (a folder is walked). Relative paths resolve against the **campaign folder**, not your shell's cwd. |
| `relax` | `true` | MLIP-relax each seed before screening. Turn it off only if the seeds are already relaxed at the same level. |
| `dedup` | `warn` | `warn` reports seed collisions and keeps them; `drop` removes them; `off` skips the check. |
| `max_atoms` | `source.defaults.max_atoms` | A seed bigger than this is refused **at read time**, with the file named — not silently carried to DFT. |

> **This is the only mode with no `generate:` block.** Seeds go straight to
> screening, so it needs no GPU for generation and no MatterGen checkpoint.

---

## `generate` — making structures (omit entirely for `structure_list`)

```yaml
generate:
  engine: mattergen
  mattergen:
    model: /path/to/checkpoint
    mode: csp
    max_batch_size: 100
    timeout_per_batch: 1800
  resources: {role: gpu, gpus: 1, time: "24:00:00"}
```

| key | default | what it does |
|---|---|---|
| `engine` | `mattergen` | The only engine implemented. |
| `mattergen.model` | *(required)* | Checkpoint directory. `csp doctor` checks its layout and whether it was CSP-trained. |
| `mattergen.mode` | `csp` | `csp` conditions on the composition you asked for. `unconditional` ignores it — rarely what you want here. |
| `mattergen.max_batch_size` | `100` | Structures per forward pass. Lower it if the GPU runs out of memory. |
| `mattergen.timeout_per_batch` | `1800` | Seconds before a batch is abandoned. Fail fast rather than hold a GPU. |
| `resources` | `{role: cpu, time: "24:00:00"}` | **Set `role: gpu, gpus: 1`** — the default role is `cpu` and generation needs a GPU. |

Omitting `generate:` is legitimate and expected for `structure_list`: the driver
treats `generate` as an optional stage and does not complain.

---

## `screen` — MLIP relaxation and deduplication

```yaml
screen:
  mlip: mattersim
  mattersim: {model: MatterSim-v1.0.0-5M.pth, fmax: 0.01, max_steps: 500, batch_size: 32}
  dedup: {matcher: {ltol: 0.2, stol: 0.2, angle_tol: 5.0}}
  resources: {role: gpu, gpus: 1, time: "24:00:00"}
```

| key | default | what it does |
|---|---|---|
| `mlip` | `mattersim` | `mattersim` is what is installed and tested here. |
| `mattersim.model` | `MatterSim-v1.0.0-5M.pth` | Checkpoint name. |
| `mattersim.fmax` | `0.01` | eV/Å force convergence. Loosen to `0.05` for a smoke test. |
| `mattersim.max_steps` | `500` | A cell still moving at this point is kept and flagged, not discarded. |
| `dedup.matcher.ltol` | `0.2` | pymatgen `StructureMatcher` fractional length tolerance. |
| `dedup.matcher.stol` | `0.2` | Site-displacement tolerance. |
| `dedup.matcher.angle_tol` | `5.0` | Degrees. |

> **There is no `batch_size`.** MLIP relaxation runs **one structure at a time**.
> The setting existed, defaulted to 32, and was read by nothing — so it described
> GPU batching that does not happen. MatterSim's own `BatchRelaxer` is unusable
> against the installed ASE (two independent breakages, one inside mattersim's
> own loop) and accepts no step limit, so an unconvergeable structure would have
> nothing to stop it. `ASE`'s optimizer is used instead: it honours `max_steps`
> and lets cspflow own the convergence test. An old `campaign.yaml` that still
> sets `batch_size` loads fine and warns; delete the line.

> **The screening MLIP must be the same one the filter hull was built with.**
> Mixing them puts candidates and hull on different scales, which looks fine and
> ranks wrongly.

---

## `reference` — the hull the filter cuts against

```yaml
reference:
  functionals: [GGA]
  thermo_type: GGA_GGA+U
  energy_scale: raw
  mode: recompute
  energy_source: dft
  prescreen_mode: mp_energies
  prescreen_hull_max: 0.20
  snapshot: true
  relax_with_mlip: true
```

| key | default | what it does |
|---|---|---|
| `functionals` | `[GGA]` | Which MP functionals to accept. |
| `thermo_type` | `GGA_GGA+U` | **Pinned.** A reference set containing more than one is refused. MP mixes functionals; a hull built across the mix is not a hull. |
| `energy_scale` | `raw` | `raw` or `mp_corrected` (MP's anion corrections). |
| `mode` | `recompute` | `recompute` builds the hull from **our own** DFT in the shared store. `mp_energies` takes MP's numbers as they are. |
| `energy_source` | `dft` | Which of the store's three energies builds the hull when `mode: recompute` — `dft`, `mlip`, or `mp`. Every stored structure carries all three, and each defines a different hull. |
| `prescreen_mode` | `mp_energies` | What the cheap early cut uses. |
| `prescreen_hull_max` | `0.20` | eV/atom, deliberately wide — this cut happens before anything expensive. |
| `snapshot` | `true` | Freezes the MP query. **Leave this on.** Without it `e_above_hull` moves with no change to any of your inputs, which is indistinguishable from a bug. |
| `snapshot_id` | `auto` | `auto` stamps a new one with the date and MP release; a fixed id reproduces an old run exactly. |
| `cache` / `recompute_cache` | `$CSPFLOW_REFERENCE/...` | Where the MP query and the recomputed energies are kept. |
| `relax_with_mlip` | `true` | Relax reference phases with the same MLIP, so reference and candidates are treated identically. |

> **The single most consequential thing on this page.** Every energy on one hull
> must come from the same DFT settings. `csp doctor` compares your `dft:` block
> against the store's and says either `matches the store (<hash>) -- the hull is
> on one scale` or warns that they differ. If it warns, **the hull still builds,
> still looks correct, and ranks wrongly.** See [10-reference-set.md](10-reference-set.md).

---

## `calibrate` — parsed, but no longer a stage

```yaml
calibrate:
  mp:    {on_fail: warn, thresholds: {...}}
  pilot: {on_fail: off,  thresholds: {...}}
```

**Calibration was removed from the funnel.** The block still parses so that
older campaign files load, and `csp run --through calibrate` is now an **error**,
not a slower run. The measurement that retired it is in
[10-reference-set.md](10-reference-set.md): once the filter hull is built on one
scale, there is nothing left for a fitted offset to correct.

You can delete the whole block. If you keep it, `pilot.on_fail` now defaults to
`off` rather than `block`.

---

## `filter` — which structures are worth DFT

```yaml
filter:
  e_above_hull_max: 0.10
  e_above_hull_max_source: calibrated
  max_per_composition: 5
  spacegroup: {min_number: 1}
```

| key | default | what it does |
|---|---|---|
| `e_above_hull_max` | `0.10` | eV/atom above the hull. **The single most consequential knob in the file.** |
| `e_above_hull_max_source` | `calibrated` | `literal` uses the number as written. `calibrated` would rescale it by a fitted MLIP-vs-DFT offset — but with calibration retired there is no fit, so it **falls back to literal and says so** in the log. Prefer writing `literal` outright: it is what actually happens. |
| `max_per_composition` | `5` | Keeps the field broad. Without it one prolific composition can fill the entire DFT budget. |
| `spacegroup.min_number` | `1` | `1` keeps everything. `3` excludes P1 and P-1. |

---

## `dft` — the expensive stage

```yaml
dft:
  recipe: magnets
  layout: runs
  combined_job: true
  potcar:     {tree: VASP6.4, functional: PBE_64, overrides: {}}
  rare_earth: {f_treatment: frozen, magnetic_order: ferri, reconstruct_ms: true}
  magnetism:  {mode: ferrimagnetic_retm, strict: true}
  ldau:       {enabled: false, ldau_type: 2, u: {}, j: {}}
  nbands: auto
  incar_overrides: {}
  max_in_flight: 200
  max_concurrent_tasks: 48
  select: {...}
```

| key | default | what it does |
|---|---|---|
| `recipe` | `magnets` | A shipped recipe name, **or a path** such as `recipe.yaml`. A relative path resolves against the campaign folder. See [09-recipes.md](09-recipes.md). |
| `layout` | `runs` | `runs`: one directory per structure, holding its script, its log and one subdirectory per step. `stages`: the older flat arrangement, one directory per structure *per step*. **A campaign whose `dft/` already holds `dft-<id>-<step>/` directories keeps `stages` whatever you write here** — the directories override the config, and the reason is printed. |
| `combined_job` | `true` | Every step of a structure in one job, instead of one job per step. Removes the queue wait between relax and static (54 min median over 29 measured pairs). Requires `layout: runs`. |
| `nbands` | `auto` | Or an integer. |
| `incar_overrides` | `{}` | Applied on top of every recipe step. Free-form — there is no whitelist. Unknown tags are written as given but warn. |
| `max_in_flight` | `200` | How many jobs this campaign has submitted at once. Counts **jobs**, which is usually not the quantity you want to cap — see `max_cores`. |
| `max_concurrent_tasks` | `48` | How many may **run** at once. **It can never exceed `max_in_flight`** — the throttle is `min(concurrent, in_flight)`, so raising this alone does nothing. |
| `max_cores` | `null` | **The cap most people actually want.** Total cores this campaign's DFT jobs may hold at once, counting **queued as well as running**. The two above count jobs: at a recipe's 64 ranks, `max_in_flight: 200` is 12,800 cores, and nothing on a typical `cpu` partition caps that. Queued jobs count because they will occupy those cores — a cap that ignored them would approve a submission the queue has already spent. Only this campaign's own jobs are counted, never anyone else's, and the figure is recomputed every cycle, so it stays right if the recipe's rank count changes or a retry rung raises it. `null` = no core cap. |

### `dft.potcar`

| key | default | what it does |
|---|---|---|
| `tree` | `VASP6.4` | `VASP6.4` or `VASP5.2`. |
| `functional` | `PBE_64` | pymatgen functional label. |
| `overrides` | `{}` | element → POTCAR symbol, e.g. `{Sm: Sm_3}`. |

### `dft.rare_earth`

| key | default | what it does |
|---|---|---|
| `f_treatment` | `frozen` | `frozen` puts the 4f electrons in the POTCAR core (Ce_3, Sm_3) — tractable, and the convention the shared store uses. `valence` computes them. |
| `magnetic_order` | `ferri` | `ferri`, `ferro` or `none`. |
| `reconstruct_ms` | `true` | Add the Hund's-rule 4f moment back when reporting, as a **separate** number. |

> **The 4f trap.** With `f_treatment: frozen`, `m_dft_raw` is ~0 for a Ce or Sm
> compound **by construction** — the 4f electrons are not in the calculation.
> That is correct behaviour, not a failed run. `m_s_reconstructed` is Hund's-rule
> bookkeeping laid on top, not a measurement. The two are reported separately
> and the schema refuses to merge them.

### `dft.magnetism`

| key | default | what it does |
|---|---|---|
| `mode` | `ferrimagnetic_retm` | `ferrimagnetic_retm` puts the RE moment antiparallel to the TM sublattice. Also `ferromagnetic`, `table`, `pymatgen`, `none`. |
| `strict` | `true` | **Fail rather than let any site take a default MAGMOM.** Leave this on: a silently-defaulted initial moment is a wrong answer that looks like a right one. |
| `table` | `{}` | element → initial moment, for `mode: table`. |
| `site_overrides` | `{}` | Per-site moments. |

### `dft.ldau`

| key | default | what it does |
|---|---|---|
| `enabled` | `false` | Set `true` only with `f_treatment: valence`. |
| `u` / `j` | `{}` | element → value in eV. |
| `ldau_type` | `2` | |

### `dft.select` — who gets a VASP job, and in what order

| key | default | what it does |
|---|---|---|
| `rank_by` | `e_above_hull_mlip` | The order Phase B works through the candidates. |
| `max_per_composition` | `3` | |
| `max_total` | `1500` | **A lifetime ceiling on the campaign, not a per-cycle throttle.** Once this many structures have entered DFT, the campaign claims nothing further — which looks exactly like "the driver stopped working". |

> **`budget_core_hours` is retired** (D143) and is ignored with a warning if
> your `campaign.yaml` still sets it. It gated submission on *projected* spend,
> and the projection for a stage with no finished job yet is the **requested
> walltime** — the ceiling, not the cost. On one campaign that made a 20,000
> core-hour budget print `BUDGET REACHED` on the first cycle with 8 of 150
> structures out and nothing overspent. Cap cores instead: `dft.max_cores`
> measures what is held right now rather than extrapolating from one job.

---

## `analyze`

```yaml
analyze:
  properties: [m_dft_raw, m_s_reconstructed, volume, spacegroup]
  report: html
```

| key | default | what it does |
|---|---|---|
| `properties` | the four above | What Stage 7 computes. `m_dft_raw` and `m_s_reconstructed` must be listed separately; asking for a merged `m_s` is refused. |
| `report` | `html` | `html` writes `report/report.html` on `csp report`; `none` skips it. |

---

## The six settings most often wrong

1. **`dft.max_total`** — a lifetime ceiling. A campaign that has stopped
   claiming has usually just reached it.
2. **`dft.max_cores` left unset** — then only *job counts* cap the run, and a
   job count is not cores: at a recipe's 64 ranks, `max_in_flight: 200` is
   12,800 cores. Set it to the cores you are willing to hold.
3. **`dft.max_concurrent_tasks` raised without `max_in_flight`** — has no
   effect, because the throttle is `min(concurrent, in_flight)`.
4. **`generate.resources.role`** — defaults to `cpu`; generation needs `gpu`.
5. **A `dft:` block that disagrees with the reference store** — the hull builds,
   looks correct, and ranks wrongly. `csp doctor` tells you.
6. **`filter.e_above_hull_max` too tight on a small campaign** — ends with zero
   candidates and no explanation.

## See also

* [04-workflows.md](04-workflows.md) — A-to-Z walkthroughs for all three source modes
* [05-campaign.md](05-campaign.md) — the same file, organised as prose with the reasoning
* [09-recipes.md](09-recipes.md) — the DFT recipe, which is a separate file
* [08-machines.md](08-machines.md) — partitions, QOS and constraints
