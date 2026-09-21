# 04 — A-to-Z workflows for all three input modes

> **Guide 04 of 13** · Previous: [03 — Choosing your input](03-sources.md) · [Guide home](README.md) · Next: [05 — `campaign.yaml` explained](05-campaign.md)

One walkthrough per way of getting structures into the funnel, from an empty
directory to a report. The three differ **only in the source stage**; everything
after it is identical, which is why they are on one page.

| mode | you provide | generation | needs a GPU | start here if |
|---|---|---|---|---|
| [`structure_list`](#c-structure_list) | the structures | none | screening only | you already have candidates — substitutions, strained cells, CIFs from a paper |
| [`composition_list`](#b-composition_list) | the formulas | MatterGen, per formula | yes | you know which phases you care about |
| [`chemical_space`](#a-chemical_space) | element groups and rules | MatterGen, per derived formula | yes | you are exploring, not checking |

**If you are new, read `structure_list` first.** It has the fewest moving parts
and is the shortest path from `csp run` to a finished VASP job.

There are working examples of all three in
[`campaigns/tests/`](../campaigns/tests/), sized to run in an afternoon.

---

## The shape of every run

```
source ──► generate ──► screen ──► dedup ──► reference ──► filter ──► dft ──► analyze
           (skipped for structure_list)
```

* `source` expands your configuration into rows in the database.
* `generate` makes structures with MatterGen (GPU).
* `screen` relaxes them with an MLIP and `dedup` removes duplicates (GPU).
* `reference` assembles the convex hull the filter will cut against.
* `filter` decides which structures are worth DFT.
* `dft` runs VASP — **one job per structure**, relax and static inside it.
* `analyze` computes moments and volumes and writes the report.

`generate` is optional: a `structure_list` campaign has no `generate:` block and
the driver does not complain.

> **There is no `calibrate` stage.** It was removed. `csp run --through calibrate`
> is an error, not a slower run. If you find that command in an old note or
> script, replace it with `--through reference`.

---

## Before any of them: the four things that must be true

Run these once, and again whenever something surprises you.

```bash
conda activate cspflow
csp version
```

**1. The reference store is reachable and covers your chemistry.** This is what
decides whether a run takes an afternoon or a fortnight: a chemical system
already in the store costs seconds, one that is not costs a reference DFT
campaign of its own.

```bash
python - <<'PY'
import csv, collections
rows = list(csv.DictReader(open('/projects/mmi/cspflow-shared/store/index.csv')))
by = collections.defaultdict(lambda: [0, 0])
for r in rows:
    by[r['chemsys']][0] += 1
    if r['ready'].strip().lower() == 'true':
        by[r['chemsys']][1] += 1
for s in ('Ce-Fe', 'Fe-Sm'):            # <-- your systems here
    n, ok = by.get(s, [0, 0])
    print(f'{s:<10} {ok}/{n} ready')
PY
```

**2. Your `dft:` block matches the store's.** Every energy on one hull has to
come from the same DFT settings. `csp doctor` compares the two and must say:

```
[OK  ] reference recipe: matches the store (6e44a4bf351d98a9) -- the hull is on one scale
```

If it warns instead, the hull still builds, still looks correct, and **ranks
wrongly**. The fix is to copy the store's `dft:` block into your campaign
verbatim.

**3. The POTCARs resolve and ENCUT clears them.**

```bash
csp doctor --elements Ce,Fe
```

**4. You are not about to use a partition that kills VASP.** Our `vasp_std` is
compiled with AVX-512. The `cpu` role in `orion.yaml` carries a `constraint`
that excludes nodes without it; the `short` (Nebula) role does **not**. Keep
DFT on `role: cpu`.

---

## A. `chemical_space`

**You name groups of elements and how many to draw from each.** The source stage
assembles every chemical system the rules allow, expands each into
stoichiometries, and hands each to MatterGen.

### A1. Create it

```bash
csp init 1 my-sweep -m orion
cd my-sweep
```

### A2. Describe the space

```yaml
source:
  - mode: chemical_space
    name: sweep
    chemical_space:
      groups:
        A: {elements: [Ce, Sm], pick: 1}
        B: {elements: [Fe, Co], pick: 1, min_fraction: 0.75}
      max_atoms_formula: 8
      max_rare_earth: 1
    defaults:
      z: {min: 1, max: 1}
      max_atoms: 20
      n_structures: {mode: fixed, count: 8}
```

**Check the size before you run it.** `max_atoms_formula` controls how many
stoichiometries get enumerated, and the count grows faster than it looks. For a
single Ce-Fe system:

| `max_atoms_formula` | 2 | 3 | 4 | 6 | 8 |
|---|--:|--:|--:|--:|--:|
| compositions | 1 | 3 | 5 | 11 | 21 |

Multiply by `n_structures.count` and by the number of systems your groups
produce. That is how many structures you are asking MatterGen for.

### A3. Point `generate` at a GPU

```yaml
generate:
  engine: mattergen
  mattergen: {model: /projects/mmi/shuo/MatterGen_checkpoints/18-55-08, mode: csp}
  resources: {role: gpu, gpus: 1, time: "12:00:00"}
```

`resources.role` defaults to `cpu`. Generation on a CPU is the most common way
a first sweep appears to hang.

### A4. Expand the source and look at it before spending anything

```bash
csp source                 # writes composition rows; submits nothing
csp status
```

This is the cheap checkpoint. If the composition count is not what you expected,
fix the groups now — before a GPU queue wait.

### A5. Run

```bash
csp doctor
sbatch /projects/mmi/Ridwan/cspflow/scripts/campaign_driver.sbatch . "" 120
```

Then jump to [Watching a run](#watching-a-run).

---

## B. `composition_list`

**You name the formulas outright.** No rules, no derivation. Use this when you
already know which phases matter.

### B1. Create and describe

```bash
csp init 2 my-shortlist -m orion && cd my-shortlist
```

```yaml
source:
  - mode: composition_list
    name: shortlist
    composition_list:
      items:
        - formula: SmFe11Ti
          z: [1, 1]
          n_structures: {mode: fixed, count: 150}   # spend more on what matters
        - formula: Sm3Fe29Ti2
          z: [1, 1]
          n_structures: {mode: fixed, count: 60}
    defaults:
      z: {min: 1, max: 2}
      max_atoms: 60
      n_structures: {mode: fixed, count: 40}
```

Per-item `n_structures` is the point of this mode: an even spread across a
shortlist wastes effort on the formulas you included for completeness.

For a long list, put it in a CSV instead:

```yaml
      from_file: inputs/compositions.csv     # formula[,z_min,z_max,n_structures]
```

`items` and `from_file` merge, so you can keep the bulk in the file and the
special cases inline.

### B2. Same as A3-A5

`generate` needs a GPU; `csp source` first; then the driver.

---

## C. `structure_list`

**You supply the structures.** No generation stage at all — seeds go straight to
MLIP screening. This is what the production CePdGe and CeFeB campaigns use.

### C1. Create it and put the seeds in

```bash
csp init 3 my-seeds -m orion && cd my-seeds
cp /somewhere/*.vasp inputs/seeds/   # inputs/seeds/ starts empty; subfolders are searched too
```

Accepted: POSCAR/`.vasp`, CIF. A folder is walked; globs work too.

**Name the files meaningfully.** The filename becomes the seed label that ends
up in the run directory name and the job name, so
`02-agentic__Ce2Fe11Co3B_x3_o5-7_Co.vasp` becomes a directory you can read at a
glance. A leading `01-`/`02-` ordering prefix is stripped; the word before `__`
is kept, because it names the provenance.

### C2. Describe them

```yaml
source:
  - mode: structure_list
    name: seeds
    structure_list:
      paths: [inputs/seeds]     # relative to the CAMPAIGN folder
      relax: true               # MLIP-relax each seed before screening
      dedup: warn               # warn | drop | off
      max_atoms: 60             # refused at read time, with the file named
```

There is **no `generate:` block**. Do not add an empty one.

### C3. Check what was read

```bash
csp source
csp status
```

Every seed should appear. A missing one was refused — `max_atoms`, or a file the
reader did not recognise — and `csp status` says which.

### C4. Run

```bash
csp doctor --elements Ce,Fe
sbatch /projects/mmi/Ridwan/cspflow/scripts/campaign_driver.sbatch . "" 60
```

---

## Watching a run

```bash
csp status                    # the driver's own snapshot; works from any node
squeue --me
tail -f csp-driver-*.log
```

`csp status` reads a status file the driver writes each cycle, not the database
— a campaign database on scratch cannot be read from another node while it is
being written.

**DFT jobs are named after the structure they are running**, one job per
structure:

```
26940187   CePdGe-0195-Ce2PdGe6-agentic_x3_o5-7_Cu   RUNNING
           -> dft/runs/0195-Ce2PdGe6-agentic_x3_o5-7_Cu/
```

The campaign name is in there because `squeue` is per user, not per campaign,
and every campaign numbers its structures from 1.

A deeper look, reconciling the database against SLURM against the filesystem:

```bash
python /projects/mmi/Ridwan/cspflow/scripts/campaign_audit.py --campaign .
```

### Running it in two phases

The driver takes a stage slice. For a large campaign it is normal to run the
cheap part to completion first, look at it, and only then commit DFT:

```bash
sbatch .../campaign_driver.sbatch . "--through reference" 120   # Phase A
# ... inspect ...
sbatch .../campaign_driver.sbatch . "--from filter" 300         # Phase B
```

Pausing is a file, so it needs no signal and survives a driver restart:

```bash
echo "waiting on the Ti POTCAR question" > PAUSED
rm PAUSED
```

---

## Reading the result

```bash
csp report
```

writes `report/report.html` (self-contained — no network, emailable) and
`report/candidates.csv`.

On disk, one directory per structure:

```
dft/
├── .potcars/POTCAR-21df8b3b98d8      shared; 109 copies became 8 files
├── claims/
└── runs/
    └── 0088-Ce8Fe56B4-strain_Ce2Fe14B_strainiso_p0.00pct/
        ├── structure.json            id, formula, seed, steps — no DB needed
        ├── SLURM_TASK                the job id that ran this directory
        ├── <jobname>.sbatch          exactly as submitted
        ├── slurm-26940187.out
        ├── relax/                    INCAR KPOINTS POSCAR CONTCAR OUTCAR …
        └── static/                   prepared during the job, from relax/CONTCAR
```

### Two results that look like failures and are not

* **`m_dft_raw ≈ 0` on a Ce or Sm compound.** With `f_treatment: frozen` the 4f
  electrons are in the POTCAR core, so they cannot contribute a computed moment.
  The computed moment lives on the transition metal; the 4f part appears
  separately as `m_s_reconstructed`, which is Hund's-rule bookkeeping, not a
  measurement.
* **A structure that "converged" but is listed as not usable.** A relaxation
  that stopped at the ionic step limit exits cleanly. It is not a crash and it is
  not a relaxed geometry.

---

## Starting over

Everything a run creates is under `workdir`. The campaign folder is input.

```bash
# cancel anything still queued FIRST
squeue --me -o "%i %j"
scancel <ids>

rm -rf /scratch/$USER/cspflow/my-campaign     # the true reset
rm -f  csp-driver-*.log
rm -rf report
```

To rerun only the failed structures after fixing an *environmental* cause — a
bad node excluded, a module repaired — rather than starting over:

```bash
python /projects/mmi/Ridwan/cspflow/scripts/requeue.py -c . --reason "no OUTCAR"
python /projects/mmi/Ridwan/cspflow/scripts/requeue.py -c . --reason "no OUTCAR" --apply
```

Without `--apply` it is a dry run. This is deliberately separate from the DFT
retry ladder, which refuses to retry a failure a recipe change cannot fix.

`requeue.py` has two other selectors: `--dead-jobs` for after a mass `scancel`,
and `-s <id>` for one structure. Adding `--from-seed` discards what is on disk
(moved to `discarded/`, never deleted) and starts that structure over from its
seed -- use it when the previous attempt is the thing you are throwing away.

---

## When something goes wrong

| symptom | first thing to check |
|---|---|
| the driver submits nothing, stages all pending | `dft.select.max_total` — it is a **lifetime** ceiling, not a per-cycle one |
| raising `max_concurrent_tasks` changed nothing | it can never exceed `max_in_flight`; the throttle is `min(concurrent, in_flight)` |
| generation seems to hang | `generate.resources.role` — it defaults to `cpu` |
| zero candidates after `filter` | `filter.e_above_hull_max` too tight, or the hull is on a different scale from the candidates |
| `unknown stage 'calibrate'` | an old `--through calibrate`; use `--through reference` |
| VASP dies instantly, no OUTCAR | an AVX-512-less node; keep DFT on `role: cpu` |

[13-troubleshooting.md](13-troubleshooting.md) has the longer list.

## See also

* [06-settings.md](06-settings.md) — every key in `campaign.yaml`, with defaults
* [09-recipes.md](09-recipes.md) — the DFT recipe
* [10-reference-set.md](10-reference-set.md) — the shared store and why one scale matters
* [`campaigns/tests/`](../campaigns/tests/) — a small, runnable example of each mode
