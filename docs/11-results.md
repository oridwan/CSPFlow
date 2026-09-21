# 11 — Reading the results

> **Guide 11 of 13** · Previous: [10 — The reference set](10-reference-set.md) · [Guide home](README.md) · Next: [12 — Command reference](12-cli.md)

## Where things are

```
my-campaign/
├── campaign.yaml
├── report/          report.html + candidates.csv   ← csp report writes here
└── results/  ──────────────────────────────┐
                                            │ symlink, made on the first run
$workdir/                        ←──────────┘
├── campaign.db      everything: structures, states, jobs, energies
├── status.json      how many are where — rewritten every cycle
├── structures.csv   what happened to EACH ONE — rewritten every cycle
├── generate/        one directory per submission
├── screen/          one directory per submission — see below
└── dft/             one directory per STRUCTURE — see below
```

`status.json` and `structures.csv` are the two files to read while a campaign
runs. The first answers "how many are where", the second "what happened to this
one", for all of them at once.

`workdir` is on scratch because it gets large. The campaign folder stays small
and holds only what you wrote plus what you would want to keep.

### Inside `dft/`

A campaign created by `csp init` uses the `runs` layout: **one directory per
structure**, holding its inputs, its script, its log and one subdirectory per
step.

```
dft/
├── .potcars/
│   └── POTCAR-21df8b3b98d8    shared; CeFeB had 109 copies of 8 distinct files
├── claims/
│   └── claim-26940187.json    what each submission took; outlives the driver
└── runs/
    └── 0088-Ce8Fe56B4-strain_Ce2Fe14B_strainiso_p0.00pct/
        ├── structure.json     id, formula, seed, steps — readable without the DB
        ├── SLURM_TASK         the job id that ran this directory
        ├── FAILED_STEP        only if something failed; names the step
        ├── <jobname>.sbatch   the script, exactly as the driver submitted it
        ├── slurm-26940187.out the job's log
        ├── relax/             INCAR KPOINTS POSCAR CONTCAR OUTCAR …
        └── static/            written during the job, from relax/CONTCAR
```

**One structure is one job.** The script and the log sit in the structure's own
directory because `--chdir` points there, and the job is named after it. Nothing
else shares its allocation, so its walltime, its memory and its failure are its
own.

The directory name is `<id>-<formula>-<seed>`. The id is zero-padded so `ls`
sorts numerically, and the seed label is kept because `dft-18-relax` tells you
nothing while `0018-Ce8Co12Fe44B4-agentic_Ce2Fe11Co3B_x3_o5-7_Co` tells you what
the structure is. It also ends a real collision: every campaign numbers its
structures from 1, so `dft-69-relax` existed in two campaigns at once with
different contents.

`structure.json` matters more than it looks. The database is not the only place
an answer should live — 37 finished calculations were once unreachable because
the database forgot them and nothing on disk said otherwise. With this file the
database is a cache that can be rebuilt from the directories.

### Inside `screen/` and `generate/`

One directory per **submission**, the same shape `dft/` uses:

```
screen/
├── batches/
│   └── ids-0001-0200-x4/
│       ├── inputs.json           what the worker was told to do
│       ├── results-task0.json    what it produced — one per chunk
│       ├── progress-task0.json   how far along a RUNNING task is
│       ├── relaxed-task0/        the relaxed cells
│       ├── <jobname>.sbatch
│       └── slurm-<jobid>_<task>.out
└── claims/                       what each submission took
```

The directory name is the **span of structure ids the whole submission covers**,
zero-padded. `-x4` means it is an array of four chunks; without it, the name is
a single chunk and the range is exactly what ran.

That suffix is not decoration. Screening batches many structures into one array,
and the name used to come from the *first chunk* — so a job spanning ids 1–200
in four chunks called itself `screen-1-50`, naming a fifth of its own work. The
same fault on the DFT side once sent ten hours of debugging to a directory that
had already finished.

`inputs.json` and `results-taskN.json` say which side of the job they are on;
they were `<tag>.manifest.json` and `<tag>.task0.json`, which did not.

### The older `stages` layout

Campaigns started before this change use `layout: stages`: one directory per
structure **per step**, flat in `dft/`, with the scripts, manifests, claims and
logs beside them. A 94-structure campaign put 428 entries at that one level.

Both are supported and **a running campaign is never switched**. The layout is
resolved against the directories, not just the config: if `dft/` already holds
`dft-<id>-<step>/` directories, `stages` is used whatever the config says, and
the reason is printed. Pointing a live campaign at a different layout would
orphan its finished work and re-run it.

### Which job ran which directory

The job name **is** the directory. One structure goes out as one job, named
`<campaign>-<run directory>`:

```
26940187   CePdGe-0195-Ce2PdGe6-agentic_x3_o5-7_Cu   RUNNING
           -> dft/runs/0195-Ce2PdGe6-agentic_x3_o5-7_Cu/
```

The campaign is in the name because `squeue` is per user, not per campaign, and
every campaign numbers its structures from 1.

Each run directory also holds `SLURM_TASK`, the job id written from inside the
job itself, so the mapping can be read from either end:

```bash
cat dft/runs/0195-*/SLURM_TASK          # directory -> job id
squeue --me --name=CePdGe-0195-Ce2PdGe6-agentic_x3_o5-7_Cu
```

This was not always true. Until D135 one sbatch covered an array of many
structures, SLURM named the whole array after task 0, and `squeue` showed
`dft-78-static` for a task really running `dft-69-relax`. If you are looking at
a campaign old enough to still have arrays in flight, `backup/scripts/whichdir.py`
resolves them.

## `csp status`

```
campaign     Fe-Sm
database     /scratch/.../Fe-Sm/campaign.db
compositions 109  across 1 chemical systems
generated    3,034 of 5,348 requested (56.7%)   <- 107 composition(s) short
structures   3034
    dft_done         165
    failed           2
    filtered_out     2867
reference    21 MP entries
jobs
    done             165
    timeout          2
    core-hours       1,276
relaxations
    mattersim:converged      3034
    vasp:relax:converged     52
    vasp:relax:not converged 115   <- not usable as a relaxed geometry
```

Three things in that output are there because they are easy to miss:

**Generation yield is printed above the structure counts.** A shortfall here is
otherwise invisible — the funnel narrows anyway, and 40% fewer candidates
entering it looks exactly like a smaller campaign.

**Relaxations are reported separately from job state.** A VASP run that exits
cleanly at the ionic step limit is `done` and **not relaxed**. In one legacy
campaign 61% of the runs were exactly that, and counting them as successes
would have put unrelaxed geometries in the results table.

**`core-hours`** is what you have actually spent, plus a projection for what is
still in flight. Both are **reporting only** — nothing is held back on them
(D143). What throttles the run is `dft.max_cores`, which counts cores held now
rather than extrapolating a cost. Read the projection knowing that a stage with
no finished job yet is priced at its requested walltime, so it is a ceiling.

## `csp status --why <id>` — why is this structure not in my results?

```
structure 12: Fe10Sm
    composition_id       2
    filter_reason        mlip e_above_hull > 0.1
    generator            mattergen
    mlip_e_above_hull    0.20335836121530093
    mlip_e_per_atom      -7.919482838023793
    mlip_relaxed         True
    origin               generated
    spacegroup           10
    state                filtered_out
    wyckoff              ['2n', '2n', '2m', '1h', '2m', '1e', '1a']
  gates
    filter:e_above_hull  FAIL  value=0.20335836121530093 threshold=0.1
```

The gate line is read from a recorded **event**, not reconstructed by
differencing counts. That distinction matters: a structure can leave a state for
more than one reason, and differencing attributes all of them to whichever gate
ran last.

## `csp report`

```bash
csp report                    # report/report.html + report/candidates.csv
csp report --limit 0          # every row, not just the first 2000
csp report --detail 50        # 3D cards + per-site moments for the top 50
csp report --no-hull          # skip the hull plots
csp report -o /somewhere/else
```

`report.html` is self-contained — no server, no build step, no network. Open it
in a browser. Six sections, in this order:

| section | what it answers |
|---|---|
| Funnel | how many structures each gate removed |
| Convex hulls | where the survivors sit, on **both** scales, drawn |
| Magnetisation | how good they are, as M, V, M/V and μ₀M |
| Candidates | the full sortable table, every column defined |
| Structures | one card per top candidate: the cell in 3D, every ion's moment |
| Columns | what each column means |

### The two convex hulls

The page draws **two** hulls and never one: a MatterSim hull whose vertices are
the store's own MatterSim energies, and a DFT hull whose vertices are our own
VASP recompute. Both are one scale on both sides.

They are not merged, and this is not a stylistic choice. Ours minus MP runs
+0.15 to +0.21 eV/atom while the selection threshold is 0.06 — three times
smaller — so a single plot carrying both would show candidates sitting below a
hull they are nowhere near.

The plot follows the dimension of the chemical system:

* **binary** — formation energy against composition, with the lower convex
  envelope drawn as a line. Distance above the hull is visible directly.
* **ternary** — the composition triangle with the hull's tie-lines. Formation
  energy is not a spatial axis here; it is in the colour and the tooltip.
* **four or more elements** — no faithful 2D projection exists, so no hull is
  drawn. A strip plot of hull distance is shown instead, and says so.

A hull that cannot be built **prints the refusal in full** rather than drawing
something. If the reference store is missing a phase, you get the missing
`mp-` ids and the `refstore.py submit` line that fixes it. A hull with one
borrowed vertex still draws, still looks correct, and is wrong by the scale
offset — so nothing is drawn.

Click any candidate point to jump to its card.

### Magnetisation

The same magnetisation in four units, ranked by the fourth:

| | |
|---|---|
| **M** | μ_B per cell — what VASP integrated |
| **V** | Å³ — the relaxed cell |
| **M/V** | μ_B/Å³ — volume-normalised, comparable across cells |
| **μ₀M** | tesla — the saturation polarisation J_s, the number a permanent magnet is judged on |
| **M/V** | emu/cm³ — the same number in the CGS units the experimental literature uses |

`1 μ_B/Å³ = 11.654 T = 9274 emu/cm³`. Nd₂Fe₁₄B is 1.61 T, printed beside the
column for scale.

Every number in this table comes from `m_dft_raw`, the cell magnetisation.
`m_s_reconstructed` is **not** in it: it is a Hund's-rule model, and mixing a
modelled number into a ranked figure of merit is exactly the confusion the two
separate columns exist to prevent.

### The structure cards

One collapsed card per top candidate (`--detail`, default 25). Opening one
builds a rotatable ball-and-stick view of **the cell the VASP relaxation ended
on** — the CONTCAR, read from the run directory, not the MLIP-relaxed cell the
database holds. If that directory is gone the card falls back to the database
geometry and says so in an orange banner.

Each card opens with **one sentence saying what the structure is and what was
changed to make it** —

> Derived from Ce₂PdGe₆. What varies: Ge sublattice, x≤2. Placed by hand during
> the session (01-manual route).

— followed by a **Provenance** block giving the parent, the sublattice varied,
how the seed was made, its staging route, the seed file name, the library path
it was copied from, and its md5.

This comes from `<campaign>/seed_provenance.csv`, written when the seeds were
staged into `inputs/`, joined on the basename of the structure's `source_path`.
Nothing is parsed out of the filename: `Ce2Al2Ge4Pd_x2_o1-5_Al.vasp` does encode
its own substitution, but a filename is a label somebody typed and the CSV is a
record something wrote.

**It says what was intended, not what was measured.** The provenance records
what the staging step was told; the cell above it is what VASP ended on. The md5
is of the seed *as supplied*, so it will never match the relaxed cell — the card
says so. A campaign with no `seed_provenance.csv` (anything generated rather than
staged) gets no provenance block and a note explaining why, rather than a guess.

Above the cards, a table counts **every** seed the campaign staged by route — not
only the ones with a card. A campaign staged from several libraries exists in
order to ask which route produced the better candidates, and that question cannot
be asked from a page that never says a candidate came from one. CePdGe, for
example:

| staging route | how the seeds were made | what varies | seeds |
|---|---|---|--:|
| `01-manual` | manual, in-session | Ge sublattice, x≤2 | 95 |
| `03-agentic` | agentic, one command | Ge sublattice, x≤3 | 78 |
| `04-mazin` | manual, new tool | Ce sublattice, partial | 25 |
| `Qiang-layer` | Qiang, hand design | layer replacement | 19 |
| `Qiang-modular` | Qiang, hand design | modular block substitution | 14 |
| `parent` | given | nothing — this IS the parent | 1 |

Beside the viewer:

* every lattice parameter, the volume, and all five magnetisation numbers
* **by element** — the sphere sum per element with its **spread** across ions.
  A large spread means inequivalent sites carry different moments, which is the
  sublattice structure a single average hides.
* **per site** — one row per ion with its s, p, d and f channels. Rare-earth
  rows are shaded, because on a frozen-4f POTCAR they are supposed to be near
  zero and a row that is not is a result.

The viewer takes a supercell (1×1×1 to 3×3×3), colours by element or by the
ion's own moment (a diverging scale, so an antiferromagnetic arrangement reads
as two colours), and writes a CIF you can download.

**Why not JSmol by default.** JSmol is ~100 MB and loads over HTTP, which the
single-file rule cannot accommodate. The built-in viewer needs no network and
is what keeps the file emailable. Pass `--jsmol-url <url>` to use JSmol where
it is already served — the web portal — and the embedded data is identical
either way.

### The funnel

How many structures each gate saw and how many it kept, in the order they ran.
From a real campaign:

```
gate                           seen   passed  rejected
filter:e_above_hull            3034      167      2867
dft:relax:converged             167       52       115
select:candidate                 11       11         0
```

Which gates appear depends on which stages ran: a full campaign also shows
`screen:validate`, `screen:converged`, `dedup`, `dedup:seed_collision` and
`filter:per_composition`.

### `structures.csv` — every structure, wherever it got to

`candidates.csv` lists what came **out**. This lists what happened to
**everything**, including the rows that stopped — which are usually the ones you
are looking for.

Written to two places, and it is the same table in both:

* `$workdir/structures.csv` — rebuilt by the driver **every cycle**, so it is
  current while the campaign is still running;
* `report/structures.csv` — written by `csp report`.

It is derived entirely from `campaign.db`. Delete it and the next cycle rebuilds
it identically; nothing reads it back. That is deliberate — a second file that
can disagree with the database is worse than no file.

The two columns that do not exist anywhere else:

| column | meaning |
|---|---|
| `stopped_at` | the gate that ended this structure's run — empty if it finished or is still moving |
| `why` | the reason recorded at that gate |

Before this, that question was `csp status --why <id>`, one id at a time: fine
for a structure you already suspect, useless for finding one you do not.

```
$ python -c "
import csv, collections
rows = [r for r in csv.DictReader(l for l in open('structures.csv') if not l.startswith('#'))]
for k, v in collections.Counter(r['stopped_at'] for r in rows).most_common():
    print(f'{v:>5}  {k or \"(finished / still moving)\"}')"

  183  (finished / still moving)
   26  filter:e_above_hull
   13  filtered_out
   10  filter:per_composition
```

**`stopped_at` and `why` always come from the same record.** Reading the gate
from the event log and the reason from the stored field is a plausible-looking
mistake: a structure that failed one gate and was *later* excluded by hand then
reports the gate it did not stop at, and sends you to the wrong directory.

The rest of the columns are the whole funnel, grouped by where the number came
from — identity, provenance (`origin`, `source_name`, `source_path`), the MLIP
(`mlip_e_per_atom`, `mlip_converged`, `e_above_hull_mlip`), our DFT
(`dft_step`, `dft_attempt`, `vasp_energy`, `dft_e_above_hull`), the physics
(`volume`, `m_dft_raw`, `m_s_reconstructed`), and `dft_dir` so you can go
straight to the files. A comment header in the file itself explains each one.

### `candidates.csv`

| column | meaning |
|---|---|
| `id` | campaign structure id — the one `--why` takes |
| `formula` | reduced formula, alphabetical (`Fe17Sm2`) |
| `n_atoms` | atoms in the cell |
| `spacegroup`, `spacegroup_symbol`, `symprec` | of the relaxed cell, at the tolerance named |
| `e_above_hull_mlip` | eV/atom, MatterSim energies against MP |
| `dft_e_above_hull` | eV/atom, your DFT against MP |
| `dft_e_formation` | eV/atom |
| `volume`, `volume_per_atom` | Å³, relaxed cell |
| `m_dft_raw` | μ_B per cell — **computed**: the cell magnetisation |
| `m_s_reconstructed` | μ_B per cell — **modelled**: TM sublattice + Hund's-rule 4f |
| `f_treatment` | which of those two is the meaningful one |
| `m_per_formula_unit` | μ_B, from `m_dft_raw` |
| `m_per_volume` | μ_B/Å³, from `m_dft_raw` |
| `mu0_m_tesla` | T — the saturation polarisation μ₀M, `m_per_volume × 11.654` |
| `m_emu_per_cc` | emu/cm³ — the same number in CGS, `m_per_volume × 9274` |
| `state` | where the structure stopped |

`m_dft_raw` and `m_s_reconstructed` are the same physical quantity computed two
ways and are **adjacent on purpose**. With `f_treatment: frozen` the 4f moment
is in the core and `m_dft_raw` is missing it; `m_s_reconstructed` adds it back
from Hund's rules. A reader who sees only one of them has been misled, so both
are always written, and `f_treatment` says which one to believe.

> **`dft_e_above_hull` is on one scale now, or it is not written at all.**
> `reference.mode: recompute` (the default) builds the hull from *our own*
> recomputed MP phases, read from the shared store at this campaign's
> `recipe_id`. A system missing even one recomputed vertex is **refused and
> named in the stage note**, and no `dft_e_above_hull` is written for it —
> borrowing MP's number for the gap would rebuild the old scale error in
> miniature, and the result would look entirely correct.
>
> So an *absent* `dft_e_above_hull` means incomplete reference coverage, not a
> failed calculation. Run `csp reference status <chemsys>` to see what is
> missing. `mode: mp_energies` opts back into MP's numbers and stamps
> `SCALE_WARNING` on the report.

## The database

`campaign.db` is one SQLite file, and it is **also** a valid ASE database. The
`systems` table is ASE's; the relational tables beside it are cspflow's.

```bash
ase gui campaign.db                     # browse structures, no export step
sqlite3 campaign.db ".tables"
```

```sql
-- the funnel, from the events that produced it
SELECT gate, COUNT(*) FROM filter_event GROUP BY gate;

-- core-hours by stage
SELECT stage, SUM(core_hours) FROM job GROUP BY stage;
```

Two conventions hold everywhere in it, and both come from measured behaviour:

* **Pending work is an explicit `state` string, never a missing key.** ASE's
  `db.select('~key')` does not return the complement, so "not yet computed" as
  an absent key is a dead end — the failure mode is a driver that silently
  never picks up any work.
* **No `NULL` for "not measured yet".** A failed or skipped measurement carries
  a state and a reason.

`provenance` records the resolved config and code state behind every set of
numbers, so a result can always be traced to the configuration that produced it.


---

## See also

* [A-to-Z workflows](04-workflows.md) — including how to read a finished run
* [Every setting in `campaign.yaml`](06-settings.md)
* [Troubleshooting](13-troubleshooting.md)
