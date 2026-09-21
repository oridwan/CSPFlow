# 10 — The reference set

> **Guide 10 of 13** · Previous: [09 — The DFT recipe](09-recipes.md) · [Guide home](README.md) · Next: [11 — Reading the results](11-results.md)

The convex hull your candidates are measured against is built from Materials
Project phases **recomputed at your own DFT settings**.

Those recomputed energies live in **one folder**, outside every campaign,
because they are expensive and every campaign in the same chemistry needs the
same ones. The reference set is global infrastructure. It is *not* a campaign:
it has no generation, no screening, no calibration — just an MLIP relaxation
and two VASP runs per structure.

---

## The store

Everything lives under `$CSPFLOW_STORE` (default `/projects/mmi/cspflow-shared/store`):

```
settings.yaml          the ONE DFT policy.  Never overridden per run.
machine.yaml           the ONE machine profile (partitions, POTCAR paths).
index.csv              THE bookkeeping file: progress + results + hull input.
index.json             the same content, for scripts.
excluded.json          phases deliberately not computed, and why.
mp-cache/              Materials Project downloads, shared across campaigns.
jobs/                  task lists and slurm logs.
structures/
    mp-13-Fe/                    <- MP id AND composition
        info.json                what this phase is and where it came from
        mlip/                    MatterSim relax  (GPU job, runs separately)
        dft_relax/     \         ONE CPU job: relax, then CONTCAR becomes the
        dft_static/    /         static POSCAR, then static
archive/               superseded layouts.  Nothing reads them.
```

The composition is in the folder name on purpose. `ls structures | grep Ce`
answers "what cerium phases do I have" without opening a database — 447, as it
happens.

---

## Everyday commands

```bash
export CSPFLOW_STORE=/projects/mmi/cspflow-shared/store

python scripts/refstore.py status              # the whole store
python scripts/refstore.py status Ce-Fe-B      # one system, sub-systems included
python scripts/refstore.py hull Ce-Fe-B        # build the hull, or say what is missing

python scripts/refstore.py add Ce-Fe-B --apply # fetch MP, create what is missing
python scripts/refstore.py submit dft --apply  # run everything unfinished
python scripts/refstore.py submit dft --chemsys Ce-Fe-B --apply   # just this system
python scripts/refstore.py submit mlip --apply # GPU relaxations

python scripts/refstore.py index               # rescan the tree -> index.csv (~15 s)
```

`index.csv` is **derived**: it is rebuilt by walking the folders and reading the
OUTCARs, so it cannot drift out of step with what is on disk. If you are ever
unsure whether it is current, run `index` again. There is no separate database
to keep in sync, and no export step.

Run `index` after a batch of jobs finishes — nothing updates it automatically.

---

## What `add` does, and the 0.5 eV/atom cut

`add Ce-Fe-B` fetches the Materials Project entries for that system, **including
every sub-system** — Ce, Fe, B, Ce-Fe, Ce-B, Fe-B all come back in one call, so
there is no need to enumerate them. It then skips:

* anything already in `structures/`
* anything more than **0.5 eV/atom** above MP's own hull

The second cut is the useful one. A phase that far above the hull cannot change
where the hull sits, so DFT on it is wasted. Change it with `--cut`; `--cut 0`
disables it.

Nothing is created until you pass `--apply`.

---

## The one rule that cannot be relaxed

Every energy on one hull must come from the **same DFT settings**. Measured here,
ours minus MP:

| phase | difference |
|---|---|
| Fe (mp-13) | +0.2068 eV/atom |
| Sm₂Fe₁₇ | +0.1870 |
| SmFe₁₁Ti | +0.1794 |
| SmFe₂ | +0.1530 |

A per-element model fits those to ~1.8 meV/atom, so the offsets are systematic,
not noise. But the selection threshold is **0.06 eV/atom** — three times smaller
than the offsets. A hull mixing two scales still builds and still looks
reasonable, and every compound it ranks is ranked wrongly.

This is why `settings.yaml` sits at the store root and nothing overrides it. You
never type a recipe id, choose a settings file, or pass `-c`. Verified: an INCAR
regenerated today from `settings.yaml` matches, tag for tag, the one VASP
actually ran on `mp-1001113-Si` months ago — the only differences are `NCORE`
and `KPAR`, which do not affect the energy.

If you must change the policy, understand that **every existing energy becomes
incomparable**. That is a rebuild of the whole store, not an edit.

---

## Why relax and static are one job

They used to be two array jobs with a scheduler round trip and a database
hand-off between them, so a structure could sit for hours after its relax
finished before anything noticed. Now one allocation does both:

1. run VASP in `dft_relax/`
2. `dft_relax/CONTCAR` becomes `dft_static/POSCAR`
3. re-derive the static k-grid **from the relaxed cell**, not from the original
4. run VASP in `dft_static/`

Step 3 is why the static inputs are written inside the job rather than at submit
time: the relaxed cell does not exist yet when you submit.

A stage that already carries a `VASP_DONE` marker is skipped, so a resubmitted
job continues rather than restarting. If a relax did not converge, `submit`
removes that marker *and* the static's — a static energy sitting on an
unrelaxed cell is not a result, and one such row was already in the store
(`mp-1181327-Fe3B`) before this check existed.

A relax that was killed at the walltime restarts from its own `CONTCAR`. Without
that, a structure that needs more than one walltime restarts from the MP cell
forever and never finishes.

---

## The MLIP half

`submit dft` runs on `Orion,Nebula,Apus` with the non-AVX-512 nodes excluded
(see [machines](08-machines.md)). Naming three partitions does not risk taking over
the smallest of them: the array's `%<throttle>` is a concurrency cap SLURM
enforces wherever the tasks land, so the ceiling is `throttle x ntasks` cores in
total, not per partition.

`submit mlip` runs MatterSim on the GPU partition and writes
`structures/<id>/mlip/{CONTCAR,relaxed.cif,mlip.json}`.

It is **optional**. Its product is a better starting geometry for the DFT relax,
which saves ionic steps; MP's own relaxed cell is a perfectly valid seed on its
own. DFT is deliberately not gated on it — gating would park thousands of
structures behind a GPU queue for a convenience.

MLIP tasks are chunked (50 structures per array task by default, `--chunk`)
because loading the model costs about 30 seconds, which would otherwise dominate.

---

## When a hull cannot be built

`hull Ce-Fe-B` refuses if any phase in the system is unfinished, and names them:

```
Ce-Fe-B: 54 usable, 2 not finished
  missing: mp-1181327-Fe3B (relax: OUTCAR has no epilogue (killed mid-run))
  missing: mp-1245108-Fe (relax: no OUTCAR)
```

It refuses rather than filling the gap with MP's number, because that would mix
the two scales above. Finish them:

```bash
python scripts/refstore.py submit dft --chemsys Ce-Fe-B --apply
```

---

## Adding a system for a new campaign

```bash
python scripts/refstore.py status Ce-Fe-B          # do we already have it?
python scripts/refstore.py add Ce-Fe-B --apply     # no -> fetch and create
python scripts/refstore.py submit dft --chemsys Ce-Fe-B --apply
# ... wait ...
python scripts/refstore.py index
python scripts/refstore.py hull Ce-Fe-B
```

New phases land in the same `structures/` tree as everything else. There is no
second location and no collect step.

---

## Two defects found on 2026-09-10, and what they cost

Both were in `scripts/`, not in the library, and both are fixed. They are
written down because each was invisible in the ordinary status output.

### 1. `KPAR` was never set, so no job had k-point parallelism

`refstore.py` called `resolve_inputs(atoms, st, dft, machine)` without
`ntasks=`. In `dft/vasp/inputs.py` the whole parallel planner sits behind
`if ntasks:`, so it never ran: every INCAR in the store shipped with `KPAR`
absent (i.e. 1) and `NCORE` at whatever the recipe literal said.

With `KPAR = 1` VASP walks the irreducible k-points one after another, so wall
time grows roughly linearly in their number even though k-points are the one
part of the calculation that is embarrassingly parallel. Measured on the ten
slowest running jobs:

| structure | atoms | irr. k-points | was | planner | speedup lost |
|---|---|---|---|---|---|
| `mp-1245108-Fe` | 100 | 8 | KPAR 1, NCORE 8 | KPAR 8, NCORE 2 | **8.0x** |
| `mp-1245078-Fe2O3` | 80 | 8 | KPAR 1, NCORE 8 | KPAR 8, NCORE 2 | **8.0x** |
| `mp-715572-Fe2O3` | 40 | 14 | KPAR 1, NCORE 8 | KPAR 8, NCORE 2 | 7.0x |
| `mp-685153-Fe2O3` | 160 | 4 | KPAR 1, NCORE 8 | KPAR 4, NCORE 2 | 4.0x |
| `mp-530050-Fe43O64` | 107 | 3 | KPAR 1, NCORE 8 | KPAR 2, NCORE 4 | 1.5x |

`refstore.py inputs` now takes `--ntasks` and defaults to `$SLURM_NTASKS`, so
inside a job the plan matches the ranks the job actually has. It prints the
choice, which is the cheap way to notice this class of bug next time:

```
$ python scripts/refstore.py inputs <folder> --stage relax
.../dft_relax  KPAR=8 NCORE=2 ntasks=32
```

`KPAR=unset` is not always a bug. A large cell has a small Brillouin zone: the
27 A cell of `mp-1215144-AsO3` gets a 1x1x1 grid, one k-point, and nothing to
split. Check the k-point count before assuming.

### 2. The static ran on relaxes that never converged

`run_vasp` in `refstore_dft.sbatch` decided a stage had succeeded by looking
for `General timing and accounting` in the OUTCAR. VASP writes that on **any
normal exit** -- and exhausting `NSW` without meeting `EDIFFG` is a normal
exit. So a relax that stopped at step 99 with `fmax = 0.182 eV/A` looked
identical to a converged one, `VASP_DONE` was written, and the unrelaxed
`CONTCAR` was carried into `dft_static/POSCAR`.

`dft/vasp/parse.py` documents this exact trap in its module docstring and was
written to avoid it -- the shell script simply reimplemented the marker test
instead of using it. **116 folders** in this store hold a completed static
computed on an unrelaxed cell.

**No wrong energy reached a hull.** `index.csv` derives `relax_converged` from
`"reached required accuracy"` independently, and `ready` requires it, so all
116 are `ready = False`. Verified store-wide: zero hull-ready rows have
`relax_converged != True`. The cost was CPU, not correctness.

The job script now has `relax_converged()`, and:

* `NSW` exhaustion continues the relax from its own CONTCAR, up to
  `REFSTORE_MAX_CONTINUE` (3) times or `REFSTORE_RELAX_BUDGET` (24 h),
  because hitting the step limit is progress, not failure;
* the carry is guarded by `relax_converged` a second time, so a resubmission
  that skips the relax block cannot walk into the static either;
* failing that, the job removes `VASP_DONE` and exits 5, leaving the structure
  for the next submit rather than buying an energy on the wrong geometry.

### A third rung on the retry ladder: ALGO

`mp-1215144-AsO3` ran from -71 eV to **+4,249 eV** over 24 ionic steps with
`fmax = 31.5 eV/A`, after 100+ `WARNING in EDDRMM: call to ZHEGV failed`. A
positive total energy for a periodic solid is not slow convergence; the
RMM-DIIS subspace diagonalisation selected by `ALGO = Fast` had broken down,
and every later step moved atoms on forces computed from corrupt wavefunctions.
It burned 14 h before being cancelled.

`retry_stage` now detects >20 EDDRMM warnings, or a positive total energy, and
switches to `ALGO = Normal` (blocked Davidson) with `IBRION = 2`. It also
**resets the geometry** from `POSCAR.seed` rather than continuing from CONTCAR
-- for this one failure class the CONTCAR is the corrupted cell, so carrying it
forward is exactly wrong. Rerun with those two tags: `DAV:` replaces `RMM:`,
zero EDDRMM warnings, energy back in range.

## "Reached max steps" is not one problem

24 structures hit `NSW = 99` without meeting `EDIFFG`, even after three
continuations each (~396 ionic steps). Measured 2026-09-10, and the first
thing to say is what is **not** wrong:

**Every one of them reached energy accuracy.** Not a single ionic step out of
99 failed its SCF -- `EDIFF = 1e-4` was satisfied every time, and most steps
needed only 12-29 of the 200 electronic cycles allowed. There is no electronic
convergence problem here at all.

A VASP relaxation has two separate accuracies and they fail independently:

| tag | what it controls | value here | status on these 24 |
|---|---|---|---|
| `EDIFF` | electronic / energy, per SCF cycle | `1e-4` eV | **reached, every step** |
| `EDIFFG` | ionic / force, what "converged" means | `-0.01` eV/A | **not reached** |

The 24 then split into two groups that need different things.

### Group B: 7 structures oscillating (fmax 0.106 - 0.510 eV/A)

These are not converging slowly, they are not converging at all. The signature
is a force trajectory that reaches its best value early and then gets *worse*:

| structure | best fmax | at step | final fmax | final energy vs best |
|---|---|---|---|---|
| `mp-1206082-CeCuGe2` | 0.152 | 23 | 0.443 | **+0.095 eV** |
| `mp-1078637-Bi` | 0.093 | 5 | 0.164 | **+0.139 eV** |
| `mp-20169-Ce2O3` | 0.070 | 72 | 0.446 | -0.618 eV |
| `mp-1182534-Fe4As3O23` | 0.213 | 6 | 0.510 | -0.066 eV |

`mp-1206082-CeCuGe2` changed direction on **28 of 28** of its last 30 steps and
finished 0.095 eV *above* an energy it had already found. That is `IBRION = 1`
-- RMM-DIIS quasi-Newton -- trusting a bad Hessian and stepping uphill.

Two also moved their cell far enough to need the basis regenerated:
`mp-20169-Ce2O3` **-24.9%** by volume, `mp-1147691-Fe2O5` **+101.7%**.

The continuation loop already restarts from CONTCAR, which regenerates the
basis. What it did **not** do was change anything else, so it reproduced the
same oscillation three times. It now switches `IBRION 1 -> 2` (conjugate
gradient) on the first continuation. IBRION chooses the path to the minimum,
never the energy at it, so the one-policy rule is untouched.

### Group A: 17 structures just above the line (fmax 0.010 - 0.048 eV/A)

Their energy is flat -- median |dE| over the last ten steps is **7.5e-4 eV**,
against 1.5e-1 eV for group B. They are sitting in a minimum with a small
residual force. For scale, a random sample of 200 **converged** structures
finishes at a median fmax of 0.0056 with none above 0.01, so these are not
far out.

Within them the residual is distributed two ways, and the ratio of the largest
per-atom force to the median one separates them:

* **concentrated** (ratio 2.8-3.9x): a few atoms hold it. `mp-1180064-O2` has
  nine of ten atoms at 0.005-0.011 and one atom at 0.031.
* **uniform** (ratio 1.0-1.5x): every atom sits at a similar value just over
  the cut. Mostly O2 molecular crystals and As oxides, where PBE without
  dispersion gives a very flat intermolecular potential.

Pulay stress is not the explanation: `ENCUT = 520` against a maximum `ENMAX`
of 400 is already the 1.3x ratio recommended for `ISIF = 3`.

## Three hulls, one folder

The store is the source of truth and it keeps being extended, so a hull is
built by READING it. There is no export step, no second database and no
`recipe_id` on the hull path. That is the same rule that makes `index.csv`
trustworthy: derive it by walking the tree and it cannot disagree with disk.

Every structure carries three energies, and each defines a hull on its own
scale:

| source | where it comes from | what it is |
|---|---|---|
| `dft` | `index.csv` `e_static_eV` | our VASP recompute -- the default |
| `mlip` | `index.csv` `e_mlip_relaxed` | MatterSim, as we ran it |
| `mp` | `mp-cache/*__GGA_GGApU.json` | Materials Project's own numbers |

```bash
python scripts/refstore.py hull Ce-Ge-Pd                 # dft
python scripts/refstore.py hull Ce-Ge-Pd --energy mlip
python scripts/refstore.py hull Ce-Ge-Pd --compare       # all three
```

```
Ce-Ge-Pd [dft]   63 entries, 23 stable
Ce-Ge-Pd [mlip]  64 entries, 19 stable
Ce-Ge-Pd [mp]    64 entries, 20 stable
```

**They are never mixed.** `dft` and `mlip` are ours -- same settings, same
store. `mp` is a different scale: ours minus MP is +0.15 to +0.21 eV/atom
against a 0.06 eV/atom selection threshold, so MP's number is for comparing
against, never for filling a gap in one of the other two (D101).

The library is `cspflow/reference/refstore.py`; `csp analyze` uses the same
function, selected by `reference.energy_source` in campaign.yaml.

### What "missing" means, and why index.csv alone cannot tell you

`coverage()` takes the list of phases a system SHOULD contain from the MP
cache, not from `index.csv`. index.csv records what the store HAS; it can
never record what the store is missing. Written the other way round -- which
is how it was written first -- a phase that was never added at all is
invisible, and the hull builds silently without it.

That is not hypothetical. `mp-22623` (Ce3Cu4Ge4) sits at `e_above_hull =
0.0000`, exactly on MP's hull, and had never been added to the store. The
Ce-Cu-Ge hull built, looked correct, and was missing a vertex.

An absent phase only counts as missing when MP puts it close enough to be a
plausible vertex (0.10 eV/atom by default). Without that cut every system
would refuse: the store deliberately skips phases far above the hull, so the
MP cache is always the larger list -- 4,101 entries against 3,463 rows.

### A cut that never fired

`refstore.py add` read `e_above_hull` off the MP entry, but the attribute is
`e_above_hull_mp`. The value was therefore always `None`, so `--cut` did
nothing at all and every `info.json` recorded a null distance. Fixed
2026-09-11; verify with

```bash
python scripts/refstore.py add Ce-Cu-Ge Co-Fe --cut 0.001   # 7 skipped, 7 to create
python scripts/refstore.py add Ce-Cu-Ge Co-Fe --cut 0.1     # 1 skipped, 13 to create
```

If the two lines report the same counts, the cut is broken again.

## ignored.json -- a third state, beside done and failed

`<store>/ignored.json` lists structures the store will **not wait for**. They
are IGNORED, not failed: nothing is wrong with them and nothing was computed
incorrectly. They sit too far above the convex hull to be able to change it,
so finishing them buys nothing.

The rule is `MP e_above_hull > 0.10 eV/atom` and not yet hull-ready. The file
records, per structure, its formula, chemical system, cell size, that distance,
and the state it was in when the decision was taken.

Three tools honour it, and all three use the word *ignored*:

```
$ refstore.py status
hull-ready       3408
ignored          43   (too far above the hull to matter -- see ignored.json; NOT failures)
outstanding      12

$ refstore.py submit dft --apply
skipping 43 structure(s) in ignored.json -- too far above the hull to matter (not failures)

$ refcheck.py
stage        ok  ignored  running  max_steps  ...
relax      3408       43        3          8
```

**Why this is worth a file rather than a rule.** It is reviewable and
reversible: delete an entry and the structure returns to the queue on the next
submit. And it is honest -- the alternative, letting them sit in a `max_steps`
or `crash` column forever, makes the store look further from done than it is
and invites work that cannot pay off. Measured on this store: `mp-1215144-AsO3`
consumed roughly 42 h of compute across a divergence, an `ALGO` change and an
SCF diagnosis before anyone checked that it sits **3.17 eV/atom** above the
hull and could never have mattered.

`excluded.json` is the *different* case: structures that cannot be computed as
given -- a vacuum cluster, a pair of atoms 0.80 A apart. Those are failures.

**Check hull distance before debugging a structure.** It is one lookup, and it
decides whether the debugging is worth doing at all.

## Is the hull ready? -- "unfinished" is not the same as "blocking"

`refstore.py hull <chemsys>` used to refuse whenever any phase in the system was
unfinished. That is too strict: a phase far above the hull cannot be ON the
hull, so waiting for it is waiting for nothing.

The command now reads MP's own `e_above_hull` from the store's fetch cache and
splits the unfinished phases in two:

```
$ python scripts/refstore.py hull Ce-Ge-Pd
Ce-Ge-Pd: 63 usable, 1 not finished

  1 unfinished but too far above MP's hull to matter (> 0.1 eV/atom) -- ignored:
    mp-683992-Ce7GePd22          MP e_above_hull 0.201

hull Ce-Ge-Pd on the raw scale, 63 entries, 23 stable
```

versus a system where the gap really does matter:

```
$ python scripts/refstore.py hull Ce-Zn
Ce-Zn: 22 usable, 1 not finished

  1 unfinished and CLOSE ENOUGH to matter (<= 0.1 eV/atom):
    mp-1188098-CeZn3             ON THE HULL        relax: ionic_step_limit
```

`--ignore-above` sets the cut (default 0.10 eV/atom); raise it to be stricter.

**Why MP's number is safe to triage with, and only to triage with.** It is
computed at MP's settings, not ours, and our energies sit 0.15-0.21 eV/atom
above MP's. But `e_above_hull` is a distance *within* a chemical system, so
that systematic offset largely cancels. It decides only whether to *wait* for
a structure. No MP energy ever reaches a hull -- those come from `dft_static`.

Ignorable phases are still printed. A phase silently dropped is how a hull
quietly becomes wrong.

## Checking what failed

`refstore.py status` tells you *how many* are unfinished. `refcheck.py` tells you
*why*, per stage, using VASP's own words:

```bash
python scripts/store-maintenance/refcheck.py                    # the counts table
python scripts/store-maintenance/refcheck.py zbrent             # list those folders
python scripts/store-maintenance/refcheck.py killed --stage relax
```

```
stage      ok  running  max_steps  zbrent  crash  avx512  killed  no_outcar  setup_error  not_started
relax    2846       27          7       9      .      29      15         64           45          421
static   2792        2          .       .      2       .       .         68           47          552

what 'setup_error' actually said:
    35  magnetism.mode='ferromagnetic' has no initial moment for ['As']
    30  ... ['Sb']      14  ... ['Au']      12  ... ['Bi']      1  ... ['Ru']
```

`setup_error` is the bucket that earns the script its keep. A structure whose
inputs cannot be written leaves **nothing** in its own folder, so scanning the
tree alone calls it `not_started` and the real cause stays buried in a slurm
log. That is why this one bucket is read from `jobs/*.out` rather than from the
structure — and it is how 92 blocked structures turned out to be one missing
line in `settings.yaml` rather than 92 separate problems.

The buckets exist because they need different fixes, and lumping them together
is how a machine problem gets treated as a physics problem:

| bucket | what it means | what to do |
|---|---|---|
| `running` | OUTCAR written in the last 20 minutes | nothing — it is in flight |
| `max_steps` | finished cleanly at `NSW`, never reached `EDIFFG` | resubmit; it restarts from its own CONTCAR |
| `zbrent` | `ZBRENT: fatal error in bracketing` — the line minimiser lost its bracket, usually within meV of the minimum | resubmit; the job retries with `IBRION=2` automatically |
| `crash` | some other fatal VASP block, e.g. `POSMAP: symmetry` or `RHOSYG: stars are not` — nearly always the static step after a relax moved atoms past `SYMPREC` | needs `ISYM=0` or a looser `SYMPREC` for that structure |
| `avx512` | `illegal instruction` — landed on a node with no AVX-512 | a **machine** problem: fix the partition, then resubmit |
| `killed` | no epilogue, no error, not running | walltime or OOM; resubmit continues from CONTCAR |
| `no_outcar` | inputs written, VASP never produced an OUTCAR | check the slurm log in `jobs/` |
| `setup_error` | the job died **before** VASP — writing the inputs raised | read the message it prints; it is usually one missing entry in `settings.yaml` |

`running` is a heuristic on the OUTCAR's age, so a job that stalled without
being killed can sit in it briefly. Everything else is read from the file.

---

## Migrating an old layout

`migrate` folds both historical layouts — a `final/<mp-id>/{relax,static}` tree
and any campaign workdir's `dft/dft-<row>-<stage>/` directories — into
`structures/<mp-id>-<formula>/`. It is a dry run unless you pass `--apply`, it
is idempotent, and it uses `mv` within one filesystem, so no data is copied.

```bash
python scripts/refstore.py migrate           # what would move
python scripts/refstore.py migrate --apply
```

Run `index` afterwards.

---

## Tools

| command | what it is for |
|---|---|
| `refstore.py status [SYS]` | progress, and what is blocking |
| `refstore.py add SYS` | fetch MP, create missing folders |
| `refstore.py submit dft\|mlip` | write a task list and sbatch it |
| `refstore.py index` | rebuild `index.csv` from the folders |
| `refstore.py hull SYS` | build the hull, or say what is missing |
| `refstore.py inputs DIR --stage S` | write VASP inputs (the job calls this) |
| `refstore.py migrate` | fold an old layout into `structures/` |
| `refcheck.py` | why runs are not finished, by stage and by VASP's own error |

---

## Which hull filters, and which hull ranks

Two different hulls, two different jobs, and they are never the same one.

| | vertices | candidates | used for |
|---|---|---|---|
| **filter hull** | store's MatterSim energies | MatterSim-relaxed | the `filter` cut |
| **final hull** | store's DFT (`energy_source: dft`) | our VASP statics | `analyze`, the report |

Both are built on **one scale on both sides**. Before 2026-09-12 the filter hull
was not: its candidates were MatterSim and its vertices were MP's raw DFT, which
misplaced about a third of Ce-chemistry candidates by more than the 0.06 eV/atom
selection threshold. Fixing that is what allowed the `calibrate` stage to be
removed — it existed to measure exactly the gap that mixing introduced. See D126.

### The one question, asked at `csp init`

> Recompute the MP reference phases with THIS campaign's own DFT settings?

* **yes** (`reference.mode: recompute`, the default) — the final hull is built
  from `$CSPFLOW_STORE`. Costs DFT for any system the store does not hold.
* **no** (`reference.mode: mp_energies`) — nothing is recomputed. Candidates are
  filtered on the MatterSim hull and their DFT energies reported raw.

`--recompute-reference` / `--no-recompute-reference` answers it without a prompt.

### What you cannot ask for

Our DFT placed on MP's hull. Over the 3,345 store phases carrying both numbers,
elemental Ce is **+1.17 eV/atom** ours-minus-MP — a different 4f POTCAR, not an
offset — and a per-element correction fitted *inside one chemistry* still leaves
42 meV/atom RMS for Ce-Ge-Pd and 103 for Ce-Fe-B, against a 60 meV/atom
threshold. `analyze` therefore leaves `dft_e_above_hull` **empty** rather than
filling it from MP, and records why under `dft_e_above_hull_absent`.

One rule: **the DFT hull exists only when its vertices are ours.** The same rule
covers a system the store only partly covers.

### Checking coverage before you run

```bash
export CSPFLOW_STORE=/projects/mmi/cspflow-shared/store
python scripts/refstore.py hull Ce-Ge-Pd --energy mlip   # the filter hull
python scripts/refstore.py hull Ce-Ge-Pd --energy dft    # the final hull
python scripts/refstore.py hull Ce-Ge-Pd --compare       # all three
```

As of 2026-09-12, across the 42 chemical systems the CePdGe campaign touches:
**mlip 42/42 complete, dft 13/42, mp 42/42**. For CeFeB's 6 systems: mlip 6/6,
dft 5/6, mp 6/6.
