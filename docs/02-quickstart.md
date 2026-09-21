# 02 — Quick start

> **Guide 02 of 13** · Previous: [01 — Installation](01-installation.md) · [Guide home](README.md) · Next: [03 — Choosing your input](03-sources.md)

About ten minutes to create and check a campaign. Generation, screening, and
VASP then take as long as your queue and structures require. This guide assumes
cspflow is [installed](01-installation.md) and `csp doctor` is happy.

## Before you start

You need four things:

- a shell where `csp version` works;
- a writable scratch location for `workdir`;
- a machine profile for your cluster (use `orion` on the MMI cluster);
- access to the shared reference store, VASP, and POTCARs for a real DFT run.

If any of those words are unfamiliar, that is fine: run `csp doctor`. It checks
them and tells you exactly which item is missing. It does not submit jobs or
change campaign data.

> **Safest first run on the MMI cluster:** use
> [`campaigns/tests/t3-structure-list`](../campaigns/tests/t3-structure-list/).
> It contains three small seed structures, skips MatterGen generation, and uses
> chemistry already covered by the reference store. The rest of this page shows
> how to create your own campaign.

## 1. Make a campaign

A campaign is a **folder**, not a file:

```bash
csp init 1 my-campaign -m orion
cd my-campaign
```

The number is the **campaign type** — what you are giving the pipeline:

| type | source mode | the question it answers | you give |
|---|---|---|---|
| `1` | `chemical_space` | "search a region of the periodic table" | element groups |
| `2` | `composition_list` | "I know which formulas I want" | formulas, inline or a CSV |
| `3` | `structure_list` | "I already have the structures" | POSCAR/CIF files |

Leave it out (`csp init my-campaign`) and init prints that menu and asks.
`csp init --help` shows it too.

```
my-campaign/
├── campaign.yaml   what to search, and how hard   ← every setting live, alternatives beside it
├── machine.yaml    partitions, walltime, modules, VASP, POTCAR trees
├── recipe.yaml     the DFT ladder: INCAR tags, k-points, per-step resources
├── inputs/         your own structures or composition lists (starts empty)
├── results/        everything the campaign produces
└── report/         report.html + candidates.csv
```

`machine.yaml` and `recipe.yaml` are **your copies**, not references into the
installed package. Edit them freely; nothing here is read-only.

`-m` picks which shipped profile to copy: `orion`, `generic_slurm` (portable
SLURM, no site specifics) or `local` (run on this machine, no scheduler).

## 2. Edit four things

Open `campaign.yaml`. It is the example campaign for your type with your name
in it. **Every line that is not a comment is a live setting**, written out even
where it equals the default. Beside each one, a comment lists what else it can be:

```yaml
magnetic_order: ferro        # ferro | ferri | none                 <- pick ONE
elements: [Sm, Tb]           # several: [Sm, Nd, Pr, ...]           <- a comma-separated list
overrides: {}                # several: {Sm: Sm_3, Ti: Ti_pv}       <- several key: value pairs
```

Changing a setting is replacing its value with one from its comment.

The four choices that matter on day one are the campaign name, work directory,
input, and filter threshold. The example below is deliberately tiny. Grow it
only after the dry run reports the count you expect.

```yaml
name: my-campaign
workdir: /scratch/$USER/cspflow/my-campaign   # where the database and jobs live

source:
  - mode: chemical_space          # ← what you are searching
    chemical_space:
      groups:
        A: {elements: [Sm], pick: 1}
        B: {elements: [Fe], pick: 1, min_fraction: 0.50}
      max_atoms_formula: 4
    defaults:
      z: {min: 1, max: 1}
      max_atoms: 8
      n_structures: {mode: fixed, count: 2}

filter:
  e_above_hull_max: 0.10          # ← eV/atom; this sets how much DFT you buy
```

Not searching a chemical space? [Choosing your input](03-sources.md) covers giving
an explicit list of formulas or a folder of structures you already have.

## 3. Check before you spend anything

```bash
csp doctor              # machine, modules, VASP, POTCARs, live SLURM limits
csp source --dry-run    # what would be enumerated — nothing is written
```

`--dry-run` runs the identical code path the real enumeration does, so the
estimate you approve is produced by the code that then does the work:

```
source 'shortlist' (mode=composition_list, enters at generate)
  compositions      34
  chemical systems  5
  structures wanted 1,394
  warning: 'SmFe11Ti' appears twice; keeping the first

total
  compositions      34
  seed structures   0
  chemical systems  5
  structures wanted 1,394

--dry-run: nothing written
```

**Read the structure count before you go on.** A chemical space has a size
multiplier hiding in it — raising `pick` from 1 to 2, or `max_atoms_formula`
from 20 to 30, can move that number by an order of magnitude.

Do not continue until both checks succeed:

- `csp doctor` ends with no hard failures;
- the dry-run composition and structure counts are small enough for the run you
  intended.

## 4. Phase A — cheap, run to completion

```bash
csp run --through reference
```

This generates structures, relaxes them all with the MLIP, drops duplicates and
builds the reference hull. It is a **barrier**: it finishes for every
composition before anything expensive begins.

> If you have seen `--through calibrate` in an older note, that stage no longer
> exists and the flag is now an error rather than a slower run.

## 5. Phase B — expensive, streamed

This is the point where cspflow starts buying VASP time. Before continuing,
check `dft.max_cores`, `dft.select.max_total`, and the resolved recipe:

```bash
csp recipe
csp config show --origins
```

```bash
csp run --from filter --watch
```

Selects who is worth a VASP job, submits under `dft.max_cores`, and keeps
cycling (`--interval`, default 300 s) — reconciling what finished, submitting
what fits. Stop it with Ctrl-C whenever you like; nothing is lost, and the same
command resumes.

## 6. Look at what came out

```bash
csp status                # progress, by stage
csp status --why 1042     # the full life history of one structure
csp report                # report/report.html + report/candidates.csv
```

`report.html` is self-contained — no server, no build step, no network. Open it
in a browser. [Reading the results →](11-results.md)

You know the run is complete when `csp status` shows no pending or active work
and `csp report` finishes successfully. A campaign may also stop earlier at
`dft.select.max_total`; that limit is a deliberate lifetime ceiling.

## Things worth knowing early

**Commands find the campaign by walking up**, so they work from any folder
inside it, not just the top.

**Stopping the watcher is safe.** Ctrl-C stops the local driver loop; it does
not erase finished work. Already submitted SLURM jobs continue to run. Use the
same `csp run --from filter --watch` command to resume reconciliation and
submission later.

**Any stage slice works**: `--only screen`, `--from dft`, `--through dedup`.
The two phases above are just two slices of one list.

**Change one value without editing a file:**

```bash
csp run --set filter.e_above_hull_max=0.05
csp config show --origins       # every resolved value, and which file set it
```

**Nothing is hidden.** `csp config show --origins` names the file behind every
setting, and `csp recipe` prints the DFT ladder fully resolved — every INCAR
tag literal, nothing deferred to a library default.

## Next

* [A-to-Z workflows](04-workflows.md) — a full walkthrough for each of the three source modes
* [Every setting in `campaign.yaml`](06-settings.md) — one page, with defaults
* [Choosing your input](03-sources.md) — the three ways in
* [`examples/`](../examples/) — three complete campaigns you can copy
* [`campaigns/tests/`](../campaigns/tests/) — three small ones sized to actually run
