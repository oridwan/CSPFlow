# 12 — Command reference

> **Guide 12 of 13** · Previous: [11 — Reading the results](11-results.md) · [Guide home](README.md) · Next: [13 — Troubleshooting](13-troubleshooting.md)

Every command takes `-c/--campaign` (default `campaign.yaml`) and `-s/--set`.

**Commands find the campaign by walking up** from the current directory, so
they work anywhere inside a campaign folder, not just at its top. The campaign
actually used is printed to stderr when it was found that way.

```
csp version    Print the version.
csp init       Create a campaign folder from example type 1, 2 or 3.
csp doctor     Check everything that can be known before a job is submitted.
csp source     Stage 0 — expand `source:` into composition and seed rows.
csp run        The driver loop: reconcile what is in flight, submit what fits, repeat.
csp status     Show campaign progress.
csp report     Write the candidate table and a self-contained HTML dashboard.
csp recipe     Print a recipe fully resolved — every tag literal, nothing deferred.
csp config     Inspect configuration.
csp reference  Manage the shared store of recomputed MP reference phases.
csp ingest     Import an existing campaign directory into a cspflow database.
csp adopt      Adopt a finished legacy campaign into cspflow's own layout.
```

---

## `csp init <type> <name>`

Creates a campaign folder from the [example campaign](../examples/) for that
type: `campaign.yaml`, your own `machine.yaml` and `recipe.yaml`, an `inputs/`
folder and a README.

| type | also accepted | source mode | the question it answers |
|---|---|---|---|
| `1` | `chemical_space`, `space`, `sweep` | `chemical_space` | "search a region of the periodic table" |
| `2` | `composition_list`, `compositions`, `formulas` | `composition_list` | "I know which formulas I want" |
| `3` | `structure_list`, `structures`, `seeds` | `structure_list` | "I already have the structures" |

```bash
csp init 1 my-sweep
csp init 3 my-seeds -m local --no-recompute-reference
```

| option | default | |
|---|---|---|
| `-d, --dir <path>` | `./<name>` | where to create it |
| `-m, --machine <str>` | `orion` | shipped profile to copy: `orion`, `generic_slurm`, `local` |
| `--recipe <str>` | `magnets` | shipped DFT recipe to copy |
| `--here` | | use the current folder instead of creating one |
| `--minimal` | | `campaign.yaml` without comments, referring to the shipped profile and recipe by name |
| `-o, --out <path>` | | write just the campaign file, at this path |
| `--force` | | overwrite existing files |
| `--recompute-reference / --no-recompute-reference` | ask | the one reference question; written as `reference.mode` |

**What is in `campaign.yaml`.** The example file for the type, with four lines
changed: `name`, `machine`, `dft.recipe` and `reference.mode`. Every setting
that matters is live, written out even where it equals the default, and its
alternatives are in a comment on the same line — `# a | b | c` means pick one;
`# several: ...` means a comma-separated list or several `key: value` pairs.

**Inputs.** Types 2 and 3 also get the example's demo inputs —
`inputs/compositions.csv`, or five Sm-Fe seeds in `inputs/seeds/` — so
`csp source --dry-run` works straight away. Replace them with your own.

**No type given.** `csp init my-campaign` (the older form) prints the three
types and asks on a terminal. With no terminal — a script, CI — it stops with
the same menu and writes nothing, rather than guessing: the three types scaffold
different funnels, and a wrong guess is a campaign that runs.

`csp init` reads the examples from the cspflow checkout, so cspflow must be
installed from it (`pip install -e .`).

## `csp doctor`

Checks everything knowable before submission and **exits non-zero on any hard
failure**, so it can gate a submit script.

| option | |
|---|---|
| `--elements <str>` | comma-separated, e.g. `Sm,Fe,Ti` — resolve POTCARs for these regardless of the campaign |
| `--fix` | create the POTCAR symlink layout |
| `-m, --machine <str>` | check a different profile |

Resolves every POTCAR, reads live SLURM limits from `sacctmgr`/`scontrol`,
verifies each configured module exists and that the VASP binary is there.

It also checks **`reference recipe`**: whether this campaign's DFT policy hashes
to the same `recipe_id` as the shared reference store. A mismatch is a warning,
not a failure, because the campaign still runs — it just reads an *empty*
reference set, so every system is refused for incomplete coverage and no
`dft_e_above_hull` is written. The shipped template does not match a
`ferromagnetic` store out of the box. See [the reference set](10-reference-set.md).

## `csp source`

Expands `source:` into rows.

| option | default | |
|---|---|---|
| `--dry-run` | | print the plan, write nothing |
| `-n, --limit <int>` | | commit only the first N composition rows (the full plan is still reported) |
| `--gpu-seconds <float>` | 0.0 | seconds per generated structure, for a time estimate |
| `--db <path>` | | write here instead of the campaign workdir |

`--limit` is what makes a large chemical space quick to sanity-check: the whole
enumeration is still reported, only the commit is capped.

## `csp run`

| option | default | |
|---|---|---|
| `--through <stage>` | | run stages up to and including this one |
| `--from <stage>` | | run stages from this one on |
| `--only <stage>` | | a single stage |
| `--watch` | | keep cycling instead of stopping when idle |
| `--interval <int>` | 300 | seconds between cycles |
| `--max-cycles <int>` | | stop after this many |
| `--dry-run` | | report what would be submitted, claim nothing |

```bash
csp run --through reference       # Phase A
csp run --from filter --watch     # Phase B
csp run --only screen
```

Stages, in order: `source`, `generate`, `screen`, `dedup`, `reference`,
`filter`, `dft`, `analyze`. A name not in that list is an error naming the list
— including `calibrate`, which was a stage and is not one any more.

`generate` is the only stage that may legitimately be absent: a `structure_list`
campaign has no `generate:` block, and the driver treats that as configuration
rather than a missing implementation.

## `csp status`

| option | default | |
|---|---|---|
| `--why <int>` | | full life history of one structure id, with every gate it passed or failed |
| `--source <auto\|db\|snapshot>` | `auto` | where to read progress from |

Works from any node, while the campaign is running.

It has to, and the database alone cannot manage it. A campaign database is
SQLite on a network filesystem, and a running driver keeps recent writes in a
`campaign.db-wal` whose index lives in shared memory — coherent within one host
and nowhere else. Read from a second node it either fails outright or, worse,
returns a clean table of numbers that stopped being true at the last checkpoint.

So `auto` reads the database only when it is both readable and current, and
otherwise reads `status.json`, which the driver rewrites every cycle from the
node that can read the database. The first two lines of output always say which
source was used and how old it is:

```
campaign     /scratch/oridwan/cspflow/CeFeB
source       driver snapshot status.json, written 2m ago (cycle 175, slurm 26930224)
             campaign.db on this node is 40m behind its write-ahead log, so it was not used
```

`--source db` forces the database and fails rather than falling back;
`--source snapshot` never opens the database at all. `--why` needs the
database, so it refuses on a node whose copy is stale instead of returning a
partial history — run it on the node holding the campaign, which `squeue` names.

If `csp doctor` reports the journal mode as WAL on a network filesystem, that
database is one a second node cannot read. It can only be converted while no
driver holds it:

```bash
sqlite3 <campaign>/campaign.db 'PRAGMA journal_mode=TRUNCATE;'
```

## `csp report`

| option | default | |
|---|---|---|
| `-o, --out <path>` | `<campaign>/report` | directory for `report.html` and `candidates.csv` |
| `--limit <int>` | 2000 | rows in the table; `0` for all |
| `--detail <int>` | 25 | structures that get a 3D card and a per-site moment table; `0` for none |
| `--no-hull` | off | skip the hull plots (they rebuild one phase diagram per chemical system) |
| `--jsmol-url <url>` | | serve the 3D view with a JSmol at this URL instead of the built-in viewer |

`--detail` is the size knob. Each card embeds a geometry, so 25 cards is a file
of a few hundred kB and 500 would be tens of MB.

`--jsmol-url` is for the web portal, where JSmol is already served (the
distribution is ~100 MB and cannot be inlined). Without it the page uses the
built-in viewer, which needs no network — which is what keeps `report.html`
emailable.

## `csp recipe [name]`

Prints a recipe fully resolved — every INCAR tag literal, `inherit:` expanded.
With no argument, prints the recipe **this campaign** would actually use.

## `csp config`

```bash
csp config show              # the fully resolved configuration
csp config show --origins    # …and which layer supplied each value
csp config defaults          # the schema's defaults, independent of any campaign
```

## `csp ingest <root>`

Imports an existing campaign directory (VASP outputs on disk) into a cspflow
database.

| option | default | |
|---|---|---|
| `-n, --limit <int>` | | only this many formula directories |
| `--source-name <str>` | `ingested` | the source name recorded |
| `--db <path>` | | write here instead of the campaign workdir |
| `-q, --quiet` | | |

Unlike a composition list, ingest **tolerates junk**: it reads a directory tree
nobody curated, so a bad directory is skipped and reported rather than fatal.

## `csp adopt <flow>`

Adopts a finished legacy campaign into cspflow's layout, reading all of a flow's
result artefacts — not just its VASP directories, which is what `ingest` does —
and writing the campaign cspflow would have written had it run the work itself.
Nothing is recomputed and nothing in the source directory is touched.

| option | default | |
|---|---|---|
| `--dest <path>` | `/scratch/$USER/cspflow_results` | where the campaign directories go |
| `--staging <path>` | `/tmp` | build here first, then move — SQLite on NFS commits ~15x slower |
| `--machine <str>` | `orion` | profile to record in `campaign.yaml` |
| `-n, --limit <int>` | | only this many structures, for a quick check |

This is specific to one project's legacy layout; it is not a general importer.
For arbitrary VASP output trees, use `csp ingest`.

## `csp reference`

> **Use `scripts/refstore.py` instead.** The reference set is one folder with one
> CSV — see [the reference set](10-reference-set.md). These subcommands are the
> older, campaign-shaped path: each extension produced its own campaign workdir
> whose results had to be exported and collected afterwards. They still work and
> the library underneath is shared, but there is no reason to start here.
>
> | you want | use |
> |---|---|
> | do we have this system? | `refstore.py status Ce-Fe-B` |
> | add a system | `refstore.py add Ce-Fe-B --apply` |
> | run what is unfinished | `refstore.py submit dft --apply` |
> | build the hull | `refstore.py hull Ce-Fe-B` |

Manages the shared store of MP phases recomputed at your own DFT settings.

Every subcommand takes `--systems-file <path>` (one chemical system per line),
bare chemsys arguments, or `--from-results <dir>`.

| subcommand | |
|---|---|
| `prefetch` | download MP thermo entries and geometries into the shared cache |
| `mlip` | MatterSim over every cached geometry |
| `report` | how well the MLIP reproduces the reference set, per system and per element |
| `plan` | what recomputing would cost, deduplicated by material id |
| `build <dest>` | write and seed a reference campaign — **submits nothing** |
| `export` | copy a finished campaign's energies into the shared store |
| `status` | which systems are ready for a hull on our own energies |
| `exclude` | mark a phase unusable, with a reason |
| `adopt` | read an existing tree of `mp-*/relax,static` folders into the store |

| option | default | |
|---|---|---|
| `--max-e-above-hull <float>` | `0.5` | skip phases this far above MP's own hull (eV/atom); `0` disables the cut |
| `--max-atoms <int>` | | skip cells larger than this — a cost cut, not a physics one |
| `--min-atoms <int>` | | removes nothing; splits a build into size tiers |
| `--dry-run` | | print the plan, write nothing |


---

## See also

* [A-to-Z workflows](04-workflows.md) — these commands in the order you use them
* [Every setting in `campaign.yaml`](06-settings.md) — what `--set` can reach
* [The stages](07-stages.md) — the list `--through`, `--from` and `--only` slice
