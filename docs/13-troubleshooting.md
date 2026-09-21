# 13 — Troubleshooting

> **Guide 13 of 13** · Previous: [12 — Command reference](12-cli.md) · [Guide home](README.md)

Errors you are likely to meet, with what they actually mean. Most of cspflow's
failures are deliberate — it prefers stopping with a named cause to producing a
number nobody can defend.

## Configuration

### `undefined variable(s) ['CSPFLOW_REFERENCE'] in '$CSPFLOW_REFERENCE/mp' (at reference.cache)`

A `$VAR` in your config is not exported. Either export it, or write the value
literally.

An undefined variable is a hard error rather than an empty string on purpose:
expanding `$SCRATCH` to `""` turns `$SCRATCH/work` into `/work`.

**The common version of this is self-inflicted.** The default MP cache is
`$CSPFLOW_REFERENCE/mp`, but *schema defaults are never expanded* — only YAML a
layer actually supplied is. So copying that default into your campaign file
turns an optional environment variable into a required one. Leave it out unless
you have exported `CSPFLOW_REFERENCE`.

### `Extra inputs are not permitted` / `filtr` — a typo

```
1 validation error for Campaign
filtr
  Extra inputs are not permitted
```

Unknown keys are rejected rather than ignored, so a misspelled key stops the
campaign instead of silently doing nothing. The name in the error is the key
that was not recognised.

### `Field required … source.0.chemical_space.groups.A.pick`

`pick` has **no default** — it is the sweep's size multiplier, and guessing it
would change how much compute you buy by an order of magnitude. Say how many
elements to take from the group.

### `unknown machine 'x'; shipped profiles: ['generic_slurm', 'local', 'orion']. Pass a path to use your own.`

`machine:` takes a shipped name or a path. In a campaign made by `csp init`,
that path is `machine.yaml` beside the campaign file.

### `unknown stage 'screeen'; the funnel is [...]`

`--only`, `--from` and `--through` take a stage name from the funnel, which the
error prints in full.

### "Which file set this value?"

```bash
csp config show --origins
```

Every resolved value with the layer that supplied it. There is no need to
reason about merge order.

## Input

### `column 4 (n_structures) is '80          # the 2:17 magnet', which is not an integer`

In a composition CSV, `#` is honoured **only at the start of a line**. A
trailing comment after data is read as part of the last cell. Put the comment
on its own line.

### `item 'Fe2Co10' is not a reduced formula … and also sets z=[1, 2]. That is ambiguous`

`Fe2Co10` reduces to `Co5Fe1` with an intrinsic multiplier of 2, so `z: [1, 2]`
could mean Z ∈ {1,2} or Z ∈ {2,4} — a factor of two in cell size. Write the
reduced formula with the range you mean.

Writing a non-reduced formula *without* a `z` is fine: it is taken as that Z,
and you are told.

### `'SmFe11Ti' appears twice; keeping the first`

You listed a formula both inline and in the CSV, or twice in one of them.
Inline items are read first and win. This is a warning, not an error — it is a
normal way to make a one-off exception to a list you otherwise maintain in a
file.

### A seed structure is rejected at ingest

Structure files are parsed by **both** pymatgen and ASE, and they must agree.
The hard failures are:

* **a VASP-4 POSCAR with no species line** — pymatgen invents hydrogen and
  helium for these, with nothing but a stderr warning; H and He would then go
  into the MLIP and into VASP
* **a disordered structure** (partial occupancies) — ASE silently discards the
  minority species and hands back a clean-looking ordered cell that is not the
  material in the file
* **the two parsers disagreeing on the composition**

These are errors at ingest rather than warnings because each of them otherwise
surfaces after GPU or DFT time is spent — or never, with a wrong number in the
results table looking exactly like a right one. Fix the file, or drop it.

### `composition_list.from_file not found: …`

Relative paths resolve against the **campaign folder** — the directory holding
`campaign.yaml` — not against wherever you are standing. `inputs/compositions.csv`
means the one in this campaign.

## Cluster

### A job dies with an error naming something unrelated

Almost always a missing module. `module load` of a module that does not exist
writes to stderr and **still exits 0**, so under `set -e` the job dies at the
*next* command.

```bash
csp doctor        # checks every configured module against `module -t avail`
```

Cluster module names change. `cuda/11.8` disappearing is what motivated the
check.

### `forrtl: severe (168): Program Exception - illegal instruction`, and no OUTCAR

Reported by the index as `relax: no OUTCAR` or `static: no OUTCAR`. The run
directory holds INCAR, KPOINTS, POSCAR, POTCAR and `vasp.out` and nothing else.

VASP was placed on a node whose CPU does not support an instruction set the
binary was compiled for — usually AVX-512. Confirm it:

```bash
grep -l "illegal instruction" */vasp.out | head        # which runs
sacct -j <id> -o JobID,NodeList%20,State                # which nodes
sinfo -n <node> -o "%20N %f"                            # what that node is
objdump -d $VASP_BIN | grep -cE '%zmm[0-9]+'            # does the binary use AVX-512
```

If the failures cluster on a handful of nodes, that is the answer. Fix it with a
`constraint` or `exclude` on the partition — see
[Mixed hardware in one partition](08-machines.md#mixed-hardware-in-one-partition) —
then resubmit. Nothing is wrong with the inputs, so the affected structures just
need to run again:

```bash
python scripts/refstore.py submit dft --apply    # reference store
python scripts/requeue.py -c <campaign> --reason "no OUTCAR" --apply    # ordinary campaign
```

`requeue.py` is deliberately separate from the DFT retry ladder. The ladder
refuses to retry a failure that a recipe change cannot fix, and it is right to:
this failure has an *environmental* cause, and retrying it before the machine
profile is fixed would just burn the same allocation again.

### `srun --mpi=pmi2` starts and then `PMPI_Init` aborts

Intel MPI needs `I_MPI_PMI_LIBRARY` pointed at the PMI library SLURM actually
provides. Without it, the pmi2 plugin rejects every client request with an
error naming neither the variable nor the library. `csp doctor` checks that
every `env` value that looks like a path exists.

### POTCARs resolve but the functional label looks wrong

Run `csp doctor --fix`. pymatgen expects directory *names* the distribution
tarballs do not use; without the symlink layout, `functional: PBE_64` can
silently match a flat-layout directory and the recorded label then misdescribes
what was used. `--fix` only adds symlinks — nothing is copied or modified.

### `unknown stage 'calibrate'` — a driver that dies in the first second

The `calibrate` stage was removed from the funnel. `--through calibrate` is now
an error rather than a slower run, so an old script, note or sbatch default
carrying that flag fails immediately.

Use `--through reference` for Phase A. The valid stages are `source`,
`generate`, `screen`, `dedup`, `reference`, `filter`, `dft`, `analyze`.

The `calibrate:` block in an existing `campaign.yaml` still parses and is
ignored; you can delete it.

### Jobs are not being submitted

`csp run --dry-run` reports what *would* be submitted and why not, without
claiming anything. The usual causes:

* **`dft.max_cores` is reached.** The driver counts the cores this campaign's
  own queued *and* running jobs hold; `csp doctor` prints the arithmetic
  (`may submit N more ... (limit: max_cores=720, 480 held -> room for 3 more)`).
  If it says `0 held` and still holds work, the cap is not what is binding —
  read the limit it names instead
* your live QOS `max_submit` is the binding limit — the driver throttles
  against what `sacctmgr`/`scontrol` say, not against `machine.yaml`
* **`dft.select.max_total` has been reached.** It is a *lifetime* ceiling on
  the campaign, not a per-cycle throttle, and a campaign sitting at it looks
  exactly like a driver that has stopped working. `csp status` shows the count
  that entered DFT against it.
* `max_concurrent_tasks` was raised without `max_in_flight`. The throttle is
  `min(concurrent, in_flight)`, so the lower one wins and raising the other
  alone changes nothing.

### The driver log shows nothing for hours, but the database is changing

Two separate things, both fixed in D144; if you see them you are running older
code.

* **The log was buffered.** A driver writes to a file, where Python holds
  output in 8 KB blocks, so a slow cycle's report never reached the log until
  the cycle ended. `scripts/campaign_driver.sbatch` now sets
  `PYTHONUNBUFFERED=1`, and `csp run` flushes every line.
* **The cycle really was slow.** Structure writes each opened their own
  database connection and scanned every key-value row; on a 14,755-structure
  campaign over NFS, Phase A's bookkeeping took nine hours after two hours of
  GPU work. The same work now takes minutes.

To see progress without the log, count states directly (read-only, safe while a
driver runs):

```bash
sqlite3 "file:$WORKDIR/campaign.db?mode=ro" \
  "SELECT value, COUNT(*) FROM text_key_values WHERE key='state' GROUP BY value;"
```

### `csp` hangs silently, and `campaign.db.lock` exists

A process running an older cspflow was killed while it held ASE's lock file.
That lock is acquired with no timeout and a doubling back-off, so every later
`csp` command waits on it for ever without a message. Make sure nothing is
using the campaign (`squeue`, `ps -u $USER | grep csp`), then remove the file:

```bash
rm "$WORKDIR/campaign.db.lock"
```

Current cspflow never creates this file (D144): SQLite's own locks are released
when a process dies.

### A failed job is not retried

Exit 127 is `command not found`, and it is **never** retried: retrying it
unchanged burns a submission slot to reproduce the identical failure. Fix the
path in `machine.yaml`.

Otherwise, a structure gets one attempt per rung of the recipe's `retry`
ladder, plus the original. `csp status --why <id>` names the remedy each
attempt used.

### A retry rule never fires

Check you wrote `when:`, not `on:`. YAML 1.1 parses a bare `on` as the boolean
`True`, so `- on: timeout` becomes `{True: 'timeout'}` and the trigger
disappears. cspflow warns about a retry rule with no `when:`.

## Results

### Phase B starts but the ranking looks wrong

Check that the candidates and the hull are on the same energy scale:

```bash
csp doctor | grep -A3 "reference recipe"
```

`matches the store (<hash>) -- the hull is on one scale` is what you want. If it
warns that the two hashes differ, the hull **still builds and still looks
correct** — it just ranks wrongly, because half its numbers came from different
DFT settings. The fix is to copy the store's `dft:` block into your campaign
verbatim.

`recipe_id` hashes the recipe's `name:` and its physics, **not** `ntasks` or
`time` — so you can shrink the allocation freely without changing the scale,
but changing ENCUT or the k-point density breaks it.

### `recipe 'x' would inherit VASP defaults for tags that change the physics`

A recipe step omits `ENCUT`, `ISPIN`, `LASPH`, `NELM` or `LORBIT`. The error
says what VASP would do instead for each. See [the recipe
guide](09-recipes.md#incar--free-form-with-two-guardrails).

### A VASP job is `done` but the structure has no relaxed geometry

A run that exits cleanly at the ionic step limit is `done` and **not relaxed**.
`csp status` reports relaxations separately from job state for this reason, and
flags `not converged` counts with `<- not usable as a relaxed geometry`. Give
the ladder an `ionic_step_limit` rung with a higher `NSW` and
`remedy: resume_from_contcar`.

### `dft_e_above_hull` looks implausible

**First check whether it was written at all.** `reference.mode: recompute`
(the default) builds the DFT hull from our own recomputed MP phases and
**refuses** any system with incomplete coverage rather than borrowing MP's
number for the missing vertex. An absent `dft_e_above_hull` therefore means
missing reference phases, not a failed job:

```bash
csp reference status Ce-Fe-B     # what is missing, at your recipe_id
```

**If it is present but implausible, check that your recipe matches the store.**

```bash
csp doctor                                        # [OK] or [WARN] reference recipe
python scripts/why_recipe_differs.py campaign.yaml # which field differs
```

A campaign whose DFT policy differs from the store's by one field gets a
different `recipe_id`, reads an empty reference set, and every system is
refused. See [the reference set](10-reference-set.md).

### "Why is this structure not in my results?"

```bash
csp status --why 1042
```

Prints everything known about it and every gate it passed or failed, with the
value and the threshold. Read from recorded events, not reconstructed by
differencing counts — a structure can leave a state for more than one reason.

### `no campaign database at …`

Nothing has run yet. `csp source` writes the first rows.

## Still stuck

* `csp doctor` — most cluster-side problems have a named check
* `csp config show --origins` — most config-side problems are a value coming
  from a layer you forgot about
* `csp recipe` — the DFT ladder fully resolved, with nothing implicit
* [`examples/`](../examples/) — three campaigns known to load, with their
  numbers asserted by the test suite
