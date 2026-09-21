# 08 — Running on your cluster

> **Guide 08 of 13** · Previous: [07 — Pipeline stages](07-stages.md) · [Guide home](README.md) · Next: [09 — The DFT recipe](09-recipes.md)

`machine.yaml` is everything that depends on **where** you run, and nothing
that depends on **what** you are searching. That split is what lets the same
campaign move between clusters by changing one line.

`csp init` copies a shipped profile into the campaign folder, so you edit a
file you can see rather than one inside `site-packages`.

```bash
csp init 1 my-campaign -m orion          # or generic_slurm, or local
```

| shipped profile | for |
|---|---|
| `generic_slurm` | any SLURM site — only options every site understands. No partition names, no modules. Start here. |
| `orion` | the UNC Charlotte research cluster; a worked example of a fully specified site |
| `local` | this machine, no scheduler — for testing |

A campaign can also name a shipped profile directly (`machine: generic_slurm`),
which is what the [examples](../examples/) do. Once you have a real cluster,
copy it in.

## The file

```yaml
scheduler: slurm

partitions:
  cpu: {name: "Orion,Apus"}
  gpu: {name: "GPU"}
  bigmem: {name: "Hydrus"}

defaults:
  nodes: 1
  ntasks: 16
  cpus_per_task: 1
  mem: 32G

limits:
  cpu: {max_submit: 2048, max_cpus: 768, max_walltime: "30-00:00:00"}
  gpu: {max_submit: 128,  max_gpus: 12}

modules:
  cpu: [intel/mkl/2024.0, intel/2024, intel-mpi/2021.11]
  gpu: []

env:
  OMP_NUM_THREADS: 1

codes:
  vasp_std: /path/to/vasp_std
  mpi_launcher: "srun --mpi=pmi2"

potcar_root: /path/to/pmg
potcar_dirs:  {PBE_64: POT_PAW_PBE_64, PBE_52: POT_GGA_PAW_PBE_52}
potcar_trees: {VASP6.4: PBE_64, VASP5.2: PBE_52}

conda: {cpu: cspflow, gpu: cspflow}
scratch: /scratch/$USER
```

## Roles, not partition names

A campaign asks for `role: gpu`. This file decides what that means. Roles are
free-form — define `bigmem`, `short`, `debug`, whatever your site has — and a
campaign referring to a role you have not defined fails with the name, before
anything is submitted.

| `partitions.<role>` | meaning |
|---|---|
| `name` | site partition name(s); comma-separated for SLURM. `""` omits `--partition` and lets the site default apply |
| `qos` | |
| `account` | |
| `constraint` | `--constraint` |
| `exclude` | nodes to avoid |

## `defaults`

`nodes`, `ntasks`, `cpus_per_task`, `mem`. Anything a stage's `resources:`
block leaves out comes from here.

## `limits` — advisory, not authoritative

`max_submit`, `max_cpus`, `max_gpus`, `max_walltime` per role. These are
**starting points only**: `csp doctor` re-reads your live limits from
`sacctmgr` and `scontrol`, and the driver throttles against what it finds, not
against what is written here. Write them so a fresh clone behaves sensibly
before `doctor` has ever run.

## `modules` and `env`

Per role. Both are checked by `csp doctor`, and this is worth understanding:

> `module load` of a **missing** module writes to stderr and still exits 0. Under
> `set -e`, the job then dies at the *next* command with an error naming
> something unrelated. `doctor` checks every configured module against
> `module -t avail` so you find out before submitting.

`doctor` also checks that every `env` value that looks like a path exists —
which catches things like Intel MPI's `I_MPI_PMI_LIBRARY` pointing at a library
your site does not have, a failure whose native error message names neither the
variable nor the library.

## `codes`

| key | meaning |
|---|---|
| `vasp_std`, `vasp_gam`, `vasp_ncl` | absolute paths, or bare names to resolve from `$PATH` |
| `mpi_launcher` | e.g. `srun --mpi=pmi2`, or plain `srun` |

## POTCARs

```yaml
potcar_root: /path/to/pmg          # or leave null to use $PMG_VASP_PSP_DIR
potcar_dirs:  {PBE_64: POT_PAW_PBE_64}
potcar_trees: {VASP6.4: PBE_64}
```

pymatgen expects specific directory *names* that the distributed tarballs do
not use. `csp doctor --fix` creates the symlink layout under `potcar_root`;
without it a `functional: PBE_64` label can silently match a flat-layout
directory, and the recorded label then misdescribes what was used.

Nothing is copied and nothing is modified — `--fix` only adds symlinks.

## `conda`

Role → environment name. One environment for every role is the normal
arrangement (mattergen depends on mattersim, so they were always meant to share
one). The mapping exists for sites where a GPU stage genuinely needs a
different environment from a CPU stage.

## `scratch`

Where job scratch goes. `$VAR` is expanded, and an undefined variable is an
error rather than an empty string.

## Site-wide campaign defaults

A machine profile may also carry a `campaign:` block. Only that block joins the
campaign layer chain, and it sits *below* the campaign file — so a site can set
a sensible default `workdir` or `archive` without overriding what a user wrote.

```yaml
# in machine.yaml
campaign:
  archive: /projects/shared/cspflow_archive
  dft:
    max_in_flight: 100
```

## Mixed hardware in one partition

A partition is not necessarily one kind of machine, and VASP is compiled for a
specific instruction set. If your binary uses AVX-512 and the scheduler happens
to place the job on a node that has no AVX-512, VASP dies on the first such
instruction:

```
forrtl: severe (168): Program Exception - illegal instruction
```

with **no OUTCAR and no vasprun.xml** — it never got far enough to open them.
The run directory holds only INCAR, KPOINTS, POSCAR, POTCAR and `vasp.out`.

This is easy to miss because it looks like an ordinary failure and it is
intermittent: it depends entirely on where the scheduler put the job. Here it
produced 21 failures that all landed on just two nodes, `str-c92` (19) and
`str-c91` (2).

**How to check your own binary:**

```bash
objdump -d $VASP_BIN | grep -cE '%zmm[0-9]+'      # AVX-512 register uses
```

Ours returns 5,285.

Do not reason about this from CPU model names — measure it. One two-minute probe
job per architecture settles it:

```bash
sbatch -p Orion,Nebula,Apus --constraint=<feature> --wrap \
  'hostname -s; lscpu | grep -m1 "Model name"; lscpu | grep -o "avx512[a-z0-9_]*" | sort -u'
```

Measured on this cluster, 2026-09-09:

| feature | nodes | CPU | AVX-512 |
|---|---|---|---|
| `caslake` | `str-c[1-26,49-66,128-167]`, `str-bm1` | Xeon Gold 6248R | yes |
| `genoa` | `str-c[188-207]`, `str-c[171-177,218-221]` | EPYC 9654P | yes |
| `turin` | `str-c[67-70]` | EPYC 9655 | yes |
| `skylake` | `str-c[74-87]` (Nebula) | Xeon Gold 6154 | yes |
| `rome` | `str-c[88-97]` | EPYC 7502 | **none** |
| `milan` | `str-abm1` | EPYC (Zen 3) | **none** |
| `broadwell` | `str-bm5` | Xeon | **none** |

Two things that catch people out. Nebula's *Skylake* is Skylake-SP, which **does**
have AVX-512 — excluding it on the name alone would throw away a whole working
partition. And the non-AVX-512 machines are not all in one contiguous range:
`str-abm1` and `str-bm5` are bigmem nodes sitting inside the ordinary `Orion`
partition, so a deny-list of `str-c[88-97]` alone still leaves two traps.

**Two ways to fix it**, both in `machine.yaml`:

```yaml
partitions:
  # allow-list: name the features you need.  A new, unknown architecture is
  # excluded until someone checks it, which is the safe direction.
  cpu: {name: "Orion,Apus", constraint: "caslake|genoa|turin|skylake|sapphire_rapids"}

  # deny-list: name the nodes you know are wrong.  Simpler, but a node added
  # next month silently becomes a trap again.
  cpu: {name: "Orion,Nebula,Apus", exclude: "str-c[88-97],str-abm1,str-bm5"}
```

`constraint` becomes `#SBATCH --constraint=...` and `exclude` becomes
`#SBATCH --exclude=...`. The `|` in a constraint is SLURM's OR, and `#SBATCH`
lines are shell comments, so it is not interpreted by bash.

Check what the pool actually contains before choosing:

```bash
sinfo -p Orion,Nebula,Apus -o "%20N %10c %f" | sort -k3
```

## Checking it

```bash
csp doctor            # exits non-zero on any hard failure
csp doctor --fix      # and build the POTCAR symlink layout
csp doctor --elements Sm,Fe,Ti    # resolve POTCARs for these, campaign or not
```

`doctor` resolves a POTCAR for every element, reads live SLURM limits, verifies
each module and the VASP binary, and reports what it could not confirm. It is
safe at the top of a submit script.


---

## See also

* [Every setting in `campaign.yaml`](06-settings.md)
* [A-to-Z workflows](04-workflows.md)
* [Troubleshooting](13-troubleshooting.md) — most cluster errors are listed there by message
