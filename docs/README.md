# cspflow user guide

This guide takes you from a clean installation to a finished crystal-structure
prediction campaign. You do **not** need to read every page before you begin.
For a first run, read Guides 01–04 in order and use the remaining pages when
you need more detail.

## New user reading order

The number is both the suggested reading order and part of the filename, so the
pages also stay in order in a file browser. It does not change any `csp`
command.

| No. | Read this | What you will be able to do afterward |
|---:|---|---|
| 01 | [Installation](01-installation.md) | Install cspflow and check Python, the cluster, VASP, POTCARs, and environment variables. |
| 02 | [Quick start](02-quickstart.md) | Create, check, run, stop, resume, and report one small campaign. |
| 03 | [Choosing your input](03-sources.md) | Choose between a chemical space, a formula list, and structures you already have. |
| 04 | [A-to-Z workflows](04-workflows.md) | Follow the complete workflow for your chosen input mode. |
| 05 | [`campaign.yaml` explained](05-campaign.md) | Understand the main campaign file block by block. |
| 06 | [Every campaign setting](06-settings.md) | Look up exact keys, defaults, accepted values, and common mistakes. |
| 07 | [Pipeline stages](07-stages.md) | Understand what the pipeline is doing and where each result comes from. |
| 08 | [Running on your cluster](08-machines.md) | Map resource roles to partitions, modules, VASP, and POTCAR locations. |
| 09 | [The DFT recipe](09-recipes.md) | Control INCAR tags, k-points, resources, and retries. |
| 10 | [The reference set](10-reference-set.md) | Build and check the shared DFT data used for convex hulls. |
| 11 | [Reading the results](11-results.md) | Read status, reports, candidate tables, and rejection reasons. |
| 12 | [Command reference](12-cli.md) | Look up every `csp` command and option. |
| 13 | [Troubleshooting](13-troubleshooting.md) | Diagnose common configuration, cluster, VASP, and result problems. |

If your group already has cspflow installed and configured, start at
[Guide 02: Quick start](02-quickstart.md). If you already have crystal structures,
Guide 03 will point you to `structure_list`, the shortest workflow.

## Working examples

| Resource | Use it for |
|---|---|
| [Examples](../examples/) | Complete configuration examples for all three input modes. Copy the closest one and edit it. |
| [Small test campaigns](../campaigns/tests/) | Short campaigns designed to verify a setup before committing a large allocation. |
| [Smoke-run campaign](../campaigns/smoke-runs/) | A concrete `structure_list` campaign with three input structures and a three-job DFT limit. |

## The three files you edit

Every campaign is a folder. Most new users only need to understand these three
files:

| File | Plain-language purpose |
|---|---|
| `campaign.yaml` | What materials to search, which candidates to keep, and how much work may run. |
| `machine.yaml` | Where jobs run: partitions, modules, executables, POTCARs, and resource limits. |
| `recipe.yaml` | How VASP runs: relaxation/static steps, INCAR settings, k-points, and retries. |

`csp init <type> <name>` creates all three, copied from the
[example campaign](../examples/) for that type — `1` chemical_space, `2`
composition_list, `3` structure_list. Start by editing `campaign.yaml`; only
change the other two when your cluster or DFT policy requires it.

## Words used throughout the guide

- A **campaign** is one folder containing configuration, state, jobs, and a report.
- A **candidate** is one proposed crystal structure moving through the pipeline.
- A **source mode** says what you provide: element rules, formulas, or structures.
- A **stage** is one step such as generation, MLIP screening, DFT, or analysis.
- The **reference store** is shared, precomputed DFT data used to place candidates
  on a convex hull. It is separate from a campaign.
- **Phase A** is the cheaper preparation and screening work. **Phase B** spends
  VASP time on the best candidates.

## How it works, in one paragraph

You describe **what to search** in `campaign.yaml`. cspflow enumerates the
compositions, generates candidate structures with a diffusion model, relaxes
every one of them with a machine-learned interatomic potential, throws away
duplicates, places the survivors on a convex hull built from Materials Project
reference phases recomputed at **your own** DFT settings, and only then spends
VASP time — streamed under a core cap, best candidates first, one job
per structure. Everything lands in one SQLite file per campaign, so a run can be
stopped, inspected and resumed at any point.

```
source ──► generate ──► screen ──► dedup ──► reference ──► filter ──► dft ──► analyze
└──────────── Phase A: cheap, run to completion ──────────┘   └ Phase B: expensive, streamed ┘
```

`generate` is skipped when you supply the structures yourself
(`mode: structure_list`).

## Three things that surprise people

* **`dft.select.max_total` is a lifetime ceiling**, not a per-cycle throttle. A
  campaign that has reached it claims nothing further, which looks exactly like
  a driver that has stopped working.
* **Every energy on one hull must come from the same DFT settings.** If your
  `dft:` block disagrees with the reference store's, the hull still builds,
  still looks correct, and ranks wrongly. `csp doctor` tells you which it is.
* **`m_dft_raw ≈ 0` on a Ce or Sm compound is correct**, not a failed run. With
  `f_treatment: frozen` the 4f electrons are in the POTCAR core and cannot
  contribute a computed moment.
