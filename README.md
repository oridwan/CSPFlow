![cspflow — thousands of generated candidate crystals funnelled down to the few that sit lowest on the energy landscape](docs/assets/cspflow-cover-v3.png)

# cspflow

End-to-end high-throughput crystal structure prediction and first-principles
discovery. Give it a chemical space, a list of compositions, or a folder of
structures; it generates candidates, screens them with an MLIP, places them on a
convex hull, runs the survivors through VASP, and reports what came out.

```
source ──► generate ──► screen ──► dedup ──► reference ──► calibrate ──► filter ──► dft ──► analyze
└─────────────── Phase A: cheap, run to completion ────────────────┘     └ Phase B: expensive, streamed ┘
```

Everything lives in one SQLite file per campaign, so a run can be stopped,
inspected and resumed at any point. Cheap MLIP screening runs as a barrier;
expensive DFT runs as a throttled stream under a core cap (`dft.max_cores`).

**📖 [User guide](docs/) · [Quick start](docs/02-quickstart.md) · [Examples](examples/)**

## Install

```bash
pip install "cspflow[full]"
export MP_API_KEY=...            # reference energies come from Materials Project
csp doctor --fix                 # check the cluster; --fix builds the POTCAR layout
```

Structure generation and MLIP screening need MatterGen and MatterSim, which pin
each other and pin torch. `./scripts/build_env.sh cspflow` builds one conda
environment holding all three. Full detail: [Installation](docs/01-installation.md).

## Run a campaign

A campaign is a **folder**, and every knob you might turn is in it:

```bash
csp init 1 my-campaign      # 1 chemical_space | 2 composition_list | 3 structure_list
cd my-campaign
```

| type | source mode | the question it answers | you give |
|---|---|---|---|
| `1` | `chemical_space` | "search a region of the periodic table" | element groups |
| `2` | `composition_list` | "I know which formulas I want" | formulas, inline or a CSV |
| `3` | `structure_list` | "I already have the structures" | POSCAR/CIF files |

The folder is a copy of [the example for that type](examples/). `csp init my-campaign`
without a type prints this menu and asks.

```
my-campaign/
├── campaign.yaml   what to search, and how hard   ← every setting live, alternatives beside it
├── machine.yaml    partitions, walltime, modules, VASP, POTCAR trees
├── recipe.yaml     the DFT ladder: INCAR tags, k-points, per-step resources
├── inputs/         your own structures or composition lists (types 2, 3: demo copies)
├── results/        everything the campaign produces
└── report/         report.html + candidates.csv
```

```bash
$EDITOR campaign.yaml                   # elements, cutoffs, how many structures
csp doctor                              # then fix whatever it flags

csp source --dry-run                    # what would be enumerated, nothing written
csp run --through reference             # Phase A: generate, screen, dedup, reference
csp run --from filter --watch           # Phase B: stream DFT under dft.max_cores

csp status                              # progress
csp status --why 1042                   # the full life history of one structure
csp report                              # report/report.html + candidates.csv
```

Commands find `campaign.yaml` by walking up from wherever you are, so they work
from any folder inside the campaign. Any stage slice works: `--only screen`,
`--from dft`, `--through dedup`.

In `campaign.yaml` the live keys are the ones you must choose; **every other key
is there too, commented out**, showing the default already in effect and
indented where it belongs — deleting the leading `# ` is the whole edit.

```bash
csp config show --origins               # every resolved value, and which file set it
csp recipe                              # this campaign's DFT ladder, fully resolved
csp run --set filter.e_above_hull_max=0.05    # change one thing without an edit
```

## Three ways in

Complete, runnable versions of all three are in [`examples/`](examples/) — every
block populated and commented, with a sample CSV and real seed structures.

**A chemical space sweep** — every system the groups allow:

```yaml
source:
  - mode: chemical_space
    chemical_space:
      groups:
        A: {elements: [Sm, Tb],                pick: 1}
        B: {elements: [Fe, Co, Ni],            pick: 1, min_fraction: 0.75}
        C: {elements: [Ti, V, Cr, Mn, Cu, Zn], pick: 1}
      max_atoms_formula: 20
```

**An explicit list of compositions** — same funnel, no enumeration:

```yaml
source:
  - mode: composition_list
    composition_list:
      items: [{formula: Sm2Fe17}, {formula: SmFe11Ti, z: [1, 2]}]
      from_file: inputs/compositions.csv   # formula[,z_min,z_max,n_structures]
```

**Structures you already have** — no generation at all; POSCARs and CIFs go
straight to MLIP relaxation and DFT:

```yaml
source:
  - mode: structure_list
    structure_list:
      paths: [inputs/seeds]
      relax: true
      dedup: warn        # 'warn', not 'drop': a curated list is not a duplicate pool
```

Several sources can run in one campaign — give each a `name` and a seed set
stays distinguishable from the sweep it is a control for.
[More →](docs/03-sources.md)

## Documentation

New here? Open the **[numbered user guide](docs/)** and read Guides 01–04 in
order. The remaining pages are references you can use as the need arises.

| | |
|---|---|
| [01 — Installation](docs/01-installation.md) | Install, the MP key, POTCARs, the ML environment |
| [02 — Quick start](docs/02-quickstart.md) | Create and safely check a small first campaign |
| [03 — Choosing your input](docs/03-sources.md) | The three source modes in depth, and the CSV format |
| [04 — A-to-Z workflows](docs/04-workflows.md) | A complete run for each source mode |
| [05 — `campaign.yaml` explained](docs/05-campaign.md) | The main file, block by block |
| [06 — Every campaign setting](docs/06-settings.md) | Exact keys, defaults, and common mistakes |
| [07 — Pipeline stages](docs/07-stages.md) | What each stage does, and why the run has two phases |
| [08 — Running on your cluster](docs/08-machines.md) | Partitions, modules, walltime, POTCAR trees, VASP |
| [09 — The DFT recipe](docs/09-recipes.md) | INCAR tags, k-points, resources, the retry ladder |
| [10 — The reference set](docs/10-reference-set.md) | The shared one-scale DFT data behind convex hulls |
| [11 — Reading the results](docs/11-results.md) | `report.html`, `candidates.csv`, the database |
| [12 — Command reference](docs/12-cli.md) | Every `csp` command and option |
| [13 — Troubleshooting](docs/13-troubleshooting.md) | The errors you will actually hit |
| [`examples/`](examples/) | Three complete campaigns with sample inputs |



## License

MIT.
