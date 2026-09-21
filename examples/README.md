# Example campaigns

Three complete campaigns, one per **campaign type**. They are also the
templates: `csp init <type> <name>` copies the settings of one of these
`campaign.yaml` files, so an example and the campaign it starts cannot drift
apart. It does **not** copy the example chemistry -- the element groups, the
formulas, the CSV or the seed files. A new campaign starts empty and waits for
yours (D153). Copy an example's `inputs/` yourself if you want to run it.

| type | folder | source mode | the question it answers | input |
|---|---|---|---|---|
| `1` | [`1-chemical-space/`](1-chemical-space/) | `chemical_space` | "search this region of the periodic table" | element groups in the YAML |
| `2` | [`2-composition-list/`](2-composition-list/) | `composition_list` | "I know which formulas I want" | inline items **and** [`inputs/compositions.csv`](2-composition-list/inputs/compositions.csv) |
| `3` | [`3-structure-list/`](3-structure-list/) | `structure_list` | "I already have the structures" | five real POSCARs in [`inputs/seeds/`](3-structure-list/inputs/seeds/) |

## To use one

```bash
csp init 2 my-shortlist      # 1, 2 or 3 -- or the mode name
cd my-shortlist
csp doctor
csp source --dry-run
```

`csp init my-shortlist` without a type prints the table above and asks. The
new campaign's `csp source --dry-run` stops until you add your formulas, and says
so; to try the example as it is, run `csp source --dry-run -c examples/2-composition-list/campaign.yaml`.

## What is in each folder

Every example is a complete campaign of three YAML files, exactly what
`csp init` writes:

| file | what it is |
|---|---|
| `campaign.yaml` | what to search, and how hard -- different for each type |
| `machine.yaml` | a copy of the shipped `orion` profile, as `csp init` makes it |
| `recipe.yaml` | a copy of the shipped `magnets` DFT recipe, as `csp init` makes it |

The machine and recipe copies are generated, not hand-edited; a test pins them
to what `csp init` produces from the shipped files.

## How to read `campaign.yaml`

Every line that is not a comment is a **live setting**, written out even where
it equals the default. The comment beside it says what else it can be:

```yaml
magnetic_order: ferro        # ferro | ferri | none                 <- pick ONE
pick: 1                      # one number, or several: [1, 2]
elements: [Sm, Tb]           # several: [Sm, Nd, Pr, ...]           <- comma-separated list
overrides: {}                # several: {Sm: Sm_3, Ti: Ti_pv}       <- several key: value pairs
```

Every value listed after `# a | b | c` is checked against the schema by
`tests/unit/test_examples.py`, so a listed option is one the pipeline accepts.

**What is deliberately not there.** Settings that apply to another type (a
`structure_list` campaign has no `generate:` block and no `source.defaults`), and
settings the schema still accepts but no code reads: `calibrate:` (removed from
the funnel, D126), `archive:`, `analyze:`, `filter.e_above_hull_max_source`,
`dft.select.max_per_composition`, and `reference.functionals / prescreen_mode /
prescreen_hull_max / snapshot / snapshot_id / relax_with_mlip / cache`. An
existing campaign that sets them still loads. See D146.

## Check without running anything

```bash
csp source --dry-run -c examples/1-chemical-space/campaign.yaml
```

```
1-chemical-space     3,384 compositions   36 chemical systems   165,456 structures
2-composition-list      34 compositions    5 chemical systems     1,394 structures
3-structure-list         0 compositions    1 chemical system          5 seeds
```

Those numbers are asserted by `tests/unit/test_examples.py`, so an example that
stops matching its own description fails the suite rather than misleading you.

## What each one is really demonstrating

**1 — chemical_space** is the type with a size multiplier hidden in it. `pick`
has no default for that reason, and `max_atoms_formula` is the dial that moves
the composition count fastest. Run `--dry-run` before you believe any sweep.

**2 — composition_list** shows both ways of giving the list at once. Inline
items are read *before* the file, and a formula in both warns and keeps the
first — so the file is the list you maintain and the inline items are the
exceptions you are making today. The example does this on purpose with
`SmFe11Ti` and warns about it when you run it.

**3 — structure_list** has no `generate:` block, which is what tells the driver
there is nothing to generate. Its seeds are five DFT-relaxed Sm-Fe structures
from a finished campaign — Sm₂Fe₁₇ (hR19), SmFe₁₂ (tI26), SmFe₅, SmFe₄, SmFe₃ —
so the parsing, composition-derivation and `max_atoms` gate all run against real
files.

## The CSV format

`formula[,z_min,z_max,n_structures]`. A header row is optional, blank cells
inherit from `source.defaults`, and `#` comments must be on a line of their own
— a trailing comment after data is read as part of the last cell.
