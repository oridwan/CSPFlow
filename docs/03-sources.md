# 03 — Choosing your input

> **Guide 03 of 13** · Previous: [02 — Quick start](02-quickstart.md) · [Guide home](README.md) · Next: [04 — A-to-Z workflows](04-workflows.md)

Every campaign starts from a `source:` list. There are three modes, and a
campaign may hold several entries at once — a seed set and the sweep it is a
control for, in one run.

Use this simple rule:

1. If you already have POSCAR, CIF, or similar structure files, choose
   `structure_list`.
2. If you know the formulas but need candidate structures, choose
   `composition_list`.
3. If you want the pipeline to explore combinations of elements and formulas,
   choose `chemical_space`.

For a first test, prefer the earliest option that fits your science.
`structure_list` has the fewest moving parts; `chemical_space` has the largest
hidden size multiplier.

| mode | you supply | enters the funnel at |
|---|---|---|
| [`chemical_space`](#1-chemical_space) | element groups and rules | `generate` |
| [`composition_list`](#2-composition_list) | formulas, inline or in a CSV | `generate` |
| [`structure_list`](#3-structure_list) | structure files you already have | `screen` |

Complete, runnable versions of all three are in [`examples/`](../examples/).

Whatever the mode, check it before you spend anything:

```bash
csp source --dry-run          # identical code path, nothing written
csp source -n 50              # commit only the first 50 composition rows
```

---

## 1. `chemical_space`

*"Search this region of the periodic table."* You name element groups and how
many elements to take from each; cspflow assembles every chemical system the
rules allow and enumerates the formulas inside each one.

```yaml
source:
  - mode: chemical_space
    name: sweep
    chemical_space:
      groups:
        A: {elements: [Sm, Tb],                pick: 1}
        B: {elements: [Fe, Co, Ni],            pick: 1, min_fraction: 0.75}
        C: {elements: [Ti, V, Cr, Mn, Cu, Zn], pick: 1}
      max_atoms_formula: 20
      max_rare_earth: 1
    defaults:
      z: {min: 1, max: 2}
      max_atoms: 40
      n_structures: {mode: per_atom, structures_per_atom: 2.0}
```

**Group keys are labels only.** `A`/`B`/`C`, or `RE`/`TM`/`X` — call them
whatever reads best.

| key | type | default | meaning |
|---|---|---|---|
| `elements` | list | required | the pool to draw from |
| `pick` | int or list of ints | **no default** | how many elements to take. `[1, 2]` emits systems that take one *and* systems that take two. |
| `min_fraction` | 0–1 | none | this group must be at least this fraction of the atoms in the reduced formula |
| `max_fraction` | 0–1 | none | …and at most this |
| `max_atoms_formula` | int | 20 | cap on the **reduced** formula (Sm₂Fe₁₇ = 19 atoms) |
| `max_rare_earth` | int or null | 1 | rare-earth **species** allowed in one system |

`pick` has no default deliberately: it is the size multiplier. One element per
group gives binaries; allowing two adds every ternary and grows the sweep by
roughly an order of magnitude.

`max_rare_earth` counts species across the whole assembled system, not within
one group — so `{Sm, Tb}` with `pick: 2` will not quietly emit Sm-Tb-Fe.

`min_fraction` is the knob that keeps a sweep pointed at what you want. Without
`B: min_fraction: 0.75`, most of the budget goes to rare-earth-rich
compositions that cannot be a magnet.

**`max_atoms_formula` is the fastest dial.** Raise it from 20 to 30 and the
composition count climbs steeply. Always `--dry-run` after changing it.

---

## 2. `composition_list`

*"I know which formulas I want."* Same funnel, no enumeration.

```yaml
source:
  - mode: composition_list
    name: shortlist
    composition_list:
      items:
        - {formula: Sm2Fe17,  z: [1, 1], max_atoms: 38}
        - {formula: SmFe11Ti, z: [1, 2], n_structures: {mode: fixed, count: 40}}
      from_file: inputs/compositions.csv
    defaults:
      z: {min: 1, max: 2}
      max_atoms: 40
```

Per item: `formula` (required), `z`, `max_atoms`, `n_structures`. Anything you
omit inherits from `source.defaults`.

> `z: [1, 2]` shorthand works **only** for a per-item `z`. In
> `source.defaults` write the mapping form, `z: {min: 1, max: 2}`.

### Inline items and a file, together

Both may be used at once. **Inline items are read first**, and a formula in
both warns and keeps the inline one. That is the intended division of labour:
the file is the list you maintain, the inline items are the exceptions you are
making today.

### The CSV format

```
formula[,z_min,z_max,n_structures]
```

```csv
# Sm-Fe magnets. Blank cells inherit from source.defaults.
formula,z_min,z_max,n_structures
Sm2Fe17,1,2,80
SmFe12,1,2,
SmFe5,,,40
```

* A header row (`formula` first) is optional and skipped if present.
* Blank cells inherit from `source.defaults`, resolved per column — a row that
  sets only `n_structures` need not restate the Z range.
* **`#` comments must be on a line of their own.** A trailing comment after
  data is read as part of the last cell, and you get
  `column 4 (n_structures) is '80  # the 2:17 magnet' … not an integer`.

### What happens to a formula you write

Formulas are canonicalised to alphabetical order and reduced, so `Sm2Fe17` is
stored and reported as `Fe17Sm2`, and `Sm3Fe29Ti2` as `Fe29Sm3Ti2`. Match that
spelling when you grep the results. Reduction has two further consequences:

* **A non-reduced formula sets Z.** `Fe2Co10` reduces to `Co5Fe1` with an
  intrinsic multiplier of 2, so it is taken as a 12-atom cell, not the 6-atom
  one you would get by silently keeping the reduced formula at Z=1. The
  substitution is reported, not done quietly.
* **A non-reduced formula plus an explicit `z` is refused.** `Fe2Co10` with
  `z: [1, 2]` could mean Z ∈ {1,2} or Z ∈ {2,4} — a factor of two in cell size,
  with neither reading obviously right. Write the reduced formula and the range
  you mean.

Two items with the same reduced formula **and** the same Z are the same work:
the later is dropped and both spellings are named. Same formula, *different* Z
are different cells and both are kept — you are told, in case you meant to
write one.

A line that cannot be parsed **stops the campaign**. This is deliberately
unlike `csp ingest`, which tolerates junk: a composition list is something you
wrote, so a mistake in it is something you want to hear about before the run,
not a quietly missing result at the end.

---

## 3. `structure_list`

*"I already have the structures."* No generation at all. Files are read,
composition is **derived** from each one, and rows enter the funnel at `screen`.

```yaml
source:
  - mode: structure_list
    name: seeds
    structure_list:
      paths: [inputs/seeds, "inputs/extra/*.cif"]
      relax: true          # MLIP-relax the seed before DFT
      dedup: warn          # 'warn' or 'drop'
      # max_atoms: 60      # defaults to source.defaults.max_atoms
```

**Omit the `generate:` block entirely** in a structure-only campaign. Its
absence is what tells the driver Stage 1 has nothing to do.

`paths` takes files, globs and directories. A directory is walked for
`.vasp`, `.poscar`, `.contcar`, `.cif`, `.xyz`, `.extxyz`, `.res`, `.json`, and
files named `POSCAR*`/`CONTCAR*`; anything else is ignored rather than guessed
at, so a README beside your seeds is not an error.

`dedup: warn` is the default because a curated list is not a duplicate pool —
if you put two similar structures in deliberately, you want to be told, not
silently reduced to one. Use `drop` when the folder is a dump.

### Why a seed can be rejected

Every file is parsed by **both pymatgen and ASE**, and they must agree on the
composition. This costs milliseconds and prevents a class of silent wrong
answer that neither library avoids alone. Measured:

| file | pymatgen | ASE |
|---|---|---|
| VASP-4 POSCAR, comment `FeCo test` | `H1 He1` + a warning | `CoFe` |
| VASP-4 POSCAR, comment `my struct` | `H1 He1` + a warning | parse error |
| CIF with a 0.5/0.5 mixed Ti/Fe site | `Ti0.5 Fe1.5`, disordered | `Fe2`, **no warning** |

pymatgen invents hydrogen and helium for a POSCAR with no species line, with
nothing but a stderr warning — in a batch loop, H and He then proceed into the
MLIP and into VASP. ASE refuses that file, which is right, but silently
discards the minority species of a partial-occupancy CIF and returns a
clean-looking ordered cell that is not the material in the file.

So these are hard errors at ingest, not warnings: a VASP-4 POSCAR with no
species line, a disordered structure, a composition the two parsers disagree
on. Each of them otherwise surfaces after GPU or DFT time is spent — or never,
with a wrong number reaching the results table looking exactly like a right one.

---

## Sizing: `defaults`

Applies to every row a source produces (per-item overrides win, in mode 2 only).

| key | default | meaning |
|---|---|---|
| `z` | `{min: 1, max: 1}` | formula units of the reduced formula to build. Sm₂Fe₁₇ at Z=2 is a 38-atom cell. |
| `max_atoms` | 40 | hard cap on `z × atoms per formula unit`. A row that would exceed it is **dropped, not shrunk**. |
| `n_structures.mode` | `per_atom` | `per_atom` scales the count with cell size; `fixed` uses `count` for every row |
| `n_structures.structures_per_atom` | — | `per_atom`: a 20-atom cell at 2.0 gets 40 structures |
| `n_structures.count` | — | `fixed`: this many, regardless of size |
| `n_structures_scope` | `per_z` | whether that count is per Z value or split across the Z range |

## Several sources in one campaign

Give each a `name`. It is how a candidate's origin stays answerable in the
results table — a seed set stays distinguishable from the sweep it is a control
for, and `csp source --dry-run` reports each separately.

`name` is optional for a single source (it defaults to `default`) and effectively
required once there is more than one.


---

## See also

* [A-to-Z workflows](04-workflows.md) — a full run-through for each of these three
  modes, from `csp init` to `report.html`
* [Every setting in `campaign.yaml`](06-settings.md) — including each mode's own
  keys with their defaults
* [`campaigns/tests/`](../campaigns/tests/) — one small, runnable campaign per mode
