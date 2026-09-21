"""Campaign scaffolds: `csp init <type> <name>` copies one of the three examples.

A campaign is one of three types, one per source mode, and each has a complete
worked example in `examples/`:

    1  chemical_space     examples/1-chemical-space/
    2  composition_list   examples/2-composition-list/
    3  structure_list     examples/3-structure-list/

Those example files ARE the templates. `csp init` reads the example's
campaign.yaml and changes four lines -- `name`, `machine`, `dft.recipe` and
`reference.mode` -- and nothing else, so an example and the campaign it scaffolds
cannot drift apart. There used to be a separate all-modes template string here,
and it had already diverged from the examples (dead `calibrate:` blocks in one,
not the other).

The machine profile and DFT recipe are NOT copied from the example folder; they
are copied from the shipped files, so `-m local` still means the local profile.
The examples' own machine.yaml / recipe.yaml are pinned by a test to be exactly
what `machine_copy` / `recipe_copy` produce.

Inputs:  kind (1|2|3 or an alias), name, machine/recipe names, reference mode.
Outputs: campaign.yaml text, README text, and the example's demo input files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# templates/__init__.py -> templates -> cspflow -> src -> repository root.
EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"


@dataclass(frozen=True)
class CampaignType:
    number: int
    mode: str
    folder: str
    question: str               # what the user is saying when they pick it
    what_you_give: str
    aliases: tuple[str, ...]
    edit_first: tuple[str, ...]  # the lines to change before anything else

    @property
    def label(self) -> str:
        return f"type {self.number}: {self.mode}"


CAMPAIGN_TYPES: dict[int, CampaignType] = {
    1: CampaignType(
        number=1, mode="chemical_space", folder="1-chemical-space",
        question="search a region of the periodic table",
        what_you_give="element groups; cspflow enumerates the formulas and generates structures",
        aliases=("chemical_space", "chemical-space", "space", "sweep"),
        edit_first=(
            "campaign.yaml  source.chemical_space.groups   the elements, and how many to pick from each",
            "campaign.yaml  max_atoms_formula, defaults     the sweep's size -- check it with --dry-run",
        ),
    ),
    2: CampaignType(
        number=2, mode="composition_list", folder="2-composition-list",
        question="I know which formulas I want",
        what_you_give="formulas, inline or in a CSV; cspflow generates structures for exactly those",
        aliases=("composition_list", "composition-list", "compositions", "formulas"),
        edit_first=(
            "inputs/compositions.csv                        DEMO list -- replace it with your formulas",
            "campaign.yaml  source.composition_list.items   the inline exceptions (or delete them)",
        ),
    ),
    3: CampaignType(
        number=3, mode="structure_list", folder="3-structure-list",
        question="I already have the structures",
        what_you_give="POSCAR/CIF files; nothing is generated, they enter the funnel at screen",
        aliases=("structure_list", "structure-list", "structures", "seeds"),
        edit_first=(
            "inputs/seeds/                                  DEMO seeds -- replace them with your files",
            "campaign.yaml  structure_list.max_atoms        must clear your largest seed",
        ),
    ),
}


def parse_type(text: str | None) -> CampaignType | None:
    """`1`, `chemical_space`, `seeds` ... -> the type, or None if unrecognised."""
    if text is None:
        return None
    key = text.strip().lower()
    for ctype in CAMPAIGN_TYPES.values():
        if key == str(ctype.number) or key in ctype.aliases:
            return ctype
    return None


def type_menu() -> str:
    """The three types, as `csp init` prints them when it has to ask."""
    lines = ["Campaign types:", ""]
    for t in CAMPAIGN_TYPES.values():
        lines.append(f"  {t.number}  {t.mode:<18} \"{t.question}\"")
        lines.append(f"     {'':<18} you give {t.what_you_give}")
    lines += ["", "  csp init 1 my-sweep     csp init 2 my-shortlist     csp init 3 my-seeds"]
    return "\n".join(lines)


def example_dir(kind: CampaignType) -> Path:
    path = EXAMPLES_DIR / kind.folder
    if not (path / "campaign.yaml").is_file():
        raise FileNotFoundError(
            f"no example campaign at {path}. `csp init` copies the examples/ folder of the "
            f"cspflow checkout, so cspflow must be installed from it (pip install -e .).")
    return path


# --------------------------------------------------------------------------
# editing the example's campaign.yaml
# --------------------------------------------------------------------------


def _set_value(text: str, prefix: str, value: str, *, current: str = r"\S+") -> str:
    """Replace the value on the ONE line starting with `prefix`, keeping its comment
    in the same column. Refuses unless exactly one line matches, so an example
    edited out of shape fails loudly instead of scaffolding the wrong setting.
    """
    pattern = re.compile(rf"^({re.escape(prefix)})({current})([ \t]*#.*)?$", re.M)

    def swap(m: re.Match) -> str:
        comment = m.group(3) or ""
        if comment:
            width = len(m.group(2)) + (len(comment) - len(comment.lstrip()))
            comment = " " * max(1, width - len(value)) + comment.lstrip()
        return f"{m.group(1)}{value}{comment}"

    new, n = pattern.subn(swap, text)
    if n != 1:
        raise ValueError(f"expected exactly one line starting {prefix!r} in the example, found {n}")
    return new


def _banner(kind: CampaignType, name: str) -> str:
    return (
        "# ===========================================================================\n"
        f"#  {name} -- cspflow campaign, {kind.label}\n"
        f"#  \"{kind.question}\"\n"
        "#\n"
        f"#  Made by `csp init {kind.number} {name}` from examples/{kind.folder}/.\n"
        "#    campaign.yaml   what to search, and how hard        <- you are here\n"
        "#    machine.yaml    scheduler, partitions, codes, POTCAR trees\n"
        "#    recipe.yaml     the DFT ladder: INCAR tags, k-points, resources\n"
        "#    inputs/         your own structures or composition lists\n"
        "#    results/        everything the campaign produces\n"
        "#\n"
        "#  Every line that is not a comment is a live setting, written out even where\n"
        "#  it equals the default. Where a setting takes one of a fixed set of values,\n"
        "#  the others are listed beside it as  # a | b | c.  To see every resolved\n"
        "#  value and which file set it:   csp config show --origins\n"
        "# ===========================================================================\n"
    )


def _drop_banner(text: str) -> str:
    """Remove the example's own leading comment block ("EXAMPLE 1 of 3 ...")."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    return "".join(lines[i:]).lstrip("\n")


def _strip_comments(text: str) -> str:
    """Drop whole-line and trailing comments, collapsing the gaps.

    This is what makes `--minimal` safe: the terse file is the annotated file
    with the annotations removed, so a knob can never exist in one and not the
    other. A trailing comment is recognised by the two spaces before its `#`,
    which is how every example writes them.
    """
    kept = []
    for line in text.splitlines():
        if re.match(r"\s*#", line):
            continue
        kept.append(re.sub(r"\s{2,}#.*$", "", line))
    out: list[str] = []
    for line in kept:
        if not line.strip() and (not out or not out[-1].strip()):
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def campaign_yaml(kind: CampaignType | int | str, *, name: str,
                  machine: str = "machine.yaml", recipe: str = "recipe.yaml",
                  minimal: bool = False, reference_mode: str | None = None) -> str:
    """The campaign file for `kind`: the example's, with four lines changed.

    `reference_mode` is the answer to the one question `csp init` asks -- whether
    the MP reference phases get recomputed at this campaign's own DFT settings
    (D126). It is written as the live value of `reference.mode`.
    """
    ctype = kind if isinstance(kind, CampaignType) else parse_type(str(kind))
    if ctype is None:
        raise ValueError(f"unknown campaign type {kind!r}; expected 1, 2 or 3")
    text = (example_dir(ctype) / "campaign.yaml").read_text()

    text = _set_value(text, "name: ", name)
    text = _set_value(text, "machine: ", machine)
    text = _set_value(text, "  recipe: ", recipe)
    if reference_mode is not None:
        if reference_mode not in ("recompute", "mp_energies"):
            raise ValueError("reference_mode must be 'recompute' or 'mp_energies'")
        text = _set_value(text, "  mode: ", reference_mode, current=r"recompute|mp_energies")

    if minimal:
        header = (f"# {name} -- cspflow campaign, {ctype.label}. `csp init {ctype.number} {name}`\n"
                  "# without --minimal writes this file with every setting's alternatives\n"
                  "# beside it, plus editable copies of the machine profile and DFT recipe.\n")
        return header + "\n" + _strip_comments(text)
    return _banner(ctype, name) + "\n" + _drop_banner(text)


def example_inputs(kind: CampaignType) -> list[tuple[str, Path]]:
    """(path relative to the campaign, source file) for the example's demo inputs."""
    root = example_dir(kind)
    inputs = root / "inputs"
    if not inputs.is_dir():
        return []
    return [(str(p.relative_to(root)), p) for p in sorted(inputs.rglob("*")) if p.is_file()]


# --------------------------------------------------------------------------
# READMEs
# --------------------------------------------------------------------------


README = """\
# @NAME@

A cspflow campaign, @LABEL@ -- "@QUESTION@".
Made by `csp init @NUMBER@ @NAME@` from `examples/@FOLDER@/`.

| file | what you change there |
|------|-----------------------|
| `campaign.yaml` | what to search, how many structures, the hull cutoff |
| `machine.yaml`  | partition, walltime, modules, VASP binary, POTCAR trees |
| `recipe.yaml`   | the DFT ladder: INCAR tags, k-point density, resources |
| `inputs/`       | @INPUTS@ |

Everything the campaign produces -- the database, the screening results, every
DFT directory -- goes in `results/` inside this folder.

## Edit first

@EDIT_FIRST@

## Run it

```bash
conda activate cspflow
csp doctor                    # check machine, codes, POTCARs, env
csp source --dry-run          # what would be searched, before anything runs
csp run --through reference   # Phase A: cheap, runs to completion
csp run --from filter --watch # Phase B: expensive, streamed under max_cores
csp status                    # where everything is
csp report                    # report/report.html + report/candidates.csv
```

Every command finds `campaign.yaml` by walking up from wherever you are, so
these work from any folder inside the campaign, not just the top of it.

## Change one thing without editing a file

```bash
csp run --set filter.e_above_hull_max=0.05
csp config show --origins     # every resolved value, and which file set it
```
"""

_INPUTS_LINE = {
    1: "nothing -- a chemical_space campaign names its elements in campaign.yaml",
    2: "`compositions.csv`: the formulas to generate (demo list; replace it)",
    3: "`seeds/`: the structures to screen (demo Sm-Fe seeds; replace them)",
}

_INPUTS_README = {
    1: """\
A chemical_space campaign reads nothing from here: its elements are the
`source.chemical_space.groups` in campaign.yaml.

Put anything you want kept beside the campaign here -- cspflow never writes to
this folder. Adding a second source that does read files (a composition CSV or
a folder of seeds) is how a sweep gets a hand-picked control group.
""",
    2: """\
`compositions.csv` is the example's DEMO list, copied by `csp init`.
Replace it with your own formulas.

Format:  formula[,z_min,z_max,n_structures]
A header row is optional, blank cells inherit from `source.defaults`, and `#`
comments must be on a line of their own -- a trailing comment after data is read
as part of the last cell.

`csp source --dry-run` shows what the list expands to. cspflow never writes to
this folder.
""",
    3: """\
`seeds/` holds the example's five DEMO structures (DFT-relaxed Sm-Fe phases),
copied by `csp init`. Replace them with your own files.

Read: .vasp .poscar .contcar .cif .xyz .extxyz .res .json, and any file named
POSCAR or CONTCAR. Anything else in the folder is ignored, so a README beside
your seeds is fine. Every file is parsed by both pymatgen and ASE, and the two
must agree on its composition.

Check `structure_list.max_atoms` in campaign.yaml clears your largest seed.
`csp source --dry-run` lists what was read. cspflow never writes to this folder.
""",
}


def workspace_readme(kind: CampaignType, *, name: str) -> str:
    edit = "\n".join(f"* `{line.split()[0]}` {' '.join(line.split()[1:])}" for line in kind.edit_first)
    return (README.replace("@NAME@", name).replace("@LABEL@", kind.label)
                  .replace("@QUESTION@", kind.question).replace("@NUMBER@", str(kind.number))
                  .replace("@FOLDER@", kind.folder).replace("@INPUTS@", _INPUTS_LINE[kind.number])
                  .replace("@EDIT_FIRST@", edit))


def inputs_readme(kind: CampaignType) -> str:
    return _INPUTS_README[kind.number]


def machine_copy(source: Path, *, name: str) -> str:
    """A machine profile copied into the campaign folder, with a note on top."""
    header = (
        f"# Scheduler profile for the '{name}' campaign.\n"
        f"# Copied from the shipped profile {source.name} so it can be edited\n"
        f"# here -- partition, walltime, modules, VASP binary, POTCAR trees.\n"
        f"# Delete this file and set `machine: {source.stem}` in campaign.yaml\n"
        f"# to go back to the shipped one and pick up its updates.\n"
        f"#\n"
    )
    return header + source.read_text()


def recipe_copy(source: Path, *, name: str) -> str:
    """A DFT recipe copied into the campaign folder, with a note on top."""
    header = (
        f"# DFT recipe for the '{name}' campaign.\n"
        f"# Copied from the shipped recipe {source.name}. Every INCAR tag, the\n"
        f"# k-point density and the per-stage resources are here and editable.\n"
        f"# `csp recipe` prints it fully resolved, with nothing deferred.\n"
        f"# Delete this file and set `dft.recipe: {source.stem}` in campaign.yaml\n"
        f"# to go back to the shipped one.\n"
        f"#\n"
    )
    return header + source.read_text()
