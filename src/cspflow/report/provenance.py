"""Where a seed came from, and what was changed to make it.

WHAT THIS ANSWERS
    A card in the report shows a relaxed cell and its numbers.  It did not say
    the one thing a reader of a substitution campaign asks first: *what is this,
    and what did we change to get it?*  `Ce2Al2Ge4Pd` in a table is a formula.
    "two Al on the Ge sublattice of Ce2PdGe6, orbits 1 and 5, placed by hand"
    is the experiment.

WHERE THE ANSWER LIVES
    `<campaign>/seed_provenance.csv`, written by `scripts/stage_campaign.py`
    when the seeds were staged into `inputs/`.  One row per seed file:

        seed_file          the name in inputs/ -- the join key
        source_tag         which staging route produced it (01-manual, ...)
        source_folder      the library it was copied from
        original_filename  what it was called there
        method             how it was made, in words
        parent             the structure it was derived from
        what_varies        which sublattice, and how far
        n_atoms, elements, md5

    The join is exact: a structure row's `source_path` key is `inputs/<seed_file>`,
    so the basename is the CSV key.  Nothing is inferred from the formula and
    nothing is parsed out of the filename -- `Ce2Al2Ge4Pd_x2_o1-5_Al.vasp` does
    encode "x=2, orbits 1 and 5, Al", but a filename is a label somebody typed
    and the CSV is a record something wrote.

WHAT IT DOES NOT CLAIM
    This is a record of what the staging step was TOLD, not a re-derivation from
    the structure.  It says what was intended; the cell on the card is what VASP
    ended on.  The md5 is of the seed **as supplied**, so it will never match the
    relaxed cell shown beside it -- which is why it is labelled as the seed's.

    A campaign with no `seed_provenance.csv` (anything generated rather than
    staged) simply gets no provenance block.  An empty block is better than an
    invented one.

INPUTS   a campaign directory, or a store path to derive one from
OUTPUTS  a `SeedRecord` per structure, and the prose line the card prints
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FILENAME = "seed_provenance.csv"

# How a `method` value reads in a sentence. The CSV holds the short label the
# staging script wrote; this is the same fact in prose, because the card is
# read by someone who was not in the session that made the seed.
METHOD_PROSE = {
    "given": "taken as given",
    "manual, in-session": "placed by hand during the session",
    "manual, new tool": "placed by hand with the partial-sublattice tool",
    "agentic, one command": "enumerated by the agent from a single command",
    "Qiang, hand design": "hand-designed",
}


@dataclass(frozen=True)
class SeedRecord:
    """One row of `seed_provenance.csv`."""

    seed_file: str
    source_tag: str = ""
    source_folder: str = ""
    original_filename: str = ""
    method: str = ""
    parent: str = ""
    what_varies: str = ""
    md5: str = ""

    @property
    def is_parent(self) -> bool:
        return self.source_tag == "parent" or self.method == "given"

    @property
    def origin_path(self) -> str:
        """Where the seed was copied from, as one path."""
        if self.source_folder and self.original_filename:
            return f"{self.source_folder}/{self.original_filename}"
        return self.original_filename or self.source_folder

    def sentence(self) -> str:
        """One line: what this is, and what was changed to get it.

        Built from the CSV's own words rather than a template with blanks, so a
        route the staging script invents later still produces a readable line
        instead of a sentence with a hole in it.
        """
        if self.is_parent:
            return (f"The parent structure, {self.parent}, taken as given. "
                    f"Nothing was substituted: this is the reference every "
                    f"other candidate in the campaign is measured against.")

        # Three clauses rather than one flowing sentence, because `what_varies`
        # is a free-text column and its values are not all the same part of
        # speech: "Fe sublattice, x<=6" is a noun phrase, "layer replacement"
        # and "modular block substitution" are not. "by varying the layer
        # replacement" is what a template with a slot produced; naming the
        # column instead reads correctly for every value the CSV can hold.
        how = METHOD_PROSE.get(self.method, self.method or "staged")
        varies = self.what_varies or "not recorded"
        parent = self.parent or "an unrecorded parent"
        route = f" ({self.source_tag} route)" if self.source_tag else ""
        return (f"Derived from {parent}. What varies: {varies}. "
                f"{how[0].upper()}{how[1:]}{route}.")


def load(campaign_dir: Path | None) -> dict[str, SeedRecord]:
    """Every seed record, keyed by seed file name. Empty when there is no file."""
    if campaign_dir is None:
        return {}
    path = Path(campaign_dir) / FILENAME
    if not path.is_file():
        return {}

    out: dict[str, SeedRecord] = {}
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                name = (row.get("seed_file") or "").strip()
                if not name:
                    continue
                out[name] = SeedRecord(
                    seed_file=name,
                    source_tag=(row.get("source_tag") or "").strip(),
                    source_folder=(row.get("source_folder") or "").strip(),
                    original_filename=(row.get("original_filename") or "").strip(),
                    method=(row.get("method") or "").strip(),
                    parent=(row.get("parent") or "").strip(),
                    what_varies=(row.get("what_varies") or "").strip(),
                    md5=(row.get("md5") or "").strip(),
                )
    except (OSError, csv.Error):
        # A malformed provenance file loses the provenance block, not the
        # report. The numbers on the page do not depend on it.
        return {}
    return out


def campaign_dir_of(store_path: Path) -> Path | None:
    """Guess the campaign folder from the database path.

    Used only when the caller did not say; `csp report` passes `cfg.base_dir`.

    The database sits at `<campaign>/results/campaign.db`, so the campaign is
    two levels up -- but `results` is a **symlink to the scratch workdir**, and
    resolving it walks out of the campaign entirely: `campaigns/CePdGe/results/
    campaign.db` resolves to `/scratch/oridwan/cspflow/CePdGe/campaign.db`,
    whose parent's parent is `/scratch/oridwan/cspflow`. So the unresolved path
    is tried first, and every candidate is checked by looking for the file
    rather than trusted.
    """
    path = Path(store_path)
    candidates = [path.parent.parent, path.parent,
                  path.absolute().parent.parent, path.absolute().parent]
    try:
        resolved = path.resolve()
        candidates += [resolved.parent.parent, resolved.parent]
    except OSError:                                              # pragma: no cover
        pass
    for candidate in candidates:
        if (candidate / FILENAME).is_file():
            return candidate
    return None


def for_structure(records: dict[str, SeedRecord], kv: dict[str, Any]
                  ) -> SeedRecord | None:
    """The record for one structure row, matched on its `source_path` basename."""
    source = kv.get("source_path")
    if not source:
        return None
    return records.get(Path(str(source)).name)


def summary(records: dict[str, SeedRecord]) -> list[dict[str, Any]]:
    """How many seeds each route contributed, for the section's opening line."""
    counts: dict[tuple[str, str, str], int] = {}
    for record in records.values():
        key = (record.source_tag, record.method, record.what_varies)
        counts[key] = counts.get(key, 0) + 1
    return [{"tag": tag, "method": method, "what_varies": varies, "n": n}
            for (tag, method, varies), n in
            sorted(counts.items(), key=lambda kv: -kv[1])]
