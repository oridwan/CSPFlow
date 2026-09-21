"""Where a campaign's DFT work lives on disk, and what it is called.

TWO LAYOUTS, BECAUSE ONE IS ALREADY RUNNING
-------------------------------------------
`stages` is what exists today: one directory per structure PER STEP, all of them
flat in `<work>/dft/`, plus a `.sbatch`, a `.tasks.json`, a `claim-*.json` and a
`slurm-*.out` beside them.  A 94-structure campaign puts **428 entries** at that
one level and 2,588 files beneath it.

`runs` is the replacement: one directory per STRUCTURE, holding its seed, its
script, its log, and one subdirectory per step.  The same campaign becomes 94
top-level entries.

Both are supported because CeFeB and CePdGe are mid-flight on `stages` and
moving 15 GB of a live campaign is exactly the sort of operation that fails
quietly.  New campaigns get `runs`; existing ones finish as they are.

WHY THE NAMES CARRY THE CHEMISTRY
---------------------------------
`dft-18-relax` says nothing.  The seed it came from was
`02-agentic__Ce2Fe11Co3B_x3_o5-7_Co.vasp` -- which says what the structure IS --
and that was thrown away at submission.  So a run directory is

    0018-Ce8Co12Fe44B4-agentic_Ce2Fe11Co3B_x3_o5-7_Co

  * the zero-padded id keeps `ls` in database order and keeps the row findable;
  * the formula makes the campaign readable without opening anything;
  * the seed label preserves the provenance that names the substitution.

It also removes a real collision: `dft-69-relax` exists in CeFeB AND in CePdGe,
because every campaign numbers its structures from 1.  Two directories with the
same name and different contents is how a job gets matched to the wrong folder.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Long names are worse than short ones once they stop fitting in a terminal.
MAX_LABEL = 48
MAX_SLUG = 96


def slugify(text: str) -> str:
    """A filesystem- and shell-safe fragment of `text`.

    Deliberately strict: only letters, digits, `_`, `-` and `.` survive. A
    directory name reaches a shell, an sbatch script, a JSON manifest and a
    glob, and the character that breaks one of those is rarely the one you
    thought about.
    """
    text = re.sub(r"\.(vasp|cif|poscar|json|yaml)$", "", str(text).strip(), flags=re.I)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return re.sub(r"_{2,}", "_", text).strip("_-.") or "x"


def seed_label(source_path: str | None) -> str:
    """The informative part of a seed filename.

    Seeds arrive as `02-agentic__Ce2Fe11Co3B_x3_o5-7_Co.vasp` or
    `parent__Ce2Fe14B.cif`. Only the leading digits go: they are a
    source-ordering prefix every seed in the batch shares, so they separate
    nothing and cost width.

    The word before `__` is KEPT. It names the source mode -- `agentic`,
    `strain`, `parent` -- and dropping it makes `parent__Ce2Fe14B.cif` and a
    substituted `Ce2Fe14B` indistinguishable, which is precisely the comparison
    the campaign exists to make.
    """
    if not source_path:
        return ""
    stem = Path(str(source_path)).name
    stem = re.sub(r"^\d+[-_]+", "", stem)          # drop a leading "02-"
    return slugify(stem)[:MAX_LABEL]


def run_slug(structure_id: int, formula: str = "", source_path: str | None = None) -> str:
    """The directory name for one structure's DFT work.

    The id leads and is zero-padded so that string sort is numeric sort -- with
    a bare id, `ls` puts structure 100 between 10 and 11, and a campaign of a few
    hundred becomes unreadable in exactly the place you go to read it.
    """
    parts = [f"{int(structure_id):04d}"]
    if formula:
        parts.append(slugify(formula))
    label = seed_label(source_path)
    if label:
        parts.append(label)
    return "-".join(parts)[:MAX_SLUG]


def structure_id_of(slug: str) -> int | None:
    """The id back out of a slug, or None if it does not carry one."""
    m = re.match(r"^(\d+)(?:-|$)", str(slug))
    return int(m.group(1)) if m else None


class Layout:
    """Path policy. One instance per campaign; `name` is what config selected."""

    def __init__(self, name: str, dft_root: Path) -> None:
        if name not in ("stages", "runs"):
            raise ValueError(f"unknown layout {name!r}; expected 'stages' or 'runs'")
        self.name = name
        self.dft_root = Path(dft_root)

    # -- directories -------------------------------------------------------

    def run_root(self, structure_id: int, formula: str = "",
                 source_path: str | None = None) -> Path:
        """The directory that owns everything about one structure."""
        if self.name == "stages":
            # There is no such directory in the old layout; the closest thing is
            # the flat dft/ root, and callers must not assume otherwise.
            return self.dft_root
        return self.dft_root / "runs" / run_slug(structure_id, formula, source_path)

    def stage_dir(self, structure_id: int, step_name: str, formula: str = "",
                  source_path: str | None = None) -> Path:
        """Where one step of one structure runs."""
        if self.name == "stages":
            return self.dft_root / f"dft-{int(structure_id)}-{step_name}"
        return self.run_root(structure_id, formula, source_path) / step_name

    # -- files -------------------------------------------------------------

    def structure_json(self, structure_id: int, formula: str = "",
                       source_path: str | None = None) -> Path:
        return self.run_root(structure_id, formula, source_path) / "structure.json"

    def potcar_store(self) -> Path:
        """Shared POTCARs.

        A campaign writes one POTCAR per run directory: 109 copies of 8 distinct
        files in CeFeB, 90 MB that grows with every structure and every campaign.
        They are identical by construction -- the same elements in the same order
        at the same functional -- so they are stored once and linked.
        """
        return self.dft_root / ".potcars"

    def is_run_dir(self, path: Path) -> bool:
        return self.name == "runs" and path.parent.name == "runs"


def resolve(configured: str, dft_root: Path) -> tuple[str, str]:
    """The layout to actually use, and why, given what is on disk.

    Configuration is an intention; the directories are a fact. A campaign that
    already has `dft-<id>-<step>/` directories has finished work arranged the
    old way, and honouring a `runs` setting would point every path at somewhere
    empty -- reporting nothing done, and resubmitting calculations that are
    sitting complete on disk.

    So evidence wins, and says so. The reverse case is symmetrical: a `runs/`
    tree with `stages` configured.
    """
    root = Path(dft_root)
    has_stage_dirs = any(root.glob("dft-*-*/OUTCAR")) or any(
        p.is_dir() and re.match(r"^dft-\d+-\w+$", p.name) for p in root.glob("dft-*")
    ) if root.is_dir() else False
    has_run_dirs = (root / "runs").is_dir() and any((root / "runs").iterdir()) \
        if root.is_dir() else False

    if has_stage_dirs and not has_run_dirs and configured != "stages":
        return "stages", (
            f"{root} already holds dft-<id>-<step> directories from the older "
            f"layout, so 'stages' is in use here regardless of the configured "
            f"'{configured}'. Pointing a running campaign at a different layout "
            f"orphans its finished work and re-runs it."
        )
    if has_run_dirs and not has_stage_dirs and configured != "runs":
        return "runs", (
            f"{root}/runs already exists, so 'runs' is in use here regardless of "
            f"the configured '{configured}'."
        )
    return configured, ""


def write_structure_json(path: Path, payload: dict) -> Path:
    """One structure's own record, beside its work.

    WHY THIS EXISTS.  On 2026-09-12 a campaign database lost 37 finished
    calculations -- two drivers shared one job table, and whichever polled first
    marked a job terminal; when that driver did not own the job's stage it could
    not read the results, and nothing looked at the row again.  The OUTCARs were
    on disk, converged, and unreachable: 31 structures sat in `dft_queued`
    holding work nobody could see.

    Nothing on disk recorded what had been decided.  With this file the database
    becomes a cache that can be rebuilt from the directories, rather than the
    only copy of the answer.

    Written to a temporary file and renamed, because a reader is likely to be
    `watch cat` or a status pass running at the same moment, and a half-written
    file shows up as a parse error at random.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


def read_structure_json(path: Path) -> dict:
    """`{}` for anything unreadable -- a missing record is not a crash."""
    import json

    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def share_potcar(directory: Path, store_dir: Path) -> Path | None:
    """Replace a written POTCAR with a link to one shared copy.

    A campaign writes one POTCAR per run directory: 109 copies of **8 distinct
    files** in CeFeB, 90 MB that grows with every structure and every campaign.
    They are identical by construction -- the same elements, in the same order,
    at the same functional.

    Content-addressed, so two structures sharing a POTCAR share the file and a
    POTCAR that differs gets its own. Returns the shared path, or None if the
    link could not be made -- in which case the real file is simply left where
    it is, because a space optimisation must never cost a calculation.
    """
    import hashlib
    import os

    src = Path(directory) / "POTCAR"
    if not src.is_file() or src.is_symlink():
        return None
    try:
        digest = hashlib.sha1(src.read_bytes()).hexdigest()[:16]
        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        shared = store_dir / f"POTCAR-{digest}"
        if not shared.exists():
            tmp = store_dir / f".{digest}.tmp"
            tmp.write_bytes(src.read_bytes())
            tmp.replace(shared)
        src.unlink()
        os.symlink(os.path.relpath(shared, src.parent), src)
        return shared
    except OSError:
        return None
