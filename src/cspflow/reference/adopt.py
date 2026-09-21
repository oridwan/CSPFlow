"""Adopt a curated tree of finished VASP runs into the computed reference cache.

`build.export` reads a campaign *database*.  That is the right tool when the
reference set is one cspflow campaign and it finished cleanly.  It was not, and
it did not: the 2,746-phase recompute ran as three tiers (`t1`, `t2`, `t3`) and
then through eight repair rounds, and the result nobody has to reason about is
the tree the operator assembled afterwards --

    final/
        EXCLUDED.json           phases dropped on purpose, with reasons
        mp-1001113/
            relax/              a complete VASP run
            static/             a complete VASP run; ITS energy reaches the hull
            mattersim.json      the MLIP half, already a serialised ComputedPhase
        mp-.../

one directory per phase, holding the run that was finally accepted.  The tier
databases still carry every abandoned attempt beside it: 149 `failed` rows, 49
still queued, and -- the trap -- 84 rows that are `failed` yet carry a
`vasp_energy` from a relax whose static never ran.  `export` keys on "has an
energy", so adopting the tiers would put 88 half-finished phases into the cache
labelled `done`.  Adopting the tree cannot, because a directory only has an
energy if its own static produced one.

What this module checks, and why each check exists
--------------------------------------------------

A directory of VASP output is not self-describing.  Three things have to be
true before its energy may join a hull, and none of them is visible in the
number itself:

1.  **The policy matches.**  `recipe_id` hashes the settings that decide an
    energy, and the cache is keyed on it.  A directory computed at a different
    `ENCUT` is not a cheaper version of the same answer, it is a different
    answer -- so `INVARIANT` tags are compared against the recipe and a
    mismatch is refused, not warned about.

2.  **The geometry survived.**  VASP_FAILURES.md failure 12: four structures in
    this very campaign had their cells destroyed by a runaway relaxation, one by
    a factor of 677, and presented as four unrelated errors.  A destroyed cell
    still produces a total energy, and that energy still builds a hull.  The
    volume ratio and the energy per atom are checked because the error messages
    were not enough.

3.  **The physics deviations are recorded.**  The retries that rescued this
    campaign changed `SYMPREC`, `ALGO`, `NELMIN`, `IBRION` and `POTIM` -- none
    of which changes the converged energy -- and, on 83 of 2,713 phases,
    `ISMEAR`, which does.  Measured across the tree: every phase shares
    `ENCUT 520`, `PREC Accurate`, `SIGMA 0.05`, no `LDAU`, `LASPH .TRUE.`,
    `LREAL .FALSE.`, `LMAXMIX 4`.  So the set is on one scale in every respect
    but the smearing, and the smearing is written onto each record rather than
    averaged into silence.

Inputs
------
    root        a directory of `mp-*/` phase directories (`final/` above)
    recipe_id   the policy the cache is keyed on, from `-c campaign-settings.yaml`

Output
------
    one JSON record per phase under `$CSPFLOW_REFERENCE/computed/<rid>/`,
    plus an `AdoptReport` summarising what was taken and what was refused.

Run it with
-----------
    csp reference adopt /projects/mmi/cspflow-shared/store/final \
        -c /projects/mmi/cspflow-shared/store/campaign-settings.yaml --dry-run
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..dft.vasp.parse import read_incar, read_job_directory
from .computed import ComputedPhase, read_record, write_record

# The step whose energy reaches the hull.  `relax` finds the geometry; `static`
# is what is computed AT that geometry, and mixing the two is a difference of
# order 10 meV/atom -- small enough to look right and large enough to matter
# against a 60 meV/atom selection threshold.
ENERGY_STEP = "static"

# Tags that must be identical or the energies are not comparable.  These are the
# basis set and the functional treatment: nothing here is a knob a retry would
# touch, so a difference means the directory belongs to a different policy.
INVARIANT = ("ENCUT", "PREC", "SIGMA", "LASPH", "LREAL", "LMAXMIX", "LDAU")

# Tags that DO change the energy and did legitimately vary across this campaign.
# Recorded per phase rather than refused: refusing would discard 83 real results
# over a difference of a few meV, and averaging them into silence would hide a
# systematic the next person has every right to audit.
PHYSICS = ("ISMEAR", "ISPIN")

# Tags a retry is allowed to move because they change how the answer was
# reached, not what it is.  Listed explicitly so that a tag which is neither
# invariant, physics, nor here shows up as unclassified instead of being waved
# through by a default.
ROBUSTNESS = ("ALGO", "SYMPREC", "NELMIN", "NELM", "IBRION", "POTIM", "NSW",
              "ISIF", "EDIFF", "EDIFFG", "SMASS", "KPAR", "NCORE", "NBANDS",
              "MAGMOM", "LORBIT", "LCHARG", "LWAVE", "ISYM", "AMIX", "BMIX",
              "AMIX_MAG", "BMIX_MAG", "LMAXTAU", "ICHARG", "ISTART", "NPAR")

# Tags with no effect on anything computed.  Separated from ROBUSTNESS rather
# than merged into it so the three lists keep meaning what they say: these are
# not knobs that were turned, they are labels.
IGNORED = ("SYSTEM",)

# Failure 12's screen, split by what a bad value actually costs.
#
# The asymmetry decides which side each check falls on.  A phase wrongly
# REFUSED leaves `coverage` incomplete, and `entries_for` then refuses the whole
# chemical system -- one bad Cu entry would block every Cu-containing hull.  A
# phase wrongly ACCEPTED adds a point to the hull, and a point only changes a
# hull if it is LOW.  So: energy is a refusal, because a wrong low energy
# rewrites the envelope; geometry is a warning, because a vacuum box is always
# high and a high point cannot define a lower envelope.
#
# Measured over the 2,713 adopted phases: energy per atom spans -10.92 to
# -1.01, so these bounds are two eV clear of the data on the tight side and
# catch failure 12's +13.3 and -396.8 eV/atom outright.
MAX_E_PER_ATOM = 0.0
MIN_E_PER_ATOM = -20.0

# Volume per atom, measured on the same set: p1 7.53, median 15.19, p99 36.53,
# then 62.50 -- and then a clean gap to 220.71, 240.89, 250.97, 390.83, 612.55.
# Everything above the gap is an isolated atom or a cluster in a vacuum box,
# which is the defect EXCLUDED.json records for mp-1188724 at 217 A^3/atom.
# 100 sits inside the gap, so it separates the two populations without a guess.
SUSPICIOUS_VOLUME_PER_ATOM = 100.0
# A relaxation that moves the cell this far has usually been handed a bad input
# rather than produced a bad output -- mp-1207665 came in at 380 A^3/atom of
# vacuum and relaxed correctly to 19.8.  Worth saying, not worth refusing.
MAX_VOLUME_RATIO = 2.0
MIN_VOLUME_RATIO = 0.5


class AdoptError(Exception):
    """A phase directory that cannot be trusted into the cache."""


@dataclass
class AdoptReport:
    """What was taken, what was refused, and on what grounds."""

    root: str = ""
    recipe_id: str = ""
    adopted: int = 0
    skipped_existing: int = 0
    excluded: int = 0
    refused: list[tuple[str, str]] = field(default_factory=list)
    physics: dict[str, int] = field(default_factory=dict)
    # Adopted, but with something worth a second look recorded on the record.
    # Listed in full rather than counted: there were six, and the point of
    # finding them is that a person reads them.
    suspect: list[tuple[str, str]] = field(default_factory=list)
    unconverged: int = 0

    def render(self) -> str:
        lines = [
            f"recipe {self.recipe_id[:16]}  from {self.root}",
            f"adopted {self.adopted}"
            + (f", {self.skipped_existing} already present" if self.skipped_existing else "")
            + (f", {self.excluded} excluded by EXCLUDED.json" if self.excluded else "")
            + (f", {len(self.refused)} refused" if self.refused else ""),
        ]
        if self.physics:
            lines.append("physics settings actually used in the "
                         f"{ENERGY_STEP} step (these enter the energy):")
            for tag, _ in sorted(self.physics.items()):
                lines.append(f"  {tag}")
        if self.unconverged:
            lines.append(f"{self.unconverged} adopted phase(s) did not report "
                         f"convergence in the {ENERGY_STEP} step")
        for mp_id, why in self.refused:
            lines.append(f"  REFUSED {mp_id}: {why}")
        if self.suspect:
            lines.append(f"adopted with a warning on the record ({len(self.suspect)}):")
            for mp_id, why in self.suspect[:20]:
                lines.append(f"  {mp_id}: {why}")
            if len(self.suspect) > 20:
                lines.append(f"  ... and {len(self.suspect) - 20} more")
        return "\n".join(lines)


def _volume(path: Path) -> float | None:
    """Cell volume of a POSCAR/CONTCAR, or None if it cannot be read."""
    try:
        from ase.io import read as ase_read

        return float(ase_read(str(path)).get_volume())
    except Exception:                                          # noqa: BLE001
        return None


def check_policy(incar: dict[str, str], expected: dict[str, tuple[str | None, str]],
                 ) -> tuple[list[str], dict[str, str], list[str]]:
    """Compare one INCAR against the policy every other phase was computed under.

    Returns `(violations, physics_used, unclassified)`.  The split is the whole
    point: a violation means this directory was computed under a different
    policy and its energy does not belong in this cache, while a physics tag
    that differs is a real result computed slightly differently, which is
    recorded and kept.

    `expected` carries where each value came from, and the message says so,
    because the two mean different things to whoever reads the refusal.  A tag
    the recipe states is policy in the strict sense.  A tag the recipe leaves to
    resolution -- `LMAXMIX` is computed from whether f sits in the valence, not
    written in the stage -- has no recipe value to compare against, so the
    standard is the rest of the tree: 2,712 directories agreeing and one
    differing is exactly the stray-build case worth catching, and it is the only
    thing that can be checked without re-resolving every structure.
    """
    violations, physics = [], {}
    # Over `expected`, not over INVARIANT: a tag absent from `expected` is one
    # nothing can vouch for -- the recipe does not state it and the tree does
    # not agree on it -- and treating that absence as "expected <unset>" would
    # refuse every directory for having a value at all.
    for tag in INVARIANT:
        if tag not in expected:
            continue
        want, where = expected[tag]
        got = incar.get(tag)
        if want is None and got is None:
            continue
        if _norm(want) != _norm(got):
            violations.append(f"{tag}={got or '<unset>'} but {where} says "
                              f"{want or '<unset>'}")
    for tag in PHYSICS:
        if tag in incar:
            physics[tag] = str(incar[tag]).strip()
    known = set(INVARIANT) | set(PHYSICS) | set(ROBUSTNESS) | set(IGNORED)
    unclassified = sorted(t for t in incar if t not in known)
    return violations, physics, unclassified


def _norm(value: Any) -> str:
    """`1e-05`, `1.0E-05` and ` 1e-5 ` are one value; `.TRUE.` and `T` are not.

    Numbers are compared as numbers because VASP, pymatgen and a hand-edited
    INCAR all spell the same float differently, and a string compare would
    refuse a directory over its formatting.
    """
    if value is None:
        return "<unset>"
    text = str(value).strip()
    try:
        return f"{float(text):.10g}"
    except ValueError:
        return text.upper()


def check_geometry(outcome, initial_volume: float | None,
                   ) -> tuple[list[str], list[str]]:
    """Failure 12's screen, as `(fatal, suspect)`.

    Of the four destroyed structures found in that failure only one had an
    obviously wrong volume; for the other three the energy per atom was the
    tell. So both are measured -- but only the energy can refuse, for the
    reason given at `MAX_E_PER_ATOM`.
    """
    fatal, suspect = [], []
    if outcome.e_per_atom is not None and not (
            MIN_E_PER_ATOM <= outcome.e_per_atom <= MAX_E_PER_ATOM):
        fatal.append(
            f"{outcome.e_per_atom:.3f} eV/atom is outside the physical range "
            f"[{MIN_E_PER_ATOM}, {MAX_E_PER_ATOM}]: the geometry is destroyed and "
            f"this energy describes a consequence, not this phase")

    final_volume = _volume(outcome.path / "CONTCAR") or _volume(outcome.path / "POSCAR")
    if final_volume and outcome.n_atoms:
        per_atom = final_volume / outcome.n_atoms
        if per_atom > SUSPICIOUS_VOLUME_PER_ATOM:
            suspect.append(
                f"{per_atom:.0f} A^3/atom: an isolated atom or a cluster in a vacuum "
                f"box, not a bulk crystal. Kept because such a phase is always high "
                f"in energy and a high point cannot define a hull's lower envelope")
    if initial_volume and final_volume:
        ratio = final_volume / initial_volume
        if not MIN_VOLUME_RATIO <= ratio <= MAX_VOLUME_RATIO:
            suspect.append(
                f"cell volume moved by {ratio:.2f}x ({initial_volume:.1f} -> "
                f"{final_volume:.1f} A^3)")
    return fatal, suspect


def read_phase(directory: Path, *, rid: str,
               expected: dict[str, tuple[str | None, str]]) -> ComputedPhase:
    """One `mp-*/` directory as a cache record, or raise `AdoptError`.

    The MLIP half is read from `mattersim.json` rather than recomputed: it is
    already a serialised `ComputedPhase` written by `csp reference mlip`, and
    re-running MatterSim to rediscover numbers that are sitting on disk would
    cost hours and could not improve them.
    """
    mp_id = directory.name
    step = directory / ENERGY_STEP
    if not (step / "OUTCAR").is_file():
        raise AdoptError(f"no {ENERGY_STEP}/OUTCAR")

    # The MLIP half, from the richest source available.  An existing cache
    # record is preferred over `mattersim.json` because `csp reference mlip`
    # may have added to it -- `mlip_carried_from` is written there and not in
    # the tree -- and re-running MatterSim to rediscover numbers already on
    # disk would cost hours and could not improve them.
    phase = read_record(rid, mp_id)
    if phase is None or not phase.counts:
        base = directory / "mattersim.json"
        if base.is_file():
            from .computed import _from_dict

            phase = _from_dict(json.loads(base.read_text()))
    if phase is None:
        phase = ComputedPhase(mp_id=mp_id, formula=mp_id, chemsys="",
                              counts={}, n_atoms=0)
    if not phase.counts:
        raise AdoptError("no composition: mattersim.json is missing or has no counts")

    outcome = read_job_directory(step)
    # BOTH conditions, and the second is the one that matters.  A static killed
    # mid-run still leaves an OSZICAR holding the last electronic step, so
    # `energy is not None` is true for a job that never finished -- the same
    # trap as the 84 tier rows that are `failed` and carry a relax energy.
    # `state == 'done'` means VASP reached its own epilogue; nothing else does.
    if outcome.state != "done":
        raise AdoptError(
            f"{ENERGY_STEP} did not finish: {outcome.exit_reason or outcome.state}"
            + (f" (an energy of {outcome.energy} is present but the run was cut short)"
               if outcome.energy is not None else ""))
    if outcome.energy is None:
        raise AdoptError(f"{ENERGY_STEP} produced no energy ({outcome.exit_reason})")
    # The guard `to_entry` would raise on later, applied here where the fix is
    # cheap: a total energy paired with the wrong composition is a hull vertex
    # wrong by exactly that ratio, and nothing downstream can see it.
    if outcome.n_atoms and outcome.n_atoms != sum(phase.counts.values()):
        raise AdoptError(
            f"{ENERGY_STEP} ran {outcome.n_atoms} atoms but mattersim.json says "
            f"{sum(phase.counts.values())}; the energy and the composition disagree")

    violations, physics, unclassified = check_policy(read_incar(step / "INCAR"), expected)
    if violations:
        raise AdoptError("computed under a different policy: " + "; ".join(violations))

    initial = _volume(directory / "relax" / "POSCAR")
    fatal, suspect = check_geometry(outcome, initial)
    if fatal:
        raise AdoptError("; ".join(fatal))

    relax = read_job_directory(directory / "relax") if (
        directory / "relax" / "OUTCAR").is_file() else None

    phase.recipe_id = rid
    phase.e_dft = float(outcome.energy)
    phase.n_atoms = sum(phase.counts.values())
    phase.dft_converged = bool(outcome.converged)
    phase.dft_volume_initial = initial
    phase.dft_volume_final = (_volume(step / "CONTCAR") or _volume(step / "POSCAR"))
    phase.core_hours = round(
        (outcome.core_hours or 0.0) + (relax.core_hours if relax else 0.0), 3) or None
    phase.settings_hash = ""
    phase.state = "done"
    phase.fail_reason = ""
    phase.incar_physics = physics
    phase.adopted_from = str(directory)
    phase.warnings = list(phase.warnings) + suspect
    if not outcome.converged:
        phase.warnings.append(f"{ENERGY_STEP} did not report convergence")
    if unclassified:
        phase.warnings.append(
            "INCAR tags not classified as invariant, physics or robustness: "
            + ", ".join(unclassified[:8]))
    return phase


def expected_tags(recipe: Any, root: Path, step: str = ENERGY_STEP,
                  ) -> dict[str, tuple[str | None, str]]:
    """What each invariant tag should be, and on whose authority.

    Preferring the recipe wherever it speaks is what stops this from being a
    vote: if every directory in a tree were wrong in the same way, a pure
    consensus would bless it.  The consensus is the fallback for the tags the
    recipe deliberately does not state because resolution computes them, and
    there it answers the only question available -- "is this directory like its
    2,712 neighbours?"  A tree where a tag is genuinely split reports no
    expectation for it rather than picking the larger half, since at that point
    the tree cannot vouch for itself and refusing everything would be an
    accusation the evidence does not support.
    """
    stage = recipe.stage(step)
    out: dict[str, tuple[str | None, str]] = {}
    derived = []
    for tag in INVARIANT:
        if tag in stage.incar:
            out[tag] = (str(stage.incar[tag]), "the recipe")
        else:
            derived.append(tag)
    if not derived:
        return out

    seen: dict[str, dict[str, int]] = {t: {} for t in derived}
    total = 0
    for directory in sorted(Path(root).glob("mp-*")):
        incar_path = directory / step / "INCAR"
        if not incar_path.is_file():
            continue
        incar = read_incar(incar_path)
        total += 1
        for tag in derived:
            key = _norm(incar.get(tag))
            seen[tag][key] = seen[tag].get(key, 0) + 1
    for tag in derived:
        counts = seen[tag]
        if len(counts) == 1 and total:
            value = next(iter(counts))
            out[tag] = (None if value == "<unset>" else value,
                        f"all {total} directories")
        # split, or nothing read: no expectation, so the tag is not checked
    return out


def adopt(root: Path, *, rid: str, recipe: Any, refresh: bool = False,
          dry_run: bool = False, progress: Any = None) -> AdoptReport:
    """Walk a phase tree into the cache.  Writes nothing when `dry_run`."""
    root = Path(root)
    report = AdoptReport(root=str(root), recipe_id=rid)
    expected = expected_tags(recipe, root)

    # The tree's own exclusion list is carried INTO the cache, not merely obeyed
    # while walking.  A phase the operator dropped on purpose has to read as
    # "decided" to `coverage` too -- otherwise it is indistinguishable from one
    # nobody has run, and it blocks every hull in its chemistry forever.
    excluded: set[str] = set()
    exclusions = root / "EXCLUDED.json"
    if exclusions.is_file():
        raw = json.loads(exclusions.read_text()) or {}
        excluded = set(raw)
        if not dry_run and raw:
            from .computed import write_excluded

            write_excluded(rid, raw)

    physics_tally: dict[str, int] = {}
    for directory in sorted(root.glob("mp-*")):
        if not directory.is_dir():
            continue
        mp_id = directory.name
        if mp_id in excluded:
            report.excluded += 1
            continue
        if not refresh:
            existing = read_record(rid, mp_id)
            if existing is not None and existing.state == "done" and existing.e_dft is not None:
                report.skipped_existing += 1
                continue
        try:
            phase = read_phase(directory, rid=rid, expected=expected)
        except AdoptError as exc:
            report.refused.append((mp_id, str(exc)))
            continue
        except Exception as exc:                               # noqa: BLE001
            report.refused.append((mp_id, f"unreadable: {exc}"))
            continue
        for note in phase.warnings:
            report.suspect.append((mp_id, note))
        if not phase.dft_converged:
            report.unconverged += 1
        key = " ".join(f"{t}={phase.incar_physics.get(t, '?')}" for t in PHYSICS)
        physics_tally[key] = physics_tally.get(key, 0) + 1
        if not dry_run:
            write_record(phase)
        report.adopted += 1
        if progress is not None:
            progress(report.adopted, mp_id)

    report.physics = {f"{k}: {v} phase(s)": v for k, v in sorted(physics_tally.items())}
    return report
