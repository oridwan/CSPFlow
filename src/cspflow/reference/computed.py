"""The computed reference store: our own DFT and our own MLIP, per MP phase.

Stage 3 places candidates against MP's energies, which are not on our scale.
Measured 2026-08-27 on four MP structures through this project's own settings,
as a static at MP's own geometry:

    Fe (mp-13)  +0.2068 eV/atom     Sm2Fe17  +0.1870
    SmFe11Ti    +0.1794             SmFe2    +0.1530

A three-element least squares fits all four to within 1.8 meV/atom -- a clean
per-element offset.  It cancels exactly in a hull built on ONE scale and not at
all in one built on two, so every `dft_e_above_hull` is inflated by the
candidate's own share of it (D101).  The fix is to stop mixing: recompute the
reference phases ourselves, once, and build the hull entirely on our numbers.

That recomputation is expensive enough to be worth doing once for everyone and
cheap enough to be worth doing at all -- 2,746 phases over the nine-rare-earth
element set, 47,622 atoms, ~12,500 core-hours against the 120,714 the campaigns
have already spent.  So it lives outside every campaign, under
`$CSPFLOW_REFERENCE`, next to the MP download it is derived from -- and
pointedly not under `~/.cache`, because unlike that download it cannot be
fetched again.

**Why the key is a recipe id and not `settings_hash`.**  `ResolvedInputs.
settings_hash` is a hash over what will actually be computed *for one
structure*: it includes the resolved k-point grid, and MAGMOM, NBANDS and
LMAXMIX, all of which depend on the cell and its species.  Two phases computed
under identical policy therefore have different `settings_hash` values, so it
identifies a job and cannot identify a cache.  `recipe_id` hashes the policy
instead -- the recipe's INCAR templates, the k-point scheme, the POTCAR tree and
f-treatment, the magnetism and LDA+U settings -- which is exactly the thing that
has to match for two energies to belong on one hull.

Both are recorded.  `recipe_id` says the stored entry is usable; the per-record
`settings_hash` says what was actually written for that structure.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..chem import canonical_formula, chemsys as chemsys_of, parse_formula
from .hull import Entry
from .mp import reference_root


class ComputedError(Exception):
    """A reference set that would produce a plausible number on a mixed scale."""


# Where our own numbers live, under the same root as the MP download.
COMPUTED_DIRNAME = "computed"
# How much of the recipe id appears in a path. Full value is kept in the record.
ID_CHARS = 16


def computed_root() -> Path:
    return reference_root() / COMPUTED_DIRNAME


def recipe_id(dft: Any, recipe: Any) -> str:
    """A hash over the POLICY that decides an energy, not over one job's inputs.

    Everything that would make two energies incomparable goes in; nothing that
    varies with the structure does.  See the module docstring for why that
    distinction is the whole point.
    """
    payload = {
        "recipe": {
            "name": getattr(recipe, "name", ""),
            "stages": [
                {
                    "name": stage.name,
                    "incar": {k: _plain(v) for k, v in sorted(stage.incar.items())},
                    "kpoints": stage.kpoints.as_dict(),
                }
                for stage in recipe.stages
            ],
        },
        "potcar": _plain(dft.potcar),
        "rare_earth": _plain(dft.rare_earth),
        "magnetism": _plain(dft.magnetism),
        "ldau": _plain(dft.ldau),
        "nbands": _plain(dft.nbands),
        "incar_overrides": {k: _plain(v) for k, v in sorted(dft.incar_overrides.items())},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _plain(value: Any) -> Any:
    """Pydantic models, enums and paths, flattened so a hash is reproducible."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in sorted(value.items())}
    if hasattr(value, "value"):                    # an Enum
        return value.value
    return value


# --------------------------------------------------------------------------


@dataclass
class ComputedPhase:
    """One MP phase, recomputed.  Energies are TOTAL for the cell computed.

    **`counts` and `n_atoms` describe the CELL, not the reduced formula.**  MP
    reports `formula_pretty` reduced -- `mp-69` is "Sm1" and its structure has
    four atoms -- and pairing a four-atom total energy with a one-atom
    composition puts a point on the hull that is wrong by a factor of four.
    Nothing about it looks wrong: the hull still builds, and every
    `e_above_hull` in that system is quietly off.  `formula` keeps MP's reduced
    string for identification and is never used arithmetically.
    """

    mp_id: str
    formula: str                       # MP's reduced label, for humans only
    chemsys: str
    counts: dict[str, int]             # the CELL that was computed
    n_atoms: int                       # sum(counts), atoms in that cell
    recipe_id: str = ""

    # --- our DFT: relax then static, at our own settings ------------------
    # `e_dft` is the TOTAL energy of `n_atoms` atoms, from the final static.
    e_dft: float | None = None
    settings_hash: str = ""
    dft_volume_initial: float | None = None      # MP's cell
    dft_volume_final: float | None = None        # after our relaxation
    dft_converged: bool = False
    core_hours: float | None = None

    # --- our MLIP: both numbers, because they answer different questions ---
    # `e_mlip_static` is at MP's own geometry and is the parity number: it
    # isolates energy error from geometry error, which have opposite
    # consequences (a uniform energy offset largely cancels in a hull, a volume
    # bias does not).  `e_mlip_relaxed` is at the MLIP's own minimum and is the
    # one comparable with `e_dft`, because that is also a relaxed number.
    # Storing one and not the other loses a question you cannot ask later
    # without redoing the work.
    e_mlip_static: float | None = None
    e_mlip_relaxed: float | None = None
    mlip_volume_initial: float | None = None     # MP's cell
    mlip_volume_final: float | None = None       # after MLIP relaxation
    mlip_converged: bool = False
    mlip_steps: int = 0
    mlip_fmax: float | None = None
    mlip_model: str = ""
    # Set when the MLIP half was computed under a DIFFERENT recipe_id and reused
    # here.  That is legitimate -- MatterSim does not read the INCAR, so its
    # energy does not depend on the DFT policy the cache is keyed on -- but it
    # is the kind of reuse that must be visible rather than inferred.  It was
    # being written into the cache JSON while not existing as a field, so every
    # read through `_from_dict` silently dropped it.
    mlip_carried_from: str = ""
    # --- provenance ------------------------------------------------------
    mp_snapshot: str = ""
    # The tags that DO enter the energy, as actually run.  `recipe_id` says
    # what the policy was; this says what this phase got, and the two differ
    # wherever a retry had to change the physics to make a run finish at all.
    # Measured on the 2,713-phase adopted set: 83 statics ran `ISMEAR 0`
    # instead of `-5` and 12 ran `ISPIN 1` instead of `2`.  Recording it is
    # what makes the difference auditable later instead of a rediscovery.
    incar_physics: dict[str, str] = field(default_factory=dict)
    # Set when the record came from a directory of finished VASP output rather
    # than from a campaign database this pipeline drove.
    adopted_from: str = ""
    state: str = "pending"             # pending | done | failed
    fail_reason: str = ""
    computed_at: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def e_dft_per_atom(self) -> float | None:
        if self.e_dft is None or not self.n_atoms:
            return None
        return self.e_dft / self.n_atoms

    @property
    def e_mlip_relaxed_per_atom(self) -> float | None:
        if self.e_mlip_relaxed is None or not self.n_atoms:
            return None
        return self.e_mlip_relaxed / self.n_atoms

    @property
    def dft_volume_drift(self) -> float | None:
        """How far our relaxation moved MP's cell."""
        return _drift(self.dft_volume_initial, self.dft_volume_final)

    @property
    def mlip_volume_drift(self) -> float | None:
        """How far the MLIP's relaxation moved MP's cell.

        Kept as its own number and never folded into an energy MAE: an energy
        offset largely cancels along a hull tie-line, a volume bias does not.
        """
        return _drift(self.mlip_volume_initial, self.mlip_volume_final)

    @property
    def geometry_disagreement(self) -> float | None:
        """Relative volume difference between our relaxed cell and the MLIP's.

        Both relaxations start from MP's geometry, so they can land in different
        local minima -- and then `e_mlip_relaxed` and `e_dft` describe different
        structures rather than the same one computed two ways.  This is the
        cheap signal that it happened.
        """
        return _drift(self.dft_volume_final, self.mlip_volume_final)

    def to_entry(self) -> Entry:
        """A hull vertex on OUR scale.

        `scale='raw'` because a VASP total energy is uncorrected.

        `source='ours-reference'` rather than plain `'ours'`: the thing that
        must never be mixed is an MP *energy*, and this is not one -- but a
        candidate of ours that rediscovers a reference phase is a genuine
        double count, and `hull.find_duplicates` can only report it if the two
        carry different source labels.  It reports rather than refuses, which
        is right here: both entries are on one scale, so the hull is correct
        either way and the note is a diagnostic, not an error.

        `run_type` is left EMPTY, and that is deliberate.  A `ComputedPhase`
        record does not store which functional produced it -- `recipe_id` does,
        and the record stores that instead.  Inventing a value here would put a
        second `run_type` into a hull whose candidates already declare one, and
        `hull.assert_one_functional` refuses a hull that mixes two.  An empty
        `run_type` means "not recorded", which that function skips by design --
        so the functional is taken from the candidates, and the guarantee that
        the two agree comes from `recipe_id` matching, which is stronger than a
        string comparison would be.
        """
        if self.e_dft is None:
            raise ComputedError(
                f"{self.mp_id} has no computed energy (state={self.state!r}"
                f"{'; ' + self.fail_reason if self.fail_reason else ''}). "
                f"It cannot enter a hull built on our own scale."
            )
        total = sum(self.counts.values())
        if total != self.n_atoms:
            raise ComputedError(
                f"{self.mp_id}: n_atoms is {self.n_atoms} but its composition sums to "
                f"{total}. A total energy paired with the wrong composition is a hull "
                f"vertex wrong by exactly that ratio, and the hull still builds. Refusing."
            )
        return Entry(label=self.mp_id, counts=dict(self.counts), energy=self.e_dft,
                     scale="raw", source="ours-reference", run_type="")


# --------------------------------------------------------------------------


def _drift(before: float | None, after: float | None) -> float | None:
    if not before or after is None:
        return None
    return (after - before) / before


def record_dir(rid: str) -> Path:
    return computed_root() / rid[:ID_CHARS]


def record_path(rid: str, mp_id: str) -> Path:
    return record_dir(rid) / f"{mp_id}.json"


def write_record(phase: ComputedPhase) -> Path:
    if not phase.recipe_id:
        raise ComputedError("a computed phase must carry the recipe_id it was computed under")
    phase.computed_at = phase.computed_at or time.strftime("%Y-%m-%dT%H:%M:%S")
    path = record_path(phase.recipe_id, phase.mp_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(asdict(phase), indent=2, sort_keys=True))
    tmp.replace(path)
    return path


def _from_dict(payload: dict[str, Any]) -> ComputedPhase:
    """Tolerant of fields this version does not know.

    The store outlives the code that wrote it.  A record carrying a field that has
    since been renamed must not take down every campaign that reads the
    directory -- it should simply lose that field, which `coverage` then sees as
    work still to do.
    """
    known = {f for f in ComputedPhase.__dataclass_fields__}
    return ComputedPhase(**{k: v for k, v in payload.items() if k in known})


def read_record(rid: str, mp_id: str) -> ComputedPhase | None:
    path = record_path(rid, mp_id)
    if not path.is_file():
        return None
    return _from_dict(json.loads(path.read_text()))


def load_all(rid: str) -> dict[str, ComputedPhase]:
    out: dict[str, ComputedPhase] = {}
    for path in sorted(record_dir(rid).glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            phase = _from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError):      # pragma: no cover
            continue
        out[phase.mp_id] = phase
    return out


def write_manifest(rid: str, dft: Any, recipe: Any, *, note: str = "") -> Path:
    """What the id stands for, in full, so the hash is auditable and not opaque."""
    path = record_dir(rid) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "recipe_id": rid,
        "note": note,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "recipe": {"name": getattr(recipe, "name", ""),
                   "stages": [{"name": s.name, "incar": {k: _plain(v) for k, v in s.incar.items()},
                               "kpoints": s.kpoints.as_dict(), "resources": s.resources}
                              for s in recipe.stages]},
        "dft": {"potcar": _plain(dft.potcar), "rare_earth": _plain(dft.rare_earth),
                "magnetism": _plain(dft.magnetism), "ldau": _plain(dft.ldau),
                "nbands": _plain(dft.nbands),
                "incar_overrides": _plain(dft.incar_overrides)},
    }, indent=2, sort_keys=True))
    return path


# --------------------------------------------------------------------------


@dataclass
class Coverage:
    """What the cache holds for one chemical system, and what it is missing."""

    chemsys: str
    recipe_id: str
    wanted: list[str] = field(default_factory=list)
    done: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    # Dropped on purpose, and therefore NOT in `wanted`.  Counted separately so
    # a system reads as "complete, 3 excluded" rather than simply complete: an
    # exclusion is a judgement about the hull, and it should stay visible to
    # whoever reads the number that judgement made possible.
    excluded: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.wanted) and not self.missing and not self.failed

    def render(self) -> str:
        return (f"{self.chemsys}: {len(self.done)}/{len(self.wanted)} computed"
                + (f", {len(self.failed)} failed" if self.failed else "")
                + (f", {len(self.missing)} missing" if self.missing else "")
                + (f", {len(self.excluded)} excluded" if self.excluded else ""))


EXCLUDED_FILE = "excluded.json"


def excluded_path(rid: str) -> Path:
    return record_dir(rid) / EXCLUDED_FILE


def load_excluded(rid: str) -> dict[str, str]:
    """Phases deliberately not computed, as `{mp_id: why}`.

    Without this, "we decided this phase cannot matter" and "nobody has run this
    phase yet" are the same state to `coverage`, and the first one blocks a hull
    forever.  Measured on this project: 33 phases, blocking 262 of 434 chemical
    systems -- 31 elemental polymorphs abandoned under the rule in
    VASP_FAILURES.md Part 4, plus the two in that tree's own `EXCLUDED.json`.

    **The rule that makes this safe, and its one exception.**  A structure can
    only move a hull if it is the sole structure at its composition, or lower
    than whatever else sits there.  An unconverged relaxation gives an UPPER
    bound on the true energy, so a phase whose upper bound already sits well
    above a converged neighbour at the same composition cannot reach the hull
    however it finishes.  A phase that is the ONLY one at its composition can
    never be excluded on that argument, whatever its energy -- and that is the
    check to make before adding an entry here, not after.

    A missing or unreadable file means nothing is excluded, which is the safe
    direction: it makes `coverage` refuse rather than silently accept.
    """
    path = excluded_path(rid)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):        # pragma: no cover
        return {}
    out: dict[str, str] = {}
    for mp_id, why in (raw or {}).items():
        if isinstance(why, dict):                  # the richer EXCLUDED.json shape
            why = why.get("why") or why.get("decision") or "excluded"
        out[str(mp_id)] = str(why)
    return out


def write_excluded(rid: str, entries: dict[str, Any], *, merge: bool = True) -> Path:
    """Record deliberate exclusions, keeping whatever reasons are already there."""
    path = excluded_path(rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, Any] = {}
    if merge and path.is_file():
        try:
            current = json.loads(path.read_text()) or {}
        except (json.JSONDecodeError, OSError):    # pragma: no cover
            current = {}
    current.update(entries)
    path.write_text(json.dumps(current, indent=2, sort_keys=True))
    return path


def coverage(chemsys: str, rid: str, *, thermo_type: str = "GGA_GGA+U") -> Coverage:
    """Which of a system's MP phases we have our own energy for.

    The wanted set comes from the MP download, so `coverage` answers the
    question a campaign actually asks -- "can this hull be built on our scale?"
    -- rather than "how many files are in a directory".
    """
    from .mp import fetch_chemsys

    result = fetch_chemsys(chemsys, thermo_type=thermo_type, energy_scale="raw")
    have = load_all(rid)
    dropped = load_excluded(rid)
    cov = Coverage(
        chemsys=chemsys, recipe_id=rid,
        wanted=[e.mp_id for e in result.entries if e.mp_id not in dropped],
        excluded=[e.mp_id for e in result.entries if e.mp_id in dropped])
    for mp_id in cov.wanted:
        phase = have.get(mp_id)
        if phase is None or phase.state == "pending":
            cov.missing.append(mp_id)
        elif phase.state == "failed" or phase.e_dft is None:
            cov.failed.append(mp_id)
        else:
            cov.done.append(mp_id)
    return cov


def entries_for(chemsys: str, rid: str, *, thermo_type: str = "GGA_GGA+U",
                allow_partial: bool = False) -> list[Entry]:
    """Hull vertices for one system on OUR energies, or a refusal.

    Partial coverage is refused by default, and that refusal is the whole point
    of this module.  A hull missing one recomputed phase falls back to MP's
    number for that vertex, which is the exact mixed-scale condition D101
    describes -- and it produces a plausible hull rather than an error, so
    nothing downstream can catch it.
    """
    cov = coverage(chemsys, rid, thermo_type=thermo_type)
    if not cov.complete and not allow_partial:
        unresolved = cov.missing + cov.failed
        raise ComputedError(
            f"{chemsys}: {len(cov.done)} of {len(cov.wanted)} reference phases have "
            f"an energy at recipe {rid[:12]}; {len(unresolved)} do not "
            f"({', '.join(unresolved[:6])}{' ...' if len(unresolved) > 6 else ''}). "
            f"Refusing rather than filling the gap from MP: a hull with one borrowed "
            f"vertex still builds and looks correct, and is wrong by the scale offset "
            f"in D101. Either finish the reference set --\n"
            f"    csp reference status {chemsys}      # what is missing\n"
            f"    csp reference build <dir> {chemsys} # compute it\n"
            f"    csp reference export -c <dir>/campaign.yaml\n"
            f"-- or accept MP's scale deliberately with `reference.mode: mp_energies`, "
            f"which reports the mixing instead of hiding it."
        )
    have = load_all(rid)
    return [have[mp_id].to_entry() for mp_id in cov.done]


def phases_needed(systems: Iterable[str], *, thermo_type: str = "GGA_GGA+U",
                  ) -> dict[str, dict[str, Any]]:
    """Every distinct MP phase across a set of systems, deduplicated by mp_id.

    One VASP calculation per phase serves every hull the phase appears in, so
    this -- not the sum over systems -- is what the build actually costs.
    """
    from .mp import fetch_chemsys

    out: dict[str, dict[str, Any]] = {}
    for chemsys in systems:
        for entry in fetch_chemsys(chemsys, thermo_type=thermo_type,
                                   energy_scale="raw").entries:
            out.setdefault(entry.mp_id, {
                "mp_id": entry.mp_id, "formula": entry.formula,
                "chemsys": entry.chemsys, "counts": dict(entry.counts),
                "n_atoms": entry.n_atoms, "e_above_hull_mp": entry.e_above_hull_mp,
            })
    return out


# --------------------------------------------------------------------------
# Deliberate exclusions
# --------------------------------------------------------------------------


@dataclass
class ExclusionProposal:
    """Phases a hull cannot use, and the ones that must be computed anyway."""

    droppable: dict[str, dict[str, Any]] = field(default_factory=dict)
    # mp_id -> why it CANNOT be dropped.  The important half: a proposal that
    # only listed what may go would quietly turn a refusal into an omission.
    keep: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"{len(self.droppable)} phase(s) can be excluded, "
                 f"{len(self.keep)} must be computed"]
        by_formula: dict[str, int] = {}
        for entry in self.droppable.values():
            by_formula[entry["reduced"]] = by_formula.get(entry["reduced"], 0) + 1
        for formula, n in sorted(by_formula.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {formula:12s} {n:3d}")
        for mp_id, why in sorted(self.keep.items()):
            lines.append(f"  MUST COMPUTE {mp_id}: {why}")
        return "\n".join(lines)


def propose_exclusions(systems: Iterable[str], rid: str, *,
                       thermo_type: str = "GGA_GGA+U") -> ExclusionProposal:
    """Which uncomputed phases cannot change any hull, by Part 4's own rule.

    From VASP_FAILURES.md Part 4: a structure only matters to a convex hull if
    it is the sole structure at its composition, or lower in energy than
    whatever else sits there.

    **The sole-entry case is a hard refusal, not a judgement.**  If nothing else
    in the set shares a phase's composition, dropping it deletes a vertex the
    hull needs, and no argument about its likely energy can recover that -- the
    hull's boundary in that direction would then be set by whichever compound
    happens to be lowest, with no visible symptom.  Those come back under
    `keep`, and the caller is expected to compute them.

    **What this does NOT establish.**  Where a converged sibling exists, this
    says only that the composition is represented -- not that the uncomputed
    phase would have landed above it.  Part 4 checked that too, against each
    unconverged run's last energy as an upper bound.  Phases that never produced
    an energy at all have no upper bound to check, so for those the sibling test
    is the whole of the evidence and the rest is the operator's judgement.  The
    reason string records which of the two applies, per phase, rather than
    letting them read alike later.
    """
    from ..chem import reduce_counts
    from .mp import fetch_chemsys

    have = {k: v for k, v in load_all(rid).items()
            if v.state == "done" and v.e_dft is not None}

    def _reduced(counts: dict[str, int]) -> str:
        from ..chem import canonical_formula

        return canonical_formula(reduce_counts(counts)[0])

    # best converged energy per reduced composition, and how many we hold
    best: dict[str, tuple[float, str]] = {}
    census: dict[str, int] = {}
    for mp_id, phase in have.items():
        key = _reduced(phase.counts)
        census[key] = census.get(key, 0) + 1
        epa = phase.e_dft / phase.n_atoms
        if key not in best or epa < best[key][0]:
            best[key] = (epa, mp_id)

    gap: dict[str, Any] = {}
    for chemsys in sorted(set(systems)):
        try:
            entries = fetch_chemsys(chemsys, thermo_type=thermo_type,
                                    energy_scale="raw").entries
        except Exception:                                     # noqa: BLE001
            continue
        for entry in entries:
            if entry.mp_id not in have:
                gap[entry.mp_id] = entry

    out = ExclusionProposal()
    for mp_id, entry in sorted(gap.items()):
        key = _reduced(dict(entry.counts))
        n = census.get(key, 0)
        if n == 0:
            out.keep[mp_id] = (
                f"{entry.formula}: nothing else in the reference set sits at "
                f"{key}. Dropping it would delete a hull vertex outright.")
            continue
        epa, best_id = best[key]
        out.droppable[mp_id] = {
            "formula": entry.formula,
            "reduced": key,
            "decision": "excluded: cannot change any hull",
            "why": (f"no energy was produced for this phase. {n} converged "
                    f"structure(s) already sit at {key}, the lowest being "
                    f"{best_id} at {epa:.4f} eV/atom, so the composition is "
                    f"represented and the hull point does not move. Note this "
                    f"phase has no upper bound of its own -- the sibling is the "
                    f"whole of the evidence (VASP_FAILURES.md Part 4)."),
            "best_sibling": best_id,
            "best_sibling_e_per_atom": round(epa, 4),
            "n_converged_siblings": n,
        }
    return out
