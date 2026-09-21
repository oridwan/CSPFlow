"""Build the computed reference cache: one campaign, then an export.

The expensive half of `computed.py` is DFT, and cspflow already knows how to
submit, throttle, retry and reconcile DFT.  So a reference build is not a new
execution path -- it is an ordinary campaign whose structures happen to be MP's
own phases:

    source     structure_list over the cached MP geometries, relax: false
    generate   nothing to generate
    screen     nothing to screen; the seeds go straight in
    dedup      off; two MP entries for one composition are polymorphs, not
               duplicates, and dropping one removes a real hull vertex
    reference  off; this campaign IS the reference
    calibrate  pilot gate off; the gate exists to protect candidate DFT from an
               unchecked MLIP, and there is no MLIP in this path
    dft        the campaign recipe, unchanged -- that is the entire point
    analyze    energies out

Every structure is seeded in state `selected`, which is where `dft_stage` claims
from, so no filter has to be configured to pass everything through.

The export then reads the finished campaign and writes one record per phase into
the shared cache, keyed on `recipe_id`.  The campaign database stays as the
audit trail; the cache is the part other campaigns read.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..chem import parse_formula
from .computed import (
    ComputedPhase, ComputedError, phases_needed, recipe_id, write_manifest,
    write_record, load_all,
)
from .mp import fetch_chemsys, fetch_structures, reference_root

# Measured on 11,644 real DFT jobs from the four adopted families: median
# core-hours against cell size, at this project's `magnets` recipe on Orion.
# Used only for the estimate `plan` prints -- nothing is scheduled from it.
COST_CURVE: list[tuple[int, float]] = [
    (10, 2.55), (12, 2.15), (14, 3.58), (16, 5.83), (18, 5.88), (20, 7.85),
    (24, 5.52), (28, 7.55), (32, 8.26), (36, 12.14), (40, 11.73), (52, 12.77),
    (60, 12.86), (76, 15.34),
]


def cost_estimate(n_atoms: int) -> float:
    """Core-hours for a relax+static on a cell this size, from measured data."""
    xs = [n for n, _ in COST_CURVE]
    if n_atoms <= xs[0]:
        return COST_CURVE[0][1] * n_atoms / xs[0]
    if n_atoms >= xs[-1]:
        return COST_CURVE[-1][1] * n_atoms / xs[-1]
    for (x0, y0), (x1, y1) in zip(COST_CURVE, COST_CURVE[1:]):
        if x0 <= n_atoms <= x1:
            return y0 + (y1 - y0) * (n_atoms - x0) / (x1 - x0)
    return COST_CURVE[-1][1]                                  # pragma: no cover


@dataclass
class Plan:
    """What a build would compute, before anything is submitted."""

    systems: list[str] = field(default_factory=list)
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
    already: list[str] = field(default_factory=list)
    recipe_id: str = ""
    dropped: dict[str, int] = field(default_factory=dict)

    @property
    def todo(self) -> list[dict[str, Any]]:
        done = set(self.already)
        return [p for mp_id, p in sorted(self.phases.items()) if mp_id not in done]

    @property
    def core_hours(self) -> float:
        return sum(cost_estimate(p["n_atoms"]) for p in self.todo)

    def render(self) -> str:
        sizes = [p["n_atoms"] for p in self.todo] or [0]
        lines = [
            f"recipe {self.recipe_id[:16]}",
            f"{len(self.systems)} chemical system(s) -> {len(self.phases)} distinct MP phases",
        ]
        if self.already:
            lines.append(f"{len(self.already)} already computed, {len(self.todo)} to do")
        lines += [
            f"{sum(sizes):,} atoms, median cell {statistics.median(sizes):.0f}, "
            f"largest {max(sizes)}",
            f"estimated {self.core_hours:,.0f} core-hours (relax + static, "
            f"from 11,644 measured jobs)",
        ]
        for reason, n in sorted(self.dropped.items()):
            lines.append(f"  excluded {n:,} phases: {reason}")
        return "\n".join(lines)


def plan(systems: Iterable[str], *, dft: Any, recipe: Any,
         thermo_type: str = "GGA_GGA+U", min_atoms: int | None = None,
         max_atoms: int | None = None,
         max_e_above_hull: float | None = None) -> Plan:
    """What a reference build would compute, after the cuts that make it affordable.

    The three filters exist for different reasons and are not interchangeable:

    *   ``max_e_above_hull`` is the only one that removes phases on physics.  A
        phase far above MP's own hull cannot become a vertex on ours either:
        D101 measured our-minus-MP offsets of 0.15-0.21 eV/atom that a
        per-element model fits to 1.8 meV/atom, so relative positions move by
        single meV.  0.2 eV/atom is two orders of magnitude of headroom.
    *   ``max_atoms`` is a cost cut, and an honest one only because the measured
        cost curve is linear and stops at 76 atoms while DFT is O(N^3).  The 61
        phases above 80 atoms are ~1,400 core-hours linear and ~3,800 cubic; the
        two largest are 240- and 232-atom C and Si allotropes that cannot be
        hull vertices at any cutoff.
    *   ``min_atoms`` removes nothing.  It exists to SPLIT a build into size
        tiers that get different `ntasks`, because resources are set once per
        array job and 32 MPI ranks on a 1-atom cell is mostly overhead.
        Resources are not in `recipe_id`, so the tiers stay one cache.
    """
    systems = sorted(set(systems))
    rid = recipe_id(dft, recipe)
    have = {mp_id for mp_id, p in load_all(rid).items()
            if p.state == "done" and p.e_dft is not None}
    phases = phases_needed(systems, thermo_type=thermo_type)
    _use_cell_sizes(systems, phases)

    dropped: dict[str, int] = {}
    def _cut(reason: str, keep) -> None:
        gone = [k for k, v in phases.items() if not keep(v)]
        for k in gone:
            del phases[k]
        if gone:
            dropped[reason] = len(gone)

    if max_e_above_hull is not None:
        # A missing e_above_hull is KEPT.  Dropping on absent data would silently
        # thin the hull wherever MP has no value, which is the opposite of safe.
        _cut(f"e_above_hull > {max_e_above_hull}",
             lambda v: v.get("e_above_hull_mp") is None
             or v["e_above_hull_mp"] <= max_e_above_hull)
    if max_atoms is not None:
        _cut(f"more than {max_atoms} atoms", lambda v: v["n_atoms"] <= max_atoms)
    if min_atoms is not None:
        _cut(f"fewer than {min_atoms} atoms", lambda v: v["n_atoms"] >= min_atoms)

    return Plan(systems=systems, phases=phases, recipe_id=rid,
                already=sorted(have & set(phases)), dropped=dropped)


def _use_cell_sizes(systems: Iterable[str], phases: dict[str, dict[str, Any]]) -> None:
    """Replace reduced-formula atom counts with the cell that will be computed.

    `formula_pretty` is reduced, so `mp-69` reads "Sm1" and its structure has
    four atoms.  Estimating cost from the reduced count understates every
    elemental phase, which is a third of the set.  Silent if the geometries are
    not cached: an estimate off by a factor is better than a refusal here.
    """
    for chemsys in sorted(set(systems)):
        try:
            structures = fetch_structures(chemsys)
        except Exception:                                     # noqa: BLE001
            continue
        for mp_id, structure in structures.items():
            if mp_id in phases:
                phases[mp_id]["n_atoms"] = len(structure)


# --------------------------------------------------------------------------


CAMPAIGN_YAML = """\
# Reference build -- MP's own phases, recomputed at this project's DFT settings.
#
# Generated by `csp reference build`.  The point of this campaign is that its
# `dft:` block is IDENTICAL to the campaigns whose hulls will use its output;
# change it here and the recipe id changes, and the cache it writes no longer
# applies to those campaigns.
#
# recipe_id: {recipe_id}
name: {name}
machine: {machine}
workdir: {workdir}

source:
  - mode: structure_list
    name: mp_reference
    structure_list:
      paths: ["{cifs}/*.cif"]
      relax: false          # MP geometries are the seed; our DFT does the relaxing
      dedup: warn           # two MP entries at one composition are polymorphs
      max_atoms: {max_atoms}

screen:
  mlip: mattersim

reference:
  # This campaign IS the reference set. It must not fetch one.
  mode: mp_energies
  prescreen_hull_max: 100.0

calibrate:
  mp:    {{on_fail: "off"}}
  pilot: {{on_fail: "off"}}   # no MLIP in this path, so nothing to gate on

filter:
  e_above_hull_max: 100.0
  e_above_hull_max_source: literal
  max_per_composition: 1000

dft:
{dft_block}

analyze:
  properties: [m_dft_raw, volume, spacegroup]
  report: html
"""


def write_campaign(dest: Path, *, systems: Iterable[str], dft: Any, recipe: Any,
                   machine: str = "orion", thermo_type: str = "GGA_GGA+U",
                   name: str = "mp-reference", min_atoms: int | None = None,
                   max_atoms: int | None = None,
                   max_e_above_hull: float | None = None) -> tuple[Path, Plan]:
    """Write the campaign directory and its CIF seeds.  Submits nothing."""
    import yaml

    the_plan = plan(systems, dft=dft, recipe=recipe, thermo_type=thermo_type,
                    min_atoms=min_atoms, max_atoms=max_atoms,
                    max_e_above_hull=max_e_above_hull)
    dest = Path(dest)
    cifs = dest / "inputs"
    cifs.mkdir(parents=True, exist_ok=True)

    wanted = {p["mp_id"] for p in the_plan.todo}
    written, missing = 0, []
    for chemsys in the_plan.systems:
        if not wanted:
            break
        try:
            structures = fetch_structures(chemsys)
        except Exception as exc:                              # pragma: no cover
            missing.append(f"{chemsys}: {exc}")
            continue
        for mp_id, structure in structures.items():
            if mp_id not in wanted:
                continue
            (cifs / f"{mp_id}.cif").write_text(structure.to(fmt="cif"))
            wanted.discard(mp_id)
            written += 1
    if wanted:
        missing.append(f"{len(wanted)} phase(s) have thermo data but no structure: "
                       f"{', '.join(sorted(wanted)[:8])}")

    # NOT the `max_atoms` filter argument: this is the campaign's own
    # `source.max_atoms` guard, which must clear the largest cell that survived
    # the cuts or the source stage would reject phases the plan just selected.
    cell_cap = max([p["n_atoms"] for p in the_plan.todo] or [40])
    dft_yaml = yaml.safe_dump(_dft_mapping(dft), sort_keys=False, default_flow_style=False)
    body = CAMPAIGN_YAML.format(
        recipe_id=the_plan.recipe_id, name=name, machine=machine, workdir=dest,
        cifs=cifs, max_atoms=cell_cap + 1,
        dft_block="\n".join(f"  {line}" for line in dft_yaml.rstrip().splitlines()),
    )
    (dest / "campaign.yaml").write_text(body)
    write_manifest(the_plan.recipe_id, dft, recipe,
                   note=f"reference build at {dest}")
    (dest / "phases.json").write_text(json.dumps(the_plan.phases, indent=2, sort_keys=True))
    if missing:
        (dest / "missing.txt").write_text("\n".join(missing) + "\n")
    return dest, the_plan


def _dft_mapping(dft: Any) -> dict[str, Any]:
    """The `dft:` block, with the two knobs a reference build must pin.

    `select.max_total` has to clear the phase count or the stage caps itself
    partway through and the cache is silently incomplete -- which is the one
    failure `computed.entries_for` cannot distinguish from work still running.
    """
    data = dft.model_dump(mode="json", exclude_defaults=False)
    data.setdefault("select", {})
    data["select"]["max_total"] = 100_000
    data["select"]["max_per_composition"] = 1000
    data["select"]["rank_by"] = "n_atoms"          # cheapest first, not by a hull
    return data


# --------------------------------------------------------------------------


def seed(store: Any, dest: Path, *, plan_: Plan) -> int:
    """Put every phase into the store already `selected`, stamped with its mp_id.

    Seeded directly rather than through `csp source` because the mp_id is the
    cache key and has to survive onto the structure row; a CIF path does not
    carry it anywhere the DFT stage would preserve.
    """
    from ase.io import read as ase_read

    from ..db.store import Origin, StructureState

    added = 0
    for phase in plan_.todo:
        path = Path(dest) / "inputs" / f"{phase['mp_id']}.cif"
        if not path.is_file():
            continue
        atoms = ase_read(str(path))
        # `formula`, `natoms` and every element symbol are ASE row attributes;
        # writing one raises from inside ASE. The mp_ prefix keeps them ours.
        store.add_structure(
            atoms, origin=Origin.seed if hasattr(Origin, "seed") else "seed",
            state=StructureState.selected,
            mp_id=phase["mp_id"], mp_formula=phase["formula"],
            chemsys=phase["chemsys"], mp_natoms=int(phase["n_atoms"]),
            # MP's own cell volume, kept so the export can report how far our
            # relaxation moved it. Not recoverable afterwards: the row's
            # `volume` is overwritten with the relaxed value.
            mp_volume=float(atoms.get_volume()),
            reference_build=1,
        )
        added += 1
    return added


def export(store: Any, *, recipe_id_: str, mp_snapshots: dict[str, str] | None = None) -> dict[str, int]:
    """Finished campaign -> one cache record per phase.

    A structure with no `vasp_energy` is written as `failed` rather than left
    absent: absent means "not attempted yet" to `coverage`, and a phase that
    diverged is not the same as a phase nobody has run.
    """
    counts = {"done": 0, "failed": 0, "skipped": 0}
    for row in store.structures():
        kv = row.key_value_pairs
        mp_id = kv.get("mp_id")
        if not mp_id:
            counts["skipped"] += 1
            continue
        try:
            elements = parse_formula(str(kv.get("mp_formula") or row.formula))
        except Exception:                                     # pragma: no cover
            counts["skipped"] += 1
            continue
        energy = kv.get("vasp_energy")
        # The composition VASP actually ran, from the structure's own species
        # list rather than from MP's reduced formula -- see ComputedPhase.
        cell = _counts_from_numbers(row.numbers) or elements
        phase = ComputedPhase(
            mp_id=str(mp_id), formula=str(kv.get("mp_formula") or row.formula),
            chemsys=str(kv.get("chemsys") or ""), counts=cell,
            n_atoms=sum(cell.values()),
            recipe_id=recipe_id_,
            e_dft=float(energy) if energy is not None else None,
            settings_hash=str(kv.get("settings_hash") or ""),
            dft_volume_initial=float(kv["mp_volume"]) if kv.get("mp_volume") else None,
            dft_volume_final=float(kv["volume"]) if kv.get("volume") else None,
            dft_converged=str(kv.get("state") or "") == "dft_done",
            mp_snapshot=(mp_snapshots or {}).get(str(kv.get("chemsys") or ""), ""),
            state="done" if energy is not None else "failed",
            fail_reason="" if energy is not None else str(kv.get("state") or "no vasp_energy"),
        )
        write_record(phase)
        counts["done" if phase.state == "done" else "failed"] += 1
    return counts


def structure_dir() -> Path | None:
    """Per-structure artifact folder: `<dir>/<mp_id>/{mlip,relax,static}`.

    The MLIP relaxation produces a geometry that was being thrown away -- only
    its energy and volume were kept -- even though it is the best available
    starting point for the DFT relaxation and the only record of what the MLIP
    actually did to the cell.  Writing it next to `relax/` and `static/` keeps
    one folder per structure telling the whole story, which is the layout
    `adopt.py` already expects from the legacy runs.

    Set `CSPFLOW_STRUCTURE_DIR` to choose the root; defaults to
    `$CSPFLOW_REFERENCE/structures`.  Set it empty to disable writing.
    """
    raw = os.environ.get("CSPFLOW_STRUCTURE_DIR")
    if raw is not None and not raw.strip():
        return None
    return Path(raw) if raw else reference_root() / "structures"


def write_mlip_geometry(mp_id: str, atoms: Any, phase: Any) -> Path | None:
    """Save the MLIP-relaxed cell as `<dir>/<mp_id>/mlip/CONTCAR` (+ mlip.json)."""
    root = structure_dir()
    if root is None or atoms is None:
        return None
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp.inputs import Poscar
    out = root / mp_id / "mlip"
    out.mkdir(parents=True, exist_ok=True)
    struct = AseAtomsAdaptor.get_structure(atoms)
    Poscar(struct).write_file(out / "CONTCAR")
    struct.to(filename=str(out / "relaxed.cif"))
    (out / "mlip.json").write_text(json.dumps({
        "mp_id": mp_id, "model": phase.mlip_model,
        "e_mlip_static": phase.e_mlip_static,
        "e_mlip_relaxed": phase.e_mlip_relaxed,
        "mlip_converged": phase.mlip_converged, "mlip_steps": phase.mlip_steps,
        "mlip_fmax": phase.mlip_fmax,
        "volume_initial": phase.mlip_volume_initial,
        "volume_final": phase.mlip_volume_final,
        "recipe_id": phase.recipe_id,
    }, indent=2, sort_keys=True))
    return out


def mlip_pass(systems: Iterable[str], *, recipe_id_: str, engine: Any,
              relax: bool = True, thermo_type: str = "GGA_GGA+U",
              refresh: bool = False, progress: Any = None) -> dict[str, int]:
    """MatterSim over every cached MP geometry, written into the same records.

    Costs GPU-seconds and no scheduler, so it runs before any DFT and answers
    the question that decides whether the DFT is worth spending: can this MLIP
    reproduce MP's own ordering in this chemistry at all?  On the four adopted
    families it could not for Gd -- 9 of 11 systems failed, Spearman 0.14 on
    Fe-Gd over 330 points -- while Sm and Tb passed.  That is an element-level
    property, and this is where it becomes visible before the money is spent.

    A record is created in state `pending` if DFT has not run yet: the MLIP half
    and the DFT half of a phase are filled in independently and in either order.
    """
    phases = phases_needed(systems, thermo_type=thermo_type)
    have = load_all(recipe_id_)
    model = getattr(engine, "model", "") or getattr(engine, "name", "")
    counts = {"static": 0, "relaxed": 0, "unconverged": 0, "failed": 0,
              "no_structure": 0}
    seen: set[str] = set()             # mp_ids handled in THIS run, see below

    for chemsys in sorted(set(systems)):
        try:
            structures = fetch_structures(chemsys)
        except Exception as exc:                              # noqa: BLE001
            if progress:
                progress(f"{chemsys}: {exc}")
            continue
        for mp_id, structure in structures.items():
            meta = phases.get(mp_id)
            if meta is None:
                continue
            if mp_id in seen:
                # A phase belongs to every system it is a sub-system of:
                # elemental Co is in all 144 Co-bearing systems here, and the
                # 432 systems name 21,889 entries that are only 2,746 distinct
                # materials. Deduplicating is an 8x saving on average, and this
                # guard has to sit outside the `refresh` test -- otherwise
                # --refresh recomputes Co 144 times rather than once.
                continue
            seen.add(mp_id)
            if not refresh and mp_id in have and have[mp_id].e_mlip_static is not None:
                continue                                      # already done
            atoms = _to_atoms(structure)
            # From the cell, never from MP's reduced formula -- see ComputedPhase.
            cell = _cell_counts(atoms)
            phase = have.get(mp_id) or ComputedPhase(
                mp_id=mp_id, formula=meta["formula"], chemsys=meta["chemsys"],
                counts=cell, n_atoms=sum(cell.values()),
                recipe_id=recipe_id_, state="pending",
            )
            phase.counts, phase.n_atoms = cell, sum(cell.values())

            # The single point is the parity number, at MP's own geometry: it
            # separates energy error from geometry error. Cheap, so it is taken
            # even when the relaxation is what we are really after.
            single = engine.single_point(atoms)
            if single.error or single.energy is None:
                counts["failed"] += 1
                phase.warnings.append(f"mlip single point: {single.error or 'no energy'}")
            else:
                phase.e_mlip_static = float(single.energy)
                counts["static"] += 1

            # The relaxation is the hull number: `e_dft` is a relaxed energy, so
            # only a relaxed MLIP energy is on the same footing.
            if relax:
                moved = engine.relax(atoms)
                if moved.error or moved.energy is None:
                    counts["failed"] += 1
                    phase.warnings.append(f"mlip relax: {moved.error or 'no energy'}")
                else:
                    phase.e_mlip_relaxed = float(moved.energy)
                    phase.mlip_volume_initial = moved.volume_before
                    phase.mlip_volume_final = moved.volume_after
                    phase.mlip_converged = bool(moved.converged)
                    phase.mlip_steps = int(moved.n_steps)
                    phase.mlip_fmax = moved.fmax
                    counts["relaxed"] += 1
                    if not moved.converged:
                        # Not a failure: `ok` and `converged` are separate for a
                        # reason. But an energy from a structure that stopped at
                        # the step limit is not a minimum, and a hull vertex has
                        # to be one.
                        counts["unconverged"] += 1
                        phase.warnings.append(
                            f"mlip relax stopped after {moved.n_steps} steps at "
                            f"fmax {moved.fmax}, not at the force criterion")
                    phase.mlip_model = str(model)
                    write_mlip_geometry(mp_id, moved.atoms, phase)
            phase.mlip_model = str(model)
            have[mp_id] = phase
            write_record(phase)
        if progress:
            progress(f"{chemsys}: {counts['static']} static, {counts['relaxed']} relaxed")

    counts["no_structure"] = len(set(phases) - set(have))
    return counts


def _to_atoms(structure):
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(structure)


def _cell_counts(atoms) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _counts_from_numbers(numbers) -> dict[str, int]:
    """Composition of a stored row, from its atomic numbers."""
    from ase.data import chemical_symbols

    counts: dict[str, int] = {}
    for z in numbers or []:
        symbol = chemical_symbols[int(z)]
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts
