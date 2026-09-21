"""The candidate table, and the funnel that produced it.

"Analysis, seeing results is difficult" was the complaint this answers, so the
shape of the answer matters: one table, every column named for exactly what it
is, and a funnel above it that says how many structures each gate removed.

The funnel is read from `filter_event`, not reconstructed by differencing state
counts.  Differencing looks equivalent and is not: a structure can leave a state
for more than one reason, and the difference then attributes all of them to the
last gate that ran.  The events say which gate rejected what.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..db.store import Store
from .magnetics import summarise

# The columns, in the order they are written, with a one-line meaning each.
# Two of these are the same physical quantity computed two ways and they are
# adjacent on purpose -- a reader who sees only one of them has been misled.
COLUMNS: list[tuple[str, str]] = [
    ("id", "campaign structure id"),
    ("formula", "reduced formula"),
    ("n_atoms", "atoms in the cell"),
    ("spacegroup", "spacegroup number of the relaxed cell"),
    ("spacegroup_symbol", "at the tolerance in `symprec`"),
    ("symprec", "symmetry tolerance the spacegroup was found at"),
    ("e_above_hull_mlip", "eV/atom, MatterSim energies against MP"),
    ("dft_e_above_hull", "eV/atom, our DFT against MP, same scale"),
    ("dft_e_formation", "eV/atom"),
    ("volume", "A^3, relaxed cell"),
    ("volume_per_atom", "A^3"),
    ("m_dft_raw", "mu_B per cell, computed: the cell magnetisation"),
    ("m_s_reconstructed", "mu_B per cell, MODELLED: TM sublattice + Hund's-rule 4f"),
    ("f_treatment", "which of the two above is meaningful"),
    ("m_per_formula_unit", "mu_B, from m_dft_raw"),
    ("m_per_volume", "mu_B / A^3, from m_dft_raw"),
    ("mu0_m_tesla", "T, saturation polarisation mu0*M, from m_dft_raw / volume"),
    ("m_emu_per_cc", "emu / cm^3, the same number in CGS"),
    ("state", "where the structure stopped"),
    # Last, and adjacent to `state`, because it qualifies every number above it:
    # a structure whose relax and static settled in different magnetic states has
    # a geometry from one and an energy from the other. Empty for a clean run.
    ("dft_warning", "why this row should not be read at face value"),
]

# What each funnel gate means, in the order they run.
GATE_ORDER = ["screen:validate", "screen:converged", "dedup", "dedup:seed_collision",
              "filter:e_above_hull", "filter:per_composition",
              # Not a filter: nothing is removed by failing it. It is in the
              # funnel so that a run whose steps disagree is counted somewhere a
              # person looks, rather than only on the row itself.
              "dft:static:consistent_with_previous"]


@dataclass
class Funnel:
    """How many structures each gate saw and how many it kept."""

    rows: list[tuple[str, int, int]] = field(default_factory=list)
    generated: int = 0
    reached_dft: int = 0

    def render(self) -> str:
        lines = [f"{'gate':<26} {'seen':>8} {'passed':>8} {'rejected':>9}"]
        for gate, seen, passed in self.rows:
            lines.append(f"{gate:<26} {seen:>8} {passed:>8} {seen - passed:>9}")
        return "\n".join(lines)


def funnel(store: Store) -> Funnel:
    """Per-gate counts, straight from the recorded events."""
    counts: dict[str, tuple[int, int]] = {}
    for row in store.sql.execute(
            "SELECT gate, COUNT(*) AS seen, "
            "       SUM(CASE WHEN passed THEN 1 ELSE 0 END) AS passed "
            "FROM filter_event GROUP BY gate"):
        counts[row["gate"]] = (int(row["seen"]), int(row["passed"] or 0))

    ordered = [(gate, *counts[gate]) for gate in GATE_ORDER if gate in counts]
    ordered += [(gate, *v) for gate, v in sorted(counts.items())
                if gate not in GATE_ORDER]
    return Funnel(rows=ordered,
                  generated=store.count_structures(),
                  reached_dft=store.count_structures(state="dft_done"))


def candidate_rows(store: Store, *, states: tuple[str, ...] = ("dft_done",),
                   limit: int | None = None) -> list[dict[str, Any]]:
    """One dictionary per candidate, keyed by `COLUMNS`.

    Sorted by DFT hull distance where it exists and by the MLIP's estimate
    otherwise, with the two kept in separate columns so the ordering is never
    mistaken for a single quantity.
    """
    names = [name for name, _ in COLUMNS]
    out: list[dict[str, Any]] = []
    for state in states:
        for row in store.structures(state=state):
            kv = row.key_value_pairs
            record: dict[str, Any] = {name: kv.get(name) for name in names}
            record["id"] = int(row.id)
            record["formula"] = kv.get("reduced_formula") or row.formula
            record["n_atoms"] = int(row.natoms)
            record["state"] = state
            # mu0*M is the number a reader compares against Nd2Fe14B, and it is
            # one multiplication from the cell magnetisation and the volume.
            # Derived here rather than stored, so it can never disagree with
            # the two columns it comes from.
            summary = summarise(kv.get("m_dft_raw"), kv.get("volume"),
                                source="m_dft_raw")
            record["mu0_m_tesla"] = summary.mu0_m
            record["m_emu_per_cc"] = summary.emu_per_cc
            if record.get("m_per_volume") is None:
                record["m_per_volume"] = summary.m_per_volume
            out.append(record)

    def key(record: dict[str, Any]) -> tuple[int, float]:
        dft = record.get("dft_e_above_hull")
        mlip = record.get("e_above_hull_mlip")
        if dft is not None:
            return (0, float(dft))
        if mlip is not None:
            return (1, float(mlip))
        return (2, 0.0)

    out.sort(key=key)
    return out[:limit] if limit else out


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """The table as CSV, with a comment header naming every column.

    The header is worth the two lines: `m_dft_raw` and `m_s_reconstructed` are
    both magnetisations and only one of them is computed, and a bare column name
    does not say which.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        for name, meaning in COLUMNS:
            handle.write(f"# {name}: {meaning}\n")
        writer = csv.DictWriter(handle, fieldnames=[n for n, _ in COLUMNS],
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
