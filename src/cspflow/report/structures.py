"""One row per structure, every structure, at whatever stage it reached.

WHY THIS EXISTS
    `candidates.csv` answers "what came out": it lists structures in `dft_done`,
    ranked. That is the right table for a result and the wrong one for a
    question, because the structures you most want to look at are usually the
    ones that are NOT in it -- the ones that stopped somewhere, and the campaign
    could not tell you where without `csp status --why <id>`, one id at a time.

    So this is the other table: every structure the campaign ever made, with the
    verdict of each stage that touched it and, for anything that stopped, the
    gate that stopped it and the reason recorded at the time.

IT IS DERIVED, NEVER AUTHORED
    Every column comes out of the campaign database. Nothing is computed here
    that is not already stored, except the two derived-on-read columns noted
    below, and nothing is written back. Delete the file and the next driver
    cycle rebuilds it identically. That is deliberate: a second file that can
    disagree with the database is worse than no file at all (D129 lost 37
    finished calculations to exactly that class of problem).

INPUTS
    a campaign Store

OUTPUTS
    `structures.csv` -- one row per structure, a comment header naming every
    column, sorted by hull distance where known and by id otherwise

RUN
    written automatically by the driver each cycle, to <workdir>/structures.csv
    and by `csp report`, to report/structures.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..db.store import Store

#: (column, what it means). The order is the order a structure meets them.
COLUMNS: list[tuple[str, str]] = [
    # -- identity ---------------------------------------------------------
    ("id", "campaign structure id"),
    ("formula", "reduced formula"),
    ("n_atoms", "atoms in the cell"),
    ("spacegroup", "spacegroup number, relaxed cell"),
    ("spacegroup_symbol", "at the tolerance in `symprec`"),
    # -- where it came from ----------------------------------------------
    ("origin", "generated | seed -- how the structure entered the campaign"),
    ("source_name", "the `name:` of the source block that produced it"),
    ("source_path", "the seed file, for structure_list sources"),
    # -- where it got to --------------------------------------------------
    ("state", "current state: new, screened, filtered_out, selected, dft_done, failed, ..."),
    ("selected", "true if the filter chose it for DFT"),
    ("stopped_at", "the gate that ended its run; empty if it is still moving"),
    ("why", "the reason recorded at that gate"),
    # -- the MLIP ---------------------------------------------------------
    ("mlip_e_per_atom", "eV/atom, MatterSim after relaxation"),
    ("mlip_converged", "did the MLIP relaxation reach fmax"),
    ("mlip_steps", "ionic steps the MLIP took"),
    ("mlip_volume_drift", "fractional cell-volume change during MLIP relaxation"),
    ("e_above_hull_mlip", "eV/atom, MLIP energies against the store's MLIP hull"),
    # -- our DFT ----------------------------------------------------------
    ("dft_step", "how far up the recipe ladder it got (0 = relax, 1 = static)"),
    ("dft_attempt", "retry attempts spent"),
    ("dft_converged", "did the last DFT step report a converged relaxation"),
    ("vasp_energy", "eV, total energy of the cell"),
    ("e_per_atom", "eV/atom"),
    ("dft_e_above_hull", "eV/atom, our DFT against the store's DFT hull, one scale"),
    ("dft_e_formation", "eV/atom"),
    # -- physics ----------------------------------------------------------
    ("volume", "A^3, relaxed cell"),
    ("volume_per_atom", "A^3"),
    ("m_dft_raw", "mu_B per cell, COMPUTED: the cell magnetisation VASP reported"),
    ("m_s_reconstructed", "mu_B per cell, MODELLED: TM sublattice + Hund's-rule 4f"),
    # -- where the files are ----------------------------------------------
    ("dft_dir", "the finished VASP directory, if there is one"),
]

#: Gates in the order they run, so "where did it stop" is answerable.
GATE_ORDER = [
    "screen:validate", "screen:converged",
    "dedup", "dedup:seed_collision",
    "filter:e_above_hull", "filter:per_composition",
    "dft:relax:converged", "dft:static:converged",
]

#: States that mean the structure is no longer moving.
TERMINAL = {"failed", "filtered_out", "dft_done"}


def _stopped(store: Store, sid: int, kv: dict) -> tuple[str, str]:
    """The gate that ended this structure's run, and the reason given there.

    THE GATE AND THE REASON MUST COME FROM THE SAME SOURCE. Taking the last
    failed event for one and the recorded reason for the other reads plausibly
    and is wrong: structure 122 of CePdGe has an unconverged relax event AND a
    later operator exclusion, and pairing them produced

        stopped_at  dft:relax:converged
        why         zero computed moment ... excluded by operator

    which says it stopped at a gate it did not stop at. A row that misattributes
    a stop is worse than one that says `filtered_out` and nothing more, because
    it sends you to the wrong place to look.

    So the terminal state chooses which record is authoritative, and both
    columns are then read from that one.

    A structure still moving returns ("", ""). An empty cell is honest; a
    made-up "in progress" would sort alongside real verdicts.
    """
    state = str(kv.get("state") or "")
    if state not in TERMINAL:
        return "", ""
    if state == "dft_done":
        return "", ""                # it finished; nothing stopped it

    # `filter_events` yields sqlite3.Row: it indexes like a mapping but has no
    # `.get`, so a missing column raises instead of returning None.
    def field(event, name):
        try:
            return event[name]
        except (IndexError, KeyError):
            return None

    try:
        failed = [e for e in store.filter_events(sid) if not field(e, "passed")]
    except Exception:                                          # noqa: BLE001
        failed = []

    def last_gate(prefixes: tuple[str, ...]) -> tuple[str, str]:
        for event in reversed(failed):
            gate = str(field(event, "gate") or "")
            if gate.startswith(prefixes):
                return gate, str(field(event, "detail") or "")
        return "", ""

    # An explicitly recorded reason wins, and drags its own gate with it.
    if state == "filtered_out" and kv.get("filter_reason"):
        gate, _ = last_gate(("filter:", "dedup"))
        return gate or "filtered_out", str(kv["filter_reason"])
    if state == "failed" and kv.get("dft_fail_reason"):
        gate, _ = last_gate(("dft:",))
        return gate or "failed", str(kv["dft_fail_reason"])
    if kv.get("fail_reason"):
        gate, _ = last_gate(("screen:", "dedup", "filter:", "dft:"))
        return gate or state, str(kv["fail_reason"])

    # Otherwise the last gate that refused it speaks for itself.
    if failed:
        event = failed[-1]
        return str(field(event, "gate") or state), str(field(event, "detail") or "")
    return state, ""


def structure_rows(store: Store) -> list[dict[str, Any]]:
    """Every structure in the campaign, as dictionaries keyed by `COLUMNS`."""
    names = [name for name, _ in COLUMNS]
    out: list[dict[str, Any]] = []
    for row in store.structures():
        kv = row.key_value_pairs
        record: dict[str, Any] = {name: kv.get(name) for name in names}
        record["id"] = int(row.id)
        record["formula"] = kv.get("reduced_formula") or row.formula
        record["n_atoms"] = int(row.natoms)
        record["state"] = kv.get("state") or ""
        record["selected"] = bool(kv.get("selected")) or record["state"] in (
            "selected", "dft_queued", "dft_running", "dft_done")
        # `volume` is one of ASE's OWN columns, not a key_value pair: reading it
        # from `kv` alone left the column blank for every structure the analyze
        # stage had not yet written, which is most of them. The row always knows
        # the current cell, so it is the fallback.
        volume = kv.get("volume") or getattr(row, "volume", None)
        record["volume"] = volume
        # Derived on read, never stored: volume per atom cannot then disagree
        # with the volume and atom count it comes from.
        if volume and record["n_atoms"]:
            record["volume_per_atom"] = round(float(volume) / record["n_atoms"], 4)
        # The ladder finishing IS the convergence statement for DFT.
        if record["state"] == "dft_done":
            record["dft_converged"] = True
        elif kv.get("dft_fail_reason"):
            record["dft_converged"] = False
        record["stopped_at"], record["why"] = _stopped(store, int(row.id), kv)
        out.append(record)

    def key(record: dict[str, Any]) -> tuple[int, float, int]:
        for n, column in enumerate(("dft_e_above_hull", "e_above_hull_mlip")):
            value = record.get(column)
            if value is not None:
                try:
                    return (n, float(value), record["id"])
                except (TypeError, ValueError):
                    pass
        return (2, 0.0, record["id"])

    out.sort(key=key)
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """The table as CSV, with a comment header naming every column.

    The header is worth its lines here for the same reason it is in
    `candidates.csv`: `m_dft_raw` and `m_s_reconstructed` are both
    magnetisations and only one of them was computed, and a bare column name
    does not say which.

    Written to a temporary file and renamed, because the driver rewrites this
    every cycle and a reader is quite likely to be a `watch` or a spreadsheet
    picking it up at the wrong moment.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in COLUMNS]
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        handle.write("# one row per structure, at whatever stage it reached\n")
        handle.write("# derived entirely from the campaign database; safe to delete\n")
        for name, meaning in COLUMNS:
            handle.write(f"#   {name:<22} {meaning}\n")
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for record in rows:
            writer.writerow({n: record.get(n) for n in names})
    tmp.replace(path)
    return path


def write(store: Store, path: Path) -> Path:
    """Build the table and write it. The one call the driver and CLI share."""
    return write_csv(structure_rows(store), path)
