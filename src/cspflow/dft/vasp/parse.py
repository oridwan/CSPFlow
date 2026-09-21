"""Readers for VASP output.

Two constraints shape this module:

* **OUTCARs are large.** The ones in `/projects/mmi/shuo/redo-new-ter-mag` run
  to 20-26 MB each, and there are thousands. Every status question is answered
  from a tail read, never by loading the file.

* **"Finished" and "converged" are different questions, and conflating them is
  the bug this exists to avoid.** Measured over 106 jobs in that campaign: 100 %
  wrote a `VASP_DONE` marker and 100 % reached VASP's own "General timing"
  epilogue -- so every one of them exited cleanly -- but only 39 % reached the
  force criterion. The other 61 % hit `NSW` and stopped. A marker file cannot
  tell you which, so we read the OUTCAR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Enough to cover VASP's epilogue plus the last few ionic steps.
_TAIL_BYTES = 200_000

_OSZICAR_IONIC = re.compile(
    r"^\s*(\d+)\s+F=\s*([-.\dE+]+)\s+E0=\s*([-.\dE+]+)(?:.*?mag=\s*([-.\dE+]+))?",
    re.MULTILINE,
)
# One SCF iteration. VASP names the line after the algorithm in use -- DAV for
# blocked Davidson, RMM for RMM-DIIS, CG for conjugate gradient -- and ALGO=Fast
# switches between them mid-run, so the name is matched loosely and only the
# iteration NUMBER is read.
_OSZICAR_ELECTRONIC = re.compile(r"^\s*([A-Z]{2,5}):\s+(\d+)\s", re.MULTILINE)

_INCAR_TAG = re.compile(r"^\s*([A-Z_]+)\s*=\s*(.+?)\s*(?:[#!].*)?$", re.MULTILINE)


def head_text(path: Path, nbytes: int = 40_000) -> str:
    """Read the first `nbytes` of a file as text.

    Needed because VASP prints the run's dimensions (NIONS, NBANDS, the POTCAR
    list) in the header and its outcome in the epilogue, so answering "how many
    atoms, and did it converge?" takes a read at each end -- still far cheaper
    than loading a 26 MB OUTCAR.
    """
    with open(path, "rb") as fh:
        return fh.read(nbytes).decode("utf8", "ignore")


def tail_text(path: Path, nbytes: int = _TAIL_BYTES) -> str:
    """Read the last `nbytes` of a file as text, tolerating binary noise."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - nbytes))
        return fh.read().decode("utf8", "ignore")


@dataclass(frozen=True)
class OszicarResult:
    n_ionic_steps: int
    energy: float | None       # F=  (free energy)
    e0: float | None           # E0= (energy with sigma->0; what we store)
    magnetisation: float | None
    # SCF iterations in the LAST ionic step. Compared against NELM, this is the
    # only evidence that a static's energy is converged -- a static has no force
    # criterion, so nothing else in the file distinguishes a converged answer
    # from one VASP gave up on. See read_job_directory.
    n_electronic_steps: int | None = None


def read_oszicar(path: Path) -> OszicarResult:
    """Last ionic step of an OSZICAR.

    `E0` is the sigma->0 energy and is the one that belongs in a hull; `F` is
    the free energy at the smearing width actually used. They differ by a few
    meV for the ISMEAR=1/SIGMA=0.05 settings in use here, which is the same
    order as the hull thresholds, so the distinction is kept explicit rather
    than left to whichever the caller happens to grab.
    """
    if not path.is_file():
        return OszicarResult(0, None, None, None, None)
    text = tail_text(path)
    matches = _OSZICAR_IONIC.findall(text)
    electronic = _OSZICAR_ELECTRONIC.findall(text)
    n_elec = int(electronic[-1][1]) if electronic else None
    if not matches:
        return OszicarResult(0, None, None, None, n_elec)
    step, f_energy, e0, mag = matches[-1]
    return OszicarResult(
        n_ionic_steps=int(step),
        energy=float(f_energy),
        e0=float(e0),
        magnetisation=float(mag) if mag else None,
        n_electronic_steps=n_elec,
    )


@dataclass(frozen=True)
class OutcarStatus:
    exists: bool
    finished: bool          # VASP wrote its epilogue -- the process ended normally
    converged: bool         # ionic relaxation reached the force criterion
    n_atoms: int | None
    elapsed_seconds: float | None

    @property
    def hit_step_limit(self) -> bool:
        """Finished cleanly but never converged: stopped at NSW."""
        return self.finished and not self.converged


def read_outcar_status(path: Path) -> OutcarStatus:
    if not path.is_file():
        return OutcarStatus(False, False, False, None, None)
    text = tail_text(path)
    finished = "General timing and accounting" in text
    converged = "reached required accuracy" in text
    elapsed = None
    m = re.search(r"Elapsed time \(sec\):\s*([\d.]+)", text)
    if m:
        elapsed = float(m.group(1))
    # NIONS is in the header, not the epilogue -- but HOW FAR into the header
    # depends on the chemistry. VASP echoes each POTCAR block twice, so a
    # ternary with verbose pseudopotentials pushes NIONS past a small window.
    #
    # Found 2026-09-15: in mp-4459 (Ce2Fe14B, PAW_PBE Ce_3 / Fe_pv / B) NIONS
    # sits at byte 40,335 -- 335 bytes past the old fixed 40,000-byte read. It
    # parsed as None, so `e_per_atom` was never derived, and the row went into
    # index.csv with an energy but no per-atom energy. 65 phases were affected,
    # nearly all of them the R2Fe14B / R2Co14B / R2Fe14C family -- exactly the
    # competing phases a 2:14:1 hull needs. A hull built on `e_per_atom_eV`
    # silently omitted them, and omitted the parent compound itself.
    #
    # So: keep the cheap 40 KB read as the fast path, and widen only when that
    # misses. 1 MB is still far less than a 26 MB OUTCAR.
    n_atoms = None
    for window in (40_000, 1_000_000):
        m = re.search(r"NIONS\s*=\s*(\d+)", head_text(path, window))
        if m:
            n_atoms = int(m.group(1))
            break
    return OutcarStatus(True, finished, converged, n_atoms, elapsed)


def read_incar(path: Path) -> dict[str, str]:
    """INCAR as a plain tag -> string mapping. No interpretation."""
    if not path.is_file():
        return {}
    return {m.group(1): m.group(2) for m in _INCAR_TAG.finditer(path.read_text(errors="ignore"))}


def incar_int(incar: dict[str, str], tag: str) -> int | None:
    raw = incar.get(tag)
    if raw is None:
        return None
    m = re.search(r"-?\d+", raw)
    return int(m.group()) if m else None


def read_potcar_symbols(path: Path) -> list[str]:
    """The ordered POTCAR symbols actually used, from the TITEL lines.

    Read from the run's own POTCAR rather than inferred from the structure, so
    what is recorded is what VASP was given.
    """
    if not path.is_file():
        return []
    symbols: list[str] = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "TITEL" in line:
                parts = line.split("=", 1)[1].split()
                if len(parts) >= 2:
                    symbols.append(parts[1])
    return symbols


@dataclass(frozen=True)
class JobOutcome:
    """Everything one relaxation directory can tell us."""

    path: Path
    state: str                  # done | failed | running | timeout
    converged: bool
    n_ionic_steps: int
    step_limit: int | None
    energy: float | None        # E0, eV
    e_per_atom: float | None
    magnetisation: float | None
    n_atoms: int | None
    slurm_id: str
    core_hours: float
    exit_reason: str
    potcar_symbols: list[str]

    @property
    def unconverged_but_finished(self) -> bool:
        return self.state == "done" and not self.converged


def _slurm_id_from_dir(directory: Path) -> str:
    """Slurm ids survive only in the stdout/stderr filenames (`vasp_<id>.out`)."""
    for pattern in ("vasp_*.out", "vasp_*.err", "*.out"):
        for candidate in sorted(directory.glob(pattern)):
            m = re.search(r"(\d{5,})", candidate.name)
            if m:
                return m.group(1)
    return ""


def read_job_directory(directory: Path) -> JobOutcome:
    """Classify one VASP run directory.

    The state machine deliberately separates process outcome from physics:
    `state` says what happened to the job, `converged` says whether the answer
    is usable. A run that finished cleanly at the ionic step limit is
    `state='done', converged=False` with `exit_reason='ionic_step_limit'` --
    not a failure, and emphatically not a success.
    """
    outcar = read_outcar_status(directory / "OUTCAR")
    oszicar = read_oszicar(directory / "OSZICAR")
    incar = read_incar(directory / "INCAR")
    nsw = incar_int(incar, "NSW")
    ibrion = incar_int(incar, "IBRION")
    n_atoms = outcar.n_atoms

    # A step with no ionic motion has no force criterion to reach, so VASP never
    # prints "reached required accuracy" and `outcar.converged` is always False.
    # Judged by the relax step's rule, a perfectly good static calculation is
    # therefore "not converged" -- and the retry ladder fires on it forever.
    #
    # Found live: a `static` step reported 200 ionic steps and converged=False,
    # having been retried once already.
    static = (nsw is not None and nsw <= 0) or ibrion == -1

    # But "finished" is not the substitute criterion, and using it as one was
    # D151. `outcar.finished` means VASP wrote its epilogue -- a statement about
    # the process, which the docstring at the top of this file exists to keep
    # apart from the physics. A static whose SCF ran out at NELM writes that
    # epilogue exactly like a converged one, so EVERY static in the
    # RE-magnets-CHGNet campaign read as converged: 2,245 of 2,245 rows, with
    # 149 of the 2,039 runs on disk having actually stopped at NELM.
    #
    # The evidence a static does leave is electronic: an SCF that converged
    # stopped below NELM, and one VASP gave up on hit it exactly. That test also
    # belongs on a relax -- an ionically converged step whose final SCF did not
    # converge is not a usable energy either -- so it is applied to both, as a
    # requirement rather than a replacement.
    nelm = incar_int(incar, "NELM")
    n_elec = oszicar.n_electronic_steps
    scf_converged = not (nelm and n_elec and n_elec >= nelm)

    converged = (outcar.converged or (static and outcar.finished)) and scf_converged

    if not outcar.exists:
        state, reason = "failed", "no OUTCAR"
    elif not outcar.finished:
        # No epilogue: the process was killed. Walltime is the usual cause.
        state, reason = "timeout", "OUTCAR has no epilogue (killed mid-run)"
    elif converged:
        state, reason = "done", ""
    elif not scf_converged:
        # Names the rung that fits: ALGO/NELM, not more ionic steps.
        state, reason = "done", "scf_not_converged"
    elif nsw is not None and oszicar.n_ionic_steps >= nsw:
        state, reason = "done", "ionic_step_limit"
    else:
        state, reason = "done", "finished without reaching the force criterion"

    e_per_atom = None
    if oszicar.e0 is not None and n_atoms:
        e_per_atom = oszicar.e0 / n_atoms

    core_hours = 0.0
    if outcar.elapsed_seconds:
        ncore = incar_int(incar, "NCORE") or 1
        core_hours = outcar.elapsed_seconds / 3600.0 * ncore

    return JobOutcome(
        path=directory,
        state=state,
        converged=converged,
        n_ionic_steps=oszicar.n_ionic_steps,
        step_limit=nsw,
        energy=oszicar.e0,
        e_per_atom=e_per_atom,
        magnetisation=oszicar.magnetisation,
        n_atoms=n_atoms,
        slurm_id=_slurm_id_from_dir(directory),
        core_hours=core_hours,
        exit_reason=reason,
        potcar_symbols=read_potcar_symbols(directory / "POTCAR"),
    )


# --------------------------------------------------------------------------
# Do two steps of one structure describe the same calculation?
# --------------------------------------------------------------------------

# A static that lands in a different magnetic state from the relaxation that
# produced its geometry is the failure this measures. The static starts its SCF
# from the MAGMOM guess again -- no WAVECAR, no CHGCAR -- so it re-finds the
# magnetic solution from scratch and can settle somewhere else. The geometry is
# then optimised in one state and the energy reported for another; neither
# number is wrong alone, and the pair is not a result.
#
# Measured over 2,067 structures of RE-magnets-CHGNet whose relax converged:
#
#   |relax -> static shift| > 60 meV/atom : 41 of 46 changed moment (89%)
#   |relax -> static shift| <= 5 meV/atom : 50 of 1753 changed moment (3%)
#
# 0.1 uB/atom is where those two populations separate. 60 meV/atom is the
# campaign's own selection threshold -- below it the shift cannot change a
# ranking decision, so flagging it would be noise. The worst seen were
# structure 13836 (-2.7 -> +4.1 uB) and structure 4470 (-8.8 -> +12.0 uB).
ENERGY_SHIFT_MEV_PER_ATOM = 60.0
MAGMOM_SHIFT_PER_ATOM = 0.1


@dataclass(frozen=True)
class StepConsistency:
    """How far a step moved from the one whose geometry it inherited."""

    energy_shift: float | None       # meV/atom, static minus previous
    magmom_shift: float | None       # uB/atom, absolute
    ok: bool
    detail: str

    @property
    def measurable(self) -> bool:
        return self.energy_shift is not None


def step_consistency(previous, current,
                     energy_limit: float = ENERGY_SHIFT_MEV_PER_ATOM,
                     magmom_limit: float = MAGMOM_SHIFT_PER_ATOM) -> StepConsistency:
    """Compare a step against the step whose relaxed geometry it started from.

    Both arguments are `JobOutcome`s. Returns `ok=True` when nothing can be
    measured: a missing number is not evidence of a problem, and a gate that
    fails on absent data teaches people to ignore it.
    """
    n_atoms = current.n_atoms or previous.n_atoms
    if previous.e_per_atom is None or current.e_per_atom is None:
        return StepConsistency(None, None, True, "")

    shift = (current.e_per_atom - previous.e_per_atom) * 1000.0

    dmag = None
    if (previous.magnetisation is not None and current.magnetisation is not None
            and n_atoms):
        dmag = abs(current.magnetisation - previous.magnetisation) / n_atoms

    if abs(shift) <= energy_limit:
        return StepConsistency(shift, dmag, True, "")

    # Over the limit. Name the likely cause rather than only the symptom -- the
    # moment is what a person would check next, and it is already in hand.
    if dmag is not None and dmag > magmom_limit:
        detail = (f"energy moved {shift:+.0f} meV/atom and the moment moved "
                  f"{dmag:.2f} uB/atom ({previous.magnetisation:+.2f} -> "
                  f"{current.magnetisation:+.2f} uB total): the two steps "
                  f"settled in different magnetic states, so the geometry and "
                  f"the energy do not describe the same calculation")
    elif dmag is not None:
        detail = (f"energy moved {shift:+.0f} meV/atom with the moment steady "
                  f"({dmag:.2f} uB/atom): not a magnetic flip -- check the "
                  f"k-mesh and ISMEAR difference between the two steps")
    else:
        detail = f"energy moved {shift:+.0f} meV/atom; no moment recorded"
    return StepConsistency(shift, dmag, False, detail)
