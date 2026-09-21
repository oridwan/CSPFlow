"""POTCAR resolution, identity and the 4f convention.

Two things this module exists to prevent, both measured on real files:

1.  **A label that misdescribes what is on disk.**  pymatgen maps functional
    labels to directory NAMES (PBE_64 -> POT_PAW_PBE_64).  A raw ``potpaw_PBE``
    tree has element folders at the top level, so ``functional: PBE_64`` fails
    while ``PBE_54`` succeeds by flat-layout fallback -- and now the recorded
    label says PBE_54 for what are actually VASP 6.4 potentials.
    ``ensure_pmg_layout`` builds the symlink tree that makes the label true.

2.  **Mixed 4f conventions inside one campaign.**  MPRelaxSet selects frozen
    ``_3`` variants for most of the rare-earth series but bare, f-in-valence
    POTCARs for Ce, Gd and Eu.  Gd2Fe17 then has 4f in valence while Tb2Fe17
    has it frozen, and their moments are not comparable.  Here the convention is
    a campaign-wide declaration applied to the WHOLE series.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from ...config.schema import RARE_EARTHS, FTreatment, Machine

# Explicit default symbols.  Not derived from any pymatgen input set: the whole
# point of the INCAR/POTCAR decision is that what we run is written down here
# where it can be read, diffed and overridden, rather than inherited from a
# library whose values move between releases.
#
# 3d series follows the VASP recommendation (semicore p for the early metals).
DEFAULT_SYMBOLS: dict[str, str] = {
    "Ti": "Ti_pv", "V": "V_pv", "Cr": "Cr_pv", "Mn": "Mn_pv", "Fe": "Fe_pv",
    "Co": "Co", "Ni": "Ni", "Cu": "Cu", "Zn": "Zn",
    "Y": "Y_sv", "Sc": "Sc_sv", "Zr": "Zr_sv", "Nb": "Nb_pv", "Mo": "Mo_pv",
    "Al": "Al", "Si": "Si", "B": "B", "C": "C", "N": "N", "O": "O",
}


class PotcarError(Exception):
    """POTCAR could not be resolved, or is not what it claims to be."""


@dataclass(frozen=True)
class PotcarInfo:
    element: str
    symbol: str
    path: Path
    titel: str
    zval: float
    enmax: float
    md5_header_hash: str
    n_f_valence: bool

    @property
    def short_hash(self) -> str:
        return self.md5_header_hash[:8]


# La is 4f0: there is no 4f shell to freeze, so VASP ships no `La_3` and never
# will.  The bare `La` dataset IS the frozen-f convention for lanthanum -- it has
# no f valence electrons to begin with -- so it satisfies
# `assert_one_f_convention` alongside Sm_3 and Gd_3 rather than violating it.
# Verified against the local PBE_64 tree 2026-09-01: La, La_GW and La_s exist;
# La_3 does not.  Every other element of the series has its `_3` variant.
NO_FROZEN_VARIANT = {"La": "La"}


def rare_earth_symbol(element: str, treatment: FTreatment) -> str:
    """Apply the campaign's 4f convention to one rare earth.

    ``frozen`` gives ``<El>_3`` across the series, Gd/Eu/Ce included --
    deliberately diverging from MP, which mixes conventions.  ``valence`` gives
    the bare symbol, with f electrons in the valence.

    La is the one element where the two coincide, because it has no 4f
    electrons.  Without this it resolves to a `La_3` that does not exist, and
    the campaign fails at the first La structure rather than at configuration
    time.
    """
    if treatment is FTreatment.frozen:
        return NO_FROZEN_VARIANT.get(element, f"{element}_3")
    return element


def potcar_symbols(
    elements: list[str],
    *,
    f_treatment: FTreatment = FTreatment.frozen,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Element -> POTCAR symbol for a campaign.  Overrides always win."""
    overrides = overrides or {}
    out: dict[str, str] = {}
    for el in elements:
        if el in overrides:
            out[el] = overrides[el]
        elif el in RARE_EARTHS:
            out[el] = rare_earth_symbol(el, f_treatment)
        else:
            out[el] = DEFAULT_SYMBOLS.get(el, el)
    return out


# --------------------------------------------------------------------------
# tree layout
# --------------------------------------------------------------------------


def tree_directory(machine: Machine, tree: str) -> tuple[str, Path]:
    """Resolve a tree label (VASP6.4) to (functional_label, directory)."""
    if not machine.potcar_root:
        raise PotcarError(
            "machine profile has no potcar_root; set it or export PMG_VASP_PSP_DIR"
        )
    functional = machine.potcar_trees.get(tree)
    if functional is None:
        known = sorted(machine.potcar_trees) or ["<none>"]
        raise PotcarError(f"machine profile has no potcar tree {tree!r}; known: {known}")
    dirname = machine.potcar_dirs.get(functional)
    if dirname is None:
        raise PotcarError(
            f"machine profile maps tree {tree!r} to functional {functional!r}, "
            f"but potcar_dirs has no entry for it"
        )
    return functional, Path(machine.potcar_root) / dirname


def ensure_pmg_layout(machine: Machine, sources: dict[str, str], *, create: bool = False
                      ) -> list[tuple[str, str]]:
    """Check (and optionally create) the pymatgen-layout symlink tree.

    ``sources`` maps a functional label to the real potpaw_PBE tree it should
    point at.  Returns a list of (status, message); nothing is created unless
    ``create`` is true.
    """
    actions: list[tuple[str, str]] = []
    if not machine.potcar_root:
        return [("fail", "no potcar_root in machine profile")]
    root = Path(machine.potcar_root)

    if not root.exists():
        if create:
            root.mkdir(parents=True, exist_ok=True)
            actions.append(("fixed", f"created {root}"))
        else:
            actions.append(("fail", f"potcar_root {root} does not exist (run with --fix)"))

    for functional, real in sources.items():
        dirname = machine.potcar_dirs.get(functional)
        if dirname is None:
            actions.append(("warn", f"no potcar_dirs entry for {functional}"))
            continue
        link = root / dirname
        real_path = Path(real)
        if not real_path.is_dir():
            actions.append(("fail", f"{functional}: source tree {real_path} not found"))
            continue
        if link.is_symlink():
            target = link.resolve()
            if target == real_path.resolve():
                actions.append(("ok", f"{dirname} -> {real_path}"))
            else:
                actions.append(
                    ("fail", f"{dirname} points at {target}, expected {real_path}")
                )
        elif link.exists():
            actions.append(("warn", f"{link} exists and is not a symlink; leaving alone"))
        elif create:
            link.symlink_to(real_path)
            actions.append(("fixed", f"created {dirname} -> {real_path}"))
        else:
            actions.append(("fail", f"{dirname} missing (run with --fix)"))
    return actions


# --------------------------------------------------------------------------
# reading a POTCAR
# --------------------------------------------------------------------------


# Measured ZVAL for both flavours across the local trees:
#
#   Nd_3 11 / Nd 14   Sm_3 11 / Sm 16   Eu_3  9 / Eu 17
#   Gd_3  9 / Gd 18   Tb_3  9 / Tb 19   Dy_3  9 / Dy 20
#   Ce_3 11 / Ce 12   <- the exception that shapes the check below
#
# No frozen-core dataset reaches 14, so "suffixed but ZVAL >= 14" is a safe
# contradiction to flag.  The converse is NOT safe: cerium's f-in-valence
# dataset carries a single 4f electron, so bare Ce sits at ZVAL 12, squarely
# inside the frozen-core range.  A lower-bound check would reject a perfectly
# good POTCAR, so the naming convention alone decides that direction.
_VALENCE_ZVAL_MIN = 14.0


def f_in_valence(element: str, symbol: str, zval: float) -> bool:
    """Does this POTCAR carry the 4f shell in the valence?

    Determined from the symbol suffix, which is the authoritative VASP naming
    convention (``Gd_3`` trivalent frozen-f, ``Eu_2`` divalent, bare ``Gd``
    f-in-valence), and cross-checked against ZVAL.

    Note for anyone tempted to use VRHFIN instead: it does not work.  Measured
    on both local trees, ``Gd_3`` and ``Gd`` report the *identical* string
    ``VRHFIN =Gd : [core=Xe4]`` while their ZVALs are 9 and 18.  VRHFIN cannot
    tell the two conventions apart; ZVAL can.
    """
    if element not in RARE_EARTHS:
        return False
    if element in NO_FROZEN_VARIANT:
        # La is 4f0.  The suffix heuristic reads bare `La` as f-in-valence and
        # is wrong here, because there are no f electrons to put in the
        # valence: ZVAL 11 (5s2 5p6 5d1 6s2) sits in the frozen-core range,
        # while every genuine f-in-valence dataset is 14-20.  Left uncorrected,
        # a La campaign trips `assert_one_f_convention` against its own
        # correctly-frozen Sm_3 and Gd_3.
        return False
    suffixed = bool(re.search(r"_\d+$", symbol))
    inferred = not suffixed
    if suffixed and zval >= _VALENCE_ZVAL_MIN:
        raise PotcarError(
            f"{symbol} is named as a frozen-core dataset but has ZVAL {zval:g}, "
            f"which is an f-in-valence count. The file does not match its name."
        )
    return inferred


def _parse_header(path: Path) -> tuple[str, float, float]:
    """Pull TITEL, ZVAL and ENMAX without needing pymatgen."""
    titel, zval, enmax = "", 0.0, 0.0
    with path.open(errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not titel and s.startswith("TITEL"):
                titel = s.split("=", 1)[1].strip()
            elif not zval and "ZVAL" in s and "POMASS" in s:
                for part in s.split(";"):
                    if "ZVAL" in part:
                        zval = float(part.split("=")[1].split()[0])
            elif not enmax and "ENMAX" in s:
                enmax = float(s.split("ENMAX")[1].split(";")[0].replace("=", "").strip())
            if titel and zval and enmax:
                break
    if not titel:
        raise PotcarError(f"{path} has no TITEL line; truncated or not a POTCAR")
    return titel, zval, enmax


def _md5_header_hash(path: Path) -> str:
    """The hash MP records in input.potcar_spec[].hash.

    Uses pymatgen when available so the value is byte-comparable with MP's own
    records; falls back to a whole-file md5, which still detects a changed
    POTCAR but cannot be compared against MP.
    """
    try:
        from pymatgen.io.vasp.inputs import PotcarSingle

        return PotcarSingle.from_file(str(path)).md5_header_hash
    except Exception:
        return "sha:" + hashlib.md5(path.read_bytes()).hexdigest()


def read_potcar(directory: Path, symbol: str, element: str) -> PotcarInfo:
    path = directory / symbol / "POTCAR"
    if not path.is_file():
        available = sorted(p.name for p in directory.glob(f"{element}*")) if directory.is_dir() else []
        raise PotcarError(
            f"no POTCAR for {element} at symbol {symbol!r} under {directory}. "
            f"Candidates present: {available or '<none>'}"
        )
    titel, zval, enmax = _parse_header(path)
    f_valence = f_in_valence(element, symbol, zval)
    return PotcarInfo(
        element=element, symbol=symbol, path=path, titel=titel, zval=zval,
        enmax=enmax, md5_header_hash=_md5_header_hash(path), n_f_valence=f_valence,
    )


def resolve_all(
    elements: list[str],
    machine: Machine,
    *,
    tree: str = "VASP6.4",
    f_treatment: FTreatment = FTreatment.frozen,
    overrides: dict[str, str] | None = None,
) -> tuple[list[PotcarInfo], list[str]]:
    """Resolve every element, returning (resolved, errors)."""
    _, directory = tree_directory(machine, tree)
    symbols = potcar_symbols(elements, f_treatment=f_treatment, overrides=overrides)
    infos: list[PotcarInfo] = []
    errors: list[str] = []
    for el in sorted(elements):
        try:
            infos.append(read_potcar(directory, symbols[el], el))
        except PotcarError as exc:
            errors.append(str(exc))
    return infos, errors


def assert_one_f_convention(infos: list[PotcarInfo]) -> None:
    """One 4f convention per campaign, enforced.

    This is the check that would have caught MP's Gd2Fe17 (4f in valence, ZVAL
    18) sitting in the same reference set as Tb2Fe17 (4f frozen, ZVAL 9).
    """
    res = [i for i in infos if i.element in RARE_EARTHS]
    if len(res) < 2:
        return
    conventions = {i.n_f_valence for i in res}
    if len(conventions) > 1:
        frozen = sorted(i.element for i in res if not i.n_f_valence)
        valence = sorted(i.element for i in res if i.n_f_valence)
        raise PotcarError(
            f"mixed 4f conventions in one campaign: frozen {frozen} vs "
            f"f-in-valence {valence}. Their moments are not comparable. "
            f"Set dft.rare_earth.f_treatment and remove any per-element override "
            f"that contradicts it."
        )


def max_enmax(infos: list[PotcarInfo]) -> float:
    """Highest ENMAX -- what VASP would default ENCUT to, per composition.

    Reported by `csp doctor` precisely because it is composition-dependent:
    267.9 eV for Sm_3/Fe/Ti but 295.4 eV with Cu present, so a default ENCUT
    silently changes between two entries of the same hull.
    """
    return max((i.enmax for i in infos), default=0.0)
